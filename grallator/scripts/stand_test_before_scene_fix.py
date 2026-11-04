import argparse
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()

import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils import clone
from isaaclab_newton.physics import NewtonCfg

from grallator.config.robot import GRALLATOR_CFG


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args.device,
        physics=NewtonCfg(),
    )

    with launch_simulation(sim_cfg, args):
        sim = SimulationContext(sim_cfg)

        # Ground at z = 0
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/Ground", ground)

        # Load Grallator with our actuator configuration
        cfg = clone(GRALLATOR_CFG)
        cfg.prim_path = "/World/Grallator"
        robot = Articulation(cfg=cfg)

        sim.set_camera_view(
            [2.0, 2.0, 1.2],
            [0.0, 0.0, 0.35],
        )

        try:
            sim.reset()
        except KeyError:
            from isaaclab_newton.physics.newton_manager import NewtonManager
            from isaaclab.sim import get_current_stage
            from pxr import UsdPhysics

            print("\n===== NEWTON DIAGNOSTICS =====", flush=True)

            stage = get_current_stage()
            roots = [
                str(p.GetPath())
                for p in stage.Traverse()
                if p.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]
            print("USD articulation roots:", roots, flush=True)

            backend = NewtonManager.backend
            if backend is not None:
                model = backend.model
                labels = getattr(model, "articulation_label", None)
                print("Newton articulation count:",
                      getattr(model, "articulation_count", None), flush=True)
                print("Newton articulation labels:",
                      list(labels) if labels is not None else "Not exposed",
                      flush=True)

            print("Clone sources:",
                  list(NewtonManager._cl_protos.keys()), flush=True)
            raise

        print("Joint count:", robot.num_joints)
        print("Joint names:", robot.joint_names)

        assert robot.num_joints == 12

        # Restore MIDDLE_POSE and initial base position
        root_pose = robot.data.default_root_pose.torch.clone()
        root_vel = robot.data.default_root_vel.torch.clone()

        q_des = robot.data.default_joint_pos.torch.clone()
        dq_des = torch.zeros_like(q_des)

        robot.write_root_pose_to_sim_index(
            root_pose=root_pose
        )
        robot.write_root_velocity_to_sim_index(
            root_velocity=root_vel
        )
        robot.write_joint_position_to_sim_index(
            position=q_des
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=dq_des
        )
        robot.reset()

        dt = sim.get_physics_dt()

        for step in range(1200):
            if not sim.is_running():
                break

            # Hold desired joint positions
            robot.actuators.target_command.set_position_index(
                value=q_des
            )

            # Desired velocity = 0
            robot.actuators.target_command.set_velocity_index(
                value=dq_des
            )

            # No feedforward effort is commanded
            robot.write_data_to_sim()

            sim.step()
            robot.update(dt)

            if step % 100 == 0:
                z = robot.data.root_pos_w.torch[0, 2].item()

                error = torch.max(
                    torch.abs(
                        robot.data.joint_pos.torch - q_des
                    )
                ).item()

                print(
                    f"t={step*dt:.2f}s | "
                    f"base_z={z:.4f}m | "
                    f"max_joint_error={error:.4f}rad"
                )

            time.sleep(0.01)


if __name__ == "__main__":
    main()
