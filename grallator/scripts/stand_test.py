import argparse
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()

import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, instantiate, replace
from isaaclab_newton.physics import NewtonCfg

from grallator.config.robot import GRALLATOR_CFG



@configclass
class GrallatorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = replace(
        GRALLATOR_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args.device,
        physics=NewtonCfg(),
    )

    with launch_simulation(sim_cfg, args):
        sim = SimulationContext(sim_cfg)

        scene = instantiate(GrallatorSceneCfg(num_envs=1, env_spacing=2.0))
        robot = scene["robot"]

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
            scene.write_data_to_sim()

            sim.step()
            scene.update(dt)

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


            if step == 1199:
                print("\n===== FINAL JOINT ERRORS =====")
                actual = robot.data.joint_pos.torch[0].tolist()
                desired = q_des[0].tolist()

                for name, target, measured in zip(
                    robot.joint_names, desired, actual
                ):
                    error = target - measured
                    print(
                        f"{name:20s} "
                        f"target={target:+.4f} "
                        f"actual={measured:+.4f} "
                        f"error={error:+.4f} rad"
                    )

                print("\nBase quaternion (xyzw):",
                      robot.data.root_quat_w.torch[0].tolist())


            if step == 1199:
                print("\n===== RUNTIME PD CHECK =====")

                q = robot.data.joint_pos.torch[0].tolist()
                dq = robot.data.joint_vel.torch[0].tolist()
                targets = q_des[0].tolist()

                kp = robot.data.joint_stiffness.torch[0].tolist()
                kd = robot.data.joint_damping.torch[0].tolist()
                limits = robot.data.joint_effort_limits.torch[0].tolist()

                for i, name in enumerate(robot.joint_names):
                    estimated_torque = (
                        kp[i] * (targets[i] - q[i])
                        - kd[i] * dq[i]
                    )

                    print(
                        f"{name:20s} "
                        f"Kp={kp[i]:.1f} "
                        f"Kd={kd[i]:.1f} "
                        f"limit={limits[i]:.1f} Nm "
                        f"velocity={dq[i]:+.4f} "
                        f"PD_torque={estimated_torque:+.2f} Nm"
                    )


            if step == 1199:
                print("\n===== FOOT HEIGHT CHECK =====")
                print("Simulation bodies:", robot.body_names)

                positions = robot.data.body_pos_w.torch[0].tolist()
                found = 0

                for name, xyz in zip(robot.body_names, positions):
                    if "foot" in name.lower():
                        found += 1
                        print(
                            f"{name}: center_z={xyz[2]:.4f} m, "
                            f"bottom_z={xyz[2]-0.04:.4f} m"
                        )

                if found == 0:
                    print(
                        "Foot links are not separate simulation bodies; "
                        "their fixed transforms must be checked separately."
                    )

            time.sleep(0.01)


if __name__ == "__main__":
    main()
