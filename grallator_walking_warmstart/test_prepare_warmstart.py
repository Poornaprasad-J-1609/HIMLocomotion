"""CPU-only tests; no Isaac Sim or Newton needed."""
import importlib.util
import tempfile
from pathlib import Path
import torch

script = Path(__file__).with_name("prepare_warmstart.py")
spec = importlib.util.spec_from_file_location("grallator_warmstart", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    source = root / "standing.pt"
    target = root / "walking.pt"
    torch.save({
        "actor_state_dict": {
            "distribution.std_param": torch.full((12,), 0.019551),
            "actor.weight": torch.arange(6.0),
            "estimator.weight": torch.ones(3),
        },
        "critic_state_dict": {"critic.weight": torch.ones(3)},
        "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(15.), "exp_avg": torch.ones(3), "exp_avg_sq": torch.ones(3)}},
            "param_groups": [{"lr": 1e-5, "params": [0], "betas": (0.9, 0.999)}],
        },
        "hio_optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(15.)}},
            "param_groups": [{"lr": 3e-4, "params": [0]}],
        },
        "iter": 1700,
        "infos": {"seed": 42},
    }, source)
    module.prepare(source, target, 0.12, 1e-4)
    old = torch.load(source, map_location="cpu", weights_only=False)
    new = torch.load(target, map_location="cpu", weights_only=False)
    assert old["iter"] == 1700 and new["iter"] == 0
    assert torch.allclose(new["actor_state_dict"]["distribution.std_param"], torch.full((12,), 0.12))
    assert torch.allclose(old["actor_state_dict"]["distribution.std_param"], torch.full((12,), 0.019551))
    assert torch.equal(new["actor_state_dict"]["actor.weight"], old["actor_state_dict"]["actor.weight"])
    assert torch.equal(new["actor_state_dict"]["estimator.weight"], old["actor_state_dict"]["estimator.weight"])
    assert new["optimizer_state_dict"]["state"] == {}
    assert new["hio_optimizer_state_dict"]["state"] == {}
    assert new["optimizer_state_dict"]["param_groups"][0]["lr"] == 1e-4
    try:
        module.prepare(source, target, 0.12, 1e-4)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected refusing overwrite")
print("PASS: checkpoint warm-start conversion; original unchanged; overwrite protection")
