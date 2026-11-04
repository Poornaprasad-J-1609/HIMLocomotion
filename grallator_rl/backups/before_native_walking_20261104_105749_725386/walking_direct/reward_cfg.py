"""Manual Grallator walking reward configuration. No LLM or Eureka dependency.

This avoids executing arbitrary LLM-generated Python during the first integration.
Candidate weights are reward *rates*, not per-step rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

ENV_VAR = "GRALLATOR_WALK_REWARD_FILE"


@dataclass(frozen=True)
class WalkingRewardCfg:
    track_xy: float = 2.0
    track_yaw: float = 1.0
    upright: float = 0.6
    height: float = 0.5
    vertical_velocity: float = 0.4
    joint_velocity: float = 0.01
    action_rate: float = 0.05
    action_magnitude: float = 0.01
    undesired_contact: float = 2.0
    xy_sigma: float = 0.30
    yaw_sigma: float = 0.40
    height_sigma: float = 0.04


WEIGHT_KEYS = (
    "track_xy", "track_yaw", "upright", "height", "vertical_velocity",
    "joint_velocity", "action_rate", "action_magnitude", "undesired_contact",
)
SCALE_KEYS = ("xy_sigma", "yaw_sigma", "height_sigma")
ALLOWED_KEYS = set(WEIGHT_KEYS + SCALE_KEYS)


def load_reward_cfg(path: str | Path | None = None) -> WalkingRewardCfg:
    """Load only vetted numeric parameters; never evaluate Python expressions."""
    if path is None:
        return WalkingRewardCfg()
    candidate_path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Candidate JSON must be an object")
    if set(payload) != {"version", "task", "parameters"}:
        raise ValueError("Expected exactly: version, task, parameters")
    if payload["version"] != 1 or payload["task"] != "grallator_flat_walking":
        raise ValueError("Invalid candidate version or task")
    values = payload["parameters"]
    if not isinstance(values, dict):
        raise ValueError("parameters must be a JSON object")
    unknown = set(values) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown reward parameters: {sorted(unknown)}")
    if len(values) != len(ALLOWED_KEYS):
        missing = ALLOWED_KEYS - set(values)
        raise ValueError(f"Missing reward parameters: {sorted(missing)}")
    validated = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite real number")
        val = float(value)
        if key in WEIGHT_KEYS and not (0.0 <= val <= 20.0):
            raise ValueError(f"{key} weight must be in [0, 20]")
        if key in SCALE_KEYS and not (0.005 <= val <= 2.0):
            raise ValueError(f"{key} sigma must be in [0.005, 2]")
        validated[key] = val
    return WalkingRewardCfg(**validated)


def load_reward_cfg_from_env() -> WalkingRewardCfg:
    return load_reward_cfg(os.environ.get(ENV_VAR) or Path(__file__).with_name("reward_weights.json"))
