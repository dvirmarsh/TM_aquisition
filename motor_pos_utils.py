"""Shared helpers for calibration sampling, motor homing, and subset-trial naming."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

DEFAULT_NUM_RANDOM_TRIALS: int = 10
CALIBRATION_H5_GROUP = "initial_calibration"


def sample_random_subset_indices(
    n_available: int,
    subset_size: int,
    trial_idx: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + int(trial_idx) * 10_007 + int(subset_size))
    return np.sort(rng.choice(n_available, size=subset_size, replace=False))


def mask_run_label(method: str, subset_size: int, trial_idx: int) -> str:
    """H5 group label for saved SLM focus masks."""
    return f"{method}_subset_{subset_size}_trial_{trial_idx}"


def parse_mask_run_label(run_label: str) -> dict:
    """Parse ``{method}_subset_{size}_trial_{trial}`` or legacy ``{method}_subset_{size}``."""
    if "_trial_" in run_label:
        prefix, trial_str = run_label.rsplit("_trial_", maxsplit=1)
        if "_subset_" in prefix:
            method, size_str = prefix.rsplit("_subset_", maxsplit=1)
            return {
                "subset_key": run_label,
                "run_type": method,
                "subset_size": int(size_str),
                "trial": int(trial_str),
            }
    if "_subset_" in run_label:
        method, size_str = run_label.rsplit("_subset_", maxsplit=1)
        return {
            "subset_key": run_label,
            "run_type": method,
            "subset_size": int(size_str),
            "trial": None,
        }
    return {"subset_key": run_label, "run_type": None, "subset_size": None, "trial": None}


def mask_run_sort_key(run_label: str) -> Tuple[int, int, str]:
    parsed = parse_mask_run_label(run_label)
    subset_size = parsed["subset_size"] or 0
    trial = parsed["trial"] if parsed["trial"] is not None else -1
    run_type = parsed["run_type"] or ""
    return (subset_size, trial, run_type)


def home_motor(motor) -> None:
    """Home the motor using ``home()`` when available, otherwise move to 0 mm."""
    if hasattr(motor, "home"):
        try:
            motor.home(blocking=True)
        except TypeError:
            motor.home()
        return
    if hasattr(motor, "move_to"):
        motor.move_to(0.0, blocking=True)
        return
    raise AttributeError("Motor object has neither home() nor move_to()")
