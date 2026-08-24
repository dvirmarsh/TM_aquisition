"""Smoke tests for the published library. Reuses helpers from demo.py."""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_published_modules():
    import TM_retrieve
    import TM_retrieval_algos
    import metrics_losses
    import mode_solver
    import motor_pos_utils
    import utils
    import demo  # noqa: F401

    assert hasattr(TM_retrieve, "TMMeasurementSystem")
    assert hasattr(TM_retrieval_algos, "optimize_tm_admm_dual_plane")
    assert hasattr(TM_retrieval_algos, "optimize_tm_slm")
    assert hasattr(metrics_losses, "PearsonCorrelationLoss")
    assert hasattr(utils, "load_or_calculate_mode_fields")
    assert hasattr(utils, "sample_tm_perturbed_mmf")
    assert hasattr(motor_pos_utils, "sample_random_subset_indices")
    assert hasattr(mode_solver, "find_guided_modes")


def test_lab_scripts_parse():
    lab = REPO_ROOT / "lab"
    for path in sorted(lab.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_admm_fits_synthetic_dual_plane_data(tmp_path):
    from demo import run_demo

    result = run_demo(
        device="cpu",
        epochs=8,
        num_measurements=None,
        seed=0,
        output_dir=tmp_path,
        cache_dir=tmp_path / "mode_fields",
        budget_multiple=1.5,
    )
    assert result["loss_last"] < result["loss_first"]
    assert result["pcc_both_image"] > 0.5
    assert result["pcc_both_fourier"] > 0.5
    assert result["n_modes"] > 0
