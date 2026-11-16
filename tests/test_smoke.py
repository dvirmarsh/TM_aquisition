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
    for path in sorted(lab.rglob("*.py")):
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


def test_fourier_camera_weight(tmp_path):
    """b = 0 is the polarization-summed model; b != 0 reweights the Fourier plane only."""
    import torch

    from demo import build_synthetic_system

    system = build_synthetic_system(device="cpu", seed=0, cache_dir=tmp_path / "mode_fields")
    system.calibrate()
    torch.manual_seed(0)
    n = system.grid_params["ny"]
    slm = torch.rand(3, n, n) * 2 * torch.pi
    with torch.no_grad():
        fields = system.forward_model_slm(slm, return_complex_field=True)   # (B, plane, pol, H, W)
        summed = system.forward_model_slm(slm)
        assert system.fourier_weight_b() == 0.0
        assert torch.allclose(summed[:, 1], fields[:, 1, 0].abs() ** 2 + fields[:, 1, 1].abs() ** 2, rtol=1e-4, atol=1e-8)

        b = 0.4
        system.fourier_pol_weights = (1.0 + b, 1.0 - b)
        weighted = system.forward_model_slm(slm)
        assert abs(system.fourier_weight_b() - b) < 1e-6
        assert torch.allclose(weighted[:, 0], summed[:, 0])                  # image plane unchanged
        expected = (1 + b) * fields[:, 1, 0].abs() ** 2 + (1 - b) * fields[:, 1, 1].abs() ** 2
        assert torch.allclose(weighted[:, 1], expected, rtol=1e-4, atol=1e-8)

        system.enable_learnable_fourier_weights(b_init=-0.3)
        learnable = system.forward_model_slm(slm)
        expected = 0.7 * fields[:, 1, 0].abs() ** 2 + 1.3 * fields[:, 1, 1].abs() ** 2
        assert torch.allclose(learnable[:, 1], expected, rtol=1e-4, atol=1e-8)
    assert "fourier_weight_beta" in dict(system.named_parameters())
    assert abs(system.fourier_weight_b() + 0.3) < 1e-6
