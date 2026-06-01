"""
Standalone TM experimental pipeline with mathematically corrected ALS variants.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from TM_retrieve import (
    TMMeasurementSystem,
    simulate_phase_stepped_measurements,
    extract_complex_interference_4step,
    optimize_tm_analytic_dual_plane,
    optimize_tm_mse_exact_line_search_dual_plane,
)
from metrics_losses import calculate_phase_aligned_frobenius_norm, PearsonCorrelationLoss
from raf21_tm import RAF21Options, raf21_tm


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_default_params(grid_n: int = 128) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    wavelength = 632.8e-9
    core_radius = 25e-6
    na = 0.22
    cladding_index = 1.457
    core_index = np.sqrt(cladding_index**2 + na**2)
    margin = 1.1
    x_range = core_radius * margin
    y_range = x_range
    f = 200e-3
    x_max = f * na
    dx_i = 2 * x_range / (grid_n - 1)
    dx_o = 2 * x_max * margin / (grid_n - 1)
    n_pad = round(wavelength * f / (dx_i * dx_o))
    mult_factor_slm = dx_o * dx_i / (wavelength * f)
    mult_factor_out = dx_i * dx_o / (wavelength * f)

    fiber_params = {
        "a": core_radius,
        "n1": core_index,
        "n0": cladding_index,
        "wavelength": wavelength,
    }
    grid_params = {
        "x_range": (-x_range, x_range),
        "y_range": (-y_range, y_range),
        "nx": grid_n,
        "ny": grid_n,
    }
    zoomfft_params = {}
    for side in ["slm", "tran"]:
        zoomfft_params[side] = {
            "a_init": 2 * np.pi / n_pad,
            "n0_init": (grid_n - 1) / 2,
            "m0_init": (grid_n - 1) / 2,
            "k0_init": (grid_n - 1) / 2,
            "l0_init": (grid_n - 1) / 2,
        }
    mult_factor = {"slm": mult_factor_slm, "tran": mult_factor_out}
    dx = {"im": dx_i, "four": dx_o, "slm": dx_o}
    return fiber_params, grid_params, zoomfft_params, mult_factor, dx


def make_tm_system(
    *,
    fiber_params: Dict,
    grid_params: Dict,
    zoomfft_params: Dict,
    mult_factor: Dict,
    dx: Dict,
    device: str,
    enforce_symmetric_tm: bool = False,
    theta_before_slm_init: float = np.pi / 8,
) -> TMMeasurementSystem:
    return TMMeasurementSystem(
        fiber_params=fiber_params,
        grid_params=grid_params,
        device=device,
        mult_factor=mult_factor,
        zoomfft_params=zoomfft_params,
        zernike_n_modes=0,
        dx=dx,
        enforce_symmetric_tm=enforce_symmetric_tm,
        theta_before_slm_init=theta_before_slm_init,
    )


def bytes_to_gb(v: int) -> float:
    return float(v) / (1024.0**3)


def current_peak_memory_gb(device: str) -> float:
    cpu_peak = tracemalloc.get_traced_memory()[1]
    if device == "cuda" and torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated()
        return max(bytes_to_gb(cpu_peak), bytes_to_gb(gpu_peak))
    return bytes_to_gb(cpu_peak)


def get_first_threshold_epoch(history: Dict, threshold: float) -> Optional[int]:
    errs = history.get("tm_error", [])
    for i, v in enumerate(errs, start=1):
        if float(v) < threshold:
            return i
    return None


def generate_dataset(
    tm_system: TMMeasurementSystem,
    n_train: int,
    n_test: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    n_total = n_train + n_test
    h = tm_system.grid_params["ny"]
    w = tm_system.grid_params["nx"]
    base_slm = torch.rand(n_total, h, w, device=tm_system.device) * 2 * np.pi
    with torch.no_grad():
        intens_4step = simulate_phase_stepped_measurements(
            tm_system,
            base_slm,
            tm=tm_system.real_tm_tensor,
            use_internal_params=True,
            apply_slm_amplitude=True,
        )
        complex_targets = extract_complex_interference_4step(intens_4step)
        raw_i0 = intens_4step[:, 0]
        ref_intensity = tm_system.forward_model_slm(
            torch.zeros(1, h, w, device=tm_system.device),
            active_polarization="y_only",
            tm=tm_system.real_tm_tensor,
            use_internal_params=True,
            apply_slm_amplitude=True,
        ).squeeze(0)
    return {
        "train_slm": base_slm[:n_train],
        "test_slm": base_slm[n_train:],
        "train_complex": complex_targets[:n_train],
        "test_complex": complex_targets[n_train:],
        "train_raw": raw_i0[:n_train],
        "test_raw": raw_i0[n_train:],
        "reference_intensity": ref_intensity,
        "all_intens_4step": intens_4step,
    }


@dataclass
class ExperimentResult:
    name: str
    converged: bool
    time_to_convergence_sec: Optional[float]
    total_wallclock_sec: float
    epochs_to_convergence: Optional[int]
    final_tm_error: float
    peak_memory_gb: float
    first_epoch_gt_focus_500: Optional[int]
    final_gt_focus: float
    matrix_shapes: Dict[str, Tuple[int, ...] | float]
    setup_notes: str
    methodology_notes: str


def _projection_mats(tm_system: TMMeasurementSystem, plane_indices: List[int]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    n_modes = len(tm_system.guided_modes)
    I_modes = torch.eye(n_modes, dtype=torch.complex64, device=tm_system.device)
    out = tm_system.output_modes_to_intensities(I_modes, return_complex_field=True, use_internal_params=True)
    p_out_x = []
    p_out_y = []
    for p in plane_indices:
        p_out_x.append(out[:, p, 0].reshape(n_modes, -1).T)  # (pixels, modes)
        p_out_y.append(out[:, p, 1].reshape(n_modes, -1).T)  # (pixels, modes)
    return p_out_x, p_out_y


def _solve_regularized(A: torch.Tensor, B: torch.Tensor, ridge_alpha: float) -> torch.Tensor:
    # Match TM_retrieve scaling: lambda * max(|diag(A)|) * I
    diag_abs_max = torch.max(torch.abs(torch.diag(A))).real
    reg_scale = ridge_alpha * torch.clamp(diag_abs_max, min=1e-12)
    reg = reg_scale * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A + reg, B)


def _focus_metric_from_intensity_map(intensity_hw: torch.Tensor, center_radius: int) -> torch.Tensor:
    h, w = intensity_hw.shape
    cy, cx = h // 2, w // 2
    center_mask = torch.zeros((h, w), dtype=torch.bool, device=intensity_hw.device)
    y_min, y_max = max(0, cy - center_radius), min(h, cy + center_radius + 1)
    x_min, x_max = max(0, cx - center_radius), min(w, cx + center_radius + 1)
    center_mask[y_min:y_max, x_min:x_max] = True
    background_mask = ~center_mask
    center_mean = intensity_hw[center_mask].mean()
    background_mean = intensity_hw[background_mask].mean()
    return center_mean / (background_mean)


def evaluate_gt_focus_metric(
    tm_system: TMMeasurementSystem,
    slm_shape: Tuple[int, int],
    num_iterations: int,
    lr: float,
    center_radius: int,
    target_plane: str,
) -> float:
    device = tm_system.device
    prev_reqs = [p.requires_grad for p in tm_system.parameters()]
    for p in tm_system.parameters():
        p.requires_grad_(False)
    slm_phase = torch.nn.Parameter(2 * torch.pi * torch.rand((1, *slm_shape), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([slm_phase], lr=lr)
    plane_idx = 0 if target_plane.lower() == "image" else 1
    with torch.enable_grad():
        for _ in range(num_iterations):
            optimizer.zero_grad()
            pred_intensity = tm_system.forward_model_slm(
                slm_phase,
                tm=tm_system.optimized_tm,
                use_internal_params=True,
                apply_slm_amplitude=True,
                active_polarization="x_only",
            )
            focus_metric = _focus_metric_from_intensity_map(pred_intensity[0, plane_idx], center_radius)
            (-focus_metric).backward()
            optimizer.step()
    with torch.no_grad():
        optimized_phase = torch.remainder(slm_phase.detach(), 2 * torch.pi)
        pred_gt = tm_system.forward_model_slm(
            optimized_phase,
            tm=tm_system.real_tm_tensor,
            use_internal_params=True,
            apply_slm_amplitude=True,
            active_polarization="x_only",
        )
        gt_focus = _focus_metric_from_intensity_map(pred_gt[0, plane_idx], center_radius)
    for p, req in zip(tm_system.parameters(), prev_reqs):
        p.requires_grad_(req)
    return float(gt_focus.item())


def optimize_dual_plane_corrected(
    tm_system: TMMeasurementSystem,
    dataset: Dict[str, torch.Tensor],
    num_epochs: int,
    ridge_alpha: float,
    ref_weight: float,
    focus_opt_iters: int,
    focus_opt_lr: float,
    focus_center_radius: int,
    focus_target_plane: str,
) -> Dict:
    history = {"tm_error": [], "ls_loss": [], "gt_focus_metric": []}
    device = tm_system.device
    N = dataset["train_slm"].shape[0]
    n_modes = len(tm_system.guided_modes)
    dummy = torch.zeros(1, dataset["train_slm"].shape[1], dataset["train_slm"].shape[2], device=device)
    with torch.no_grad():
        Z_sig = tm_system.slm_to_input_modes(dataset["train_slm"], active_polarization="x_only", use_internal_params=True, apply_slm_amplitude=True)
        Z_ref = tm_system.slm_to_input_modes(dummy, active_polarization="y_only", use_internal_params=True, apply_slm_amplitude=True)
    p_out_x, p_out_y = _projection_mats(tm_system, [0, 1])
    W = [dataset["train_complex"][:, p].reshape(N, -1) for p in [0, 1]]  # (batch, pixels)
    W_ref_base = [dataset["reference_intensity"][p].reshape(-1) for p in [0, 1]]
    T = tm_system.optimized_tm.detach().clone()
    # Hardcoded ALS-only budget per directive.
    for _ in range(num_epochs):
        with torch.no_grad():
            Y_sig = T @ Z_sig
            Y_ref = T @ Z_ref

            # Phase A.1 corrected dual-plane: sum operators before Gram.
            AHA = torch.zeros((n_modes, n_modes), dtype=torch.complex64, device=device)
            AHW = torch.zeros((n_modes, N), dtype=torch.complex64, device=device)
            # gamma weighting applied ONLY after Gram/target construction.
            gamma = torch.mean(torch.abs(W[0]) ** 2) / (torch.mean(torch.abs(W[1]) ** 2))*8e-2
            for p in [0, 1]:
                weight = 1.0 if p == 0 else gamma
                E_rx = p_out_x[p] @ Y_ref
                E_ry = p_out_y[p] @ Y_ref
                A_ref_p = torch.conj(E_rx) * p_out_x[p] + torch.conj(E_ry) * p_out_y[p]
                A_H_A_p = A_ref_p.T.conj() @ A_ref_p
                W_p = W[p]  # (batch, pixels)
                A_H_W_p = A_ref_p.T.conj() @ W_p.T
                AHA = AHA + weight * A_H_A_p
                AHW = AHW + weight * A_H_W_p
            Y_sig = _solve_regularized(AHA, AHW, ridge_alpha)

            # Phase A.2: batch-aware Y_ref update; sum over batch and planes.
            M_tot = torch.zeros((n_modes, n_modes), dtype=torch.complex64, device=device)
            V_tot = torch.zeros((n_modes, 1), dtype=torch.complex64, device=device)
            for p in [0, 1]:
                weight = 1.0 if p == 0 else gamma
                E_sx = p_out_x[p] @ Y_sig
                E_sy = p_out_y[p] @ Y_sig
                I_sx = torch.sum(torch.abs(E_sx) ** 2, dim=1)
                I_sy = torch.sum(torch.abs(E_sy) ** 2, dim=1)
                I_sxy = torch.sum(E_sx * torch.conj(E_sy), dim=1)
                p_x_h = torch.conj(p_out_x[p].T)
                p_y_h = torch.conj(p_out_y[p].T)
                M_p = (
                    p_x_h @ (I_sx.unsqueeze(1) * p_out_x[p])
                    + p_y_h @ (I_sy.unsqueeze(1) * p_out_y[p])
                    + p_x_h @ (I_sxy.unsqueeze(1) * p_out_y[p])
                    + p_y_h @ (torch.conj(I_sxy).unsqueeze(1) * p_out_x[p])
                )
                S_x = torch.sum(E_sx * torch.conj(W[p].T), dim=1)
                S_y = torch.sum(E_sy * torch.conj(W[p].T), dim=1)
                V_p = p_x_h @ S_x.unsqueeze(1) + p_y_h @ S_y.unsqueeze(1)
                M_tot = M_tot + weight * M_p
                V_tot = V_tot + weight * V_p
            Y_ref = _solve_regularized(M_tot, V_tot, ridge_alpha)

            # Exact amplitude anchor from isolated reference intensity.
            I_ref_base = []
            for p in [0, 1]:
                E_ref_x = p_out_x[p] @ Y_ref
                E_ref_y = p_out_y[p] @ Y_ref
                I_ref_base.append((torch.abs(E_ref_x) ** 2 + torch.abs(E_ref_y) ** 2).reshape(-1))
            numerator = torch.tensor(0.0, device=device)
            denominator = torch.tensor(1e-12, device=device)
            for p in [0, 1]:
                weight = 1.0 if p == 0 else gamma
                numerator = numerator + weight * torch.sum(I_ref_base[p] * W_ref_base[p])
                denominator = denominator + weight * torch.sum(I_ref_base[p] ** 2)
            g_sq = numerator / denominator
            g_val = torch.sqrt(torch.clamp(g_sq, min=1e-8, max=1e8))
            g_final = g_val.to(torch.complex64)
            Y_sig = Y_sig / g_final
            Y_ref = Y_ref * g_final

            # TM update with swept reference weight.
            wref = math.sqrt(N / 3.0) * math.sqrt(ref_weight)
            Z_comb = torch.cat([Z_sig, Z_ref * wref], dim=1)
            Y_comb = torch.cat([Y_sig, Y_ref * wref], dim=1)
            ZHZ = Z_comb @ Z_comb.T.conj()
            T = _solve_regularized(ZHZ, Z_comb @ Y_comb.T.conj(), ridge_alpha).T.conj()
            tm_system.optimized_tm_tensor = T
            tm_err, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, tm_system.optimized_tm)
            history["tm_error"].append(float(tm_err))
            # proxy LS loss
            ls_loss = 0.0
            for p in [0, 1]:
                Yp = (p_out_x[p] @ Y_sig) * torch.conj(p_out_x[p] @ Y_ref) + (p_out_y[p] @ Y_sig) * torch.conj(p_out_y[p] @ Y_ref)
                ls_loss += float(torch.mean(torch.abs(Yp.T - W[p]) ** 2))
            history["ls_loss"].append(ls_loss)
            focus_gt = evaluate_gt_focus_metric(
                tm_system,
                slm_shape=(dataset["train_slm"].shape[1], dataset["train_slm"].shape[2]),
                num_iterations=focus_opt_iters,
                lr=focus_opt_lr,
                center_radius=focus_center_radius,
                target_plane=focus_target_plane,
            )
            history["gt_focus_metric"].append(focus_gt)
            # Keep fixed 20 ALS epochs by design.
    return history


def generate_pbs_dataset(tm_system: TMMeasurementSystem, base_slm: torch.Tensor) -> Dict[str, torch.Tensor]:
    phases = [0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]
    I_x = []
    I_y = []
    with torch.no_grad():
        for ph in phases:
            E = tm_system.forward_model_slm(
                base_slm + ph,
                tm=tm_system.real_tm_tensor,
                use_internal_params=True,
                apply_slm_amplitude=True,
                active_polarization="both",
                return_complex_field=True,
            )
            E_im_x = E[:, 0, 0]
            E_im_y = E[:, 0, 1]
            I_x.append(torch.abs(E_im_x) ** 2)
            I_y.append(torch.abs(E_im_y) ** 2)
        I_x = torch.stack(I_x, dim=1)
        I_y = torch.stack(I_y, dim=1)
        W_x = ((I_x[:, 0] - I_x[:, 2]) / 2.0) + 1j * ((I_x[:, 3] - I_x[:, 1]) / 2.0)
        W_y = ((I_y[:, 0] - I_y[:, 2]) / 2.0) + 1j * ((I_y[:, 3] - I_y[:, 1]) / 2.0)

        # True isolated reference intensities (y-only arm, zero SLM phase).
        _, h_slm, w_slm = base_slm.shape
        dummy_slm = torch.zeros(1, h_slm, w_slm, device=tm_system.device, dtype=base_slm.dtype)
        E_ref = tm_system.forward_model_slm(
            dummy_slm,
            tm=tm_system.real_tm_tensor,
            use_internal_params=True,
            apply_slm_amplitude=True,
            active_polarization="y_only",
            return_complex_field=True,
        )
        E_im_x = E_ref[:, 0, 0]
        E_im_y = E_ref[:, 0, 1]
        ref_x = torch.abs(E_im_x) ** 2
        ref_y = torch.abs(E_im_y) ** 2

    return {
        "complex_x": W_x,
        "complex_y": W_y,
        "ref_x": ref_x,
        "ref_y": ref_y,
    }


def optimize_pbs_single_plane_corrected(
    tm_system: TMMeasurementSystem,
    dataset: Dict[str, torch.Tensor],
    pbs: Dict[str, torch.Tensor],
    num_epochs: int,
    ridge_alpha: float,
    ref_weight: float,
    focus_opt_iters: int,
    focus_opt_lr: float,
    focus_center_radius: int,
    focus_target_plane: str,
) -> Dict:
    history = {"tm_error": [], "ls_loss": [], "gt_focus_metric": []}
    device = tm_system.device
    N = dataset["train_slm"].shape[0]
    n_modes = len(tm_system.guided_modes)
    dummy = torch.zeros(1, dataset["train_slm"].shape[1], dataset["train_slm"].shape[2], device=device)
    with torch.no_grad():
        Z_sig = tm_system.slm_to_input_modes(dataset["train_slm"], active_polarization="x_only", use_internal_params=True, apply_slm_amplitude=True)
        Z_ref = tm_system.slm_to_input_modes(dummy, active_polarization="y_only", use_internal_params=True, apply_slm_amplitude=True)

    p_out_x_l, p_out_y_l = _projection_mats(tm_system, [0])
    p_out_x = p_out_x_l[0]
    p_out_y = p_out_y_l[0]
    W_x = pbs["complex_x"][:N].reshape(N, -1)  # (batch, pixels)
    W_y = pbs["complex_y"][:N].reshape(N, -1)  # (batch, pixels)
    ref_x_meas = pbs["ref_x"].reshape(-1)
    ref_y_meas = pbs["ref_y"].reshape(-1)
    P_H_P = p_out_x.T.conj() @ p_out_x + p_out_y.T.conj() @ p_out_y

    T = tm_system.optimized_tm.detach().clone()
    # Hardcoded ALS-only budget per directive.
    weights = np.linspace(-4, -3, num_epochs)
    for i in range(num_epochs):
        with torch.no_grad():
            Y_sig = T @ Z_sig
            Y_ref = T @ Z_ref

            # Phase A.1 PBS corrected (independent x/y operators).
            E_rx = p_out_x @ Y_ref
            E_ry = p_out_y @ Y_ref
            phase_x = torch.exp(1j * torch.angle(E_rx))
            phase_y = torch.exp(1j * torch.angle(E_ry))
            E_rx = torch.sqrt(ref_x_meas).unsqueeze(1) * phase_x
            E_ry = torch.sqrt(ref_y_meas).unsqueeze(1) * phase_y
            A_ref_x = torch.conj(E_rx) * p_out_x
            A_ref_y = torch.conj(E_ry) * p_out_y
            A_H_A_x = A_ref_x.T.conj() @ A_ref_x
            A_H_A_y = A_ref_y.T.conj() @ A_ref_y
            A_H_W_x = A_ref_x.T.conj() @ W_x.T
            A_H_W_y = A_ref_y.T.conj() @ W_y.T
            AHA = A_H_A_x + A_H_A_y
            AHW = A_H_W_x + A_H_W_y
            Y_sig = _solve_regularized(AHA, AHW, ridge_alpha)

            # Phase A.2 PBS corrected with batch-summed normal equations.
            E_sx = p_out_x @ Y_sig
            E_sy = p_out_y @ Y_sig
            I_sx = torch.sum(torch.abs(E_sx) ** 2, dim=1)
            I_sy = torch.sum(torch.abs(E_sy) ** 2, dim=1)
            p_x_h = torch.conj(p_out_x.T)
            p_y_h = torch.conj(p_out_y.T)
            M_sig_x = p_x_h @ (I_sx.unsqueeze(1) * p_out_x)
            M_sig_y = p_y_h @ (I_sy.unsqueeze(1) * p_out_y)
            S_x = torch.sum(E_sx * torch.conj(W_x.T), dim=1)
            S_y = torch.sum(E_sy * torch.conj(W_y.T), dim=1)
            V_sig_x = p_x_h @ S_x.unsqueeze(1)
            V_sig_y = p_y_h @ S_y.unsqueeze(1)
            Y_ref = _solve_regularized(M_sig_x + M_sig_y, V_sig_x + V_sig_y, ridge_alpha)

            # Gerchberg–Saxton-style amplitude replacement + mode projection (isolated reference).
            E_rx_est = p_out_x @ Y_ref
            E_ry_est = p_out_y @ Y_ref
            phase_x = torch.exp(1j * torch.angle(E_rx_est))
            phase_y = torch.exp(1j * torch.angle(E_ry_est))
            E_rx_ideal = torch.sqrt(ref_x_meas).unsqueeze(1) * phase_x
            E_ry_ideal = torch.sqrt(ref_y_meas).unsqueeze(1) * phase_y
            P_H_E = p_out_x.T.conj() @ E_rx_ideal + p_out_y.T.conj() @ E_ry_ideal
            Y_ref = _solve_regularized(P_H_P, P_H_E, ridge_alpha)

            wref = math.sqrt(N / 3.0) * math.sqrt(ref_weight)
            Z_comb = torch.cat([Z_sig, Z_ref * wref], dim=1)
            Y_comb = torch.cat([Y_sig, Y_ref * wref], dim=1)
            ZHZ = Z_comb @ Z_comb.T.conj()
            T = _solve_regularized(ZHZ, Z_comb @ Y_comb.T.conj(), ridge_alpha).T.conj()
            tm_system.optimized_tm_tensor = T

        with torch.no_grad():
            tm_err, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, tm_system.optimized_tm)
            history["tm_error"].append(float(tm_err))
            ls_loss = float(torch.mean(torch.abs((p_out_x @ Y_sig).T * torch.conj((p_out_x @ Y_ref).T) - W_x) ** 2))
            ls_loss += float(torch.mean(torch.abs((p_out_y @ Y_sig).T * torch.conj((p_out_y @ Y_ref).T) - W_y) ** 2))
            history["ls_loss"].append(ls_loss)
            focus_gt = evaluate_gt_focus_metric(
                tm_system,
                slm_shape=(dataset["train_slm"].shape[1], dataset["train_slm"].shape[2]),
                num_iterations=focus_opt_iters,
                lr=focus_opt_lr,
                center_radius=focus_center_radius,
                target_plane=focus_target_plane,
            )
            history["gt_focus_metric"].append(focus_gt)
            # Keep fixed 20 ALS epochs by design.
    return history


def generate_intensity_dataset_with_input_modes(
    tm_system: TMMeasurementSystem,
    n_train: int,
    n_test: int,
    seed: int,
    forward_batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    set_seed(seed)
    n_total = n_train + n_test
    h = tm_system.grid_params["ny"]
    w = tm_system.grid_params["nx"]
    base_slm = torch.rand(n_total, h, w, device=tm_system.device) * 2 * np.pi
    with torch.no_grad():
        input_mode_batches = []
        intensity_batches = []
        for start in range(0, n_total, forward_batch_size):
            end = min(start + forward_batch_size, n_total)
            slm_chunk = base_slm[start:end]
            input_modes_chunk = tm_system.slm_to_input_modes(
                slm_chunk,
                use_internal_params=True,
                apply_slm_amplitude=True,
                active_polarization="x_only",
            )
            output_modes_chunk = tm_system.input_modes_to_output_modes(input_modes_chunk, tm=tm_system.real_tm_tensor)
            intensity_chunk = tm_system.output_modes_to_intensities(output_modes_chunk, use_internal_params=True)
            input_mode_batches.append(input_modes_chunk)
            intensity_batches.append(intensity_chunk)
        input_modes = torch.cat(input_mode_batches, dim=1)
        intensity_targets = torch.cat(intensity_batches, dim=0)
    return {
        "train_input_modes": input_modes[:, :n_train],
        "test_input_modes": input_modes[:, n_train:],
        "train_intensity": intensity_targets[:n_train],
        "test_intensity": intensity_targets[n_train:],
    }


def _upsample_phase_kron(phase_low: torch.Tensor, ratio_h: int, ratio_w: int) -> torch.Tensor:
    """Replicate each low-res pixel into a ratio_h x ratio_w block (equivalent to
    ``kron(phase, ones(ratio_h, ratio_w))`` along the trailing two spatial axes)."""
    return phase_low.repeat_interleave(ratio_h, dim=-2).repeat_interleave(ratio_w, dim=-1)


def _downsample_intensity_avg(intensity: torch.Tensor, ratio_h: int, ratio_w: int) -> torch.Tensor:
    """Block-mean downsample the trailing two spatial axes by (ratio_h, ratio_w)."""
    if ratio_h == 1 and ratio_w == 1:
        return intensity
    leading = intensity.shape[:-2]
    h, w = intensity.shape[-2], intensity.shape[-1]
    flat = intensity.reshape(-1, 1, h, w)
    pooled = torch.nn.functional.avg_pool2d(flat, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w))
    return pooled.reshape(*leading, h // ratio_h, w // ratio_w)


def _resolve_raf21_ratios(
    tm_system: TMMeasurementSystem,
    n_pix_slm: Optional[int],
    n_pix_out: Optional[int],
) -> Tuple[int, int, int, int, int, int]:
    """Validate and resolve (n_pix_slm, n_pix_out) against the system grid; return
    (n_pix_slm, n_pix_out, ratio_slm_h, ratio_slm_w, ratio_out_h, ratio_out_w)."""
    h = int(tm_system.grid_params["ny"])
    w = int(tm_system.grid_params["nx"])
    if n_pix_slm is None or n_pix_slm <= 0:
        n_pix_slm = h
    if n_pix_out is None or n_pix_out <= 0:
        n_pix_out = h
    if h % n_pix_slm != 0 or w % n_pix_slm != 0:
        raise ValueError(
            f"raf21_n_pix_slm={n_pix_slm} must divide the SLM grid ({h}x{w})."
        )
    if h % n_pix_out != 0 or w % n_pix_out != 0:
        raise ValueError(
            f"raf21_n_pix_out={n_pix_out} must divide the output grid ({h}x{w})."
        )
    return (
        n_pix_slm,
        n_pix_out,
        h // n_pix_slm,
        w // n_pix_slm,
        h // n_pix_out,
        w // n_pix_out,
    )


def generate_raf21_dataset(
    tm_system: TMMeasurementSystem,
    n_train: int,
    seed: int,
    n_pix_slm: Optional[int] = None,
    n_pix_out: Optional[int] = None,
    forward_batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Build the RAF2-1 pixel-basis dataset (Cheng et al., 2023) at user-chosen
    resolutions for the SLM input (n_pix_slm x n_pix_slm) and the camera output
    (n_pix_out x n_pix_out). Both must divide the system's physical SLM grid.

    Pipeline:
      - Sample low-res unit-amp SLM phases (n_train, n_pix_slm, n_pix_slm).
      - Kronecker-upsample by `ratio_slm = grid_n / n_pix_slm` so each low-res pixel
        becomes a constant-phase ratio_slm x ratio_slm block on the physical SLM.
      - Run the full forward model with the GT TM and `active_polarization="x_only"`.
      - Take the image-plane x-polarization intensity (single output polarization, as
        required by RAF2-1), and average-pool it down to (n_pix_out, n_pix_out).

    The returned matrix shapes match the RAF2-1 convention:
      - y_intensity: (n_pix_out^2, n_train), the algorithm's "N^2 x P".
      - x_probes:    (n_pix_slm^2, n_train), the algorithm's "M^2 x P".
    """
    n_pix_slm, n_pix_out, r_slm_h, r_slm_w, r_out_h, r_out_w = _resolve_raf21_ratios(
        tm_system, n_pix_slm, n_pix_out
    )
    set_seed(seed)
    device = tm_system.device
    base_slm_low = torch.rand(n_train, n_pix_slm, n_pix_slm, device=device) * 2 * np.pi
    intensity_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n_train, forward_batch_size):
            end = min(start + forward_batch_size, n_train)
            slm_full = _upsample_phase_kron(base_slm_low[start:end], r_slm_h, r_slm_w)
            E = tm_system.forward_model_slm(
                slm_full,
                tm=tm_system.real_tm_tensor,
                use_internal_params=True,
                apply_slm_amplitude=True,
                active_polarization="x_only",
                return_complex_field=True,
            )
            # E shape: (batch, 2, 2, H, W) -> [plane=0=image, pol=0=x].
            I_full = (E[:, 0, 0].abs() ** 2)
            I_down = _downsample_intensity_avg(I_full, r_out_h, r_out_w)
            intensity_chunks.append(I_down.contiguous())
        intensity_low = torch.cat(intensity_chunks, dim=0)  # (n_train, n_pix_out, n_pix_out)
    # Match MATLAB `SpekAmp ./ max(...)` normalization: rescale amplitudes (not intensities)
    # so that max(amplitude) = 1. Phase information is preserved; the algorithm becomes
    # numerically stable for any output dynamic range.
    amp = intensity_low.clamp(min=0).sqrt()
    amp_max = amp.max().clamp(min=1e-30)
    amp_norm = amp / amp_max
    y_intensity = (amp_norm ** 2).reshape(n_train, -1).T.contiguous()  # (n_pix_out^2, n_train)
    x_probes = torch.exp(1j * base_slm_low).reshape(n_train, -1).T.contiguous()  # (n_pix_slm^2, n_train)
    return {
        "slm_phase_low": base_slm_low,
        "y_intensity": y_intensity,
        "x_probes": x_probes,
        "amp_max": amp_max.detach(),
        "n_pix_slm": n_pix_slm,
        "n_pix_out": n_pix_out,
        "ratio_slm": (r_slm_h, r_slm_w),
        "ratio_out": (r_out_h, r_out_w),
    }


def evaluate_raf21_focus_metric(
    tm_system: TMMeasurementSystem,
    a_pixel: torch.Tensor,
    n_pix_slm: int,
    n_pix_out: int,
    num_iterations: int,
    lr: float,
    center_radius: int,
    target_plane: str,
) -> Dict[str, float]:
    """
    Optimize the SLM phase against the RAF2-1 retrieved pixel-basis TM `a_pixel`, then
    validate the resulting SLM on the GT system in the same low-resolution space the
    algorithm operates in.

    Both the SLM optimization and the GT validation use a focus metric on an
    (n_pix_out x n_pix_out) intensity map: the SLM optimization uses the low-res TM
    directly; the validation upsamples the optimized low-res phase to the physical SLM
    grid (Kronecker product with ones(ratio_slm)), runs the full forward model with the
    GT TM, and average-pools the resulting full-res image-plane x-polarization intensity
    back to (n_pix_out x n_pix_out) before measuring focus.

    Returns:
        {
            "gt_focus_low": focus measured on the binned (n_pix_out x n_pix_out) intensity
                            (directly comparable to what the algorithm sees);
            "gt_focus_full": focus measured on the full physical-resolution intensity, with
                            the center mask scaled by ratio_out so the same physical region
                            is used (sanity check on the high-res sensor).
        }
    """
    n_pix_slm, n_pix_out, r_slm_h, r_slm_w, r_out_h, r_out_w = _resolve_raf21_ratios(
        tm_system, n_pix_slm, n_pix_out
    )
    device = tm_system.device
    a_pixel = a_pixel.to(device=device, dtype=torch.complex64)
    if a_pixel.shape != (n_pix_out * n_pix_out, n_pix_slm * n_pix_slm):
        raise ValueError(
            "a_pixel shape mismatch: expected "
            f"{(n_pix_out * n_pix_out, n_pix_slm * n_pix_slm)}, got {tuple(a_pixel.shape)}"
        )

    prev_reqs = [p.requires_grad for p in tm_system.parameters()]
    for p in tm_system.parameters():
        p.requires_grad_(False)
    slm_phase_low = torch.nn.Parameter(
        2 * torch.pi * torch.rand((n_pix_slm, n_pix_slm), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([slm_phase_low], lr=lr)
    plane_idx = 0 if target_plane.lower() == "image" else 1
    a_pixel_detached = a_pixel.detach()
    with torch.enable_grad():
        for _ in range(num_iterations):
            optimizer.zero_grad()
            x_flat = torch.exp(1j * slm_phase_low).reshape(-1)
            e_out = a_pixel_detached @ x_flat
            i_pred = (e_out.abs() ** 2).reshape(n_pix_out, n_pix_out)
            focus_metric = _focus_metric_from_intensity_map(i_pred, center_radius)
            (-focus_metric).backward()
            optimizer.step()
    with torch.no_grad():
        phi_low = torch.remainder(slm_phase_low.detach(), 2 * torch.pi)
        phi_full = _upsample_phase_kron(phi_low.unsqueeze(0), r_slm_h, r_slm_w)
        pred_gt = tm_system.forward_model_slm(
            phi_full,
            tm=tm_system.real_tm_tensor,
            use_internal_params=True,
            apply_slm_amplitude=True,
            active_polarization="x_only",
            return_complex_field=True,
        )  # (1, 2, 2, H, W)
        I_full = (pred_gt[0, plane_idx, 0].abs() ** 2)  # (H, W)
        I_low = _downsample_intensity_avg(I_full, r_out_h, r_out_w)  # (n_pix_out, n_pix_out)
        gt_focus_low = _focus_metric_from_intensity_map(I_low, center_radius)
        # Full-resolution focus on the *same physical region*: scale the center radius by
        # the output-binning ratio so the center mask covers the equivalent area.
        full_radius = max(1, (2 * center_radius + 1) * max(r_out_h, r_out_w) // 2)
        gt_focus_full = _focus_metric_from_intensity_map(I_full, full_radius)
    for p, req in zip(tm_system.parameters(), prev_reqs):
        p.requires_grad_(req)
    return {
        "gt_focus_low": float(gt_focus_low.item()),
        "gt_focus_full": float(gt_focus_full.item()),
    }


def run_raf21_experiment(
    tm_system: TMMeasurementSystem,
    raf_dataset: Dict[str, torch.Tensor],
    iters: int,
    mu: float,
    ratio: float,
    beta: float,
    gamma: float,
    npower_iter: int,
    seed: int,
    focus_opt_iters: int,
    focus_opt_lr: float,
    focus_center_radius: int,
    focus_target_plane: str,
) -> Dict:
    """
    One-shot RAF2-1 retrieval + GT focus evaluation. RAF2-1 is run end-to-end (no per-epoch
    bookkeeping), so the history holds a single entry that matches the schema used by
    `run_experiment` (`tm_error`, `ls_loss`, `gt_focus_metric`).
    """
    device = tm_system.device
    y_intensity = raf_dataset["y_intensity"].to(device)
    x_probes = raf_dataset["x_probes"].to(device)
    n_pix_slm = int(raf_dataset["n_pix_slm"])
    n_pix_out = int(raf_dataset["n_pix_out"])
    opts = RAF21Options(mu=mu, iters=iters, ratio=ratio, beta=beta, gamma=gamma, npower_iter=npower_iter)
    gen = torch.Generator(device=device).manual_seed(seed)
    a_pixel = raf21_tm(y_intensity, x_probes, opts, generator=gen)

    # Final-iteration LS measurement error in the spirit of the MATLAB errCurve: sum over
    # output pixels of l2-norms between predicted and measured amplitudes, normalized.
    with torch.no_grad():
        amp_pred = (a_pixel @ x_probes).abs()
        amp_meas = y_intensity.clamp(min=0).sqrt()
        ls_err = float(torch.linalg.vector_norm(amp_pred - amp_meas, ord=2, dim=1).sum() / amp_pred.shape[0])

    focus_results = evaluate_raf21_focus_metric(
        tm_system,
        a_pixel,
        n_pix_slm=n_pix_slm,
        n_pix_out=n_pix_out,
        num_iterations=focus_opt_iters,
        lr=focus_opt_lr,
        center_radius=focus_center_radius,
        target_plane=focus_target_plane,
    )

    return {
        # RAF2-1 lives in the pixel basis and is not directly comparable to the mode-basis
        # Frobenius error tracked elsewhere; nan keeps the threshold logic from firing.
        "tm_error": [float("nan")],
        "ls_loss": [ls_err],
        # Report the low-res focus (the metric the algorithm actually optimizes) in the
        # primary `gt_focus_metric` series so it lines up with the other experiments;
        # the full-resolution number is kept on the side for inspection.
        "gt_focus_metric": [focus_results["gt_focus_low"]],
        "gt_focus_full_res": [focus_results["gt_focus_full"]],
    }


def _tm_trainable_params(tm_system: TMMeasurementSystem) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = [tm_system._optimized_tm_param]
    tm_system._optimized_tm_param.requires_grad_(True)
    if getattr(tm_system, "enforce_symmetric_tm", False):
        tm_system.diag_in.requires_grad_(True)
        tm_system.diag_out.requires_grad_(True)
        params.extend([tm_system.diag_in, tm_system.diag_out])
    return params


def _predict_intensity_from_input_modes(tm_system: TMMeasurementSystem, input_modes: torch.Tensor) -> torch.Tensor:
    output_modes = tm_system.input_modes_to_output_modes(input_modes, tm=tm_system.optimized_tm)
    return tm_system.output_modes_to_intensities(output_modes, use_internal_params=True)


def _predict_intensity_from_input_modes_chunked(
    tm_system: TMMeasurementSystem,
    input_modes: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    n = input_modes.shape[1]
    preds = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        preds.append(_predict_intensity_from_input_modes(tm_system, input_modes[:, start:end]))
    return torch.cat(preds, dim=0)


def optimize_tm_from_input_modes_adam(
    tm_system: TMMeasurementSystem,
    train_input_modes: torch.Tensor,
    train_intensity_targets: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    focus_opt_iters: int,
    focus_opt_lr: float,
    focus_center_radius: int,
    focus_target_plane: str,
    slm_shape: Tuple[int, int],
    test_input_modes: Optional[torch.Tensor] = None,
    test_intensity_targets: Optional[torch.Tensor] = None,
) -> Dict:
    criterion = PearsonCorrelationLoss(dim=(2, 3), center=True)
    optimizer = torch.optim.Adam(_tm_trainable_params(tm_system), lr=learning_rate)
    n_samples = train_intensity_targets.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    history = {"epoch_losses": [], "test_losses": [], "tm_error": [], "gt_focus_metric": []}

    for _ in range(num_epochs):
        total_epoch_loss = 0.0
        indices = torch.randperm(n_samples, device=train_intensity_targets.device)
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            batch_input_modes = train_input_modes[:, batch_idx].detach().contiguous()
            batch_targets = train_intensity_targets[batch_idx].detach().contiguous()
            optimizer.zero_grad()
            pred = _predict_intensity_from_input_modes(tm_system, batch_input_modes)
            loss = criterion(pred, batch_targets)
            loss.backward()
            optimizer.step()
            total_epoch_loss += float(loss.item())

        avg_loss = total_epoch_loss / max(1, n_batches)
        history["epoch_losses"].append(avg_loss)
        if test_input_modes is not None and test_intensity_targets is not None:
            with torch.no_grad():
                test_pred = _predict_intensity_from_input_modes_chunked(tm_system, test_input_modes, batch_size)
                history["test_losses"].append(float(criterion(test_pred, test_intensity_targets).item()))
        else:
            history["test_losses"].append(0.0)
        with torch.no_grad():
            tm_err, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, tm_system.optimized_tm)
            history["tm_error"].append(float(tm_err))
        focus_gt = evaluate_gt_focus_metric(
            tm_system,
            slm_shape=slm_shape,
            num_iterations=focus_opt_iters,
            lr=focus_opt_lr,
            center_radius=focus_center_radius,
            target_plane=focus_target_plane,
        )
        history["gt_focus_metric"].append(focus_gt)
    return history


def run_experiment(
    name: str,
    tm_system: TMMeasurementSystem,
    history: Dict,
    threshold: float,
    start: float,
    setup_notes: str,
    methodology_notes: str,
    shapes: Dict[str, Tuple[int, ...] | float],
) -> ExperimentResult:
    end = time.perf_counter()
    final_tm_error = float(history["tm_error"][-1]) if history.get("tm_error") else float("inf")
    gt_focus_hist = history.get("gt_focus_metric", [])
    final_gt_focus = float(gt_focus_hist[-1]) if gt_focus_hist else float("nan")
    first_gt_focus_500 = next((i + 1 for i, v in enumerate(gt_focus_hist) if float(v) > 500.0), None)
    hit_epoch = get_first_threshold_epoch(history, threshold)
    converged = hit_epoch is not None
    epoch_time = (end - start) / max(1, len(history.get("tm_error", [])))
    t_conv = epoch_time * hit_epoch if converged else None
    peak_mem = current_peak_memory_gb(tm_system.device)
    return ExperimentResult(
        name=name,
        converged=converged,
        time_to_convergence_sec=t_conv,
        total_wallclock_sec=end - start,
        epochs_to_convergence=hit_epoch,
        final_tm_error=final_tm_error,
        peak_memory_gb=peak_mem,
        first_epoch_gt_focus_500=first_gt_focus_500,
        final_gt_focus=final_gt_focus,
        matrix_shapes=shapes,
        setup_notes=setup_notes,
        methodology_notes=methodology_notes,
    )


def summarize_table(results: List[ExperimentResult]) -> str:
    header = (
        "| Experiment | Time to Convergence (s) | Epochs | Final TM Error | Peak Memory (GB) | First Epoch GT Focus > 500 | Final GT Focus |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for r in results:
        t = f"{r.time_to_convergence_sec:.2f}" if r.time_to_convergence_sec is not None else "N/A"
        e = str(r.epochs_to_convergence) if r.epochs_to_convergence is not None else "N/A"
        f500 = str(r.first_epoch_gt_focus_500) if r.first_epoch_gt_focus_500 is not None else "N/A"
        rows.append(
            f"| {r.name} | {t} | {e} | {r.final_tm_error:.4f} | {r.peak_memory_gb:.3f} | {f500} | {r.final_gt_focus:.2f} |"
        )
    return "\n".join([header] + rows)


def recommendation(results: List[ExperimentResult]) -> str:
    feasible = [r for r in results if r.converged and r.final_tm_error < 16.0]
    if not feasible:
        return "No configuration met TM error < 16. Increase epochs or revise initialization/masking."
    best = min(feasible, key=lambda r: r.time_to_convergence_sec if r.time_to_convergence_sec is not None else float("inf"))
    return f"Recommended production candidate: {best.name} (fastest convergent configuration under TM error < 16)."


def run_suite(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    fiber_params, grid_params, zoomfft_params, mult_factor, dx = build_default_params(args.grid_n)
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"
    epochs = args.epochs
    # Load baseline TM system from checkpoint instead of constructing it from init args.
    tm_system_path = args.tm_system_path
    if not os.path.isabs(tm_system_path):
        tm_system_path = os.path.join(os.getcwd(), tm_system_path)
    if not os.path.exists(tm_system_path):
        raise FileNotFoundError(f"TM system checkpoint not found: {tm_system_path}")
    loaded_tm_system = torch.load(tm_system_path, map_location=device, weights_only=False)
    if not isinstance(loaded_tm_system, TMMeasurementSystem):
        raise TypeError(f"Loaded object is not TMMeasurementSystem: {type(loaded_tm_system)}")
    loaded_tm_system = loaded_tm_system.to(device)

    # Match TM_retrieve.py behavior: fresh init physics + injected measured real TM.
    source_system = make_tm_system(
        fiber_params=fiber_params,
        grid_params=grid_params,
        zoomfft_params=zoomfft_params,
        mult_factor=mult_factor,
        dx=dx,
        device=device,
        enforce_symmetric_tm=False,
        theta_before_slm_init=np.pi / 8,
    )
    with torch.no_grad():
        source_system.real_tm_tensor.copy_(loaded_tm_system.real_tm_tensor)
    # loaded_tm_system is no longer needed after copying the GT TM.
    del loaded_tm_system
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    source_system.device = device
    source_system.calibrate()
    real_tm = source_system.real_tm_tensor.detach().clone()

    def fresh_system() -> TMMeasurementSystem:
        # Clone precomputed mode basis and modules once per experiment instead of
        # reconstructing everything from scratch.
        s = copy.deepcopy(source_system)
        with torch.no_grad():
            s.real_tm_tensor.copy_(real_tm)
            # Reset optimized TM to the class default initializer by creating a fresh tensor.
            s.optimized_tm_tensor = s.optimized_tm_tensor.detach().clone()
        s.calibrate()
        return s

    results: List[ExperimentResult] = []
    histories: Dict[str, Dict] = {}

    def cleanup_system(sys_obj: TMMeasurementSystem) -> None:
        del sys_obj
        _release_cuda_memory()

    def _release_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Exp 1 (corrected dual-plane ALS) and Exp 2 (corrected PBS single-plane ALS)
    # are disabled in this suite; their implementations remain available as
    # `optimize_dual_plane_corrected` / `optimize_pbs_single_plane_corrected`.

    # Experiment 3: intensity-based TM retrieval from precomputed input modes.
    src_intensity = copy.deepcopy(source_system)
    src_intensity = src_intensity.to(device)
    src_intensity.device = device
    src_intensity.calibrate()
    ds_intensity = generate_intensity_dataset_with_input_modes(
        src_intensity,
        args.n_train_intensity,
        args.n_test_intensity,
        args.seed,
        args.intensity_forward_batch_size,
    )
    sys_adam = copy.deepcopy(src_intensity)
    with torch.no_grad():
        sys_adam.real_tm_tensor.copy_(src_intensity.real_tm_tensor.detach())
        sys_adam.optimized_tm_tensor = sys_adam.optimized_tm_tensor.detach().clone()
    sys_adam.calibrate()
    tracemalloc.start()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    h_adam = optimize_tm_from_input_modes_adam(
        sys_adam,
        ds_intensity["train_input_modes"],
        ds_intensity["train_intensity"],
        epochs,
        args.learning_rate,
        args.batch_size,
        args.focus_opt_iters,
        args.focus_opt_lr,
        args.focus_center_radius,
        args.focus_target_plane,
        (grid_params["ny"], grid_params["nx"]),
        ds_intensity["test_input_modes"],
        ds_intensity["test_intensity"],
    )
    r_adam = run_experiment(
        "Exp 3: Adam",
        sys_adam,
        h_adam,
        args.tm_error_threshold,
        t0,
        "Intensity measurements only.",
        "Precompute input modes once; optimize TM using output-mode to intensity mapping.",
        {"train_intensity": tuple(ds_intensity["train_intensity"].shape)},
    )
    tracemalloc.stop()
    histories["exp3_adam"] = h_adam
    results.append(r_adam)
    cleanup_system(sys_adam)
    cleanup_system(src_intensity)
    _release_cuda_memory()

    # Experiment 5: Autograd-free analytic Wirtinger optimizer (dual-plane RAF).
    # Reuses the Exp 3 dataset (same seed / sample count) instead of regenerating
    # it, and runs before RAF2-1 so large pixel-basis allocations do not linger.
    sys_analytic = fresh_system()
    tracemalloc.start()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    h_analytic = optimize_tm_analytic_dual_plane(
        sys_analytic,
        ds_intensity["train_input_modes"],
        ds_intensity["train_intensity"],
        num_epochs=epochs,
        learning_rate=args.analytic_learning_rate,
        momentum=args.analytic_momentum,
        beta=args.analytic_beta,
        epsilon=1e-8,
    )
    focus_gt_analytic = evaluate_gt_focus_metric(
        sys_analytic,
        slm_shape=(grid_params["ny"], grid_params["nx"]),
        num_iterations=args.focus_opt_iters,
        lr=args.focus_opt_lr,
        center_radius=args.focus_center_radius,
        target_plane=args.focus_target_plane,
    )
    h_analytic["gt_focus_metric"] = [focus_gt_analytic]
    r_analytic = run_experiment(
        "Exp 5: Analytic RAF Dual-Plane",
        sys_analytic,
        h_analytic,
        args.tm_error_threshold,
        t0,
        "Dual-plane (image + Fourier) unpolarized intensity measurements, full-batch.",
        "Autograd-free Wirtinger-gradient update on T with RAF dynamic weighting; static plane balance gamma = L_0/L_1; Polyak momentum and in-place T.sub_(velocity, alpha=lr).",
        {"train_intensity": tuple(ds_intensity["train_intensity"].shape)},
    )
    tracemalloc.stop()
    histories["exp5_analytic"] = h_analytic
    results.append(r_analytic)
    cleanup_system(sys_analytic)
    _release_cuda_memory()

    # Experiment 6: Autograd-free NLCG (exact line search + Polak-Ribiere+)
    # on the dual-plane intensity-MSE loss. Reuses the Exp 3/5 intensity dataset.
    sys_nlcg = fresh_system()
    tracemalloc.start()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    h_nlcg = optimize_tm_mse_exact_line_search_dual_plane(
        sys_nlcg,
        ds_intensity["train_input_modes"],
        ds_intensity["train_intensity"],
        num_epochs=epochs,
        resync_interval=args.nlcg_resync_interval,
        epsilon=1e-8,
    )
    focus_gt_nlcg = evaluate_gt_focus_metric(
        sys_nlcg,
        slm_shape=(grid_params["ny"], grid_params["nx"]),
        num_iterations=args.focus_opt_iters,
        lr=args.focus_opt_lr,
        center_radius=args.focus_center_radius,
        target_plane=args.focus_target_plane,
    )
    h_nlcg["gt_focus_metric"] = [focus_gt_nlcg]
    r_nlcg = run_experiment(
        "Exp 6: NLCG Exact Line Search Dual-Plane",
        sys_nlcg,
        h_nlcg,
        args.tm_error_threshold,
        t0,
        "Dual-plane (image + Fourier) unpolarized intensity measurements, full-batch.",
        "Autograd-free Non-Linear Conjugate Gradient on intensity-MSE: Polak-Ribiere+ conjugate direction with exact (closed-form quartic) line search; static plane balance gamma = L_0/L_1; in-place forward state with periodic resync.",
        {"train_intensity": tuple(ds_intensity["train_intensity"].shape), "resync_interval": args.nlcg_resync_interval},
    )
    tracemalloc.stop()
    histories["exp6_nlcg"] = h_nlcg
    results.append(r_nlcg)
    cleanup_system(sys_nlcg)
    del ds_intensity
    _release_cuda_memory()

    # Experiment 4: RAF2-1 pixel-basis benchmark from Cheng et al. (raf_21/).
    # Low-resolution parameterization: each "raf21 SLM pixel" drives a constant-phase block
    # on the physical SLM and intensities are binned to a smaller output grid. This is what
    # makes the pixel-basis TM tractable on a 128x128 system (full-res would need a 16384^2
    # complex matrix).
    raf_sys = fresh_system()
    n_pix_slm_raf, n_pix_out_raf, _, _, _, _ = _resolve_raf21_ratios(
        raf_sys,
        args.raf21_n_pix_slm if args.raf21_n_pix_slm > 0 else None,
        args.raf21_n_pix_out if args.raf21_n_pix_out > 0 else None,
    )
    n_pix_slm_total = n_pix_slm_raf * n_pix_slm_raf
    n_pix_out_total = n_pix_out_raf * n_pix_out_raf
    if args.n_train_raf21 > 0:
        n_train_raf = int(args.n_train_raf21)
    else:
        # RAF2-1 needs P >= ~5 * (SLM degrees of freedom) for stability.
        n_train_raf = int(5 * n_pix_slm_total)
        print(
            f"[raf21] n_train_raf21 not provided; defaulting to 5 * n_pix_slm^2 = {n_train_raf} "
            f"(n_pix_slm={n_pix_slm_raf})."
        )
    print(
        f"[raf21] retrieving a ({n_pix_out_total} x {n_pix_slm_total}) pixel-basis TM "
        f"from P={n_train_raf} single-pol intensity measurements "
        f"(n_pix_slm={n_pix_slm_raf}, n_pix_out={n_pix_out_raf})."
    )
    # Rough peak memory footprint: x_probes (n_pix_slm^2, P) complex64 + intermediate of
    # same size + the (n_pix_out^2, n_pix_slm^2) TM.
    raf_mem_gb = (
        16.0 * n_pix_slm_total * n_train_raf
        + 8.0 * n_pix_out_total * n_pix_slm_total
    ) / (1024.0 ** 3)
    if raf_mem_gb > 8.0:
        print(
            f"[raf21] WARNING: estimated peak memory ~{raf_mem_gb:.1f} GB. Consider "
            f"--raf21-n-pix-slm / --raf21-n-pix-out smaller or --n-train-raf21 lower."
        )
    raf_dataset = generate_raf21_dataset(
        raf_sys,
        n_train_raf,
        args.seed,
        n_pix_slm=n_pix_slm_raf,
        n_pix_out=n_pix_out_raf,
    )
    tracemalloc.start()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    h_raf = run_raf21_experiment(
        raf_sys,
        raf_dataset,
        iters=args.raf21_iters,
        mu=args.raf21_mu,
        ratio=args.raf21_ratio,
        beta=args.raf21_beta,
        gamma=args.raf21_gamma,
        npower_iter=args.raf21_npower_iter,
        seed=args.seed,
        focus_opt_iters=args.focus_opt_iters,
        focus_opt_lr=args.focus_opt_lr,
        focus_center_radius=args.focus_center_radius,
        focus_target_plane=args.focus_target_plane,
    )
    r_raf = run_experiment(
        "Exp 4: RAF2-1 pixel-basis",
        raf_sys,
        h_raf,
        args.tm_error_threshold,
        t0,
        (
            f"Single-polarization image-plane intensities, {n_train_raf} unit-amp SLM probes. "
            f"SLM controlled at {n_pix_slm_raf}x{n_pix_slm_raf} (Kronecker-upsampled to the "
            f"physical SLM), camera binned to {n_pix_out_raf}x{n_pix_out_raf}."
        ),
        (
            "Pixel-basis TM (N^2xM^2) retrieved with RAF2-1 (Cheng et al., 2023); SLM optimized "
            "in the low-res space against the retrieved TM, then upsampled and validated on "
            "the GT system. The reported focus is measured on the same low-res-binned camera "
            "the algorithm sees; gt_focus_full_res reports the same metric on the unbinned image."
        ),
        {
            "y_intensity": tuple(raf_dataset["y_intensity"].shape),
            "x_probes": tuple(raf_dataset["x_probes"].shape),
            "iters": args.raf21_iters,
            "n_pix_slm": n_pix_slm_raf,
            "n_pix_out": n_pix_out_raf,
        },
    )
    tracemalloc.stop()
    histories["exp4_raf21"] = h_raf
    results.append(r_raf)
    cleanup_system(raf_sys)
    del raf_dataset
    _release_cuda_memory()

    report = {
        "experimental_setup": {
            "device": device,
            "seed": args.seed,
            "threshold": args.tm_error_threshold,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "n_train_ls": args.n_train,
            "n_test_ls": args.n_test,
            "n_train_intensity": args.n_train_intensity,
            "n_test_intensity": args.n_test_intensity,
            "intensity_forward_batch_size": args.intensity_forward_batch_size,
            "focus_target_plane": args.focus_target_plane,
            "grid_shape": (source_system.grid_params["ny"], source_system.grid_params["nx"]),
            "tm_system_path": tm_system_path,
            "raf21": {
                "n_train": int(n_train_raf),
                "iters": args.raf21_iters,
                "mu": args.raf21_mu,
                "ratio": args.raf21_ratio,
                "beta": args.raf21_beta,
                "gamma": args.raf21_gamma,
                "npower_iter": args.raf21_npower_iter,
                "n_pix_slm": int(n_pix_slm_raf),
                "n_pix_out": int(n_pix_out_raf),
            },
        },
        "results": [asdict(r) for r in results],
        "performance_table_markdown": summarize_table(results),
        "analysis": [
            "Dual-plane corrected ALS enforces the physically consistent summed operator before Gram assembly.",
            "PBS corrected ALS separates x/y operators and targets, improving physical faithfulness and avoiding channel mixing bias.",
            "ALS-only evaluation isolates Phase-A convergence behavior without gradient-descent confounders.",
            "Reference-weight sweep changes conditioning of TM updates; best weight is selected empirically.",
            "Intensity-only TM retrieval from precomputed input modes isolates TM optimization from SLM-path compute.",
            "RAF2-1 (Cheng et al., 2023) provides a pixel-basis benchmark: an N^2xM^2 TM is fit to single-polarization output intensities via amplitude flow with a 2/3 RAF2 -> 1/3 RAF1 schedule. To stay tractable on a 128x128 grid, the SLM is parameterized at a coarser n_pix_slm grid (replicated to the physical SLM via Kronecker product) and the camera is binned to n_pix_out; the recovered TM has shape (n_pix_out^2, n_pix_slm^2).",
            "Exp 5 (analytic RAF dual-plane) bypasses autograd entirely: the Wirtinger gradient grad_T = (sum_p w_p P_p^H [R_p E_p]) Z_in^H is hand-implemented with stacked-polarization batched matmuls under torch.no_grad(). RAF reweighting is recomputed each iteration; the inter-plane balancing factor gamma = L_0/L_1 is fixed once from the measured amplitude energies; momentum updates run in place on T.",
            "Exp 6 (NLCG dual-plane intensity-MSE) is also autograd-free: a Polak-Ribiere+ conjugate-gradient direction is taken on the exact intensity-MSE Wirtinger gradient delta_T = (sum_p w_p P_p^H [2 R_p E_p]) Z_in^H, and the step length is solved in closed form because the loss is quartic in alpha along the line (its cubic derivative is rooted, real roots filtered, global minimizer chosen). The forward state (Y, E) is advanced in place via E <- E - alpha V_E and only resynced every resync_interval epochs; static plane balance gamma = L_0/L_1 is fixed once at epoch 0.",
        ],
        "conclusion": recommendation(results),
        "novel_technique_breakdown": {
            "name": "Reference-weight sweep over corrected operators",
            "stage_1": "Run corrected ALS over a coarse weight grid.",
            "stage_2": "Select the best weight by final TM error under the convergence threshold.",
            "why_it_works": "Balances the reference anchor against signal equations to improve numerical conditioning.",
        },
    }
    return {"report": report, "histories": histories}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TM optimization experimental suite runner")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--tm-system-path", type=str, default="tm_system_rotated.pt")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--grid-n", type=int, default=64)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-train-intensity", type=int, default=5000)
    p.add_argument("--n-test-intensity", type=int, default=200)
    p.add_argument("--intensity-forward-batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--focus-opt-iters", type=int, default=50)
    p.add_argument("--focus-opt-lr", type=float, default=0.1)
    p.add_argument("--focus-center-radius", type=int, default=1)
    p.add_argument("--focus-target-plane", type=str, default="image", choices=["image", "fourier"])
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument("--tm-error-threshold", type=float, default=16.0)
    p.add_argument("--reference-weights", type=float, nargs="+", default=[0.25])
    p.add_argument("--n-train-raf21", type=int, default=0, help="Number of unit-amp SLM probes for RAF2-1 (0 -> default 5 * n_pix_slm^2).")
    p.add_argument("--raf21-iters", type=int, default=80, help="RAF2-1 gradient-descent iterations.")
    p.add_argument("--raf21-mu", type=float, default=3.0, help="RAF2-1 step size (matches the MATLAB default).")
    p.add_argument("--raf21-ratio", type=float, default=2.0 / 3.0, help="RAF2 -> RAF1 switch ratio.")
    p.add_argument("--raf21-beta", type=float, default=5.0)
    p.add_argument("--raf21-gamma", type=float, default=0.5)
    p.add_argument("--raf21-npower-iter", type=int, default=30)
    p.add_argument("--raf21-n-pix-slm", type=int, default=32, help="Low-res SLM side (must divide --grid-n; 0 -> grid_n).")
    p.add_argument("--raf21-n-pix-out", type=int, default=64, help="Low-res output side (must divide --grid-n; 0 -> grid_n).")
    p.add_argument("--analytic-learning-rate", type=float, default=3e-7, help="Step size for Exp 5 analytic Wirtinger optimizer.")
    p.add_argument("--analytic-momentum", type=float, default=0.6, help="Polyak momentum for Exp 5 analytic Wirtinger optimizer.")
    p.add_argument("--analytic-beta", type=float, default=4.0, help="RAF dynamic-weighting beta for Exp 5 analytic Wirtinger optimizer.")
    p.add_argument("--nlcg-resync-interval", type=int, default=10, help="Forward-model resync / CG-reset interval for Exp 6 NLCG optimizer.")
    p.add_argument("--report-json", type=str, default="tm_experiment_report.json")
    p.add_argument("--history-json", type=str, default="tm_experiment_histories.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_suite(args)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(out["report"], f, indent=2)
    with open(args.history_json, "w", encoding="utf-8") as f:
        json.dump(out["histories"], f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print("\n=== Experimental Report (Markdown Table) ===")
    print(out["report"]["performance_table_markdown"])
    print("\n=== Conclusion ===")
    print(out["report"]["conclusion"])
    print(f"\nSaved report to: {os.path.abspath(args.report_json)}")
    print(f"Saved histories to: {os.path.abspath(args.history_json)}")


if __name__ == "__main__":
    main()
