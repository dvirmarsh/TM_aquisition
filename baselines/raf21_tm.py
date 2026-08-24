"""
RAF2-1 transmission-matrix retrieval (Cheng et al.), ported from MATLAB `RAF21_tm_cpu.m`.

Ported from MATLAB `RAF21_tm_cpu.m` (Cheng et al., Adv. Photon. Nexus 2, 066005 (2023), doi: 10.1117/1.APN.2.6.066005).
Forward model in the original code: intensity Y = abs(X * A).^2 with complex probe matrix X
and complex TM estimate A (same transpose / dimension conventions as MATLAB).

This module keeps the same initialization (weighted power iterations), two-phase RAF2/RAF1
gradient weights controlled by `ratio`, and the same per-iteration gradient / output
normalization as the CPU MATLAB file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch


@dataclass
class RAF21Options:
    mu: float = 3.0
    iters: int = 200
    ratio: float = 2.0 / 3.0
    beta: float = 5.0
    gamma: float = 0.5
    npower_iter: int = 30


def _a_pr_to_pixel_tm(a_pr: torch.Tensor) -> torch.Tensor:
    """Map internal (n, m) RAF state to pixel-basis TM (m, n), without variance normalization."""
    a = a_pr.conj().transpose(0, 1).contiguous()
    huge = torch.abs(a) > 1e10
    if huge.any():
        a = torch.where(huge, torch.exp(1j * torch.angle(a)), a)
    return a


def _finalize_a_pr(a_pr: torch.Tensor) -> torch.Tensor:
    a = _a_pr_to_pixel_tm(a_pr)
    re = a.real
    im = a.imag
    var_total = ((re - re.mean()).pow(2) + (im - im.mean()).pow(2)).mean()
    scale = torch.sqrt(var_total).clamp(min=1e-30)
    return a / scale.to(a.dtype)


def compute_raf21_ls_loss(
    a_pixel: torch.Tensor,
    y_intensity: torch.Tensor,
    x_probes: torch.Tensor,
) -> float:
    amp_pred = (a_pixel @ x_probes).abs()
    amp_meas = y_intensity.clamp(min=0).sqrt()
    return float(torch.linalg.vector_norm(amp_pred - amp_meas, ord=2, dim=1).sum() / amp_pred.shape[0])


def _raf21_grad_step(
    a_pr: torch.Tensor,
    *,
    compt: int,
    cutoff: int,
    xfun: Callable[[torch.Tensor], torch.Tensor],
    xinvfun: Callable[[torch.Tensor], torch.Tensor],
    ymag: torch.Tensor,
    beta: float,
    mu: float,
    p: int,
) -> torch.Tensor:
    curr_y = xfun(a_pr)
    wgt = torch.abs(curr_y) / (torch.abs(curr_y) + beta * ymag).clamp(min=1e-30)
    if compt <= cutoff:
        resid = (curr_y - (ymag**2) * curr_y / torch.abs(curr_y).clamp(min=1e-30)) * wgt
    else:
        resid = (curr_y - ymag * curr_y / torch.abs(curr_y).clamp(min=1e-30)) * wgt
    grad = xinvfun(resid) / float(p)
    a_pr = a_pr - mu * grad
    huge = torch.abs(a_pr) > 1e10
    if huge.any():
        a_pr = torch.where(huge, torch.exp(1j * torch.angle(a_pr)), a_pr)
    return a_pr


def raf21_tm_epochs(
    y_intensity: torch.Tensor,
    x_probes: torch.Tensor,
    opts: RAF21Options,
    *,
    generator: Optional[torch.Generator] = None,
    init_a_pr: Optional[torch.Tensor] = None,
    on_epoch_end: Optional[Callable[[int, float, torch.Tensor, float], None]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """
    Run RAF2-1 with per-iteration bookkeeping.

    Each epoch is one gradient step. Epoch 0 timing includes random initialization and
    the spectral (power-iteration) initialization; later epochs time only their gradient step.
  """
    if y_intensity.ndim != 2 or x_probes.ndim != 2:
        raise ValueError("y_intensity and x_probes must be 2D tensors.")
    m, p = y_intensity.shape
    n, p2 = x_probes.shape
    if p2 != p:
        raise ValueError(f"Probe count mismatch: y has p={p}, x has p={p2}.")

    device = y_intensity.device
    dtype_c = torch.complex64
    y = y_intensity.to(dtype=torch.float32).transpose(0, 1).contiguous()
    ymag = torch.sqrt(torch.clamp(y, min=0.0))
    x_row = torch.conj(x_probes.transpose(0, 1)).to(device=device, dtype=dtype_c).contiguous()
    x_adj = x_row.conj().transpose(-2, -1).contiguous()

    def xfun(i_mat: torch.Tensor) -> torch.Tensor:
        return x_row @ i_mat

    def xinvfun(y_mat: torch.Tensor) -> torch.Tensor:
        return x_adj @ y_mat

    epoch_times: List[float] = []
    ls_losses: List[float] = []
    cutoff = int(torch.ceil(torch.tensor(opts.ratio * opts.iters, device=device, dtype=torch.float32)).item())

    for epoch_idx in range(opts.iters):
        t0 = time.perf_counter()
        if epoch_idx == 0:
            if init_a_pr is not None:
                if init_a_pr.shape != (n, m):
                    raise ValueError(f"init_a_pr expected {(n, m)}, got {tuple(init_a_pr.shape)}")
                a_pr = init_a_pr.to(device=device, dtype=dtype_c).contiguous()
            elif generator is not None:
                a_pr = torch.randn((n, m), generator=generator, device=device, dtype=torch.float32)
                a_pr = a_pr + 1j * torch.randn((n, m), generator=generator, device=device, dtype=torch.float32)
            else:
                a_pr = torch.randn((n, m), device=device, dtype=torch.float32)
                a_pr = a_pr + 1j * torch.randn((n, m), device=device, dtype=torch.float32)
            if init_a_pr is None:
                a_pr = (a_pr / (2**0.5)).to(dtype_c)
            col_norm = torch.linalg.vector_norm(a_pr, ord=2, dim=0, keepdim=True).clamp(min=1e-30)
            a_pr = a_pr / col_norm

            normest = torch.sqrt(torch.mean(y, dim=0, keepdim=True)).to(dtype_c)
            y_sorted, _ = torch.sort(y, dim=0, descending=False)
            matlab_row = int(round(float(p) / 1.3))
            matlab_row = max(1, min(p, matlab_row))
            python_row = matlab_row - 1
            ythresh = y_sorted[python_row : python_row + 1, :]
            ind = y >= ythresh
            weights = ((ymag * ind.to(ymag.dtype)) ** opts.gamma).to(dtype_c)

            for _ in range(opts.npower_iter):
                a_pr = xinvfun(weights * xfun(a_pr))
                col_norm = torch.linalg.vector_norm(a_pr, ord=2, dim=0, keepdim=True).clamp(min=1e-30)
                a_pr = a_pr / col_norm
            a_pr = normest * a_pr

        a_pr = _raf21_grad_step(
            a_pr,
            compt=epoch_idx + 1,
            cutoff=cutoff,
            xfun=xfun,
            xinvfun=xinvfun,
            ymag=ymag,
            beta=opts.beta,
            mu=opts.mu,
            p=p,
        )
        a_pixel = _a_pr_to_pixel_tm(a_pr)
        ls_loss = compute_raf21_ls_loss(a_pixel, y_intensity, x_probes)
        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)
        ls_losses.append(ls_loss)
        if on_epoch_end is not None:
            on_epoch_end(epoch_idx, elapsed, a_pixel, ls_loss)

    return _finalize_a_pr(a_pr), {"epoch_times_sec": epoch_times, "ls_loss": ls_losses}


def raf21_tm(
    y_intensity: torch.Tensor,
    x_probes: torch.Tensor,
    opts: RAF21Options,
    *,
    generator: Optional[torch.Generator] = None,
    init_a_pr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Retrieve complex matrix A such that measurements match |X @ A|^2 in amplitude flow.

    Args:
        y_intensity: (m, p) intensities — m outputs (e.g. camera pixels), p probes (columns).
        x_probes: (n, p) complex input fields / SLM patterns (column p is probe p).
        opts: Hyperparameters matching MATLAB defaults in `RAF21_tm_cpu.m`.
        generator: Optional RNG for the complex Gaussian initialization of A.
        init_a_pr: If provided, shape (n, m) complex, skip random init and use this matrix
            (still column-normalized as in MATLAB). For tests / parity checks only.

    Returns:
        A_pr: (m, n) complex (same layout as MATLAB after final transpose), single-precision
        on the same device as inputs.

    Notes:
        - Internally matches `RAF21_tm_cpu.m`: `y -> y.'`, `ymag = sqrt(y)`, `x = x.'` then
          `xH = x.'` (using the **updated** `x`), so `Xfun(I) = x * I` with `x` of shape `(p, n)` and
          `Xinvfun(Y) = xH * Y` with `xH` of shape `(n, p)` (Hermitian transpose of `x`).
    """
    a_pixel, _ = raf21_tm_epochs(
        y_intensity,
        x_probes,
        opts,
        generator=generator,
        init_a_pr=init_a_pr,
    )
    return a_pixel


def recover_tm_from_sensor_matrix(
    p_out_x: torch.Tensor,
    h_est: torch.Tensor,
    ridge_alpha: float,
) -> torch.Tensor:
    """
    Given image-plane x-polarization projection P (pixels, modes) and effective matrix
    H_est ≈ P @ T, recover T via Tikhonov normal equations:

        (P^H P + λ diag(P^H P)) T = P^H H_est

    which matches the ridge scaling style used in `TM_experimental_pipeline._solve_regularized`.
    """
    if p_out_x.shape[0] != h_est.shape[0]:
        raise ValueError("Row count of P must match row count of H_est.")
    php = p_out_x.conj().transpose(-2, -1) @ p_out_x
    phh = p_out_x.conj().transpose(-2, -1) @ h_est
    diag_abs_max = torch.max(torch.abs(torch.diag(php))).real
    reg_scale = ridge_alpha * torch.clamp(diag_abs_max, min=1e-12)
    reg = reg_scale * torch.eye(php.shape[0], device=php.device, dtype=php.dtype)
    return torch.linalg.solve(php + reg, phh)
