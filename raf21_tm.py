"""
RAF2-1 transmission-matrix retrieval (Cheng et al.), ported from MATLAB `RAF21_tm_cpu.m`.

Reference implementation: `raf_21/RAF21_TM_retrieval/algs/RAF21_tm_cpu.m`.
Forward model in the original code: intensity Y = abs(X * A).^2 with complex probe matrix X
and complex TM estimate A (same transpose / dimension conventions as MATLAB).

This module keeps the same initialization (weighted power iterations), two-phase RAF2/RAF1
gradient weights controlled by `ratio`, and the same per-iteration gradient / output
normalization as the CPU MATLAB file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class RAF21Options:
    mu: float = 3.0
    iters: int = 200
    ratio: float = 2.0 / 3.0
    beta: float = 5.0
    gamma: float = 0.5
    npower_iter: int = 30


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
    if y_intensity.ndim != 2 or x_probes.ndim != 2:
        raise ValueError("y_intensity and x_probes must be 2D tensors.")
    m, p = y_intensity.shape
    n, p2 = x_probes.shape
    if p2 != p:
        raise ValueError(f"Probe count mismatch: y has p={p}, x has p={p2}.")

    device = y_intensity.device
    dtype_c = torch.complex64
    y = y_intensity.to(dtype=torch.float32).transpose(0, 1).contiguous()  # (p, m)
    ymag = torch.sqrt(torch.clamp(y, min=0.0))
    x_row = torch.conj(x_probes.transpose(0, 1)).to(device=device, dtype=dtype_c).contiguous()  # (p, n), MATLAB `x = x.'`
    x_adj = x_row.conj().transpose(-2, -1).contiguous()  # (n, p), MATLAB `xH = x.'` after `x` was updated

    def xfun(i_mat: torch.Tensor) -> torch.Tensor:
        return x_row @ i_mat

    def xinvfun(y_mat: torch.Tensor) -> torch.Tensor:
        return x_adj @ y_mat

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
        a_pr = (a_pr / (2 ** 0.5)).to(dtype_c)
    col_norm = torch.linalg.vector_norm(a_pr, ord=2, dim=0, keepdim=True).clamp(min=1e-30)
    a_pr = a_pr / col_norm

    normest = torch.sqrt(torch.mean(y, dim=0, keepdim=True)).to(dtype_c)  # (1, m)
    y_sorted, _ = torch.sort(y, dim=0, descending=False)
    matlab_row = int(round(float(p) / 1.3))
    matlab_row = max(1, min(p, matlab_row))
    python_row = matlab_row - 1
    ythresh = y_sorted[python_row : python_row + 1, :]  # (1, m), matches MATLAB ysort(round(p/1.3), :)
    ind = y >= ythresh
    weights = ((ymag * ind.to(ymag.dtype)) ** opts.gamma).to(dtype_c)

    for _ in range(opts.npower_iter):
        a_pr = xinvfun(weights * xfun(a_pr))
        col_norm = torch.linalg.vector_norm(a_pr, ord=2, dim=0, keepdim=True).clamp(min=1e-30)
        a_pr = a_pr / col_norm

    a_pr = normest * a_pr

    cutoff = int(torch.ceil(torch.tensor(opts.ratio * opts.iters, device=device, dtype=torch.float32)).item())
    compt = 1
    while compt <= opts.iters:
        curr_y = xfun(a_pr)
        wgt = torch.abs(curr_y) / (torch.abs(curr_y) + opts.beta * ymag).clamp(min=1e-30)
        if compt <= cutoff:
            resid = (curr_y - (ymag ** 2) * curr_y / torch.abs(curr_y).clamp(min=1e-30)) * wgt
        else:
            resid = (curr_y - ymag * curr_y / torch.abs(curr_y).clamp(min=1e-30)) * wgt
        grad = xinvfun(resid) / float(p)
        a_pr = a_pr - opts.mu * grad
        # Numerical safety: MATLAB stays finite in double precision and only saturates after
        # the loop, but complex64 overflows ~1e38. In severely under-sampled regimes
        # (P < 5 * N), iterates can blow up; clip blown-up entries to unit-magnitude in place
        # so the algorithm completes and returns a defined (still poor) estimate instead of
        # propagating NaNs.
        huge = torch.abs(a_pr) > 1e10
        if huge.any():
            a_pr = torch.where(huge, torch.exp(1j * torch.angle(a_pr)), a_pr)
        compt += 1

    # MATLAB `A_pr = A_pr'` is Hermitian (conjugate) transpose for complex matrices.
    # Using a plain transpose here would return conj(A_true), which silently inverts the
    # optimal-focus SLM phase when the recovered matrix is used as a forward model.
    a_pr = a_pr.conj().transpose(0, 1).contiguous()

    huge = torch.abs(a_pr) > 1e10
    if huge.any():
        a_pr = torch.where(huge, torch.exp(1j * torch.angle(a_pr)), a_pr)

    # MATLAB std on a complex matrix uses |z - mean(z)|, i.e., var(re) + var(im); reproduce
    # that exactly here rather than flattening to real-valued samples.
    re = a_pr.real
    im = a_pr.imag
    var_total = ((re - re.mean()).pow(2) + (im - im.mean()).pow(2)).mean()
    scale = torch.sqrt(var_total).clamp(min=1e-30)
    a_pr = a_pr / scale.to(dtype_c)
    return a_pr


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
