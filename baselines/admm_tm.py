"""
ADMM-based transmission-matrix retrieval from intensity-only measurements.

Implements the alternating-direction method of multipliers (ADMM) reconstruction of
Wei et al., "Millisecond vectorial wave-front shaping through scattering media via full
transmission-matrix measurement", Phys. Rev. Applied 25, 064049 (2026) (doi: 10.1103/t85q-16jp).

The paper controls two input polarizations and retrieves the *vectorial* TM via the coupled
model c = |A x + B y|. Here we control only a single polarization, so the model reduces to the
*scalar* phase-retrieval problem

    c = |A x|,

where, for a fixed output location, A (M x N) collects the M structured input probes over the
N controllable SLM modes, x is the unknown TM column at that output location, and c are the
measured output amplitudes. All output locations (camera pixels) share the same probe matrix A,
so the K columns of the TM are solved simultaneously in a single batched linear algebra pass.

The ADMM iteration (scalar specialization of Eqs. (4)-(11) in the paper):

    x_n  = (A^H A)^-1 A^H (u_{n-1} + lambda_{n-1})          # Eq. (6) with B y dropped
    d_n  = A x_n - lambda_{n-1}                              # Eq. (9)
    u_n  = (mu c + rho |d_n|) / (mu + rho) * d_n / |d_n|     # Eq. (10)
    lam_n = lambda_{n-1} + (u_n - A x_n)                     # Eq. (11)

Conventions match `raf21_tm.py`/`run_raf_offline.py`:
    y_intensity : (K, P) output intensities  (K = n_pix_out^2 detector pixels, P probes)
    x_probes    : (N, P) complex input SLM fields (N = n_pix_slm^2 modes, column p = probe p)
    a_pixel     : (K, N) complex retrieved TM such that E_out = a_pixel @ E_in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from raf21_tm import compute_raf21_ls_loss


@dataclass
class ADMMOptions:
    # Measurement / model weights (paper: mu=5, rho=4 -> stable convergence in 20-30 iters).
    mu: float = 5.0
    rho: float = 4.0
    iters: int = 30
    # Orthogonality-promoting initialization of x0 (named in the paper, but not cited or
    # parameterized; TAF definition assumed, see _orthogonality_promoting_init).
    npower_iter: int = 30
    # Fraction of largest-amplitude measurements kept for the power iteration
    # (TAF, arXiv:1605.08285, Algorithm 1 uses |I0| = ceil(m/6); this code keeps 5/6).
    init_keep_ratio: float = 5.0 / 6.0
    # Tikhonov ridge added to A^H A for a numerically stable closed-form x-update.
    ridge: float = 1e-6


def _orthogonality_promoting_init(
    a_fwd: torch.Tensor,
    c_amp: torch.Tensor,
    *,
    npower_iter: int,
    keep_ratio: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Orthogonality-promoting estimate of x0 for the scalar model c = |A x|.

    Batched over the K output locations (columns of `c_amp`). Returns X0 of shape (N, K).

    Wei et al. name an "orthogonality-promoting estimate" but neither cite nor specify it; this
    is the simplest form of the method that introduced the term (Wang, Giannakis & Eldar,
    truncated amplitude flow, arXiv:1605.08285, Algorithm 1). The exact variant Wei et al.
    ran is unknown.

    Probes with the smallest measured amplitudes are nearly orthogonal to x, so x0 should be
    the direction most orthogonal to them: the smallest eigenvector of A_S^H A_S. Because
    A^H A ~ P*I for random-phase probes, that is approximately the leading eigenvector of
    A_T^H A_T, with T the remaining keep_ratio * P largest amplitudes, which power iteration
    finds quickly (the direct smallest-eigenvector iteration needs ~1000 steps here).

    Simplifications (no effect for this problem): no per-probe norm normalization (phase-only
    probes all have norm sqrt(N)) and no scaling of x0 (u0 = c * u0/|u0| discards it).
    """
    p, n = a_fwd.shape
    k = c_amp.shape[1]
    x = torch.randn((n, k), generator=generator, device=a_fwd.device, dtype=a_fwd.dtype)

    keep_count = max(1, min(p, round(p * keep_ratio)))
    top = torch.topk(c_amp, keep_count, dim=0).indices  # (keep_count, K) strongest probes
    mask = torch.zeros_like(c_amp).scatter_(0, top, 1.0).to(a_fwd.dtype)
    a_adj = a_fwd.conj().transpose(0, 1)

    for _ in range(npower_iter):
        x = a_adj @ (mask * (a_fwd @ x))
        x = x / torch.linalg.vector_norm(x, dim=0, keepdim=True).clamp(min=1e-30)
    return x


def admm_tm_epochs(
    y_intensity: torch.Tensor,
    x_probes: torch.Tensor,
    opts: ADMMOptions,
    *,
    generator: Optional[torch.Generator] = None,
    on_epoch_end: Optional[Callable[[int, float, torch.Tensor, float], None]] = None,
) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
    """Run scalar ADMM TM retrieval with per-iteration bookkeeping.

    Each epoch is one ADMM iteration. Epoch 0 timing additionally includes building the
    (A^H A)^-1 A^H solver and the orthogonality-promoting initialization.

    Args:
        y_intensity: (K, P) measured output intensities.
        x_probes:    (N, P) complex input SLM fields.
        opts:        ADMM hyperparameters.
        generator:   Optional RNG for the orthogonality-promoting initialization.
        on_epoch_end: Callback (epoch, epoch_seconds, a_pixel, ls_loss) called every iteration,
            where a_pixel is the (K, N) TM estimate at that iteration.

    Returns:
        a_pixel: (K, N) complex retrieved TM.
        history: includes per-epoch metrics plus one-element setup and initialization timings.
    """
    if y_intensity.ndim != 2 or x_probes.ndim != 2:
        raise ValueError("y_intensity and x_probes must be 2D tensors.")
    k_out, p = y_intensity.shape
    n, p2 = x_probes.shape
    if p2 != p:
        raise ValueError(f"Probe count mismatch: y has P={p}, x has P={p2}.")

    device = y_intensity.device
    dtype_c = torch.complex64

    # ---- Iteration-independent setup (computed once, before the ADMM loop) ----
    # The probe matrix A, its Hermitian, the closed-form x-update operator P_inv, the amplitude
    # targets, and the initialization do not change between iterations, so they are
    # hoisted out of the loop. The one-time cost is timed and attributed to epoch 0 (mirroring
    # `raf21_tm_epochs`), so the reported per-epoch timing budget is unchanged.
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    t_setup = time.perf_counter()

    # A (P, N): row p is probe p over the N modes; C (P, K): measured amplitudes per output.
    a_fwd = x_probes.transpose(0, 1).to(device=device, dtype=dtype_c).contiguous()
    a_adj = a_fwd.conj().transpose(0, 1).contiguous()  # (N, P) = A^H
    c_amp = torch.sqrt(torch.clamp(y_intensity, min=0.0)).to(torch.float32).transpose(0, 1).contiguous()  # (P, K)
    c_amp_c = c_amp.to(dtype_c)

    # Closed-form x-update operator P_inv = (A^H A + ridge I)^-1 A^H, shape (N, P).
    gram = a_adj @ a_fwd  # (N, N)
    ridge = opts.ridge * torch.clamp(torch.max(torch.abs(torch.diag(gram))).real, min=1e-12)
    gram = gram + ridge * torch.eye(n, device=device, dtype=dtype_c)
    p_inv = torch.linalg.solve(gram, a_adj)  # (N, P)

    # Constant pieces of the weighted amplitude projection (Eq. 10).
    mu_c = opts.mu * c_amp_c  # the measurement-fidelity term mu * c
    inv_mu_rho = 1.0 / (opts.mu + opts.rho)

    # Paper initialization: orthogonality-promoting x0, followed by Eq. (4), amplitude
    # normalization, and lambda0 = 0. Synchronization makes this a real CUDA wall time.
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    t_init = time.perf_counter()
    x0 = _orthogonality_promoting_init(
        a_fwd,
        c_amp,
        npower_iter=opts.npower_iter,
        keep_ratio=opts.init_keep_ratio,
        generator=generator,
    )
    ax0 = a_fwd @ x0  # (P, K)
    u = c_amp_c * ax0 / ax0.abs().clamp(min=1e-30)
    lam = torch.zeros_like(u)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    initialization_seconds = time.perf_counter() - t_init
    setup_seconds = time.perf_counter() - t_setup

    epoch_times: List[float] = []
    ls_losses: List[float] = []
    corrs: List[float] = []
    x_prev: Optional[torch.Tensor] = None

    for epoch_idx in range(opts.iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        t0 = time.perf_counter()

        # x-update (Eq. 6, scalar): X = P_inv @ (u + lambda).
        x = p_inv @ (u + lam)  # (N, K)
        ax = a_fwd @ x  # (P, K)
        # Physical prediction corrected for accumulated residual (Eq. 9).
        d = ax - lam
        d_abs = d.abs()
        # Weighted amplitude projection (Eq. 10): keep phase of d, blend amplitudes.
        u = ((mu_c + opts.rho * d_abs) * inv_mu_rho) * (d / d_abs.clamp(min=1e-30))
        # Dual / Lagrange-multiplier update (Eq. 11).
        lam = lam + (u - ax)

        a_pixel = x.transpose(0, 1).contiguous()  # (K, N) = TM
        ls_loss = compute_raf21_ls_loss(a_pixel, y_intensity, x_probes)

        # Convergence diagnostic: mean column correlation between successive estimates.
        if x_prev is not None:
            num = torch.abs((x_prev.conj() * x).sum(dim=0))
            den = (
                torch.linalg.vector_norm(x_prev, ord=2, dim=0)
                * torch.linalg.vector_norm(x, ord=2, dim=0)
            ).clamp(min=1e-30)
            corr = float((num / den).mean().item())
        else:
            corr = 0.0
        x_prev = x
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)
        elapsed = time.perf_counter() - t0
        if epoch_idx == 0:
            elapsed += setup_seconds  # one-time setup attributed to the first epoch
        epoch_times.append(elapsed)
        ls_losses.append(ls_loss)
        corrs.append(corr)
        if on_epoch_end is not None:
            on_epoch_end(epoch_idx, elapsed, a_pixel, ls_loss)

    return a_pixel, {
        "epoch_times_sec": epoch_times,
        "ls_loss": ls_losses,
        "corr": corrs,
        "setup_time_sec": [setup_seconds],
        "initialization_time_sec": [initialization_seconds],
    }


def admm_tm(
    y_intensity: torch.Tensor,
    x_probes: torch.Tensor,
    opts: ADMMOptions,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Retrieve the scalar TM a_pixel (K, N) such that |a_pixel @ x_probes|^2 matches y_intensity."""
    a_pixel, _ = admm_tm_epochs(y_intensity, x_probes, opts, generator=generator)
    return a_pixel
