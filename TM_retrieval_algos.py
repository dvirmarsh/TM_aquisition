"""
Modal transmission-matrix retrieval algorithms used in the manuscript.

Two published methods:

* ``optimize_tm_admm_dual_plane`` — Algorithm 1, fast modal ADMM retrieval.
  Autograd-free; runs under ``no_grad``. Alternates a closed-form TM update,
  a joint polarization-preserving amplitude projection, and a leaky dual
  update (``gamma_t = 0.9 ** floor(t / 15)``, ``rho = 0`` for the first 45
  iterations then ``0.1``).
* ``optimize_tm_slm`` — Algorithm 2, differentiable forward model with
  aberration corrections. Adam jointly fits the TM and the aberration /
  geometry nuisance parameters by minimizing the PCC loss.

Both operate on a **calibrated** ``TMMeasurementSystem``. Intensity is
polarization-summed, ``I = |E_x|^2 + |E_y|^2``, and produced for two planes
at once (index 0 = image, index 1 = Fourier).

Shared ADMM contract::

    history = optimize_tm_admm_dual_plane(
        tm_system,                 # a CALIBRATED TMMeasurementSystem
        train_input_modes,         # Z_in, shape (N_modes, Batch)
        train_intensity_targets,   # (Batch, 2, H, W); 0 = image, 1 = Fourier
        num_epochs=..., planes='both'|'image'|'fourier',
        init_method='default_beta_prior',
        resync_interval=...,
        on_epoch_end=lambda epoch, secs, sys: ...,
    )
    # -> {'epoch_losses': [...], 'epoch_times': [...]}

``init_method='default_beta_prior'`` keeps the straight-fiber prior written by
``calibrate()`` / ``reinitialize_optimized_tm()``:
``T0 = diag(exp(i beta_l L)) + Delta`` with complex Gaussian ``Delta``,
``sigma = 0.01``.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.optim as optim

from metrics_losses import (
    PearsonCorrelationLoss,
)
from TM_retrieve import TMMeasurementSystem


def _synchronize_if_cuda(device) -> None:
    """Wall-clock barrier around ADMM timers. No-op on CPU; numerics unchanged."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(device)


def _plane_indices_from_planes(planes: str) -> Tuple[int, ...]:
    """Map ``'image'`` / ``'fourier'`` / ``'both'`` to plane indices (0=image, 1=fourier)."""
    planes_l = planes.lower()
    if planes_l == 'image':
        return (0,)
    if planes_l == 'fourier':
        return (1,)
    if planes_l == 'both':
        return (0, 1)
    raise ValueError(f"planes must be 'image', 'fourier', or 'both', got {planes!r}")


def _init_tm_dual_plane(
    tm_system: 'TMMeasurementSystem',
    init_method: str,
    P_stacked: List[torch.Tensor],
    Z_in: torch.Tensor,
    train_intensity_targets: torch.Tensor,
    init_power_iters: int,
    init_num_pixels: int,
    n_pixels: int,
    init_eig_rel_threshold: float,
) -> float:
    """Initialize ``tm_system.optimized_tm_tensor`` for the dual-plane solvers.

    Sets the starting TM in place and returns the wall-clock seconds spent.

    * ``'random'``: standard scaled complex Gaussian noise (the from-scratch baseline).
    * ``'identity'``: identity mapping in the common guided-mode basis.

    ``default_beta_prior`` is handled by ``optimize_tm_admm_dual_plane`` itself
    (it leaves the TM written by ``calibrate()`` / ``reinitialize_optimized_tm()``).

    Args:
        tm_system: calibrated measurement system (provides the TM setter).
        init_method: ``'random'`` or ``'identity'``.
        P_stacked: unused; kept for call-signature compatibility.
        Z_in: input modes ``(n_modes, batch)``.
        train_intensity_targets: unused; kept for call-signature compatibility.
        init_power_iters: unused; kept for call-signature compatibility.
        init_num_pixels: unused; kept for call-signature compatibility.
        n_pixels: unused; kept for call-signature compatibility.
        init_eig_rel_threshold: unused; kept for call-signature compatibility.

    Returns:
        Initialization wall-clock time in seconds.
    """
    del P_stacked, train_intensity_targets, init_power_iters, init_num_pixels
    del n_pixels, init_eig_rel_threshold

    device = Z_in.device
    n_modes = Z_in.shape[0]

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    if init_method == 'random':
        re = torch.randn((n_modes, n_modes), device=device, dtype=torch.float32)
        im = torch.randn((n_modes, n_modes), device=device, dtype=torch.float32)
        T_init = ((re + 1j * im) / math.sqrt(2.0)).to(torch.complex64)
    elif init_method == 'identity':
        T_init = torch.eye(n_modes, device=device, dtype=torch.complex64)
    else:
        raise ValueError(
            "init_method must be 'random' or 'identity', "
            f"got {init_method!r}"
        )

    tm_system.optimized_tm_tensor = T_init.contiguous()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0


def optimize_tm_slm(
    tm_system: TMMeasurementSystem,
    slm_patterns: torch.Tensor,
    intensity_targets: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    learning_rate_phys: float = 0.005,
    learning_rate_aberrations: float = 0.005,
    batch_size: int = 32,
    planes: str = 'both',
    optimize_aberrations: bool = True,
    test_slm_patterns: Optional[torch.Tensor] = None,
    test_intensity_targets: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Optimize TM and Aberrations using Adam optimizer with 3 separate learning rates.
    
    Uses the TMMeasurementSystem's internal nn.Parameters directly:
    - TM: optimized_tm, slm_complex_amplitude
    - Physical parameters: ZoomFFT params, polarization angles
    - Aberration parameters: Zernike coefficients
    
    Args:
        tm_system: TMMeasurementSystem nn.Module with learnable parameters
        slm_patterns: SLM phase patterns, shape (num_measurements, H, W)
        intensity_targets: Target intensities, shape (num_measurements, 2, H, W)
        num_epochs: Number of optimization epochs
        learning_rate: Learning rate for TM parameters (default 0.01)
        learning_rate_phys: Learning rate for physical parameters (ZoomFFT, polarization) (default 0.005)
        learning_rate_aberrations: Learning rate for aberration parameters (Zernike) (default 0.005)
        batch_size: Batch size for mini-batch gradient descent
        planes: Which planes to use for loss ('image', 'fourier', or 'both')
        optimize_aberrations: If True, optimize all parameters. If False, only optimize TM.
        test_slm_patterns: Optional test SLM patterns for test loss calculation
        test_intensity_targets: Optional test intensity targets for test loss
    
    Returns:
        Dict containing training history
    """

    print("\n" + "="*60)
    print("PHASE 2: Gradient-Based Optimization")
    print("="*60)
    print(f"Adam Optimization. Optimize Aberrations: {optimize_aberrations}")
    print(f"LR_TM: {learning_rate} | LR_Phys: {learning_rate_phys} | LR_Aberrations: {learning_rate_aberrations}")
    
    # --- 1. Configure which parameters to optimize ---
    # TM parameters (only the transmission matrix itself)
    tm_system._optimized_tm_param.requires_grad_(True)
    
    # Physical parameters (SLM amplitude + polarization)
    tm_system.slm_complex_amplitude.requires_grad_(True)
    tm_system.theta_before_slm.requires_grad_(optimize_aberrations)
    tm_system.theta_after_slm.requires_grad_(optimize_aberrations)
    
    # Aberration parameters (Zernike + ZoomFFT)
    zernike_params = [tm_system.zernike_before_ft, tm_system.zernike_after_ft,
                      tm_system.zernike_image_plane, tm_system.zernike_fourier_domain]
    for param in zernike_params:
        param.requires_grad_(optimize_aberrations)
    
    zoomfft_params_list = []
    for param_dict in [tm_system.zoomfft_slm_params, tm_system.zoomfft_out_fwd_params,
                       tm_system.zoomfft_out_inv_params]:
        for param in param_dict.values():
            param.requires_grad_(optimize_aberrations)
            if optimize_aberrations:
                zoomfft_params_list.append(param)
    
    # --- 2. Setup Optimizer with 3 learning rates ---
    tm_params = [tm_system._optimized_tm_param]
    # Optional camera weight of the Fourier-plane camera
    # (TMMeasurementSystem.enable_learnable_fourier_weights); fitted with the TM.
    fourier_weight_beta = getattr(tm_system, "fourier_weight_beta", None)
    if fourier_weight_beta is not None:
        fourier_weight_beta.requires_grad_(True)
        tm_params.append(fourier_weight_beta)
    
    # Physical: SLM amplitude + polarization angles
    physical_params = [tm_system.slm_complex_amplitude]
    if optimize_aberrations:
        physical_params.extend([tm_system.theta_before_slm, tm_system.theta_after_slm])
    
    aberration_params = [p for p in zernike_params if p.requires_grad] + zoomfft_params_list
    
    param_groups = [
        {'params': tm_params, 'lr': learning_rate},
        {'params': physical_params, 'lr': learning_rate_phys},
        {'params': aberration_params, 'lr': learning_rate_aberrations}
    ]
    optimizer = optim.Adam(param_groups)
    
    criterion = PearsonCorrelationLoss(dim=(0, 2, 3), center=True)
    planes_l = planes.lower()
    num_measurements = slm_patterns.shape[0]
    num_batches = (num_measurements + batch_size - 1) // batch_size
    history = {
        'epoch_losses': [],
        'test_losses': [],
        'tm_norm': []
    }

    print(f"Measurements: {num_measurements} | Batches/Epoch: {num_batches}")

    # --- 3. Training Loop ---
    for epoch in range(num_epochs):
        total_epoch_loss = 0.0

        # Shuffle data
        indices = torch.randperm(num_measurements)

        # Loop over batches
        for i in range(num_batches):
            optimizer.zero_grad()
            start = i * batch_size
            end = min(start + batch_size, num_measurements)
            batch_idx = indices[start:end]
            
            # Detach batch inputs to avoid "backward through graph a second time" errors.
            # We only need gradients for model parameters, not for the data.
            batch_slm = slm_patterns[batch_idx].detach().contiguous()
            batch_target = intensity_targets[batch_idx].detach().contiguous()
            
            # Forward pass using internal parameters (use_internal_params=True by default)
            pred_intensity = tm_system.forward_model_slm(batch_slm)

            # Select Planes
            if planes_l == 'image': 
                pred, target = pred_intensity[:, 0:1], batch_target[:, 0:1]
            elif planes_l == 'fourier': 
                pred, target = pred_intensity[:, 1:2], batch_target[:, 1:2]
            else: 
                pred, target = pred_intensity, batch_target

            # Loss (Pearson correlation)
            loss = criterion(pred, target)
            
            # Backward
            loss.backward()
            
            # Step optimizer
            optimizer.step()
            
            # Track loss for reporting
            total_epoch_loss += loss.item()
        
        # Logging
        avg_loss = total_epoch_loss / num_batches
        
        # Calculate test loss if test data provided
        test_loss = 0.0
        if test_slm_patterns is not None and test_intensity_targets is not None:
            with torch.no_grad():
                test_pred = tm_system.forward_model_slm(test_slm_patterns)
                test_loss = criterion(test_pred, test_intensity_targets).item()
        
        tm_norm = torch.norm(tm_system.optimized_tm).item()

        history['epoch_losses'].append(avg_loss)
        history['test_losses'].append(test_loss)
        history['tm_norm'].append(tm_norm)

        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}, Test Loss={test_loss:.6f}, TM Norm={tm_norm:.6f}")

    print("Optimization Completed.")
    return history


def admm_step_relaxation_schedule(
    epoch_idx: int,
    *,
    rho_lo: float = 0.0,
    rho_hi: float = 0.1,
    switch_epoch: int = 45,
) -> float:
    """Step schedule for ADMM over-relaxation: ``rho_lo`` for the first
    ``switch_epoch`` epochs (1-based count), then ``rho_hi``.

    Default (``rho = 0`` for 45 iterations, then ``0.1``) matches the
    manuscript: the hard projection descends fastest at first, while the
    partial step reaches a lower final residual.
    """
    return float(rho_lo if (epoch_idx + 1) <= int(switch_epoch) else rho_hi)


def _default_admm_relaxation_schedule(epoch_idx: int) -> float:
    """Default ADMM rho: 0 for epochs 1-45, then 0.1."""
    return admm_step_relaxation_schedule(epoch_idx)


def optimize_tm_admm_dual_plane(
    tm_system: 'TMMeasurementSystem',
    train_input_modes: torch.Tensor,
    train_intensity_targets: torch.Tensor,
    num_epochs: int = 100,
    resync_interval: int = 10,
    tikhonov_epsilon: float = 1e-6,
    admm_relaxation: float = 0.3,
    admm_relaxation_schedule: Optional[Callable[[int], float]] = _default_admm_relaxation_schedule,
    init_method: str = 'default_beta_prior',
    init_power_iters: int = 50,
    init_num_pixels: int = 500,
    init_eig_rel_threshold: float = 1e-3,
    planes: str = 'both',
    on_epoch_end: Optional[Callable[[int, float, 'TMMeasurementSystem'], None]] = None,
    lambda_decay_interval: int = 15,
    lambda_decay_factor: float = 0.8,
) -> Dict:
    """
    ADMM dual-plane TM retrieval: alternates an analytic TM update with
    per-pixel intensity projection onto the measured magnitude manifold.

    Forward model per plane p in {image, fourier} and polarization c in {x, y}:
        I_meas,p = |E_x,p|^2 + |E_y,p|^2,  E_c,p = P_c,p T A,
    with unified modal TM T (n_modes x n_modes), input probing matrix A = Z_in
    (n_modes x batch), and mode projection P_c,p (n_pixels x n_modes).

    Offline precomputation:
        Intensities and input modes are globally rescaled to O(1) for float32
        stability. Plane weights w_p = 1 / mean(I_p) balance image vs Fourier.
        M_inv = pinv(sum_{c,p} w_p P_c,p^H P_c,p), Cov_in_inv = pinv(A A^H).

    Each epoch:
      1. T <- M_inv (B A^H) Cov_in_inv with B = sum_{c,p} w_p P_c,p^H (U_c,p - Lambda_c,p).
      2. Forward E, project U onto sqrt(I_meas) via per-pixel scaling.
      3. Lambda <- Lambda + E - U.

    Initialization may use the existing beta prior, random noise, or identity.
    Auxiliary fields U are
    seeded by one forward pass + intensity projection so the first TM update is
    not driven by zero fields. The optimized TM is stored in this scaled space.

    Args:
        tm_system: Calibrated TM measurement system.
        train_input_modes: Z_in, shape (N_modes, Batch).
        train_intensity_targets: Dual-plane intensities, shape (Batch, 2, H, W).
        num_epochs: Number of ADMM iterations.
        resync_interval: Accepted for API compatibility with other dual-plane
            solvers; unused by ADMM.
        tikhonov_epsilon: Accepted for API compatibility; unused (pinv used instead).
        admm_relaxation: Constant over-relaxation parameter rho used only when
            ``admm_relaxation_schedule`` is None. When 0.0, hard magnitude
            projection ``S = C / D_mag``; otherwise
            ``S = (C + rho D_mag) / ((1 + rho) D_mag)``.
        admm_relaxation_schedule: ``rho = schedule(epoch)`` (0-based epoch index)
            for init and each epoch. Default is
            ``_default_admm_relaxation_schedule`` (rho=0 for 45 epochs, then
            0.1). Pass ``None`` to use constant ``admm_relaxation``.
        init_method: ``'default_beta_prior'``, ``'random'``, or ``'identity'``.
        init_power_iters: unused unless a non-default init is requested; kept
            for call-signature compatibility.
        init_num_pixels: unused unless a non-default init is requested.
        init_eig_rel_threshold: unused unless a non-default init is requested.
        planes: Which measurement planes to optimize against ('image', 'fourier',
            or 'both'; index 0 = image, 1 = Fourier in ``train_intensity_targets``).
        on_epoch_end: Optional ``on_epoch_end(epoch, epoch_seconds, tm_system)``.
        lambda_decay_interval: Every this many epochs the dual-update coefficient is
            multiplied by ``lambda_decay_factor``.
        lambda_decay_factor: Multiplicative decay of the dual-update coefficient.

    Returns:
        history with ``epoch_losses`` (weighted mean dual-plane intensity MSE)
        and ``epoch_times``.
    """
    del resync_interval, tikhonov_epsilon  # unused; kept for caller compatibility

    device = tm_system.device
    history: Dict[str, List[float]] = {'epoch_losses': [], 'epoch_times': []}
    intensity_eps = 1e-12
    active_planes = _plane_indices_from_planes(planes)

    with torch.no_grad():
        n_modes = len(tm_system.guided_modes)
        batch_size = train_input_modes.shape[1]
        n_pixels = tm_system.grid_params['ny'] * tm_system.grid_params['nx']

        I_modes = torch.eye(n_modes, dtype=torch.complex64, device=device)
        P_complex = tm_system.output_modes_to_intensities(
            I_modes, return_complex_field=True, use_internal_params=True
        )
        P_stacked: List[torch.Tensor] = []
        P_stacked_H: List[torch.Tensor] = []
        for p in (0, 1):
            P_p = torch.stack([
                P_complex[:, p, 0].reshape(n_modes, n_pixels).T,
                P_complex[:, p, 1].reshape(n_modes, n_pixels).T,
            ], dim=0).contiguous()
            P_stacked.append(P_p)
            P_stacked_H.append(P_p.conj().transpose(-1, -2).contiguous())
        del P_complex, I_modes

        # Overlaps used only to build M_inv; free them immediately after.
        P_overlap = {
            p: P_stacked_H[p][0] @ P_stacked[p][0] + P_stacked_H[p][1] @ P_stacked[p][1]
            for p in active_planes
        }

        # --- 1. Global Scaling (Prevent Float32 Overflow) ---
        I_global_scale = torch.mean(train_intensity_targets).clamp(min=1e-12)
        Z_scale = train_input_modes.abs().max().clamp(min=1e-12)

        # === A. OFFLINE PRECOMPUTATION (Independent of Measurements) ===
        Z_in = (train_input_modes / Z_scale).to(device=device, dtype=torch.complex64).contiguous()
        Z_in_H = Z_in.conj().transpose(-1, -2).contiguous()

        Cov_in = Z_in @ Z_in_H
        Cov_in_inv = torch.linalg.pinv(Cov_in, hermitian=True, rcond=1e-4)
        del Cov_in

        # === B. ONLINE INITIALIZATION (Dependent on Measurements) ===
        _synchronize_if_cuda(device)
        online_start_time = time.perf_counter()

        targets = (train_intensity_targets / I_global_scale).to(device=device, dtype=torch.float32)
        I_meas_list: Dict[int, torch.Tensor] = {}
        C_list: Dict[int, torch.Tensor] = {}
        W_list: Dict[int, float] = {}

        for p in active_planes:
            I_meas = targets[:, p].reshape(batch_size, n_pixels).T.contiguous()
            I_meas_list[p] = I_meas
            C_list[p] = torch.sqrt(I_meas.clamp(min=0.0) + intensity_eps)
            mean_I = float(torch.mean(I_meas).clamp(min=intensity_eps))
            W_list[p] = 1.0 / mean_I
        del targets

        M_overlap = torch.zeros_like(P_overlap[active_planes[0]])
        for p in active_planes:
            M_overlap.add_(P_overlap[p], alpha=W_list[p])
        del P_overlap
        M_inv = torch.linalg.pinv(M_overlap, hermitian=True, rcond=1e-4)
        del M_overlap

        if init_method == 'default_beta_prior':
            init_method_seconds = 0.0
        else:
            init_method_seconds = _init_tm_dual_plane(
                tm_system,
                init_method,
                P_stacked,
                train_input_modes,
                train_intensity_targets,
                init_power_iters,
                init_num_pixels,
                n_pixels,
                init_eig_rel_threshold,
            )

        # Convert the physical-space initialization to ADMM's normalized coordinates.
        T_physical_init = tm_system.optimized_tm.detach().clone().contiguous()
        T = T_physical_init * (Z_scale / torch.sqrt(I_global_scale))
        del T_physical_init
        Y_init = T @ Z_in

        init_loss = 0.0
        for p in active_planes:
            E_x_init = torch.matmul(P_stacked[p][0], Y_init)
            E_y_init = torch.matmul(P_stacked[p][1], Y_init)
            I_pred_init = E_x_init.real ** 2 + E_x_init.imag ** 2 + E_y_init.real ** 2 + E_y_init.imag ** 2
            plane_mse = float(torch.mean((I_pred_init - I_meas_list[p]) ** 2))
            init_loss += plane_mse * (W_list[p] ** 2)
            del E_x_init, E_y_init, I_pred_init

        init_loss *= 0.5
        physical_init_loss = init_loss * float(I_global_scale ** 2)

        print(
            f"[ADMM Init] planes={planes} | Method: {init_method} | "
            f"Init_Weighted_MSE: {physical_init_loss:.4e}"
        )
        history['init_seconds'] = float(init_method_seconds)
        history['init_loss'] = float(physical_init_loss)
        history['init_method'] = init_method

        U: Dict[int, List[torch.Tensor]] = {}

        def _rho_at(epoch_idx: int) -> float:
            if admm_relaxation_schedule is not None:
                return float(admm_relaxation_schedule(epoch_idx))
            return float(admm_relaxation)

        rho = _rho_at(0)
        use_relaxed_projection = rho != 0.0

        for p in active_planes:
            E_x_init = torch.matmul(P_stacked[p][0], Y_init)
            E_y_init = torch.matmul(P_stacked[p][1], Y_init)
            D_mag_init = torch.sqrt(
                E_x_init.real ** 2 + E_x_init.imag ** 2 + E_y_init.real ** 2 + E_y_init.imag ** 2 + intensity_eps
            )

            if use_relaxed_projection:
                target_mag = (C_list[p] + rho * D_mag_init) / (1.0 + rho)
                S_p_init = target_mag / D_mag_init
            else:
                S_p_init = C_list[p] / D_mag_init

            U[p] = [S_p_init * E_x_init, S_p_init * E_y_init]
            del E_x_init, E_y_init, D_mag_init, S_p_init
        del Y_init

        Lambda: Dict[int, List[torch.Tensor]] = {
            p: [
                torch.zeros((n_pixels, batch_size), dtype=torch.complex64, device=device),
                torch.zeros((n_pixels, batch_size), dtype=torch.complex64, device=device),
            ]
            for p in active_planes
        }

        _synchronize_if_cuda(device)
        online_init_seconds = time.perf_counter() - online_start_time

        lambda_coeff = 1.0
        history['admm_relaxation_schedule'] = []

        for epoch in range(num_epochs):
            _synchronize_if_cuda(device)
            epoch_start = time.perf_counter()

            rho = _rho_at(epoch)
            use_relaxed_projection = rho != 0.0
            history['admm_relaxation_schedule'].append(float(rho))

            # Step 1: analytic TM update.
            B = torch.zeros((n_modes, batch_size), dtype=torch.complex64, device=device)
            for p in active_planes:
                for pol in (0, 1):
                    V_cp = U[p][pol] - Lambda[p][pol]
                    B.add_(P_stacked_H[p][pol] @ V_cp, alpha=W_list[p])
                    del V_cp
            T = M_inv @ (B @ Z_in_H) @ Cov_in_inv
            del B

            # Step 2: forward + intensity projection. Build E plane-by-plane so we
            # never double-buffer all 4 pol/plane fields at once.
            Y = T @ Z_in
            epoch_loss_eval = 0.0
            for p in active_planes:
                E_x = torch.matmul(P_stacked[p][0], Y)
                E_y = torch.matmul(P_stacked[p][1], Y)
                I_pred = E_x.real ** 2 + E_x.imag ** 2 + E_y.real ** 2 + E_y.imag ** 2
                plane_mse = float(torch.mean((I_pred - I_meas_list[p]) ** 2))
                epoch_loss_eval += plane_mse * (W_list[p] ** 2)
                del I_pred

                D_x = E_x + Lambda[p][0]
                D_y = E_y + Lambda[p][1]
                D_mag = torch.sqrt(
                    D_x.real ** 2 + D_x.imag ** 2 + D_y.real ** 2 + D_y.imag ** 2 + intensity_eps
                )
                if use_relaxed_projection:
                    target_mag = (C_list[p] + rho * D_mag) / (1.0 + rho)
                    S_p = target_mag / D_mag
                    del target_mag
                else:
                    S_p = C_list[p] / D_mag
                del D_mag

                U_x = S_p * D_x
                U_y = S_p * D_y
                del D_x, D_y, S_p

                # Step 3: Lambda update (needs E and new U).
                Lambda[p][0].add_(E_x).sub_(U_x).mul_(lambda_coeff)
                Lambda[p][1].add_(E_y).sub_(U_y).mul_(lambda_coeff)
                del E_x, E_y
                U[p][0] = U_x
                U[p][1] = U_y
            del Y

            if (epoch + 1) % lambda_decay_interval == 0:
                lambda_coeff *= lambda_decay_factor

            epoch_loss_eval *= 0.5

            _synchronize_if_cuda(device)
            epoch_seconds = time.perf_counter() - epoch_start

            if epoch == 0:
                epoch_seconds += online_init_seconds

            tm_system.optimized_tm_tensor = T

            history['epoch_losses'].append(float(epoch_loss_eval))
            history['epoch_times'].append(float(epoch_seconds))

            print(
                f"[ADMM] Epoch {epoch}: planes={planes} | Weighted MSE = {epoch_loss_eval:.5e} | "
                f"time={epoch_seconds:.2f}s"
            )

            # Free ADMM field buffers before the focus callback so SLM Adam
            # does not share the GPU with a full dual-plane E workspace.
            if on_epoch_end is not None:
                on_epoch_end(epoch, epoch_seconds, tm_system)

    return history
