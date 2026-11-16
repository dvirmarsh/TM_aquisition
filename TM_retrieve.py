"""
Transmission Matrix Retrieval System

This module implements a complete system for retrieving transmission matrices (TMs)
from intensity measurements using optimization methods.

Camera weight (optional). A beam splitter between the two cameras can transmit and reflect
the two polarization components in different ratios. Intensity data fix only the weights
of one camera relative to the other, so the image-plane camera keeps |E_x|^2 + |E_y|^2
and the Fourier-plane camera records (1 + b)|E_x|^2 + (1 - b)|E_y|^2. Set a fixed value
with ``tm_system.fourier_pol_weights = (1 + b, 1 - b)`` or fit b together with the TM
after ``tm_system.enable_learnable_fourier_weights(b_init)``. The default, b = 0, is the
polarization-summed model used for all results of the paper except the test of the
polarization axes (Supplementary Section S7).
"""

import math

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from mode_solver import find_guided_modes
from utils import (
    load_or_calculate_mode_fields,
    sample_tm_perturbed_mmf,
    ZoomFFT2D_Optimizable,
    ZernikeAberrationLayer,
)


class TMMeasurementSystem(nn.Module):
    """
    A class that handles the physical environment for transmission matrix measurements.
    
    This class manages:
    - Guided mode calculations and storage
    - Real and optimized TM sampling
    - Intensity calculations (I = |E_x|^2 + |E_y|^2)
    - Mode distribution sampling
    - Field calculations
    
    As an nn.Module, it contains learnable parameters for:
    - Transmission matrix (optimized_tm)
    - Zernike aberration coefficients (4 sets)
    - ZoomFFT parameters (3 sets)
    - SLM complex amplitude field
    """
    
    def __init__(self, fiber_params: Dict, grid_params: Dict, device: str = None,
                 mult_factor: Dict = None, zoomfft_params: Optional[Dict] = None, 
                 zernike_n_modes: int = 8, lab_data: bool = False, dx: Dict = None,
                 theta_before_slm_init: float = 0.0, cache_dir: Optional[str] = None):
        """
        Initialize the TM measurement system.
        
        Args:
            fiber_params: Dictionary containing fiber parameters
                - 'a': Core radius in meters
                - 'n1': Core refractive index
                - 'n0': Cladding refractive index
                - 'wavelength': Wavelength in meters
            grid_params: Dictionary containing grid parameters
                - 'x_range': Tuple of (x_min, x_max)
                - 'y_range': Tuple of (y_min, y_max)
                - 'nx': Number of x grid points
                - 'ny': Number of y grid points
            device: Device to use for computations ('cpu' or 'cuda'). If None, auto-detect.
            mult_factor: Dictionary containing multiplication factors for each side
                - 'tran': Multiplication factor for tran→fourier transform
                - 'slm': Multiplication factor for slm→fourier transform
            zoomfft_params: Dictionary with ZoomFFT base parameters for each side
                - 'tran': Dictionary with ZoomFFT base parameters for tran→fourier transform
                - 'slm': Dictionary with ZoomFFT base parameters for slm→fourier transform
            zernike_n_modes: Number of Zernike modes to use (default 12)
            lab_data: If True, use lab data instead of sampling a TM
            dx: Dictionary containing pixel sizes for each side
                - 'im': Pixel size in the transmission image plane
                - 'four': Pixel size in the transmission fourier plane
                - 'slm': Pixel size of the SLM pattern
            theta_before_slm_init: Initial polarization angle before SLM in radians (default 0.0).
            cache_dir: Directory for the mode-field cache. Default is the current working
                directory (same as the paper scripts). The demo passes data/mode_fields/.
        """
        super().__init__()
        self.fiber_params = fiber_params
        self.grid_params = grid_params
        self.lab_data = lab_data
        # (w_x, w_y) of the Fourier-plane camera relative to the image-plane camera.
        # (1, 1) is the polarization-summed model; see the module docstring.
        self.fourier_pol_weights = (1.0, 1.0)
        if dx is None:
            dx = {
                'im': 1.0,
                'four': 1.0,
                'slm': 1.0
            }
        self.dx = dx
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = 'cpu'
                print("CUDA not available, using CPU")
        else:
            self.device = device
            if device == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'
        
        # Store N_pad and mult_factors BEFORE mode_fields calculation
        if mult_factor is None:
            mult_factor = {
                'tran': 1.0,
                'slm': 1.0
            }
        self.mult_factor = mult_factor
        
        # Create coordinate grids
        x = np.linspace(grid_params['x_range'][0], grid_params['x_range'][1], grid_params['nx'])
        y = np.linspace(grid_params['y_range'][0], grid_params['y_range'][1], grid_params['ny'])
        self.X, self.Y = np.meshgrid(x, y)
        
        # Calculate guided modes
        print("Calculating guided modes...")
        self.guided_modes = find_guided_modes(
            fiber_params['a'], 
            fiber_params['n1'], 
            fiber_params['n0'], 
            fiber_params['wavelength']
        )
        
        # Calculate and store mode fields (load if exists, calculate if not)
        # If use_slm=True, only store image plane mode fields (not Fourier plane)
        # Note: mode_fields uses mult_factor_out for Fourier plane calculation
        self.mode_fields = load_or_calculate_mode_fields(
            self.guided_modes,
            self.fiber_params,
            self.grid_params,
            self.X,
            self.Y,
            self.device,
            self.mult_factor['tran'],
            image_plane_only=True,
            cache_dir=cache_dir,
        )
        # Store n_order for each mode (same order as mode_fields last dimension)
        self.mode_n_orders = [m['n_order'] for m in self.guided_modes]
        self.mode_powers = torch.sum(torch.abs(self.mode_fields[0, 0, :, :, :])**2 + torch.abs(self.mode_fields[0, 1, :, :, :])**2, dim=(0,1)) # (N_modes,)
        
        # Calculate sum of intensities inside the core for each mode
        # Convert coordinate grids to torch tensors
        X_torch = torch.tensor(self.X, device=self.device, dtype=torch.float32)
        Y_torch = torch.tensor(self.Y, device=self.device, dtype=torch.float32)
        
        # Create mask for points inside the core (r <= a)
        r = torch.sqrt(X_torch**2 + Y_torch**2)
 
        # Sample real TM and initialization TM
        N_modes = len(self.guided_modes)
        print(f"Sampling transmission matrices for {N_modes} modes...")
        
        if not lab_data:
            # Real TM (the one we want to find)
            self.real_tm = sample_tm_perturbed_mmf(
                N_modes=N_modes,
                loss_parameter=1.0,  # 100% transmission efficiency
                time_reversal_symmetry_broken=False,
                noise_level=0.0
            )
        
        optimized_tm_init = self._make_optimized_tm_init_numpy(self.guided_modes)

        # Convert to torch tensors
        # real_tm_tensor is registered as a buffer (not learnable, but moves with module)
        if not lab_data:
            self.register_buffer('real_tm_tensor', torch.tensor(self.real_tm, dtype=torch.complex64, device=device))

        self._optimized_tm_param = nn.Parameter(torch.tensor(optimized_tm_init, dtype=torch.complex64, device=device))
        
        # Initialize aberration models
        H, W = grid_params['ny'], grid_params['nx']
        
        # Initialize ZoomFFT parameters
        if zoomfft_params is None:
            # Default values - user should provide these. Keyed per side, since
            # the loop below indexes zoomfft_params['tran'] / ['slm'].
            zoomfft_params = {
                side: {
                    'a_init': 1.0,
                    'n0_init': (H - 1) / 2,
                    'm0_init': (W - 1) / 2,
                    'k0_init': (H - 1) / 2,
                    'l0_init': (W - 1) / 2,
                }
                for side in ('tran', 'slm')
            }
            print("Warning: Using default ZoomFFT parameters. Consider providing zoomfft_params.")
        
        # Create ZoomFFT instance (can be reused with different parameters)
        self.zoomfft = {}
        for side in ['tran', 'slm']:
            self.zoomfft[side] = ZoomFFT2D_Optimizable(
                N=H, M=W,
                a_init=zoomfft_params[side]['a_init'],
                n0_init=zoomfft_params[side]['n0_init'],
                m0_init=zoomfft_params[side]['m0_init'],
                k0_init=zoomfft_params[side]['k0_init'],
                l0_init=zoomfft_params[side]['l0_init']
            ).to(self.device)
        
        # Create single Zernike aberration layer (stores only the basis, not parameters)
        # We use the same layer with different coefficients for each aberration location
        self.zernike_layer = ZernikeAberrationLayer(H, W, n_modes=zernike_n_modes).to(self.device)
        
        self.zernike_n_modes = zernike_n_modes
        
        # Initialize learnable Zernike coefficients (4 sets, all initialized to zero)
        self.zernike_before_ft = nn.Parameter(torch.zeros(zernike_n_modes, device=self.device))
        self.zernike_after_ft = nn.Parameter(torch.zeros(zernike_n_modes, device=self.device))
        self.zernike_image_plane = nn.Parameter(torch.zeros(zernike_n_modes, device=self.device))
        self.zernike_fourier_domain = nn.Parameter(torch.zeros(zernike_n_modes, device=self.device))
        
        # Initialize learnable ZoomFFT parameters (3 sets: slm, out_fwd, out_inv)
        # gamma is fixed at 0, other params initialized to 1.0
        def _create_zoomfft_param_dict():
            return nn.ParameterDict({
                's_a_x': nn.Parameter(torch.tensor(1.0, device=self.device)),
                's_a_y': nn.Parameter(torch.tensor(1.0, device=self.device)),
                's_n0': nn.Parameter(torch.tensor(1.0, device=self.device)),
                's_m0': nn.Parameter(torch.tensor(1.0, device=self.device)),
                's_k0': nn.Parameter(torch.tensor(1.0, device=self.device)),
                's_l0': nn.Parameter(torch.tensor(1.0, device=self.device))
            })
        
        self.zoomfft_slm_params = _create_zoomfft_param_dict()
        self.zoomfft_out_fwd_params = _create_zoomfft_param_dict()
        self.zoomfft_out_inv_params = _create_zoomfft_param_dict()
        
        # parameters for the polarization tilting
        self.theta_before_slm = nn.Parameter(torch.tensor(theta_before_slm_init, device=self.device))  # not perfect x polarization
        self.theta_after_slm = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=False)  # tilting after SLM

        # Fixed gamma parameter (not learnable)
        self.register_buffer('gamma_fixed', torch.tensor(0.0, device=self.device))
        
        # Initialize learnable SLM complex amplitude field (1 inside circle of radius 60, 0 outside)
        # Create coordinate grids centered at (H/2, W/2)
        y_coords = torch.arange(H, dtype=torch.float32, device=self.device) - (H - 1) / 2
        x_coords = torch.arange(W, dtype=torch.float32, device=self.device) - (W - 1) / 2
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # Calculate distance from center
        r = torch.sqrt(X**2 + Y**2)
        # Create mask: 1 inside circle (r <= 60), 0 outside
        circle_mask = (r <= 64.0).to(torch.complex64)
        self.slm_complex_amplitude = nn.Parameter(circle_mask)
        
        # --- Performance Optimization: Pre-flatten mode fields ---
        # Image plane mode fields (always present)
        # mode_fields shape: (num_planes, 2, H, W, N_modes)
        self.register_buffer('mode_fields_flat_x', 
            self.mode_fields[0, 0, :, :, :].reshape(-1, N_modes).contiguous())
        self.register_buffer('mode_fields_flat_y', 
            self.mode_fields[0, 1, :, :, :].reshape(-1, N_modes).contiguous())
        
        # Cache conjugated versions for projection operations
        self.register_buffer('mode_fields_flat_x_conj', torch.conj(self.mode_fields_flat_x))
        self.register_buffer('mode_fields_flat_y_conj', torch.conj(self.mode_fields_flat_y))
        
        # Pre-flatten for calculate_fields_from_mode_dist (single plane case)
        # Shape: (1, 2, H, W, N_modes) -> (1*2*H*W, N_modes)
        selected_fields_single = self.mode_fields[0:1].contiguous()
        P_single, C, H, W, N_modes_field = selected_fields_single.shape
        self.register_buffer('mode_fields_matrix_single', 
            selected_fields_single.reshape(P_single * C * H * W, N_modes_field))
        
        # Calibration state: set to True after calibrate() is called
        self._calibrated = False

        print(f"System initialized with {N_modes} guided modes")
        print(f"Grid size: {grid_params['nx']} x {grid_params['ny']}")
        print(f"Aberration models initialized: ZoomFFT and {zernike_n_modes} Zernike modes")
        print(f"Performance optimizations: Pre-flattened mode fields for fast projection")
    
    @staticmethod
    def _make_optimized_tm_init_numpy(
        guided_modes: List[Dict],
        noise_std: float = 0.01,
    ) -> np.ndarray:
        """diag(exp(i*beta_mode)) + complex Gaussian noise (TM starting point)."""
        beta_modes = np.array([m['beta'] for m in guided_modes], dtype=np.float64)
        n_modes = len(beta_modes)
        noise = noise_std * (
            np.random.randn(n_modes, n_modes) + 1j * np.random.randn(n_modes, n_modes)
        ).astype(np.complex64)
        return np.diag(np.exp(1j * beta_modes)).astype(np.complex64) + noise

    def _make_optimized_tm_init_tensor(self, noise_std: float = 0.01) -> torch.Tensor:
        init_np = self._make_optimized_tm_init_numpy(self.guided_modes, noise_std=noise_std)
        return torch.tensor(init_np, dtype=torch.complex64, device=self.device)

    def reinitialize_optimized_tm(self, noise_std: float = 0.01) -> torch.Tensor:
        """
        Reset the learnable TM to diag(exp(i*beta_mode)) + noise, using guided-mode
        propagation constants from ``self.guided_modes``.
        """
        init = self._make_optimized_tm_init_tensor(noise_std=noise_std)
        with torch.no_grad():
            self._optimized_tm_param.data.copy_(init)
        return self.optimized_tm.detach().clone()

    def validate_aberrated_mode_basis(self, atol: float = 1e-4) -> Dict[str, object]:
        """
        Check that calibrated output modes and learned SLM-path aberrations are active.

        After phase-1 aberration learning and ``calibrate()``, PCC / intensity forward
        models must use:
          - cached aberrated output mode fields (_cal_* buffers), and
          - learned SLM-side Zernike / ZoomFFT params in ``slm_to_input_modes``.
        """
        report: Dict[str, object] = {
            'calibrated': bool(self._calibrated),
            'output_calibrated_buffers_present': False,
            'output_calibrated_matches_recompute': False,
            'input_aberrations_active': False,
            'zernike_image_plane_norm': float(torch.norm(self.zernike_image_plane).item()),
            'zernike_fourier_domain_norm': float(torch.norm(self.zernike_fourier_domain).item()),
            'zernike_before_ft_norm': float(torch.norm(self.zernike_before_ft).item()),
            'zernike_after_ft_norm': float(torch.norm(self.zernike_after_ft).item()),
        }

        if not self._calibrated:
            return report

        has_cal_buffers = hasattr(self, '_cal_image_x') and hasattr(self, '_cal_fourier_pre_x')
        report['output_calibrated_buffers_present'] = has_cal_buffers
        if not has_cal_buffers:
            return report

        with torch.no_grad():
            n_modes = len(self.guided_modes)
            unit = torch.zeros(n_modes, 1, dtype=torch.complex64, device=self.device)
            unit[0] = 1.0

            E_fast = self.output_modes_to_intensities(
                unit, return_complex_field=True, use_internal_params=True
            )
            E_manual_x = torch.matmul(self._cal_image_x, unit).reshape(
                self.grid_params['ny'], self.grid_params['nx']
            )
            E_manual_y = torch.matmul(self._cal_image_y, unit).reshape(
                self.grid_params['ny'], self.grid_params['nx']
            )
            E_manual = torch.stack([E_manual_x, E_manual_y], dim=0).unsqueeze(0)
            report['output_calibrated_matches_recompute'] = bool(
                torch.allclose(E_fast[0, 0, 0], E_manual[0, 0], atol=atol)
            )

            E_zero_out = self.output_modes_to_intensities(
                unit,
                return_complex_field=True,
                use_internal_params=True,
                zernike_coeffs_image_plane=torch.zeros_like(self.zernike_image_plane),
                zernike_coeffs_fourier_domain=torch.zeros_like(self.zernike_fourier_domain),
            )
            report['output_differs_from_zero_aberration'] = bool(
                not torch.allclose(E_fast, E_zero_out, atol=atol)
            )

            slm = torch.rand(1, self.grid_params['ny'], self.grid_params['nx'], device=self.device) * (
                2 * np.pi
            )
            z_in_learned = self.slm_to_input_modes(
                slm, use_internal_params=True, apply_slm_amplitude=True, active_polarization='x_only'
            )
            z_in_zero = self.slm_to_input_modes(
                slm,
                use_internal_params=True,
                apply_slm_amplitude=True,
                active_polarization='x_only',
                zernike_coeffs_before_ft=torch.zeros_like(self.zernike_before_ft),
                zernike_coeffs_after_ft=torch.zeros_like(self.zernike_after_ft),
            )
            report['input_aberrations_active'] = bool(
                not torch.allclose(z_in_learned, z_in_zero, atol=atol)
            )

        report['ok'] = (
            report['calibrated']
            and report['output_calibrated_buffers_present']
            and report['output_calibrated_matches_recompute']
            and report['input_aberrations_active']
        )
        return report

    @property
    def optimized_tm_tensor(self) -> torch.Tensor:
        """Backward compatibility property for optimized_tm_tensor."""
        return self.optimized_tm
    
    @optimized_tm_tensor.setter
    def optimized_tm_tensor(self, value: torch.Tensor):
        """Backward compatibility setter for optimized_tm_tensor."""
        self._optimized_tm_param.data = value

    @property
    def optimized_tm(self) -> torch.Tensor:
        """Return the current optimized TM."""
        return self._optimized_tm_param
    
    def project_field_to_modes(self, E_field: torch.Tensor) -> torch.Tensor:
        """
        Project a field onto image plane mode fields to get mode distribution.
        Supports both single polarization (E_y only) and both polarizations (E_x, E_y).
        
        Args:
            E_field: Field in image plane, shape:
                - (H, W) or (batch, H, W) for y-polarized field (backward compatible)
                - (batch, 2, H, W) for both x and y polarizations
                - (2, H, W) for both x and y polarizations (single batch)
            
        Returns:
            torch.Tensor: Mode amplitudes, shape (N_modes,) or (N_modes, batch)
        """
        # Use pre-flattened and conjugated mode fields for performance
        N_modes = self.mode_fields_flat_x_conj.shape[1]
        
        # Check if input has both polarizations (shape includes dimension for E_x/E_y)
        has_both_polarizations = (E_field.dim() == 4 and E_field.shape[-3] == 2) or (E_field.dim() == 3 and E_field.shape[0] == 2)
        
        if has_both_polarizations:
            # Both polarizations provided: (batch, 2, H, W) or (2, H, W)
            if E_field.dim() == 3:
                # Single batch: (2, H, W) -> add batch dimension
                E_field = E_field.unsqueeze(0)  # (1, 2, H, W)
                single_batch = True
            else:
                single_batch = False
            
            batch_size = E_field.shape[0]
            E_x = E_field[:, 0, :, :]  # (batch, H, W)
            E_y = E_field[:, 1, :, :]  # (batch, H, W)
            
            # Flatten fields
            E_x_flat = E_x.reshape(batch_size, -1)  # (batch, H*W)
            E_y_flat = E_y.reshape(batch_size, -1)  # (batch, H*W)
            
            # Use pre-flattened conjugated mode fields (no reshape/conj overhead)
            # Project both components: sum(E_x * conj(mode_field_x) + E_y * conj(mode_field_y))
            mode_amplitudes_x = torch.matmul(E_x_flat, self.mode_fields_flat_x_conj)  # (batch, N_modes)
            mode_amplitudes_y = torch.matmul(E_y_flat, self.mode_fields_flat_y_conj)  # (batch, N_modes)
            mode_amplitudes = (mode_amplitudes_x + mode_amplitudes_y) / self.mode_powers  # (batch, N_modes)
            
            if single_batch:
                return mode_amplitudes.squeeze(0).T  # (N_modes,)
            return mode_amplitudes.T  # (N_modes, batch)
        else:
            # Backward compatibility: only y-polarization (H, W) or (batch, H, W)
            # Handle batch dimension
            if E_field.dim() == 2:
                # Single field: (H, W)
                E_y_flat = E_field.flatten()  # (H*W,)
                # Use pre-flattened conjugated mode fields (no reshape/conj overhead)
                # Project: sum(E_y * conj(mode_field)) over spatial grid
                mode_amplitudes = torch.sum(E_y_flat.unsqueeze(1) * self.mode_fields_flat_y_conj, dim=0) / self.mode_powers # (N_modes,)
                return mode_amplitudes
            else:
                # Batch: (batch, H, W)
                batch_size = E_field.shape[0]
                E_y_flat = E_field.reshape(batch_size, -1)  # (batch, H*W)
                # Use pre-flattened conjugated mode fields (no reshape/conj overhead)
                # Project: sum(E_y * conj(mode_field)) over spatial grid for each batch
                mode_amplitudes = torch.matmul(E_y_flat, self.mode_fields_flat_y_conj) / self.mode_powers # (batch, N_modes)
                return mode_amplitudes.T  # (N_modes, batch)
    
    def calculate_fields_from_mode_dist(self, mode_distribution: torch.Tensor) -> torch.Tensor:
        """
        Calculate complex electric fields (E_x, E_y) directly from mode distribution (no TM applied).
        
        Args:
            mode_distribution: Tensor of shape (N_modes, batch_size).
        Returns:
            torch.Tensor: (batch_size, 1, 2, H, W) — image plane, (E_x, E_y).
        """
        # Image-plane mode fields only.
        # Shape: (P*C*H*W, N_modes)
        total_fields_flat = torch.matmul(self.mode_fields_matrix_single, mode_distribution)  # (P*C*H*W, batch)
        batch_size = mode_distribution.shape[1]
        # Reconstruct spatial dimensions
        P = 1  # Single plane
        C = self.mode_fields.shape[1]  # 2 (polarizations)
        H = self.mode_fields.shape[2]
        W = self.mode_fields.shape[3]
        total_fields = total_fields_flat.view(P, C, H, W, batch_size)
        total_fields = total_fields.permute(4, 0, 1, 2, 3)
        return total_fields
    
    @torch.no_grad()
    def calibrate(self) -> None:
        """
        Pre-compute aberrated output mode fields in both planes using the current optimized parameters.

        After calibration, ``output_modes_to_intensities`` replaces its full ZoomFFT + Zernike
        pipeline with a single matrix multiplication per plane.  The speedup comes from linearity:
        every operation between the mode-distribution vector and the final field is linear, so the
        combined operator can be pre-applied once to each basis mode and cached.

        The cached matrices (registered as non-learnable buffers) store the response of each mode
        at each spatial pixel **after** all fixed aberrations have been applied:

        Cached matrices:
          1. ``_cal_fourier_pre``  – mode fields after (Zernike_image → FFT × mult).
             Used for ``I_fourier`` (the Fourier-plane intensity before the Fourier-domain Zernike).
          2. ``_cal_fourier_post`` – mode fields after (Zernike_image → FFT × mult → Zernike_fourier).
             Returned as ``E_fourier`` when ``return_complex_field=True``.
          3. ``_cal_image``        – mode fields after the full chain (→ IFFT ÷ mult).
             Used for ``I_image`` and returned as ``E_image`` when ``return_complex_field=True``.

        All matrices are stored with shape ``(H*W, N_modes)`` per polarisation component so that
        the calibrated forward simply computes::

            E_flat = cal_matrix @ output_mode_dist   # (H*W, batch)

        Call this method once after all parameters have been optimised.  Re-call if any output-path
        parameter (Zernike image/Fourier, ZoomFFT out_fwd/out_inv) changes after the fact.

        Note: the calibrated path is only engaged when ``use_internal_params=True`` and no explicit
        override parameters are passed to ``output_modes_to_intensities``.
        """
        self._calibrated = False  # prevent stale fast-path during recomputation

        H = self.grid_params['ny']
        W = self.grid_params['nx']
        N_modes = self.mode_fields.shape[-1]
        mult = self.mult_factor['tran']

        zoomfft_fwd = self._get_zoomfft_params_dict(self.zoomfft_out_fwd_params)
        zoomfft_inv = self._get_zoomfft_params_dict(self.zoomfft_out_inv_params)
        z_image   = self.zernike_image_plane
        z_fourier = self.zernike_fourier_domain

        def _modes_to_flat(mode_tensor_2HWM: torch.Tensor) -> torch.Tensor:
            """(2, H, W, N_modes) → (N_modes, 2, H, W) batch-friendly layout."""
            return mode_tensor_2HWM.permute(3, 0, 1, 2).contiguous()  # (N_modes, 2, H, W)

        def _flatten_spatial(e: torch.Tensor) -> torch.Tensor:
            """(N_modes, 2, H, W) per-pol → (H*W, N_modes)."""
            # e[:, pol] shape: (N_modes, H, W) → transpose to (H*W, N_modes)
            return e.reshape(N_modes, H * W).T.contiguous()

        # ------------------------------------------------------------------
        # Single chain:
        # mode_m  →  Zernike_image  →  FFT×mult  →  [I_fourier here]
        #         →  Zernike_fourier  →  IFFT÷mult  →  [I_image here]
        # ------------------------------------------------------------------
        image_modes = _modes_to_flat(self.mode_fields[0])          # (N_modes, 2, H, W)
        image_modes_ab = self.zernike_layer(image_modes, z_image)  # (N_modes, 2, H, W)

        e_four_pre = self.zoomfft['tran'](image_modes_ab, inverse=False, **zoomfft_fwd) * mult
        e_four_post = self.zernike_layer(e_four_pre, z_fourier)
        e_imag = self.zoomfft['tran'](e_four_post, inverse=True, **zoomfft_inv) / mult

        self.register_buffer('_cal_fourier_pre_x',  _flatten_spatial(e_four_pre[:, 0]))
        self.register_buffer('_cal_fourier_pre_y',  _flatten_spatial(e_four_pre[:, 1]))
        self.register_buffer('_cal_fourier_post_x', _flatten_spatial(e_four_post[:, 0]))
        self.register_buffer('_cal_fourier_post_y', _flatten_spatial(e_four_post[:, 1]))
        self.register_buffer('_cal_image_x',        _flatten_spatial(e_imag[:, 0]))
        self.register_buffer('_cal_image_y',        _flatten_spatial(e_imag[:, 1]))

        self._calibrated = True
        path = "default"
        reinit_tm = self.reinitialize_optimized_tm()
        validation = self.validate_aberrated_mode_basis()
        print(f"Calibration complete ({path} path). "
              f"Aberrated output mode fields pre-computed for {N_modes} modes "
              f"({H}×{W} grid). Fast matmul path enabled in output_modes_to_intensities.")
        print(
            f"Optimized TM reinitialized to diag(exp(i*beta)) + noise "
            f"(Frobenius norm={torch.norm(reinit_tm).item():.4f})."
        )
        if validation.get('ok'):
            print("Aberrated mode basis validation: OK "
                  "(calibrated output buffers + learned SLM-path aberrations active).")
        else:
            print(f"Aberrated mode basis validation: {validation}")

    def _get_zoomfft_params_dict(self, param_dict: nn.ParameterDict) -> Dict:
        """Convert nn.ParameterDict to regular dict with gamma included."""
        return {
            's_a_x': param_dict['s_a_x'],
            's_a_y': param_dict['s_a_y'],
            'gamma': self.gamma_fixed,
            's_n0': param_dict['s_n0'],
            's_m0': param_dict['s_m0'],
            's_k0': param_dict['s_k0'],
            's_l0': param_dict['s_l0']
        }

    def slm_to_input_modes(
        self,
        slm_pattern: torch.Tensor,
        use_internal_params: bool = True,
        zoomfft_params_slm: Optional[Dict] = None,
        zernike_coeffs_before_ft: Optional[torch.Tensor] = None,
        zernike_coeffs_after_ft: Optional[torch.Tensor] = None,
        apply_slm_amplitude: bool = True,
        active_polarization: str = 'both'
    ) -> torch.Tensor:
        """
        SLM pattern → input mode distribution (fiber input).
        Resolves only SLM-path parameters (zoomfft_slm, zernike before/after FT).

        Args:
            slm_pattern: SLM phase pattern, shape (batch, H, W).
            use_internal_params: If True, use internal learnable params; else use non-aberrative defaults.
            zoomfft_params_slm: ZoomFFT parameters for SLM→Fourier transform.
            zernike_coeffs_before_ft, zernike_coeffs_after_ft: Zernike coefficients before/after FT.
            apply_slm_amplitude: If True, multiply by learnable slm_complex_amplitude.
            active_polarization: 'both', 'x_only', or 'y_only'. Zero the other component at SLM.

        Returns:
            input_mode_dist: Shape (N_modes, batch).
        """
        # Resolve SLM-path parameters only
        if use_internal_params:
            if zoomfft_params_slm is None:
                zoomfft_params_slm = self._get_zoomfft_params_dict(self.zoomfft_slm_params)
            if zernike_coeffs_before_ft is None:
                zernike_coeffs_before_ft = self.zernike_before_ft
            if zernike_coeffs_after_ft is None:
                zernike_coeffs_after_ft = self.zernike_after_ft
        else:
            default_zoomfft_params = {
                's_a_x': 1.0, 's_a_y': 1.0, 'gamma': 0.0,
                's_n0': 1.0, 's_m0': 1.0, 's_k0': 1.0, 's_l0': 1.0
            }
            if zoomfft_params_slm is None:
                zoomfft_params_slm = default_zoomfft_params
            if zernike_coeffs_before_ft is None:
                zernike_coeffs_before_ft = torch.zeros(self.zernike_n_modes, device=self.device)
            if zernike_coeffs_after_ft is None:
                zernike_coeffs_after_ft = torch.zeros(self.zernike_n_modes, device=self.device)

        batch_size, H, W = slm_pattern.shape

        # Start with unit field, split polarization before SLM
        cos_theta_before = torch.cos(self.theta_before_slm)
        sin_theta_before = torch.sin(self.theta_before_slm)
        E_field_slm = torch.zeros(batch_size, 2, H, W, dtype=torch.complex64, device=self.device)
        # E_field_slm[:, 0, :, :] = cos_theta_before
        E_field_slm[:, 1, :, :] = sin_theta_before

        # Only x-polarization is modulated by SLM pattern
        E_field_slm[:, 0, :, :] = cos_theta_before * torch.exp(1j * slm_pattern)

        # Isolate polarizations when requested (e.g. for 4-step internal reference)
        if active_polarization == 'x_only':
            E_field_slm[:, 1, :, :] = 0.0
        elif active_polarization == 'y_only':
            E_field_slm[:, 0, :, :] = 0.0

        # Apply aberration before Fourier transform
        E_field_slm = self.zernike_layer(E_field_slm, zernike_coeffs_before_ft)

        if apply_slm_amplitude:
            E_field_slm = E_field_slm * self.slm_complex_amplitude.unsqueeze(0).unsqueeze(0)

        # Rotate polarization by theta_after_slm
        cos_theta_after = torch.cos(self.theta_after_slm)
        sin_theta_after = torch.sin(self.theta_after_slm)
        E_x_slm = E_field_slm[:, 0, :, :]
        E_y_slm = E_field_slm[:, 1, :, :]
        E_x_slm_rotated = cos_theta_after * E_x_slm - sin_theta_after * E_y_slm
        E_y_slm_rotated = sin_theta_after * E_x_slm + cos_theta_after * E_y_slm

        # Fourier transform (SLM plane → Fourier plane)
        E_x_fourier = self.zoomfft['slm'](E_x_slm_rotated, inverse=False, **zoomfft_params_slm)
        E_y_fourier = self.zoomfft['slm'](E_y_slm_rotated, inverse=False, **zoomfft_params_slm)

        # Apply aberration after Fourier transform
        E_x_fourier = self.zernike_layer(E_x_fourier, zernike_coeffs_after_ft)
        E_y_fourier = self.zernike_layer(E_y_fourier, zernike_coeffs_after_ft)
        E_input_fourier = torch.stack([E_x_fourier, E_y_fourier], dim=1)

        # Project onto mode fields → input mode distribution
        input_mode_dist = self.project_field_to_modes(E_input_fourier)
        return input_mode_dist

    def input_modes_to_output_modes(
        self,
        input_mode_dist: torch.Tensor,
        tm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Input mode distribution → output mode distribution (TM application).

        Args:
            input_mode_dist: Shape (N_modes, batch).
            tm: Transmission matrix, shape (N_modes, N_modes). If None, uses self.optimized_tm.

        Returns:
            output_mode_dist: Shape (N_modes, batch).
        """
        if tm is None:
            tm = self.optimized_tm

        output_mode_dist = torch.matmul(tm, input_mode_dist)
        return output_mode_dist

    def slm_to_output_modes(
        self,
        slm_pattern: torch.Tensor,
        tm: Optional[torch.Tensor] = None,
        use_internal_params: bool = True,
        zoomfft_params_slm: Optional[Dict] = None,
        zernike_coeffs_before_ft: Optional[torch.Tensor] = None,
        zernike_coeffs_after_ft: Optional[torch.Tensor] = None,
        apply_slm_amplitude: bool = True,
        active_polarization: str = 'both'
    ) -> torch.Tensor:
        """
        SLM pattern → output mode distribution.
        Concatenation of slm_to_input_modes and input_modes_to_output_modes.

        Args:
            slm_pattern: SLM phase pattern, shape (batch, H, W).
            tm: Transmission matrix, shape (N_modes, N_modes). If None, uses self.optimized_tm.
            use_internal_params: If True, use internal learnable params; else use non-aberrative defaults.
            zoomfft_params_slm: ZoomFFT parameters for SLM→Fourier transform.
            zernike_coeffs_before_ft, zernike_coeffs_after_ft: Zernike coefficients before/after FT.
            apply_slm_amplitude: If True, multiply by learnable slm_complex_amplitude.
            active_polarization: 'both', 'x_only', or 'y_only'. Zero the other component at SLM.

        Returns:
            output_mode_dist: Shape (N_modes, batch).
        """
        input_mode_dist = self.slm_to_input_modes(
            slm_pattern,
            use_internal_params=use_internal_params,
            zoomfft_params_slm=zoomfft_params_slm,
            zernike_coeffs_before_ft=zernike_coeffs_before_ft,
            zernike_coeffs_after_ft=zernike_coeffs_after_ft,
            apply_slm_amplitude=apply_slm_amplitude,
            active_polarization=active_polarization
        )
        return self.input_modes_to_output_modes(input_mode_dist, tm=tm)

    def enable_learnable_fourier_weights(self, b_init: float = 0.0) -> None:
        """
        Make the Fourier-camera weights (1 + b, 1 - b) trainable.

        b = tanh(beta) keeps the weights positive; beta starts at atanh(b_init). The sign of b
        only selects which polarization component is labelled x. ``optimize_tm_slm`` fits beta
        with the learning rate of the TM. The fast ADMM does not use the weights.
        """
        b_init = float(min(max(float(b_init), -0.999), 0.999))
        self.fourier_weight_beta = nn.Parameter(
            torch.tensor(math.atanh(b_init), dtype=torch.float32, device=self.device)
        )

    def fourier_weight_b(self) -> float:
        """Current b of the Fourier-camera weights (1 + b, 1 - b)."""
        beta = getattr(self, "fourier_weight_beta", None)
        if beta is not None:
            return float(torch.tanh(beta.detach()))
        w_x, w_y = getattr(self, "fourier_pol_weights", (1.0, 1.0))
        return (float(w_x) - float(w_y)) / (float(w_x) + float(w_y))

    def _fourier_plane_intensity(self, E_x: torch.Tensor, E_y: torch.Tensor) -> torch.Tensor:
        """Fourier-plane camera intensity w_x |E_x|^2 + w_y |E_y|^2; (1, 1) by default."""
        beta = getattr(self, "fourier_weight_beta", None)
        if beta is not None:
            b = torch.tanh(beta)
            return (1.0 + b) * E_x.abs() ** 2 + (1.0 - b) * E_y.abs() ** 2
        w_x, w_y = getattr(self, "fourier_pol_weights", (1.0, 1.0))
        return w_x * E_x.abs() ** 2 + w_y * E_y.abs() ** 2

    def output_modes_to_intensities(
        self,
        output_mode_dist: torch.Tensor,
        use_internal_params: bool = True,
        zoomfft_params_output_forward: Optional[Dict] = None,
        zoomfft_params_output_inverse: Optional[Dict] = None,
        zernike_coeffs_image_plane: Optional[torch.Tensor] = None,
        zernike_coeffs_fourier_domain: Optional[torch.Tensor] = None,
        return_complex_field: bool = False
    ) -> torch.Tensor:
        """
        Stage 2 of forward model: output mode distribution → intensities (image and Fourier planes).
        Resolves only output-path parameters (ZoomFFT output forward/inverse, Zernike image/Fourier).

        Args:
            output_mode_dist: Shape (N_modes, batch).
            use_internal_params: If True, use internal learnable params; else use non-aberrative defaults.
            zoomfft_params_output_forward: ZoomFFT parameters for output image→Fourier.
            zoomfft_params_output_inverse: ZoomFFT parameters for Fourier→image (inverse).
            zernike_coeffs_image_plane: Zernike coefficients in image plane.
            zernike_coeffs_fourier_domain: Zernike coefficients in Fourier domain.
            return_complex_field: If True, return complex E fields (batch, 2, 2, H, W) instead of intensities.

        Returns:
            If return_complex_field False: intensities, shape (batch, 2, H, W) where [:, 0] = image, [:, 1] = Fourier.
            If return_complex_field True: complex E, shape (batch, 2, 2, H, W) where [:, plane, pol, :, :], pol 0=x, 1=y.
        """
        H = self.grid_params['ny']
        W = self.grid_params['nx']

        # ------------------------------------------------------------------
        # Fast calibrated path: single matrix multiplication per plane.
        # Active when calibrate() has been called and no per-call overrides
        # are provided (i.e. we use the exact parameters seen during calibration).
        # ------------------------------------------------------------------
        use_calibrated = (
            self._calibrated
            and use_internal_params
            and zoomfft_params_output_forward is None
            and zoomfft_params_output_inverse is None
            and zernike_coeffs_image_plane is None
            and zernike_coeffs_fourier_domain is None
        )

        if use_calibrated:
            batch_size = output_mode_dist.shape[1]

            # Fourier-plane field before Fourier-domain Zernike  (for I_fourier)
            efp_x = torch.matmul(self._cal_fourier_pre_x, output_mode_dist)  # (H*W, batch)
            efp_y = torch.matmul(self._cal_fourier_pre_y, output_mode_dist)
            I_fourier = self._fourier_plane_intensity(efp_x, efp_y).T.reshape(batch_size, H, W)

            # Final image-plane field  (for I_image)
            ei_x = torch.matmul(self._cal_image_x, output_mode_dist)         # (H*W, batch)
            ei_y = torch.matmul(self._cal_image_y, output_mode_dist)
            I_image = (ei_x.abs()**2 + ei_y.abs()**2).T.reshape(batch_size, H, W)

            if return_complex_field:
                # E_fourier returned = field after Fourier-domain Zernike
                efs_x = torch.matmul(self._cal_fourier_post_x, output_mode_dist)
                efs_y = torch.matmul(self._cal_fourier_post_y, output_mode_dist)
                E_fourier = torch.stack([
                    efs_x.T.reshape(batch_size, H, W),
                    efs_y.T.reshape(batch_size, H, W)
                ], dim=1)  # (batch, 2, H, W)
                E_image = torch.stack([
                    ei_x.T.reshape(batch_size, H, W),
                    ei_y.T.reshape(batch_size, H, W)
                ], dim=1)  # (batch, 2, H, W)
                return torch.stack([E_image, E_fourier], dim=1)

            return torch.stack([I_image, I_fourier], dim=1)

        # ------------------------------------------------------------------
        # Full pipeline path (used during optimisation or when overrides given)
        # ------------------------------------------------------------------
        if use_internal_params:
            if zoomfft_params_output_forward is None:
                zoomfft_params_output_forward = self._get_zoomfft_params_dict(self.zoomfft_out_fwd_params)
            if zoomfft_params_output_inverse is None:
                zoomfft_params_output_inverse = self._get_zoomfft_params_dict(self.zoomfft_out_inv_params)
            if zernike_coeffs_image_plane is None:
                zernike_coeffs_image_plane = self.zernike_image_plane
            if zernike_coeffs_fourier_domain is None:
                zernike_coeffs_fourier_domain = self.zernike_fourier_domain
        else:
            default_zoomfft_params = {
                's_a_x': 1.0, 's_a_y': 1.0, 'gamma': 0.0,
                's_n0': 1.0, 's_m0': 1.0, 's_k0': 1.0, 's_l0': 1.0
            }
            if zoomfft_params_output_forward is None:
                zoomfft_params_output_forward = default_zoomfft_params
            if zoomfft_params_output_inverse is None:
                zoomfft_params_output_inverse = default_zoomfft_params
            if zernike_coeffs_image_plane is None:
                zernike_coeffs_image_plane = torch.zeros(self.zernike_n_modes, device=self.device)
            if zernike_coeffs_fourier_domain is None:
                zernike_coeffs_fourier_domain = torch.zeros(self.zernike_n_modes, device=self.device)

        output_fields_image = self.calculate_fields_from_mode_dist(output_mode_dist)
        E_output = output_fields_image[:, 0, :, :, :]
        E_output = self.zernike_layer(E_output, zernike_coeffs_image_plane)
        E_fourier = self.zoomfft['tran'](E_output, inverse=False, **zoomfft_params_output_forward) * self.mult_factor['tran']
        I_fourier = self._fourier_plane_intensity(E_fourier[:, 0], E_fourier[:, 1])
        E_fourier = self.zernike_layer(E_fourier, zernike_coeffs_fourier_domain)
        E_image = self.zoomfft['tran'](E_fourier, inverse=True, **zoomfft_params_output_inverse) / self.mult_factor['tran']
        I_image = torch.abs(E_image[:, 0])**2 + torch.abs(E_image[:, 1])**2
        if return_complex_field:
            return torch.stack([E_image, E_fourier], dim=1)

        return torch.stack([I_image, I_fourier], dim=1)

    def forward_model_slm(
        self,
        slm_pattern: torch.Tensor,
        tm: Optional[torch.Tensor] = None,
        use_internal_params: bool = True,
        zoomfft_params_slm: Optional[Dict] = None,
        zoomfft_params_output_forward: Optional[Dict] = None,
        zoomfft_params_output_inverse: Optional[Dict] = None,
        zernike_coeffs_before_ft: Optional[torch.Tensor] = None,
        zernike_coeffs_after_ft: Optional[torch.Tensor] = None,
        zernike_coeffs_image_plane: Optional[torch.Tensor] = None,
        zernike_coeffs_fourier_domain: Optional[torch.Tensor] = None,
        apply_slm_amplitude: bool = True,
        active_polarization: str = 'both',
        return_complex_field: bool = False
    ) -> torch.Tensor:
        """
        Forward model: SLM pattern → intensities (image and Fourier planes).
        Concatenation of: (1) slm_to_output_modes and (2) output_modes_to_intensities.
        Uses ZoomFFT for all Fourier transforms (batched, supports arbitrary leading dimensions).
        
        The model includes polarization aberrations:
        - Before SLM: Field is split into x and y components based on theta_before_slm
          (x = cos(theta_before), y = sin(theta_before)). Only x-component is modulated by SLM.
        - After SLM: Polarization is rotated by theta_after_slm before projection to modes.
        
        Args:
            slm_pattern: SLM phase pattern, shape (H, W) or (batch, H, W)
            tm: Transmission matrix to use, shape (N_modes, N_modes). If None, uses self.optimized_tm
            use_internal_params: If True, use internal learnable parameters (Zernike, ZoomFFT).
                                If False, use provided external parameters (for simulation).
            zoomfft_params_slm: ZoomFFT parameters for SLM→Fourier transform
                Dict with keys: s_a_x, s_a_y, gamma, s_n0, s_m0, s_k0, s_l0
                If None and use_internal_params=True, uses internal params.
                If None and use_internal_params=False, uses non-aberrative values.
            zoomfft_params_output_forward: ZoomFFT parameters for output image→Fourier
            zoomfft_params_output_inverse: ZoomFFT parameters for Fourier→image (inverse)
            zernike_coeffs_before_ft: Zernike coefficients before FT, shape (n_modes,)
            zernike_coeffs_after_ft: Zernike coefficients after FT, shape (n_modes,)
            zernike_coeffs_image_plane: Zernike coefficients in image plane, shape (n_modes,)
            zernike_coeffs_fourier_domain: Zernike coefficients in Fourier domain, shape (n_modes,)
            apply_slm_amplitude: If True, multiply by learnable slm_complex_amplitude (default True)
            active_polarization: 'both', 'x_only', or 'y_only'. Zero the other component at SLM (default 'both').
            return_complex_field: If True, return complex E fields (batch, 2, 2, H, W) instead of intensities (default False).
            
        Returns:
            If return_complex_field False: Intensities, shape (batch, 2, H, W) where [:, 0] = image, [:, 1] = Fourier.
            If return_complex_field True: Complex E, shape (batch, 2, 2, H, W) where [:, plane, pol, :, :], pol 0=x, 1=y.
        """
        # Handle batch dimension
        if slm_pattern.dim() == 2:
            slm_pattern = slm_pattern.unsqueeze(0)
            single_batch = True
        else:
            single_batch = False

        # Stage 1: SLM → output mode distribution (resolves SLM-path params inside)
        output_mode_dist = self.slm_to_output_modes(
            slm_pattern, tm=tm, use_internal_params=use_internal_params,
            zoomfft_params_slm=zoomfft_params_slm,
            zernike_coeffs_before_ft=zernike_coeffs_before_ft,
            zernike_coeffs_after_ft=zernike_coeffs_after_ft,
            apply_slm_amplitude=apply_slm_amplitude,
            active_polarization=active_polarization
        )
        # Stage 2: output modes → intensities (or complex fields if return_complex_field)
        result = self.output_modes_to_intensities(
            output_mode_dist,
            use_internal_params=use_internal_params,
            zoomfft_params_output_forward=zoomfft_params_output_forward,
            zoomfft_params_output_inverse=zoomfft_params_output_inverse,
            zernike_coeffs_image_plane=zernike_coeffs_image_plane,
            zernike_coeffs_fourier_domain=zernike_coeffs_fourier_domain,
            return_complex_field=return_complex_field
        )

        if single_batch:
            result = result.squeeze(0)

        return result
    
    def sample_unknown_aberrations(
        self,
        zernike_std: float = 0.0,
        zoomfft_std: float = 0.0
    ) -> Dict:
        """
        Sample small unknown aberrations for the real system.
        
        Args:
            zernike_std: Standard deviation for Zernike coefficients (default 0.1)
            zoomfft_std: Standard deviation for ZoomFFT parameter deviations (default 0.05)
            
        Returns:
            Dict containing sampled aberration parameters:
                - zoomfft_params_slm: ZoomFFT parameters for SLM→Fourier
                - zoomfft_params_output_forward: ZoomFFT parameters for output image→Fourier
                - zoomfft_params_output_inverse: ZoomFFT parameters for Fourier→image (inverse)
                - zernike_coeffs_before_ft: Zernike coefficients before FT
                - zernike_coeffs_after_ft: Zernike coefficients after FT
                - zernike_coeffs_image_plane: Zernike coefficients in image plane
                - zernike_coeffs_fourier_domain: Zernike coefficients in Fourier domain
        """
        # Sample Zernike coefficients (small random values)
        zernike_coeffs_before_ft = torch.randn(self.zernike_n_modes, device=self.device) * zernike_std
        zernike_coeffs_after_ft = torch.randn(self.zernike_n_modes, device=self.device) * zernike_std
        zernike_coeffs_image_plane = torch.randn(self.zernike_n_modes, device=self.device) * zernike_std
        zernike_coeffs_fourier_domain = torch.randn(self.zernike_n_modes, device=self.device) * zernike_std
        
        def _sample_zoomfft_params():
            # gamma is kept at 0 (no shear aberration) to avoid instability
            # Only sample scaling and shift parameters
            return {
                's_a_x': 1.0 + torch.randn(1, device=self.device).item() * zoomfft_std,
                's_a_y': 1.0 + torch.randn(1, device=self.device).item() * zoomfft_std,
                'gamma': 0.0,  # Fixed at 0 - no shear aberration sampled
                's_n0': 1.0 + torch.randn(1, device=self.device).item() * zoomfft_std,
                's_m0': 1.0 + torch.randn(1, device=self.device).item() * zoomfft_std,
                's_k0': 1.0 + torch.randn(1, device=self.device).item() * zoomfft_std,
                's_l0': 1.0 + torch.randn(1, device=self.device).item() * zoomfft_std
            }
        
        return {
            'zoomfft_params_slm': _sample_zoomfft_params(),
            'zoomfft_params_output_forward': _sample_zoomfft_params(),
            'zoomfft_params_output_inverse': _sample_zoomfft_params(),
            'zernike_coeffs_before_ft': zernike_coeffs_before_ft,
            'zernike_coeffs_after_ft': zernike_coeffs_after_ft,
            'zernike_coeffs_image_plane': zernike_coeffs_image_plane,
            'zernike_coeffs_fourier_domain': zernike_coeffs_fourier_domain
        }

    
    def sample_measurements_slm(self, num_measurements: int, use_aberrations: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample SLM phase patterns and calculate measurements using forward model.
        
        Args:
            num_measurements: Number of measurements to sample
            use_aberrations: If True, apply random aberrations to targets (default).
                           If False, use no aberrations (for testing convergence).
            
        Returns:
            Tuple of (slm_patterns, intensity_measurements)
                - slm_patterns: (num_measurements, H, W) - SLM phase patterns (0-2π)
                - intensity_measurements: (num_measurements, 2, H, W) - intensities
        """
        print(f"Sampling {num_measurements} SLM pattern measurements (aberrations={use_aberrations})...")
        H, W = self.grid_params['ny'], self.grid_params['nx']
        
        # Sample random SLM phase patterns (0 to 2π)
        slm_patterns = torch.rand(num_measurements, H, W, device=self.device) * 2 * np.pi
        
        if use_aberrations:
            # Sample unknown aberrations for the real system
            unknown_aberrations = self.sample_unknown_aberrations()
            
            # Calculate measurements using forward model with real TM and unknown aberrations
            # use_internal_params=False to avoid using learnable parameters for simulation
            intensity_measurements = self.forward_model_slm(
                slm_patterns,
                tm=self.real_tm_tensor,
                use_internal_params=False,
                apply_slm_amplitude=False,
                zoomfft_params_slm=unknown_aberrations['zoomfft_params_slm'],
                zoomfft_params_output_forward=unknown_aberrations['zoomfft_params_output_forward'],
                zoomfft_params_output_inverse=unknown_aberrations['zoomfft_params_output_inverse'],
                zernike_coeffs_before_ft=unknown_aberrations['zernike_coeffs_before_ft'],
                zernike_coeffs_after_ft=unknown_aberrations['zernike_coeffs_after_ft'],
                zernike_coeffs_image_plane=unknown_aberrations['zernike_coeffs_image_plane'],
                zernike_coeffs_fourier_domain=unknown_aberrations['zernike_coeffs_fourier_domain']
            )
        else:
            # No aberrations - use identity ZoomFFT params and zero Zernike coeffs
            # use_internal_params=False to avoid using learnable parameters for simulation
            intensity_measurements = self.forward_model_slm(
                slm_patterns,
                tm=self.real_tm_tensor,
                use_internal_params=False,
                apply_slm_amplitude=False
            )
        
        return slm_patterns, intensity_measurements
