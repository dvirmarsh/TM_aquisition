"""
Transmission Matrix Retrieval System

This module implements a complete system for retrieving transmission matrices (TMs)
from intensity measurements using optimization methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint as cp
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mode_solver import find_guided_modes
from tm_sampler import sample_tm_perturbed_mmf
from utils import (
    load_or_calculate_mode_fields,
    clear_mode_fields_cache,
    mode_fields_cache_exists,
    plot_results as utils_plot_results,
    plot_training_intensity_example as utils_plot_training_intensity_example,
    plot_training_history as utils_plot_training_history,
    ZoomFFT2D_Optimizable,
    ZernikeAberrationLayer,
    run_spectral_initialization,
    solve_fast_cholesky
)
from metrics_losses import (
    calculate_phase_aligned_frobenius_norm,
    calculate_phase_aligned_tm_error,
    evaluate_reconstruction as metrics_evaluate_reconstruction,
    PearsonCorrelationLoss,
    ComplexCosineLoss
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
                 enforce_symmetric_tm: bool = False, theta_before_slm_init: float = 0.0):
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
            enforce_symmetric_tm: If True, optimize only the lower-triangular TM
                and reconstruct a symmetric TM as tm_tri + tm_tri.T during forward use.
            theta_before_slm_init: Initial polarization angle before SLM in radians (default 0.0).
                E.g. pi/4 to excite the y-polarized reference for 4-step internal-reference retrieval.
        """
        super().__init__()
        self.fiber_params = fiber_params
        self.grid_params = grid_params
        self.lab_data = lab_data
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
            image_plane_only=True
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
        
        # Initialization TM (starting point for optimization): diag(exp(1j*beta_mode))
        beta_modes = np.array([m['beta'] for m in self.guided_modes], dtype=np.float64)
        # Add random noise with std 0.01 to the diagonal initialization
        noise = 0.01 * (np.random.randn(len(beta_modes), len(beta_modes)) + 1j * np.random.randn(len(beta_modes), len(beta_modes))).astype(np.complex64)
        optimized_tm_init = np.diag(np.exp(1j * beta_modes)).astype(np.complex64) + noise
        
        # Convert to torch tensors
        # real_tm_tensor is registered as a buffer (not learnable, but moves with module)
        if not lab_data:
            self.register_buffer('real_tm_tensor', torch.tensor(self.real_tm, dtype=torch.complex64, device=device))

        # --- Optimized TM parameter (optionally symmetric) ---
        self.enforce_symmetric_tm = enforce_symmetric_tm
        if enforce_symmetric_tm:
            self.register_buffer('tril_mask', torch.tril(torch.ones(N_modes, N_modes, device=device, dtype=torch.bool)))
            init_tri = torch.tensor(optimized_tm_init, dtype=torch.complex64, device=device)
            init_tri = torch.where(self.tril_mask, init_tri, torch.zeros_like(init_tri))
            self._optimized_tm_param = nn.Parameter(init_tri)
        else:
            self._optimized_tm_param = nn.Parameter(torch.tensor(optimized_tm_init, dtype=torch.complex64, device=device))
        
        # Initialize aberration models
        H, W = grid_params['ny'], grid_params['nx']
        
        # Initialize ZoomFFT parameters
        if zoomfft_params is None:
            # Default values - user should provide these
            zoomfft_params = {
                'a_init': 1.0,
                'n0_init': 0.0,
                'm0_init': 0.0,
                'k0_init': 0.0,
                'l0_init': 0.0
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
        
        # When enforce_symmetric_tm is True, add Fourier-plane mode fields using transmission ZoomFFT
        if self.enforce_symmetric_tm:
            z_fwd = self._get_zoomfft_params_dict(self.zoomfft_out_fwd_params)
            image_plane = self.mode_fields  # (1, 2, H, W, N_modes)
            fields_for_fft = image_plane.permute(4, 0, 1, 2, 3)  # (N_modes, 1, 2, H, W)
            fourier_plane = self.zoomfft['tran'](fields_for_fft, inverse=False, **z_fwd)  # (N_modes, 1, 2, H, W)
            fourier_plane = fourier_plane * self.mult_factor['tran']
            fourier_plane = fourier_plane.permute(1, 2, 3, 4, 0)  # (1, 2, H, W, N_modes)
            # Detach to prevent stale computational graph during training
            fourier_plane = fourier_plane.detach()
            self.mode_fields = torch.cat([image_plane, fourier_plane], dim=0)  # (2, 2, H, W, N_modes)
            self.register_buffer('mode_n_orders_tensor', torch.tensor(self.mode_n_orders, dtype=torch.float32, device=self.device))
            self.output_tilt = nn.Parameter(torch.zeros(2, device=self.device))  # (2,) - tilt angles for each output plane
            self.slm_tilt = nn.Parameter(torch.tensor(0.0, device=self.device))  # (1,) - SLM tilt angle
            self.diag_in = nn.Parameter(torch.ones(N_modes, device=self.device, dtype=torch.complex64))
            self.diag_out = nn.Parameter(torch.ones(N_modes, device=self.device, dtype=torch.complex64))
        
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
        
        # If symmetric TM with Fourier plane, also pre-flatten for two-plane case
        if self.enforce_symmetric_tm and self.mode_fields.shape[0] == 2:
            # Fourier plane fields
            self.register_buffer('mode_fields_flat_x_fourier', 
                self.mode_fields[1, 0, :, :, :].reshape(-1, N_modes).contiguous())
            self.register_buffer('mode_fields_flat_y_fourier', 
                self.mode_fields[1, 1, :, :, :].reshape(-1, N_modes).contiguous())
            
            # Pre-flatten for calculate_fields_from_mode_dist (two-plane case)
            # Shape: (2, 2, H, W, N_modes) -> (2, 2*H*W, N_modes)
            selected_fields_dual = self.mode_fields.contiguous()
            self.register_buffer('mode_fields_matrix_dual', 
                selected_fields_dual.view(2, C * H * W, N_modes_field))
        
        print(f"System initialized with {N_modes} guided modes")
        print(f"Grid size: {grid_params['nx']} x {grid_params['ny']}")
        print(f"Aberration models initialized: ZoomFFT and {zernike_n_modes} Zernike modes")
        print(f"Performance optimizations: Pre-flattened mode fields for fast projection")
    
    def enable_torch_compile(self, mode='reduce-overhead'):
        """
        Enable torch.compile for forward methods to fuse CUDA kernels.
        
        NOTE: torch.compile currently disabled due to incompatibility with complex number
        gradients in PyTorch 2.x. The method is kept for API compatibility but does nothing.
        
        Main speedup (2-3x) comes from spectral initialization and pre-flattened tensors.
        
        Args:
            mode: Compilation mode (ignored, kept for API compatibility)
        
        Note: torch.compile will be re-enabled when PyTorch fixes complex gradient issues.
        """
        import torch
        
        # torch.compile is currently disabled due to complex number gradient issues
        # in PyTorch's inductor backend. See TORCH_COMPILE_FIX.md for details.
        print("Note: torch.compile is currently disabled due to PyTorch complex gradient issues.")
        print("Main speedup (2-3x) comes from spectral initialization + pre-flattened tensors.")
        print("This is sufficient for the target 2-4x overall speedup.")
        return
        
        # Original torch.compile code kept for when PyTorch fixes the issue:
        # if not torch.cuda.is_available():
        #     print("Warning: torch.compile is most effective on CUDA. Skipping compilation on CPU.")
        #     return
        # 
        # try:
        #     if not hasattr(torch, 'compile'):
        #         print("Warning: torch.compile not available. Requires PyTorch 2.0+. Skipping.")
        #         return
        #     
        #     self.output_modes_to_intensities = torch.compile(
        #         self.output_modes_to_intensities,
        #         mode=mode,
        #         fullgraph=False
        #     )
        #     print("✓ torch.compile enabled.")
        # except Exception as e:
        #     print(f"Warning: Could not enable torch.compile: {e}")
        #     print("Continuing without compilation.")
    
    @property
    def optimized_tm_tensor(self) -> torch.Tensor:
        """Backward compatibility property for optimized_tm_tensor."""
        return self.optimized_tm
    
    @optimized_tm_tensor.setter
    def optimized_tm_tensor(self, value: torch.Tensor):
        """Backward compatibility setter for optimized_tm_tensor."""
        if self.enforce_symmetric_tm:
            # Store only the lower triangle; upper is ignored
            value_tri = torch.where(self.tril_mask, value, torch.zeros_like(value))
            self._optimized_tm_param.data = value_tri
        else:
            self._optimized_tm_param.data = value

    @property
    def optimized_tm(self) -> torch.Tensor:
        """
        Return the current optimized TM.
        If enforce_symmetric_tm is True, reconstruct as TM_tri + TM_tri.T
        (diagonal is intentionally doubled to match requested behavior).
        """
        if self.enforce_symmetric_tm:
            tri = torch.tril(self._optimized_tm_param)
            return tri + tri.transpose(-2, -1)
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
            mode_distribution: Tensor of shape (N_modes, batch_size) or (2, N_modes, batch_size)
                for two-plane mode dist (used when enforce_symmetric_tm and mode_fields has 2 planes).
        Returns:
            torch.Tensor: (batch_size, P, 2, H, W) where P is number of planes (1 or 2), 2 is (E_x, E_y)
        """
        if mode_distribution.dim() == 3 and mode_distribution.shape[0] == 2:
            # (2, N_modes, batch) -> use both planes of mode_fields, return (batch, 2, 2, H, W)
            # Use pre-flattened mode_fields_matrix_dual: (2, C*H*W, N_modes)
            # Batched matmul: (2, C*H*W, N_modes) @ (2, N_modes, batch) -> (2, C*H*W, batch)
            batch_size = mode_distribution.shape[2]
            total_fields_flat = torch.bmm(self.mode_fields_matrix_dual, mode_distribution)  # (2, C*H*W, batch)
            # Reconstruct spatial dimensions from pre-computed shape info
            C = self.mode_fields.shape[1]  # 2 (polarizations)
            H = self.mode_fields.shape[2]
            W = self.mode_fields.shape[3]
            total_fields = total_fields_flat.view(2, C, H, W, batch_size).permute(4, 0, 1, 2, 3)  # (batch, 2, 2, H, W)
            return total_fields
        # Single plane case: use pre-flattened mode_fields_matrix_single
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
        if self.enforce_symmetric_tm:
            input_mode_dist = input_mode_dist * torch.exp(1j * self.mode_n_orders_tensor.unsqueeze(1) * self.slm_tilt) * self.diag_in.unsqueeze(1)

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
            output_mode_dist: Shape (N_modes, batch) or (2, N_modes, batch) when enforce_symmetric_tm.
        """
        if tm is None:
            tm = self.optimized_tm

        output_mode_dist = torch.matmul(tm, input_mode_dist)

        if self.enforce_symmetric_tm:
            output_mode_dist = output_mode_dist.unsqueeze(0)
            phase = self.mode_n_orders_tensor.unsqueeze(1) * self.output_tilt.unsqueeze(0)
            output_mode_dist = output_mode_dist * torch.exp(1j * phase.permute(1, 0).unsqueeze(2)) * self.diag_out.unsqueeze(0).unsqueeze(2)

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
            output_mode_dist: Shape (N_modes, batch) or (2, N_modes, batch) when enforce_symmetric_tm.
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
            output_mode_dist: Shape (N_modes, batch) or (2, N_modes, batch) when enforce_symmetric_tm.
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
        # Resolve output-path parameters only
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

        if self.enforce_symmetric_tm:
            # calculate_fields_from_mode_dist expects (2, N_modes, batch) for two planes; expand if needed
            if output_mode_dist.dim() == 2:
                output_mode_dist = output_mode_dist.unsqueeze(0).expand(2, -1, -1)
            elif output_mode_dist.dim() == 3 and output_mode_dist.shape[0] == 1:
                output_mode_dist = output_mode_dist.expand(2, -1, -1)
            output_fields_both = self.calculate_fields_from_mode_dist(output_mode_dist)
            E_image_plane = output_fields_both[:, 0, :, :, :]
            E_fourier_plane = output_fields_both[:, 1, :, :, :]
            E_image_plane = self.zernike_layer(E_image_plane, zernike_coeffs_image_plane)
            E_fourier_from_image = self.zoomfft['tran'](E_image_plane, inverse=False, **zoomfft_params_output_forward) * self.mult_factor['tran']
            I_fourier = torch.abs(E_fourier_from_image[:, 0])**2 + torch.abs(E_fourier_from_image[:, 1])**2
            E_fourier_plane = self.zernike_layer(E_fourier_plane, zernike_coeffs_fourier_domain)
            E_image_from_fourier = self.zoomfft['tran'](E_fourier_plane, inverse=True, **zoomfft_params_output_inverse) / self.mult_factor['tran']
            I_image = torch.abs(E_image_from_fourier[:, 0])**2 + torch.abs(E_image_from_fourier[:, 1])**2
            if return_complex_field:
                return torch.stack([E_image_from_fourier, E_fourier_from_image], dim=1)
        else:
            output_fields_image = self.calculate_fields_from_mode_dist(output_mode_dist)
            E_output = output_fields_image[:, 0, :, :, :]
            E_output = self.zernike_layer(E_output, zernike_coeffs_image_plane)
            E_fourier = self.zoomfft['tran'](E_output, inverse=False, **zoomfft_params_output_forward) * self.mult_factor['tran']
            I_fourier = torch.abs(E_fourier[:, 0])**2 + torch.abs(E_fourier[:, 1])**2
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
    
    def evaluate_reconstruction(self) -> Dict:
        """
        Evaluate the quality of TM reconstruction.
        
        Returns:
            Dict containing evaluation metrics
        """
        # Sample SLM patterns for evaluation
        def sample_slm_fn(batch_size):
            H, W = self.grid_params['ny'], self.grid_params['nx']
            return torch.rand(batch_size, H, W, device=self.device) * 2 * np.pi
        
        # Calculate intensity from SLM patterns
        def calculate_intensity_slm_fn(slm_patterns, use_real_tm):
            tm = self.real_tm_tensor if use_real_tm else self.optimized_tm
            # Use forward_model_slm with no aberrations for evaluation
            intensities = self.forward_model_slm(slm_patterns, tm, use_internal_params=False, apply_slm_amplitude=False)
            return intensities
        
        return metrics_evaluate_reconstruction(
            self.real_tm_tensor,
            self.optimized_tm,
            sample_slm_fn,
            calculate_intensity_slm_fn,
        )

    
    def plot_results(self, evaluation_results: Dict):
        """
        Plot optimization results and evaluation metrics.
        
        Args:
            evaluation_results: Results from evaluate_reconstruction()
        """
        # Sample SLM patterns for plotting
        def sample_slm_fn(batch_size):
            H, W = self.grid_params['ny'], self.grid_params['nx']
            return torch.rand(batch_size, H, W, device=self.device) * 2 * np.pi
        
        # Calculate intensity from SLM patterns
        def calculate_intensity_slm_fn(slm_patterns, use_real_tm):
            tm = self.real_tm_tensor if use_real_tm else self.optimized_tm
            # Use forward_model_slm with no aberrations for plotting
            intensities = self.forward_model_slm(slm_patterns, tm, use_internal_params=False, apply_slm_amplitude=False)
            return intensities
        
        utils_plot_results(
            evaluation_results,
            sample_slm_fn,
            calculate_intensity_slm_fn
        )

    
    def plot_training_intensity_example(
        self,
        mode_distributions: torch.Tensor,
        intensity_targets: torch.Tensor,
        example_idx: Optional[int] = None,
        plane: str = 'image'
    ):
        """
        Plot an example intensity comparison from the training set.
        
        Args:
            mode_distributions: Tensor of shape (N_modes, num_measurements)
            intensity_targets: Tensor of shape (num_measurements, 2, H, W)
            example_idx: Optional index of the training sample to visualize
            plane: 'image' or 'fourier' plane to visualize
        """
        utils_plot_training_intensity_example(
            mode_distributions,
            intensity_targets,
            self.calculate_intensity,
            example_idx,
            plane
        )
    
    def plot_training_history(self, history: Dict):
        """
        Plot training history showing losses as a function of epoch.
        
        Args:
            history: Training history from optimize_tm()
        """
        utils_plot_training_history(history)

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
    use_spectral_init: bool = True,
    spectral_init_iters: int = 15,
    spectral_inner_iters: int = 3,
    enable_compile: bool = False
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
        use_spectral_init: If True, run spectral initialization before gradient descent (default True)
        spectral_init_iters: Number of spectral initialization outer iterations (default 15)
        spectral_inner_iters: Number of inner refinement iterations per spectral step (default 3)
        enable_compile: If True, enable torch.compile (default False - disabled due to PyTorch complex gradient issues)
    
    Returns:
        Dict containing training history
    """
    # --- Performance Optimization: Enable torch.compile ---
    if enable_compile:
        tm_system.enable_torch_compile(mode='reduce-overhead')
    
    # --- Spectral Initialization (if enabled) ---
    if use_spectral_init:
        print("\n" + "="*60)
        print("PHASE 1: Spectral Initialization")
        print("="*60)
        run_spectral_initialization(
            tm_system,
            slm_patterns,
            intensity_targets,
            num_iters=spectral_init_iters,
            num_inner_iters=spectral_inner_iters,
            ridge_alpha=1e-4
        )
        print("="*60 + "\n")
    
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
    # TM: transmission matrix + diag_in/diag_out when enforce_symmetric_tm
    tm_params = [tm_system._optimized_tm_param]
    if getattr(tm_system, 'enforce_symmetric_tm', False):
        tm_system.diag_in.requires_grad_(True)
        tm_system.diag_out.requires_grad_(True)
        tm_params.extend([tm_system.diag_in, tm_system.diag_out])
    
    # Physical: SLM amplitude + polarization angles
    physical_params = [tm_system.slm_complex_amplitude]
    if optimize_aberrations:
        physical_params.extend([tm_system.theta_before_slm, tm_system.theta_after_slm])
    
    # Aberrations: Zernike + ZoomFFT + (when enforce_symmetric_tm) output_tilt, slm_tilt
    aberration_params = [p for p in zernike_params if p.requires_grad] + zoomfft_params_list
    if getattr(tm_system, 'enforce_symmetric_tm', False):
        tm_system.output_tilt.requires_grad_(True)
        tm_system.slm_tilt.requires_grad_(True)
        aberration_params.extend([tm_system.output_tilt, tm_system.slm_tilt])
    
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
        'tm_error': [],
        'tm_norm': [],
        'tm_frobenius_error_aligned': []
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
        
        # Calculate TM error if ground truth available
        tm_error = 0.0
        if hasattr(tm_system, 'real_tm_tensor'):
            with torch.no_grad():
                tm_error, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, tm_system.optimized_tm)
                tm_norm = torch.norm(tm_system.optimized_tm).item()
        else:
            tm_norm = torch.norm(tm_system.optimized_tm).item()
        
        history['epoch_losses'].append(avg_loss)
        history['test_losses'].append(test_loss)
        history['tm_error'].append(tm_error)
        history['tm_norm'].append(tm_norm)
        history['tm_frobenius_error_aligned'].append(tm_error)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}, Test Loss={test_loss:.6f}, TM Error={tm_error:.4f}, TM Norm={tm_norm:.6f}")

    print("Optimization Completed.")
    return history


def simulate_phase_stepped_measurements(
    tm_system: TMMeasurementSystem,
    base_slm_patterns: torch.Tensor,
    max_batch_size: Optional[int] = None,
    **forward_kwargs
) -> torch.Tensor:
    """
    Simulate 4-step phase-stepping holography: for each base SLM pattern, add global phases
    [0, π/2, π, 3π/2], run the forward model with both polarizations, and stack intensities.

    Args:
        tm_system: TMMeasurementSystem instance.
        base_slm_patterns: Base SLM phase patterns, shape (N, H, W).
        max_batch_size: If set, process patterns in chunks of this size to reduce memory (e.g. for enforce_symmetric_tm).
        **forward_kwargs: Optional kwargs passed to forward_model_slm (e.g. tm=, use_internal_params=).

    Returns:
        Intensities stacked over phase steps, shape (N, 4, num_planes, H, W) where index 1 is the 4 phase steps.
    """
    N = base_slm_patterns.shape[0]
    if max_batch_size is None or N <= max_batch_size:
        # Single pass
        phases = [0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]
        step_intensities = []
        for phase in phases:
            shifted_patterns = base_slm_patterns + phase
            intensities = tm_system.forward_model_slm(
                shifted_patterns,
                active_polarization='both',
                **forward_kwargs
            )
            step_intensities.append(intensities)
        return torch.stack(step_intensities, dim=1)
    # Chunked pass to avoid OOM (e.g. when enforce_symmetric_tm=True uses 2x field tensors)
    chunk_size = max_batch_size
    all_steps = []
    for phase in [0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]:
        step_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = base_slm_patterns[start:end] + phase
            step_chunks.append(tm_system.forward_model_slm(
                chunk,
                active_polarization='both',
                **forward_kwargs
            ))
        all_steps.append(torch.cat(step_chunks, dim=0))
    return torch.stack(all_steps, dim=1)


def extract_complex_interference_4step(intensities_4step: torch.Tensor) -> torch.Tensor:
    """
    Extract the complex interference term from 4-step phase-stepped intensity measurements.

    Expects intensities from simulate_phase_stepped_measurements with shape (N, 4, ...).
    Uses the standard 4-step formula:
      Y = ((I_0 - I_π) / 2) + 1j * ((I_3π/2 - I_π/2) / 2).

    Args:
        intensities_4step: Intensities tensor, shape (N, 4, num_planes, H, W) or (N, 4, H, W).

    Returns:
        Complex interference target Y_target, same trailing shape as one step (e.g. (N, num_planes, H, W) or (N, H, W)).
    """
    I_0 = intensities_4step[:, 0]
    I_pi_2 = intensities_4step[:, 1]
    I_pi = intensities_4step[:, 2]
    I_3pi_2 = intensities_4step[:, 3]
    Y_target = ((I_0 - I_pi) / 2.0) + 1j * ((I_3pi_2 - I_pi_2) / 2.0)
    return Y_target


def optimize_tm_internal_reference_4step(
    tm_system: TMMeasurementSystem,
    base_slm_patterns: torch.Tensor,
    complex_interference_targets: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    learning_rate_phys: float = 0.005,
    learning_rate_aberrations: float = 0.005,
    batch_size: int = 64,
    image_plane_index: int = 0,
    optimize_aberrations: bool = True,
    real_tm: Optional[torch.Tensor] = None,
    test_slm_patterns: Optional[torch.Tensor] = None,
    test_complex_targets: Optional[torch.Tensor] = None,
    raw_intensity_targets: Optional[torch.Tensor] = None,
    test_raw_intensities: Optional[torch.Tensor] = None,
    lambda_intensity: float = 0.0,
    **forward_kwargs
) -> Dict:

    # 1. Enable Gradients
    tm_system._optimized_tm_param.requires_grad_(True)
    tm_system.slm_complex_amplitude.requires_grad_(True)
    tm_system.theta_before_slm.requires_grad_(optimize_aberrations)
    tm_system.theta_after_slm.requires_grad_(optimize_aberrations)

    zernike_params = [tm_system.zernike_before_ft, tm_system.zernike_after_ft,
                      tm_system.zernike_image_plane, tm_system.zernike_fourier_domain]
    for param in zernike_params:
        param.requires_grad_(optimize_aberrations)

    zoomfft_params_list = []
    for param_dict in [tm_system.zoomfft_slm_params, tm_system.zoomfft_out_fwd_params, tm_system.zoomfft_out_inv_params]:
        for param in param_dict.values():
            param.requires_grad_(optimize_aberrations)
            if optimize_aberrations:
                zoomfft_params_list.append(param)

    # 2. Setup Adam Optimizer (E_ref_opt is removed, optimizing TM directly)
    param_groups = [
        {'params': [tm_system._optimized_tm_param], 'lr': learning_rate},
        {'params': [tm_system.slm_complex_amplitude], 'lr': learning_rate_phys}
    ]
    if optimize_aberrations:
        param_groups[1]['params'].extend([tm_system.theta_before_slm, tm_system.theta_after_slm])
        aberration_params = [p for p in zernike_params if p.requires_grad] + zoomfft_params_list
        if aberration_params:
            param_groups.append({'params': aberration_params, 'lr': learning_rate_aberrations})

    if getattr(tm_system, 'enforce_symmetric_tm', False):
        tm_system.diag_in.requires_grad_(True)
        tm_system.diag_out.requires_grad_(True)
        param_groups[0]['params'].extend([tm_system.diag_in, tm_system.diag_out])

    optimizer = optim.Adam(param_groups)
    criterion_cosine = ComplexCosineLoss()
    # Mini-batch S-VARPRO flattens each sample to vectors, so use 2D correlation over feature axis.
    criterion_intensity = PearsonCorrelationLoss(dim=1, center=True)

    num_measurements = base_slm_patterns.shape[0]
    num_batches = (num_measurements + batch_size - 1) // batch_size
    history = {'epoch_losses': [], 'tm_frobenius_error_aligned': [], 'test_losses': []}
    true_tm = real_tm if real_tm is not None else getattr(tm_system, 'real_tm_tensor', None)

    # Dummy pattern used to generate the reference field
    dummy_pattern = torch.zeros(1, *base_slm_patterns.shape[1:], device=base_slm_patterns.device, dtype=base_slm_patterns.dtype)

    print(f"Starting Adam Optimization. Batches/Epoch: {num_batches}")

    # 3. Standard Training Loop
    for epoch in range(num_epochs):
        indices = torch.randperm(num_measurements, device=base_slm_patterns.device)
        total_loss_val = 0.0

        for i in range(num_batches):
            optimizer.zero_grad()

            start = i * batch_size
            end = min(start + batch_size, num_measurements)
            batch_idx = indices[start:end]

            batch_slm = base_slm_patterns[batch_idx].detach()
            batch_target = complex_interference_targets[batch_idx].detach()

            # Forward passes (Calculated directly from the active TM)
            E_mod = tm_system.forward_model_slm(batch_slm, active_polarization='x_only', return_complex_field=True, **forward_kwargs)
            E_ref = tm_system.forward_model_slm(dummy_pattern, active_polarization='y_only', return_complex_field=True, **forward_kwargs)

            E_mod_x, E_mod_y = E_mod[:, image_plane_index, 0], E_mod[:, image_plane_index, 1]
            E_ref_x, E_ref_y = E_ref[:, image_plane_index, 0], E_ref[:, image_plane_index, 1]

            # 1. Complex Cosine Loss (Relative Holography)
            Y_pred = (E_mod_x * torch.conj(E_ref_x)) + (E_mod_y * torch.conj(E_ref_y))
            target_slice = batch_target[:, image_plane_index] if batch_target.dim() == 4 else batch_target

            Y_pred_flat = Y_pred.reshape(Y_pred.shape[0], -1)
            target_flat = target_slice.reshape(target_slice.shape[0], -1)
            loss_data = criterion_cosine(Y_pred_flat, target_flat)

            # 2. Pearson Correlation Loss (Absolute Intensity Anchor)
            loss_intensity = 0.0
            if raw_intensity_targets is not None:
                batch_raw_target = raw_intensity_targets[batch_idx].detach()
                raw_target_slice = batch_raw_target[:, image_plane_index] if batch_raw_target.dim() == 4 else batch_raw_target

                # Calculate intensity directly from the fields (saves a forward pass)
                I_pred = torch.abs(E_mod_x + E_ref_x)**2 + torch.abs(E_mod_y + E_ref_y)**2

                loss_intensity = criterion_intensity(I_pred, raw_target_slice)

            # Total Loss
            loss = loss_data + (lambda_intensity * loss_intensity)

            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()

        avg_loss = total_loss_val / num_batches
        history['epoch_losses'].append(avg_loss)

        # Metrics
        frob_norm_str, test_loss_str = "", ""
        if true_tm is not None:
            with torch.no_grad():
                frob_norm, phi_opt = calculate_phase_aligned_frobenius_norm(true_tm, tm_system.optimized_tm)
            history['tm_frobenius_error_aligned'].append(frob_norm)
            frob_norm_str = f", TM Error={frob_norm:.6f}"

        if test_slm_patterns is not None and test_complex_targets is not None:
            with torch.no_grad():
                E_mod_test = tm_system.forward_model_slm(test_slm_patterns, active_polarization='x_only', return_complex_field=True, **forward_kwargs)
                E_ref_test = tm_system.forward_model_slm(dummy_pattern, active_polarization='y_only', return_complex_field=True, **forward_kwargs)

                E_mod_x_t, E_mod_y_t = E_mod_test[:, image_plane_index, 0], E_mod_test[:, image_plane_index, 1]
                E_ref_x_t, E_ref_y_t = E_ref_test[:, image_plane_index, 0], E_ref_test[:, image_plane_index, 1]

                Y_pred_test = (E_mod_x_t * torch.conj(E_ref_x_t)) + (E_mod_y_t * torch.conj(E_ref_y_t))
                target_test = test_complex_targets[:, image_plane_index] if test_complex_targets.dim() == 4 else test_complex_targets

                Y_pred_flat_t = Y_pred_test.reshape(Y_pred_test.shape[0], -1)
                target_flat_t = target_test.reshape(target_test.shape[0], -1)
                test_loss_data = criterion_cosine(Y_pred_flat_t, target_flat_t).item()

                test_loss_intensity = 0.0
                if test_raw_intensities is not None:
                    raw_test_slice = test_raw_intensities[:, image_plane_index] if test_raw_intensities.dim() == 4 else test_raw_intensities
                    I_pred_test = torch.abs(E_mod_x_t + E_ref_x_t)**2 + torch.abs(E_mod_y_t + E_ref_y_t)**2
                    test_loss_intensity = criterion_intensity(I_pred_test, raw_test_slice).item()

                test_loss = test_loss_data + lambda_intensity * test_loss_intensity

            history['test_losses'].append(test_loss)
            test_loss_str = f", Test Loss={test_loss:.6f}"

        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}{frob_norm_str}{test_loss_str}")

    return history


def optimize_tm_alternating_with_aberrations(
    tm_system: TMMeasurementSystem,
    slm_patterns: torch.Tensor,
    intensity_targets: torch.Tensor,
    num_epochs: int = 20,
    inner_steps: int = 1,
    learning_rate_Y: float = 0.1,
    learning_rate_phys: float = 0.01,
    learning_rate_aberration: Optional[float] = None,
    lambda_consistency: float = 0.1,
    ridge_alpha: float = 1e-4,
    batch_size: int = 512,
    test_slm_patterns: Optional[torch.Tensor] = None,
    test_intensity_targets: Optional[torch.Tensor] = None,
    use_spectral_init: bool = True,
    spectral_init_iters: int = 15,
    spectral_inner_iters: int = 3,
    enable_compile: bool = False
) -> Dict:
    """
    Alternating optimization for TM retrieval with aberration correction.
    
    This method alternates between:
    1. Gradient Descent: Optimize output mode coefficients (Y) and physical parameters
       (Zernike, ZoomFFT aberrations) while keeping TM fixed
    2. Exact Solver: Update TM using closed-form least squares solution given
       current Y and input modes X
    
    The alternating approach decouples the non-convex TM optimization from aberration
    learning, often leading to better convergence than joint optimization.
    
    Args:
        tm_system: TMMeasurementSystem instance with learnable parameters
        slm_patterns: SLM phase patterns, shape (N_samples, H, W)
        intensity_targets: Target intensities, shape (N_samples, 2, H, W)
        num_epochs: Number of alternating epochs (default 20)
        inner_steps: Gradient descent steps per batch (default 1)
        learning_rate_Y: Learning rate for output mode coefficients (default 0.1)
        learning_rate_phys: Learning rate for other physical params (default 0.01).
            Used for theta_before_slm, slm_complex_amplitude, and if symmetric:
            output_tilt, slm_tilt, diag_in, diag_out.
        learning_rate_aberration: Learning rate for ZoomFFT and Zernike params (default: learning_rate_phys)
        lambda_consistency: Weight for ||Y - TX|| consistency loss (default 0.1)
        ridge_alpha: Regularization for TM solver (default 1e-4)
        batch_size: Mini-batch size (default 512)
        test_slm_patterns: Optional test SLM patterns for test loss calculation
        test_intensity_targets: Optional test intensity targets for test loss
        use_spectral_init: If True, run spectral initialization before optimization (default True)
        spectral_init_iters: Number of spectral initialization outer iterations (default 15)
        spectral_inner_iters: Number of inner refinement iterations per spectral step (default 3)
        enable_compile: If True, enable torch.compile for faster execution on CUDA (default True)
        
    Returns:
        Dict containing training history with keys:
            - loss_intensity_train: Intensity matching loss per epoch
            - loss_consistency_train: Y-TX consistency loss per epoch
            - loss_intensity_test: Test intensity loss per epoch (if test data provided)
            - tm_error: Phase-aligned TM error per epoch (if ground truth available)
    """
    # --- Performance Optimization: Enable torch.compile ---
    if enable_compile:
        tm_system.enable_torch_compile(mode='reduce-overhead')
    
    # --- Spectral Initialization (if enabled) ---
    if use_spectral_init:
        print("\n" + "="*60)
        print("PHASE 1: Spectral Initialization")
        print("="*60)
        run_spectral_initialization(
            tm_system,
            slm_patterns,
            intensity_targets,
            num_iters=spectral_init_iters,
            num_inner_iters=spectral_inner_iters,
            ridge_alpha=ridge_alpha
        )
        print("="*60 + "\n")
    
    if learning_rate_aberration is None:
        learning_rate_aberration = learning_rate_phys
    N_samples = slm_patterns.shape[0]
    N_modes = len(tm_system.guided_modes)
    device = tm_system.device
    num_batches = (N_samples + batch_size - 1) // batch_size
    
    print(f"Starting Alternating Optimization with Aberration Correction...")
    print(f"Samples: {N_samples} | Modes: {N_modes} | Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size} | Inner Steps: {inner_steps}")
    print(f"LR_Y: {learning_rate_Y} | LR_aberration: {learning_rate_aberration} | LR_phys: {learning_rate_phys} | Lambda: {lambda_consistency}")
    
    # --- 1. Initialize Output Modes (Y_est) ---
    # Initialize Y = T_init @ X_init using current system state
    with torch.no_grad():
        print("Initializing output mode estimates...")
        X_init_list = []
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, N_samples)
            batch_slm = slm_patterns[start:end]
            batch_X = tm_system.slm_to_input_modes(batch_slm, use_internal_params=True)
            X_init_list.append(batch_X.T)  # Store as (Batch, Modes)
        
        X_full_init = torch.cat(X_init_list, dim=0)  # (Samples, Modes)
        Y_init = X_full_init @ tm_system.optimized_tm.T  # (Samples, Modes)
    
    # Make Y_est a learnable parameter
    Y_est = nn.Parameter(Y_init.detach().clone())
    
    # --- 2. Configure Parameters ---
    # Lock TM during gradient descent phase
    tm_system._optimized_tm_param.requires_grad_(False)
    
    # Enable all other parameters (aberrations, SLM amplitude, etc.)
    for param in tm_system.parameters():
        if param is not tm_system._optimized_tm_param:
            param.requires_grad_(True)
    
    # Collect physical parameters by group for separate learning rates
    zoomfft_param_names = {'zoomfft_slm_params', 'zoomfft_out_fwd_params', 'zoomfft_out_inv_params'}
    zernike_param_names = {'zernike_before_ft', 'zernike_after_ft', 'zernike_image_plane', 'zernike_fourier_domain'}
    zoomfft_params = []
    zernike_params = []
    other_phys_params = []
    for n, p in tm_system.named_parameters():
        if '_optimized_tm_param' in n or not p.requires_grad:
            continue
        # ParameterDict names are like "zoomfft_slm_params.s_a_x"
        base_name = n.split('.')[0]
        if base_name in zoomfft_param_names:
            zoomfft_params.append(p)
        elif base_name in zernike_param_names:
            zernike_params.append(p)
        else:
            other_phys_params.append(p)
    
    # --- 3. Setup Optimizer and Loss Functions ---
    param_groups = [{'params': [Y_est], 'lr': learning_rate_Y}]
    aberration_params = zoomfft_params + zernike_params
    if aberration_params:
        param_groups.append({'params': aberration_params, 'lr': learning_rate_aberration})
    if other_phys_params:
        param_groups.append({'params': other_phys_params, 'lr': learning_rate_phys})
    optimizer = optim.Adam(param_groups)
    
    criterion_intensity = PearsonCorrelationLoss(dim=(0, 2, 3), center=True)
    criterion_consistency = ComplexCosineLoss()
    
    history = {
        'loss_intensity_train': [],
        'loss_consistency_train': [],
        'loss_intensity_test': [],
        'tm_error': []
    }
    
    # --- 4. Main Alternating Loop ---
    for epoch in range(num_epochs):
        total_loss_I = 0.0
        total_loss_consist = 0.0
        indices = torch.randperm(N_samples)
        
        # === PHASE 1: Gradient Descent (Optimize Y and Physical Parameters) ===
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, N_samples)
            batch_idx = indices[start:end]
            
            # Prepare batch data (fixed per batch). Detach to avoid double-backward errors.
            batch_slm = slm_patterns[batch_idx].detach().contiguous()
            batch_target = intensity_targets[batch_idx].detach().contiguous()
            
            # Inner gradient steps
            for _ in range(inner_steps):
                optimizer.zero_grad()
                # Re-read Y_est slice each step so we don't reuse a tensor from a freed graph
                batch_Y_est = Y_est[batch_idx]
                
                # A. Calculate input modes X (gradients flow to input aberrations)
                batch_X = tm_system.slm_to_input_modes(batch_slm, use_internal_params=True)
                batch_X = batch_X.T  # (Batch, Modes)
                
                # B. Intensity loss: Match Y_est to measured intensities
                # output_modes_to_intensities expects (N_modes, Batch)
                pred_intensity = tm_system.output_modes_to_intensities(
                    batch_Y_est.T, 
                    use_internal_params=True
                )
                loss_I = criterion_intensity(pred_intensity, batch_target)
                
                # C. Consistency loss: Enforce Y ≈ T @ X
                # T is fixed (no_grad), X has gradients, Y has gradients
                with torch.no_grad():
                    current_T = tm_system.optimized_tm
                
                Y_from_physics = batch_X @ current_T.T  # (Batch, Modes)
                loss_consist = criterion_consistency(batch_Y_est, Y_from_physics)
                
                # Total loss
                loss = loss_I + lambda_consistency * loss_consist
                
                loss.backward()
                optimizer.step()
            
            total_loss_I += loss_I.item() / num_batches
            total_loss_consist += loss_consist.item() / num_batches
        
        # === PHASE 2: Exact Least Squares (Update TM) ===
        with torch.no_grad():
            # Recompute full X matrix with updated input aberrations
            X_new_list = []
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, N_samples)
                batch_slm = slm_patterns[start:end]
                batch_x = tm_system.slm_to_input_modes(batch_slm, use_internal_params=True)
                X_new_list.append(batch_x)  # (Modes, Batch)
            
            X_full_new = torch.cat(X_new_list, dim=1)  # (Modes, Samples)
            Y_phys = tm_system.optimized_tm @ X_full_new
            Y_full_new = Y_est.T  # (Modes, Samples)
            dot_prod = torch.sum(Y_full_new.conj() * Y_phys, dim=0)
            phase_factor = dot_prod/(torch.abs(dot_prod) + 1e-8)
            real_energy = torch.sum(intensity_targets, dim=(1,2,3)) # Shape (N_samples,)
            current_energy = torch.norm(Y_full_new, dim=0)**2
            scale_factor = torch.sqrt(real_energy/(current_energy + 1e-8))
            Y_full_aligned = Y_full_new * phase_factor.unsqueeze(0) * scale_factor.unsqueeze(0)
            # Solve T = Y @ X^dagger with ridge regularization
            new_T = solve_fast_cholesky(Y_full_aligned, X_full_new, alpha=ridge_alpha)
            
            # Update system TM
            tm_system.optimized_tm_tensor = new_T
        
        # === Calculate Test Loss (if test data provided) ===
        test_loss = 0.0
        if test_slm_patterns is not None and test_intensity_targets is not None:
            with torch.no_grad():
                # Calculate predictions on test set using current system state
                test_pred = tm_system.forward_model_slm(test_slm_patterns)
                test_loss = criterion_intensity(test_pred, test_intensity_targets).item()
        
        # === Calculate TM Error (if ground truth available) ===
        tm_err = 0.0
        if hasattr(tm_system, 'real_tm_tensor'):
            with torch.no_grad():
                tm_err, _ = calculate_phase_aligned_frobenius_norm(
                    tm_system.real_tm_tensor, tm_system.optimized_tm
                )
        
        # Record history
        history['loss_intensity_train'].append(total_loss_I)
        history['loss_consistency_train'].append(total_loss_consist)
        history['loss_intensity_test'].append(test_loss)
        history['tm_error'].append(tm_err)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss_I={total_loss_I:.5f}, Loss_consist={total_loss_consist:.5f}, "
              f"Test_Loss={test_loss:.5f}, TM_Error={tm_err:.5f}")
    
    print("Alternating Optimization Completed.")
    return history



def optimize_tm_lifted_als_4step(
    tm_system: TMMeasurementSystem,
    base_slm_patterns: torch.Tensor,
    complex_interference_targets: torch.Tensor,
    raw_intensity_targets: torch.Tensor,
    reference_intensity_target: torch.Tensor,  # The isolated y-polarization measurement
    num_epochs: int = 100,
    burn_in_epochs: int = 80,
    inner_als_steps: int = 3,
    inner_phys_steps: int = 1,
    learning_rate_phys: float = 0.001,
    ridge_alpha: float = 1e-4,
    batch_size: int = 256,
    test_slm_patterns: Optional[torch.Tensor] = None,
    test_complex_targets: Optional[torch.Tensor] = None,
    test_raw_intensities: Optional[torch.Tensor] = None,
    lambda_intensity: float = 0.5,
    **forward_kwargs) -> Dict:
    
    device = base_slm_patterns.device
    N_samples = base_slm_patterns.shape[0]
    N_modes = len(tm_system.guided_modes)
    num_batches = (N_samples + batch_size - 1) // batch_size
    
    clean_forward_kwargs = dict(forward_kwargs)
    clean_forward_kwargs.pop("use_internal_params", None)
    
    print("="*60)
    print(f"Starting Split-Variable ALS with Exact 1D Reference Amplitude Anchor...")
    print("="*60)

    tm_system._optimized_tm_param.requires_grad_(False)
    phys_params = [p for n, p in tm_system.named_parameters() if p.requires_grad and '_optimized_tm_param' not in n]

    criterion_cosine = ComplexCosineLoss()
    criterion_intensity = PearsonCorrelationLoss(dim=(1, 2), center=True)

    # Prepare dual-plane targets: W[p] has shape (H*W, N_samples)
    if complex_interference_targets.dim() == 3:
        complex_interference_targets = complex_interference_targets.unsqueeze(0)
    if raw_intensity_targets.dim() == 3:
        raw_intensity_targets = raw_intensity_targets.unsqueeze(0)
    if reference_intensity_target.dim() == 4:
        reference_intensity_target = reference_intensity_target.squeeze(0)

    W = [complex_interference_targets[:, p].reshape(N_samples, -1).T for p in range(2)]
    W_ref_base = [reference_intensity_target[p].reshape(-1) for p in range(2)]
    
    history = {'epoch_losses': [], 'tm_error': []}
    T = tm_system.optimized_tm_tensor.clone().detach()

    optimizer = optim.Adam(phys_params, lr=learning_rate_phys)
    dummy_pattern = torch.zeros(1, *base_slm_patterns.shape[1:], device=device, dtype=base_slm_patterns.dtype)

    # --- Scheduled ALS Loop ---
    for epoch in range(num_epochs):
        
        # ==========================================
        # PHASE 0: Dynamic Target Scaling & Basis Recomputation
        # ==========================================
        with torch.no_grad():
            # Dynamically scale the isolated I_pi/2 measurement to match holography power split
            sin_theta = torch.sin(tm_system.theta_before_slm.detach())
            W_ref_scaled = W_ref_base  # * (sin_theta ** 2)

            Z_ref = tm_system.slm_to_input_modes(dummy_pattern, active_polarization='y_only', use_internal_params=True, **clean_forward_kwargs)
            
            Z_slm_list = []
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, N_samples)
                batch_Z = tm_system.slm_to_input_modes(base_slm_patterns[start:end], active_polarization='x_only', use_internal_params=True, **clean_forward_kwargs)
                Z_slm_list.append(batch_Z)
            Z_slm = torch.cat(Z_slm_list, dim=1)

            I_modes = torch.eye(N_modes, dtype=torch.complex64, device=device)
            P_out_complex = tm_system.output_modes_to_intensities(I_modes, return_complex_field=True, use_internal_params=True)
            P_out_x = [P_out_complex[:, p, 0].reshape(N_modes, -1).T for p in range(2)]
            P_out_y = [P_out_complex[:, p, 1].reshape(N_modes, -1).T for p in range(2)]
            P_out_x_H = [torch.conj(P_out_x[p].T) for p in range(2)]
            P_out_y_H = [torch.conj(P_out_y[p].T) for p in range(2)]
            gamma = torch.mean(torch.abs(W[0]) ** 2) / (torch.mean(torch.abs(W[1]) ** 2) + 1e-12)

            Y_sig = T @ Z_slm
            Y_ref = T @ Z_ref

        # ==========================================
        # PHASE A: Multi-Step Pure LS Core
        # ==========================================
        for ias in range(inner_als_steps):
            with torch.no_grad():
                # 1. Y_sig Update
                A_H_A_total = torch.zeros((N_modes, N_modes), dtype=torch.complex64, device=device)
                A_H_W_total = torch.zeros((N_modes, N_samples), dtype=torch.complex64, device=device)
                for p in range(2):
                    plane_weight = 1.0 if p == 0 else gamma
                    E_ref_x_p = P_out_x[p] @ Y_ref
                    E_ref_y_p = P_out_y[p] @ Y_ref
                    A_ref_p = torch.conj(E_ref_x_p) * P_out_x[p] + torch.conj(E_ref_y_p) * P_out_y[p]
                    A_H_A_total = A_H_A_total + plane_weight * (A_ref_p.T.conj() @ A_ref_p)
                    A_H_W_total = A_H_W_total + plane_weight * (A_ref_p.T.conj() @ W[p])
                reg_A = ridge_alpha * torch.max(torch.abs(torch.diag(A_H_A_total))) * torch.eye(N_modes, dtype=torch.complex64, device=device)
                Y_sig = torch.linalg.solve(A_H_A_total + reg_A, A_H_W_total)

                # 2. Y_ref Update (Constrained by ALL holography images)
                M_sig_total = torch.zeros((N_modes, N_modes), dtype=torch.complex64, device=device)
                V_sig_total = torch.zeros((N_modes, 1), dtype=torch.complex64, device=device)
                for p in range(2):
                    plane_weight = 1.0 if p == 0 else gamma
                    E_sig_x_p = P_out_x[p] @ Y_sig
                    E_sig_y_p = P_out_y[p] @ Y_sig
                    I_sig_x_p = torch.sum(torch.abs(E_sig_x_p) ** 2, dim=1)
                    I_sig_y_p = torch.sum(torch.abs(E_sig_y_p) ** 2, dim=1)
                    I_sig_xy_p = torch.sum(E_sig_x_p * torch.conj(E_sig_y_p), dim=1)
                    M_sig_p = P_out_x_H[p] @ (I_sig_x_p.unsqueeze(1) * P_out_x[p]) + \
                              P_out_y_H[p] @ (I_sig_y_p.unsqueeze(1) * P_out_y[p]) + \
                              P_out_x_H[p] @ (I_sig_xy_p.unsqueeze(1) * P_out_y[p]) + \
                              P_out_y_H[p] @ (torch.conj(I_sig_xy_p).unsqueeze(1) * P_out_x[p])
                    S_x_p = torch.sum(E_sig_x_p * torch.conj(W[p]), dim=1)
                    S_y_p = torch.sum(E_sig_y_p * torch.conj(W[p]), dim=1)
                    V_sig_p = P_out_x_H[p] @ S_x_p.unsqueeze(1) + P_out_y_H[p] @ S_y_p.unsqueeze(1)
                    M_sig_total = M_sig_total + plane_weight * M_sig_p
                    V_sig_total = V_sig_total + plane_weight * V_sig_p

                reg_M = ridge_alpha * torch.max(torch.abs(torch.diag(M_sig_total))) * torch.eye(N_modes, dtype=torch.complex64, device=device)
                Y_ref = torch.linalg.solve(M_sig_total + reg_M, V_sig_total)
                
                # 3. Global Phase Anchor
                # phase_align = torch.sum(torch.conj(Y_ref) * Z_ref)
                # phase_factor = phase_align / (torch.abs(phase_align) + 1e-12)
                # Y_ref = Y_ref * phase_factor

                # 4. Exact 1D Amplitude Anchor using the new I_pi/2 measurement
                I_ref_base = []
                for p in range(2):
                    E_ref_x_base_p = P_out_x[p] @ Y_ref
                    E_ref_y_base_p = P_out_y[p] @ Y_ref
                    I_ref_base.append((torch.abs(E_ref_x_base_p) ** 2 + torch.abs(E_ref_y_base_p) ** 2).reshape(-1))

                # Closed-form minimization of weighted dual-plane error
                numerator = 0.0
                denominator = 1e-12
                for p in range(2):
                    plane_weight = 1.0 if p == 0 else gamma
                    numerator = numerator + plane_weight * torch.sum(I_ref_base[p] * W_ref_scaled[p])
                    denominator = denominator + plane_weight * torch.sum(I_ref_base[p] ** 2)
                g_sq = numerator / denominator
                g_val = torch.sqrt(torch.clamp(g_sq, min=1e-8, max=1e8))
                
                # Apply the exact amplitude split
                g_final = g_val.to(torch.complex64)
                Y_sig = Y_sig / g_final
                Y_ref = Y_ref * g_final

                # 5. TM Update
                weight = (N_samples/3) ** 0.5
                Z_comb = torch.cat([Z_slm, Z_ref * weight], dim=1)
                Y_comb = torch.cat([Y_sig, Y_ref * weight], dim=1)
                
                Z_H_Z = Z_comb @ Z_comb.T.conj()
                reg_T = ridge_alpha * torch.max(torch.abs(torch.diag(Z_H_Z))) * torch.eye(N_modes, dtype=torch.complex64, device=device)
                T = torch.linalg.solve(Z_H_Z + reg_T, Z_comb @ Y_comb.T.conj()).T.conj()
                
                Y_sig = T @ Z_slm
                Y_ref = T @ Z_ref
        
        with torch.no_grad():
            tm_system.optimized_tm_tensor = T
            weighted_complex_losses = []
            for p in range(2):
                plane_weight = 1.0 if p == 0 else gamma
                E_mod_x_p, E_mod_y_p = P_out_x[p] @ Y_sig, P_out_y[p] @ Y_sig
                E_ref_x_eval_p, E_ref_y_eval_p = P_out_x[p] @ Y_ref, P_out_y[p] @ Y_ref
                Y_pred_p = (E_mod_x_p * torch.conj(E_ref_x_eval_p)) + (E_mod_y_p * torch.conj(E_ref_y_eval_p))
                loss_complex_p = 1.0 - torch.mean(
                    torch.abs(torch.sum(Y_pred_p.T * torch.conj(W[p].T), dim=1)) /
                    (torch.norm(Y_pred_p.T, dim=1) * torch.norm(W[p].T, dim=1) + 1e-8)
                )
                weighted_complex_losses.append(plane_weight * loss_complex_p)
            loss_complex_ls = weighted_complex_losses[0] + weighted_complex_losses[1]

        # ==========================================
        # PHASE 1.5: Calculate Optimal Input Targets (Stable Cholesky Solve)
        # ==========================================
        with torch.no_grad():
            T_H_T = T.T.conj() @ T
            reg_Z = ridge_alpha * torch.max(torch.abs(torch.diag(T_H_T))) * torch.eye(N_modes, dtype=torch.complex64, device=device)
            L_Z = torch.linalg.cholesky(T_H_T + reg_Z)
            Z_target_sig = torch.cholesky_solve(T.T.conj() @ Y_sig, L_Z)
            Z_target_ref = torch.cholesky_solve(T.T.conj() @ Y_ref, L_Z)

        # ==========================================
        # PHASE B: Decoupled Aberration Anchor (Full-Batch Gradient Accumulation)
        # ==========================================
        total_phys_loss = 0.0
        
        if epoch < burn_in_epochs:
            status_msg = "[Burn-In: LS Only]"
        else:
            status_msg = "[Split-Variable Fine-Tuning]"
            for _ in range(inner_phys_steps):
                optimizer.zero_grad(set_to_none=True)
                epoch_loss_accum = 0.0

                for i in range(num_batches):
                    start = i * batch_size
                    end = min(start + batch_size, N_samples)
                    batch_len = end - start

                    batch_slm = base_slm_patterns[start:end].detach()
                    batch_target_raw = raw_intensity_targets[start:end].detach()
                    batch_target_complex = complex_interference_targets[start:end].detach()

                    # 1. INPUT NETWORK (SLM -> Z_pred). Target is Z_target.
                    Z_pred_sig = tm_system.slm_to_input_modes(batch_slm, active_polarization='x_only', use_internal_params=True, **clean_forward_kwargs)
                    Z_pred_ref = tm_system.slm_to_input_modes(dummy_pattern, active_polarization='y_only', use_internal_params=True, **clean_forward_kwargs)
                    
                    loss_input_sig = criterion_cosine(Z_pred_sig.T, Z_target_sig[:, start:end].T)
                    loss_input_ref = criterion_cosine(Z_pred_ref.T, Z_target_ref.T)
                    loss_input = loss_input_sig + loss_input_ref

                    # 2. OUTPUT NETWORK (Y_fixed -> Camera). Target is Lab Data.
                    E_mod = tm_system.output_modes_to_intensities(Y_sig[:, start:end].detach(), return_complex_field=True, use_internal_params=True)
                    E_ref = tm_system.output_modes_to_intensities(Y_ref.detach(), return_complex_field=True, use_internal_params=True)
                    E_mod_x, E_mod_y = E_mod[:, :, 0], E_mod[:, :, 1]
                    E_ref_x, E_ref_y = E_ref[:, :, 0], E_ref[:, :, 1]

                    sin_theta_opt = torch.sin(tm_system.theta_before_slm)
                    W_ref_opt_scaled = [W_ref_base[p] * (sin_theta_opt ** 2) for p in range(2)]
                    plane_losses = []
                    for p in range(2):
                        I_pred_p = torch.abs(E_mod_x[:, p] + E_ref_x[:, p])**2 + torch.abs(E_mod_y[:, p] + E_ref_y[:, p])**2
                        loss_intensity_p = criterion_intensity(I_pred_p, batch_target_raw[:, p])

                        I_pred_ref_p = torch.abs(E_ref_x[:, p])**2 + torch.abs(E_ref_y[:, p])**2
                        loss_intensity_ref_p = criterion_intensity(I_pred_ref_p, W_ref_opt_scaled[p].unsqueeze(0))

                        Y_pred_batch_p = (E_mod_x[:, p] * torch.conj(E_ref_x[:, p])) + (E_mod_y[:, p] * torch.conj(E_ref_y[:, p]))
                        Y_pred_flat_p = Y_pred_batch_p.reshape(Y_pred_batch_p.shape[0], -1)
                        target_flat_complex_p = batch_target_complex[:, p].reshape(batch_target_complex.shape[0], -1)
                        loss_hologram_p = criterion_cosine(Y_pred_flat_p, target_flat_complex_p)

                        plane_losses.append(loss_hologram_p + lambda_intensity * (loss_intensity_p + loss_intensity_ref_p))

                    loss_output = plane_losses[0] + gamma * plane_losses[1]
                    
                    # 3. COMBINED DECOUPLED LOSS
                    loss_total = loss_input + loss_output
                    scaled_loss = loss_total * (batch_len / N_samples)
                    scaled_loss.backward()

                    epoch_loss_accum += scaled_loss.item()

                torch.nn.utils.clip_grad_norm_(phys_params, max_norm=1.0)
                optimizer.step()
                total_phys_loss = epoch_loss_accum

        tm_err = 0.0
        if hasattr(tm_system, 'real_tm_tensor'):
            tm_err, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, tm_system.optimized_tm)
            
        history['epoch_losses'].append(loss_complex_ls.item())
        history['tm_error'].append(tm_err)
        
        adam_loss_str = f"{total_phys_loss:.5f}" if epoch >= burn_in_epochs else "N/A"
        print(f"Epoch {epoch+1}/{num_epochs} {status_msg}: LS Data Loss={loss_complex_ls.item():.5f}, Adam Loss={adam_loss_str}, TM Error={tm_err:.5f}")

    return history


def main(optimization_mode: str = "4step_internal"):
    """
    Example usage of the TM retrieval system.
    """
    wavelength = 632.8e-9
    core_radius = 25e-6
    NA = 0.22
    cladding_index = 1.457
    core_index = np.sqrt(cladding_index**2 + NA**2)
    margin = 1.1
    x_range = core_radius*margin
    N_0 = 128
    dx_i = 2*x_range/(N_0-1)
    y_range = x_range
    f = 200e-3 # focal length of the lens
    x_max = f*NA
    dx_o = 2*x_max*margin/(N_0-1)
    N_pad = round(wavelength*f/(dx_i*dx_o))
    mult_factor_slm = dx_o*dx_i/(wavelength*f)
    mult_factor_out = dx_i*dx_o/(wavelength*f)
    
    # Define fiber parameters
    fiber_params = {
        'a': core_radius,  # 25 micron core radius
        'n1': core_index,  # Core refractive index
        'n0': cladding_index,  # Cladding refractive index
        'wavelength': wavelength  # 632.8 nm wavelength
    }
    
    # Define grid parameters
    grid_params = {
        'x_range': (-x_range, x_range),  # ±27 microns
        'y_range': (-y_range, y_range),  # ±27 microns
        'nx': N_0,
        'ny': N_0
    }
    
    # Define ZoomFFT parameters (for SLM mode)
    zoomfft_params = {}
    for side in ['slm', 'tran']:
        zoomfft_params[side] = {
            'a_init': 2*np.pi/N_pad,
            'n0_init': (N_0 - 1) / 2,
            'm0_init': (N_0 - 1) / 2,
            'k0_init': (N_0 - 1) / 2,
            'l0_init': (N_0 - 1) / 2
        }
    mult_factor = {
        'slm': mult_factor_slm,
        'tran': mult_factor_out
    }
    dx = {
        'im': dx_i,
        'four': dx_o,
        'slm': dx_o
    }
    # Initialize the system
    print("Initializing TM Measurement System...")
    tm_system = TMMeasurementSystem(
        fiber_params, grid_params, 
        device='cuda', 
        mult_factor=mult_factor,
        zoomfft_params=zoomfft_params,
        zernike_n_modes=8,
        dx=dx,
        enforce_symmetric_tm=False,
        theta_before_slm_init=np.pi/8
    )

    # ===== 4-step internal reference TM retrieval simulation =====
    optimization_mode = "4step_lifted_als"
    if optimization_mode in ("4step_internal", "4step_lifted_als"):
        print("\n" + "="*60)
        print("4-step internal reference TM retrieval simulation")
        print("="*60)
        num_4step_train = 3000
        num_4step_test = 200
        base_slm_patterns_4step = torch.rand(
            num_4step_train + num_4step_test,
            grid_params['ny'], grid_params['nx'],
            device=tm_system.device
        ) * 2 * np.pi
        # Simulate 4-step measurements using the real TM with same aberrations/amplitude as optimizer.
        # no_grad: we only need target values, not a computation graph (saves large GPU memory).
        # Chunk when enforce_symmetric_tm to avoid OOM (symmetric path uses 2x field tensors).
        max_batch_4step = 512 if getattr(tm_system, 'enforce_symmetric_tm', False) else None
        with torch.no_grad():
            intensities_4step = simulate_phase_stepped_measurements(
                tm_system,
                base_slm_patterns_4step,
                max_batch_size=max_batch_4step,
                tm=tm_system.real_tm_tensor,
                use_internal_params=True,
                apply_slm_amplitude=True
            )
            complex_targets_4step = extract_complex_interference_4step(intensities_4step)
        train_slm_4step = base_slm_patterns_4step[:num_4step_train]
        train_targets_4step = complex_targets_4step[:num_4step_train]
        test_slm_4step = base_slm_patterns_4step[num_4step_train:num_4step_train + num_4step_test]
        test_targets_4step = complex_targets_4step[num_4step_train:num_4step_train + num_4step_test]
        train_raw_intensities_4step = intensities_4step[:num_4step_train, 0]
        test_raw_intensities_4step = intensities_4step[num_4step_train:num_4step_train + num_4step_test, 0]
        with torch.no_grad():
            reference_intensity_4step = tm_system.forward_model_slm(
                torch.zeros(
                    1,
                    grid_params['ny'],
                    grid_params['nx'],
                    device=tm_system.device
                ),
                active_polarization='y_only',
                tm=tm_system.real_tm_tensor,
                use_internal_params=True,
                apply_slm_amplitude=True
            ).squeeze(0)
        print(f"Train patterns: {train_slm_4step.shape[0]}, test patterns: {test_slm_4step.shape[0]}")
        if optimization_mode == "4step_internal":
            history_4step = optimize_tm_internal_reference_4step(
                tm_system=tm_system,
                base_slm_patterns=train_slm_4step,
                complex_interference_targets=train_targets_4step,
                num_epochs=100,
                learning_rate=1e-2,
                learning_rate_phys=1e-2,
                learning_rate_aberrations=1e-3,
                batch_size=128,
                image_plane_index=0,
                optimize_aberrations=True,
                real_tm=tm_system.real_tm_tensor,
                test_slm_patterns=test_slm_4step,
                test_complex_targets=test_targets_4step,
                raw_intensity_targets=train_raw_intensities_4step,
                test_raw_intensities=test_raw_intensities_4step,
                lambda_intensity=0.5,
                use_internal_params=True,
                apply_slm_amplitude=True
            )
        elif optimization_mode == "4step_lifted_als":
            history_4step = optimize_tm_lifted_als_4step(
                tm_system=tm_system,
                base_slm_patterns=train_slm_4step,
                complex_interference_targets=train_targets_4step,
                raw_intensity_targets=train_raw_intensities_4step,
                reference_intensity_target=reference_intensity_4step,
                num_epochs=300,
                learning_rate_phys=0.001,
                ridge_alpha=1e-4,
                batch_size=256,
                test_slm_patterns=test_slm_4step,
                test_complex_targets=test_targets_4step,
                test_raw_intensities=test_raw_intensities_4step,
                use_internal_params=True,
                apply_slm_amplitude=True
            )
        else:
            raise ValueError(f"Unknown 4-step optimization mode: {optimization_mode}")
        print("4-step simulation done.")
        print("="*60 + "\n")

    # ===== SLM Pattern Optimization =====
    elif optimization_mode == "slm":
        # Set these flags to test different configurations:
        use_aberrations_in_target = True  # If True, targets have aberrations; if False, no aberrations
        optimize_aberrations = True       # If True, optimize TM+aberrations; if False, optimize TM only
        
        print("\n" + "="*60)
        print(f"SLM Pattern Optimization Mode")
        print(f"  Target aberrations: {use_aberrations_in_target}")
        print(f"  Optimize aberrations: {optimize_aberrations}")
        print("="*60)
        
        # Sample SLM measurements
        print("\nSampling SLM measurements...")
        slm_patterns, intensity_measurements = tm_system.sample_measurements_slm(
            num_measurements=4000, use_aberrations=use_aberrations_in_target)
        
        # Optimize TM (and optionally aberrations)
        print("\nOptimizing transmission matrix...")
        import time
        start_time = time.time()
        history = optimize_tm_slm(
            tm_system=tm_system,
            slm_patterns=slm_patterns,
            intensity_targets=intensity_measurements,
            num_epochs=100,
            learning_rate=0.001,
            batch_size=256,
            planes='both',
            optimize_aberrations=optimize_aberrations
        )
        end_time = time.time()
        print(f"Optimization time: {end_time - start_time:.2f} seconds")
    else:
        raise ValueError(f"Unknown optimization_mode: {optimization_mode}")
    
    
    # Evaluate results
    print("\nEvaluating reconstruction quality...")
    evaluation_results = tm_system.evaluate_reconstruction()
    
    # Print results
    print(f"\nReconstruction Results:")
    print(f"TM Frobenius Error (Phase-Aligned, Normalized): {evaluation_results['tm_frobenius_error_aligned_normalized']:.6f}")
    print(f"Optimal Phase: {evaluation_results['optimal_phase_degrees']:.2f}° ({evaluation_results['optimal_phase_radians']:.4f} rad)")
    print(f"Intensity Mean Error (Normalized): {evaluation_results['intensity_mean_error_normalized']:.6f}")
    
    # Plot training history
    print("\nPlotting training history...")
    tm_system.plot_training_history(history)
    
    # Plot evaluation results
    print("\nPlotting evaluation results...")
    tm_system.plot_results(evaluation_results)


if __name__ == "__main__":
    main()
