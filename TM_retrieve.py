"""
Transmission Matrix Retrieval System

This module implements a complete system for retrieving transmission matrices (TMs)
from intensity measurements using optimization methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from mode_solver import find_guided_modes
from tm_sampler import sample_tm_perturbed_mmf
from utils import (
    fourier_plane,
    load_or_calculate_mode_fields,
    clear_mode_fields_cache,
    mode_fields_cache_exists,
    sample_mode_distribution as utils_sample_mode_distribution,
    plot_results as utils_plot_results,
    plot_training_intensity_example as utils_plot_training_intensity_example,
    plot_training_history as utils_plot_training_history,
    ZoomFFT2D_Optimizable,
    ZernikeAberrationLayer
)
from metrics_losses import (
    calculate_phase_aligned_frobenius_norm,
    calculate_phase_aligned_tm_error,
    evaluate_reconstruction as metrics_evaluate_reconstruction
)


class TMMeasurementSystem:
    """
    A class that handles the physical environment for transmission matrix measurements.
    
    This class manages:
    - Guided mode calculations and storage
    - Real and optimized TM sampling
    - Intensity calculations (I = |E_x|^2 + |E_y|^2)
    - Mode distribution sampling
    - Field calculations
    """
    
    def __init__(self, fiber_params: Dict, grid_params: Dict, device: str = None, N_pad: int = None,
                 mult_factor_slm: float = None, mult_factor_out: float = None,
                 zoomfft_params: Optional[Dict] = None, zernike_n_modes: int = 12, use_slm: bool = False):
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
            N_pad: Padding size for FFT
            mult_factor_slm: Multiplication factor for SLM→Fourier transform
            mult_factor_out: Multiplication factor for output FFT (multiply FFT, divide IFFT)
            zoomfft_params: Dictionary with ZoomFFT base parameters:
                - 'a_init': Base parameter a
                - 'n0_init': Base parameter n0
                - 'm0_init': Base parameter m0
                - 'k0_init': Base parameter k0
                - 'l0_init': Base parameter l0
                If None, will use default values calculated from system parameters
            zernike_n_modes: Number of Zernike modes to use (default 12)
            use_slm: If True, system is configured for SLM-based optimization (only image plane mode fields stored)
        """
        self.fiber_params = fiber_params
        self.grid_params = grid_params
        self.use_slm = use_slm
        
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
        self.N_pad = N_pad
        self.mult_factor_slm = mult_factor_slm if mult_factor_slm is not None else 1.0
        self.mult_factor_out = mult_factor_out if mult_factor_out is not None else 1.0
        
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
            self.N_pad,
            self.mult_factor_out,
            image_plane_only=self.use_slm
        )
        
        # Sample real TM and initialization TM
        N_modes = len(self.guided_modes)
        print(f"Sampling transmission matrices for {N_modes} modes...")
        
        # Real TM (the one we want to find)
        self.real_tm = sample_tm_perturbed_mmf(
            N_modes=N_modes,
            loss_parameter=1.0,  # 100% transmission efficiency
            time_reversal_symmetry_broken=False,
            noise_level=0.0
        )
        
        # Initialization TM (starting point for optimization)
        self.optimized_tm = sample_tm_perturbed_mmf(
            N_modes=N_modes,
            loss_parameter=1.0,  # Slightly different for initialization
            time_reversal_symmetry_broken=False,
            noise_level=0.0
        )
        
        # Convert to torch tensors
        self.real_tm_tensor = torch.tensor(self.real_tm, dtype=torch.complex64, device=device)
        self.optimized_tm_tensor = torch.tensor(self.optimized_tm, dtype=torch.complex64, device=device)
        
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
        self.zoomfft = ZoomFFT2D_Optimizable(
            N=H, M=W,
            a_init=zoomfft_params['a_init'],
            n0_init=zoomfft_params['n0_init'],
            m0_init=zoomfft_params['m0_init'],
            k0_init=zoomfft_params['k0_init'],
            l0_init=zoomfft_params['l0_init']
        ).to(self.device)
        
        # Create single Zernike aberration layer (stores only the basis, not parameters)
        # We use the same layer with different coefficients for each aberration location
        self.zernike_layer = ZernikeAberrationLayer(H, W, n_modes=zernike_n_modes).to(self.device)
        
        self.zernike_n_modes = zernike_n_modes
        
        print(f"System initialized with {N_modes} guided modes")
        print(f"Grid size: {grid_params['nx']} x {grid_params['ny']}")
        print(f"Aberration models initialized: ZoomFFT and {zernike_n_modes} Zernike modes")
    
    def clear_mode_fields_cache(self):
        """
        Clear the cached mode_fields file if it exists.
        """
        clear_mode_fields_cache(self.fiber_params, self.grid_params)
    
    def mode_fields_cache_exists(self) -> bool:
        """
        Check if the mode_fields cache file exists.
        
        Returns:
            bool: True if cache exists, False otherwise
        """
        return mode_fields_cache_exists(self.fiber_params, self.grid_params)
    
    def project_field_to_modes(self, E_y_field: torch.Tensor) -> torch.Tensor:
        """
        Project a y-polarized field onto image plane mode fields to get mode distribution.
        
        Args:
            E_y_field: Field in image plane, shape (H, W) or (batch, H, W)
            
        Returns:
            torch.Tensor: Mode amplitudes, shape (N_modes,) or (N_modes, batch)
        """
        # Get image plane y-component mode fields: (H, W, N_modes)
        # mode_fields shape: (2, 2, H, W, N_modes) where [0, 1] = image plane, y-component
        mode_fields_y = self.mode_fields[0, 1, :, :, :]  # (H, W, N_modes)
        N_modes = mode_fields_y.shape[2]
        
        # Handle batch dimension
        if E_y_field.dim() == 2:
            # Single field: (H, W)
            E_y_flat = E_y_field.flatten()  # (H*W,)
            mode_fields_flat = mode_fields_y.reshape(-1, N_modes)  # (H*W, N_modes)
            # Project: sum(E_y * conj(mode_field)) over spatial grid
            mode_amplitudes = torch.sum(E_y_flat.unsqueeze(1) * torch.conj(mode_fields_flat), dim=0)  # (N_modes,)
            return mode_amplitudes
        else:
            # Batch: (batch, H, W)
            batch_size = E_y_field.shape[0]
            E_y_flat = E_y_field.reshape(batch_size, -1)  # (batch, H*W)
            mode_fields_flat = mode_fields_y.reshape(-1, N_modes)  # (H*W, N_modes)
            # Project: sum(E_y * conj(mode_field)) over spatial grid for each batch
            mode_amplitudes = torch.matmul(E_y_flat, torch.conj(mode_fields_flat))  # (batch, N_modes)
            return mode_amplitudes.T  # (N_modes, batch)
    
    def calculate_fields_from_mode_dist(self, mode_distribution: torch.Tensor, planes: str = 'both') -> torch.Tensor:
        """
        Calculate complex electric fields (E_x, E_y) directly from mode distribution (no TM applied).
        
        Args:
            mode_distribution: Tensor of shape (N_modes, batch_size)
            planes: 'both', 'image', or 'fourier'
        Returns:
            torch.Tensor: (batch_size, P, 2, H, W) where P is number of planes (1 or 2), 2 is (E_x, E_y)
        """
        planes_l = planes.lower()
        if self.use_slm and (planes_l == 'fourier' or planes_l == 'both'):
            if planes_l == 'fourier':
                raise ValueError("Fourier plane mode fields not available when use_slm=True. Only image plane is stored.")
            # For 'both', only return image plane when use_slm=True
            selected_fields = self.mode_fields[0:1]
        elif planes_l == 'both':
            selected_fields = self.mode_fields
        elif planes_l == 'image':
            selected_fields = self.mode_fields[0:1]
        elif planes_l == 'fourier':
            selected_fields = self.mode_fields[1:2]
        else:
            raise ValueError("planes must be 'both', 'image', or 'fourier'")
        selected_fields = selected_fields.contiguous()
        P, C, H, W, N_modes = selected_fields.shape
        field_matrix = selected_fields.reshape(P * C * H * W, N_modes)
        total_fields_flat = torch.matmul(field_matrix, mode_distribution)  # (P*C*H*W, batch)
        batch_size = mode_distribution.shape[1]
        total_fields = total_fields_flat.view(P, C, H, W, batch_size)
        total_fields = total_fields.permute(4, 0, 1, 2, 3)
        return total_fields
    
    def forward_model_slm(
        self,
        slm_pattern: torch.Tensor,
        tm: torch.Tensor,
        zoomfft_params_slm: Optional[Dict] = None,
        zoomfft_params_output_forward: Optional[Dict] = None,
        zoomfft_params_output_inverse: Optional[Dict] = None,
        zernike_coeffs_before_ft: Optional[torch.Tensor] = None,
        zernike_coeffs_after_ft: Optional[torch.Tensor] = None,
        zernike_coeffs_image_plane: Optional[torch.Tensor] = None,
        zernike_coeffs_fourier_domain: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward model: SLM pattern → intensities (image and Fourier planes).
        Uses ZoomFFT for all Fourier transforms (batched, supports arbitrary leading dimensions).
        
        Args:
            slm_pattern: SLM phase pattern, shape (H, W) or (batch, H, W)
            tm: Transmission matrix to use, shape (N_modes, N_modes)
            zoomfft_params_slm: ZoomFFT parameters for SLM→Fourier transform
                Dict with keys: s_a_x, s_a_y, gamma, s_n0, s_m0, s_k0, s_l0
                If None, uses non-aberrative values (all 1.0 except gamma=0.0)
            zoomfft_params_output_forward: ZoomFFT parameters for output image→Fourier
                If None, uses non-aberrative values
            zoomfft_params_output_inverse: ZoomFFT parameters for Fourier→image (inverse)
                If None, uses non-aberrative values
            zernike_coeffs_before_ft: Zernike coefficients before FT, shape (n_modes,)
            zernike_coeffs_after_ft: Zernike coefficients after FT, shape (n_modes,)
            zernike_coeffs_image_plane: Zernike coefficients in image plane, shape (n_modes,)
            zernike_coeffs_fourier_domain: Zernike coefficients in Fourier domain, shape (n_modes,)
            
        Returns:
            torch.Tensor: Intensities, shape (batch, 2, H, W) where [:, 0] = image plane, [:, 1] = Fourier plane
        """
        # Handle batch dimension
        if slm_pattern.dim() == 2:
            slm_pattern = slm_pattern.unsqueeze(0)  # (1, H, W)
            single_batch = True
        else:
            single_batch = False
        
        batch_size, H, W = slm_pattern.shape
        
        # Default non-aberrative parameters
        default_zoomfft_params = {
            's_a_x': 1.0, 's_a_y': 1.0, 'gamma': 0.0,
            's_n0': 1.0, 's_m0': 1.0, 's_k0': 1.0, 's_l0': 1.0
        }
        if zoomfft_params_slm is None:
            zoomfft_params_slm = default_zoomfft_params
        if zoomfft_params_output_forward is None:
            zoomfft_params_output_forward = default_zoomfft_params
        if zoomfft_params_output_inverse is None:
            zoomfft_params_output_inverse = default_zoomfft_params
        
        if zernike_coeffs_before_ft is None:
            zernike_coeffs_before_ft = torch.zeros(self.zernike_n_modes, device=self.device)
        if zernike_coeffs_after_ft is None:
            zernike_coeffs_after_ft = torch.zeros(self.zernike_n_modes, device=self.device)
        if zernike_coeffs_image_plane is None:
            zernike_coeffs_image_plane = torch.zeros(self.zernike_n_modes, device=self.device)
        if zernike_coeffs_fourier_domain is None:
            zernike_coeffs_fourier_domain = torch.zeros(self.zernike_n_modes, device=self.device)
        
        # Step 1: SLM pattern → y-polarized field
        E_y_slm = torch.exp(1j * slm_pattern)  # (batch, H, W)
        
        # Step 2: Apply aberration before Fourier transform
        E_y_slm = self.zernike_layer(E_y_slm, zernike_coeffs_before_ft)  # (batch, H, W)
        
        # Step 3: Fourier transform (SLM plane → Fourier plane = fiber input)
        # ZoomFFT supports batched input via broadcasting
        # Multiply by mult_factor_slm for correct scaling
        E_y_fourier = self.zoomfft(E_y_slm, inverse=False, **zoomfft_params_slm) * self.mult_factor_slm  # (batch, H, W)
        
        # Step 4: Apply aberration after Fourier transform
        E_y_fourier = self.zernike_layer(E_y_fourier, zernike_coeffs_after_ft)  # (batch, H, W)
        
        # Step 5: Project onto image plane mode fields to get input mode distribution
        input_mode_dist = self.project_field_to_modes(E_y_fourier)  # (N_modes, batch)
        
        # Step 6: Apply TM to get output mode distribution
        output_mode_dist = torch.matmul(tm, input_mode_dist)  # (N_modes, batch)
        
        # Step 7: Calculate output fields in image plane directly from output mode distribution
        output_fields_image = self.calculate_fields_from_mode_dist(output_mode_dist, planes='image')
        # output_fields_image shape: (batch, 1, 2, H, W) [plane, E_x/E_y]
        E_output = output_fields_image[:, 0, :, :, :]  # (batch, 2, H, W)
        
        # Step 8: Apply aberration in image plane (both polarizations at once)
        E_output = self.zernike_layer(E_output, zernike_coeffs_image_plane)  # (batch, 2, H, W)
        
        # Step 9: Fourier transform → Fourier plane (both polarizations at once)
        # Multiply by mult_factor_out for correct scaling
        E_fourier = self.zoomfft(E_output, inverse=False, **zoomfft_params_output_forward) * self.mult_factor_out  # (batch, 2, H, W)
        
        # Step 10: Calculate Fourier plane intensity
        I_fourier = torch.abs(E_fourier[:, 0])**2 + torch.abs(E_fourier[:, 1])**2  # (batch, H, W)
        
        # Step 11: Apply aberration in Fourier domain (both polarizations at once)
        E_fourier = self.zernike_layer(E_fourier, zernike_coeffs_fourier_domain)  # (batch, 2, H, W)
        
        # Step 12: Inverse Fourier transform → image plane (both polarizations at once)
        # Divide by mult_factor_out for energy conservation
        E_image = self.zoomfft(E_fourier, inverse=True, **zoomfft_params_output_inverse) / self.mult_factor_out  # (batch, 2, H, W)
        
        # Step 13: Calculate image plane intensity
        I_image = torch.abs(E_image[:, 0])**2 + torch.abs(E_image[:, 1])**2  # (batch, H, W)
        
        # Combine intensities: (batch, 2, H, W)
        intensities = torch.stack([I_image, I_fourier], dim=1)  # (batch, 2, H, W)
        
        if single_batch:
            intensities = intensities.squeeze(0)  # (2, H, W)
        
        return intensities
    
    def sample_unknown_aberrations(
        self,
        zernike_std: float = 0.1,
        zoomfft_std: float = 0.05
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
    
    def sample_mode_distribution(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample random mode distributions.
        
        Args:
            batch_size: Number of mode distributions to sample
            
        Returns:
            torch.Tensor: Complex tensor of shape (N_modes, batch_size) 
                         representing mode amplitudes
        """
        N_modes = len(self.guided_modes)
        return utils_sample_mode_distribution(N_modes, batch_size, self.device)
    
    def calculate_intensity(self, mode_distribution: torch.Tensor, use_real_tm: bool = True) -> torch.Tensor:
        """
        Calculate the intensity I = |E_x|^2 + |E_y|^2 for both image and Fourier planes.
        Args:
            mode_distribution: Tensor of shape (N_modes, batch_size) 
            use_real_tm: If True, use real TM; if False, use optimized TM
        Returns:
            torch.Tensor: Intensity of shape (batch_size, 2, H, W) [2: image, fourier]
        """
        N_modes, batch_size = mode_distribution.shape
        tm = self.real_tm_tensor if use_real_tm else self.optimized_tm_tensor
        # Compute total fields for both planes at once: (batch, 2, 2, H, W) [E_x, E_y]
        total_fields = self.calculate_fields_with_tm(mode_distribution, tm)
        E_x = total_fields[:, :, 0]
        E_y = total_fields[:, :, 1]
        I_both = torch.abs(E_x)**2 + torch.abs(E_y)**2
        return I_both
    
    def sample_measurements(self, num_measurements: int, measurement_mode: str = 'intensity', 
                            use_slm: bool = None, use_aberrations: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample M mode distributions or SLM patterns and calculate measurements for each, for both planes.
        Args:
            num_measurements: Number of measurements to sample
            measurement_mode: 'intensity' for I = |E_x|^2 + |E_y|^2, 'field' for complex fields (E_x, E_y)
            use_slm: If True, sample SLM patterns instead of mode distributions. If None, uses self.use_slm.
            use_aberrations: If True (and use_slm=True), apply aberrations to targets
        Returns:
            Tuple of (inputs, measurements)
                - if use_slm=False:
                    - inputs: (N_modes, num_measurements) - mode distributions
                    - measurements: (num_measurements, 2, H, W) or (num_measurements, 2, 2, H, W)
                - if use_slm=True:
                    - inputs: (num_measurements, H, W) - SLM phase patterns
                    - measurements: (num_measurements, 2, H, W) - intensities
        """
        if use_slm is None:
            use_slm = self.use_slm
        if use_slm:
            return self.sample_measurements_slm(num_measurements, use_aberrations=use_aberrations)
        else:
            print(f"Sampling {num_measurements} measurements...")
            mode_distributions = self.sample_mode_distribution(num_measurements)
            if measurement_mode == 'field':
                field_measurements = self.calculate_fields_with_tm(mode_distributions, self.real_tm_tensor)
                return mode_distributions, field_measurements
            else:
                intensity_measurements = self.calculate_intensity_with_tm(mode_distributions, self.real_tm_tensor)
                return mode_distributions, intensity_measurements
    
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
            intensity_measurements = self.forward_model_slm(
                slm_patterns,
                self.real_tm_tensor,
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
            intensity_measurements = self.forward_model_slm(
                slm_patterns,
                self.real_tm_tensor
                # All aberration params default to non-aberrative values
            )
        
        return slm_patterns, intensity_measurements
    
    def calculate_intensity_with_tm(self, mode_distribution: torch.Tensor, tm: torch.Tensor) -> torch.Tensor:
        """
        Calculate intensity I = |E_x|^2 + |E_y|^2 for both planes using a specific TM (for optimization).
        Args:
            mode_distribution: Tensor of shape (N_modes, batch_size)
            tm: Transmission matrix to use
        Returns:
            torch.Tensor: Intensity of shape (batch_size, 2, H, W)
        """
        N_modes, batch_size = mode_distribution.shape
        # Total fields for both planes at once: (batch, 2, 2, H, W) [E_x, E_y]
        total_fields = self.calculate_fields_with_tm(mode_distribution, tm)
        E_x = total_fields[:, :, 0]
        E_y = total_fields[:, :, 1]
        I_both = torch.abs(E_x)**2 + torch.abs(E_y)**2
        return I_both

    def calculate_fields(self, mode_distribution: torch.Tensor, use_real_tm: bool = True) -> torch.Tensor:
        """
        Calculate complex electric fields (E_x, E_y) for both planes.
        Args:
            mode_distribution: Tensor of shape (N_modes, batch_size)
            use_real_tm: If True, use real TM; if False, use optimized TM
        Returns:
            torch.Tensor: (batch_size, 2, 2, H, W) [planes: image/fourier, components: E_x/E_y]
        """
        tm = self.real_tm_tensor if use_real_tm else self.optimized_tm_tensor
        return self.calculate_fields_with_tm(mode_distribution, tm)

    def calculate_fields_with_tm(self, mode_distribution: torch.Tensor, tm: torch.Tensor, planes: str = 'both') -> torch.Tensor:
        """
        Calculate complex electric fields (E_x, E_y) for both planes with a specific TM.
        Args:
            mode_distribution: Tensor of shape (N_modes, batch_size)
            tm: Transmission matrix to use
            planes: 'both', 'image', or 'fourier'
        Returns:
            torch.Tensor: (batch_size, P, 2, H, W) where P is number of planes (1 or 2), 2 is (E_x, E_y)
        """
        output_modes_nm_b = torch.matmul(tm, mode_distribution)  # (N_modes, batch)
        planes_l = planes.lower()
        if self.use_slm and (planes_l == 'fourier' or planes_l == 'both'):
            if planes_l == 'fourier':
                raise ValueError("Fourier plane mode fields not available when use_slm=True. Only image plane is stored.")
            # For 'both', only return image plane when use_slm=True
            selected_fields = self.mode_fields[0:1]
        elif planes_l == 'both':
            selected_fields = self.mode_fields
        elif planes_l == 'image':
            selected_fields = self.mode_fields[0:1]
        elif planes_l == 'fourier':
            selected_fields = self.mode_fields[1:2]
        else:
            raise ValueError("planes must be 'both', 'image', or 'fourier'")
        selected_fields = selected_fields.contiguous()
        P, C, H, W, N_modes = selected_fields.shape
        field_matrix = selected_fields.reshape(P * C * H * W, N_modes)
        total_fields_flat = torch.matmul(field_matrix, output_modes_nm_b)  # (P*C*H*W, batch)
        batch_size = mode_distribution.shape[1]
        total_fields = total_fields_flat.view(P, C, H, W, batch_size)
        total_fields = total_fields.permute(4, 0, 1, 2, 3)
        return total_fields
    
    def calculate_phase_aligned_tm_error(self) -> Tuple[float, float]:
        """
        Calculate the phase-aligned Frobenius norm between real and optimized TMs.
        
        Returns:
            Tuple of (frobenius_norm, optimal_phase_radians)
        """
        return calculate_phase_aligned_tm_error(
            self.real_tm_tensor, 
            self.optimized_tm_tensor
        )
    
    def evaluate_reconstruction(self) -> Dict:
        """
        Evaluate the quality of TM reconstruction.
        
        Returns:
            Dict containing evaluation metrics
        """
        if self.use_slm:
            # Sample SLM patterns for evaluation
            def sample_slm_fn(batch_size):
                H, W = self.grid_params['ny'], self.grid_params['nx']
                return torch.rand(batch_size, H, W, device=self.device) * 2 * np.pi
            
            # Calculate intensity from SLM patterns
            def calculate_intensity_slm_fn(slm_patterns, use_real_tm):
                tm = self.real_tm_tensor if use_real_tm else self.optimized_tm_tensor
                # Use forward_model_slm with no aberrations for evaluation
                intensities = self.forward_model_slm(slm_patterns, tm)
                return intensities
            
            return metrics_evaluate_reconstruction(
                self.real_tm_tensor,
                self.optimized_tm_tensor,
                sample_slm_fn,
                calculate_intensity_slm_fn,
                use_slm=True
            )
        else:
            return metrics_evaluate_reconstruction(
                self.real_tm_tensor,
                self.optimized_tm_tensor,
                self.sample_mode_distribution,
                self.calculate_intensity,
                use_slm=False
            )
    
    def plot_results(self, evaluation_results: Dict):
        """
        Plot optimization results and evaluation metrics.
        
        Args:
            evaluation_results: Results from evaluate_reconstruction()
        """
        if self.use_slm:
            # Sample SLM patterns for plotting
            def sample_slm_fn(batch_size):
                H, W = self.grid_params['ny'], self.grid_params['nx']
                return torch.rand(batch_size, H, W, device=self.device) * 2 * np.pi
            
            # Calculate intensity from SLM patterns
            def calculate_intensity_slm_fn(slm_patterns, use_real_tm):
                tm = self.real_tm_tensor if use_real_tm else self.optimized_tm_tensor
                # Use forward_model_slm with no aberrations for plotting
                intensities = self.forward_model_slm(slm_patterns, tm)
                return intensities
            
            utils_plot_results(
                evaluation_results,
                sample_slm_fn,
                calculate_intensity_slm_fn,
                use_slm=True
            )
        else:
            utils_plot_results(
                evaluation_results,
                self.sample_mode_distribution,
                self.calculate_intensity,
                use_slm=False
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


def optimize_tm(
    tm_system: TMMeasurementSystem,
    mode_distributions: torch.Tensor, 
    intensity_targets: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    planes: str = 'both'
) -> Dict:
    """
    Optimize the transmission matrix to match the target measurements.
    
    This is a standalone optimization function that operates on a TMMeasurementSystem instance.
    
    Args:
        tm_system: TMMeasurementSystem instance containing the physical environment
        mode_distributions: Pre-sampled mode distributions (N_modes, M)
        intensity_targets: Target intensity measurements I = |E_x|^2 + |E_y|^2 (M, 2, H, W)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization (not used by LBFGS, kept for compatibility)
        batch_size: Batch size for training
        planes: 'both', 'image', or 'fourier' - which plane(s) to optimize on
    Returns:
        Dict containing optimization history and final results
    """
    print("Starting TM optimization...")
    optimized_tm = nn.Parameter(tm_system.optimized_tm_tensor.clone())
    
    # Use LBFGS optimizer
    optimizer = optim.LBFGS([optimized_tm], lr=1, max_iter=20, history_size=10, 
                             line_search_fn="strong_wolfe")
    print(f"Using LBFGS optimizer with MSE loss")
    
    # MSE loss criterion
    criterion = nn.MSELoss()
    
    # Validate and set plane selection
    planes_l = planes.lower()
    if planes_l not in ['both', 'image', 'fourier']:
        raise ValueError("planes must be 'both', 'image', or 'fourier'")
    print(f"Optimizing on {planes_l} plane(s)")
    
    history = {
        'epoch_losses': [],
        'mse_losses': [],
        'tm_norm': [],
        'tm_frobenius_error_aligned': [],
        'learning_rates': []
    }
    num_measurements = mode_distributions.shape[1]
    num_batches_per_epoch = (num_measurements + batch_size - 1) // batch_size
    print(f"Training for {num_epochs} epochs")
    print(f"Total measurements: {num_measurements}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {num_batches_per_epoch}")

    for epoch in range(num_epochs):
        indices = torch.randperm(num_measurements)
        
        # LBFGS requires processing all data in one closure
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            batch_count = 0
            
            # Process all batches
            for batch_idx in range(num_batches_per_epoch):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_measurements)
                batch_indices = indices[start_idx:end_idx]
                batch_modes = mode_distributions[:, batch_indices]
                batch_targets_intensity = intensity_targets[batch_indices]  # (batch, 2, H, W)
                batch_predictions_intensity = tm_system.calculate_intensity_with_tm(batch_modes, optimized_tm)  # (batch, 2, H, W)
                
                # Select planes based on the planes parameter
                if planes_l == 'image':
                    batch_targets_intensity = batch_targets_intensity[:, 0:1]  # (batch, 1, H, W)
                    batch_predictions_intensity = batch_predictions_intensity[:, 0:1]  # (batch, 1, H, W)
                elif planes_l == 'fourier':
                    batch_targets_intensity = batch_targets_intensity[:, 1:2]  # (batch, 1, H, W)
                    batch_predictions_intensity = batch_predictions_intensity[:, 1:2]  # (batch, 1, H, W)
                
                # Convert intensity to amplitude (square root of intensity) for loss calculation
                batch_targets_amplitude = torch.sqrt(batch_targets_intensity + 1e-10)
                batch_predictions_amplitude = torch.sqrt(batch_predictions_intensity + 1e-10)
                
                # Compute MSE loss over selected plane(s) using amplitudes
                batch_loss = criterion(batch_predictions_amplitude*1e5, batch_targets_amplitude*1e5)
                
                total_loss += batch_loss
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            avg_loss.backward()
            return avg_loss
        
        # LBFGS step
        loss = optimizer.step(closure)
        avg_epoch_loss = loss.item()
        
        # Calculate MSE for history
        with torch.no_grad():
            sample_batch_idx = 0
            start_idx = sample_batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_measurements)
            batch_indices = indices[start_idx:end_idx]
            batch_modes = mode_distributions[:, batch_indices]
            batch_targets_intensity = intensity_targets[batch_indices]
            batch_predictions_intensity = tm_system.calculate_intensity_with_tm(batch_modes, optimized_tm)
            
            if planes_l == 'image':
                batch_targets_intensity = batch_targets_intensity[:, 0:1]
                batch_predictions_intensity = batch_predictions_intensity[:, 0:1]
            elif planes_l == 'fourier':
                batch_targets_intensity = batch_targets_intensity[:, 1:2]
                batch_predictions_intensity = batch_predictions_intensity[:, 1:2]
            
            batch_targets_amplitude = torch.sqrt(batch_targets_intensity + 1e-10)
            batch_predictions_amplitude = torch.sqrt(batch_predictions_intensity + 1e-10)
            
            mse_loss = criterion(batch_predictions_amplitude, batch_targets_amplitude)
            avg_mse_loss = mse_loss.item()
        
        history['epoch_losses'].append(avg_epoch_loss)
        history['mse_losses'].append(avg_mse_loss)
        history['tm_norm'].append(torch.norm(optimized_tm).item())
        tm_error_aligned, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, optimized_tm)
        history['tm_frobenius_error_aligned'].append(tm_error_aligned)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_epoch_loss:.6f}, TM Error (Aligned) = {tm_error_aligned:.3f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
    
    # Update the optimized TM in the system
    tm_system.optimized_tm_tensor = optimized_tm.detach()
    print("Optimization completed!")
    return history


def optimize_tm_slm(
    tm_system: TMMeasurementSystem,
    slm_patterns: torch.Tensor,
    intensity_targets: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    planes: str = 'both',
    optimize_aberrations: bool = True
) -> Dict:
    """
    Optimize the transmission matrix and aberrations to match target measurements from SLM patterns.
    
    Args:
        tm_system: TMMeasurementSystem instance containing the physical environment
        slm_patterns: SLM phase patterns (num_measurements, H, W)
        intensity_targets: Target intensity measurements (num_measurements, 2, H, W)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization (not used by LBFGS, kept for compatibility)
        batch_size: Batch size for training
        planes: 'both', 'image', or 'fourier' - which plane(s) to optimize on
        optimize_aberrations: If True, optimize TM + aberrations. If False, optimize TM only.
    Returns:
        Dict containing optimization history and final results
    """
    print(f"Starting TM optimization with SLM patterns (optimize_aberrations={optimize_aberrations})...")
    
    # Initialize optimizable parameters
    optimized_tm = nn.Parameter(tm_system.optimized_tm_tensor.clone())
    
    # Helper to create ZoomFFT parameter dict
    # gamma is fixed at 0 (not optimized) to avoid instability in LBFGS
    def _create_zoomfft_params(optimizable: bool):
        if optimizable:
            return {
                's_a_x': nn.Parameter(torch.tensor(1.0, device=tm_system.device)),
                's_a_y': nn.Parameter(torch.tensor(1.0, device=tm_system.device)),
                'gamma': torch.tensor(0.0, device=tm_system.device),  # Fixed, not optimized
                's_n0': nn.Parameter(torch.tensor(1.0, device=tm_system.device)),
                's_m0': nn.Parameter(torch.tensor(1.0, device=tm_system.device)),
                's_k0': nn.Parameter(torch.tensor(1.0, device=tm_system.device)),
                's_l0': nn.Parameter(torch.tensor(1.0, device=tm_system.device))
            }
        else:
            # All fixed (not optimized)
            return {
                's_a_x': torch.tensor(1.0, device=tm_system.device),
                's_a_y': torch.tensor(1.0, device=tm_system.device),
                'gamma': torch.tensor(0.0, device=tm_system.device),
                's_n0': torch.tensor(1.0, device=tm_system.device),
                's_m0': torch.tensor(1.0, device=tm_system.device),
                's_k0': torch.tensor(1.0, device=tm_system.device),
                's_l0': torch.tensor(1.0, device=tm_system.device)
            }
    
    # Helper to get only optimizable params from zoomfft dict (excludes gamma)
    def _get_optimizable_zoomfft_params(params_dict):
        return [v for k, v in params_dict.items() if k != 'gamma' and isinstance(v, nn.Parameter)]
    
    # Initialize aberration parameters from non-aberrative start point
    zoomfft_params_slm = _create_zoomfft_params(optimize_aberrations)
    zoomfft_params_output_forward = _create_zoomfft_params(optimize_aberrations)
    zoomfft_params_output_inverse = _create_zoomfft_params(optimize_aberrations)
    
    # Zernike coefficients (one set for each aberration location)
    if optimize_aberrations:
        zernike_coeffs_before_ft = nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device))
        zernike_coeffs_after_ft = nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device))
        zernike_coeffs_image_plane = nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device))
        zernike_coeffs_fourier_domain = nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device))
    else:
        zernike_coeffs_before_ft = torch.zeros(tm_system.zernike_n_modes, device=tm_system.device)
        zernike_coeffs_after_ft = torch.zeros(tm_system.zernike_n_modes, device=tm_system.device)
        zernike_coeffs_image_plane = torch.zeros(tm_system.zernike_n_modes, device=tm_system.device)
        zernike_coeffs_fourier_domain = torch.zeros(tm_system.zernike_n_modes, device=tm_system.device)
    
    # Collect parameters for optimizer
    if optimize_aberrations:
        all_params = [optimized_tm] + \
                     _get_optimizable_zoomfft_params(zoomfft_params_slm) + \
                     _get_optimizable_zoomfft_params(zoomfft_params_output_forward) + \
                     _get_optimizable_zoomfft_params(zoomfft_params_output_inverse) + \
                     [zernike_coeffs_before_ft, zernike_coeffs_after_ft, zernike_coeffs_image_plane, zernike_coeffs_fourier_domain]
        aberr_params_count = 18 + 4 * tm_system.zernike_n_modes
    else:
        all_params = [optimized_tm]  # Only TM
        aberr_params_count = 0
    
    # Use LBFGS optimizer
    optimizer = optim.LBFGS(all_params, lr=1, max_iter=20, history_size=10, 
                             line_search_fn="strong_wolfe")
    print(f"Using LBFGS optimizer with MSE loss")
    print(f"Optimizing: TM ({optimized_tm.numel()} params)" + (f" + aberrations ({aberr_params_count} params)" if optimize_aberrations else " (no aberrations)"))
    
    # MSE loss criterion
    criterion = nn.MSELoss()
    
    # Validate and set plane selection
    planes_l = planes.lower()
    if planes_l not in ['both', 'image', 'fourier']:
        raise ValueError("planes must be 'both', 'image', or 'fourier'")
    print(f"Optimizing on {planes_l} plane(s)")
    
    history = {
        'epoch_losses': [],
        'mse_losses': [],
        'tm_norm': [],
        'tm_frobenius_error_aligned': [],
        'learning_rates': []
    }
    num_measurements = slm_patterns.shape[0]
    num_batches_per_epoch = (num_measurements + batch_size - 1) // batch_size
    print(f"Training for {num_epochs} epochs")
    print(f"Total measurements: {num_measurements}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {num_batches_per_epoch}")

    for epoch in range(num_epochs):
        indices = torch.randperm(num_measurements)
        
        # LBFGS requires processing all data in one closure
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            batch_count = 0
            
            # Convert parameter dicts for forward model (keep as tensors for gradients)
            zoomfft_slm_dict = {k: v for k, v in zoomfft_params_slm.items()}
            zoomfft_output_forward_dict = {k: v for k, v in zoomfft_params_output_forward.items()}
            zoomfft_output_inverse_dict = {k: v for k, v in zoomfft_params_output_inverse.items()}
            
            # Process all batches
            for batch_idx in range(num_batches_per_epoch):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_measurements)
                batch_indices = indices[start_idx:end_idx]
                batch_slm = slm_patterns[batch_indices]  # (batch, H, W)
                batch_targets_intensity = intensity_targets[batch_indices]  # (batch, 2, H, W)
                
                # Forward model with current parameters
                batch_predictions_intensity = tm_system.forward_model_slm(
                    batch_slm,
                    optimized_tm,
                    zoomfft_params_slm=zoomfft_slm_dict,
                    zoomfft_params_output_forward=zoomfft_output_forward_dict,
                    zoomfft_params_output_inverse=zoomfft_output_inverse_dict,
                    zernike_coeffs_before_ft=zernike_coeffs_before_ft,
                    zernike_coeffs_after_ft=zernike_coeffs_after_ft,
                    zernike_coeffs_image_plane=zernike_coeffs_image_plane,
                    zernike_coeffs_fourier_domain=zernike_coeffs_fourier_domain
                )  # (batch, 2, H, W)
                
                # Select planes based on the planes parameter
                if planes_l == 'image':
                    batch_targets_intensity = batch_targets_intensity[:, 0:1]  # (batch, 1, H, W)
                    batch_predictions_intensity = batch_predictions_intensity[:, 0:1]  # (batch, 1, H, W)
                elif planes_l == 'fourier':
                    batch_targets_intensity = batch_targets_intensity[:, 1:2]  # (batch, 1, H, W)
                    batch_predictions_intensity = batch_predictions_intensity[:, 1:2]  # (batch, 1, H, W)
                
                # Convert intensity to amplitude (square root of intensity) for loss calculation
                batch_targets_amplitude = torch.sqrt(batch_targets_intensity + 1e-10)
                batch_predictions_amplitude = torch.sqrt(batch_predictions_intensity + 1e-10)
                
                # Compute MSE loss over selected plane(s) using amplitudes
                batch_loss = criterion(batch_predictions_amplitude*1e3, batch_targets_amplitude*1e3)
                
                total_loss += batch_loss
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            avg_loss.backward()
            return avg_loss
        
        # LBFGS step
        loss = optimizer.step(closure)
        avg_epoch_loss = loss.item()
        
        # Calculate MSE for history
        with torch.no_grad():
            sample_batch_idx = 0
            start_idx = sample_batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_measurements)
            batch_indices = indices[start_idx:end_idx]
            batch_slm = slm_patterns[batch_indices]
            batch_targets_intensity = intensity_targets[batch_indices]
            
            zoomfft_slm_dict = {k: v for k, v in zoomfft_params_slm.items()}
            zoomfft_output_forward_dict = {k: v for k, v in zoomfft_params_output_forward.items()}
            zoomfft_output_inverse_dict = {k: v for k, v in zoomfft_params_output_inverse.items()}
            
            batch_predictions_intensity = tm_system.forward_model_slm(
                batch_slm,
                optimized_tm,
                zoomfft_params_slm=zoomfft_slm_dict,
                zoomfft_params_output_forward=zoomfft_output_forward_dict,
                zoomfft_params_output_inverse=zoomfft_output_inverse_dict,
                zernike_coeffs_before_ft=zernike_coeffs_before_ft,
                zernike_coeffs_after_ft=zernike_coeffs_after_ft,
                zernike_coeffs_image_plane=zernike_coeffs_image_plane,
                zernike_coeffs_fourier_domain=zernike_coeffs_fourier_domain
            )
            
            if planes_l == 'image':
                batch_targets_intensity = batch_targets_intensity[:, 0:1]
                batch_predictions_intensity = batch_predictions_intensity[:, 0:1]
            elif planes_l == 'fourier':
                batch_targets_intensity = batch_targets_intensity[:, 1:2]
                batch_predictions_intensity = batch_predictions_intensity[:, 1:2]
            
            batch_targets_amplitude = torch.sqrt(batch_targets_intensity + 1e-10)
            batch_predictions_amplitude = torch.sqrt(batch_predictions_intensity + 1e-10)
            
            mse_loss = criterion(batch_predictions_amplitude, batch_targets_amplitude)
            avg_mse_loss = mse_loss.item()
        
        history['epoch_losses'].append(avg_epoch_loss)
        history['mse_losses'].append(avg_mse_loss)
        history['tm_norm'].append(torch.norm(optimized_tm).item())
        tm_error_aligned, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, optimized_tm)
        history['tm_frobenius_error_aligned'].append(tm_error_aligned)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print(f"Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_epoch_loss:.6f}, TM Error (Aligned) = {tm_error_aligned:.3f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
    
    # Update the optimized TM in the system
    tm_system.optimized_tm_tensor = optimized_tm.detach()
    print("Optimization completed!")
    return history


def main(use_slm: bool = False):
    """
    Example usage of the TM retrieval system.
    
    Args:
        use_slm: If True, optimize using SLM patterns and aberrations.
                 If False, optimize using mode distributions (no aberrations).
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
    zoomfft_params = {
        'a_init': 2*np.pi/N_pad,
        'n0_init': (N_0 - 1) / 2,
        'm0_init': (N_0 - 1) / 2,
        'k0_init': (N_0 - 1) / 2,
        'l0_init': (N_0 - 1) / 2
    }
    
    # Initialize the system
    print("Initializing TM Measurement System...")
    tm_system = TMMeasurementSystem(
        fiber_params, grid_params, 
        device='cuda', 
        N_pad=N_pad, 
        mult_factor_slm=mult_factor_slm,
        mult_factor_out=mult_factor_out,
        zoomfft_params=zoomfft_params,
        use_slm=use_slm,
        zernike_n_modes=8
    )
    
    if use_slm:
        # ===== SLM Pattern Optimization =====
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
        slm_patterns, intensity_measurements = tm_system.sample_measurements(
            num_measurements=2500, 
            use_aberrations=use_aberrations_in_target
        )
        
        # Optimize TM (and optionally aberrations)
        print("\nOptimizing transmission matrix...")
        history = optimize_tm_slm(
            tm_system=tm_system,
            slm_patterns=slm_patterns,
            intensity_targets=intensity_measurements,
            num_epochs=10,
            learning_rate=0.001,
            batch_size=512,
            planes='both',
            optimize_aberrations=optimize_aberrations
        )
    else:
        # ===== Mode Distribution Optimization (no aberrations) =====
        print("\n" + "="*60)
        print("Mode Distribution Optimization Mode (no aberrations)")
        print("="*60)
        
        # Sample mode distribution measurements
        print("\nSampling measurements...")
        mode_distributions, intensity_measurements = tm_system.sample_measurements(
            num_measurements=2500
        )
        
        # Optimize TM only
        print("\nOptimizing transmission matrix...")
        history = optimize_tm(
            tm_system=tm_system,
            mode_distributions=mode_distributions,
            intensity_targets=intensity_measurements,
            num_epochs=15,
            learning_rate=0.001,
            batch_size=512,
            planes='both'
        )
    
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
    
    # Plot training intensity example (only for mode distribution mode)
    if not use_slm:
        print("\nPlotting training-set intensity example...")
        tm_system.plot_training_intensity_example(mode_distributions, intensity_measurements, plane='image')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TM Retrieval Optimization')
    parser.add_argument('--slm', action='store_true', 
                        help='Use SLM pattern optimization with aberrations (default: mode distribution)')
    args = parser.parse_args()
    main(use_slm=True)
