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
    
    def __init__(self, fiber_params: Dict, grid_params: Dict, device: str = None,
                 mult_factor: Dict = None, zoomfft_params: Optional[Dict] = None, 
                 zernike_n_modes: int = 8, lab_data: bool = False, dx: Dict = None):
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
        """
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
        self.mode_powers = torch.sum(torch.abs(self.mode_fields[0, 0, :, :, :])**2 + torch.abs(self.mode_fields[0, 1, :, :, :])**2, dim=(0,1)) * self.dx['im']**2 # (N_modes,)
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
        
        # Initialization TM (starting point for optimization)
        self.optimized_tm = sample_tm_perturbed_mmf(
            N_modes=N_modes,
            loss_parameter=1.0,  # Slightly different for initialization
            time_reversal_symmetry_broken=False,
            noise_level=0.0
        )
        
        # Convert to torch tensors
        if not lab_data:
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
        
        print(f"System initialized with {N_modes} guided modes")
        print(f"Grid size: {grid_params['nx']} x {grid_params['ny']}")
        print(f"Aberration models initialized: ZoomFFT and {zernike_n_modes} Zernike modes")
    
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
            mode_amplitudes = torch.sum(E_y_flat.unsqueeze(1) * torch.conj(mode_fields_flat), dim=0) * self.dx['im']**2 # (N_modes,)
            return mode_amplitudes
        else:
            # Batch: (batch, H, W)
            batch_size = E_y_field.shape[0]
            E_y_flat = E_y_field.reshape(batch_size, -1)  # (batch, H*W)
            mode_fields_flat = mode_fields_y.reshape(-1, N_modes)  # (H*W, N_modes)
            # Project: sum(E_y * conj(mode_field)) over spatial grid for each batch
            mode_amplitudes = torch.matmul(E_y_flat, torch.conj(mode_fields_flat)) * self.dx['im']**2 / self.mode_powers # (batch, N_modes)
            return mode_amplitudes.T  # (N_modes, batch)
    
    def calculate_fields_from_mode_dist(self, mode_distribution: torch.Tensor) -> torch.Tensor:
        """
        Calculate complex electric fields (E_x, E_y) directly from mode distribution (no TM applied).
        
        Args:
            mode_distribution: Tensor of shape (N_modes, batch_size)
        Returns:
            torch.Tensor: (batch_size, P, 2, H, W) where P is number of planes (1 or 2), 2 is (E_x, E_y)
        """
        selected_fields = self.mode_fields[0:1]
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
        E_y_slm = torch.exp(1j * slm_pattern) # (batch, H, W)
        
        # Step 2: Apply aberration before Fourier transform
        E_y_slm = self.zernike_layer(E_y_slm, zernike_coeffs_before_ft)  # (batch, H, W)
        
        # Step 3: Fourier transform (SLM plane → Fourier plane = fiber input)
        # ZoomFFT supports batched input via broadcasting
        # Multiply by mult_factor_slm for correct scaling
        E_y_fourier = self.zoomfft['slm'](E_y_slm, inverse=False, **zoomfft_params_slm)# * self.mult_factor['slm']  # (batch, H, W)
        
        # Step 4: Apply aberration after Fourier transform
        E_y_fourier = self.zernike_layer(E_y_fourier, zernike_coeffs_after_ft)  # (batch, H, W)
        
        # Step 5: Project onto image plane mode fields to get input mode distribution
        input_mode_dist = self.project_field_to_modes(E_y_fourier)  # (N_modes, batch)
        
        # Step 6: Apply TM to get output mode distribution
        output_mode_dist = torch.matmul(tm, input_mode_dist)  # (N_modes, batch)
        
        # Step 7: Calculate output fields in image plane directly from output mode distribution
        output_fields_image = self.calculate_fields_from_mode_dist(output_mode_dist)
        # output_fields_image shape: (batch, 1, 2, H, W) [plane, E_x/E_y]
        E_output = output_fields_image[:, 0, :, :, :]  # (batch, 2, H, W)
        
        # Step 8: Apply aberration in image plane (both polarizations at once)
        E_output = self.zernike_layer(E_output, zernike_coeffs_image_plane)  # (batch, 2, H, W)
        
        # Step 9: Fourier transform → Fourier plane (both polarizations at once)
        # Multiply by mult_factor_out for correct scaling
        E_fourier = self.zoomfft['tran'](E_output, inverse=False, **zoomfft_params_output_forward) * self.mult_factor['tran']  # (batch, 2, H, W)
        
        # Step 10: Calculate Fourier plane intensity
        I_fourier = torch.abs(E_fourier[:, 0])**2 + torch.abs(E_fourier[:, 1])**2  # (batch, H, W)
        
        # Step 11: Apply aberration in Fourier domain (both polarizations at once)
        E_fourier = self.zernike_layer(E_fourier, zernike_coeffs_fourier_domain)  # (batch, 2, H, W)
        
        # Step 12: Inverse Fourier transform → image plane (both polarizations at once)
        # Divide by mult_factor_out for energy conservation
        E_image = self.zoomfft['tran'](E_fourier, inverse=True, **zoomfft_params_output_inverse) / self.mult_factor['tran']  # (batch, 2, H, W)
        
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
        zoomfft_std: float = 0.01
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
            tm = self.real_tm_tensor if use_real_tm else self.optimized_tm_tensor
            # Use forward_model_slm with no aberrations for evaluation
            intensities = self.forward_model_slm(slm_patterns, tm)
            return intensities
        
        return metrics_evaluate_reconstruction(
            self.real_tm_tensor,
            self.optimized_tm_tensor,
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
            tm = self.real_tm_tensor if use_real_tm else self.optimized_tm_tensor
            # Use forward_model_slm with no aberrations for plotting
            intensities = self.forward_model_slm(slm_patterns, tm)
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
    batch_size: int = 32,
    planes: str = 'both',
    optimize_aberrations: bool = True
) -> Dict:
    """
    Optimize TM and Aberrations using Adam with Full Epoch Gradient Accumulation.
    (Takes 1 step per epoch using the average gradient of the entire dataset).
    """
    print(f"Starting Adam Optimization (Full Batch / Gradient Accumulation). Optimize Aberrations: {optimize_aberrations}")
    
    # --- 1. Initialize Parameters ---
    optimized_tm = nn.Parameter(tm_system.optimized_tm_tensor.clone())
    
    # Helper for ZoomFFT parameters
    def _create_zoomfft_params(optimizable: bool):
        grad = optimizable
        return {
            's_a_x': nn.Parameter(torch.tensor(1.0, device=tm_system.device, requires_grad=grad)),
            's_a_y': nn.Parameter(torch.tensor(1.0, device=tm_system.device, requires_grad=grad)),
            'gamma': torch.tensor(0.0, device=tm_system.device), # Fixed
            's_n0': nn.Parameter(torch.tensor(1.0, device=tm_system.device, requires_grad=grad)),
            's_m0': nn.Parameter(torch.tensor(1.0, device=tm_system.device, requires_grad=grad)),
            's_k0': nn.Parameter(torch.tensor(1.0, device=tm_system.device, requires_grad=grad)),
            's_l0': nn.Parameter(torch.tensor(1.0, device=tm_system.device, requires_grad=grad))
        }

    # Initialize Params
    zoomfft_slm = _create_zoomfft_params(optimize_aberrations)
    zoomfft_out_fwd = _create_zoomfft_params(optimize_aberrations)
    zoomfft_out_inv = _create_zoomfft_params(optimize_aberrations)
    
    requires_grad = optimize_aberrations
    zernike_coeffs = {
        'before': nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device, requires_grad=requires_grad)),
        'after': nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device, requires_grad=requires_grad)),
        'image': nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device, requires_grad=requires_grad)),
        'fourier': nn.Parameter(torch.zeros(tm_system.zernike_n_modes, device=tm_system.device, requires_grad=requires_grad))
    }

    # Collect optimizable parameters
    all_params = [optimized_tm]
    if optimize_aberrations:
        for d in [zoomfft_slm, zoomfft_out_fwd, zoomfft_out_inv]:
            all_params += [v for k, v in d.items() if isinstance(v, nn.Parameter) and v.requires_grad]
        all_params += list(zernike_coeffs.values())
    z_params = []
    for d in [zoomfft_slm, zoomfft_out_fwd, zoomfft_out_inv]:
        z_params += [v for k, v in d.items() if isinstance(v, nn.Parameter) and v.requires_grad]
    # --- 2. Setup Optimizer ---
    optimizer = optim.Adam(all_params, lr=learning_rate)
    # optimizer = optim.Adam([
    #     {'params': [optimized_tm], 'lr': 0.001}, # TM Learning Rate
    #     {'params': z_params, 'lr': 0.001}, # Higher LR for aberrations
    #     {'params': list(zernike_coeffs.values()), 'lr': 0.001}
    # ])
    
    criterion = nn.L1Loss()
    planes_l = planes.lower()
    num_measurements = slm_patterns.shape[0]
    num_batches = (num_measurements + batch_size - 1) // batch_size
    history = {
        'epoch_losses': [],
        'tm_error': [],
        'tm_norm': [],
        'tm_frobenius_error_aligned': []
    }

    print(f"Measurements: {num_measurements} | Batches/Epoch: {num_batches} | LR: {learning_rate}")

    # --- 3. Training Loop ---
    for epoch in range(num_epochs):
        # Zero gradients at the START of the epoch
        optimizer.zero_grad()
        
        total_epoch_loss = 0.0
        
        # Helper dicts (unwrapped)
        z_slm = {k: v for k, v in zoomfft_slm.items()}
        z_fwd = {k: v for k, v in zoomfft_out_fwd.items()}
        z_inv = {k: v for k, v in zoomfft_out_inv.items()}

        # Shuffle (less critical for full batch, but good practice if batch norms were involved)
        indices = torch.randperm(num_measurements)

        # Loop over batches to ACCUMULATE gradients
        for i in range(num_batches):
            optimizer.zero_grad()
            start = i * batch_size
            end = min(start + batch_size, num_measurements)
            batch_idx = indices[start:end]
            
            # Forward
            pred_intensity = tm_system.forward_model_slm(
                slm_patterns[batch_idx], optimized_tm,
                zoomfft_params_slm=z_slm,
                zoomfft_params_output_forward=z_fwd,
                zoomfft_params_output_inverse=z_inv,
                zernike_coeffs_before_ft=zernike_coeffs['before'],
                zernike_coeffs_after_ft=zernike_coeffs['after'],
                zernike_coeffs_image_plane=zernike_coeffs['image'],
                zernike_coeffs_fourier_domain=zernike_coeffs['fourier']
            )

            # Select Planes
            if planes_l == 'image': 
                pred, target = pred_intensity[:, 0:1], intensity_targets[batch_idx][:, 0:1]
            elif planes_l == 'fourier': 
                pred, target = pred_intensity[:, 1:2], intensity_targets[batch_idx][:, 1:2]
            else: 
                pred, target = pred_intensity, intensity_targets[batch_idx]

            # Loss (Amplitude)
            pred_amp = torch.sqrt(pred)
            target_amp = torch.sqrt(target)
            
            # Normalization (Crucial for stability)
            norm_factor = torch.mean(target_amp).detach()
            loss = criterion(pred_amp / norm_factor, target_amp / norm_factor)
            
            # Scale loss by 1/num_batches so the gradients are averaged, not summed
            # This mimics taking the mean gradient over the full dataset
            loss_scaled = loss / num_batches
            
            # Backward - Accumulates gradients into .grad buffers
            loss_scaled.backward()
            
            # Track 'true' loss for reporting
            total_epoch_loss += loss.item()

        # Step Optimizer - ONCE per epoch
        # (This uses the gradients accumulated over all batches)
            optimizer.step()
        
        # Logging
        avg_loss = total_epoch_loss / num_batches
        with torch.no_grad():
            tm_error, _ = calculate_phase_aligned_frobenius_norm(tm_system.real_tm_tensor, optimized_tm)
            tm_norm = torch.norm(optimized_tm).item()
        
        history['epoch_losses'].append(avg_loss)
        history['tm_error'].append(tm_error)
        history['tm_norm'].append(tm_norm)
        history['tm_frobenius_error_aligned'].append(tm_error)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}, TM Error={tm_error:.4f}, TM Norm={tm_norm:.6f}")

    # Final Update
    tm_system.optimized_tm_tensor = optimized_tm.detach()
    print("Optimization Completed.")
    return history


def main():
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
        dx=dx
    )
    
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
        batch_size=512,
        planes='both',
        optimize_aberrations=optimize_aberrations
    )
    end_time = time.time()
    print(f"Optimization time: {end_time - start_time:.2f} seconds")
    
    
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
