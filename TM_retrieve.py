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
import matplotlib.pyplot as plt
from mode_solver import find_guided_modes, calculate_wave_distribution
from tm_sampler import sample_tm_perturbed_mmf
from tqdm import tqdm
import os
import hashlib

def fourier_plane(x: torch.Tensor, N_pad: int, mult_factor: float) -> torch.Tensor:
    """
    Calculate the field in the Fourier plane.
    N_pad and mult_factor must be provided explicitly.
    """
    N_0 = x.shape[0]
    device = x.device
    center = N_pad // 2
    X_F = torch.fft.fftshift(torch.fft.fft2(x, s=(N_pad, N_pad)))
    y, x_ = torch.meshgrid(
        torch.arange(N_0, device=device),
        torch.arange(N_0, device=device),
        indexing='ij')
    fourier_plane = X_F[center - N_0 // 2:center + N_0 // 2, center - N_0 // 2:center + N_0 // 2]
    phase_factor = (torch.pi / N_pad) * (N_0 - 1)
    fourier_plane = fourier_plane * torch.exp(1j * phase_factor * (x_ + y))
    return fourier_plane * mult_factor


def calculate_phase_aligned_frobenius_norm(T1: torch.Tensor, T2: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate the Frobenius norm between two matrices with optimal phase alignment.
    
    This function:
    1. Calculates phi_opt = arg(Tr(T2^H * T1))
    2. Returns the Frobenius norm of ||T1 - e^(j*phi_opt) * T2||
    
    Args:
        T1: First complex matrix (real TM)
        T2: Second complex matrix (optimized TM)
        
    Returns:
        Tuple of (frobenius_norm, optimal_phase_radians)
    """
    # Calculate the trace of T2^H * T1
    trace_product = torch.trace(torch.conj(torch.transpose(T2, 0, 1)) @ T1)/torch.trace(torch.conj(torch.transpose(T2, 0, 1)) @ T2)
    
    # Calculate the optimal phase
    phi_opt = torch.angle(trace_product)
    
    # Calculate the phase-aligned difference
    phase_factor = torch.exp(1j * phi_opt)
    aligned_diff = T1 - phase_factor * T2
    
    # Calculate Frobenius norm
    frobenius_norm = torch.norm(aligned_diff, 'fro')
    
    return frobenius_norm.item(), phi_opt.item()


class TMMeasurementSystem:
    """
    A class that handles transmission matrix measurements and optimization.
    
    This class manages:
    - Guided mode calculations and storage
    - Real and optimized TM sampling
    - Intensity calculations (I = |E_x|^2 + |E_y|^2)
    - Mode distribution sampling
    """
    
    def __init__(self, fiber_params: Dict, grid_params: Dict, device: str = None, N_pad: int = None, mult_factor: float = None):
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
        """
        self.fiber_params = fiber_params
        self.grid_params = grid_params
        
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
        
        # Store N_pad and mult_factor BEFORE mode_fields calculation
        self.N_pad = N_pad
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
        self.mode_fields = self._load_or_calculate_mode_fields()
        
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
        
        print(f"System initialized with {N_modes} guided modes")
        print(f"Grid size: {grid_params['nx']} x {grid_params['ny']}")
    
    def _calculate_mode_fields(self) -> torch.Tensor:
        """
        Calculate and store mode field components needed for intensity calculation in both image and Fourier planes.
        
        Returns:
            torch.Tensor: Tensor of shape (2, 2, H, W, N_modes) containing
                         (E_x, E_y) for each plane, with the last dim indexing modes
        """
        N_modes = len(self.guided_modes)
        H, W = self.X.shape
        # 2: [0]=image plane, [1]=fourier plane
        # 2: [0]=E_x, [1]=E_y
        mode_fields = torch.zeros((2, 2, H, W, N_modes), dtype=torch.complex64, device=self.device)
        for i, mode in tqdm(enumerate(self.guided_modes), total=N_modes, desc="Calculating mode fields"):
            fields = calculate_wave_distribution(mode, self.fiber_params, self.X, self.Y)
            r = np.sqrt(self.X**2 + self.Y**2)
            theta = np.arctan2(self.Y, self.X)
            E_r = fields['Er']
            E_theta = fields['E_theta']
            E_x = E_r * np.cos(theta) - E_theta * np.sin(theta)
            E_y = E_r * np.sin(theta) + E_theta * np.cos(theta)
            # Normalize using intensity I = |E_x|^2 + |E_y|^2
            I_tot = np.sum(np.abs(E_x)**2 + np.abs(E_y)**2)
            E_x = torch.tensor(E_x / np.sqrt(I_tot), dtype=torch.complex64, device=self.device)
            E_y = torch.tensor(E_y / np.sqrt(I_tot), dtype=torch.complex64, device=self.device)
            # Image plane fields
            mode_fields[0, 0, :, :, i] = E_x
            mode_fields[0, 1, :, :, i] = E_y
            # Fourier plane fields (use self.N_pad and self.mult_factor)
            mode_fields[1, 0, :, :, i] = fourier_plane(E_x, self.N_pad, self.mult_factor)
            mode_fields[1, 1, :, :, i] = fourier_plane(E_y, self.N_pad, self.mult_factor)
        return mode_fields
    
    def _get_mode_fields_filename(self) -> str:
        """
        Generate a unique filename for mode_fields based on fiber and grid parameters.
        
        Returns:
            str: Filename for the mode_fields cache
        """
        # Create a hash of the parameters to ensure uniqueness
        param_string = f"{self.fiber_params['a']}_{self.fiber_params['n1']}_{self.fiber_params['n0']}_{self.fiber_params['wavelength']}_{self.grid_params['x_range']}_{self.grid_params['y_range']}_{self.grid_params['nx']}_{self.grid_params['ny']}"
        param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]
        
        return f"mode_fields_{param_hash}.pt"
    
    def _load_or_calculate_mode_fields(self) -> torch.Tensor:
        """
        Load mode_fields from file if it exists, otherwise calculate and save them.
        
        Returns:
            torch.Tensor: Mode fields tensor
        """
        filename = self._get_mode_fields_filename()
        
        if os.path.exists(filename):
            print(f"Loading mode_fields from {filename}...")
            try:
                mode_fields = torch.load(filename, map_location=self.device)
                expected_modes = len(self.guided_modes)
                if mode_fields.dim() == 5 and mode_fields.shape[-1] != expected_modes:
                    # Legacy layout: (N_modes, 2, 4, H, W) or (N_modes, 2, 2, H, W) -> convert to (2, 2, H, W, N_modes)
                    if mode_fields.shape[0] == expected_modes:
                        mode_fields = mode_fields.permute(1, 2, 3, 4, 0).contiguous()
                        # If old format had 4 components, extract only E_x, E_y (indices 0, 1)
                        if mode_fields.shape[1] == 4:
                            mode_fields = mode_fields[:, 0:2, :, :, :]
                            print("Converted legacy mode_fields layout (4 components) to new format (2,2,H,W,N_modes) - keeping only E_x, E_y.")
                        else:
                            print("Converted legacy mode_fields layout to new format (2,2,H,W,N_modes).")
                    else:
                        raise ValueError(
                            f"Incompatible mode_fields shape {mode_fields.shape} for {expected_modes} modes."
                        )
                # Also handle old format with 4 components in the correct shape
                if mode_fields.dim() == 5 and mode_fields.shape[1] == 4:
                    mode_fields = mode_fields[:, 0:2, :, :, :]
                    print("Extracted E_x, E_y from old 4-component format.")
                print(f"Successfully loaded mode_fields with shape: {mode_fields.shape}")
                return mode_fields
            except Exception as e:
                print(f"Error loading mode_fields from {filename}: {e}")
                print("Will recalculate mode_fields...")
        
        print("Calculating mode_fields (this may take a while)...")
        mode_fields = self._calculate_mode_fields()
        
        # Save the calculated mode_fields
        print(f"Saving mode_fields to {filename}...")
        try:
            torch.save(mode_fields, filename)
            print(f"Successfully saved mode_fields to {filename}")
        except Exception as e:
            print(f"Warning: Could not save mode_fields to {filename}: {e}")
        
        return mode_fields
    
    def clear_mode_fields_cache(self):
        """
        Clear the cached mode_fields file if it exists.
        """
        filename = self._get_mode_fields_filename()
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Cleared mode_fields cache: {filename}")
            except Exception as e:
                print(f"Error clearing cache {filename}: {e}")
        else:
            print(f"No cache file found: {filename}")
    
    def mode_fields_cache_exists(self) -> bool:
        """
        Check if the mode_fields cache file exists.
        
        Returns:
            bool: True if cache exists, False otherwise
        """
        filename = self._get_mode_fields_filename()
        return os.path.exists(filename)
    
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
        
        # Sample complex Gaussian random amplitudes with columns as independent distributions
        real_part = torch.randn(N_modes, batch_size, device=self.device)
        imag_part = torch.randn(N_modes, batch_size, device=self.device)
        
        # Normalize each column to unit power
        power = torch.sum(real_part**2 + imag_part**2, dim=0, keepdim=True)
        amplitudes = (real_part + 1j * imag_part) / torch.sqrt(power)
        
        return amplitudes
    
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
    
    def sample_measurements(self, num_measurements: int, measurement_mode: str = 'intensity') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample M mode distributions and calculate measurements for each, for both planes.
        Args:
            num_measurements: Number of measurements to sample
            measurement_mode: 'intensity' for I = |E_x|^2 + |E_y|^2, 'field' for complex fields (E_x, E_y)
        Returns:
            Tuple of (mode_distributions, measurements)
                - mode_distributions: (N_modes, num_measurements)
                - measurements:
                    * if 'intensity': (num_measurements, 2, H, W)
                    * if 'field': (num_measurements, 2, 2, H, W)
        """
        print(f"Sampling {num_measurements} measurements...")
        mode_distributions = self.sample_mode_distribution(num_measurements)
        if measurement_mode == 'field':
            field_measurements = self.calculate_fields_with_tm(mode_distributions, self.real_tm_tensor)
            return mode_distributions, field_measurements
        else:
            intensity_measurements = self.calculate_intensity_with_tm(mode_distributions, self.real_tm_tensor)
            return mode_distributions, intensity_measurements
    
    def optimize_tm(self, 
                   mode_distributions: torch.Tensor, 
                   intensity_targets: torch.Tensor = None,
                   num_epochs: int = 100,
                   learning_rate: float = 0.01,
                   batch_size: int = 32,
                   planes: str = 'both') -> Dict:
        """
        Optimize the transmission matrix to match the target measurements.
        Args:
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
        optimized_tm = nn.Parameter(self.optimized_tm_tensor.clone())
        
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
                    batch_predictions_intensity = self.calculate_intensity_with_tm(batch_modes, optimized_tm)  # (batch, 2, H, W)
                    
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
                batch_predictions_intensity = self.calculate_intensity_with_tm(batch_modes, optimized_tm)
                
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
            tm_error_aligned, _ = calculate_phase_aligned_frobenius_norm(self.real_tm_tensor, optimized_tm)
            history['tm_frobenius_error_aligned'].append(tm_error_aligned)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            print(f"Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_epoch_loss:.6f}, TM Error (Aligned) = {tm_error_aligned:.3f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        self.optimized_tm_tensor = optimized_tm.detach()
        print("Optimization completed!")
        return history
    
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
        if planes_l == 'both':
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
        return calculate_phase_aligned_frobenius_norm(
            self.real_tm_tensor, 
            self.optimized_tm_tensor
        )
    
    def evaluate_reconstruction(self) -> Dict:
        """
        Evaluate the quality of TM reconstruction.
        
        Returns:
            Dict containing evaluation metrics
        """
        # Calculate singular values
        real_sv = torch.linalg.svd(self.real_tm_tensor, full_matrices=False)[1]
        optimized_sv = torch.linalg.svd(self.optimized_tm_tensor, full_matrices=False)[1]
        
        # Calculate phase-aligned reconstruction error
        tm_error_aligned, optimal_phase = self.calculate_phase_aligned_tm_error()
        tm_error_aligned_normalized = tm_error_aligned / torch.norm(self.real_tm_tensor, 'fro')
        
        # Test on new measurements
        test_modes = self.sample_mode_distribution(100)
        real_intensity = self.calculate_intensity(test_modes, use_real_tm=True)
        predicted_intensity = self.calculate_intensity(test_modes, use_real_tm=False)
        
        intensity_error = torch.mean(torch.abs(real_intensity - predicted_intensity))
        intensity_error_normalized = intensity_error / torch.mean(torch.abs(real_intensity))
        
        return {
            'tm_frobenius_error_aligned': tm_error_aligned,
            'tm_frobenius_error_aligned_normalized': tm_error_aligned_normalized,
            'optimal_phase_radians': optimal_phase,
            'optimal_phase_degrees': optimal_phase * 180 / np.pi,
            'intensity_mean_error': intensity_error.item(),
            'intensity_mean_error_normalized': intensity_error_normalized.item(),
            'real_singular_values': real_sv.cpu().numpy(),
            'optimized_singular_values': optimized_sv.cpu().numpy()
        }
    
    def plot_results(self, evaluation_results: Dict):
        """
        Plot optimization results and evaluation metrics.
        
        Args:
            evaluation_results: Results from evaluate_reconstruction()
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot singular values comparison
        axes[0, 0].plot(evaluation_results['real_singular_values'], 'b-', label='Real TM', linewidth=2)
        axes[0, 0].plot(evaluation_results['optimized_singular_values'], 'r--', label='Optimized TM', linewidth=2)
        axes[0, 0].set_xlabel('Singular Value Index')
        axes[0, 0].set_ylabel('Singular Value')
        axes[0, 0].set_title('Singular Values Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot TM error
        tm_error_aligned_norm = float(evaluation_results['tm_frobenius_error_aligned_normalized'])
        intensity_error_norm = float(evaluation_results['intensity_mean_error_normalized'])
        
        axes[0, 1].bar(['TM Error\n(Phase-Aligned)', 'Intensity Error'], 
                      [tm_error_aligned_norm, intensity_error_norm])
        axes[0, 1].set_ylabel('Normalized Error')
        axes[0, 1].set_title('Reconstruction Errors')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot example intensity comparison (image plane only)
        test_modes = self.sample_mode_distribution(1)
        real_intensity = self.calculate_intensity(test_modes, use_real_tm=True)
        predicted_intensity = self.calculate_intensity(test_modes, use_real_tm=False)
        
        # Select image plane (index 0) for visualization
        im1 = axes[1, 0].imshow(real_intensity[0, 0].cpu().numpy(), cmap='viridis')
        axes[1, 0].set_title('Real Intensity (Image Plane)')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(predicted_intensity[0, 0].cpu().numpy(), cmap='viridis')
        axes[1, 1].set_title('Predicted Intensity (Image Plane)')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
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
        if mode_distributions is None or intensity_targets is None:
            print("Training data not provided; skipping training intensity plot.")
            return
        num_measurements = mode_distributions.shape[1]
        if num_measurements == 0:
            print("Training set is empty; skipping training intensity plot.")
            return
        plane_idx = 0 if plane.lower() == 'image' else 1
        if example_idx is None:
            example_idx = np.random.randint(0, num_measurements)
        example_idx = int(np.clip(example_idx, 0, num_measurements - 1))
        with torch.no_grad():
            mode_sample = mode_distributions[:, example_idx:example_idx + 1]
            predicted_intensity = self.calculate_intensity(mode_sample, use_real_tm=False)[0, plane_idx]
        target_intensity = intensity_targets[example_idx, plane_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im1 = axes[0].imshow(target_intensity.detach().cpu().numpy(), cmap='viridis')
        axes[0].set_title(f'Training GT Intensity ({plane.capitalize()} plane)')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(predicted_intensity.detach().cpu().numpy(), cmap='viridis')
        axes[1].set_title(f'Training Pred Intensity ({plane.capitalize()} plane)')
        plt.colorbar(im2, ax=axes[1])
        
        fig.suptitle(f'Training Example #{example_idx}')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history: Dict):
        """
        Plot training history showing losses as a function of epoch.
        
        Args:
            history: Training history from optimize_tm()
        """
        mse_losses = history.get('mse_losses') or []
        tm_frobenius_error_aligned = history.get('tm_frobenius_error_aligned') or []
        learning_rates = history.get('learning_rates') or []
        has_mse = len(mse_losses) > 0
        has_tm_error_aligned = len(tm_frobenius_error_aligned) > 0
        has_learning_rates = len(learning_rates) > 0
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot total loss
        axes[0, 0].plot(history['epoch_losses'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Training Loss vs Epoch')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')  # Log scale for better visualization
        
        # Plot MSE loss
        if has_mse:
            axes[0, 1].plot(mse_losses, 'r-', label='MSE Loss', linewidth=2, marker='s', markersize=4)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('MSE Loss vs Epoch')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        else:
            axes[0, 1].axis('off')
        
        # Plot TM norm
        axes[1, 0].plot(history['tm_norm'], 'purple', linewidth=2, marker='d', markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('TM Norm')
        axes[1, 0].set_title('Transmission Matrix Norm vs Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot TM Frobenius error
        if has_tm_error_aligned:
            axes[1, 1].plot(tm_frobenius_error_aligned, 'blue', linewidth=2, marker='^', markersize=4, label='Phase-Aligned')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('TM Frobenius Error')
            axes[1, 1].set_title('TM Reconstruction Error (Phase-Aligned) vs Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        else:
            # Fallback if no TM error data
            axes[1, 1].plot(history['epoch_losses'], 'b-', linewidth=2, marker='o', markersize=4)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Total Loss')
            axes[1, 1].set_title('Training Loss vs Epoch (Linear Scale)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        if has_learning_rates:
            axes[2, 0].plot(learning_rates, 'orange', linewidth=2, marker='*', markersize=6)
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Learning Rate')
            axes[2, 0].set_title('Learning Rate vs Epoch')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].set_yscale('log')
        
        # Hide the unused subplot
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== Training History Summary ===")
        print(f"Initial Loss: {history['epoch_losses'][0]:.6f}")
        print(f"Final Loss: {history['epoch_losses'][-1]:.6f}")
        print(f"Loss Reduction: {history['epoch_losses'][0]/history['epoch_losses'][-1]:.2f}x")
        
        if has_mse:
            print(f"Initial MSE Loss: {mse_losses[0]:.6f}")
            print(f"Final MSE Loss: {mse_losses[-1]:.6f}")
        
        print(f"Initial TM Norm: {history['tm_norm'][0]:.6f}")
        print(f"Final TM Norm: {history['tm_norm'][-1]:.6f}")
        
        if has_tm_error_aligned:
            print(f"Initial TM Error (Phase-Aligned): {tm_frobenius_error_aligned[0]:.6f}")
            print(f"Final TM Error (Phase-Aligned): {tm_frobenius_error_aligned[-1]:.6f}")
            print(f"TM Error Reduction (Phase-Aligned): {tm_frobenius_error_aligned[0]/tm_frobenius_error_aligned[-1]:.2f}x")
        
        print("=" * 35)


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
    mult_factor = dx_i*dx_o/(wavelength*f)
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
    
    # Initialize the system, pass N_pad and mult_factor
    print("Initializing TM Measurement System...")
    tm_system = TMMeasurementSystem(fiber_params, grid_params, device='cuda', N_pad=N_pad, mult_factor=mult_factor)
    
    # Sample measurements
    print("\nSampling measurements...")
    mode_distributions, intensity_measurements = tm_system.sample_measurements(num_measurements=2500)
    
    # Optimize TM
    print("\nOptimizing transmission matrix...")
    history = tm_system.optimize_tm(
        mode_distributions=mode_distributions,
        intensity_targets=intensity_measurements,
        num_epochs=15,
        learning_rate=0.001,
        batch_size=512,
        planes='both',
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
    
    # Plot example from the training set
    print("\nPlotting training-set intensity example...")
    tm_system.plot_training_intensity_example(mode_distributions, intensity_measurements, plane='image')


if __name__ == "__main__":
    main()
