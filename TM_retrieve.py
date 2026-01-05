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
    plot_training_history as utils_plot_training_history
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
        self.mode_fields = load_or_calculate_mode_fields(
            self.guided_modes,
            self.fiber_params,
            self.grid_params,
            self.X,
            self.Y,
            self.device,
            self.N_pad,
            self.mult_factor
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
        
        print(f"System initialized with {N_modes} guided modes")
        print(f"Grid size: {grid_params['nx']} x {grid_params['ny']}")
    
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
        return metrics_evaluate_reconstruction(
            self.real_tm_tensor,
            self.optimized_tm_tensor,
            self.sample_mode_distribution,
            self.calculate_intensity
        )
    
    def plot_results(self, evaluation_results: Dict):
        """
        Plot optimization results and evaluation metrics.
        
        Args:
            evaluation_results: Results from evaluate_reconstruction()
        """
        utils_plot_results(
            evaluation_results,
            self.sample_mode_distribution,
            self.calculate_intensity
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
    history = optimize_tm(
        tm_system=tm_system,
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
