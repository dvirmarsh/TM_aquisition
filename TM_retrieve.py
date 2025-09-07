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

class PearsonCorrelationLoss(nn.Module):
    """
    Pearson correlation loss for measuring correlation between predicted and target values.
    
    This loss function measures the negative correlation between predictions and targets,
    encouraging the model to learn the relative patterns rather than just absolute values.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize Pearson correlation loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(PearsonCorrelationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Pearson correlation loss.
        
        Args:
            predictions: Predicted values (B, 2, H, W) or (B, H, W)
            targets: Target values (B, 2, H, W) or (B, H, W)
            
        Returns:
            Loss value (negative correlation)
        """
        # Flatten tensors for correlation calculation
        if predictions.dim() == 4:
            B, P, H, W = predictions.shape
            pred_flat = predictions.view(B * P, -1)
            target_flat = targets.view(B * P, -1)
        else:
            pred_flat = predictions.view(predictions.size(0), -1)
            target_flat = targets.view(targets.size(0), -1)
        # Calculate means
        pred_mean = torch.mean(pred_flat, dim=1, keepdim=True)
        target_mean = torch.mean(target_flat, dim=1, keepdim=True)
        # Center the data
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        # Calculate correlation
        numerator = torch.sum(pred_centered * target_centered, dim=1)
        pred_var = torch.sum(pred_centered ** 2, dim=1)
        target_var = torch.sum(target_centered ** 2, dim=1)
        denominator = torch.sqrt(pred_var * target_var + 1e-8)  # Add small epsilon for numerical stability
        # Calculate correlation coefficient
        correlation = numerator / denominator
        # Return negative correlation as loss (we want to maximize correlation)
        loss = 1 - correlation
        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that combines MSE, Pearson correlation loss, and symmetry loss.
    """
    
    def __init__(self, mse_weight: float = 1.0, correlation_weight: float = 0.1, symmetry_weight: float = 0.0):
        """
        Initialize combined loss.
        
        Args:
            mse_weight: Weight for MSE loss
            correlation_weight: Weight for Pearson correlation loss
            symmetry_weight: Weight for symmetry loss
        """
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.symmetry_weight = symmetry_weight
        self.mse_loss = nn.MSELoss()
        self.correlation_loss = PearsonCorrelationLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, tm: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: Predicted values
            targets: Target values
            tm: Transmission matrix (for symmetry loss)
        Returns:
            Combined loss value
        """
        mse = self.mse_loss(predictions*1e5, targets*1e5)
        correlation = self.correlation_loss(predictions, targets)
        symmetry = 0.0
        if self.symmetry_weight > 0.0 and tm is not None:
            # Hermitian symmetry: penalize ||tm - tm.T.conj||_F
            symmetry = torch.norm(tm - tm.transpose(-2, -1), p='fro') / tm.numel()
        return self.mse_weight * mse + self.correlation_weight * correlation + self.symmetry_weight * symmetry


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
    - Poynting vector calculations
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
        Calculate and store mode field components needed for S_z calculation in both image and Fourier planes.
        
        Returns:
            torch.Tensor: Tensor of shape (N_modes, 2, 4, H, W) containing
                         (E_x, E_y, H_x, H_y) for each mode, for both image and Fourier planes
        """
        N_modes = len(self.guided_modes)
        H, W = self.X.shape
        # 2: [0]=image plane, [1]=fourier plane
        mode_fields = torch.zeros((N_modes, 2, 4, H, W), dtype=torch.complex64, device=self.device)
        for i, mode in tqdm(enumerate(self.guided_modes), total=N_modes, desc="Calculating mode fields"):
            fields = calculate_wave_distribution(mode, self.fiber_params, self.X, self.Y)
            r = np.sqrt(self.X**2 + self.Y**2)
            theta = np.arctan2(self.Y, self.X)
            E_r = fields['Er']
            E_theta = fields['E_theta']
            E_x = E_r * np.cos(theta) - E_theta * np.sin(theta)
            E_y = E_r * np.sin(theta) + E_theta * np.cos(theta)
            H_r = fields['Hr']
            H_theta = fields['H_theta']
            H_x = H_r * np.cos(theta) - H_theta * np.sin(theta)
            H_y = H_r * np.sin(theta) + H_theta * np.cos(theta)
            P_tot = np.sum(0.5 * np.real(E_x * np.conj(H_y) - E_y * np.conj(H_x)))
            E_x = torch.tensor(E_x / np.sqrt(P_tot), dtype=torch.complex64, device=self.device)
            E_y = torch.tensor(E_y / np.sqrt(P_tot), dtype=torch.complex64, device=self.device)
            H_x = torch.tensor(H_x / np.sqrt(P_tot), dtype=torch.complex64, device=self.device)
            H_y = torch.tensor(H_y / np.sqrt(P_tot), dtype=torch.complex64, device=self.device)
            # Image plane fields
            mode_fields[i, 0, 0] = E_x
            mode_fields[i, 0, 1] = E_y
            mode_fields[i, 0, 2] = H_x
            mode_fields[i, 0, 3] = H_y
            # Fourier plane fields (use self.N_pad and self.mult_factor)
            mode_fields[i, 1, 0] = fourier_plane(E_x, self.N_pad, self.mult_factor)
            mode_fields[i, 1, 1] = fourier_plane(E_y, self.N_pad, self.mult_factor)
            mode_fields[i, 1, 2] = fourier_plane(H_x, self.N_pad, self.mult_factor)
            mode_fields[i, 1, 3] = fourier_plane(H_y, self.N_pad, self.mult_factor)
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
            torch.Tensor: Complex tensor of shape (batch_size, N_modes) 
                         representing mode amplitudes
        """
        N_modes = len(self.guided_modes)
        
        # Sample complex Gaussian random amplitudes
        real_part = torch.randn(batch_size, N_modes, device=self.device)
        imag_part = torch.randn(batch_size, N_modes, device=self.device)
        
        # Normalize to unit power
        amplitudes = (real_part + 1j * imag_part) / torch.sqrt(torch.sum(real_part**2 + imag_part**2, dim=1, keepdim=True))
        
        return amplitudes
    
    def calculate_sz(self, mode_distribution: torch.Tensor, use_real_tm: bool = True) -> torch.Tensor:
        """
        Calculate the Poynting vector in the Z direction for both image and Fourier planes.
        Args:
            mode_distribution: Tensor of shape (batch_size, N_modes) 
            use_real_tm: If True, use real TM; if False, use optimized TM
        Returns:
            torch.Tensor: S_z of shape (batch_size, 2, H, W) [2: image, fourier]
        """
        batch_size, N_modes = mode_distribution.shape
        tm = self.real_tm_tensor if use_real_tm else self.optimized_tm_tensor
        output_modes = torch.einsum('nm,bm->bn', tm, mode_distribution)
        # For both planes
        S_z_planes = []
        for plane in range(2):
            total_fields = torch.einsum('bm,mfhw->bfhw', output_modes, self.mode_fields[:, plane])
            E_x = total_fields[:, 0]
            E_y = total_fields[:, 1]
            H_x = total_fields[:, 2]
            H_y = total_fields[:, 3]
            S_z = 0.5 * torch.real(E_x * torch.conj(H_y) - E_y * torch.conj(H_x))
            S_z_planes.append(S_z)
        S_z_both = torch.stack(S_z_planes, dim=1)  # (batch, 2, H, W)
        return S_z_both
    
    def sample_measurements(self, num_measurements: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample M mode distributions and calculate S_z for each, for both planes.
        Args:
            num_measurements: Number of measurements to sample
        Returns:
            Tuple of (mode_distributions, sz_measurements)
                - mode_distributions: (num_measurements, N_modes)
                - sz_measurements: (num_measurements, 2, H, W)
        """
        print(f"Sampling {num_measurements} measurements...")
        mode_distributions = self.sample_mode_distribution(num_measurements)
        sz_measurements = self.calculate_sz_with_tm(mode_distributions, self.real_tm_tensor)
        return mode_distributions, sz_measurements
    
    def optimize_tm(self, 
                   mode_distributions: torch.Tensor, 
                   sz_targets: torch.Tensor,
                   num_epochs: int = 100,
                   learning_rate: float = 0.01,
                   batch_size: int = 32,
                   use_combined_loss: bool = True,
                   mse_weight: float = 1.0,
                   correlation_weight: float = 0.1,
                   symmetry_weight: float = 0.0) -> Dict:
        """
        Optimize the transmission matrix to match the target measurements (both planes).
        Args:
            mode_distributions: Pre-sampled mode distributions (M, N_modes)
            sz_targets: Target S_z measurements (M, 2, H, W)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            use_combined_loss: Whether to use combined MSE + correlation loss
            mse_weight: Weight for MSE loss component
            correlation_weight: Weight for Pearson correlation loss component
            symmetry_weight: Weight for symmetry loss component
        Returns:
            Dict containing optimization history and final results
        """
        print("Starting TM optimization...")
        optimized_tm = nn.Parameter(self.optimized_tm_tensor.clone())
        optimizer = optim.Adam([optimized_tm], lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            min_lr=1e-4,
            threshold=1e-2
        )
        if use_combined_loss:
            criterion = CombinedLoss(mse_weight=mse_weight, correlation_weight=correlation_weight, symmetry_weight=symmetry_weight)
            print(f"Using combined loss (MSE weight: {mse_weight}, Correlation weight: {correlation_weight}, Symmetry weight: {symmetry_weight})")
        else:
            criterion = nn.MSELoss()
            print("Using MSE loss only")
        history = {
            'epoch_losses': [],
            'mse_losses': [],
            'correlation_losses': [],
            'symmetry_losses': [],
            'tm_norm': [],
            'tm_frobenius_error': [],
            'tm_frobenius_error_aligned': [],
            'learning_rates': []
        }
        num_measurements = mode_distributions.shape[0]
        num_batches_per_epoch = (num_measurements + batch_size - 1) // batch_size
        print(f"Training for {num_epochs} epochs")
        print(f"Total measurements: {num_measurements}")
        print(f"Batch size: {batch_size}")
        print(f"Batches per epoch: {num_batches_per_epoch}")
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            mse_losses = []
            correlation_losses = []
            symmetry_losses = []
            indices = torch.randperm(num_measurements)
            for batch_idx in range(num_batches_per_epoch):
                optimizer.zero_grad()
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_measurements)
                batch_indices = indices[start_idx:end_idx]
                batch_modes = mode_distributions[batch_indices]
                batch_targets = sz_targets[batch_indices]  # (batch, 2, H, W)
                batch_predictions = self.calculate_sz_with_tm(batch_modes, optimized_tm)  # (batch, 2, H, W)
                # Compute loss over both planes (sum or mean over plane axis)
                if use_combined_loss:
                    mse_loss = nn.MSELoss()(batch_predictions*1e5, batch_targets*1e5)
                    correlation_loss = PearsonCorrelationLoss()(batch_predictions*1e5, batch_targets*1e5)
                    symmetry_loss = torch.norm(optimized_tm - optimized_tm.transpose(-2, -1), p='fro') if symmetry_weight > 0.0 else 0.0
                    loss = mse_weight * mse_loss + correlation_weight * correlation_loss + symmetry_weight * symmetry_loss
                    mse_losses.append(mse_loss.item())
                    correlation_losses.append(correlation_loss.item())
                    symmetry_losses.append(symmetry_loss.item() if isinstance(symmetry_loss, torch.Tensor) else symmetry_loss)
                else:
                    loss = criterion(batch_predictions*1e5, batch_targets*1e5)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / num_batches_per_epoch
            history['epoch_losses'].append(avg_epoch_loss)
            history['tm_norm'].append(torch.norm(optimized_tm).item())
            if use_combined_loss:
                avg_mse_loss = sum(mse_losses) / len(mse_losses) if mse_losses else 0
                avg_correlation_loss = sum(correlation_losses) / len(correlation_losses) if correlation_losses else 0
                avg_symmetry_loss = sum(symmetry_losses) / len(symmetry_losses) if symmetry_losses else 0
                history['mse_losses'].append(avg_mse_loss)
                history['correlation_losses'].append(avg_correlation_loss)
                history['symmetry_losses'].append(avg_symmetry_loss)
            tm_error = torch.norm(self.real_tm_tensor - optimized_tm, 'fro').item()
            tm_error_aligned, _ = calculate_phase_aligned_frobenius_norm(self.real_tm_tensor, optimized_tm)
            history['tm_frobenius_error'].append(tm_error)
            history['tm_frobenius_error_aligned'].append(tm_error_aligned)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            scheduler.step(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_epoch_loss:.6f}, TM Error = {tm_error:.6f}, TM Error (Aligned) = {tm_error_aligned:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")
        self.optimized_tm_tensor = optimized_tm.detach()
        print("Optimization completed!")
        return history
    
    def calculate_sz_with_tm(self, mode_distribution: torch.Tensor, tm: torch.Tensor) -> torch.Tensor:
        """
        Calculate S_z for both planes using a specific TM (for optimization).
        Args:
            mode_distribution: Tensor of shape (batch_size, N_modes)
            tm: Transmission matrix to use
        Returns:
            torch.Tensor: S_z of shape (batch_size, 2, H, W)
        """
        batch_size, N_modes = mode_distribution.shape
        output_modes = torch.einsum('nm,bm->bn', tm, mode_distribution)
        S_z_planes = []
        for plane in range(2):
            total_fields = torch.einsum('bm,mfhw->bfhw', output_modes, self.mode_fields[:, plane])
            E_x = total_fields[:, 0]
            E_y = total_fields[:, 1]
            H_x = total_fields[:, 2]
            H_y = total_fields[:, 3]
            S_z = 0.5 * torch.real(E_x * torch.conj(H_y) - E_y * torch.conj(H_x))
            S_z_planes.append(S_z)
        S_z_both = torch.stack(S_z_planes, dim=1)
        return S_z_both
    
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
        
        # Calculate reconstruction error
        tm_error = torch.norm(self.real_tm_tensor - self.optimized_tm_tensor, 'fro')
        tm_error_normalized = tm_error / torch.norm(self.real_tm_tensor, 'fro')
        
        # Calculate phase-aligned reconstruction error
        tm_error_aligned, optimal_phase = self.calculate_phase_aligned_tm_error()
        tm_error_aligned_normalized = tm_error_aligned / torch.norm(self.real_tm_tensor, 'fro')
        
        # Test on new measurements
        test_modes = self.sample_mode_distribution(100)
        real_sz = self.calculate_sz(test_modes, use_real_tm=True)
        predicted_sz = self.calculate_sz(test_modes, use_real_tm=False)
        
        sz_error = torch.mean(torch.abs(real_sz - predicted_sz))
        sz_error_normalized = sz_error / torch.mean(torch.abs(real_sz))
        
        return {
            'tm_frobenius_error': tm_error.item(),
            'tm_frobenius_error_normalized': tm_error_normalized.item(),
            'tm_frobenius_error_aligned': tm_error_aligned,
            'tm_frobenius_error_aligned_normalized': tm_error_aligned_normalized,
            'optimal_phase_radians': optimal_phase,
            'optimal_phase_degrees': optimal_phase * 180 / np.pi,
            'sz_mean_error': sz_error.item(),
            'sz_mean_error_normalized': sz_error_normalized.item(),
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
        tm_error_norm = float(evaluation_results['tm_frobenius_error_normalized'])
        tm_error_aligned_norm = float(evaluation_results['tm_frobenius_error_aligned_normalized'])
        sz_error_norm = float(evaluation_results['sz_mean_error_normalized'])
        
        axes[0, 1].bar(['TM Error', 'TM Error\n(Phase-Aligned)', 'S_z Error'], 
                      [tm_error_norm, tm_error_aligned_norm, sz_error_norm])
        axes[0, 1].set_ylabel('Normalized Error')
        axes[0, 1].set_title('Reconstruction Errors')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot example S_z comparison (image plane only)
        test_modes = self.sample_mode_distribution(1)
        real_sz = self.calculate_sz(test_modes, use_real_tm=True)
        predicted_sz = self.calculate_sz(test_modes, use_real_tm=False)
        
        # Select image plane (index 0) for visualization
        im1 = axes[1, 0].imshow(real_sz[0, 0].cpu().numpy(), cmap='viridis')
        axes[1, 0].set_title('Real S_z (Image Plane)')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(predicted_sz[0, 0].cpu().numpy(), cmap='viridis')
        axes[1, 1].set_title('Predicted S_z (Image Plane)')
        plt.colorbar(im2, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history: Dict):
        """
        Plot training history showing losses as a function of epoch.
        
        Args:
            history: Training history from optimize_tm()
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot total loss
        axes[0, 0].plot(history['epoch_losses'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Training Loss vs Epoch')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')  # Log scale for better visualization
        
        # Plot MSE and correlation losses separately if available
        if 'mse_losses' in history and 'correlation_losses' in history:
            axes[0, 1].plot(history['mse_losses'], 'r-', label='MSE Loss', linewidth=2, marker='s', markersize=4)
            axes[0, 1].plot(history['correlation_losses'], 'g-', label='Correlation Loss', linewidth=2, marker='^', markersize=4)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Loss Components vs Epoch')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        
        # Plot TM norm
        axes[1, 0].plot(history['tm_norm'], 'purple', linewidth=2, marker='d', markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('TM Norm')
        axes[1, 0].set_title('Transmission Matrix Norm vs Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot TM Frobenius error
        if 'tm_frobenius_error' in history:
            axes[1, 1].plot(history['tm_frobenius_error'], 'red', linewidth=2, marker='s', markersize=4, label='Regular')
            if 'tm_frobenius_error_aligned' in history:
                axes[1, 1].plot(history['tm_frobenius_error_aligned'], 'blue', linewidth=2, marker='^', markersize=4, label='Phase-Aligned')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('TM Frobenius Error')
            axes[1, 1].set_title('TM Reconstruction Error vs Epoch')
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
        if 'learning_rates' in history:
            axes[2, 0].plot(history['learning_rates'], 'orange', linewidth=2, marker='*', markersize=6)
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
        
        if 'mse_losses' in history and 'correlation_losses' in history:
            print(f"Initial MSE Loss: {history['mse_losses'][0]:.6f}")
            print(f"Final MSE Loss: {history['mse_losses'][-1]:.6f}")
            print(f"Initial Correlation Loss: {history['correlation_losses'][0]:.6f}")
            print(f"Final Correlation Loss: {history['correlation_losses'][-1]:.6f}")
        
        print(f"Initial TM Norm: {history['tm_norm'][0]:.6f}")
        print(f"Final TM Norm: {history['tm_norm'][-1]:.6f}")
        
        if 'tm_frobenius_error' in history:
            print(f"Initial TM Frobenius Error: {history['tm_frobenius_error'][0]:.6f}")
            print(f"Final TM Frobenius Error: {history['tm_frobenius_error'][-1]:.6f}")
            print(f"TM Error Reduction: {history['tm_frobenius_error'][0]/history['tm_frobenius_error'][-1]:.2f}x")
            
            if 'tm_frobenius_error_aligned' in history:
                print(f"Initial TM Error (Phase-Aligned): {history['tm_frobenius_error_aligned'][0]:.6f}")
                print(f"Final TM Error (Phase-Aligned): {history['tm_frobenius_error_aligned'][-1]:.6f}")
                print(f"TM Error Reduction (Phase-Aligned): {history['tm_frobenius_error_aligned'][0]/history['tm_frobenius_error_aligned'][-1]:.2f}x")
        
        print("=" * 35)


def main():
    """
    Example usage of the TM retrieval system.
    """
    wavelength = 632.8e-9
    core_radius = 13e-6
    NA = 0.22
    cladding_index = 1.457
    core_index = np.sqrt(cladding_index**2 + NA**2)
    margin = 1.1
    x_range = core_radius*margin
    N_0 = 256
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
    mode_distributions, sz_measurements = tm_system.sample_measurements(num_measurements=4000)
    
    # Optimize TM
    print("\nOptimizing transmission matrix...")
    history = tm_system.optimize_tm(
        mode_distributions=mode_distributions,
        sz_targets=sz_measurements,
        num_epochs=200,
        learning_rate=0.00001,
        batch_size=128,
        use_combined_loss=True,
        mse_weight=5.0,
        correlation_weight=0.0,
        symmetry_weight=1.0
    )
    
    # Evaluate results
    print("\nEvaluating reconstruction quality...")
    evaluation_results = tm_system.evaluate_reconstruction()
    
    # Print results
    print(f"\nReconstruction Results:")
    print(f"TM Frobenius Error (Normalized): {evaluation_results['tm_frobenius_error_normalized']:.6f}")
    print(f"TM Frobenius Error (Phase-Aligned, Normalized): {evaluation_results['tm_frobenius_error_aligned_normalized']:.6f}")
    print(f"Optimal Phase: {evaluation_results['optimal_phase_degrees']:.2f}° ({evaluation_results['optimal_phase_radians']:.4f} rad)")
    print(f"S_z Mean Error (Normalized): {evaluation_results['sz_mean_error_normalized']:.6f}")
    
    # Plot training history
    print("\nPlotting training history...")
    tm_system.plot_training_history(history)
    
    # Plot results
    tm_system.plot_results(evaluation_results)


if __name__ == "__main__":
    main()
