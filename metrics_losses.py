"""
Metrics and Loss Functions for Transmission Matrix Retrieval

This module contains all metrics and loss functions used for evaluating
and optimizing transmission matrices.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


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
    phase_factor = trace_product
    aligned_diff = T1 - phase_factor * T2
    
    # Calculate Frobenius norm
    frobenius_norm = torch.norm(aligned_diff, 'fro')
    
    return frobenius_norm.item(), phi_opt.item()


def calculate_phase_aligned_tm_error(real_tm: torch.Tensor, optimized_tm: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate the phase-aligned Frobenius norm between real and optimized TMs.
    
    Args:
        real_tm: Real transmission matrix tensor
        optimized_tm: Optimized transmission matrix tensor
    
    Returns:
        Tuple of (frobenius_norm, optimal_phase_radians)
    """
    return calculate_phase_aligned_frobenius_norm(real_tm, optimized_tm)


class PearsonCorrelationLoss(nn.Module):
    """
    Pearson correlation loss module for real tensors (intensities).
    
    The loss is defined as 1 - Pearson correlation coefficient, so:
    - Perfect correlation (r=1) gives loss = 0
    - No correlation (r=0) gives loss = 1
    - Perfect negative correlation (r=-1) gives loss = 2
    """
    
    def __init__(self, dim: Optional[int] = None, eps: float = 1e-8):
        """
        Initialize Pearson correlation loss module.
        
        Args:
            dim: Dimension along which to compute correlation. If None, flattens tensors.
            eps: Small value to avoid division by zero
        """
        super(PearsonCorrelationLoss, self).__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate Pearson correlation loss between two real tensors (intensities).
        
        Args:
            x: First real tensor (intensities)
            y: Second real tensor (intensities, same shape as x)
        
        Returns:
            Pearson correlation loss tensor (1 - correlation coefficient)
        """
        # Flatten tensors if no dimension specified
        if self.dim is None:
            x = x.flatten()
            y = y.flatten()
        
        # Center the variables
        x_centered = x - torch.mean(x, dim=self.dim, keepdim=True) if self.dim is not None else x - torch.mean(x)
        y_centered = y - torch.mean(y, dim=self.dim, keepdim=True) if self.dim is not None else y - torch.mean(y)
        
        # Calculate covariance
        if self.dim is None:
            covariance = torch.sum(x_centered * y_centered)
        else:
            covariance = torch.sum(x_centered * y_centered, dim=self.dim, keepdim=True)
        
        # Calculate standard deviations
        x_std = torch.sqrt(torch.sum(x_centered ** 2, dim=self.dim, keepdim=True) + self.eps) if self.dim is not None else torch.sqrt(torch.sum(x_centered ** 2) + self.eps)
        y_std = torch.sqrt(torch.sum(y_centered ** 2, dim=self.dim, keepdim=True) + self.eps) if self.dim is not None else torch.sqrt(torch.sum(y_centered ** 2) + self.eps)
        
        # Calculate Pearson correlation coefficient
        correlation = covariance / (x_std * y_std + self.eps)
        
        # Return loss as 1 - correlation (clamped to avoid numerical issues)
        correlation = torch.clamp(correlation, -1.0, 1.0)
        loss = 1.0 - correlation
        
        return loss.mean()


def evaluate_reconstruction(
    real_tm_tensor: torch.Tensor,
    optimized_tm_tensor: torch.Tensor,
    sample_input_fn,
    calculate_intensity_fn,
    use_slm: bool = False
) -> Dict:
    """
    Evaluate the quality of TM reconstruction.
    
    Args:
        real_tm_tensor: Real transmission matrix tensor
        optimized_tm_tensor: Optimized transmission matrix tensor
        sample_input_fn: Function to sample inputs:
            - If use_slm=False: (batch_size) -> (N_modes, batch_size) mode distributions
            - If use_slm=True: (batch_size) -> (batch_size, H, W) SLM patterns
        calculate_intensity_fn: Function to calculate intensity:
            - If use_slm=False: (mode_dist, use_real_tm) -> intensity
            - If use_slm=True: (slm_patterns, use_real_tm) -> intensity
        use_slm: If True, evaluation uses SLM patterns; if False, uses mode distributions
    
    Returns:
        Dict containing evaluation metrics
    """
    # Calculate singular values
    real_sv = torch.linalg.svd(real_tm_tensor, full_matrices=False)[1]
    optimized_sv = torch.linalg.svd(optimized_tm_tensor, full_matrices=False)[1]
    
    # Calculate phase-aligned reconstruction error
    tm_error_aligned, optimal_phase = calculate_phase_aligned_tm_error(real_tm_tensor, optimized_tm_tensor)
    tm_error_aligned_normalized = tm_error_aligned / torch.norm(real_tm_tensor, 'fro')
    
    # Test on new measurements
    test_inputs = sample_input_fn(100)
    real_intensity = calculate_intensity_fn(test_inputs, use_real_tm=True)
    predicted_intensity = calculate_intensity_fn(test_inputs, use_real_tm=False)
    
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

