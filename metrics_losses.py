"""
Metrics and Loss Functions for Transmission Matrix Retrieval

This module contains all metrics and loss functions used for evaluating
and optimizing transmission matrices.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class PearsonCorrelationLoss(nn.Module):
    """
    Pearson correlation loss module for real tensors (intensities).
    
    The loss is defined as 1 - Pearson correlation coefficient, so:
    - Perfect correlation (r=1) gives loss = 0
    - No correlation (r=0) gives loss = 1
    - Perfect negative correlation (r=-1) gives loss = 2
    """
    
    def __init__(self, dim: Optional[int] = None, eps: float = 1e-8, center: bool = True):
        """
        Initialize Pearson correlation loss module.
        
        Args:
            dim: Dimension along which to compute correlation. If None, flattens tensors.
            eps: Small value to avoid division by zero
        """
        super(PearsonCorrelationLoss, self).__init__()
        self.dim = dim
        self.eps = eps
        self.center = center
    
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
        if self.center:
            x_centered = x - torch.mean(x, dim=self.dim, keepdim=True) if self.dim is not None else x - torch.mean(x)
            y_centered = y - torch.mean(y, dim=self.dim, keepdim=True) if self.dim is not None else y - torch.mean(y)
        else:
            x_centered = x
            y_centered = y
        
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


def _focus_metric_from_intensity_map(intensity_hw: torch.Tensor, center_radius: int) -> torch.Tensor:
    """Center-disk mean intensity over background mean (focus figure of merit)."""
    h, w = intensity_hw.shape
    cy, cx = h // 2, w // 2
    center_mask = torch.zeros((h, w), dtype=torch.bool, device=intensity_hw.device)
    y_min, y_max = max(0, cy - center_radius), min(h, cy + center_radius + 1)
    x_min, x_max = max(0, cx - center_radius), min(w, cx + center_radius + 1)
    center_mask[y_min:y_max, x_min:x_max] = True
    background_mask = ~center_mask
    center_mean = intensity_hw[center_mask].mean()
    background_mean = intensity_hw[background_mask].mean()
    return center_mean / (background_mean)
