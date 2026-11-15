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


# Camera frames are cropped to FIBER_CROP_MARGIN x the radius of the region the fiber can
# illuminate (core radius in the image plane, NA * f_obj in the Fourier plane) and resized
# to n_out pixels (lab/run_wavefront_control.py::_load_geometry).
FIBER_CROP_MARGIN = 1.1


def disk_support_mask(shape: Tuple[int, int], radius_px: float) -> np.ndarray:
    """Boolean disk of radius ``radius_px`` centered on ((h-1)/2, (w-1)/2); flip-symmetric."""
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.ogrid[:h, :w]
    return (yy - (h - 1) / 2.0) ** 2 + (xx - (w - 1) / 2.0) ** 2 <= float(radius_px) ** 2


def nominal_fiber_support_mask(shape: Tuple[int, int], margin: float = FIBER_CROP_MARGIN) -> np.ndarray:
    """
    Fiber core (image plane) or NA disk (Fourier plane) in a cropped-and-resized frame:
    radius (frame size / 2) / margin. Equal to the radius computed from the physical
    geometry (TMFocusPipeline.support_radius_px) to within ~0.1 px.
    """
    return disk_support_mask(shape, min(shape[0], shape[1]) / (2.0 * margin))


def enhancement_factor(intensity: np.ndarray, target_mask: np.ndarray, support_mask: np.ndarray) -> float:
    """eta = <I>_target / <I>_(support \\ target), all masks boolean with the frame's shape."""
    background_mask = support_mask & ~target_mask
    if not background_mask.any():
        raise ValueError("Enhancement background is empty: target window covers the whole fiber support.")
    return float(intensity[target_mask].mean() / intensity[background_mask].mean())


# Focus figure of merit for single-spot focusing (offline simulations and lab replays).
PBR_PEAK_SEARCH_RADIUS_PX = 3
PBR_WINDOW_HALF_WIDTH_PX = 1
PBR_DEFINITION = (
    "PBR = I_peak / mean(I over fiber support minus the 3x3 window centred on the peak); "
    "I_peak = brightest pixel within 3 px of the target; "
    "support = fiber core disk (image plane) or NA disk (Fourier plane)"
)


def peak_to_background_ratio(
    intensity,
    support_mask: Optional[np.ndarray] = None,
    target_yx: Optional[Tuple[int, int]] = None,
    search_radius: int = PBR_PEAK_SEARCH_RADIUS_PX,
    half_width: int = PBR_WINDOW_HALF_WIDTH_PX,
) -> Tuple[float, Tuple[int, int]]:
    """
    PBR = I_peak / <I>_(support \\ window); returns (pbr, (peak_y, peak_x)).

    ``intensity`` is a 2D numpy array or torch tensor (evaluated without gradients).
    I_peak is the brightest pixel within ``search_radius`` px (square neighbourhood) of
    ``target_yx`` (default: the frame centre (h // 2, w // 2), the offline focus target).
    The window is the (2 * half_width + 1)^2 square centred on the peak; the background is
    the fiber support outside it. ``support_mask`` defaults to nominal_fiber_support_mask.
    """
    if isinstance(intensity, torch.Tensor):
        intensity = intensity.detach().cpu().numpy()
    intensity = np.asarray(intensity, dtype=np.float64)
    h, w = intensity.shape
    if support_mask is None:
        support_mask = nominal_fiber_support_mask((h, w))
    support_mask = np.asarray(support_mask, dtype=bool)
    if target_yx is None:
        target_yx = (h // 2, w // 2)
    yy, xx = np.ogrid[:h, :w]
    near = (np.abs(yy - int(target_yx[0])) <= search_radius) & (np.abs(xx - int(target_yx[1])) <= search_radius)
    search = support_mask & near
    if not search.any():
        search = support_mask
    py, px = np.unravel_index(int(np.argmax(np.where(search, intensity, -np.inf))), intensity.shape)
    window = np.zeros((h, w), dtype=bool)
    window[max(0, py - half_width):py + half_width + 1, max(0, px - half_width):px + half_width + 1] = True
    background = support_mask & ~window
    if not background.any():
        raise ValueError("PBR background is empty.")
    return float(intensity[py, px] / intensity[background].mean()), (int(py), int(px))


def _focus_metric_from_intensity_map(
    intensity_hw: torch.Tensor,
    center_radius: int,
    support_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Enhancement factor: mean intensity in the center window over the mean intensity in
    the fiber support (core / NA disk) outside that window. `support_mask` defaults to
    the nominal disk for the frame size (nominal_fiber_support_mask).
    """
    h, w = intensity_hw.shape
    cy, cx = h // 2, w // 2
    center_mask = torch.zeros((h, w), dtype=torch.bool, device=intensity_hw.device)
    y_min, y_max = max(0, cy - center_radius), min(h, cy + center_radius + 1)
    x_min, x_max = max(0, cx - center_radius), min(w, cx + center_radius + 1)
    center_mask[y_min:y_max, x_min:x_max] = True
    if support_mask is None:
        support_mask = torch.from_numpy(nominal_fiber_support_mask((h, w))).to(intensity_hw.device)
    background_mask = support_mask & ~center_mask
    center_mean = intensity_hw[center_mask].mean()
    background_mean = intensity_hw[background_mask].mean()
    return center_mean / (background_mean)
