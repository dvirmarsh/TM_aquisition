import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import hashlib
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from mode_solver import find_guided_modes, calculate_wave_distribution

# --- 1. Define the ZoomFFT Class ---
class ZoomFFT2D_Optimizable(nn.Module):
    def __init__(self, N, M, a_init, n0_init, m0_init, k0_init, l0_init):
        super().__init__()
        self.N = N
        self.M = M
        self.a_base = a_init
        self.n0_base = n0_init
        self.m0_base = m0_init
        self.k0_base = k0_init
        self.l0_base = l0_init

        # FFT Padded Sizes
        self.fft_n = 2**math.ceil(math.log2(2 * N - 1))
        self.fft_m = 2**math.ceil(math.log2(2 * M - 1))

        # --- Buffers (Coordinates) ---
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32)
        m = torch.arange(M, dtype=torch.float32)
        l = torch.arange(M, dtype=torch.float32)

        self.register_buffer('n_vec', n.view(N, 1))
        self.register_buffer('n_sq_vec', n.view(N, 1)**2)
        self.register_buffer('k_vec', k.view(N, 1))
        self.register_buffer('k_sq_vec', k.view(N, 1)**2)

        self.register_buffer('m_vec', m.view(1, M))
        self.register_buffer('m_sq_vec', m.view(1, M)**2)
        self.register_buffer('l_vec', l.view(1, M))
        self.register_buffer('l_sq_vec', l.view(1, M)**2)

        # --- Buffers (Kernel Indices) ---
        eff_t_n = torch.zeros(self.fft_n, dtype=torch.float32)
        eff_t_n[:N] = torch.arange(N, dtype=torch.float32)
        eff_t_n[self.fft_n-(N-1):] = -torch.arange(1, N, dtype=torch.float32).flip(0)

        eff_t_m = torch.zeros(self.fft_m, dtype=torch.float32)
        eff_t_m[:M] = torch.arange(M, dtype=torch.float32)
        eff_t_m[self.fft_m-(M-1):] = -torch.arange(1, M, dtype=torch.float32).flip(0)

        self.register_buffer('kernel_n_sq', (eff_t_n**2).view(-1, 1))
        self.register_buffer('kernel_m_sq', (eff_t_m**2).view(1, -1))
        self.register_buffer('kernel_cross', eff_t_n.view(-1, 1) * eff_t_m.view(1, -1))

    def forward(self, x, inverse=False, s_a_x=1.0, s_a_y=1.0, gamma=0.0, s_n0=1.0, s_m0=1.0, s_k0=1.0, s_l0=1.0):
        # 1. Effective Parameters
        A_y = self.a_base * s_a_y
        A_x = self.a_base * s_a_x
        n0 = self.n0_base * s_n0
        m0 = self.m0_base * s_m0
        k0 = self.k0_base * s_k0
        l0 = self.l0_base * s_l0

        is_inverse = inverse

        # INVERSE FIX: Do not negate b and c!
        # Negating A (scaling) effectively reverses the shear Phase (A*b)
        # without flipping the geometry parameter itself.
        if is_inverse:
            A_y, A_x = -A_y, -A_x
            n0, k0 = k0, n0
            m0, l0 = l0, m0

            # Normalization
            N_eff = 2 * math.pi / abs(self.a_base)
            norm_factor = (1.0 / (N_eff**2))
            gamma = -gamma
        else:
            # Use torch.sqrt if inputs are tensors, otherwise np.sqrt
            if isinstance(s_a_x, torch.Tensor):
                norm_factor = torch.sqrt(s_a_x * s_a_y - gamma**2/(A_x*A_y))
            else:
                norm_factor = np.sqrt(s_a_x * s_a_y -gamma**2/(A_x*A_y))

            
        # --- 2. Pre-Chirp Phase ---
        # Standard Quadratic
        phase_n = A_y * (k0 * self.n_vec - 0.5 * self.n_sq_vec)
        phase_m = A_x * (l0 * self.m_vec - 0.5 * self.m_sq_vec)

        # Artifact Cancellation
        cancel_nm = -gamma * (self.n_vec * self.m_vec)

        # Linear Shift (using GAMMA)
        if not is_inverse:
            shift_pre_n = (gamma * l0) * self.n_vec
            shift_pre_m = (gamma * k0) * self.m_vec
        else:
            shift_pre_n = (gamma * l0) * self.n_vec
            shift_pre_m = (gamma * k0) * self.m_vec

        x_pre = x * torch.exp(1j * (phase_n + phase_m + cancel_nm + shift_pre_n + shift_pre_m))

        # --- 3. Kernel Generation ---
        phase_H = (0.5 * A_y * self.kernel_n_sq) + \
                  (0.5 * A_x * self.kernel_m_sq) + \
                  (gamma * self.kernel_cross)

        H_2d = torch.fft.fft2(torch.exp(1j * phase_H))

        # --- 4. Convolution ---
        X_fft = torch.fft.fft2(x_pre, s=(self.fft_n, self.fft_m))
        out_fft = torch.fft.ifft2(X_fft * H_2d)
        out = out_fft[..., :self.N, :self.M]

        # --- 5. Post-Chirp Phase ---
        # Standard Quadratic
        phase_k = A_y * (n0 * self.k_vec - 0.5 * self.k_sq_vec - n0 * k0)
        phase_l = A_x * (m0 * self.l_vec - 0.5 * self.l_sq_vec - m0 * l0)

        # Artifact Cancellation
        cancel_kl = -gamma * (self.k_vec * self.l_vec)

        # Linear Shift (using GAMMA)
        if not is_inverse:
            shift_post_k = (gamma * m0) * self.k_vec
            shift_post_l = (gamma * n0) * self.l_vec
        else:
            shift_post_k = (gamma * m0) * self.k_vec
            shift_post_l = (gamma * n0) * self.l_vec

        final_phase = phase_k + phase_l + cancel_kl + shift_post_k + shift_post_l

        return (out * torch.exp(1j * final_phase)) * norm_factor

# --- 2. Define the abberation model ---

class ZernikeAberrationLayer(nn.Module):
    def __init__(self, N, M, n_modes=12, grid_radius=1.0):
        """
        Args:
            N, M: Grid dimensions.
            n_modes: Number of modes (starting from Noll index 4: Defocus).
            grid_radius: Radius of the aperture relative to grid edge.
        """
        super().__init__()
        self.N = N
        self.M = M
        self.n_modes = n_modes
        
        # 1. Create Coordinate Grid
        y = torch.linspace(-1, 1, N)
        x = torch.linspace(-1, 1, M)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        rho = torch.sqrt(X**2 + Y**2) / grid_radius
        phi = torch.atan2(Y, X)
        
        # 2. Generate Zernike Bases
        if n_modes == 0:
            # Handle case when no modes are requested
            # Create empty tensor with shape (0, N, M)
            self.register_buffer('zernike_stack', torch.empty(0, N, M))
        else:
            zernike_list = []

            current_noll = 4
            for i in range(n_modes):
                n, m = self._noll_to_nm(current_noll)
                z_map = self._zernike_polar(n, m, rho, phi)
                zernike_list.append(z_map)
                
                current_noll += 1
                
            # Shape: (n_modes, N, M)
            self.register_buffer('zernike_stack', torch.stack(zernike_list, dim=0))

    def forward(self, x, coeffs):
        """
        Args:
            x: Input with shape (..., N, M) where ... can be any leading dimensions
               (e.g., (batch, N, M), (batch, polarization, N, M), etc.)
            coeffs: Global coefficients (n_modes,). 
                    Same coeffs applied to every image in batch.
        """
        # Handle case when no modes are requested (n_modes=0)
        if self.n_modes == 0:
            return x
        
        # Calculate single phase screen from global coefficients
        # coeffs: (n_modes)
        # zernike_stack: (n_modes, N, M)
        # Result: (N, M)
        total_phase = torch.einsum('m, mhw -> hw', coeffs, self.zernike_stack)
        
        # Create aberration term
        aberration = torch.exp(1j * total_phase)
        
        # Broadcast (N, M) to match x's shape (..., N, M)
        # aberration shape is (N, M), x has arbitrary leading dims
        return x * aberration

    def _noll_to_nm(self, j):
        """Convert Noll index j to (n, m) and print."""
        n = 0
        while (n + 1) * (n + 2) / 2 < j:
            n += 1
        
        m_index = j - n * (n + 1) // 2
        
        if n % 2 == 0:
            m = 2 * (m_index // 2)
        else:
            m = 2 * ((m_index - 1) // 2) + 1
            
        if j % 2 == 0:
            m = m
        else:
            m = -m
            
        return n, m

    def _zernike_polar(self, n, m, rho, phi):
        R_nm = self._zernike_radial(n, abs(m), rho)
        if m > 0:
            return R_nm * torch.cos(m * phi)
        elif m < 0:
            return R_nm * torch.sin(-m * phi)
        else:
            return R_nm

    def _zernike_radial(self, n, m, rho):
        R = torch.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            c = ((-1) ** k * math.factorial(n - k)) / (
                math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
            )
            R += c * (rho ** (n - 2 * k))
        return R

# --- Utility Functions for TM Retrieval ---

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


def calculate_mode_fields(
    guided_modes,
    fiber_params: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    device: str,
    N_pad: int,
    mult_factor: float,
    image_plane_only: bool = False
) -> torch.Tensor:
    """
    Calculate and store mode field components needed for intensity calculation.
    
    Args:
        guided_modes: List of guided modes
        fiber_params: Dictionary containing fiber parameters
        X: X coordinate grid
        Y: Y coordinate grid
        device: Device to use for computations
        N_pad: Padding size for FFT
        mult_factor: Multiplication factor for Fourier plane
        image_plane_only: If True, only calculate image plane fields (for SLM mode)
    
    Returns:
        torch.Tensor: 
            - If image_plane_only=False: (2, 2, H, W, N_modes) [planes: image/fourier, components: E_x/E_y]
            - If image_plane_only=True: (1, 2, H, W, N_modes) [only image plane]
    """
    N_modes = len(guided_modes)
    H, W = X.shape
    # 2: [0]=image plane, [1]=fourier plane (or 1 if image_plane_only)
    # 2: [0]=E_x, [1]=E_y
    num_planes = 1 if image_plane_only else 2
    mode_fields = torch.zeros((num_planes, 2, H, W, N_modes), dtype=torch.complex64, device=device)
    for i, mode in tqdm(enumerate(guided_modes), total=N_modes, desc="Calculating mode fields"):
        fields = calculate_wave_distribution(mode, fiber_params, X, Y)
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        E_r = fields['Er']
        E_theta = fields['E_theta']
        E_x = E_r * np.cos(theta) - E_theta * np.sin(theta)
        E_y = E_r * np.sin(theta) + E_theta * np.cos(theta)
        # Normalize using intensity I = |E_x|^2 + |E_y|^2
        I_tot = np.sum(np.abs(E_x)**2 + np.abs(E_y)**2)
        E_x = torch.tensor(E_x / np.sqrt(I_tot), dtype=torch.complex64, device=device)
        E_y = torch.tensor(E_y / np.sqrt(I_tot), dtype=torch.complex64, device=device)
        # Image plane fields
        mode_fields[0, 0, :, :, i] = E_x
        mode_fields[0, 1, :, :, i] = E_y
        # Fourier plane fields (only if not image_plane_only)
        if not image_plane_only:
            mode_fields[1, 0, :, :, i] = fourier_plane(E_x, N_pad, mult_factor)
            mode_fields[1, 1, :, :, i] = fourier_plane(E_y, N_pad, mult_factor)
    return mode_fields


def get_mode_fields_filename(fiber_params: Dict, grid_params: Dict) -> str:
    """
    Generate a unique filename for mode_fields based on fiber and grid parameters.
    
    Args:
        fiber_params: Dictionary containing fiber parameters
        grid_params: Dictionary containing grid parameters
    
    Returns:
        str: Filename for the mode_fields cache
    """
    # Create a hash of the parameters to ensure uniqueness
    param_string = f"{fiber_params['a']}_{fiber_params['n1']}_{fiber_params['n0']}_{fiber_params['wavelength']}_{grid_params['x_range']}_{grid_params['y_range']}_{grid_params['nx']}_{grid_params['ny']}"
    param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]
    
    return f"mode_fields_{param_hash}.pt"


def load_or_calculate_mode_fields(
    guided_modes,
    fiber_params: Dict,
    grid_params: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    device: str,
    N_pad: int,
    mult_factor: float,
    image_plane_only: bool = False
) -> torch.Tensor:
    """
    Load mode_fields from file if it exists, otherwise calculate and save them.
    
    Args:
        guided_modes: List of guided modes
        fiber_params: Dictionary containing fiber parameters
        grid_params: Dictionary containing grid parameters
        X: X coordinate grid
        Y: Y coordinate grid
        device: Device to use for computations
        N_pad: Padding size for FFT
        mult_factor: Multiplication factor for Fourier plane
        image_plane_only: If True, only calculate/store image plane mode fields (for SLM mode)
    
    Returns:
        torch.Tensor: Mode fields tensor
            - If image_plane_only=False: (2, 2, H, W, N_modes) [planes: image/fourier, components: E_x/E_y]
            - If image_plane_only=True: (1, 2, H, W, N_modes) [only image plane]
    """
    filename = get_mode_fields_filename(fiber_params, grid_params)
    
    if os.path.exists(filename):
        print(f"Loading mode_fields from {filename}...")
        try:
            mode_fields = torch.load(filename, map_location=device)
            expected_modes = len(guided_modes)
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
            
            # If image_plane_only=True, extract only image plane (index 0)
            if image_plane_only and mode_fields.shape[0] == 2:
                mode_fields = mode_fields[0:1, :, :, :, :]  # Keep only image plane
                print("Extracted image plane only (SLM mode).")
            
            print(f"Successfully loaded mode_fields with shape: {mode_fields.shape}")
            return mode_fields
        except Exception as e:
            print(f"Error loading mode_fields from {filename}: {e}")
            print("Will recalculate mode_fields...")
    
    print(f"Calculating mode_fields (this may take a while)... {'[Image plane only]' if image_plane_only else ''}")
    mode_fields = calculate_mode_fields(guided_modes, fiber_params, X, Y, device, N_pad, mult_factor, image_plane_only=image_plane_only)
    
    # Save the calculated mode_fields
    print(f"Saving mode_fields to {filename}...")
    try:
        torch.save(mode_fields, filename)
        print(f"Successfully saved mode_fields to {filename}")
    except Exception as e:
        print(f"Warning: Could not save mode_fields to {filename}: {e}")
    
    return mode_fields


def clear_mode_fields_cache(fiber_params: Dict, grid_params: Dict):
    """
    Clear the cached mode_fields file if it exists.
    
    Args:
        fiber_params: Dictionary containing fiber parameters
        grid_params: Dictionary containing grid parameters
    """
    filename = get_mode_fields_filename(fiber_params, grid_params)
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"Cleared mode_fields cache: {filename}")
        except Exception as e:
            print(f"Error clearing cache {filename}: {e}")
    else:
        print(f"No cache file found: {filename}")


def mode_fields_cache_exists(fiber_params: Dict, grid_params: Dict) -> bool:
    """
    Check if the mode_fields cache file exists.
    
    Args:
        fiber_params: Dictionary containing fiber parameters
        grid_params: Dictionary containing grid parameters
    
    Returns:
        bool: True if cache exists, False otherwise
    """
    filename = get_mode_fields_filename(fiber_params, grid_params)
    return os.path.exists(filename)


def sample_mode_distribution(N_modes: int, batch_size: int, device: str) -> torch.Tensor:
    """
    Sample random mode distributions.
    
    Args:
        N_modes: Number of modes
        batch_size: Number of mode distributions to sample
        device: Device to use for computations
            
    Returns:
        torch.Tensor: Complex tensor of shape (N_modes, batch_size) 
                     representing mode amplitudes
    """
    # Sample complex Gaussian random amplitudes with columns as independent distributions
    real_part = torch.randn(N_modes, batch_size, device=device)
    imag_part = torch.randn(N_modes, batch_size, device=device)
    
    # Normalize each column to unit power
    power = torch.sum(real_part**2 + imag_part**2, dim=0, keepdim=True)
    amplitudes = (real_part + 1j * imag_part) / torch.sqrt(power)
    
    return amplitudes


# --- Plotting Functions ---

def plot_results(
    evaluation_results: Dict,
    sample_input_fn,
    calculate_intensity_fn,
    use_slm: bool = False
):
    """
    Plot optimization results and evaluation metrics.
    
    Args:
        evaluation_results: Results from evaluate_reconstruction()
        sample_input_fn: Function to sample inputs:
            - If use_slm=False: (batch_size) -> (N_modes, batch_size) mode distributions
            - If use_slm=True: (batch_size) -> (batch_size, H, W) SLM patterns
        calculate_intensity_fn: Function to calculate intensity:
            - If use_slm=False: (mode_dist, use_real_tm) -> intensity
            - If use_slm=True: (slm_patterns, use_real_tm) -> intensity
        use_slm: If True, plotting uses SLM patterns; if False, uses mode distributions
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
    test_inputs = sample_input_fn(1)
    real_intensity = calculate_intensity_fn(test_inputs, use_real_tm=True)
    predicted_intensity = calculate_intensity_fn(test_inputs, use_real_tm=False)
    
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
    mode_distributions: torch.Tensor,
    intensity_targets: torch.Tensor,
    calculate_intensity_fn,
    example_idx: Optional[int] = None,
    plane: str = 'image'
):
    """
    Plot an example intensity comparison from the training set.
    
    Args:
        mode_distributions: Tensor of shape (N_modes, num_measurements)
        intensity_targets: Tensor of shape (num_measurements, 2, H, W)
        calculate_intensity_fn: Function to calculate intensity (mode_dist, use_real_tm) -> intensity
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
        predicted_intensity = calculate_intensity_fn(mode_sample, use_real_tm=False)[0, plane_idx]
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


def plot_training_history(history: Dict):
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