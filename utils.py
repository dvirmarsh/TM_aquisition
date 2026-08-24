import torch
import torch.nn as nn
import numpy as np
import math
import os
import hashlib
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from mode_solver import find_guided_modes, calculate_wave_distribution
import h5py
from sklearn.model_selection import train_test_split

def sample_tm_perturbed_mmf(
    N_modes: int,
    loss_parameter: float,
    time_reversal_symmetry_broken: bool = False,
    noise_level: Optional[float] = None,
    singular_value_method: str = "scaling"
) -> np.ndarray:
    """
    Sample a transmission matrix of a perturbed multi-mode fiber.
    
    This function generates a complex-valued transmission matrix that statistically
    resembles those of real perturbed MMFs, incorporating mode coupling, loss,
    and appropriate random matrix ensembles.
    
    Parameters
    ----------
    N_modes : int
        Number of input/output modes. Determines the dimension of the square TM (N_modes x N_modes).
    loss_parameter : float
        Value between 0 and 1 controlling average transmission efficiency and singular value spread.
        Higher values (closer to 1) imply less loss, lower values imply more loss.
    time_reversal_symmetry_broken : bool
        If True, uses Gaussian Unitary Ensemble (GUE) for complex random matrices.
        If False, uses Gaussian Orthogonal Ensemble (GOE) for Hermitian matrices.
    noise_level : float, optional
        Standard deviation of complex Gaussian noise to add to the final TM.
        If None, no noise is added.
    singular_value_method : str, optional
        Method for generating modified singular values:
        - "scaling": Simple scaling of original singular values
        - "distribution": Distribution-based approach for strong coupling regime
        Default is "scaling".
    
    Returns
    -------
    np.ndarray
        Complex-valued transmission matrix of size (N_modes x N_modes).
    
    Raises
    ------
    ValueError
        If parameters are invalid (e.g., N_modes <= 0, loss_parameter not in [0,1]).
    
    Notes
    -----
    The algorithm follows these steps:
    1. Generate base random matrix according to time-reversal symmetry
    2. Perform SVD to get unitary matrices and singular values
    3. Modify singular values to incorporate loss and MDL
    4. Reconstruct TM using modified singular values
    5. Optionally add noise
    
    References
    ----------
    [1] M. Plöschner, T. Tyc, and T. Čižmár, "Seeing through chaos in multimode fibres",
        Nature Photonics 9, 529-535 (2015).
    [2] H. Cao and J. Wiersig, "Dielectric microcavities: Model systems for wave chaos
        and non-Hermitian physics", Reviews of Modern Physics 87, 61 (2015).
    [3] M. L. Mehta, "Random Matrices", Academic Press (2004).
    [4] A. P. Mosk, A. Lagendijk, G. Lerosey, and M. Fink, "Controlling waves in space
        and time for imaging and focusing in complex media", Nature Photonics 6, 283-292 (2012).
    [5] S. Rotter and S. Gigan, "Light fields in complex media: Mesoscopic scattering
        meets wave control", Reviews of Modern Physics 89, 015005 (2017).
    [6] C. W. J. Beenakker, "Random-matrix theory of quantum transport",
        Reviews of Modern Physics 69, 731 (1997).
    """
    
    # Input validation
    if N_modes <= 0:
        raise ValueError("N_modes must be a positive integer")
    
    if not 0 <= loss_parameter <= 1:
        raise ValueError("loss_parameter must be between 0 and 1")
    
    if singular_value_method not in ["scaling", "distribution"]:
        raise ValueError("singular_value_method must be 'scaling' or 'distribution'")
    
    if noise_level is not None and noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    
    # Step 1: Determine matrix dimensions
    N = N_modes
    
    # Step 2: Generate base random matrix (A)
    if time_reversal_symmetry_broken:
        # Gaussian Unitary Ensemble (GUE) - complex random matrix
        # Each element drawn from standard complex Gaussian distribution
        A = (np.random.normal(0, 1/np.sqrt(2), (N, N)) + 
             1j * np.random.normal(0, 1/np.sqrt(2), (N, N)))
    else:
        # Gaussian Orthogonal Ensemble (GOE) - real symmetric matrix
        # Generate real Gaussian matrix and make it symmetric
        A_real = np.random.normal(0, 1, (N, N))
        A = (A_real + A_real.T) / 2  # Make symmetric
        # For optical TMs, we typically need complex values even with time-reversal symmetry
        # Add small imaginary component for generality
        imag_part = 1j*np.random.normal(0, 0.1, (N, N))
        imag_part = 1/2*(imag_part + imag_part.T)
        A = A + imag_part
    
    # Step 3: Perform Singular Value Decomposition (SVD)
    U, Sigma_diag_A, V_dagger = np.linalg.svd(A, full_matrices=False)
    # V_dagger = V_dagger.conj().T  # Convert to V_dagger format
    
    # Step 4: Generate modified singular values to incorporate loss and MDL
    if singular_value_method == "scaling":
        # Option A: Simple scaling
        max_sigma = np.max(Sigma_diag_A)
        if max_sigma > 0:
            Sigma_diag_TM = (Sigma_diag_A / max_sigma) * loss_parameter
        else:
            Sigma_diag_TM = np.zeros_like(Sigma_diag_A)
    
    elif singular_value_method == "distribution":
        # Option B: Distribution-based approach for strong coupling regime
        # Generate singular values following a distribution characteristic of MMFs
        # Use a skewed distribution towards lower values to represent loss
        
        # Generate random values from uniform distribution
        uniform_vals = np.random.uniform(0, 1, N)
        
        # Transform to follow a distribution that favors lower values (more loss)
        # Using a power law transformation: sigma = uniform_val^power * loss_parameter
        power = 2.0  # Adjustable parameter - higher values create more skew towards low values
        Sigma_diag_TM = (uniform_vals ** power) * loss_parameter
        
        # Ensure values are in [0, 1]
        Sigma_diag_TM = np.clip(Sigma_diag_TM, 0, 1)
    
    # Step 5: Reconstruct the sampled TM
    # Create diagonal matrix from singular values
    Sigma_matrix = np.diag(Sigma_diag_TM)
    
    # TM_sampled = U * Sigma_diag_TM * V_dagger
    TM_sampled = U @ Sigma_matrix @ V_dagger
    
    # Step 6: Add noise (optional)
    if noise_level is not None and noise_level > 0:
        # Add complex Gaussian noise
        noise_real = np.random.normal(0, noise_level/np.sqrt(2), TM_sampled.shape)
        noise_imag = np.random.normal(0, noise_level/np.sqrt(2), TM_sampled.shape)
        noise_matrix = noise_real + 1j * noise_imag
        TM_sampled = TM_sampled + noise_matrix
    
    return TM_sampled


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
    def __init__(self, N, M, n_modes=12, grid_radius=1.0, start_noll=4):
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
        mask = (rho <= 1.0)
        # 2. Generate Zernike Bases
        if n_modes == 0:
            # Handle case when no modes are requested
            # Create empty tensor with shape (0, N, M)
            self.register_buffer('zernike_stack', torch.empty(0, N, M))
        else:
            zernike_list = []

            current_noll = start_noll
            for i in range(n_modes):
                n, m = self._noll_to_nm(current_noll)
                z_map = self._zernike_polar(n, m, rho, phi)
                z_map = z_map*mask.float()
                if m == 0:
                    norm = math.sqrt(n + 1)
                else:
                    norm = math.sqrt(2 * (n + 1))
                z_map = z_map * norm
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
    # Build on CPU to avoid CUDA OOM when GPU is already full (e.g. sweep runs)
    mode_fields = torch.zeros((num_planes, 2, H, W, N_modes), dtype=torch.complex64, device='cpu')
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
        E_x = torch.tensor(E_x / np.sqrt(I_tot), dtype=torch.complex64, device='cpu')
        E_y = torch.tensor(E_y / np.sqrt(I_tot), dtype=torch.complex64, device='cpu')
        # Image plane fields
        mode_fields[0, 0, :, :, i] = E_x
        mode_fields[0, 1, :, :, i] = E_y
        # Fourier plane fields (only if not image_plane_only)
        if not image_plane_only:
            mode_fields[1, 0, :, :, i] = fourier_plane(E_x, N_pad, mult_factor)
            mode_fields[1, 1, :, :, i] = fourier_plane(E_y, N_pad, mult_factor)
    return mode_fields.to(device)


def get_mode_fields_filename(
    fiber_params: Dict,
    grid_params: Dict,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Generate a unique filename for mode_fields based on fiber and grid parameters.

    Args:
        fiber_params: Dictionary containing fiber parameters
        grid_params: Dictionary containing grid parameters
        cache_dir: Directory to place the cache file. Default (None) writes
            ``mode_fields_<hash>.pt`` in the current working directory, matching
            the paper scripts.

    Returns:
        str: Filename for the mode_fields cache
    """
    # Create a hash of the parameters to ensure uniqueness
    param_string = f"{fiber_params['a']}_{fiber_params['n1']}_{fiber_params['n0']}_{fiber_params['wavelength']}_{grid_params['x_range']}_{grid_params['y_range']}_{grid_params['nx']}_{grid_params['ny']}"
    param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]
    name = f"mode_fields_{param_hash}.pt"
    if cache_dir is None:
        return name
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def load_or_calculate_mode_fields(
    guided_modes,
    fiber_params: Dict,
    grid_params: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    device: str,
    mult_factor: float,
    image_plane_only: bool = True,  # TODO: remove this
    N_pad: int = 128,  # TODO: remove this
    cache_dir: Optional[str] = None,
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
    filename = get_mode_fields_filename(fiber_params, grid_params, cache_dir=cache_dir)
    
    if os.path.exists(filename):
        print(f"Loading mode_fields from {filename}...")
        try:
            # Load to CPU first to avoid CUDA OOM when GPU is already full (e.g. sweep runs)
            mode_fields = torch.load(filename, map_location='cpu')
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
            return mode_fields.to(device)
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










# Functions to load and process real MMF measurement data from H5 files.
def load_real_mmf_data(h5_file_path: str, experiment_group: str, train_test_split_ratio: float = 0.8, random_seed: int = 42, im_factor: float = 1.0):
    """
    Load real MMF measurement data from an H5 file.
    
    Args:
        h5_file_path: Path to the H5 file containing the measurements
        experiment_group: Name of the group/experiment to load (e.g., 'experiment_1', 'group_0')
        train_test_split_ratio: Ratio of data to use for training (default 0.8 = 80% train, 20% test)
        random_seed: Random seed for train/test split reproducibility
    
    Returns:
        Tuple of (train_data, test_data) where each is a dict with:
            - 'slm_patterns': SLM phase masks (N, H, W)
            - 'intensity_image': Image plane intensities (N, H, W)
            - 'intensity_fourier': Fourier plane intensities (N, H, W)
            - 'intensity_targets': Combined intensities (N, 2, H, W) where [:, 0] = image, [:, 1] = fourier
    """
    print(f"Loading data from {h5_file_path}, group: {experiment_group}")
    
    with h5py.File(h5_file_path, 'r') as f:
        if experiment_group not in f.keys():
            raise ValueError(f"Group '{experiment_group}' not found in H5 file. Available groups: {list(f.keys())}")
        
        group = f[experiment_group]
        
        # Try different possible dataset names
        slm_key = None
        img_key = None
        fourier_key = None
        
        for key in group.keys():
            key_lower = key.lower()
            if 'slm' in key_lower or ('phase' in key_lower or 'mask' in key_lower):
                slm_key = key
            elif 'image' in key_lower and 'transmission' in key_lower:
                img_key = key
            elif 'fourier' in key_lower and 'transmission' in key_lower:
                fourier_key = key
        
        # Fallback to default names if not found
        if slm_key is None:
            slm_key = 'slm_phase_mask'
        if img_key is None:
            img_key = 'transmission_image_intensity'
        if fourier_key is None:
            fourier_key = 'transmission_fourier_intensity'
        
        # Load the three datasets
        slm_phase = group[slm_key][:]  # (N, H, W)
        slm_phase = np.flip(slm_phase, axis=(-2,-1))  # flip because of the 4F system
        intensity_image = group[img_key][:]  # (N, H, W)
        intensity_image = np.flip(intensity_image, axis=(-2,-1))  # flip because of the 4F system and BS
        intensity_fourier = group[fourier_key][:]  # (N, H, W)
        # intensity_fourier = np.flip(intensity_fourier, axis=(-2))  # flip because of the 4F system and BS
        
        print(f"Using dataset keys: {slm_key}, {img_key}, {fourier_key}")
        
        print(f"Loaded data shapes:")
        print(f"  SLM phase masks: {slm_phase.shape}, dtype: {slm_phase.dtype}")
        print(f"  Image intensities: {intensity_image.shape}, dtype: {intensity_image.dtype}")
        print(f"  Fourier intensities: {intensity_fourier.shape}, dtype: {intensity_fourier.dtype}")
        
        # Convert uint8 phase masks to phase values: 0-255 -> 0 to 2π*255/256
        # Formula: phase = uint8_value * (2π * 255/256) / 255 = uint8_value * 2π / 256
        if slm_phase.dtype == np.uint8 or slm_phase.dtype == 'uint8':
            print("Converting uint8 phase masks to phase values (0-255 -> 0 to 2π*255/256)...")
            slm_phase = slm_phase.astype(np.float32) * 2 * np.pi / 256.0
            print(f"slm_phase max: {slm_phase.max()} | slm_phase min: {slm_phase.min()}")
        elif slm_phase.max() <= 1.0 and slm_phase.min() >= 0.0:
            # If already normalized to [0, 1], scale to [0, 2π]
            print("Scaling phase masks from [0, 1] to [0, 2π]...")
            slm_phase = slm_phase.astype(np.float32) * 2 * np.pi
        else:
            # Assume already in phase range [0, 2π]
            slm_phase = slm_phase.astype(np.float32)
        
        # Combine intensities: (N, 2, H, W) where [:, 0] = image, [:, 1] = fourier
        intensity_targets = np.stack([intensity_image, intensity_fourier], axis=1)  # (N, 2, H, W)
        
        # Convert to torch tensors
        slm_phase_tensor = torch.from_numpy(slm_phase).float()
        intensity_targets_tensor = torch.from_numpy(intensity_targets).float()
        intensity_targets_tensor[:,0] = intensity_targets_tensor[:,0] * im_factor
        # Split into train and test
        num_samples = slm_phase_tensor.shape[0]
        indices = np.arange(num_samples)
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=1 - train_test_split_ratio, 
            random_state=random_seed
        )
        
        train_data = {
            'slm_patterns': slm_phase_tensor[train_indices],
            'intensity_image': torch.from_numpy(intensity_image[train_indices]).float(),
            'intensity_fourier': torch.from_numpy(intensity_fourier[train_indices]).float(),
            'intensity_targets': intensity_targets_tensor[train_indices]
        }
        
        test_data = {
            'slm_patterns': slm_phase_tensor[test_indices],
            'intensity_image': torch.from_numpy(intensity_image[test_indices]).float(),
            'intensity_fourier': torch.from_numpy(intensity_fourier[test_indices]).float(),
            'intensity_targets': intensity_targets_tensor[test_indices]
        }
        
        print(f"\nData split:")
        print(f"  Training samples: {len(train_indices)} ({len(train_indices)/num_samples*100:.1f}%)")
        print(f"  Test samples: {len(test_indices)} ({len(test_indices)/num_samples*100:.1f}%)")
        
        return train_data, test_data


def list_h5_experiments(h5_file_path: str):
    """
    List all available experiment groups in an H5 file.
    
    Args:
        h5_file_path: Path to the H5 file
    
    Returns:
        List of group names
    """
    with h5py.File(h5_file_path, 'r') as f:
        groups = list(f.keys())
        print(f"Available experiment groups in {h5_file_path}:")
        for i, group in enumerate(groups):
            print(f"  {i}: {group}")
        return groups



