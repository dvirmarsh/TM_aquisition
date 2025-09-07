"""
Multi-Mode Fiber Transmission Matrix Sampler

This module implements algorithms for sampling transmission matrices (TMs) 
of perturbed multi-mode fibers (MMFs) based on random matrix theory.
"""

import numpy as np
from typing import Optional, Union
import warnings


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


def analyze_tm_properties(tm: np.ndarray) -> dict:
    """
    Analyze properties of a transmission matrix.
    
    Parameters
    ----------
    tm : np.ndarray
        Transmission matrix to analyze.
    
    Returns
    -------
    dict
        Dictionary containing various TM properties.
    """
    # Singular values
    singular_values = np.linalg.svd(tm, compute_uv=False)
    
    # Transmission efficiency (average of singular values)
    transmission_efficiency = np.mean(singular_values)
    
    # Mode-dependent loss (standard deviation of singular values)
    mdl = np.std(singular_values)
    
    # Condition number (ratio of largest to smallest singular value)
    condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
    
    # Unitarity measure (how far from unitary)
    # For unitary matrix, singular values should all be 1
    unitarity_deviation = np.std(singular_values - 1)
    
    # Frobenius norm
    frobenius_norm = np.linalg.norm(tm, 'fro')
    
    return {
        'singular_values': singular_values,
        'transmission_efficiency': transmission_efficiency,
        'mode_dependent_loss': mdl,
        'condition_number': condition_number,
        'unitarity_deviation': unitarity_deviation,
        'frobenius_norm': frobenius_norm,
        'matrix_shape': tm.shape
    }


def sample_multiple_tms(
    N_modes: int,
    loss_parameter: float,
    time_reversal_symmetry_broken: bool,
    num_samples: int,
    **kwargs
) -> list:
    """
    Sample multiple transmission matrices with the same parameters.
    
    Parameters
    ----------
    N_modes : int
        Number of modes.
    loss_parameter : float
        Loss parameter.
    time_reversal_symmetry_broken : bool
        Time-reversal symmetry flag.
    num_samples : int
        Number of TMs to sample.
    **kwargs
        Additional arguments passed to sample_tm_perturbed_mmf.
    
    Returns
    -------
    list
        List of sampled transmission matrices.
    """
    tms = []
    for _ in range(num_samples):
        tm = sample_tm_perturbed_mmf(
            N_modes=N_modes,
            loss_parameter=loss_parameter,
            time_reversal_symmetry_broken=time_reversal_symmetry_broken,
            **kwargs
        )
        tms.append(tm)
    return tms 