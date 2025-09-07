import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kn, jn_zeros
from scipy.optimize import root_scalar

# ==============================================================================
# Function 1: Find all guided modes
# ==============================================================================

def find_guided_modes(a, n1, n0, wavelength, plot_dispersion=False):
    """
    Calculates and finds all guided modes of a step-index optical fiber
    by solving the dispersion relation under the weakly guided approximation.
    This function automatically determines the number of modes to search for.

    Args:
        a (float): Core radius in meters.
        n1 (float): Refractive index of the core.
        n0 (float): Refractive index of the cladding.
        wavelength (float): Wavelength of light in meters.
        plot_dispersion (bool): If True, plots the dispersion relation for each azimuthal mode.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              guided mode and contains its parameters:
              'lp_m', 'lp_l', 'mode_type', 'n_order', 'l_order',
              'v', 'u', 'w', 'beta'.
    """
    # --- 1. Calculate fiber parameters
    k0 = 2 * np.pi / wavelength
    v_number = k0 * a * np.sqrt(n1**2 - n0**2)
    
    print(f"Fiber V-number: {v_number:.3f}")
    if v_number <= 2.405:
        print("Fiber is single-mode (only LP01 propagates).")
    else:
        print("Fiber is multi-mode.")

    guided_modes = []

    # --- 2. Define the unified dispersion equation for LP modes (Eq. 3.74)
    def dispersion_equation(u, v, m):
        """The characteristic equation for LP modes."""
        if u <= 1e-9 or u >= v:
            return np.inf
        w = np.sqrt(v**2 - u**2)
        
        if m == 0:
            left = jv(0, u) / (u * jv(1, u))
            right = kn(0, w) / (w * kn(1, w))
            return left - right
        
        left = jv(m, u) / (u * jv(m - 1, u))
        right = -kn(m, w) / (w * kn(m - 1, w))
        return left - right

    # --- 3. Iterate through LP mode numbers (m, l) to find all possible modes
    m = 0
    while True:
        # Dynamically determine cutoffs for the current m
        cutoffs = []
        l_idx = 0
        while True:
            if m == 0:
                # LP0l -> HE1l. LP01 (l=1) has vc=0. LP0l (l>1) has vc = j_{0, l-1}
                if l_idx == 0:
                    next_cutoff = 0.0
                else:
                    next_cutoff = jn_zeros(1, l_idx)[-1]
            else:
                # LPml -> cutoff is j_{m-1, l}
                next_cutoff = jn_zeros(m - 1, l_idx + 1)[-1]
            cutoffs.append(next_cutoff)
            if next_cutoff > v_number:
                break
            l_idx += 1

        # If the first cutoff for this m is > V, no higher m's can exist.
        if cutoffs[0] > v_number:
            break

        found_us = []  # Store found roots for plotting
        # For the current m, find all radial modes (l) that are guided
        for l_idx, vc in enumerate(cutoffs):
            l = l_idx + 1 # l is a 1-based index
            
            if v_number <= vc:
                break # No more guided radial modes for this m

            # Determine the search interval for the root u
            bracket_low = vc
            if (l_idx + 1) < len(cutoffs):
                bracket_high = min(v_number, cutoffs[l_idx + 1])
            else:
                bracket_high = v_number

            # --- 4. Solve for u
            try:
                sol = root_scalar(
                    lambda u: dispersion_equation(u, v_number, m),
                    bracket=(bracket_low + 1e-6, bracket_high - 1e-6),
                    method='brentq'
                )
                if sol.converged:
                    u = sol.root
                    found_us.append(u)
                    w = np.sqrt(v_number**2 - u**2)
                    beta = np.sqrt((k0 * n1)**2 - (u / a)**2)

                    mode_info = {'lp_m': m, 'lp_l': l, 'v': v_number, 'u': u, 'w': w, 'beta': beta}
                    
                    if m == 0:
                        [guided_modes.append({**mode_info, 'mode_type': 'HE', 'n_order': 1, 'l_order': l, 'psi_idx': psi_idx}) for psi_idx in [0, 1]]
                    elif m == 1:
                        guided_modes.append({**mode_info, 'mode_type': 'TE', 'n_order': 0, 'l_order': l})
                        guided_modes.append({**mode_info, 'mode_type': 'TM', 'n_order': 0, 'l_order': l})
                        [guided_modes.append({**mode_info, 'mode_type': 'HE', 'n_order': 2, 'l_order': l, 'psi_idx': psi_idx}) for psi_idx in [0, 1]]
                    else:
                        [guided_modes.append({**mode_info, 'mode_type': 'EH', 'n_order': m - 1, 'l_order': l, 'psi_idx': psi_idx}) for psi_idx in [0, 1]]
                        [guided_modes.append({**mode_info, 'mode_type': 'HE', 'n_order': m + 1, 'l_order': l, 'psi_idx': psi_idx}) for psi_idx in [0, 1]]
            except (ValueError, RuntimeError):
                pass
        # Plot the dispersion relation for this m (only if requested)
        if plot_dispersion:
            u_grid = np.linspace(0, v_number, 10000)
            disp_values = []
            for u_val in u_grid:
                if u_val <= 1e-9 or u_val >= v_number - 1e-9:
                    disp_values.append(np.inf)
                else:
                    disp_values.append(dispersion_equation(u_val, v_number, m))
            disp_values = np.array(disp_values)
            plt.figure(figsize=(10, 6))
            plt.plot(u_grid, disp_values, 'b-', linewidth=2, label=f'Dispersion equation (m={m})')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.axvline(x=v_number, color='r', linestyle='--', alpha=0.7, label=f'V-number = {v_number:.3f}')
            if found_us:
                plt.plot(found_us, [0]*len(found_us), 'ro', markersize=8, label=f'Found {len(found_us)} roots')
                for i, u_root in enumerate(found_us):
                    plt.annotate(f'u={u_root:.3f}', (u_root, 0), xytext=(10, 10), 
                               textcoords='offset points', fontsize=10, 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            else:
                plt.text(0.5*v_number, 0.2, 
                        'No guided modes found', ha='center', va='center', 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            plt.xlabel('u (normalized transverse wavenumber)')
            plt.ylabel('Dispersion equation value')
            plt.title(f'Dispersion Relation for Azimuthal Mode m={m}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([-0.5, 0.5])
            plt.tight_layout()
            plt.show()
        m += 1
        if m > 20: break # Safety break

    return guided_modes


# ==============================================================================
# Function 2: Calculate Wave Distribution
# ==============================================================================

def calculate_wave_distribution(mode_params, fiber_params, X, Y):
    """
    Calculates the electric and magnetic field distribution for a given mode.

    Args:
        mode_params (dict): A dictionary for a single mode from find_guided_modes.
        fiber_params (dict): Dictionary with 'a', 'n1', 'n0', 'wavelength'.
        X, Y (np.ndarray): 2D grids of X and Y coordinates.

    Returns:
        dict: A dictionary with the complex field components in cylindrical coords
              {'Er', 'E_theta', 'Ez', 'Hr', 'H_theta', 'Hz'}.
    """
    # --- 1. Extract parameters
    mp, fp = mode_params, fiber_params
    A = 1 # Arbitrary amplitude constant
    
    r, theta = np.sqrt(X**2 + Y**2), np.arctan2(Y, X)
    core_mask, clad_mask = r <= fp['a'], r > fp['a']
    
    # Initialize fields
    field_names = ['Er', 'E_theta', 'Ez', 'Hr', 'H_theta', 'Hz']
    fields = {name: np.zeros_like(X, dtype=complex) for name in field_names}

    # --- 2. Constants and mode properties
    n = mp['n_order']
    u, w, beta = mp['u'], mp['w'], mp['beta']
    a, n1, n0 = fp['a'], fp['n1'], fp['n0']
    c = 299792458
    omega = 2 * np.pi * c / fp['wavelength']
    mu0 = 4 * np.pi * 1e-7
    epsilon0 = 1 / (mu0 * c**2)
    
    r_core, theta_core = r[core_mask], theta[core_mask]
    r_clad, theta_clad = r[clad_mask], theta[clad_mask]

    # --- 3. Calculate fields based on mode type
    if mp['mode_type'] == 'TE': # n=0
        # Core (Eqs. 3.22)
        fields['E_theta'][core_mask] = -1j * omega * mu0 * (a/u) * A * jv(1, u * r_core / a)
        fields['Hr'][core_mask] = 1j * beta * (a/u) * A * jv(1, u * r_core / a)
        fields['Hz'][core_mask] = A * jv(0, u * r_core / a)
        # Cladding (Eqs. 3.23)
        clad_norm = jv(0, u) / kn(0, w)
        fields['E_theta'][clad_mask] = 1j * omega * mu0 * (a/w) * A * clad_norm * kn(1, w * r_clad / a)
        fields['Hr'][clad_mask] = -1j * beta * (a/w) * A * clad_norm * kn(1, w * r_clad / a)
        fields['Hz'][clad_mask] = A * clad_norm * kn(0, w * r_clad / a)
        
    elif mp['mode_type'] == 'TM': # n=0
        # Core (Eqs. 3.30)
        fields['Er'][core_mask] = 1j * beta * (a/u) * A * jv(1, u * r_core / a)
        fields['Ez'][core_mask] = A * jv(0, u * r_core / a)
        fields['H_theta'][core_mask] = 1j * omega * epsilon0 * n1**2 * (a/u) * A * jv(1, u * r_core / a)
        # Cladding (Eqs. 3.31)
        clad_norm = jv(0, u) / kn(0, w)
        fields['Er'][clad_mask] = -1j * beta * (a/w) * A * clad_norm * kn(1, w * r_clad / a)
        fields['Ez'][clad_mask] = A * clad_norm * kn(0, w * r_clad / a)
        fields['H_theta'][clad_mask] = -1j * omega * epsilon0 * n0**2 * (a/w) * A * clad_norm * kn(1, w * r_clad / a)

    elif mp['mode_type'] in ['HE', 'EH']:
        psi_idx = mp['psi_idx']
        s = -1 if mp['mode_type'] == 'HE' else 1
        clad_norm = jv(n, u) / kn(n, w)
        
        # Core fields (Eqs. 3.44)
        term1_c = (1 - s) / 2 * jv(n - 1, u * r_core / a)
        term2_c = (1 + s) / 2 * jv(n + 1, u * r_core / a)
        fields['Er'][core_mask] = -1j*A*beta*(a/u) * (term1_c - term2_c) * np.cos(n*theta_core + psi_idx*np.pi/2)
        fields['E_theta'][core_mask] = 1j*A*beta*(a/u) * (term1_c + term2_c) * np.sin(n*theta_core + psi_idx*np.pi/2)
        fields['Ez'][core_mask] = A * jv(n, u * r_core / a) * np.cos(n*theta_core + psi_idx*np.pi/2)
        fields['Hr'][core_mask] = -1j*A*omega*epsilon0*n1**2*(a/u) * (term1_c + term2_c) * np.sin(n*theta_core + psi_idx*np.pi/2)
        fields['H_theta'][core_mask] = -1j*A*omega*epsilon0*n1**2*(a/u) * (term1_c - term2_c) * np.cos(n*theta_core + psi_idx*np.pi/2)
        fields['Hz'][core_mask] = -A * (beta/(omega*mu0)) * s * jv(n, u * r_core / a) * np.sin(n*theta_core + psi_idx*np.pi/2)
        
        # Cladding fields (Eqs. 3.45) - using weak guidance s1=s0=s
        term1_cl = (1 - s) / 2 * kn(n - 1, w * r_clad / a)
        term2_cl = (1 + s) / 2 * kn(n + 1, w * r_clad / a)
        fields['Er'][clad_mask] = -1j*A*beta*(a/w)*clad_norm * (term1_cl + term2_cl) * np.cos(n*theta_clad + psi_idx*np.pi/2)
        fields['E_theta'][clad_mask] = 1j*A*beta*(a/w)*clad_norm * (-term1_cl + term2_cl) * np.sin(n*theta_clad + psi_idx*np.pi/2)
        fields['Ez'][clad_mask] = A * clad_norm * kn(n, w * r_clad / a) * np.cos(n*theta_clad + psi_idx*np.pi/2)
        fields['Hr'][clad_mask] = -1j*A*omega*epsilon0*n0**2*(a/w)*clad_norm * (-term1_cl + term2_cl) * np.sin(n*theta_clad + psi_idx*np.pi/2)
        fields['H_theta'][clad_mask] = -1j*A*omega*epsilon0*n0**2*(a/w)*clad_norm * (term1_cl + term2_cl) * np.cos(n*theta_clad + psi_idx*np.pi/2)
        fields['Hz'][clad_mask] = -A * (beta/(omega*mu0)) * s * clad_norm * kn(n, w * r_clad / a) * np.sin(n*theta_clad + psi_idx*np.pi/2)

    return fields

# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    # --- Fiber Parameters ---
    fiber_core_radius = 25e-6
    fiber_n_core = 1.457
    fiber_n_cladding = 1.44
    # Use a shorter wavelength to make the fiber multi-mode for demonstration
    operating_wavelength = 0.633e-6
    
    # --- Find Modes ---
    print("="*50 + "\n1. Finding Guided Modes\n" + "="*50)
    all_modes = find_guided_modes(
        a=fiber_core_radius, n1=fiber_n_core, n0=fiber_n_cladding, wavelength=operating_wavelength
    )

    print(f"\nFound {len(all_modes)} guided modes (including degeneracies).")
    for i, mode in enumerate(all_modes):
        print(f"  {i+1:2d}: LP_{mode['lp_m']}{mode['lp_l']} -> {mode['mode_type']}_{mode['n_order']}{mode['l_order']} (β={1e-6*mode['beta']:.3e})")

    # --- Calculate and Plot a specific mode ---
    if not all_modes:
        print("\nNo guided modes found to plot.")
    else:
        print("\n" + "="*50 + "\n2. Calculating and Plotting Field Distribution\n" + "="*50)
        
        # Plot a higher-order mode to see azimuthal dependence, e.g., HE21 (from LP11)
        # Find the index for HE21 mode
        try:
            mode_to_plot_index = next(i for i, mode in enumerate(all_modes) if mode['mode_type'] == 'HE' and mode['n_order'] == 2)
        except StopIteration:
            print("HE21 mode not found, plotting fundamental mode instead.")
            mode_to_plot_index = 0
            
        chosen_mode = all_modes[mode_to_plot_index]
        print(f"Plotting Mode: LP_{chosen_mode['lp_m']}{chosen_mode['lp_l']} -> {chosen_mode['mode_type']}_{chosen_mode['n_order']}{chosen_mode['l_order']}")

        grid_size, plot_range = 200, fiber_core_radius * 3
        x = np.linspace(-plot_range, plot_range, grid_size)
        y = np.linspace(-plot_range, plot_range, grid_size)
        X, Y = np.meshgrid(x, y)
        
        fiber_params = {'a': fiber_core_radius, 'n1': fiber_n_core, 'n0': fiber_n_cladding, 'wavelength': operating_wavelength}
        
        cyl_fields = calculate_wave_distribution(chosen_mode, fiber_params, X, Y)

        # Calculate Poynting Vector z-component: Sz = 1/2 Re(Er H_theta* - E_theta Hr*)
        Sz = 0.5 * np.real(cyl_fields['Er'] * np.conj(cyl_fields['H_theta']) - cyl_fields['E_theta'] * np.conj(cyl_fields['Hr']))
        
        # Convert E-fields to Cartesian for vector plot
        theta = np.arctan2(Y, X)
        Ex = cyl_fields['Er'] * np.cos(theta) - cyl_fields['E_theta'] * np.sin(theta)
        Ey = cyl_fields['Er'] * np.sin(theta) + cyl_fields['E_theta'] * np.cos(theta)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Mode: {chosen_mode['mode_type']}_{chosen_mode['n_order']}{chosen_mode['l_order']} (LP_{chosen_mode['lp_m']}{chosen_mode['lp_l']})", fontsize=16)

        # Plot Poynting Vector
        ax = axes[0]
        im = ax.imshow(Sz, extent=[-plot_range*1e6, plot_range*1e6, -plot_range*1e6, plot_range*1e6], cmap='inferno', origin='lower')
        core_circle = plt.Circle((0, 0), fiber_core_radius*1e6, color='white', fill=False, linestyle='--', linewidth=1.5)
        ax.add_artist(core_circle)
        ax.set_title('Poynting Vector ($S_z$)')
        ax.set_xlabel('x (μm)'), ax.set_ylabel('y (μm)')
        fig.colorbar(im, ax=ax, label='Power Flow (a.u.)')

        # Plot Vector Field
        ax = axes[1]
        skip = 15
        ax.quiver(X[::skip, ::skip]*1e6, Y[::skip, ::skip]*1e6, 
                  np.real(Ex[::skip, ::skip]), np.real(Ey[::skip, ::skip]), 
                  color='white', pivot='middle', scale=np.max(np.abs(Ex))*20)
        ax.set_facecolor('black')
        core_circle_2 = plt.Circle((0, 0), fiber_core_radius*1e6, color='cyan', fill=False, linestyle='--', linewidth=1.5)
        ax.add_artist(core_circle_2)
        ax.set_title('Transverse Electric Field (Re{$E_x, E_y$})')
        ax.set_xlabel('x (μm)'), ax.set_ylabel('y (μm)')
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
