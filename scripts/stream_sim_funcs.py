# -*- coding: utf-8 -*-
"""
Stream-simulation funcs
===========

Utility functions for simulating Nbody GC streams with various methods.

Docstring style
---------------
All docstrings follow the NumPy/SciPy convention
(see https://numpydoc.readthedocs.io/).

Notes
-----
- This file is intended as a library of helper functions, **not** a standalone script.
- Type hints follow PEP 484/PEP 604, with postponed evaluation enabled via
  ``from __future__ import annotations``.
"""
# Standard library
from __future__ import annotations
from typing import Any, Callable 
__docformat__ = "numpy"  # optional, descriptive only
import os, math, sys, warnings, inspect, zipfile

# Third-party
import numpy as np
from tqdm.auto import tqdm, trange
from scipy import special
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Custom packages
try:
    import agama
except ImportError:
    raise RuntimeError("agama package required for potential calculation.")

# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1, length=1, velocity=1)
# agama.setNumThreads(48)  # Explicitly set the number of threads

def _compute_vel_disp_from_Potential(
    pot_for_dynFric_sigma: agama.Potential, 
    grid_r: np.ndarray | None = None
) -> Callable:
    """
    Compute the velocity dispersion profile for the host potential.

    Parameters
    ----------
    pot_for_dynFric_sigma : agama.Potential
        Potential model used to compute the velocity dispersion profile. 
        Ideally an axisymmetric/spherical symmetric model.
    grid_r : optional, np.ndarray
        Grid to compute dispersion function
        
    Returns
    -------
    Callable
        A function that computes the velocity dispersion at a given radius.
    """
    if grid_r is None:
        grid_r = np.logspace(-1, 2, 16)  # grid from 0.1 to 100 kpc
    
    try:
        df_host = agama.DistributionFunction(type='quasispherical', potential=pot_for_dynFric_sigma)
        grid_sig = agama.GalaxyModel(pot_for_dynFric_sigma, df_host).moments(
            np.column_stack((grid_r, grid_r * 0, grid_r * 0)), dens=False, vel=False, vel2=True)[:, 0] ** 0.5
        logspl = agama.Spline(np.log(grid_r), np.log(grid_sig))
        return lambda r: np.exp(logspl(np.log(r)))
    
    except:
        print('Using precomputed velocity dispersion profiles.')
        # Fallback to a default velocity dispersion profile
        grid_sig_init = np.array([158.34386609, 200.12076947, 208.35638186, 207.53478107,
                                  197.97276146, 195.18822847, 188.6893688, 183.74527079,
                                  187.35960162, 193.26190609, 173.27866017, 143.68049751,
                                  132.84412575, 121.76024275, 106.50314755, 104.28241804])
        logspl_init = agama.Spline(np.log(grid_r), np.log(grid_sig_init))
        
        return lambda r: np.exp(logspl_init(np.log(r)))

def _dynamical_friction_acceleration(
    pos: numpy.ndarray, vel: numpy.ndarray, 
    pot_host: agama.Potential, 
    mass: float, 
    sigma_r_func: Callable, 
    t: float = 0
) -> np.ndarray:
    """
    Compute the dynamical friction acceleration for a point mass in the host galaxy.

    Parameters
    ----------
    pos : np.ndarray
        Position vector of the satellite.
    vel : np.ndarray
        Velocity vector of the satellite.
    pot_host : agama.Potential
        Potential of the host galaxy.
    mass : float
        Mass of the satellite.
    sigma_r_func : Callable
        Precomputed function for velocity dispersion at a given radius.
    t : float, optional
        Time at which to evaluate the potential (default: 0).

    Returns
    -------
    np.ndarray
        Acceleration vector due to dynamical friction.
    """
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    rho = pot_host.density(pos, t=t)
    coulombLog = 3.0
    X = v / (2**0.5 * sigma_r_func(r))
    return -vel / v * (4 * np.pi * agama.G**2 * mass * rho * coulombLog *
                       (special.erf(X) - 2 / np.pi**0.5 * X * np.exp(-X**2)) / v**2)

def integrate_orbit_with_dynamical_friction(
    ic: np.ndarray, 
    pot_host: agama.Potential, 
    mass: float, 
    time_total: float,
    time_end: float, 
    pot_for_dynFric_sigma: agama.Potential, 
    trajsize: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the orbit of a massive particle in the host galaxy, accounting for dynamical friction.
    Dynamical friction cam be turned off with mass=0. 

    Parameters
    ----------
    ic : np.ndarray
        Initial conditions (position and velocity) of the satellite.
    pot_host : agama.Potential
        Potential of the host galaxy.
    mass : float
        Fixed mass of the satellite.    
    time_total : float
        Total time to integrate the orbit.
    time_end : float
        End time of the simulation.
    pot_for_dynFric_sigma : agama.Potential
        Potential model used to compute the velocity dispersion profile.
    trajsize : int, optional
         Trajsize of the orbit to construct. default : 0, saves all adaptive time steps. 
         
    Returns
    -------
    tuple
        A tuple containing the times and trajectory of the satellite.
    """
    time_sat, orbit_sat = agama.orbit(ic=ic, potential=pot_host, 
                                      time=-time_total, 
                                      timestart=time_end, 
                                      trajsize=trajsize)
    if mass == 0:
        return time_sat, orbit_sat
        
    # Precompute the velocity dispersion profile
    sigma_r_func = _compute_vel_disp_from_Potential(pot_for_dynFric_sigma)
    
    def equations_of_motion(t: float|np.ndarray, xv: np.ndarray) -> np.ndarray:
        """returns vel & computed accelerations"""
        pos, vel = xv[:3], xv[3:6]
        acc = pot_host.force(pos, t=t) + _dynamical_friction_acceleration(pos, vel, pot_host, mass, sigma_r_func, t=t)
        return np.hstack((vel, acc))
    
    # Solve the ODE using a more efficient integrator (DOP853 or LSODA)
    sol = solve_ivp(equations_of_motion, 
                    [time_end, time_end - time_total], ic, 
                    method='DOP853', dense_output=True,
                    rtol=1e-8,  # High relative tolerance
                    atol=1e-10,  # High absolute tolerance
                   )
    
    print('Dynamical Friction is turned on')
    
    # Return the internal time steps and corresponding trajectory
    return time_sat, sol.sol(time_sat).T

def _get_prog_GalaxyModel(
    initmass: float, 
    scaleradius: float, 
    prog_pot_kind: str,  
    **kwargs: Any) -> tuple[agama.Potential, agama.DistributionFunction]:
    """
    Create a satellite potential and distribution function based on the specified profile.

    Parameters
    ----------
    initmass : float
        Initial mass of the satellite.
    scaleradius : float
        Scale radius of the satellite.
    prog_pot_kind : str
        Type of potential profile. Must be one of ["King", "Plummer", "Plummer_withRcut"].
    **kwargs : dict
        Additional parameters for the potential profile. For 'King' profile, these include:
        - W0 : float, optional
            Central dimensionless potential for King profile (default: 3).
        - trunc : float, optional
            Truncation parameter for King profile (default: 1).

    Returns
    -------
    tuple
        A tuple containing the satellite potential and distribution function.
    """
    if prog_pot_kind.lower() == 'plummer': 
        pot_sat = agama.Potential(type='Plummer', mass=initmass, scaleRadius=scaleradius)
    
    elif prog_pot_kind.lower() == 'plummer_withrcut': 
        pot_sat = agama.Potential(type='Spheroid', mass=initmass, scaleRadius=scaleradius, 
                                  outerCutoffRadius=4 * scaleradius,  # Set the cutoff radius
                                  gamma=0,   # Core-like behavior (Plummer has a flat core)
                                  beta=5,    # Outer slope (Plummer falls off as r^-5)
                                  alpha=2,   # Inner slope (density falls off like r^-2.5)
                                  cutoffStrength=3  # Controls sharpness of the cutoff
                                 )

    elif prog_pot_kind.lower() == 'king':
        # Use kwargs to get W0 and trunc, or default values if not provided
        W0 = kwargs.get('W0', 3)  # Default W0 = 3
        trunc = kwargs.get('trunc', 1)  # Default trunc = 1
        # print(f'King Profile with W0={W0:.2f}, trunc={trunc:.2f}')
        pot_sat = agama.Potential(type='King', mass=initmass, 
                                  scaleRadius=scaleradius, 
                                  W0=W0, trunc=trunc)
    
    else:
        raise ValueError(f"Unsupported progenitor potential kind: {prog_pot_kind}")
    
    return pot_sat, agama.DistributionFunction(type='quasispherical', potential=pot_sat)

def _find_prog_pot_Nparticles(
    xv: np.ndarray, 
    prog: np.ndarray, 
    masses: np.ndarray | None = None, 
    **potential_kwargs: dict)-> tuple[np.ndarray, int]:
    """
    Define the Progenitor potential from N particle system.

    Parameters
    ----------
    xv : np.ndarray (N,6)
        Phase-space coordinates [x,y,z,vx,vy,vz] for N particles.
    prog : np.ndarray (6, )
        COM phase-space of the progenitor at current time.
    masses : np.ndarray (N,), optional
        Particle masses (uniform if None).
    potential_kwargs : dict
        Additional arguments for agama.Potential.

    Returns
    -------
    xv_prog : np.ndarray (6,)
        Phase-space coordinates of the most bound particle.
    prog_idx : int
        Index of the progenitor particle in the original array.
    """
    xv = np.asarray(xv)
    assert xv.ndim == 2 and xv.shape[1] == 6, "Input must be Nx6 array."

    N = len(xv)
    if masses is None:
        masses = np.ones(N) / N
    else:
        masses = np.asarray(masses)
        assert len(masses) == N, "Masses must match particle count."

    # Shift all particles to the centroid's frame
    xv_rel = xv.copy()
    xv_rel -= prog
    
    # Set up potential using the shifted positions
    pot_params = {
        'type': 'multipole',
        'particles': (xv_rel[:, :3], masses),
        'symmetry': 's',
    }
    pot_params.update(potential_kwargs)

    # Compute energies and find the most bound particle
    pot_sat = agama.Potential(**pot_params)
    
    return pot_sat, prog

################################################################################
######################## Restricted Nbdoy Method #####################
################################################################################

def run_restricted_Nbody(
    pot_host: agama.Potential, 
    initmass: float,
    sat_cen_present: np.ndarray | tuple, 
    scaleradius: float | None = None, 
    num_particles: int = int(1e4), 
    prog_pot_kind: str = 'King', 
    xv_init: np.ndarray | None = None,
    dynFric: bool = False, 
    pot_for_dynFric_sigma = None,
    time_total: float = 3.0, 
    time_end: float = 13.78, 
    step_size: int = 10, 
    save_rate: int = 300,
    trajsize_each_step: int = 10, 
    verbose: bool = False, 
    accuracy_integ: float = 1e-6, 
    **kwargs: Any) -> dict[str, np.ndarray]:
    """
    Simulate the restricted/collisionless N-body system of a satellite orbiting 
    a host potential with/without dynamical friction (basically 0 for GCs).
    
    Parameters
    ----------
    pot_host : agama.Potential
        The gravitational potential of the host system.
    initmass : float
        Initial mass of the satellite (must be positive).    
    sat_cen_present : np.ndarray or tuple of shape (6,)
        Present-day position and velocity of the satellite center.
        If xv_init is provided, this is assumed to be the progenitor position. 
    scaleradius : float, optional
        Initial scale radius of the satellite (must be positive). 
        Ignored if xv_init is provided.       
    num_particles : int, default 10000
        Number of stream particles (must be positive).   
    prog_pot_kind : {'King', 'Plummer', 'Plummer_withRcut'}, default 'King'
        Progenitor initial potential profile. Defaults to 'King' in case of 
        parameter error.
    xv_init : np.ndarray of shape (N, 6), optional
        If provided:
        
        - Uses these particles directly (no rewinding/sampling)
        - Code integrates forward to reach time_end based on time_total, 
          assuming the initial time is time_end - time_total
        - Ignores sat_cen_present/scaleradius parameters
        - Computes progenitor from this array    
    dynFric : bool, default False
        Whether to integrate the initial orbit with dynamical friction enabled.
    pot_for_dynFric_sigma : agama.Potential, optional
        Potential model used to compute the moment (velocity dispersion profile).
        If None, uses precomputed & saved profiles in the code.
    time_total : float, default 3.0
        Total time to rewind the satellite's orbit in Gyr (must be non-negative).
    time_end : float, default 13.78
        The end time of the simulation in Gyr.
    step_size : int, default 10
        Number of time steps to group together for each iteration (must be positive).
    save_rate : int, default 300
        Number of time steps between integrated outputs (must be at least 1). 
        If save_rate = 1, only the final time step is output.
    trajsize_each_step : int, default 10
        Number of trajectory points saved per integration step (must be positive).
    verbose : bool, default False
        If True, print detailed output from the orbit integrator.
    accuracy_integ : float, default 1e-6
        Accuracy parameter for the orbit integrator (must be positive).
    **kwargs : dict, optional
        Additional parameters for the progenitor potential profile. For 'King' 
        profile, these include:
        
        - W0 : float, default 3
            Central dimensionless potential for King profile.
        - trunc : float, default 1
            Truncation parameter for King profile.
    
    Returns
    -------
    dict
        Dictionary containing simulation results with keys:
        
        - 'save_times' : np.ndarray
            Array of times at which interpolated results are evaluated.
        - 'orbit_sat_interp' : np.ndarray
            Interpolated satellite orbit evaluated at `save_times`.
        - 'phase_space_interp' : np.ndarray
            Interpolated phase-space coordinates of all particles evaluated at `save_times`.
        - 'bound_mass_interp' : np.ndarray
            Interpolated bound mass evaluated at `save_times`.
    
    Notes
    -----
    This function performs a restricted N-body simulation where test particles 
    (representing the satellite's stars) move in the combined potential of the 
    host galaxy and the satellite itself. The satellite's orbit can optionally 
    include dynamical friction effects.
    
    The simulation workflow depends on the xv_init parameter:
    
    - If xv_init is None: The satellite orbit is rewound from sat_cen_present 
      by time_total, particles are sampled from the chosen potential profile, 
      then integrated forward.
    - If xv_init is provided: The given particles are integrated forward directly 
      from time (time_end - time_total) to time_end.
    """
    # Assertions to check parameter validity
    assert hasattr(pot_host, 'potential') or isinstance(pot_host, object), "pot_host must be a valid agama.Potential object."
    assert isinstance(initmass, (float, int)) and initmass > 0, "initmass must be a positive number."
    assert isinstance(sat_cen_present, (np.ndarray, tuple)) and len(sat_cen_present) == 6, \
    "sat_cen_present must be a NumPy array or tuple of shape (6,)."
    assert isinstance(dynFric, bool), "dynFric must be a boolean."
    assert pot_for_dynFric_sigma is None or hasattr(pot_for_dynFric_sigma, 'potential'), \
    "pot_for_dynFric_sigma must be None or a valid agama.Potential object."    

    # --- Input validation ---
    xv = np.copy(xv_init) if xv_init is not None else None
    
    if xv is None: ##args only required if Nbody particles are not parsed.
        assert isinstance(scaleradius, (float, int)) and scaleradius > 0, "scaleradius must be a positive number."
        assert isinstance(num_particles, int) and num_particles > 0, "num_particles must be a positive integer."
        assert prog_pot_kind in ["King", "Plummer", "Plummer_withRcut"], \
            "prog_pot_kind must be one of ['King', 'Plummer', 'Plummer_withRcut']."

    else:
        # Conditional checks for xv_init vs sat_cen_present/scaleradius
        assert xv.ndim == 2 and xv.shape[1] == 6, "xv_init must be Nx6 array"
        assert xv.shape[0] >= 1, "Need at least 1 particle in xv_init"
        # Warn if unused parameters are provided
        if scaleradius is not None:
            warnings.warn("scaleradius ignored when xv_init provided", UserWarning)
        
    assert isinstance(time_total, (float, int)) and time_total >= 0, "time_total must be non-negative."
    assert isinstance(time_end, (float, int)), "time_end must be a numeric value."
    assert isinstance(step_size, int) and step_size > 0, "step_size must be a positive integer."
    assert isinstance(save_rate, int) and save_rate >= 1, "save_rate must be at least 1."
    assert isinstance(trajsize_each_step, int) and trajsize_each_step > 0, "trajsize_each_step must be a positive integer."
    assert isinstance(verbose, bool), "verbose must be a boolean."
    assert isinstance(accuracy_integ, (float, int)) and accuracy_integ > 0, "accuracy_integ must be a positive number."

    # Function implementation goes here...    
    
    # Integrate the orbit of the progenitor from its present-day posvel (at time t_end) back in time
    # for an interval time_total, storing the trajectory at every timestep of the ODE integrator
    # (as specified by trajsize=0) to ensure an accurate interpolation
    
    
    if xv is None:
        # Rewind the progenitor location with or without DF  
        time_sat, orbit_sat = integrate_orbit_with_dynamical_friction(sat_cen_present, pot_host, initmass if dynFric else 0,
                                                                      time_total, time_end, pot_for_dynFric_sigma)        
        print(f'Prog orbit resolution: {len(time_sat)} steps.')

        # Reverse time arrays to make them increasing in time
        time_sat  = time_sat[::-1]
        orbit_sat = orbit_sat[::-1]

        # create IC's for the progenitor based on the progenitor mass and scale radius.   
        # Use the default King profile as an overall better choice
        pot_sat, df_sat = _get_prog_GalaxyModel(initmass, scaleradius, prog_pot_kind, **kwargs)
        xv, masses = agama.GalaxyModel(pot_sat, df_sat).sample(num_particles) # phase-space samples of the distribution func.
        
        # Displace the sampled satellite particles with initial progenitor conditions
        xv += orbit_sat[0]
    else: 
        # Use existing particles and find progenitor
        masses = np.full(len(xv), initmass/len(xv))  # Distribute mass equally
        # The code only works in integrating forward with starting with init conds after defining a Prog w/o dyn Fric.
        pot_sat, prog_init = _find_prog_pot_Nparticles(xv, sat_cen_present, masses=masses) #find the prog and construct pot.
        # integrate the progenitor orbit forward
        time_sat, orbit_sat = agama.orbit(ic=prog_init, potential=pot_host, time=time_total, timestart=time_end-time_total, trajsize=0)
            
    bound_mass = [initmass]
    num_steps = int(np.floor(len(time_sat) / step_size))  # Number of iterations
    
    print(f'No. of timesteps to update Progenitor potential: {num_steps}.')
    
    # Initialize lists to store phase-space coordinates and times
    phase_space_list = []  # List to store particle trajectories
    time_list = []         # List to store corresponding times
    
    # Iterate through the time steps
    try:
        from tqdm.auto import trange
        loop_iter = trange(num_steps+1, desc="Simulating Restricted N-body")
    except ImportError:
        loop_iter = range(num_steps+1)
        
    for i in loop_iter:
        start_index = i * step_size
        end_index = (i + 1) * step_size

        if end_index >= len(time_sat):
            end_index = -1
        
        # Restricted N-body setup from begin to end time step.       
        time_begin = time_sat[start_index]
        time_end_step = time_sat[end_index]
        time_step = time_end_step - time_begin  # The actual integration time
        
        # Avoid floating point precision issues & ensure last step is not duplicated.
        if math.isclose(time_sat[-1], time_begin, abs_tol=1e-6) or time_sat[-1] < time_begin:
            bound_mass.append(np.sum(masses[bound])) ##to ensure same shape of the bound_mass arr at the end time. 
            continue  # Skip this iteration if time_step is too small
        
        # Initialize the time-dependent total potential (host + moving sat) on this time interval
        pot_sat_moving = agama.Potential(potential=pot_sat, center=np.column_stack([time_sat, orbit_sat]))
        pot_total = agama.Potential(pot_host, pot_sat_moving)
        
        # Adaptive-integrate the particle trajectories with high resolution (Save rate set by trajsize_each_step)
        traj = agama.orbit(
            ic=xv, potential=pot_total, trajsize=trajsize_each_step, verbose=verbose,
            time=time_step, timestart=time_begin, accuracy=accuracy_integ
        )
        
        # traj is a list of N tuples: [(time, phase_space), ...] for N particle.
        # Extract time and phase-space coordinates
        traj_times = traj[0][0]  # Shape: (trajsize_each_step,)
        traj_phase_space = np.array([t[1] for t in traj])  # Shape: (N, trajsize_each_step, 6)
        
        # Save the trajectories and corresponding times
        phase_space_list.append(traj_phase_space)  
        time_list.append(traj_times)  
        
        # Update xv to the final position of this integration step
        xv = traj_phase_space[:, -1]  # Shape: (N, 6)
        
        # Update the potential of the satellite (using a spherical monopole approximation),
        # Compute satellite-relative coordinates
        xv_rel = xv - orbit_sat[end_index]        
        try:
            pot_sat = agama.Potential(type='multipole', particles=(xv_rel[:, 0:3], masses), symmetry='s')
        except Exception as e:
            print(f'Sat pot failed at {time_sat[end_index]:.2f} Gyr.')
            return (time_sat, orbit_sat), xv, bound_mass
        
        # Determine which particles remain bound to the satellite
        bound = pot_sat.potential(xv_rel[:,0:3]) + 0.5 * np.sum(xv_rel[:,3:6]**2, axis=1) < 0
        bound_frac = bound.sum() / xv.shape[0]
        
        print(f'\rBound Frac: {bound_frac:.4f} at T: {time_sat[end_index]:.2f} Gyr.', end='\r', flush=True)
        # sys.stdout.write(f'\rBound Frac: {bound_frac:.4f} at T: {time_sat[end_index]:.2f} Gyr.')
        # sys.stdout.flush()
        bound_mass.append(np.sum(masses[bound]))
        
        if end_index == -1:
            break # Exit the loop immediately.
    # Final time step bound fraction.
    print(f'\rBound Frac: {bound_frac:.4f} at T: {time_sat[end_index]:.2f} Gyr.')        
    
    # Combine all saved trajectories into a single 3D array using np.vstack for faster concatenation
    bound_mass = np.hstack(bound_mass)
    phase_space = np.hstack(phase_space_list)  # Shape: (num_steps * trajsize_each_step, N, 6)
    time_traj = np.hstack(time_list)           # Shape: (num_steps * trajsize_each_step,)
    
    # Interpolate bound mass, phase-space coordinates, and satellite orbit
    save_times = np.linspace(time_end - time_total, time_end, save_rate) if save_rate > 1 else time_end
    
    # try saving the interpolated bound mass as well.
    try:
        flag=True
        time_steps_bound_comp = np.hstack([time_sat[::step_size], time_sat[-1]]) # we start at 0 --> -1 step.
        # Figure out unique times for interpolation of saved trajectory.
        _, unique_inds = np.unique(time_steps_bound_comp, return_index=True)
        bound_mass_interp = interp1d(time_steps_bound_comp[unique_inds], bound_mass[unique_inds], 
                                     kind='cubic', fill_value="extrapolate")
        
        bound_mass_extract = np.minimum(bound_mass_interp(save_times), bound_mass_interp(save_times)[0])
        
    except Exception as e:
        print(e)
        flag=False
        print(f'Time shape: {time_sat[::step_size].shape}, bound_mass shape: {bound_mass.shape}')
        print(f'Returning uninterpolated mass')
    
    # Figure out unique times for interpolation of saved trajectory.
    _, unique_inds = np.unique(time_traj, return_index=True)

    phase_space_interp = interp1d(time_traj[unique_inds], phase_space[:, unique_inds], 
                                  kind='cubic', axis=1, fill_value="extrapolate")

    orbit_sat_interp = interp1d(time_sat, orbit_sat, kind='cubic', axis=0, fill_value="extrapolate")

    # Evaluate & return interpolated results at save_times.        
    return  {'times': save_times, 
             'prog_xv': orbit_sat_interp(save_times), 
             'part_xv': phase_space_interp(save_times), 
             'bound_mass': bound_mass_extract if flag else (time_sat[::step_size], bound_mass), 
             # 'pot_sat': pot_sat,
            }

################################################################################
############################## PARTICLE SPRAY Method ###########################
################################################################################
def _get_jacobi_rad_vel_mtx(
    pot_host: agama.Potential, 
    orbit_sat: np.ndarray, 
    mass_sat: float, 
    G: float = agama.G,
    t: float | np.ndarray = 0, 
    eigenvalue_method: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Jacobi radius, velocity offset, and local coordinate system rotation matrices.
    
    Parameters
    ----------
    pot_host : agama.Potential
        Gravitational potential of the host system.
    orbit_sat : array_like of shape (N, 6)
        Satellite orbit positions and velocities in format [x, y, z, vx, vy, vz].
        Units: [kpc, km/s].
    mass_sat : float
        Satellite mass in solar masses [M_sun].
    G : float, default agama.G
        Gravitational constant.
    t : float or array_like of shape (N,), default 0
        Time(s) at which to evaluate potential derivatives. If float, same time
        is used for all orbit points. If array, must match orbit_sat length.
    eigenvalue_method : bool, default True
        Method for computing Jacobi radius:
        
        - True: Use tidal tensor eigenvalues (more accurate)
        - False: Use radial potential derivative approximation (faster)
    
    Returns
    -------
    r_jacobi : np.ndarray of shape (N,)
        Jacobi radii in kpc. The tidal radius within which the satellite
        can retain bound material.
    v_jacobi : np.ndarray of shape (N,)
        Velocity scales in km/s. Characteristic velocity scale for the
        satellite at the Jacobi radius.
    R : np.ndarray of shape (N, 3, 3)
        Rotation matrices transforming from global to satellite-aligned 
        coordinate system. Each matrix R[i] has rows corresponding to:
        
        - Row 0: radial direction (satellite center to host center)
        - Row 1: azimuthal direction (perpendicular to radial, in orbital plane)  
        - Row 2: angular momentum direction (perpendicular to orbital plane)
    
    Notes
    -----
    The Jacobi radius represents the tidal truncation radius of the satellite,
    beyond which material becomes unbound due to host tidal forces.
    
    **Jacobi radius calculation:**
    
    - Eigenvalue method: :math:`r_j = \\left(\\frac{GM_{sat}}{|\\lambda_{max} + \\Omega^2|}\\right)^{1/3}`
    - Approximation method: Uses radial potential derivative
    
    where :math:`\\lambda_{max}` is the maximum eigenvalue of the tidal tensor and 
    :math:`\\Omega` is the orbital frequency.
    
    **Velocity scale:**
    :math:`v_j = \\Omega \\times r_j`
    
    **Coordinate system:**
    The rotation matrices define a satellite-centric coordinate system that
    co-moves and co-rotates with the satellite, useful for analyzing tidal
    streams and bound/unbound particle dynamics.
    """    
    orbit_sat = np.asarray(orbit_sat)
    N = len(orbit_sat)
    pos, vel = orbit_sat[:, :3], orbit_sat[:, 3:6]
    
    # Precompute geometric quantities
    r = np.linalg.norm(pos, axis=1)
    r_sq = r**2 + 1e-50

    # Angular velocity squared
    L = np.cross(pos, vel)
    L_mag = np.linalg.norm(L, axis=1)
    Omega_sq = (L_mag / r_sq)**2

    # Get potential derivatives
    der2 = pot_host.eval(pos, der=True, t=t) #Shape (N,6): [d²Φ/dx², d²Φ/dy², d²Φ/dz², d²Φ/dxdy, d²Φ/dydz, d²Φ/dzdx]

    if eigenvalue_method:
        # Construct tidal tensor (3x3 for each point)
        tidal_tensor = np.zeros((N, 3, 3))
        tidal_tensor[:, 0, 0] = der2[:, 0]  # xx
        tidal_tensor[:, 1, 1] = der2[:, 1]  # yy
        tidal_tensor[:, 2, 2] = der2[:, 2]  # zz
        tidal_tensor[:, 0, 1] = tidal_tensor[:, 1, 0] = der2[:, 3]  # xy
        tidal_tensor[:, 1, 2] = tidal_tensor[:, 2, 1] = der2[:, 4]  # yz
        tidal_tensor[:, 0, 2] = tidal_tensor[:, 2, 0] = der2[:, 5]  # zx

        eigenvalues = np.linalg.eigvalsh(tidal_tensor)
        lambda_tidal = eigenvalues[:, -1]  # Most compressive eigenvalue
        denominator = lambda_tidal + Omega_sq
    else:
        x, y, z = pos.T
        d2Phi_dr2 = -(x**2*der2[:, 0] + y**2*der2[:, 1] + z**2*der2[:, 2] +
                     2*x*y*der2[:, 3] + 2*y*z*der2[:, 4] + 2*z*x*der2[:, 5]) / r_sq
        denominator = Omega_sq - d2Phi_dr2

    # Jacobi radius and velocity
    r_jacobi = (G * mass_sat / abs(denominator))**(1/3)
    v_jacobi = np.sqrt(Omega_sq) * r_jacobi

    # Construct rotation matrices (row-based)
    R = np.zeros((N, 3, 3))
    e_r = pos / (r[:, None] + 1e-50)    # Radial direction (row 0)
    e_L = L / (L_mag[:, None] + 1e-50)  # Angular momentum (row 2)
    
    # Azimuthal direction (row 1 = e_r × e_L)
    e_phi = np.cross(e_L, e_r, axis=1)
    e_phi_norm = np.linalg.norm(e_phi, axis=1, keepdims=True)
    e_phi = np.divide(e_phi, e_phi_norm, where=e_phi_norm!=0)

    R[:, 0, :] = e_r
    R[:, 1, :] = e_phi
    R[:, 2, :] = e_L

    return r_jacobi, v_jacobi, R

def create_ic_particle_spray_chen2025(
    orbit_sat: np.ndarray, 
    mass_sat: float, 
    rj: np.ndarray, 
    R: np.ndarray, 
    G: float = agama.G
) -> np.ndarray:
    """
    Create initial conditions using Chen+2025 method incorporating:
    - Realistic escape velocity distributions
    - Correlated phase-space offsets
    - Angular dispersion models
    
    Parameters
    ----------
    orbit_sat : ndarray (N, 6)
        Satellite orbit positions/velocities [kpc, kpc/Myr]
    mass_sat : float
        Satellite mass [Msun]
    rj : ndarray (N,)
        Jacobi radii at each orbit point [kpc]
    R : ndarray (N,3,3)
        Rotation matrices to satellite frame at each point
    G : float, default agama.G
        Gravitational constant. 
        
    Returns
    -------
    ic_stream : ndarray (2N,6)
        Initial conditions for stream particles
    
    Notes
    -----
    - Covariance matrix derived from N-body simulations
    - Escape velocity calculated as v_esc = sqrt(2GM/r)
    - Angular parameters represent spherical coordinate offsets
    """
    
    N = len(orbit_sat)
    
    # Expand for leading/trailing arms (2 particles per orbit point)
    # assign positions and velocities (in the satellite reference frame) of particles
    # leaving the satellite at both lagrange points (interleaving positive and negative offsets).
    # R = np.repeat(R, 2, axis=0)  # FIX: Match 2N shape
    r_tidal = np.repeat(rj, 2) #* np.tile([1, -1], N)
    
    # Covariance matrix from Chen+2025 calibration
    mean = np.array([1.6, -30, 0, 1, 20, 0]) # [r, phi, theta, vr, alpha, beta]
    cov = np.array([
        [0.1225,   0,   0, 0, -4.9,   0],
        [     0, 529,   0, 0,    0,   0],
        [     0,   0, 144, 0,    0,   0],
        [     0,   0,   0, 0,    0,   0],
        [  -4.9,   0,   0, 0,  400,   0],
        [     0,   0,   0, 0,    0, 484],
    ]) # Units: [kpc, deg, deg, km/s, deg, deg]
    
    # Generate correlated offsets
    rng = np.random.default_rng(0)
    posvel = rng.multivariate_normal(mean, cov, size=2*N)
    
    # Convert to physical quantities
    Dr = posvel[:, 0] * r_tidal  # Radial offset [kpc]
    phi = np.deg2rad(posvel[:, 1])  # Azimuth [rad]
    theta = np.deg2rad(posvel[:, 2])  # Polar [rad]
    
    # Escape velocity at each offset radius
    v_esc = np.sqrt(2 * G * mass_sat / np.abs(Dr))
    Dv = posvel[:, 3] * v_esc  # Velocity magnitude [kpc/Myr]
    
    # Velocity angles
    alpha = np.deg2rad(posvel[:, 4])  # Velocity azimuth [rad]
    beta = np.deg2rad(posvel[:, 5])  # Velocity polar [rad]
    
    # Convert to Cartesian offsets
    dx = Dr * np.cos(theta) * np.cos(phi)
    dy = Dr * np.cos(theta) * np.sin(phi)
    dz = Dr * np.sin(theta)
    dvx = Dv * np.cos(beta) * np.cos(alpha)
    dvy = Dv * np.cos(beta) * np.sin(alpha)
    dvz = Dv * np.sin(beta)
    
    # Construct offset arrays
    offset_pos = np.column_stack([dx, dy, dz]) # position and velocity of particles in the reference frame
    offset_vel = np.column_stack([dvx, dvy, dvz]) # centered on the progenitor and aligned with its orbit
    
    # Transform to host frame
    ic_stream = np.tile(orbit_sat, 2).reshape(2 * N, 6)   # same but in the host-centered frame
    
    # trailing arm
    ic_stream[::2, 0:3] += np.einsum('ni,nij->nj', offset_pos[::2], R)
    ic_stream[::2, 3:6] += np.einsum('ni,nij->nj', offset_vel[::2], R)
    
    # leading arm
    ic_stream[1::2, 0:3] += np.einsum('ni,nij->nj', -offset_pos[1::2], R)
    ic_stream[1::2, 3:6] += np.einsum('ni,nij->nj', -offset_vel[1::2], R)
        
    return ic_stream

def create_ic_particle_spray_fardal2015(
    orbit_sat: np.ndarray, 
    rj: np.ndarray, 
    vj: np.ndarray, 
    R: np.ndarray, 
    gala_modified: bool = True
) -> np.ndarray:
    """
    Create initial conditions using Fardal+2015 particle spray method.
    
    Implements the "classic" particle spray technique with parameters from:
    Fardal, M. A., et al. (2015). MNRAS, 452(1), 301-317.
    
    Parameters
    ----------
    orbit_sat : ndarray (N,6)
        Satellite orbit positions/velocities [kpc, kpc/Myr]
    rj : ndarray (N,)
        Jacobi radii at each orbit point [kpc]
    vj : ndarray (N,)
        Velocity scales at each orbit point [kpc/Myr]
    R : ndarray (N,3,3)
        Rotation matrices to satellite frame at each point
    gala_modified : bool, optional
        Use Gala's modified dispersion parameters (default: True)
    
    Returns
    -------
    ic_stream : ndarray (2N,6)
        Initial conditions for stream particles
        
    Notes
    -----
    - Generates two particles per orbit point (leading/trailing arms)
    - Position/velocity offsets follow asymmetric Gaussian distributions
    - Rotation matrices must be pre-expanded to 2N before calling
    """
    N = len(rj)
    # Expand quantities for leading/trailing arms
    rj = np.repeat(rj, 2) * np.tile([1, -1], N)  # Alternate signs for arms
    vj = np.repeat(vj, 2) * np.tile([1, -1], N)
    R = np.repeat(R, 2, axis=0)  # Critical: match 2N shape
    
    # Configure dispersion parameters
    params = {
        'mean_x': 2.0,
        'disp_x': 0.5 if gala_modified else 0.4,
        'disp_z': 0.5,
        'mean_vy': 0.3,
        'disp_vy': 0.5 if gala_modified else 0.4,
        'disp_vz': 0.5
    }
    
    # Generate offsets in satellite frame
    rng = np.random.default_rng(0)
    rx = rng.normal(loc=params['mean_x'], scale=params['disp_x'], size=2*N)
    rz = rng.normal(scale=params['disp_z'], size=2*N) * rj
    rvy = rng.normal(loc=params['mean_vy'], scale=params['disp_vy'], size=2*N) * vj * (rx if gala_modified else 1)
    rvz = rng.normal(scale=params['disp_vz'], size=2*N) * vj
    
    # Scale radial positions
    rx *= rj
    
    # Construct offset arrays
    offset_pos = np.column_stack([rx, np.zeros(2*N), rz])
    offset_vel = np.column_stack([np.zeros(2*N), rvy, rvz])
    
    # Transform to host frame
    ic_stream = np.tile(orbit_sat, 2).reshape(2*N, 6)
    ic_stream[:, :3] += np.einsum('ni,nij->nj', offset_pos, R)
    ic_stream[:, 3:6] += np.einsum('ni,nij->nj', offset_vel, R)
    
    return ic_stream

def create_stream_particle_spray(
    pot_host: agama.Potential, 
    initmass: float, 
    sat_cen_present: np.ndarray | tuple, 
    scaleradius: float, 
    num_particles: int = int(1e4), 
    prog_pot_kind: str = 'King',
    time_total: float = 3.0, 
    time_end: float = 13.78, 
    save_rate: int = 1,
    dynFric: bool = False, 
    pot_for_dynFric_sigma: agama.Potential | None = None,
    gala_modified: bool = True, 
    add_perturber: dict[str, float] = {'mass': 0, 'scaleRadius': 0.05},
    create_ic_method: Callable = create_ic_particle_spray_chen2025,
    verbose: bool = False, 
    accuracy_integ: float = 1e-6, 
    eigenvalue_method: bool = True,
    **kwargs: Any) -> dict[str, np.ndarray]:
    """
    Construct a stream using the particle-spray method.
    
    Parameters
    ----------
    pot_host : agama.Potential
        The gravitational potential of the host system.
    initmass : float
        Initial mass of the satellite (must be positive).    
    sat_cen_present : np.ndarray or tuple of shape (6,)
        Present-day position and velocity of the satellite center.
        If xv_init is not None, this is assumed to be the progenitor position. 
    scaleradius : float
        Initial scale radius of the satellite (must be positive). 
    num_particles : int, default 10000
        Number of stream particles (must be positive).   
    prog_pot_kind : {'King', 'Plummer', 'Plummer_withRcut'}, default 'King'
        Progenitor initial potential profile. Defaults to 'King' in case of 
        parameter error.
    time_total : float, default 3.0
        Total time to rewind the satellite's orbit in Gyr (must be non-negative).
    time_end : float, default 13.78
        The end time of the simulation in Gyr.    
    save_rate : int, default 1
        Number of snapshots to save during integration (≥1). 
        If 1: only final positions. If >1: interpolates to save_rate evenly spaced times.        
    dynFric : bool, default False
        Whether to integrate the initial orbit with dynamical friction enabled.
    pot_for_dynFric_sigma : agama.Potential, optional
        Potential model used to compute the moment (velocity dispersion profile).
        If None, uses precomputed & saved profiles in the code.    
    gala_modified : bool, default True 
        If True, use modified parameters as in Gala, otherwise the ones from 
        the original paper.
    add_perturber : dict, default {'mass': 0, 'scaleRadius': 0.05}
        Dictionary specifying perturber properties. Must contain:
        
        - 'mass' : float 
            Mass of perturber in M_sun (default: 0)
        - 'scaleRadius' : float 
            Scale radius of perturber in kpc (default: 0.05)
            
        A non-zero mass enables the perturber potential.    
        (No tunable parameters are available - this code should be adapted 
        to specific scenarios).
    create_ic_method : Callable, default create_ic_particle_spray_chen2025
        Function to use for creating initial conditions for particle spray.
        Must have compatible signature with the default method.
    verbose : bool, default False
        If True, print detailed output from the orbit integrator.
    accuracy_integ : float, default 1e-6
        Accuracy parameter for the orbit integrator (must be positive).
    eigenvalue_method : bool, default True
        Whether to use eigenvalues of the tidal tensor. If False, uses the 
        projection along radial direction.
    **kwargs : dict
        Additional parameters for the progenitor potential profile. For 'King' 
        profile, these include:
        
        - W0 : float, default 3
            Central dimensionless potential for King profile.
        - trunc : float, default 1
            Truncation parameter for King profile.
            
    Returns
    -------
    dict
        Dictionary containing simulation results with keys:
        
        - 'times' : np.ndarray of shape (Nsaves,)
            Array of save times in Gyr.
        - 'prog_xv' : np.ndarray of shape (Nsaves, 6)
            Progenitor positions and velocities at each save time.
            Format: [x, y, z, vx, vy, vz] in [kpc, km/s].
        - 'part_xv' : np.ndarray of shape (Nparticles, Nsaves, 6)
            Stream particle states at each save time.
            Contains NaN values where particles weren't yet released.
            Format: [x, y, z, vx, vy, vz] in [kpc, km/s].
    
    Notes
    -----
    This function implements the particle-spray method for generating stellar 
    streams from disrupting satellites. The method progressively releases 
    particles from the satellite as it orbits through the host potential.
    
    The particle spray approach models tidal stripping by:
    
    1. Rewinding the satellite orbit by `time_total` 
    2. Progressively releasing particles at tidal radius as satellite evolves
    3. Tracking all particles forward to present day (`time_end`)
    
    The `create_ic_method` parameter allows customization of the initial 
    condition generation, enabling different spray algorithms or modifications 
    to the Chen et al. (2025) approach.
    """
    # Assertions to check parameter validity
    assert hasattr(pot_host, 'potential') or isinstance(pot_host, object), "pot_host must be a valid agama.Potential object."
    assert isinstance(initmass, (float, int)) and initmass > 0, "initmass must be a positive number."
    assert isinstance(sat_cen_present, (np.ndarray, tuple)) and len(sat_cen_present) == 6, \
    "sat_cen_present must be a NumPy array or tuple of shape (6,)."
    assert isinstance(scaleradius, (float, int)) and scaleradius > 0, "scaleradius must be a positive number."
    assert isinstance(dynFric, bool), "dynFric must be a boolean."
    assert pot_for_dynFric_sigma is None or hasattr(pot_for_dynFric_sigma, 'potential'), \
    "pot_for_dynFric_sigma must be None or a valid agama.Potential object."    
    assert isinstance(time_total, (float, int)) and time_total >= 0, "time_total must be non-negative."
    assert isinstance(time_end, (float, int)), "time_end must be a numeric value."
    assert isinstance(save_rate, int) and save_rate >= 1, "save_rate must be at least 1."
    assert isinstance(verbose, bool), "verbose must be a boolean."
    assert isinstance(accuracy_integ, (float, int)) and accuracy_integ > 0, "accuracy_integ must be a positive number."
    assert isinstance(add_perturber, dict), "add_perturber must be a dictionary"
    assert 'mass' in add_perturber and 'scaleRadius' in add_perturber, \
        "add_perturber must contain 'mass' and 'scaleRadius' keys"
    assert isinstance(add_perturber['mass'], (float, int)) and add_perturber['mass'] >= 0, \
        "add_perturber['mass'] must be non-negative"
    assert isinstance(add_perturber['scaleRadius'], (float, int)) and add_perturber['scaleRadius'] > 0, \
        "add_perturber['scaleRadius'] must be positive"
       
    # number of points on the orbit: each point produces two stream particles (leading and trailing arms)
    N = num_particles//2 + 1

    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at N points
    time_sat, orbit_sat = integrate_orbit_with_dynamical_friction(sat_cen_present, 
                                                                  pot_host, initmass if dynFric else 0,
                                                                  time_total, time_end, pot_for_dynFric_sigma,
                                                                  trajsize=N)        
    # Reverse time arrays to make them increasing in time
    # remove the 0th point (the present-day posvel) and reverse the arrays to make them increasing in time
    time_sat  = time_sat[::-1]
    orbit_sat = orbit_sat[::-1]
    
    # create IC's for the progenitor based on the progenitor mass and scale radius.   
    # Use the default King profile as an overall better choice
    pot_sat, _ = _get_prog_GalaxyModel(initmass, scaleradius, prog_pot_kind, **kwargs)
    pot_sat_moving = agama.Potential(potential=pot_sat, center=np.column_stack([time_sat, orbit_sat]))
    
    # The total potential is now composed of three parts: the host galaxy, the progenitor, and the perturber
    if add_perturber['mass'] > 0:    
        if verbose: print(f'Adding a perturber on a self-consistent orbit with mass: {add_perturber["mass"]:.2e}.')
        # Get the subhalo's phase-space position at impact time from the geometry calculation
        w_subhalo_impact = add_perturber['w_subhalo_impact']  # 6D phase-space at t_impact
        time_impact = add_perturber['time_impact']  # Time of impact (in Gyr, typically negative)
        
        # Step 1: Integrate the subhalo BACKWARDS from impact time to the initial simulation time
        # This gives us the initial conditions for the subhalo at time_init
        w_subhalo_init = agama.orbit(
            potential=pot_host, 
            ic=w_subhalo_impact, 
            time=time_end - time_total - time_impact,  # integrate backwards: from time_impact to time_init
            timestart=time_end + time_impact,  # starting at time_impact
            trajsize=1
        )[1][0]
        
        # Step 2: Integrate the subhalo FORWARD from initial time to present (time_end)
        # Store the trajectory at every internal timestep (trajsize=0)
        traj_perturber = np.column_stack(agama.orbit(
            potential=pot_host, 
            ic=w_subhalo_init, 
            time=time_total,  # integrate forward for the full simulation duration
            timestart=time_end - time_total,  # starting at time_init
            trajsize=0  # save trajectory at all internal timesteps
        ))

        # return time_sat, orbit_sat, traj_perturber        
        # Step 3: Create the moving NFW potential for the perturber
        pot_perturber_moving = agama.Potential(
            type='nfw', 
            mass=add_perturber['mass'], 
            scaleRadius=add_perturber['scaleRadius'], 
            center=traj_perturber
        )
        
        pot_total = agama.Potential(pot_host, pot_sat_moving, pot_perturber_moving)
        
    else:
        pot_total = agama.Potential(pot_host, pot_sat_moving)
        
    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at both Lagrange points
    rj, vj, R = _get_jacobi_rad_vel_mtx(pot_host, orbit_sat, initmass, t=time_sat, eigenvalue_method=eigenvalue_method)

    method_args = {'orbit_sat': orbit_sat, 'mass_sat': initmass,
                   'rj': rj, 'vj': vj, 'R': R, 'gala_modified': gala_modified}

    # Inspect the expected parameters of the create_ic_method
    sig = inspect.signature(create_ic_method)
    expected_params = list(sig.parameters.keys())

    # Filter the arguments to those expected by the create_ic_method
    filtered_args = {k: v for k, v in method_args.items() if k in expected_params}  
    
    # Generate initial conditions using the selected method
    ic_stream = create_ic_method(**filtered_args)
    time_seed = np.repeat(time_sat, 2)

    # Generate save times (array from initial to final integration time)
    save_times = np.linspace(time_end - time_total, time_end - 1e-6, save_rate) if save_rate > 1 else time_end - 1e-6 # clip for floating points
    
    # ======== Modified Orbit Integration ========
    # Configure trajectory saving based on save_rate
    if save_rate > 1:
        if verbose: print(f'Interpolating particle trajs in time.')
        
        # Get dense output for interpolation (uses ODE internal steps)
        trajsize = 0  # 0=save all internal steps    
        # Interpolate progenitor orbit to save times
        prog_interp = interp1d(time_sat, orbit_sat, axis=0, kind='cubic', 
                               fill_value='extrapolate')
        
        prog_xv = prog_interp(save_times)
    
    # Integrate all particle orbits with modified trajsize. Saving interpolators instead of output. 
    result = agama.orbit(
        potential=pot_total, 
        ic=ic_stream[:-2],
        timestart=time_seed[:-2],
        time=time_end-time_seed[:-2],
        dtype=object, # Agama's inbuild trajectory interpolator. 
        # trajsize=trajsize,
        accuracy=accuracy_integ,
        verbose=verbose,
    )    
    
    # print('Return full dense output')    
    # ======== Particle Trajectory from the interpolator ========
    part_xv = np.stack([orbit(save_times) for orbit in result], axis=0)    
    
    return {'times': np.around(save_times, decimals=5) if save_rate > 1 else time_sat,
            'prog_xv': prog_xv if save_rate > 1 else orbit_sat,
            'part_xv': part_xv,}

    # # old interpolator legacy code -> Replaced with Agama's in built agama.Orbit type.     
    # if trajsize == 1:
    #     return {'times': time_sat, 
    #             'prog_xv': orbit_sat,
    #             'part_xv': np.vstack(result[:, 1]),
    #            }
    # else:
    #     ##interpolation code goes here . . . . 
    #     # ======== Particle Trajectory Interpolation ========
    #     # Initialize particle array with NaNs
    #     part_xv = np.full((num_particles, save_rate, 6), np.nan)
    #     for i, particle_orbit in tqdm(enumerate(result)):
            
    #         # Each particle_orbit contains (times, traj) arrays
    #         particle_times = particle_orbit[0]
    #         particle_traj = particle_orbit[1]
            
    #         try:
    #             # Create interpolator for this particle's orbit
    #             interp = interp1d(particle_times, particle_traj, axis=0,
    #                               kind='linear', fill_value=np.nan, bounds_error=False)
    #         except Exception as e:
    #             # pass
    #         #     print(e)
    #         #     print(i, len(particle_times), particle_traj.shape)
    #             return result, time_seed, time_sat, orbit_sat
            
    #         part_xv[i] = interp(save_times)
    #         part_xv[i, -1] = particle_traj[-1]            
    #         # # Optional: Replace NaNs with progenitor track (commented out)
    #         # if np.isnan(part_xv[i]).any():
    #         #     valid_times = ~np.isnan(part_xv[i,:,0])
    #         #     part_xv[i,~valid_times] = prog_xv[~valid_times]            
        
    #     return {'times': save_times,
    #             'prog_xv': prog_xv,
    #             'part_xv': part_xv,}


def generate_stream_coords(
    xv: np.ndarray,
    xv_prog: np.ndarray | list = [],
    degrees: bool = True,
    optimizer_fit: bool = False,
    fit_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert galactocentric phase space (x, y, z, vx, vy, vz)
    into stream-aligned coordinates (phi1, phi2) for single or multiple streams.

    Parameters
    ----------
    xv : np.ndarray, shape (N, 6) or (S, N, 6)
        Particle positions/velocities in galactocentric coordinates. 
        S = number of streams/Time steps, N = number of particles.
    xv_prog : np.ndarray of shape (6,) or (S, 6), optional
        Progenitor phase space vector(s). If not provided, auto-estimated per stream.
    degrees : bool, default True
        If True, angles are returned in degrees, otherwise radians.
    optimizer_fit : bool, default False
        If True, optimize rotation in phi1-phi2 plane per stream.
    fit_kwargs : dict, optional
        Extra args for scipy.optimize.minimize.

    Returns
    -------
    phi1 : np.ndarray
        Stream longitude. Shape (N,) for single stream or (S, N) for multiple.
    phi2 : np.ndarray  
        Stream latitude. Shape (N,) for single stream or (S, N) for multiple.
    """
    xv = np.asarray(xv)
    
    # Normalize input to 3D: (S, N, 6)
    if xv.ndim == 2:
        # Single stream case: (N, 6) -> (1, N, 6)
        xv = xv[None, ...]
        was_single = True
    elif xv.ndim == 3:
        was_single = False
    else:
        raise ValueError(f"xv must be 2D (N, 6) or 3D (S, N, 6), got shape {xv.shape}")
    
    # Multiple streams
    S, N, D = xv.shape
    assert D == 6, "Each particle must have 6 phase-space values"

    # Handle progenitor input
    xv_prog = np.asarray(xv_prog) if len(xv_prog) > 0 else np.array([])
    
    if xv_prog.size == 0:
        # Auto-detect progenitor per stream (closest to median position)
        med = np.median(xv[:, :, :3], axis=1)  # (S, 3)
        dists = np.linalg.norm(xv[:, :, :3] - med[:, None, :], axis=2)  # (S, N)
        idxs = np.argmin(dists, axis=1)  # (S,)
        xv_prog = np.array([xv[s, idxs[s]] for s in range(S)])  # (S, 6)
    else:
        # Normalize progenitor to 2D: (S, 6)
        if xv_prog.ndim == 1:
            if was_single:
                xv_prog = xv_prog[None, :]  # (1, 6)
            else:
                # Broadcast single progenitor to all streams
                import warnings
                warnings.warn(f"Single progenitor provided for {S} streams. "
                            f"Broadcasting same progenitor to all streams.", 
                            UserWarning, stacklevel=2)
                xv_prog = np.tile(xv_prog[None, :], (S, 1))  # (S, 6)
        elif xv_prog.ndim == 2:
            if xv_prog.shape[0] != S:
                raise ValueError(f"Number of progenitors ({xv_prog.shape[0]}) must match number of streams ({S})")
        else:
            raise ValueError(f"xv_prog must be 1D (6,) or 2D (S, 6), got shape {xv_prog.shape}")
    
    assert xv_prog.shape == (S, 6), f"Expected xv_prog shape (S={S}, 6), got {xv_prog.shape}"

    # Compute stream basis vectors for each progenitor
    L = np.cross(xv_prog[:, :3], xv_prog[:, 3:])  # (S, 3)
    L /= np.linalg.norm(L, axis=1)[:, None]       # Normalize (S, 3)

    xhat = xv_prog[:, :3] / np.linalg.norm(xv_prog[:, :3], axis=1)[:, None]  # (S, 3)
    zhat = L
    yhat = np.cross(zhat, xhat)  # (S, 3)

    # Stack into basis matrices: (S, 3, 3)
    R = np.stack([xhat, yhat, zhat], axis=-1)  # (S, 3, 3)

    # Project particles into new frame
    # coords = np.einsum('sni,sij->snj', xv[:, :, :3], R)  # (S, N, 3), slower.
    coords = xv[:, :, :3] @ R  # (S, N, 3) @ (S, 3, 3) -> (S, N, 3), faster.
    xs, ys, zs = coords[..., 0], coords[..., 1], coords[..., 2]
    rs = np.sqrt(xs**2 + ys**2 + zs**2)

    # Compute phi1, phi2
    phi1 = np.arctan2(ys, xs)
    phi2 = np.arcsin(zs / rs)

    # Optional rotation optimization
    theta_opt = None
    if optimizer_fit:
        from scipy.optimize import minimize
        theta_opt = np.empty(S)
        for s in range(S):
            def _cost_fn(theta):
                c, s_ = np.cos(theta), np.sin(theta)
                p1 =  c * phi1[s] - s_ * phi2[s]
                p2 =  s_ * phi1[s] + c * phi2[s]
                return np.sum(p2**2)

            res = minimize(_cost_fn, x0=0.0, **(fit_kwargs or {}))
            theta = res.x.item()
            theta_opt[s] = theta
            c, s_ = np.cos(theta), np.sin(theta)
            phi1[s], phi2[s] = c * phi1[s] - s_ * phi2[s], s_ * phi1[s] + c * phi2[s]

    # Convert to degrees if requested
    if degrees:
        phi1 = np.degrees(phi1)
        phi2 = np.degrees(phi2)
        # if theta_opt is not None:
        #     theta_opt = np.degrees(theta_opt)
    
    # Squeeze back to original dimensionality if input was single stream
    if was_single:
        phi1 = phi1[0]  # (N,)
        phi2 = phi2[0]  # (N,) 
        # if theta_opt is not None:
        #     theta_opt = theta_opt[0]  # scalar
    
    return phi1, phi2, # theta_opt # not sure if theta_opt is ever required. 
################################################################################
############################## OUTPUT ORGANIZATION #############################
################################################################################        
## Stream output reading functions with zarr/ dask if applicable.
# Check for zarr import
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

def read_zarr_group(
    zarr_path: str, 
    group_name: str = "stream_idx_0"
) -> dict[str, np.ndarray] | None:
    """
    Read a specific group from a zarr file and return its contents as a dictionary.
    
    Parameters
    ----------
    zarr_path : str
        Path to the zarr file or directory containing the zarr store.
    group_name : str, default "stream_idx_0"
        Name of the group to read from the zarr store.
    
    Returns
    -------
    dict or None
        Dictionary containing the arrays in the specified group with array names
        as keys and numpy arrays as values. Returns None if an error occurs
        during reading or if the group doesn't exist.
    
    Raises
    ------
    ImportError
        If zarr is not installed.
    
    Notes
    -----
    This function requires the zarr package to be installed. Zarr provides
    efficient storage and retrieval of large numerical arrays with support
    for compression and chunking.
    
    The function will return None instead of raising exceptions for file
    access errors or missing groups to allow graceful error handling in
    calling code.
    
    Examples
    --------
    >>> # Read default group
    >>> data = read_zarr_group("my_stream_data.zarr")
    >>> if data is not None:
    ...     print(f"Available arrays: {list(data.keys())}")
    
    >>> # Read specific group
    >>> stream_data = read_zarr_group("streams.zarr", "stream_idx_1") 
    >>> if stream_data is not None:
    ...     positions = stream_data['positions']
    ...     velocities = stream_data['velocities']
    """
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Reading the output data from streams is configured to work with zarr. "
            "Please install zarr to proceed."
        )

    try:
        # Open the zarr file in read mode
        root = zarr.open(zarr_path, mode="r")

        # Access the group
        if group_name in root:
            group = root[group_name]
        else:
            raise ValueError(f"Group '{group_name}' not found in zarr file.")

        # Read the arrays in the group into a dictionary
        data_dict = {key: group[key][:] for key in group.array_keys()}

        return data_dict

    except Exception as e:
        print(f"Error reading zarr group '{group_name}' from '{zarr_path}': {e}")
        return None
        
def read_zarr_group_from_zip(
    zip_path: str, 
    group_name: str = "stream_idx_0"
) -> dict[str, np.ndarray] | None:
    """
    Read a specific group from a zipped zarr file and return its contents as a dictionary.
    
    Parameters
    ----------
    zip_path : str
        Path to the zipped zarr file (e.g., "data.zarr.zip" or "streams.zip").
    group_name : str, default "stream_idx_0"
        Name of the group to read from the zarr store within the zip archive.
    
    Returns
    -------
    dict or None
        Dictionary containing the arrays in the specified group with array names
        as keys and numpy arrays as values. Returns None if an error occurs
        during reading, if the zip file doesn't exist, or if the group doesn't exist.
    
    Raises
    ------
    ImportError
        If zarr is not installed.
    
    Notes
    -----
    This function requires the zarr package to be installed. It uses zarr's
    built-in support for reading from zip archives, which allows for compressed
    storage of zarr datasets.
    
    The function will return None instead of raising exceptions for file
    access errors, corrupt zip files, or missing groups to allow graceful 
    error handling in calling code.
    
    Zip-based zarr storage is useful for:
    - Archiving completed datasets
    - Reducing storage footprint
    - Distributing zarr data as single files
    
    Examples
    --------
    >>> # Read from a zipped zarr file
    >>> data = read_zarr_group_from_zip("stream_results.zarr.zip")
    >>> if data is not None:
    ...     print(f"Available arrays: {list(data.keys())}")
    ...     positions = data.get('positions')
    
    >>> # Read specific group from zip
    >>> stream_data = read_zarr_group_from_zip("all_streams.zip", "stream_idx_5")
    >>> if stream_data is not None:
    ...     times = stream_data['times']
    ...     particle_coords = stream_data['part_xv']
    """
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Reading the output data from streams is configured to work with zarr. "
            "Please install zarr to proceed."
        )
    store = None
    
    try:
        # Open the zipped zarr file using ZipStore
        store = zarr.ZipStore(zip_path, mode='r')
        root = zarr.open(store, mode="r")

        # Access the group
        if group_name in root:
            group = root[group_name]
        else:
            raise ValueError(f"Group '{group_name}' not found in zipped zarr file.")

        # Read the arrays in the group into a dictionary
        data_dict = {key: group[key][:] for key in group.array_keys()}

        return data_dict

    except Exception as e:
        print(f"Error reading zarr group '{group_name}' from zipped file '{zip_path}': {e}")
        return None
    finally:
        # Close the store to release resources
        if store is not None:
            store.close()
            
def open_zipped_zarr_root(zip_path: str) -> zarr.hierarchy.Group | None:
    """
    Open a zipped zarr file and return its root for quick and easy access.
    
    Parameters
    ----------
    zip_path : str
        Path to the zipped zarr file (e.g., "data.zarr.zip" or "streams.zip").
    
    Returns
    -------
    zarr.hierarchy.Group or None
        The root group of the zarr store, providing access to all contained
        groups and arrays. Returns None if an error occurs during opening.
    
    Raises
    ------
    ImportError
        If zarr is not installed.
    
    Notes
    -----
    This function opens a zipped zarr store and returns the root group object,
    which provides hierarchical access to all data within the archive.
    
    **Important**: The underlying ZipStore remains open as long as the returned
    root object is in use. This is necessary for the root object to function
    properly. The store will be automatically closed when the root object
    is garbage collected.
    
    The root group can be used to:
    - Navigate the zarr hierarchy (groups and subgroups)
    - Access arrays directly: ``root['array_name'][:]``
    - List available groups: ``list(root.groups())``
    - List available arrays: ``list(root.arrays())``
    
    Examples
    --------
    >>> # Open zipped zarr and explore structure
    >>> root = open_zipped_zarr_root("stream_data.zarr.zip")
    >>> if root is not None:
    ...     print("Available groups:", list(root.group_keys()))
    ...     print("Available arrays:", list(root.array_keys()))
    
    >>> # Access specific data through root
    >>> root = open_zipped_zarr_root("results.zip")
    >>> if root is not None:
    ...     stream_group = root['stream_idx_0']
    ...     positions = stream_group['positions'][:]
    ...     times = stream_group['times'][:]
    
    >>> # Direct array access
    >>> root = open_zipped_zarr_root("simulation.zarr.zip")
    >>> if root is not None:
    ...     metadata = root.attrs  # Access root attributes
    ...     data_array = root['my_array'][:]  # Direct array access
    """
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Reading the output data from streams is configured to work with zarr. "
            "Please install zarr to proceed."
        )
    
    store = None
    try:
        # Open the zipped zarr file using ZipStore
        store = zarr.ZipStore(zip_path, mode='r')
        root = zarr.open(store, mode="r")
        return root
    except Exception as e:
        print(f"Error opening zipped zarr file '{zip_path}': {e}")
        return None
    finally:
        # Note: Do not close the store here, as the root object needs it to remain open.
        pass
        
def zarr_to_zip(
    zarr_dir_path: str, 
    zip_path: str | None = None, 
    verbose: bool = True, 
    compression_level: int = 9
) -> str:
    """
    Convert a Zarr directory store to a compressed ZIP file using Zarr's native capabilities.
    
    Parameters
    ----------
    zarr_dir_path : str
        Path to existing Zarr directory store to be converted.
    zip_path : str, optional
        Output ZIP file path. If None, defaults to the same name as zarr_dir_path
        with a .zip extension appended.
    verbose : bool, default True
        If True, show progress output during conversion.
    compression_level : int, default 9
        ZIP compression level:
        
        - 0: No compression (store only)
        - 1-9: Compression levels (1=fastest, 9=best compression)
        
    Returns
    -------
    str
        Path to the created ZIP file.
    
    Raises
    ------
    ValueError
        If zarr_dir_path is not a valid Zarr directory store.
    ImportError
        If zarr is not installed.
    OSError
        If there are file system errors during conversion.
    
    Notes
    -----
    This function creates a compressed archive of a Zarr directory store,
    making it portable and reducing storage requirements. The resulting
    ZIP file can be opened using `open_zipped_zarr_root()` or 
    `read_zarr_group_from_zip()`.
    
    **Compression benefits:**
    - Zarr arrays often compress very well due to chunking and data patterns
    - Level 9 compression typically provides significant space savings
    - ZIP format is widely supported and portable
    
    **Performance considerations:**
    - Higher compression levels take longer but produce smaller files
    - Level 6-9 usually provides good compression/speed tradeoffs
    - Very large Zarr stores may take considerable time to compress
    
    Examples
    --------
    >>> # Convert with default settings (maximum compression)
    >>> zip_file = zarr_to_zip("my_stream_data.zarr")
    >>> print(f"Created: {zip_file}")  # "my_stream_data.zarr.zip"
    
    >>> # Specify custom output path and compression
    >>> zip_file = zarr_to_zip(
    ...     "simulation_results.zarr", 
    ...     "archived_results.zip",
    ...     compression_level=6
    ... )
    
    >>> # Quick conversion with no compression (just archiving)
    >>> zip_file = zarr_to_zip(
    ...     "temp_data.zarr", 
    ...     compression_level=0,
    ...     verbose=False
    ... )
    """
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Converting zarr directories requires zarr to be installed. "
            "Please install zarr to proceed."
        )
    
    # Validate inputs
    if not os.path.isdir(zarr_dir_path):
        raise ValueError(f"Not a valid Zarr directory store: {zarr_dir_path}")
        
    # Configure ZIP compression (level not directly controllable)
    compression = zipfile.ZIP_DEFLATED if compression_level > 0 else zipfile.ZIP_STORED

    # Generate default ZIP path if not provided
    if zip_path is None:
        base_path = zarr_dir_path.rstrip(os.sep)
        if base_path.endswith('.zarr'):
            base_path = base_path[:-5]
        zip_path = f"{base_path}.zip"

    # Use context managers for safe handling
    with zarr.DirectoryStore(zarr_dir_path) as src_store, \
         zarr.ZipStore(zip_path, mode='w', compression=compression) as dest_store:

        src_root = zarr.open(src_store, mode='r')
        dest_root = zarr.open(dest_store, mode='w')
        
        # Copy with optional progress logging
        log_func = print if verbose else None
        zarr.copy_all(src_root, dest_root, log=log_func)
        
    return zip_path