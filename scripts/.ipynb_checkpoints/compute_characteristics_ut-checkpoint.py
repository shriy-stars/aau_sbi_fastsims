# -*- coding: utf-8 -*-
"""
Module-name
===========

Utility functions for galaxy dynamics and cosmological simulations.

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
__docformat__ = "numpy"  # optional, descriptive only

import contextlib, errno, io, logging, os, sys, pickle, time, warnings, re
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Third-party
import h5py
import matplotlib.image
import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize, root_scalar
from sklearn.neighbors import KernelDensity

# Custom packages
import agama

# define the physical units used in the code: the choice below corresponds to
# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1, length=1, velocity=1)

# list of labels for symmetry spec
symmlabel = {'a':'axi','s':'sph','t':'triax','n':'none'}

######################################################################################
################################READING FUNCTIONS#####################################
######################################################################################

def snaps_rels(sim_dir: Path | str, sep: str = r'\s+') -> pd.DataFrame:
    r"""
    Read snapshot_times.txt robustly from sim_dir and return a dataframe with canonical columns:
      'snap', 'scale-factor', 'redshift', 'time[Gyr]', 'time_width[Myr]'

    Parameters
    ----------
    sim_dir : Path | str
        Directory containing 'snapshot_times.txt'.
    sep : str | None, optional
        Separator to use for pd.read_csv. Default: r'\s+' (whitespace regex).
        If sep is None, it will be interpreted as r'\s+' as well.
        NOTE: regex separators require engine='python' in pandas.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical columns described above (may include extra columns found
        in the file; canonical columns missing in the file are present but filled with NaN).
    """
    snapshot_file = Path(sim_dir) / "snapshot_times.txt"
    if not snapshot_file.exists():
        raise FileNotFoundError(f"snapshot_times.txt not found in {sim_dir}")

    def _normalize_token(tok: str) -> str:
        s = tok.strip().lower()
        s = re.sub(r"[\[\]\(\)\,]", "", s)
        s = re.sub(r"[^0-9a-z]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    token_to_canonical = {
        "i": "snap", "snap": "snap", "index": "snap",
        "scale_factor": "scale-factor", "scale-factor": "scale-factor", "a": "scale-factor",
        "scalefactor": "scale-factor",
        "redshift": "redshift", "z": "redshift",
        "time_gyr": "time[Gyr]", "timegyr": "time[Gyr]", "time": "time[Gyr]", "t": "time[Gyr]",
        "lookback_time_gyr": "lookback-time[Gyr]", "lookback": "lookback-time[Gyr]", "lookback_time": "lookback-time[Gyr]",
        "time_width_myr": "time_width[Myr]", "timewidth": "time_width[Myr]", "time_width": "time_width[Myr]",
        "time-width": "time_width[Myr]"
    }

    # collect comment lines to find a header-like comment
    comment_lines = []
    with open(snapshot_file, "r") as f:
        for raw in f:
            if raw.lstrip().startswith("#"):
                comment_lines.append(raw.rstrip("\n"))
    header_tokens = None
    if comment_lines:
        for line in reversed(comment_lines):
            stripped = line.lstrip("#").strip()
            words = re.split(r"\s+", stripped)
            alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
            if len(alpha_words) >= 2:
                header_tokens = words
                break

    # Prepare read_csv kwargs WITHOUT using delim_whitespace (avoids FutureWarning)
    sep_to_use = sep if sep is not None else r'\s+'
    read_kwargs = {"comment": "#", "header": None}
    # If sep looks like a regex for whitespace (r'\s+'), or contains regex tokens, use engine='python'
    # engine='python' is required for regex separators in pandas
    read_kwargs["sep"] = sep_to_use
    read_kwargs["engine"] = "python"

    df = pd.read_csv(snapshot_file, **read_kwargs)

    if df.shape[0] == 0:
        cols = ["snap", "scale-factor", "redshift", "time[Gyr]", "time_width[Myr]"]
        return pd.DataFrame(columns=cols)

    ncols = df.shape[1]
    mapped_names = None

    if header_tokens:
        normalized = [_normalize_token(t) for t in header_tokens]
        mapped = [token_to_canonical.get(tok, None) for tok in normalized]
        plausible_indices = [i for i, t in enumerate(header_tokens) if re.search(r"[A-Za-z]", t)]
        plausible_mapped = [mapped[i] if mapped[i] is not None else header_tokens[i] for i in plausible_indices]

        if len(plausible_mapped) == ncols:
            mapped_names = plausible_mapped
        else:
            if len(plausible_mapped) >= ncols:
                mapped_names = plausible_mapped[-ncols:]
            else:
                mapped_names = plausible_mapped + [f"col{j}" for j in range(len(plausible_mapped), ncols)]

    if mapped_names is None:
        stats = {}
        for c in df.columns:
            col = df[c].dropna()
            stats[c] = {
                "min": float(col.min()) if col.size else np.nan,
                "max": float(col.max()) if col.size else np.nan,
                "mean": float(col.mean()) if col.size else np.nan,
                "is_integer": bool(np.allclose(col, np.round(col), atol=1e-8)) if col.size else False,
                "n_unique": col.nunique() if col.size else 0
            }

        def score_snap(c):
            s = 0
            if stats[c]["is_integer"]:
                s += 2
            colvals = df[c].values
            if np.all(np.diff(colvals) >= 0):
                s += 1
            if stats[c]["min"] >= 0 and stats[c]["max"] < 1e6:
                s += 0.5
            return s

        def score_scale_factor(c):
            s = 0
            if stats[c]["min"] >= -0.1 and stats[c]["max"] <= 1.2:
                s += 3
            if not stats[c]["is_integer"]:
                s += 0.5
            return s

        def score_redshift(c):
            s = 0
            if stats[c]["min"] >= 0 and stats[c]["max"] > 1.0:
                s += 2
            if stats[c]["max"] > 10:
                s += 1
            return s

        def score_time_gyr(c):
            s = 0
            if stats[c]["min"] >= -1 and stats[c]["max"] <= 20:
                s += 2
            if not stats[c]["is_integer"]:
                s += 0.5
            return s

        def score_time_width_myr(c):
            s = 0
            if stats[c]["min"] >= -1 and stats[c]["max"] < 1e6:
                s += 0.5
            if stats[c]["max"] < 1e5:
                s += 1.0
            if not stats[c]["is_integer"]:
                s += 0.2
            return s

        cols = list(df.columns)
        scores = {c: {} for c in cols}
        for c in cols:
            scores[c]["snap"] = score_snap(c)
            scores[c]["scale-factor"] = score_scale_factor(c)
            scores[c]["redshift"] = score_redshift(c)
            scores[c]["time[Gyr]"] = score_time_gyr(c)
            scores[c]["time_width[Myr]"] = score_time_width_myr(c)

        assigned = {}
        available = set(cols)
        for canonical in ["snap", "scale-factor", "redshift", "time[Gyr]", "time_width[Myr]"]:
            best_col, best_score = None, -1.0
            for c in list(available):
                sc = scores[c][canonical]
                if sc > best_score:
                    best_score = sc
                    best_col = c
            if best_col is not None and best_score > 0:
                assigned[canonical] = best_col
                available.remove(best_col)

        mapped_names = []
        col_to_canon = {v: k for k, v in assigned.items()}
        for c in cols:
            mapped_names.append(col_to_canon.get(c, f"col{c}"))

    if len(mapped_names) != ncols:
        mapped_names = [f"col{c}" for c in range(ncols)]

    df.columns = mapped_names

    canonical_cols = ["snap", "scale-factor", "redshift", "time[Gyr]", "time_width[Myr]"]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = np.nan

    try:
        df["snap"] = pd.to_numeric(df["snap"], errors="coerce").astype("Int64")
    except Exception:
        df["snap"] = pd.to_numeric(df["snap"], errors="coerce")

    for c in ["scale-factor", "redshift", "time[Gyr]", "time_width[Myr]"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    extra_cols = [c for c in df.columns if c not in canonical_cols]
    df = df[canonical_cols + extra_cols]

    return df.reset_index(drop=True)

def read_host_file(sim_dir: Path | str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Read host coordinate data from an HDF5 file.

    Parameters
    ----------
    sim_dir (str): Path to the simulation directory.

    Returns
    -------
    pos (numpy.ndarray): Array of host positions with shape (n_hosts, 3).
    vel (numpy.ndarray): Array of host velocities with shape (n_hosts, 3).

    Raises
    ------
    FileNotFoundError: If the HDF5 file is not found or cannot be opened.
    KeyError: If the required datasets 'host.position' or 'host.velocity' are not found.

    Example
    -------
    pos, vel = read_host_file('/path/to/simulation')
    
    """
    # Construct the file path to the host coordinate HDF5 file
    file_dir = f'{sim_dir}/track/host_coordinates.hdf5'

    if os.path.isfile(file_dir):
        try:
            with h5py.File(file_dir, "r") as f:
                # List all groups in the HDF5 file
                print("Keys: %s" % f.keys())

                # Read the host position and velocity datasets
                pos = np.array(list(f['host.position'])).reshape((-1, 3))
                vel = np.array(list(f['host.velocity'])).reshape((-1, 3))

            return pos, vel
        except KeyError as ke:
            print(f"KeyError: The required dataset '{ke.args[0]}' not found in the HDF5 file.")
            raise
        except Exception as e:
            print(f"Error occurred while reading the HDF5 file: {e}")
            raise
    else:
        print("Host coordinate HDF5 file not found in the specified simulation directory.")
        return None, None

def read(sim_dir: Path | str, nsnap: int | float, 
         species: str | list[str] | None = None, 
         snap_dir: str='output/', 
         snapshot_value_kind: str ='index', 
         assign_hosts: bool = True, 
         assign_hosts_rotation: bool = False,
         sort_dark_by_id: bool = False,
         assign_pointers: bool = False,
         verbose: bool = True,
         **kwargs):
    """
    Reads particle data from snapshot files based on the gizmo_analysis package.

    Parameters
    ----------
    sim_dir (str): The simulation directory containing 'snapshot_time.txt' and 'output/snapshot_**.hdf5' files.
    nsnap (int): The snapshot number to read.
    species (list, optional): A list of particle species to read. Default is ['gas', 'star', 'dark'].
                              It could also be a single species string (e.g., 'gas').
    snap_dir (str, optional): None! This parameter is not used in the function and is here for compatibility.
    snapshot_value_kind (str, optional): The kind of snapshot value to use. Default is 'index'.
    assign_hosts (bool, optional): If True, assign host halos to particles. Default is True.
    assign_hosts_rotation (bool, optional): If True, assign halo rotation information to particles. Default is False.
    sort_dark_by_id (bool, optional): If True, sort dark matter particles by ID. Default is False.
    assign_pointers (bool, optional): If True, assign particle pointers. Default is False.
    verbose (bool, optional): If False, suppress standard output. Default is True.
    **kwargs: Additional keyword arguments to be passed to the underlying function.

    Returns
    -------
    dict: A dictionary containing particle data for the specified species and snapshot.

    Note
    ----
    This function uses the gizmo_analysis package to read particle data from snapshot files.
    The snapshot data should be present in the specified `sim_dir`, and the `nsnap` should correspond
    to a valid snapshot number. The particle data is returned as a dictionary with keys for each species,
    containing information such as positions, velocities, masses, IDs, etc.

    Examples
    --------
    # Read gas, star, and dark matter particle data for snapshot 500 in the 'output/' directory
    particle_data = read(sim_dir="/path/to/sim_directory", nsnap=500, species=['gas', 'star', 'dark'])

    # Read only dark matter particle data for snapshot 100
    dark_matter_data = read(sim_dir="/path/to/sim_directory", nsnap=100, species='dark')

    """
    if species is None: species = ['gas', 'star', 'dark']
    import gizmo_analysis as ga
    import utilities as ut
    
    if verbose:
        part = ga.io.Read.read_snapshots(species=species,
                                         snapshot_value_kind=snapshot_value_kind,    
                                         snapshot_values=nsnap,
                                         simulation_directory=sim_dir+'/',
                                         snapshot_directory=snap_dir,
                                         particle_subsample_factor=1,
                                         assign_hosts=assign_hosts,
                                         assign_hosts_rotation=assign_hosts_rotation,
                                         sort_dark_by_id=sort_dark_by_id,
                                         assign_pointers=assign_pointers, **kwargs
                                        )
    else:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            part = ga.io.Read.read_snapshots(species=species,
                                             snapshot_value_kind=snapshot_value_kind,    
                                             snapshot_values=nsnap,
                                             simulation_directory=sim_dir+'/',
                                             snapshot_directory=snap_dir,
                                             particle_subsample_factor=1,
                                             assign_hosts=assign_hosts,
                                             assign_hosts_rotation=assign_hosts_rotation,
                                             sort_dark_by_id=sort_dark_by_id,
                                             assign_pointers=assign_pointers, **kwargs
                                            )
    
    try:
        if verbose:
            print(f'\n Done reading data for snapshot : {nsnap:03} \n')
    except Exception as e:
        print(e)
        
    return part

## Reading rotation mtxs.
def get_rotation_matrix(sim_dir: Path | str, nsnap: int, 
                        halo: str | None = None, 
                        file_ext: str | None = None,
                        coord_prop: bool = False, # return coord props
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
    """
    Returns the ctr, vctr, and, rotation mtx from sim_dir.  
    
    Parameters
    ----------    
    sim_dir: simulation directory for coords file
    nsnap: snapshot number
    file_ext: file extension of models [spl, DR, Nbody]
    coord_prop: returns ctr, vctr, rot_mat
        
    Returns
    -------
    rot_mat [3x3] np array. 
    ctr [3,], vctr [3,], rot_mat [3,3] (if coord_prop = True)

    """
    
    file_name = f'{nsnap}_coords'
    
    if halo:
        file_name += f'_{halo}'
        
    if file_ext:
        file_name += f'_{file_ext}'

    file_path = os.path.join(sim_dir, f'{file_name}.txt')

    with open(file_path, 'r') as fr:
        lines = fr.readlines()
        ctr = np.array(list(map(lambda line: line.strip().split(" "), lines[3:6])), dtype=np.float32).flatten()
        vctr = np.array(list(map(lambda line: line.strip().split(" "), lines[7:10])), dtype=np.float32).flatten()
        rot = np.array(list(map(lambda line: line.strip().split(" "), lines[-3:])), dtype=np.float32)
    
    if coord_prop:
        return ctr, vctr, rot
    return rot


## Reading Rockstar dark matter particle assignments. 
def read_particle_assignments(sim_dir: Path | str, nsnap: int, 
                              rockstar_directory: str ='halo/rockstar_dm/', 
                              verbose: bool = True) -> dict[int, list[int]]:
    """
    Reads dark matter particle assignments from Rockstar arranged by halo_ID: [particle IDs]
    dictionary structure.

    Parameters
    ----------
    sim_dir (str): 
        The simulation directory containing the Rockstar particle assignment data.
    nsnap (int): 
        The snapshot number to read the particle assignments for.
    rockstar_directory (str, optional): Default is 'halo/rockstar_dm_new/'.
        The subdirectory path within `sim_dir` where Rockstar data is stored.
    verbose (bool, optional). Default is True: 
        Print detailed information.

    Returns
    -------
    dict: 
        A dictionary mapping halo_IDs to lists of particle IDs belonging to each halo.

    Note
    ----
    This function reads the dark matter particle assignments computed by Rockstar for a specific snapshot.
    The particle assignments table should have been precomputed using a C++ pipeline and saved as a pickle file.
    The function will check if a corresponding pickle file exists in the given `sim_dir` before attempting to read
    the particle assignments from the original particle table. If the pickle file is not found, the function will
    generate it from the particle table and save it for faster access in future calls.

    To match particle IDs in the simulation snapshot, the particle ID for the dark matter species should be
    accessed as part['dark']['id'] from the Gizmo snapshot data.

    Examples
    --------
        # Read particle assignments for snapshot 10 in the 'halo/rockstar_dm_new/' directory
        particle_assignments = read_particle_assignments(sim_dir="/path/to/sim_directory", nsnap=500)

    """
    import os, pickle
    
    file_path = f'{sim_dir}/{rockstar_directory}catalog_hdf5/dark_assignment_{nsnap:03d}.pickle'
    
    if os.path.isfile(file_path):
        if verbose:
            print('Reading particle assignments from pickle file.')
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print('the DM assignment pickle file maybe corrupted, looking for the DM table.')
            
    if verbose:
        print('Generating pickle file from particle table')
    
    def process_line(line):
        halo_id, particle_ids = line.split(":")
        return {int(halo_id): list(map(int, particle_ids.split(",")[:-1]))}
    
    particle_file_dir = f'{sim_dir}/{rockstar_directory}catalog/particle_table:{nsnap:03d}'
    final_dict = {}
    
    with open(particle_file_dir, "r") as f:
        [final_dict.update(line_dict) for line_dict in map(process_line, f.read().strip().split("\n"))]
    
    if verbose:
        print(f'Saving assignments @ {file_path}')
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print('Files saved!')
    except Exception as e:
        print(e, f'failed for {sim_dir} at nsnap: {nsnap}')
        if verbose:
            print('Saving failed. Make sure you have writing permissions to the dir or change the save dir.')
        
    return final_dict



## Reading/modifying agama potentials

def generate_lmax_pairs(lmax: int = 0, mmax: int | None = None) -> list[tuple[int, int]]:
    """Generate (l, m) pairs for quantum numbers with optional constraints.
    
    Parameters
    ----------
    lmax (int): Maximum angular momentum number (l ≥ 0)
    mmax (int, optional): Maximum magnetic quantum number (0 ≤ m ≤ mmax)
        
    Returns
    -------
    list: List of (l, m) tuples sorted by l then m
        
    Raises
    ------
    AssertionError: If invalid input constraints are violated
        
    Examples
    --------
    >>> generate_lmax_pairs(1)
    [(0, 0), (1, 0), (1, 1)]
    
    >>> generate_lmax_pairs(2, mmax=1)
    [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)]
    
    """
    # Input validation
    assert lmax >= 0, "lmax must be ≥ 0"
    if mmax is not None:
        assert mmax >= 0, "mmax must be ≥ 0 when specified"
    
    return [
        (l, m)
        for l in range(lmax + 1)
        for m in range(
            (min(l, mmax) + 1)  # Enforces m ≤ l AND m ≤ mmax (if specified)
            if mmax is not None 
            else (l + 1)  # Default case: m ≤ l
        )
    ]

def read_agama_pot(sim_dir: Path | str, nsnap: int, sym: str = 'n', 
                   mult_l: int = 4, kind: str = 'whole',
                   specific_lm_mult: list[tuple[int, int]] | None = None, 
                   specific_m_cylspl: list[tuple[int, int]] | None =None, ## set the specific (l, +-m) to nonzero for mult and specific (+-m) to nonzero for cylspl
                   negative_m: bool = True, ##this ensure -ve m's are always included with their positive counter parts.  
                   file_ext: str = 'DR', 
                   out_acc: bool = False, 
                   halo: str | None = None,
                   verbose: bool = True,#prints info.
                   return_coefs_as_df: bool = False,
                   save_dir: str | None =None  # New: Directory to save modified files
                  ) -> agama.Potential | pd.DataFrame:
    """
    Reads and processes AGAMA potential files, allowing for the application of 
    specific modifications and saving options.

    This function reads an AGAMA potential file for a given simulation snapshot 
    and converts it into an appropriate AGAMA Potential object based on specified 
    options for multipole expansion and cylindrical spline coefficients. Optional 
    modifications to the potential coefficients can be applied, with support for 
    custom saving paths for the modified files.

    Parameters
    ----------
    sim_dir : str
        The base directory containing the simulation data.
    
    nsnap : int
        Snapshot number for the simulation.

    sym : str, optional
        The symmetry type for the expansion; options are:
        'a' for axisymmetric, 'n' for none, 't' for triaxial 
        (default is 'n').

    mult_l : int, optional
        The maximum order of the multipole expansion (default is 4).

    kind : str, optional
        Specifies which model component to load; options are:
        'whole' for the full model, 'dark' for the dark halo, and 
        'bar' for the bar component (default is 'whole').

    specific_lm_mult : list of tuples, optional
        Specifies specific (l, m) multipole coefficients to retain 
        in the model. Each tuple in the list should be of the form 
        (l, m), where l and m are integers (default is None).
        If `negative_m=True`, the function ensures that both 
        positive and negative m coefficients are included.

    specific_m_cylspl : list of int, optional
        List of integer m values to retain in the cylindrical spline 
        expansion. Each value should correspond to a desired azimuthal 
        harmonic term (default is None). If `negative_m=True`, both 
        positive and negative m coefficients are retained for each m.

    negative_m : bool, optional
        If True, ensures that all negative m coefficients are included 
        alongside positive counterparts in the specified multipole or 
        cylindrical spline expansions (default is True).

    file_ext : str, optional
        File extension indicating the type of model files to use; 
        typical values are 'spl', 'DR', or 'Nbody' (default is 'DR').

    out_acc : bool, optional
        If True, reads files from an alternate directory for output 
        acceleration snapshot models (default is False).

    halo : str or None, optional
        Specifies which halo component to load if different than 
        the entire model. Options are None (default), 'MW' for Milky 
        Way, or 'LMC' for Large Magellanic Cloud.

    verbose : bool, optional
        If True, prints detailed information about the operations 
        being performed and paths being used (default is True).

    return_coefs_as_df : bool, optional
        If True, returns the multipole expansion Phi coefficients 
        as a custom pandas DataFrame class rather than as an AGAMA 
        Potential object (default is False).

    save_dir : str or None, optional
        Specifies the directory to save modified potential coefficient 
        files. If set to None, modified files are saved in place 
        in the original directory (default is None).

    Returns
    -------
    agama.Potential or pandas.DataFrame
        Returns an AGAMA Potential object of the specified component 
        ('whole', 'dark', or 'bar') with applied modifications, or a 
        pandas DataFrame containing multipole expansion coefficients 
        if `return_coefs_as_df=True`.
    """
    
    import agama, os
    symmlabel = {'a':'axi','s':'sph','t':'triax','n':'none'}
    file_name_bar = f'{sim_dir}/potential/10kpc/{nsnap}.bar.{symmlabel[sym]}_{mult_l}.coef_cylsp'
    file_name_dark = f'{sim_dir}/potential/10kpc/{nsnap}.dark.{symmlabel[sym]}_{mult_l}.coef_mul'
    
    if specific_lm_mult is not None and negative_m:
        specific_lm_mult = _add_negative_m(specific_lm_mult)
    if verbose:
        print(f'Mult exp with (l, m): {specific_lm_mult}')
            
        
    if out_acc:
        file_name_bar = f'{sim_dir}/potential/10kpc/out_acc/{nsnap}.bar.{symmlabel[sym]}_{mult_l}.coef_cylsp'
        file_name_dark = f'{sim_dir}/potential/10kpc/out_acc/{nsnap}.dark.{symmlabel[sym]}_{mult_l}.coef_mul'
                
    if halo:
        file_name_bar = f'{sim_dir}/potential/10kpc/{nsnap}.bar.{symmlabel[sym]}_{mult_l}.{halo}.coef_cylsp'
        file_name_dark = f'{sim_dir}/potential/10kpc/{nsnap}.dark.{symmlabel[sym]}_{mult_l}.{halo}.coef_mul'
    
    if file_ext:
        file_name_bar += f'_{file_ext}'
        file_name_dark += f'_{file_ext}'
        
    # try:
        # If specific_lm_mult is provided, modify the potential coefficients
    if specific_lm_mult is not None:
        file_name_dark = _modify_multipole_coefficients(file_name_dark, specific_lm_mult, save_dir=save_dir, verbose=verbose)
    
    if return_coefs_as_df:
        return read_mult_coefs(file_name_dark)
        
    
    pxr_dark = agama.Potential(file_name_dark)
    
    # except Exception as e:
    #     if verbose:
    #         print(e)
    #     package_dir = os.path.dirname(agama.__file__)
    #     os.system(f'perl {package_dir}/data/convertcoefs.pl {file_name_dark}')
    #     os.system(f'mv {file_name_dark}.ini {file_name_dark}')
    #     pxr_dark = agama.Potential(file_name_dark)
    #     if verbose:
    #         print(f'Updated pot dark files created')

            
    # try:
    if specific_m_cylspl is not None:
        file_name_bar = modify_cylspl_coefficients(file_name_bar, specific_m_cylspl, negative_m, save_dir=save_dir, verbose=verbose)

    pxr_bar = agama.Potential(file_name_bar)
        
    # except Exception as e:
    #     if verbose:
    #         print(e) 
    #     package_dir = os.path.dirname(agama.__file__)
    #     os.system(f'perl {package_dir}/data/convertcoefs.pl {file_name_bar}')
    #     os.system(f'mv {file_name_bar}.ini {file_name_bar}')
    #     pxr_bar = agama.Potential(file_name_bar)
    #     if verbose:
    #         print(f'Updated pot bar files created')
    
    if kind == 'dark':
        if verbose:
            print('potential dark model: spherical harmonics BFE')
        return pxr_dark

    elif kind == 'bar':
        if verbose:
            print('potential bar model: Azimuthal harmonics BFE')
        return pxr_bar
    else: 
        if verbose:
            print('entire potential model: spherical+Azimuthal harmonics BFE')
        return agama.Potential(agama.Potential(file_name_bar), agama.Potential(file_name_dark))

def _add_negative_m(lm_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Expands a list of (l, m) pairs to include negative m values, ensuring 
    symmetry, and returns the list sorted by increasing l and then by m.

    This function processes a list of (l, m) tuples, adding the corresponding 
    negative m value for each positive m. It then sorts the resulting list 
    in ascending order of l and m values.

    Parameters
    ----------
    lm_pairs : list of tuples
        List of tuples, each representing an (l, m) pair of coefficients. 
        For each (l, m) pair where m is positive, the function will add 
        the corresponding negative m pair (l, -m) to the list.

    Returns
    -------
    list of tuples
        A sorted list of (l, m) pairs, including both positive and negative m 
        values, in ascending order of l and then m.
    """
    
    expanded_lm_pairs = set()
    
    # Add negative m for each positive m
    for l, m in lm_pairs:
        expanded_lm_pairs.add((l, m))
        if m != 0:
            expanded_lm_pairs.add((l, -m))
    
    # Convert to sorted list: sort first by l, then by m
    sorted_lm_pairs = sorted(expanded_lm_pairs, key=lambda pair: (pair[0], pair[1]))
    
    return sorted_lm_pairs

def _modify_multipole_coefficients(file_name: str, lm_pairs: list[tuple[int, int]], 
                                  save_dir: str | None =None, 
                                  verbose: bool = False) -> str:
    """
    Modifies the multipole coefficients in the specified AGAMA coefficient file 
    based on the provided (l, m) pairs, allowing optional saving to a custom directory.

    This function reads the multipole coefficient file, zeroes out all coefficients
    not included in the provided list of (l, m) pairs, and saves the modified file 
    to a specified directory if provided.

    Parameters
    ----------
    file_name : str
        Path to the original multipole coefficient file.

    lm_pairs : list of tuples
        List of tuples, each representing an (l, m) pair of multipole coefficients
        to retain. All other (l, m) coefficients will be zeroed.

    save_dir : str or None, optional
        Directory to save the modified coefficient file. If None, the modified 
        file is saved in the same directory as `file_name` with a modified name 
        (default is None).

    verbose : bool, optional
        If True, prints detailed information about modifications and paths 
        used for saving the modified file (default is False).

    Returns
    -------
    str
        Path to the modified file.
    """
    
    # Read the file
    with open(file_name, 'r') as f:
        lines = f.readlines()
    f.close()
    
    print('File is empty' if len(lines) < 1 else '')
    
    # Find gridSizeR dynamically
    gridSizeR = None
    for line in lines:
        if line.startswith("gridSizeR="):
            gridSizeR = int(line.split('=')[1].strip())
            break

    if gridSizeR is None:
        raise ValueError("gridSizeR could not be found in the file.")
    
    # Initialize indices for #Phi and #dPhi/dr sections
    start_phi_coef_idx = None
    end_phi_coef_idx = None
    start_dphi_coef_idx = None
    end_dphi_coef_idx = None

    # Find the start and end of the coefficients for #Phi and #dPhi/dr
    for i, line in enumerate(lines):
        if line.startswith("#Phi") or line.startswith("#rho"):
            start_phi_coef_idx = i + 2  # Coefficients start 2 lines after #Phi (header + data)
            end_phi_coef_idx = start_phi_coef_idx + gridSizeR  # Coefficients block spans `gridSizeR` rows
        elif line.startswith("#dPhi/dr"):
            start_dphi_coef_idx = i + 2  # Coefficients start 2 lines after #dPhi/dr
            end_dphi_coef_idx = start_dphi_coef_idx + gridSizeR

    # Parse the header for coefficients (found right before the start of the data for #Phi)
    header_line = lines[start_phi_coef_idx - 1].strip().split('\t')  # Columns separated by tabs

    # Generate the expanded (l, m) pairs
    lm_pairs_to_keep = _add_negative_m(lm_pairs)

    lm_indices_to_keep = []

    # Loop over the header line starting from the second item (first one is radius)
    for idx, coef in enumerate(header_line[1:], start=1):
        # Extract l and m by splitting the string 'l=x,m=y' and removing 'l=' and 'm='
        l_value = int(coef.split(',')[0][2:])  # removes 'l=' prefix and converts to int
        m_value = int(coef.split(',')[1][2:])  # removes 'm=' prefix and converts to int
        
        # Check if the current (l, m) pair is in lm_pairs_to_keep
        if (l_value, m_value) in lm_pairs_to_keep:
            lm_indices_to_keep.append(idx)

    # Zero out coefficients in the #Phi block
    for i in range(start_phi_coef_idx, end_phi_coef_idx):
        coef_values = lines[i].strip().split('\t')
        new_row = [coef_values[0]]  # Keep the radius column
        
        # Zero out unwanted coefficients
        for idx in range(1, len(coef_values)):
            if idx in lm_indices_to_keep:
                new_row.append(coef_values[idx])
            else:
                new_row.append('0.0')  # Zero out unwanted coefficients
        
        lines[i] = '\t'.join(new_row)  # Update the line with new values

    # Repeat the same for the #dPhi/dr section
    for i in range(start_dphi_coef_idx, end_dphi_coef_idx):
        coef_values = lines[i].strip().split('\t')
        new_row = [coef_values[0]]  # Keep the radius column
        
        # Zero out unwanted coefficients
        for idx in range(1, len(coef_values)):
            if idx in lm_indices_to_keep:
                new_row.append(coef_values[idx])
            else:
                new_row.append('0.0')  # Zero out unwanted coefficients
        
        lines[i] = '\t'.join(new_row)  # Update the line with new values

    # Write the modified lines back to a temporary file
    temp_file_name = os.path.join(save_dir, os.path.basename(file_name) + ".modified") if save_dir else file_name + ".modified"
    # Write lines ensuring no extra spaces or newlines
    with open(temp_file_name, 'w') as f:
        for line in lines:
            # Strip any leading/trailing whitespace and ensure a single newline at the end
            f.write(line.rstrip() + '\n')
        
        # Add an empty line before the #dPhi/dr section
        if start_dphi_coef_idx > 0:
            f.write('\n')  # Add an empty line
    f.close()

    if verbose:
        print(f'Temp file written at: {temp_file_name}')
    
    # Return the temporary file name for further processing
    return temp_file_name

def modify_cylspl_coefficients(file_name: str, m_lists: list[int], negative_m: bool, 
                               save_dir: str | None = None, verbose: bool = False) -> str:
    """
    Modifies the Cylindrical Spline (CylSpline) coefficients in the specified 
    AGAMA coefficient file based on the provided list of m values, with an option 
    to include negative m values.

    This function reads the CylSpline coefficient file, zeroes out all coefficients 
    for m values not included in the provided list, and saves the modified file to a 
    specified directory if provided.

    Parameters
    ----------
    file_name : str
        Path to the original CylSpline coefficient file.

    m_lists : list of int
        List of integer m values to retain in the cylindrical spline expansion. 
        All other m coefficients will be zeroed.

    negative_m : bool
        If True, ensures that each positive m value specified in `m_lists` is 
        accompanied by its negative counterpart in the retained coefficients.

    save_dir : str or None, optional
        Directory to save the modified coefficient file. If None, the modified 
        file is saved in the same directory as `file_name` with a modified name 
        (default is None).

    verbose : bool, optional
        If True, prints detailed information about modifications and paths 
        used for saving the modified file (default is False).

    Returns
    -------
    str
        Path to the modified file.
    """    
    # Read the file
    with open(file_name, 'r') as f:
        lines = f.readlines()
    f.close()
    # Find gridSizeR, gridSizez, and mmax dynamically
    gridSizeR = None
    gridSizez = None
    mmax = None
    
    for line in lines:
        if line.startswith("gridSizeR="):
            gridSizeR = int(line.split('=')[1].strip())
        if line.startswith("gridSizez=") or line.startswith("gridSizeZ="):
            gridSizez = int(line.split('=')[1].strip())
        if line.startswith("mmax="):
            mmax = int(line.split('=')[1].strip())
    
    if gridSizeR is None or gridSizez is None or mmax is None:
        raise ValueError("gridSizeR, gridSizez, or mmax could not be found in the file.")
    
    # Expand m-lists to automatically include negative m values
    expanded_m_lists = []
    for m in m_lists:
        expanded_m_lists.append(m)
        if m != 0:
            if negative_m:
                expanded_m_lists.append(-m)
                
    expanded_m_lists = sorted(expanded_m_lists)
    
    if verbose:
        print(f'Cylspl exp with (m): {expanded_m_lists}')

    # Initialize variables to track where the coefficients for each m value start
    start_phi_coef_idx = None
    m_indices = {}
    current_m = None

    # Step through the file and identify blocks of coefficients
    zero_out_block = False
    current_m = None

    for i, line in enumerate(lines):
        # Look for lines that match the m-value pattern (e.g. "-1\t#m" or "1\t#m")
        if '\t#m' in line:
            current_m = int(line.split('\t')[0])  # Extract the m value before the tab
            
            # If current m is in the expanded list, start modifying this block
            if current_m not in expanded_m_lists:
                zero_out_block = True
                # if verbose:
                #     print(f'Processing m = {current_m}, starting at line {i}')
        
        # When in an m-block, we need to handle the rows with coefficients
        if zero_out_block:
            # The line that starts with "#R(row)\z(col)" is a header, so we skip it
            if line.startswith("#R(row)"):                
                continue
            if '\t#m' in line:
                continue
            
            # After the header, each row has coefficients we need to modify
            if not line.startswith("#") and current_m not in expanded_m_lists:
                row_values = line.strip().split()

                # Keep the first entry (the radial grid point) intact, zero out the rest
                radial_grid_value = row_values[0]
                zeroed_row = [radial_grid_value] + ['0.0'] * gridSizez

                # Update the line with the zeroed coefficients
                lines[i] = ' '.join(zeroed_row) + '\n'

    # Write the modified content to a new temporary file
    temp_file_name = os.path.join(save_dir, os.path.basename(file_name) + ".modified") if save_dir else file_name + ".modified"
    with open(temp_file_name, 'w') as f:
        f.writelines(lines)
    f.close()
        
    if verbose:
        print(f'Temp file written at: {temp_file_name}')
    
    return temp_file_name    

class MultipoleDataFrame(pd.DataFrame):
    """
    A custom DataFrame subclass for handling multipole expansion coefficient data.
    
    This class provides methods to compute the radial and total power for each
    harmonic order in a multipole expansion, enabling users to analyze data
    efficiently while retaining full functionality of a pandas DataFrame.
    
    Methods
    -------
    compute_radial_power(l, use_quadrature=False)
        Computes radial power for a given harmonic order `l` across all radial grids.
        
    compute_total_power(l, use_quadrature=False)
        Computes total power for a given harmonic order `l` by summing radial powers.
        
    """
    
    @property
    def _constructor(self):
        return MultipoleDataFrame
    
    def compute_radial_power(self, l: int, use_quadrature: bool = True) -> pd.Series:
        """
        Computes the radial power for a given harmonic order `l` across all radial grids.
        
        Parameters
        ----------
        l : int
            The harmonic order for which the radial power is to be computed.
        use_quadrature : bool, optional (default=True)
            If True, computes power as the sum of squares of coefficients for all m.
            If False, computes power as the sum of absolute values of coefficients for all m.
        
        Returns
        -------
        pd.Series
            A Series representing the radial power across all radial grids, 
            indexed by the radius.
            
        Notes
        -----
        - Radial power at a given radius `r` for harmonic order `l`:
          - When `use_quadrature=False`: sum(|C_{l,m}(r)| for all m, -l <= m <= l)
          - When `use_quadrature=True`: sum(C_{l,m}(r)^2 for all m, -l <= m <= l)
        """
        columns = [col for col in self.columns if f"l={l}," in col]
        
        if use_quadrature:
            radial_power = self[columns].pow(2).sum(axis=1)
        else:
            radial_power = self[columns].abs().sum(axis=1)
        
        return pd.Series(radial_power.values, index=self['radius'], name=f'Radial Power (l={l})')
    
    def compute_total_power(self, l: int, use_quadrature: bool = True) -> float:
        """
        Computes the total power for a given harmonic order `l` by summing radial powers.
        
        Parameters
        ----------
        l : int
            The harmonic order for which the total power is to be computed.
        use_quadrature : bool, optional (default=True)
            If True, computes power as the sum of squares of coefficients for all m and radii.
            If False, computes power as the sum of absolute values of coefficients for all m and radii.
            
        Returns
        -------
        float
            The total power for harmonic order `l`, summing over all radial grids.
            
        Notes
        -----
        - Total power for harmonic order `l`:
          - When `use_quadrature=False`: sum(|C_{l,m}(r)| for all m, -l <= m <= l, across all radii r)
          - When `use_quadrature=True`: sum(C_{l,m}(r)^2 for all m, -l <= m <= l, across all radii r)
        """
        radial_power_series = self.compute_radial_power(l, use_quadrature=use_quadrature)
        total_power = radial_power_series.sum()
        
        return total_power


def read_mult_coefs(file_path: str):
    """
    Reads multipole expansion coefficients from a formatted text file, 
    specifically the section for the potential or density coefficients labeled 
    with `#Phi` or `#rho`, and returns a MultipoleDataFrame with additional 
    methods to compute radial and total powers.
    
    Parameters
    ----------
    file_path : str
        Path to the text file containing multipole expansion data.
        
    Returns
    -------
    MultipoleDataFrame
        A custom DataFrame with additional methods `compute_radial_power` and 
        `compute_total_power` for analyzing multipole expansion data.
        
    Example
    -------
    >>> df = read_mult_coefs('path/to/multipole_file.txt')
    >>> radial_power_l2 = df.compute_radial_power(2)
    >>> total_power_l2 = df.compute_total_power(2)
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    start_row = None
    end_row = None
    
    for i, line in enumerate(lines):
        # Check for either #Phi or #rho as the start indicator
        if '#Phi' in line or '#rho' in line:
            start_row = i + 1
        elif start_row is not None and line.strip() == '':
            end_row = i
            break
    
    end_row = end_row or len(lines)
    headers = lines[start_row].replace('#radius', 'radius').split()
    
    data = pd.read_csv(file_path, sep=r'\s+', skiprows=start_row+1, 
                       nrows=end_row-start_row-1, names=headers, engine='python')
    
    return MultipoleDataFrame(data)

def create_evolv_potential_files(sim_dir: Path | str, snap_range: tuple[int, int] = (300, 600), 
                                 create_baryons: bool = True, 
                                 create_dark_matter: bool = True, 
                                 baryon_model: str = "*.bar.none_4.coef_cylsp_spl", 
                                 dark_matter_model: str = "*.dark.none_4.coef_mul_spl", 
                                 baryon_file_name: str | None = None, 
                                 dark_matter_file_name: str | None = None, 
                                 verbose: bool = True) -> None:
    """
    Create evolving potential files for baryons and dark matter from snapshots.

    Parameters
    ----------
    sim_dir : Path | str
        Simulation directory containing `snapshot_times.txt` and the `potential/10kpc/` subdirectory.
    snap_range : tuple[int, int], optional
        Inclusive snapshot range to include (start, end), by default (300, 600).
    create_baryons : bool, optional
        Whether to create the baryon potential file, by default True.
    create_dark_matter : bool, optional
        Whether to create the dark-matter potential file, by default True.
    baryon_model : str, optional
        Baryon model name/pattern, by default `"*.bar.none_4.coef_cylsp_spl"`.
    dark_matter_model : str, optional
        Dark-matter model name/pattern, by default `"*.dark.none_4.coef_mul_spl"`.
    baryon_file_name : str | None, optional
        Custom filename to save baryon potentials; if `None` a default name is used.
    dark_matter_file_name : str | None, optional
        Custom filename to save dark-matter potentials; if `None` a default name is used.
    verbose : bool, optional
        Whether to print progress messages, by default True.

    Raises
    ------
    FileNotFoundError
        If expected files are missing in `sim_dir/potential/10kpc/`.
    """
    
    # Step 1: Load the snapshot data
    snapshot_df = snaps_rels(sim_dir)
    if snapshot_df is None:
        print("Snapshot file could not be loaded. Exiting.")
        return

    # Step 2: Filter the snapshot data based on the snap_range
    filtered_df = snapshot_df[(snapshot_df['snap'] >= snap_range[0]) & (snapshot_df['snap'] <= snap_range[1])]

    # Step 3: Determine output directory
    potential_dir = os.path.join(sim_dir, "potential", "10kpc")
    os.makedirs(potential_dir, exist_ok=True)

    # Step 4: Create baryon file if requested
    if create_baryons:
        if baryon_file_name is None:
            baryon_file_name = "cyl_spl_bar_evolv.ini"
        baryon_file_path = os.path.join(potential_dir, baryon_file_name)
        _create_potential_file(filtered_df, baryon_file_path, baryon_model, verbose)
        _check_missing_files(filtered_df, potential_dir, baryon_model)

    # Step 5: Create dark matter file if requested
    if create_dark_matter:
        if dark_matter_file_name is None:
            dark_matter_file_name = "mult_halo_evolv.ini"
        dark_matter_file_path = os.path.join(potential_dir, dark_matter_file_name)
        _create_potential_file(filtered_df, dark_matter_file_path, dark_matter_model, verbose)
        _check_missing_files(filtered_df, potential_dir, dark_matter_model)

    if verbose:
        print("All necessary files are present. Exiting successfully.")

def _create_potential_file(filtered_df, file_path: Path | str, model_name: str, verbose: bool = True):
    """
    Helper function to create a potential file from the filtered DataFrame.

    Parameters
    ----------
    filtered_df (pd.DataFrame): The filtered DataFrame containing snapshot data.
    file_path (str): The output file path for the potential file.
    model_name (str): The model name to use for the file content.
    verbose (bool): Whether to print std output. Default is True. Errors are still printed.
    """
    lines = [
        "[Potential]",
        "type = Evolving",
        "interpLinear = True",
        "Timestamps"
    ]

    for _, row in filtered_df.iterrows():
        time_gyr = row['time[Gyr]']
        snap_num = int(row['snap'])  # Ensure the snapshot number is formatted as an integer
        filename = f"{snap_num}{model_name.replace('*', '')}"
        lines.append(f"{time_gyr} {filename}")

    # Write to the output file
    with open(file_path, "w") as f:
        f.write("\n".join(lines))
    
    if verbose:
        print(f"File created: {file_path}")

def _check_missing_files(filtered_df, potential_dir, model_name):
    """
    Helper function to check for missing files in the potential directory.

    Parameters
    ----------
    filtered_df (pd.DataFrame): 
        The filtered DataFrame containing snapshot data.
    potential_dir (str): 
        The directory where the potential files should be located.
    model_name (str): 
        The model name pattern used to generate the filenames.
        
    Raises
    ------
    FileNotFoundError: If any expected files are missing.
    """
    missing_files = []
    for snap_num in filtered_df['snap']:
        expected_file = os.path.join(potential_dir, f"{int(snap_num)}{model_name.replace('*', '')}")
        if not os.path.exists(expected_file):
            missing_files.append(expected_file)

    if missing_files:
        raise FileNotFoundError(f"The following files are missing:\n" + "\n".join(missing_files))


######################################################################################
####################### Coordinate Transformations ###################################
######################################################################################

#### convert centering ####
###########################
def pynbody_halo(particles, mask=0):
    
    """
    Creates a PyNbody compatible halo from a FIRE-2 snaphot after accurate centering.
    
    Parameters
    ----------
    particles: FIRE 2 part directory with stars and dark matter after correct centering.
    
    Returns
    -------
    halo_pynb: PyNbody halo object.
    """
    import pynbody 
    if type(mask)==int :
        ndark = len(particles['dark']['mass'])
    else:
        ndark = len(mask[mask!=0])
    nstar = len(particles['star']['mass'])
    halo_pynb = pynbody.new(dark=int(ndark), star=int(nstar), order='dark,star')
    print(np.shape(mask), np.shape(particles['dark'].prop('host.distance')))
    dmpart = particles['dark'].prop('host.distance')[mask]
    print(np.shape(dmpart), ndark)
    halo_pynb.dark['pos'] = dmpart
    halo_pynb.dark['vel'] = particles['dark'].prop('host.velocity')[mask]
    halo_pynb.dark['mass'] = particles['dark'].prop('mass')[mask]

    halo_pynb.star['pos'] = particles['star'].prop('host.distance')
    halo_pynb.star['vel'] = particles['star'].prop('host.velocity')
    halo_pynb.star['mass'] = particles['star'].prop('mass')
    halo_pynb.dark['pos'].units = 'kpc'
    halo_pynb.dark['vel'].units = 'km s**-1'
    halo_pynb.dark['mass'].units = 'Msol'

    halo_pynb.star['pos'].units = 'kpc'
    halo_pynb.star['vel'].units = 'km s**-1'
    halo_pynb.star['mass'].units = 'Msol'

    return halo_pynb

def make_pynbody_rotations(halo):
    """
    Computes the rot mat to align galactic disk with XY plane.
    
    Parameters
    ----------
    halo: PyNbody halo objecy with species and spec_props such as mass, pos, and vel.
    
    Returns
    -------
    Tx_sideon: rot_mat (3x3) [np.dot(pos, rot.T) gives (x, y, z)]
    """
    import pynbody 
    cen = halo[pynbody.filt.Sphere("5 kpc")]
    Lh = pynbody.analysis.angmom.ang_mom_vec(cen)
    # pynbody.analysis.angmom.sideon()

    Tx_faceon = pynbody.analysis.angmom.calc_faceon_matrix(Lh)
    # Tx_sideon = pynbody.analysis.angmom.calc_sideon_matrix(Lh)
    return Tx_faceon #returns x, y, z

#### coordinate fields ####
###########################

# -------------------------
# Helper: input checking
# -------------------------
def _as_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Ensure xyz is an ndarray of floats and has last dimension == 3.
    Accepts array-like inputs (including pandas.DataFrame) via np.asarray.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim < 1 or xyz.shape[-1] != 3:
        raise ValueError("Input must be array-like with last dimension == 3 (shape (..., 3)).")
    return xyz

def Cart2Sph(
    xyz: np.ndarray,
    mollweide: bool = False,
) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    xyz : np.ndarray
        Positions in Cartesian coordinates, shape (..., 3).
    mollweide : bool, default False
        If True, adjust phi for Mollweide projection (wrap to (-pi, pi]).
        Theta remains the polar (colatitude) angle from +z in radians.

    Returns
    -------
    np.ndarray
        Spherical coordinates with order ['rho','theta','phi'] and shape (..., 3).

        - rho : radial distance
        - theta : polar angle (colatitude) in radians measured from +z
            * Always in [0, pi]
            * For Healpy Mollweide plotting (mollweide=True), theta remains colatitude
              — do NOT convert to latitude, because Healpy expects colatitude.
        - phi : azimuthal angle in radians
            * [0, 2*pi) if mollweide=False
            * (-pi, pi] if mollweide=True, as required by Healpy's Mollweide projection.

        NaNs in any input coordinate propagate to all outputs for that point.
    """    
    xyz = _as_xyz(xyz)
    # extract components (works for any leading dims)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    xy = x**2 + y**2
    rho = np.sqrt(xy + z**2)
    theta = np.arctan2(np.sqrt(xy), z)  # polar (colatitude) angle from +z
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2 * np.pi)        # map to [0, 2*pi)

    if mollweide:
        # theta = (np.pi / 2) - theta     # convert to latitude
        phi = np.where(phi > np.pi, phi - 2*np.pi, phi)  # (-pi, pi]

    pts = np.empty_like(xyz)
    pts[..., 0] = rho
    pts[..., 1] = theta
    pts[..., 2] = phi

    # NaN handling: if any input coordinate is NaN, set all outputs for that point to NaN
    invalid = np.any(np.isnan(xyz), axis=-1)
    if np.any(invalid):
        pts[invalid] = np.nan 
    
    return pts
    
def Cart2Cyl(xyz: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to cylindrical coordinates.

    Parameters
    ----------
    xyz : np.ndarray or pd.DataFrame
        Positions in Cartesian coordinates, shape (N, 3).

    Returns
    -------
    np.ndarray
        Cylindrical coordinates with order ['R','phi','z'] and shape (..., 3).
        phi in [0, 2*pi).
    """
    xyz = _as_xyz(xyz)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    R = np.sqrt(x**2 + y**2) # R
    phi = np.arctan2(y, x) # phi
    phi = np.mod(phi, 2 * np.pi) # phi in 0->2pi

    pts = np.empty_like(xyz)
    pts[..., 0] = R
    pts[..., 1] = phi
    pts[..., 2] = z

    invalid = np.any(np.isnan(xyz), axis=-1)
    if np.any(invalid):
        pts[invalid] = np.nan

    return pts

def Sph2Cart(
    xyz: np.ndarray | pd.DataFrame,
    return_df: bool = False
) -> pd.DataFrame | np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    xyz : np.ndarray or pd.DataFrame
        Positions in spherical coordinates ['rho','theta','phi'], shape (N, 3).
    return_df : bool, default False
        If True, return a pandas DataFrame; otherwise return a NumPy array.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Cartesian coordinates with columns ['X','Y','Z'] if return_df=True,
        otherwise an (N,3) NumPy array.
    """
    # Ensure numpy array
    if isinstance(xyz, pd.DataFrame):
        xyz = xyz.values
    elif not isinstance(xyz, np.ndarray):
        raise TypeError("xyz must be a np.ndarray or pd.DataFrame")

    ptsnew = np.zeros_like(xyz)
    ptsnew[:,0] = xyz[:,0] * np.sin(xyz[:,1]) * np.cos(xyz[:,2])  # X
    ptsnew[:,1] = xyz[:,0] * np.sin(xyz[:,1]) * np.sin(xyz[:,2])  # Y
    ptsnew[:,2] = xyz[:,0] * np.cos(xyz[:,1])                     # Z

    if return_df:
        return pd.DataFrame(ptsnew, columns=['X','Y','Z'])
    else:
        return ptsnew
        
def Cyl2Cart(
    xyz: np.ndarray | pd.DataFrame,
    return_df: bool = False
) -> pd.DataFrame | np.ndarray:
    """
    Convert cylindrical coordinates to Cartesian coordinates.

    Parameters
    ----------
    xyz : np.ndarray or pd.DataFrame
        Positions in cylindrical coordinates ['R','phi','z'], shape (N, 3).
    return_df : bool, default False
        If True, return a pandas DataFrame; otherwise return a NumPy array.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Cartesian coordinates with columns ['X','Y','Z'] if return_df=True,
        otherwise an (N,3) NumPy array.
    """
    # Ensure numpy array
    if isinstance(xyz, pd.DataFrame):
        xyz = xyz.values
    elif not isinstance(xyz, np.ndarray):
        raise TypeError("xyz must be a np.ndarray or pd.DataFrame")

    ptsnew = np.zeros_like(xyz)
    ptsnew[:,0] = xyz[:,0] * np.cos(xyz[:,1])  # X = R * cos(phi)
    ptsnew[:,1] = xyz[:,0] * np.sin(xyz[:,1])  # Y = R * sin(phi)
    ptsnew[:,2] = xyz[:,2]                     # Z (same)

    if return_df:
        return pd.DataFrame(ptsnew, columns=['X','Y','Z'])
    else:
        return ptsnew

def Sph2Cyl(
    xyz: np.ndarray | pd.DataFrame,
    return_df: bool = False
) -> pd.DataFrame | np.ndarray:
    """
    Convert spherical coordinates to cylindrical coordinates directly.

    Parameters
    ----------
    xyz : np.ndarray or pd.DataFrame
        Positions in spherical coordinates ['rho','theta','phi'], shape (N, 3).
    return_df : bool, default False
        If True, return a pandas DataFrame; otherwise return a NumPy array.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Cylindrical coordinates with columns ['R','phi','z'] if return_df=True,
        otherwise an (N,3) NumPy array.
    """
    # Ensure numpy array
    if isinstance(xyz, pd.DataFrame):
        xyz = xyz.values
    elif not isinstance(xyz, np.ndarray):
        raise TypeError("xyz must be a np.ndarray or pd.DataFrame")

    ptsnew = np.zeros_like(xyz)
    ptsnew[:,0] = xyz[:,0] * np.sin(xyz[:,1])  # R = rho * sin(theta)
    ptsnew[:,1] = xyz[:,2]                     # phi (same)
    ptsnew[:,2] = xyz[:,0] * np.cos(xyz[:,1])  # z = rho * cos(theta)

    if return_df:
        return pd.DataFrame(ptsnew, columns=['R','phi','z'])
    else:
        return ptsnew



def Cyl2Sph(
    xyz: np.ndarray | pd.DataFrame,
    return_df: bool = False
) -> pd.DataFrame | np.ndarray:
    """
    Convert cylindrical coordinates to spherical coordinates directly.

    Parameters
    ----------
    xyz : np.ndarray or pd.DataFrame
        Positions in cylindrical coordinates ['R','phi','z'], shape (N, 3).
    return_df : bool, default False
        If True, return a pandas DataFrame; otherwise return a NumPy array.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Spherical coordinates with columns ['rho','theta','phi'] if return_df=True,
        otherwise an (N,3) NumPy array.
    """
    # Ensure numpy array
    if isinstance(xyz, pd.DataFrame):
        xyz = xyz.values
    elif not isinstance(xyz, np.ndarray):
        raise TypeError("xyz must be a np.ndarray or pd.DataFrame")

    ptsnew = np.zeros_like(xyz)
    ptsnew[:,0] = np.sqrt(xyz[:,0]**2 + xyz[:,2]**2)  # rho = sqrt(R^2 + z^2)
    ptsnew[:,1] = np.arctan2(xyz[:,0], xyz[:,2])      # theta = atan2(R, z)
    ptsnew[:,2] = xyz[:,1]                            # phi (same)

    if return_df:
        return pd.DataFrame(ptsnew, columns=['rho','theta','phi'])
    else:
        return ptsnew

#### vector fields ####
###########################
def vec_Cart2Sph(pos_cart: np.ndarray, vec_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts complex vectors and positions calculated in cartesian coordinates to vectors in spherical coordinate
    and returns it along with spherical positions.
    e.g use it force vectors and etc
    
    Parameters
    ----------
    pos_cart: Nx3 dimensional array of positions in cartesian coords.
    vec_cart : Nx3 dimensional array of vectors in cartesian coords.

    Returns
    -------
    pos_sph : positions in spherical coords (Nx3) [R,theta,phi]
    F=vec_sph : vectors in spherical coords (NX3) [vec_R,vec_theta,vec_phi]

    """
    print(f'Converting coordinates to spherical')
    pos_sph = Cart2Sph(pos_cart)
    #print(f'Done Converting!')
    vec_cart = np.expand_dims(vec_cart,2)
    print(f'finding transformation mtx')
    mtx = np.array([[np.sin(pos_sph[:,1])*np.cos(pos_sph[:,2]), np.sin(pos_sph[:,1])*np.sin(pos_sph[:,2]), np.cos(pos_sph[:,1])]
                  ,[np.cos(pos_sph[:,1])*np.cos(pos_sph[:,2]), np.cos(pos_sph[:,1])*np.sin(pos_sph[:,2]), -np.sin(pos_sph[:,1])]
                  ,[-np.sin(pos_sph[:,2]), np.cos(pos_sph[:,2]), np.zeros_like(pos_sph[:,0])]])
    mtx = np.moveaxis(mtx, -1, 0)##Reshaping to move number of rows first.
    vec_sph = np.squeeze(np.matmul(mtx,vec_cart), axis=2)
    print(f'enjoy your vectors in spherical coords')
    return pos_sph, vec_sph

def vec_Sph2Cart(pos_sph: np.ndarray, vec_sph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts complex vectors and positions calculated in spherical coordinates to vectors in cartesian coordinate
    and returns it along with cartesian positions.
    e.g use it force vectors and etc

    Parameters
    ----------
    pos_sph: Nx3 dimensional array of positions in cartesian coords [rho, theta, phi].
    vec_sph : Nx3 dimensional array of vectors in cartesian coords [vec_R, vec_theta, vec_phi].

    Returns
    -------
    pos_cart : positions in spherical coords (Nx3)[X, Y, Z]
    vec_cart : vectors in spherical coords (NX3) [vec_X, vec_Y, vec_Z]

    """
    print(f'Converting coordinates to spherical')
    pos_cart = Sph2Cart(pos_sph)
    #print(f'Done Converting!')
    vec_sph = np.expand_dims(vec_sph,2)
    print(f'finding transformation mtx')
    mtx = np.array([[np.sin(pos_sph[:,1])*np.cos(pos_sph[:,2]), np.sin(pos_sph[:,1])*np.sin(pos_sph[:,2]), np.cos(pos_sph[:,1])]
                  ,[np.cos(pos_sph[:,1])*np.cos(pos_sph[:,2]), np.cos(pos_sph[:,1])*np.sin(pos_sph[:,2]), -np.sin(pos_sph[:,1])]
                  ,[-np.sin(pos_sph[:,2]), np.cos(pos_sph[:,2]), np.zeros_like(pos_sph[:,0])]])
    mtx = np.moveaxis(mtx, -1, 0)##Reshaping to move number of rows first.
    mtx = np.moveaxis(mtx, -1, -2)##taking tranpose of last two axis
    vec_cart = np.squeeze(np.matmul(mtx,vec_sph), axis=2)
    print(f'enjoy your vectors in spherical coords')
    return pos_cart, vec_cart

def vec_Cart2Cyl(pos_cart: np.ndarray, vec_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts complex vectors and positions calculated in cartesian coordinates to vectors in cylindrical coordinates
    and returns it along with cylindrical positions.
    e.g use it force vectors and etc

    Parameters
    ----------
    pos_cart: Nx3 dimensional array of positions in cartesian coords.
    vec_cart : Nx3 dimensional array of vectors in cartesian coords.

    Returns
    -------
    pos_cyl : positions in spherical coords (Nx3) [R, phi, z]
    vec_cyl : vectors in spherical coords (NX3) [vec_R, vec_phi, vec_z]

    """
    print(f'Converting coordinates to cylindrical')
    pos_cyl = Cart2Cyl(pos_cart)
    print(f'Done Converting!')
    vec_cart = np.expand_dims(vec_cart,2)
    print(f'finding transformation mtx')
    mtx = np.array([[np.cos(pos_cyl[:,1]), np.sin(pos_cyl[:,1]), np.zeros_like(pos_cyl[:,0])]
                  ,[-np.sin(pos_cyl[:,1]), np.cos(pos_cyl[:,1]), np.zeros_like(pos_cyl[:,0])]
                  ,[np.zeros_like(pos_cyl[:,0]), np.zeros_like(pos_cyl[:,0]), np.ones_like(pos_cyl[:,0])]])
    mtx = np.moveaxis(mtx, -1, 0)##Reshaping to move number of rows first.
    vec_cyl = np.squeeze(np.matmul(mtx,vec_cart), axis=2)
    print(f'enjoy your vectors in cylindrical coords')
    return pos_cyl, vec_cyl

def vec_Cyl2Cart(pos_cyl: np.ndarray, vec_cyl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts complex vectors and positions calculated in cartesian coordinates to vectors in cylindrical coordinates
    and returns it along with cylindrical positions.
    e.g use it force vectors and etc

    Parameters
    ----------
    pos_cyl: Nx3 dimensional array of positions in cartesian coords [R, phi, Z].
    vec_cyl : Nx3 dimensional array of vectors in cartesian coords [vec_R, vec_phi, vec_Z].

    Returns
    -------
    pos_cyl : positions in spherical coords (Nx3) [X, Y, Z]
    vec_cyl : vectors in spherical coords (NX3) [vec_X, vec_Y, vec_Z]

    """
    print(f'Converting coordinates to cylindrical')
    pos_cart = Cyl2Cart(pos_cyl).values
    print(f'Done Converting!')
    vec_cyl = np.expand_dims(vec_cyl,2)
    print(f'finding transformation mtx')
    mtx = np.array([[np.cos(pos_cyl[:,1]), np.sin(pos_cyl[:,1]), np.zeros_like(pos_cyl[:,0])]
                  ,[-np.sin(pos_cyl[:,1]), np.cos(pos_cyl[:,1]), np.zeros_like(pos_cyl[:,0])]
                  ,[np.zeros_like(pos_cyl[:,0]), np.zeros_like(pos_cyl[:,0]), np.ones_like(pos_cyl[:,0])]])
    mtx = np.moveaxis(mtx, -1, 0)##Reshaping to move number of rows first.
    mtx = np.moveaxis(mtx, -1, -2)##taking tranpose of last two axis
    vec_cart = np.squeeze(np.matmul(mtx,vec_cyl), axis=2)
    print(f'enjoy your vectors in cylindrical coords')
    return pos_cart, vec_cart

def convert_to_vel_los(xv: np.ndarray | list, reference_xv: list | np.ndarray = []) -> np.ndarray | float:
    """
    Convert Galactocentric phase-space coordinates to line-of-sight (radial) velocity.
    
    This function computes the line-of-sight velocity from Galactocentric Cartesian 
    coordinates by projecting the velocity vector onto the unit radial direction.
    Optionally subtracts a reference phase-space point before computing velocities.
    
    Parameters
    ----------
    xv : array_like
        Phase-space coordinates in Galactocentric Cartesian frame. Can be:
        - Shape (6,): Single point [x, y, z, vx, vy, vz]
        - Shape (N, 6): N points, each row [x, y, z, vx, vy, vz] 
        - Shape (M, N, 6): M objects with N points each
        Positions in kiloparsecs (kpc), velocities in km/s.
        
    reference_xv : array_like, optional
        Reference phase-space point(s) to subtract before computing v_los. Can be:
        - Empty list [] or None: No reference subtraction (default)
        - Shape (6,): Single reference [x, y, z, vx, vy, vz] applied to all
        - Shape (N, 6): N reference points for N input points (when xv is (N, 6))
        - Shape (M, 6): M reference points for M objects (when xv is (M, N, 6))
        - Shape (M, N, 6): Full reference array matching xv shape
        
    Returns
    -------
    v_los : float or ndarray
        Line-of-sight velocity in km/s. Output shape depends on input:
        - float: When xv is shape (6,)
        - Shape (N,): When xv is shape (N, 6)  
        - Shape (M, N): When xv is shape (M, N, 6)
        
    Raises
    ------
    AssertionError
        If input dimensions are incompatible or last dimension is not 6.
        
    Notes
    -----
    Line-of-sight velocity is computed as v_los = v· r̂, where r̂ is the unit 
    radial vector. After optional reference subtraction, the velocity vector
    is projected onto the direction from the origin to the position.
    
    Examples
    --------
    >>> # Single point
    >>> xv = [8.0, 0.0, 0.0, 0.0, 220.0, 0.0]  # Solar position/velocity
    >>> v_los = compute_vel_los(xv)
    
    >>> # Multiple points
    >>> xv = np.random.randn(100, 6)  # 100 random phase-space points
    >>> v_los = compute_vel_los(xv)  # Shape (100,)
    
    >>> # Multiple streams with reference
    >>> xv = np.random.randn(5, 1000, 6)  # 5 streams, 1000 points each
    >>> ref = np.random.randn(5, 6)       # 5 reference points
    >>> v_los = compute_vel_los(xv, ref)  # Shape (5, 1000)
    """
    
    # Convert inputs to numpy arrays
    xv = np.asarray(xv, dtype=float)
    if len(reference_xv) > 0:
        reference_xv = np.asarray(reference_xv, dtype=float)
    
    # Validate input shapes
    assert xv.shape[-1] == 6, f"Last dimension of xv must be 6, got {xv.shape[-1]}"
    assert xv.ndim in [1, 2, 3], f"xv must be 1D, 2D, or 3D array, got {xv.ndim}D"
    
    # Handle reference validation and broadcasting
    if len(reference_xv) > 0:
        assert reference_xv.shape[-1] == 6, f"Last dimension of reference_xv must be 6, got {reference_xv.shape[-1]}"
        
        # Validate reference shape compatibility
        if xv.ndim == 1:  # xv shape (6,)
            assert reference_xv.shape == (6,), f"For 1D xv, reference must be shape (6,), got {reference_xv.shape}"
        elif xv.ndim == 2:  # xv shape (N, 6)
            valid_ref_shapes = [(6,), (xv.shape[0], 6)]
            assert reference_xv.shape in valid_ref_shapes, \
                f"For 2D xv {xv.shape}, reference must be {valid_ref_shapes}, got {reference_xv.shape}"
        elif xv.ndim == 3:  # xv shape (M, N, 6)
            valid_ref_shapes = [(6,), (xv.shape[0], 6), (xv.shape[0], xv.shape[1], 6)]
            assert reference_xv.shape in valid_ref_shapes, \
                f"For 3D xv {xv.shape}, reference must be {valid_ref_shapes}, got {reference_xv.shape}"
        
        # Subtract reference (broadcasting handles shape compatibility)
        xv_rel = xv - reference_xv
    else:
        xv_rel = xv
    
    # Extract positions and velocities
    pos = xv_rel[..., :3]  # Shape: (..., 3)
    vel = xv_rel[..., 3:6]  # Shape: (..., 3)
    
    # Compute radial distances
    r_mag = np.linalg.norm(pos, axis=-1, keepdims=True)  # Shape: (..., 1)
    
    # Avoid division by zero
    assert np.all(r_mag > 0), "Position vectors cannot have zero magnitude"
    
    # Compute unit radial vectors
    r_unit = pos / r_mag  # Shape: (..., 3)
    
    # Compute line-of-sight velocity (dot product)
    v_los = np.sum(vel * r_unit, axis=-1)  # Shape: (...)
    
    # Return scalar for 1D input
    if xv.ndim == 1:
        return float(v_los)
    
    return v_los

#######################################################################
################Stream coordinate transformations######################
#######################################################################

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

def get_observed_stream_coords(
    xv: np.ndarray,
    xv_prog: Union[np.ndarray, list] = [],
    degrees: bool = True,
    optimizer_fit: bool = False,
    fit_kwargs: dict | None = None,
    galcen_distance: float = 8.122,
    galcen_v_sun: tuple = (12.9, 245.6, 7.78),
    z_sun: float = 0.0208,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Galactocentric phase-space coordinates to observed sky + stream coordinates.

    This function converts input Galactocentric positions + velocities into observed
    coordinates (Right Ascension, Declination, line-of-sight velocity) and stream
    coordinates (phi1, phi2). The transformation is fully vectorized and uses
    `generate_stream_coords` internally to compute stream-aligned coordinates.

    Parameters
    ----------
    xv : np.ndarray
        Nx6 array of phase-space points in Galactocentric Cartesian coordinates.
        Each row should be ``[x, y, z, vx, vy, vz]`` where positions are in
        kiloparsecs (kpc) and velocities are in km s-1. N is the number of points.
    xv_prog : np.ndarray or list, optional
        Progenitor phase-space point(s) used to define the stream frame. Can be:
        - a length-6 array representing a single progenitor ``[x, y, z, vx, vy, vz]``,
        - an Mx6 array of progenitor samples, or
        - an empty list (``[]``) to indicate no progenitor provided (default).
        If an empty list is provided the stream frame will be determined from `xv`
        or from defaults inside `generate_stream_coords` (behavior depends on implementation).
    degrees : bool, optional
        If True (default), angles (RA, Dec, phi1, phi2) are returned in **degrees**.
        If False, angles are returned in **radians**.
    optimizer_fit : bool, optional
        If True, run an optimizer-based fit when computing the stream frame (slower).
        Default is False.
    fit_kwargs : dict or None, optional
        Additional keyword arguments forwarded to the internal fitting routine used
        when ``optimizer_fit`` is True. If None (default), defaults are used.
    galcen_distance : float, optional
        Distance from Sun to Galactic center in kpc. Default is 8.122 kpc.
    galcen_v_sun : tuple, optional
        Sun's Galactocentric velocity components (Ux, Vy, Wz) in km s-1.
        Default is ``(12.9, 245.6, 7.78)``.
    z_sun : float, optional
        Height of the Sun above the Galactic midplane in kpc. Default is 0.0208 kpc.

    Returns
    -------
    ra : numpy.ndarray
        Right Ascension of each input point (shape ``(N,)``). Units: degrees if
        ``degrees=True``, otherwise radians.
    dec : numpy.ndarray
        Declination of each input point (shape ``(N,)``). Units: degrees/radians.
    v_los : numpy.ndarray
        Line-of-sight velocity for each input point (shape ``(N,)``). Units: km s-1.
    phi1 : numpy.ndarray
        Stream longitude / primary stream coordinate for each point (shape ``(N,)``).
        Units: degrees if ``degrees=True`` otherwise radians.
    phi2 : numpy.ndarray
        Stream latitude / secondary stream coordinate for each point (shape ``(N,)``).
        Units: degrees/radians.

    Raises
    ------
    ValueError
        If ``xv`` does not have shape ``(N, 6)`` or if ``galcen_v_sun`` is not length 3.
    TypeError
        If inputs are of incompatible dtypes (e.g., non-numeric arrays).

    Notes
    -----
    - The function expects Galactocentric Cartesian coordinates with the usual
      right-handed convention (x toward the Galactic center, y in the direction
      of Galactic rotation, z toward the North Galactic Pole). If your data use a
      different convention, convert first.
    - Units are assumed to be kpc for positions and km s-1 for velocities.
    - The function is vectorized: all arrays are processed without Python loops.
    - Passing an empty list for ``xv_prog`` is allowed for backward compatibility,
      but using ``None`` or a proper array is recommended. Consider changing the
      signature to ``xv_prog: Optional[np.ndarray] = None`` to avoid a mutable
      default argument.

    Examples
    --------
    >>> # single point
    >>> xv = np.array([[8.0, 0.0, 0.02, 10.0, 220.0, 5.0]])
    >>> ra, dec, vlos, p1, p2 = get_observed_stream_coords(xv)
    >>> ra.shape, dec.shape
    ((1,), (1,))

    >>> # many points
    >>> xv = np.random.randn(100, 6)  # kpc, km/s
    >>> ra, dec, vlos, p1, p2 = get_observed_stream_coords(xv, degrees=False)
    """
    
    # Ensure a batch dimension
    is_batch = (xv.ndim == 3)
    if not is_batch:
        xv = xv[None, ...]  # (1, N, 6)

    S, N, _ = xv.shape

    # Handle progenitors in batch
    if xv_prog is None or (hasattr(xv_prog, 'size') and np.asarray(xv_prog).size == 0):
        # auto-detect per-stream (handles both None and empty list/array)
        xv_prog = np.stack([
            stream[np.argmin(np.linalg.norm(stream[:, :3] - np.median(stream[:, :3], axis=0), axis=1))]
            for stream in xv
        ])
    else:
        xv_prog = np.asarray(xv_prog)
        if xv_prog.ndim == 1:
            assert xv_prog.shape == (6,), f"1D xv_prog must have shape (6,), got {xv_prog.shape}"
            xv_prog = np.broadcast_to(xv_prog, (S, 6))
        else:
            # xv_prog is 2D - should be either (1, 6) or (S, 6)
            assert xv_prog.shape == (S, 6) or xv_prog.shape == (1, 6), \
                f"2D xv_prog must have shape ({S}, 6) or (1, 6), got {xv_prog.shape}"
            
            # If it's (1, 6), broadcast to (S, 6)
            if xv_prog.shape == (1, 6):
                xv_prog = np.broadcast_to(xv_prog, (S, 6))

    # 1) Flatten for Agama transforms
    flat = xv.reshape(-1, 6)
    x, y, z, vx, vy, vz = flat.T

    # 2) Get Galactic → ICRS
    l, b, dist, pml, pmb, vlos = agama.getGalacticFromGalactocentric(
        x, y, z, vx, vy, vz,
        galcen_distance=galcen_distance,
        galcen_v_sun=galcen_v_sun,
        z_sun=z_sun,
    )
    ra, dec = agama.transformCelestialCoords(agama.fromGalactictoICRS, l, b)

    if degrees:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    # 3) Reshape back to (S, N)
    ra   = ra.reshape(S, N)
    dec  = dec.reshape(S, N)
    vlos = vlos.reshape(S, N)

    # 4) Compute (phi1, phi2) in one batch call
    phi1, phi2 = generate_stream_coords(
        xv, xv_prog,
        degrees=degrees,
        optimizer_fit=optimizer_fit,
        fit_kwargs=fit_kwargs,
    )
    # phi1, phi2 are each shape (S, N) when xv.ndim==3

    # 5) Drop the batch axis if it was a single stream
    if not is_batch:
        return ra[0], dec[0], vlos[0], phi1[0], phi2[0]
    return ra, dec, vlos, phi1, phi2


def produce_stream_plot(
    xv_stream: np.ndarray,
    color: str = "ro",
    ax: np.ndarray | None = None,
    xv_prog: np.ndarray | list[float] | None = None,
    alpha_lim: tuple[float | None, float | None] = (None, None),
    delta_lim: tuple[float | None, float | None] = (None, None),
    Phi1_lim: tuple[float | None, float | None] = (None, None),
    Phi2_lim: tuple[float | None, float | None] = (None, None),
    ms: float = 0.5,
    mew: float = 0,
) -> tuple[Figure | None, np.ndarray]:
    
    """
    Produce a 2×3 panel of stream diagnostic plots for a given particle set.

    Parameters
    ----------
    xv_stream : ndarray, shape (N, 6)
        Stream particle positions and velocities in galactocentric coordinates
        (x, y, z, vx, vy, vz).
    color : str, optional
        Matplotlib format string for line/marker color and style (default 'ro').
    ax : array_like of Axes or None, optional
        If provided, must be a (2,3) array of existing Matplotlib Axes to plot into.
        If None (default), a new figure and axes are created.
    xv_prog : array_like, shape (6,), optional
        Progenitor’s position and velocity for computing observed coordinates.
    alpha_lim : tuple of float or None, optional
        x-axis limits for RA plots in degrees (default (None, None)).
    delta_lim : tuple of float or None, optional
        y-axis limits for Dec plots in degrees (default (None, None)).
    Phi1_lim : tuple of float or None, optional
        x-axis limits for φ₁ stream longitude (default (None, None)).
    Phi2_lim : tuple of float or None, optional
        y-axis limits for φ₂ stream latitude (default (None, None)).
    ms : float, optional
        Marker size for scatter/line plots (default 0.5).
    mew : float, optional
        Marker edge width for scatter/line plots (default 0).

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if a new one was created, otherwise None.
    ax : ndarray of Axes, shape (2, 3)
        Array of Axes where the data have been plotted.
    """
    
    return_fig_ax = False
    if ax is None:
        return_fig_ax = True
        fig, ax = plt.subplots(2, 3, figsize=(9, 6), dpi=300)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)        
        
    for i, ax_dummy in enumerate(ax.flatten()):
        if ax[i // ax.shape[1], i % ax.shape[1]] != ax[0, 1]:  # Skip ax[0, 1]
            ax_dummy.set_aspect('equal')
        
        ax[0, 0].set_xlabel(r'$\alpha$ [deg]')
        ax[0, 0].set_ylabel(r'$\delta$ [deg]')
        ax[0, 1].set_xlabel(r'$\alpha$ [deg]')
        ax[0, 1].set_ylabel(r'$v_{\rm los}$ [km/s]')
        ax[0, 2].set_xlabel(r'$\phi_1$ [deg]')
        ax[0, 2].set_ylabel(r'$\phi_2$ [deg]')

        ax[1, 0].set_xlabel('X [kpc]')
        ax[1, 0].set_ylabel('Y [kpc]')
        ax[1, 1].set_xlabel('X [kpc]')
        ax[1, 1].set_ylabel('Z [kpc]')
        ax[1, 2].set_xlabel('Y [kpc]')
        ax[1, 2].set_ylabel('Z [kpc]')

    ra, dec, vlos, phi1, phi2 = get_observed_stream_coords(xv_stream, xv_prog)
    
    ax[0, 0].plot(ra, dec, color, ms=ms, mew=mew)
    ax[0, 1].plot(ra, vlos, color, ms=ms, mew=mew)
    ax[0, 2].plot(phi1, phi2, color, ms=ms, mew=mew)

    ax[0, 0].set_xlim(alpha_lim)
    ax[0, 0].set_ylim(delta_lim)
    ax[0, 1].set_xlim(alpha_lim)
    
    ax[0, 2].set_xlim(Phi1_lim)
    ax[0, 2].set_ylim(Phi2_lim)
    
    ax[1, 0].plot(xv_stream[:,0], xv_stream[:,1], color, ms=ms, mew=mew)
    ax[1, 1].plot(xv_stream[:,0], xv_stream[:,2], color, ms=ms, mew=mew)
    ax[1, 2].plot(xv_stream[:,1], xv_stream[:,2], color, ms=ms, mew=mew)
    plt.tight_layout()
    if return_fig_ax: return fig, ax

def return_stream_plots(Nbody_out, time_step=-1, x_axis=0, y_axis=2, LMC_traj=[], three_d_plot=False, interactive=False):

    """
    Generate a three‐panel evolution plot for an N‐body stream simulation.

    Parameters
    ----------
    Nbody_out : dict
        Dictionary containing simulation outputs with keys:
        - 'times' : array_like, shape (T,)
        - 'prog_xv' : ndarray, shape (T,6)
        - 'part_xv' : ndarray, shape (N,T,6) or (N,6)
        - 'bound_mass' : array_like, shape (T,), optional
    time_step : int, optional
        Index of the snapshot to plot for particle positions (default -1, last snapshot).
    x_axis : int, {0,1,2}, optional
        Coordinate index for the x‐axis in the right panel (0→X,1→Y,2→Z; default 0).
    y_axis : int, {0,1,2}, optional
        Coordinate index for the y‐axis in the right panel (default 2).
    LMC_traj : ndarray, shape (M,4) or empty, optional
        Trajectory of the LMC to overplot, where each row is (t, x, y, z).
        If empty (default), no LMC trajectory is drawn.
    three_d_plot : bool, optional
        If True, the middle panel is a 3D trajectory; otherwise it shows bound fraction vs time.
    interactive : bool, optional
        If True, enable interactive Matplotlib widget mode (requires IPython).
        Only valid when `three_d_plot=True`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : list of Axes
        List of three Axes objects: [distance vs time, middle panel, right panel].
    """
    
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    # Set up backend check
    if interactive and not three_d_plot:
        raise ValueError("Interactive mode only works with 3D plots")
        
    # Configure matplotlib backend
    if interactive:
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'widget')
        except ImportError:
            raise RuntimeError("Interactive mode requires IPython")
    
    # Create figure and axes
    fig = plt.figure(figsize=(12, 3), dpi=300)
    ax = [None, None, None]
    ax[0] = fig.add_subplot(131)
    ax[2] = fig.add_subplot(133)
    
    # Create middle axis with 3D projection if needed
    if three_d_plot:
        ax[1] = fig.add_subplot(132, projection='3d')
    else:
        ax[1] = fig.add_subplot(132)
    
    plt.subplots_adjust(wspace=0.25)

    # Common labels
    axes_label = {0:r'{\bf X [kpc]}', 1:r'{\bf Y [kpc]}', 2:r'{\bf Z [kpc]}'}
    ax[0].set_xlabel(r'{\bf T [Gyr]}'); ax[0].set_ylabel(r'{\bf $\mathbf{d_{cen}}$ [kpc]}')
    ax[2].set_xlabel(axes_label[x_axis]); ax[2].set_ylabel(axes_label[y_axis])

    # Configure middle axis labels
    if three_d_plot:
        ax[1].set_xlabel(axes_label[0])
        ax[1].set_ylabel(axes_label[1])
        ax[1].set_zlabel(axes_label[2])
        # ax[1].zaxis.set_label_position("left")
        
    else:
        ax[1].set_xlabel(r'{\bf T [Gyr]}'); ax[1].set_ylabel(r'{\bf Bound Frac}')

    # --- Plot common elements ---
    # Distance vs time plot
    distances = np.linalg.norm(Nbody_out['prog_xv'][:, :3], axis=1)
    ax[0].plot(Nbody_out['times'][:][:time_step], distances[:time_step], c='r', lw=0.5)
    
    # --- Middle panel content ---
    if not three_d_plot:
        try:  # Original bound fraction plot
            bound_frac = Nbody_out['bound_mass'][:] / Nbody_out['bound_mass'][0]
            ax[1].plot(Nbody_out['times'][:], bound_frac, c='r', lw=0.5)
        except Exception as e:
            ax[1].plot(Nbody_out['times'][:], np.zeros_like(Nbody_out['times'][:]), c='r', lw=0.5)
    else:  # 3D trajectory plot
        from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
        # Plot progenitor trajectory
        prog_x = Nbody_out['prog_xv'][:, 0]
        prog_y = Nbody_out['prog_xv'][:, 1]
        prog_z = Nbody_out['prog_xv'][:, 2]
        
        if time_step == -1:
            ax[1].plot(prog_x, prog_y, prog_z, c='r', lw=0.5)
        else:
            ax[1].plot(prog_x[:time_step+1], prog_y[:time_step+1], prog_z[:time_step+1], c='r', lw=0.5)
        
        # Plot particles
        try:
            part_x = Nbody_out['part_xv'][:, time_step, 0]
            part_y = Nbody_out['part_xv'][:, time_step, 1]
            part_z = Nbody_out['part_xv'][:, time_step, 2]
        except Exception as e:
            part_x = Nbody_out['part_xv'][:, 0]
            part_y = Nbody_out['part_xv'][:, 1]
            part_z = Nbody_out['part_xv'][:, 2]
        ax[1].scatter(part_x, part_y, part_z, s=0.15, alpha=0.15, c='m')
        # Make axis panes transparent
        ax[1].xaxis.pane.fill = False
        ax[1].yaxis.pane.fill = False
        ax[1].zaxis.pane.fill = False
        ax[1].grid(False) # This line turns off the grid
        
        # Automatically determine best view angle using PCA
        try:
            # Get particle coordinates
            try:
                part_x = Nbody_out['part_xv'][:, time_step, 0]
                part_y = Nbody_out['part_xv'][:, time_step, 1]
                part_z = Nbody_out['part_xv'][:, time_step, 2]
            except Exception as e:
                part_x = Nbody_out['part_xv'][:, 0]
                part_y = Nbody_out['part_xv'][:, 1]
                part_z = Nbody_out['part_xv'][:, 2]

            # Compute PCA
            data = np.vstack([part_x, part_y, part_z]).T
            data_centered = data - np.mean(data, axis=0)
            
            if len(data) > 1 and not np.allclose(data_centered, 0):
                cov = np.cov(data_centered, rowvar=False)
                eigen_vals, eigen_vecs = np.linalg.eigh(cov)
                order = eigen_vals.argsort()[::-1]
                eigen_vecs = eigen_vecs[:, order]
                
                # Get third principal component direction
                pc3 = eigen_vecs[:, 2]
                direction = -pc3  # View along PC3 direction
                dx, dy, dz = direction

                # Calculate angles
                azim = np.degrees(np.arctan2(dy, dx))
                r = np.linalg.norm(direction)
                phi = np.arccos(dz/r) if r != 0 else 0
                elev = 90 - np.degrees(phi)

                # Apply view angle
                ax[1].view_init(elev=max(0, min(90, elev)), 
                            azim=azim % 360)
            else:
                ax[1].view_init(elev=30, azim=45)
        except Exception as e:
            ax[1].view_init(elev=30, azim=45)
        

    # --- Common pericenter calculations ---
    inverted_distances = -distances
    peaks, _ = find_peaks(inverted_distances)
    if len(peaks) > 0:
        for peak in peaks:
            ax[0].axvline(Nbody_out['times'][peak], color='gray', lw=0.5, alpha=0.5, ls='--')
            if not three_d_plot:
                ax[1].axvline(Nbody_out['times'][peak], color='gray', lw=0.5, alpha=0.5, ls='--')

    # --- Right panel content ---
    # Initial conditions star
    ax[2].scatter(Nbody_out['prog_xv'][0, x_axis], Nbody_out['prog_xv'][0, y_axis], 
                  facecolor='none', edgecolor='r', s=50, marker='*', linewidth=0.3, zorder=3)
    ax[2].scatter(Nbody_out['prog_xv'][-1, x_axis], Nbody_out['prog_xv'][-1, y_axis], 
                  facecolor='none', edgecolor='k', s=50, marker='*', linewidth=0.3, zorder=3)
    
    # Particles scatter
    try:
        ax[2].scatter(Nbody_out['part_xv'][:, time_step, x_axis], 
                      Nbody_out['part_xv'][:, time_step, y_axis], s=0.15, alpha=0.15, c='m')
    except Exception as e:
        ax[2].scatter(Nbody_out['part_xv'][:, x_axis], 
                      Nbody_out['part_xv'][:, y_axis], s=0.15, alpha=0.15, c='m')
    
    # Progenitor trajectory
    if time_step == -1:
        ax[2].plot(Nbody_out['prog_xv'][:, x_axis], Nbody_out['prog_xv'][:, y_axis], lw=0.5, ls='--', c='gray')
    else:
        ax[2].plot(Nbody_out['prog_xv'][:time_step+1, x_axis], 
                   Nbody_out['prog_xv'][:time_step+1, y_axis], lw=0.5, ls='--', c='gray')

    # --- Axis limit management ---
    # For 3D plot
    if three_d_plot:
        ax[1].relim()
        ax[1].autoscale_view()
        ax[1].set_autoscale_on(False)  # Lock 3D axis limits
    
    # For right panel (2D)
    ax[2].relim()
    ax[2].autoscale_view()
    ax[2].autoscale(False)
    
    # For left panel
    ax[0].autoscale(False)

    # --- LMC trajectory handling ---
    if len(LMC_traj) > 0:
        # Left panel
        ax[0].plot(LMC_traj[:, 0], np.linalg.norm(LMC_traj[:, 1:], axis=1), 
                   c='gray', lw=2, alpha=0.4, ls='--')
        
        # Right panel
        ax[2].plot(LMC_traj[:, x_axis+1], LMC_traj[:, y_axis+1], 
                   c='gray', lw=2, alpha=0.4, ls='--')
        ax[2].scatter(LMC_traj[-1, x_axis+1], LMC_traj[-1, y_axis+1], 
                      facecolor='none', edgecolor='gray', s=50, marker='*', linewidth=1, zorder=3)
        
        # 3D panel
        if three_d_plot:
            ax[1].plot(LMC_traj[:, 1], LMC_traj[:, 2], LMC_traj[:, 3], 
                       c='gray', lw=2, alpha=0.4, ls='--')
            ax[1].scatter(LMC_traj[-1, 1], LMC_traj[-1, 2], LMC_traj[-1, 3], 
                          facecolor='none', edgecolor='gray', s=50, marker='*', linewidth=1, zorder=3)

    # Green reference lines
    if x_axis == 2 or y_axis == 2:
        ax[2].plot([-10, 10], [0, 0], c='g', alpha=0.5)

        
    # Interactive controls setup
    if interactive and three_d_plot:
        # Enable mouse interaction
        ax[1].mouse_init()
        
        # Optional: Add rotation controls
        def on_move(event):
            if event.inaxes == ax[1]:
                ax[1].view_init(elev=ax[1].elev, azim=ax[1].azim)
                
        fig.canvas.mpl_connect('motion_notify_event', on_move)
     
    # Don't block execution in interactive mode
    if not interactive:
        plt.show()
    else:
        plt.show(block=False)           

######################################################################################
#################################### Utilities #######################################
######################################################################################

## Obtain Star formation rates. 
def get_SFR(sim_dir=None, nsnap=None, 
              rmax=None, dt=1 / 1000, # 1 Myr in Gyr
              part = None, cosmology=None,
              tmax=13.7965284579874688,  # this is for Latte galaxies
              tmin_sfr=0.1,  # Gyr, this is here because there is a bug I'm still working out :/
             ):
    
    if part is None:
        if sim_dir is None or nsnap is None:
            raise ValueError("sim_dir and nsnap cannot be None if part is None.")
        part = cc.read(sim_dir, nsnap, ['star'])
    
    if rmax is None:
        try:
            hal = halo.io.IO.read_catalogs("index", nsnap, sim_dir, rockstar_directory='halo/rockstar_dm_new/')
            rmax = 5 * hal["star.radius.50"][hal["host.index"][0]]
        except Exception as e:
            print(e)
            rmax = 20
        
        print('f rmax:',rmax)
    
    if cosmology == None:
        import utilities as ut
        cosmology = ut.cosmology.CosmologyClass(source="agora")

    # select stars within a particular radius
    mask = part["star"].prop("host.distance.total") <= rmax
    ages = part["star"].prop("age")[mask]
    formation_time = part["star"].prop("form.time")[mask]
    formation_mass = part["star"].prop("form.mass")[mask]

    time_edges = np.arange(tmax, 0, -dt)[::-1]
    SFRs, time_edges = np.histogram(
        formation_time, weights=formation_mass / (dt * 1e9), bins=time_edges
    )

    t_mask = time_edges[:-1] >= tmin_sfr

    z_edges = cosmology.convert_time("redshift", "time", time_edges[:-1][t_mask])

    df = pd.DataFrame(
        {"time": time_edges[:-1][t_mask], "redshift": z_edges, "sfr": SFRs[t_mask]}
    )

    return df

##perform loess smoothing
def loess_smoothing(x: np.ndarray | list[float], y: np.ndarray | list[float], 
                    frac: float = 0.5, **kwargs: Any) -> np.ndarray:
    """
    Perform Loess smoothing on the given data.

    Parameters
    ----------
    x (array-like): The x values of the data points to be smoothed.
    y (array-like): The y values of the data points to be smoothed.
    frac (float, optional): The fraction of data points to be used in each local regression.
        Must be in the range [0, 1]. Defaults to 0.5.

    Returns
    -------
    array-like: The smoothed y values.

    Raises
    ------
    ValueError: If x and y have different lengths.
    ValueError: If frac is not in the range [0, 1].
    
    """
    # Check that x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Check that frac is in the range [0, 1]
    if not 0 <= frac <= 1:
        raise ValueError("frac must be in the range [0, 1].")

    # Perform Loess smoothing
    import statsmodels.api as sm
    loess = sm.nonparametric.lowess(y, x, frac=frac, **kwargs)

    # Return smoothed values
    return loess ##x, y

## Returns arr minus masked indices. 
def return_unmasked_array(
    np_arr: np.ndarray, mask_indices: list[int] | np.ndarray
) -> np.ndarray:
    """
    Return elements of an array excluding the masked indices.

    Parameters
    ----------
    np_arr : numpy.ndarray
        Input array from which elements will be selected.
    mask_indices : array-like of int
        Indices to mask (exclude).

    Returns
    -------
    numpy.ndarray
        Array containing elements of `np_arr` except those at `mask_indices`.

    Examples
    --------
    >>> arr = np.array([10, 20, 30, 40, 50])
    >>> return_unmasked_array(arr, [1, 3])
    array([10, 30, 50])
    """
    if len(mask_indices) == 0:
        return np_arr
    mask = np.zeros(len(np_arr), dtype=bool)
    mask[mask_indices] = True
    return np_arr[~mask]


## Remove DM substructure tracked by rockstar. 
def remove_dark_substructure(sim_dir: Path | 'str', nsnap: int,
                             part: dict|None = None, hal_IDs: list | None = None,
                             mass_cuts: list[float, float] = [1e9, 1e11],  # [lower_limit, upper_limit] on mass
                             return_sub: bool = False, **kwargs: Any) -> np.ndarray:
    """
    Removes substructure by halo mass_limits using particle assignments from Rockstar.

    Parameters
    ----------
    sim_dir (str): The simulation directory where the required data is stored.
    nsnap (int): The snapshot index indicating which snapshot to process.
    part (dict, optional): Particle directory from the Gizmo snapshot. 
                           If not provided, it will be read from the simulation directory and snapshot index.
    hal_IDs (list, optional): A list of halo IDs. 
                              If provided, the function will remove substructures associated with these halos 
                              based on the mass limits. If not provided, the function will use `mass_cuts` 
                              to determine the halo IDs to remove.
    mass_cuts (list, optional): A list containing two elements representing the mass limits 
                                [lower_limit, upper_limit] on halo mass in solar masses.
    return_sub (bool, optional): If True, the function also returns the removed particle indices.
    **kwargs: Additional keyword arguments to be passed to underlying functions.

    Returns
    -------
    np.array: An array containing the indices of DM particles after removing substructures 
              by mass limits on halos.

    Examples
    --------
    # Remove substructures with specified hal_IDs and get the remaining DM particle indices
    remaining_indices = remove_dark_substructure(sim_dir="/path/to/sim_directory", 
                                                 nsnap=500, 
                                                 hal_IDs=[1, 2, 3],
                                                 mass_cuts=[1e10, 1e12],
                                                 return_sub=False)

    # Remove substructures using mass_cuts and get the remaining DM particle indices and removed substructure indices
    remaining_indices, removed_indices = remove_dark_substructure(sim_dir="/path/to/sim_directory", 
                                                                  nsnap=500, 
                                                                  mass_cuts=[1e9, 1e11],
                                                                  return_sub=True)
                                                                      
    Notes
    -----
    - Particle assignments table should be precomputed using a C++ pipeline and provided.
    - If `hal_IDs` is not provided, the function will impose `mass_cuts` to determine the halos to remove.
    - Please include the rockstar_dm_new/ catalogs or the catalogs with dark particle assignments. 
    
    """
    import halo_analysis as halo
    if hal_IDs is None:
        hal = halo.io.IO.read_catalogs('index', nsnap, sim_dir, **kwargs)
        mass_filt = np.where((hal['mass'] > mass_cuts[0]) & ((hal['mass'] < mass_cuts[1])))[0]
        hal_IDs = hal['id'][mass_filt]
    
    final_dict = read_particle_assignments(sim_dir, nsnap, **kwargs)
    dark_IDs = np.hstack([final_dict[key] for key in hal_IDs])
    
    if part is None:
        part = read(sim_dir, nsnap, sort_dark_by_id=True, verbose=False, **kwargs)
    
    remove_dark_inds = np.isin(part['dark']['id'], dark_IDs)
    
    if return_sub:
        return np.where(~remove_dark_inds)[0], np.where(remove_dark_inds)[0]##main halo, substrucutre
    
    return np.where(~remove_dark_inds)[0]

## Quick N-body force computation [only works for limited N].
from numba import njit, prange, set_num_threads

@njit(fastmath=True, cache=True, inline='always')
def _get_force_kernel(r2: float, h: float, kernel_id: int) -> float:
    """
    Selectable force kernel function.
    Returns the 1/r^3 equivalent factor for the force calculation.

    kernel_id:
      0 = Newtonian (regularized)
      1 = Plummer
      2 = Spline (Monaghan 1992)
      3 = Dehnen k=1 (falcON default)
      4 = Dehnen k=2
    """
    # --- 0: Newtonian (with tiny regularization for safety) ---
    if kernel_id == 0:
        return 1.0 / (r2* np.sqrt(r2))

    # --- 1: Plummer Kernel ---
    if kernel_id == 1:
        denom = r2 + h * h
        return 1.0 / (denom * np.sqrt(denom))
 
    # --- 2: Dehnen k=1 (C2 correction, falcON default) ---
    # Pot1 = [(x^2 + h^2)^(-1/2) + 0.5*h^2*(x^2 + h^2)^(-3/2)] 
    # kernel_force = -dPot/dx = x * [(x^2 + h^2)^(-3/2) + (3/2)*h^2*(x^2 + h^2)^(-5/2)].
    # Remember the kernel is 1/x of that for 1/L^3 units.
    if kernel_id == 2:  # P1
        denom = r2 + h * h
        sqrt_denom = np.sqrt(denom)
        term1 = 1.0 / (denom * sqrt_denom)           # 1/(r²+h²)^(3/2) - same as Plummer
        term2 = 1.5 * h * h / (denom * denom * sqrt_denom)  # 1.5*h²/(r²+h²)^(5/2)
        return term1 + term2
   
    # --- 3: Dehnen k=2 (C4 correction) ---
    # Pot1 = [(x^2 + h^2)^(-1/2) + (1/2)*h^2*(x^2 + h^2)^(-3/2) + (3/4)*h^4*(x^2 + h^2)^(-5/2)] 
    # kernel_force = -dPot/dx = x * [(x^2 + h^2)^(-3/2) + (3/2)*h^2*(x^2 + h^2)^(-5/2) + (15/4)*h^4*(x^2 + h^2)^(-7/2)].
    if kernel_id == 3:  # P2
        denom = r2 + h * h
        sqrt_denom = np.sqrt(denom)
        hh = h * h
        term1 = 1.0 / (denom * sqrt_denom)                    # base term
        term2 = 1.5 * hh / (denom * denom * sqrt_denom)       # h² correction
        term3 = 3.75 * hh * hh / (denom * denom * denom * sqrt_denom)  # h⁴ correction
        return term1 + term2 + term3
    
    r = np.sqrt(r2)    

    # --- Below are all compact kernels. ---
    # For all compact-support kernels, the force is Newtonian for r >= h
    if r > h:
        return 1.0 / (r * r * r)

    hinv = 1.0 / h
    h3inv = hinv * hinv * hinv
    q = r * hinv
    q2 = q * q
    
    # --- 4: Spline Kernel --- (Monaghan 1992)
    if kernel_id == 4:
        if q <= 0.5:
            return h3inv * (10.666666666666666 + q2 * (-38.4 + 32.0 * q))
        else:
            return h3inv * (21.333333333333333 - 48.0 * q + 38.4 * q2
                           - 10.666666666666667 * q2 * q
                           - 0.06666666666666667 / (q2 * q))
            
    # Fallback for safety, though the wrapper should prevent this
    return 0.0

@njit(parallel=True, fastmath=True, cache=True)
def _compute_forces_direct(pos: np.ndarray, mass: np.ndarray, h: np.ndarray, 
                           kernel_id: int, r_eps: float) -> np.ndarray:
    """
    fast N-body force computation with a selectable kernel.
    
    Parameters
    ----------
    pos : np.ndarray, shape (N, 3)
        Particle positions.
    mass : np.ndarray, shape (N,)
        Particle masses.
    h : np.ndarray, shape (N,)
        Softening lengths for each particle.
    kernel_id : int
        Identifier for the gravitational softening kernel to use.
        Maps to a specific kernel function (e.g., Plummer, spline, etc.).
    r_eps : float
        Small value added to distances to avoid divide-by-zero errors.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Computed accelerations for each particle in Cartesian coordinates.
    """
    
    N = pos.shape[0]
    forces = np.zeros((N, 3), dtype=pos.dtype)

    for i in prange(N):
        fx, fy, fz = 0.0, 0.0, 0.0
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        hi = h[i]

        for j in range(N):
            if i == j:
                continue
            
            # cache mass and h for j to reduce indexing.
            mj = mass[j]
            hj = h[j]
            
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi

            r2 = dx*dx + dy*dy + dz*dz + r_eps*r_eps
                        
            # The softening length h_ij is the max of the two particles
            # combine softening lengths
            h_ij = hi if hi >= hj else hj

            # Get the force factor from our unified kernel function
            kernel_val = _get_force_kernel(r2, h_ij, kernel_id)

            factor = mj * kernel_val

            fx += factor * dx
            fy += factor * dy
            fz += factor * dz

        forces[i, 0] = fx
        forces[i, 1] = fy
        forces[i, 2] = fz

    return forces

def compute_nbody_forces(pos: np.ndarray, mass: np.ndarray | float, softening: float = 0.0, 
                         G: float = 4.30092e-6, kernel: str = 'spline', 
                         nthreads: int | None = None, dtype: str = 'float64') -> np.ndarray:
    """
    Compute N-body gravitational accelerations (direct O(N^2) pairwise) with numba.
    Comparable speeds to fastest FMM/Tree-PM codes for <100_000 bodies parallelized with 48 cores.
    CAUTION: Even with optimizations and kernels, the direct-Nbody is infact O(N^2). Be aware!!
    
    User is responsible for correct pos, mass unit conversions...
    
    Parameters
    ----------
    pos : array_like, shape (N, 3)
        Particle positions.
    mass : array_like, shape (N,) or float
        Particle masses. If float, all particles have the same mass.
    softening : float or array_like, shape (N,), optional
        Softening length(s). If scalar, a same-valued array of length N is used.
        Defaults to 0.0, which implies pure Newtonian gravity.
    G : float, optional
        Gravitational constant. Defaults to 4.30092e-6. [Length: kpc, vel: km/s, Mass: Msun]
    kernel : {'newtonian', 'plummer', 'spline', 'dehnen_k1', 'dehnen_k2'}, optional
        Softening kernel to use:
        - 'newtonian': Pure 1/r^2 gravity with small epsilon regularization
        - 'plummer': Plummer softening (always softened)
        - 'dehnen_k1': Dehnen P1 (C2 correction) kernel (falcON default)
        - 'dehnen_k2': Dehnen P2 (C4 correction) kernel 
        COMPACT SUPPORT KERNELS:
        - 'spline': Cubic spline kernel (Monaghan 1992), compact support.
    nthreads : int or None, optional
        Number of threads for numba parallel loops.
    dtype : {'float64', 'float32'} or numpy dtype, optional
        Floating type to use for computation. Default 'float64'.

    Returns
    -------
    forces : ndarray, shape (N, 3)
        Gravitational accelerations on each particle.

    Notes
    -----
    - Spline kernels have compact support at radius `softening` and become
      exactly Newtonian for r >= softening.
    - The Plummer kernel is always softened and never becomes purely Newtonian.
    - For momentum conservation, softening lengths are combined using max().
    - Takes about 200secs for 1M bodies on 48 cores. 
    """
    # Validate dtype
    if isinstance(dtype, str):
        if dtype not in ('float64', 'float32'):
            raise ValueError("dtype must be 'float64' or 'float32'")
        dtype_np = np.float64 if dtype == 'float64' else np.float32
    else:
        dtype_np = np.dtype(dtype).type
    
    # Convert inputs and ensure contiguous arrays of chosen dtype
    pos = np.ascontiguousarray(np.asarray(pos, dtype=dtype_np))
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be shape (N, 3)")
        
    N = pos.shape[0]

    # Convert mass
    if np.isscalar(mass): mass = np.full(N, mass, dtype=dtype_np)

    else:
        mass = np.ascontiguousarray(np.asarray(mass, dtype=dtype_np))    
        if mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
            raise ValueError("mass must be shape (N,)")
    
    # Set numba threads if requested
    if nthreads is not None:
        set_num_threads(int(nthreads))
    
    # Prepare softening/smoothing parameter: ALWAYS an array of length N
    if np.isscalar(softening):
        soft_arr = np.full(N, float(softening), dtype=dtype_np)
    else:
        soft_arr = np.ascontiguousarray(np.asarray(softening, dtype=dtype_np))
        if soft_arr.shape[0] != N:
            raise ValueError("softening array must have length N")

    if np.any(soft_arr == 0):
        print("Warning: softening=0 detected, Using kernel='newtonian'.")
        kernel = 'newtonian'
    
    # Map the kernel string to a unique integer ID
    kernel_map = {
        'newtonian': 0,
        'plummer': 1,
        'dehnen_k1': 2,
        'dehnen_k2': 3,
        'spline': 4,
    }
    
    if kernel.lower() not in kernel_map:
        print(f'{kernel} not found in available. using spline kernel.')
        kernel = 'spline'
    
    kernel_id = kernel_map[kernel.lower()]    
    
    # Automatically select r_eps appropriate for dtype
    if dtype_np is np.float32:
        r_eps = dtype_np(1e-4)
    else:
        r_eps = dtype_np(1e-6)
    
    # func call to compute forces
    return G * _compute_forces_direct(pos, mass, soft_arr, kernel_id, r_eps)


def compute_iterative_boundness(
    positions_dark: np.ndarray, 
    velocity_dark: np.ndarray, 
    mass_dark: np.ndarray | float,
    positions_star: np.ndarray | None = None, 
    velocity_star: np.ndarray | None = None, 
    mass_star: np.ndarray | float | None = None,
    center_position: np.ndarray | list = [], 
    center_velocity: np.ndarray | list = [],
    recursive_iter_converg: int = 50, 
    potential_compute_method: str = 'tree',
    BFE_lmax: int = 8, 
    softening_tree: float = 0.03, 
    G: float = 4.30092e-6,
    center_method: str = 'density_peak', 
    center_params: Any = None, 
    center_vel_with_KDE: bool = True,
    center_on: str = 'star', 
    vel_rmax: float = 5.0, 
    tol_frac_change: float = 0.0001,
    verbose: bool = True, 
    return_history: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    
    """
    Compute boundness of DM (and optionally stars) via iterative unbinding.
    
    Uses either Agama multipole expansion (BFE) or pyfalcon tree code to compute
    gravitational potential. Iteratively removes unbound particles until convergence.
    Boundness criterion: total energy = potential + kinetic < 0.

    Parameters
    ----------
    positions_dark : (N_d, 3) array
        Dark matter positions (kpc).
    velocity_dark : (N_d, 3) array
        Dark matter velocities (km/s).
    mass_dark : (N_d,) array or scalar
        Dark matter masses (M_sun). If scalar, applied to all particles.
    positions_star : (N_s, 3) array, optional
        Star positions (kpc).
    velocity_star : (N_s, 3) array, optional
        Star velocities (km/s).
    mass_star : (N_s,) array or scalar, optional
        Star masses (M_sun). If scalar, applied to all particles.
    center_position : (3,) array, optional
        User-specified center position (kpc). If empty, computed automatically.
    center_velocity : (3,) array, optional
        User-specified center velocity (km/s). If empty, computed automatically.
    
    Algorithm Parameters
    --------------------
    recursive_iter_converg : int, default=50
        Maximum iterations for convergence.
    potential_compute_method : {'tree', 'bfe'}, default='tree'
        Potential computation method:
        - 'tree': pyfalcon tree code (more accurate with sensible choice of softening)
        - 'bfe': Agama multipole expansion (approximate.)
    BFE_lmax : int, default=8
        Maximum multipole order for BFE method.
    softening_tree : float, default=0.03
        Gravitational softening length for tree method (kpc). 
        Recommended: ~0.1-1× your simulation's force softening.
    G : float, default=4.30092e-6
        Gravitational constant in units (L:kpc, vel: km/s, M:Msun units).
    tol_frac_change : float, default=0.0001
        Convergence tolerance on bound fraction change between iterations.

    Centering Parameters
    -------------------
    center_method : {'density_peak', 'kde', 'shrinking_sphere'}, default='density_peak'
        Method for finding center position.
    center_vel_with_KDE : bool, default=True
        Use KDE for velocity centering (vs simple average within vel_rmax).
    center_on : {'star', 'dark', 'both'}, default='star'
        Which particle type to use for center finding.
    center_params : dict, optional
        Additional parameters for center finding method.
    vel_rmax : float, default=5.0
        Maximum radius for velocity centering when not using KDE (kpc).

    Output Control
    --------------
    verbose : bool, default=True
        Print iteration diagnostics.
    return_history : bool, default=False
        Return bound masks for each iteration.

    Returns
    -------
    If no stars provided:
        bound_dark_final : (N_d,) bool array
            Final bound mask for dark matter (True=bound, False=unbound).
        [bound_history_dm] : list of arrays, optional
            Bound masks for each iteration if return_history=True.
            
    If stars provided:
        bound_dark_final : (N_d,) bool array
        bound_star_final : (N_s,) bool array  
        [bound_history_dm, bound_history_star] : list of arrays, optional
    
    Notes
    -----
    - Ensure agama.setUnits() matches your data units before calling
    - For tree method: softening should ideally match simulation resolution.
    - For BFE method: higher lmax = more accurate but slower        
    """

    # Input validation and normalization
    # ==================================

    # Validate string parameters (case-insensitive)
    potential_compute_method = potential_compute_method.lower()
    center_method = center_method.lower()
    center_on = center_on.lower()
    
    valid_potential_methods = ['tree', 'bfe']
    valid_center_methods = ['density_peak', 'kde', 'shrinking_sphere']
    valid_center_on = ['star', 'dark', 'both']
    
    assert potential_compute_method in valid_potential_methods, \
        f"potential_compute_method must be one of {valid_potential_methods}, got '{potential_compute_method}'"
    assert center_method in valid_center_methods, \
        f"center_method must be one of {valid_center_methods}, got '{center_method}'"
    assert center_on in valid_center_on, \
        f"center_on must be one of {valid_center_on}, got '{center_on}'"
    
    # Validate dark matter inputs
    positions_dark = np.asarray(positions_dark)
    velocity_dark = np.asarray(velocity_dark)
    assert positions_dark.shape[1] == 3, "positions_dark must have shape (N, 3)"
    assert velocity_dark.shape[1] == 3, "velocity_dark must have shape (N, 3)"
    assert positions_dark.shape[0] == velocity_dark.shape[0], \
        "positions_dark and velocity_dark must have same length"
    
    n_dark = positions_dark.shape[0]
    
    # Handle mass_dark (scalar or array)
    if np.isscalar(mass_dark):
        mass_dark = np.full(n_dark, mass_dark)
    else:
        mass_dark = np.asarray(mass_dark)
        assert len(mass_dark) == n_dark, \
            f"mass_dark array length ({len(mass_dark)}) must match positions ({n_dark})"
    
    # Validate star inputs if provided
    has_stars = positions_star is not None
    if has_stars:
        positions_star = np.asarray(positions_star)
        assert velocity_star is not None, "velocity_star required when positions_star provided"
        assert mass_star is not None, "mass_star required when positions_star provided"
        
        velocity_star = np.asarray(velocity_star)
        assert positions_star.shape[1] == 3, "positions_star must have shape (N, 3)"
        assert velocity_star.shape[1] == 3, "velocity_star must have shape (N, 3)"
        assert positions_star.shape[0] == velocity_star.shape[0], \
            "positions_star and velocity_star must have same length"
            
        n_star = positions_star.shape[0]
        
        # Handle mass_star (scalar or array)
        if np.isscalar(mass_star):
            mass_star = np.full(n_star, mass_star)
        else:
            mass_star = np.asarray(mass_star)
            assert len(mass_star) == n_star, \
                f"mass_star array length ({len(mass_star)}) must match positions ({n_star})"
    
    # Validate center_on makes sense given available data
    if center_on == 'star' and not has_stars:
        raise ValueError("center_on='star' requires star data to be provided")
    # Validate numerical parameters
    assert recursive_iter_converg > 0, "recursive_iter_converg must be positive"
    assert BFE_lmax > 0, "BFE_lmax must be positive"
    assert softening_tree > 0, "softening_tree must be positive"
    assert G > 0, "G must be positive"
    assert vel_rmax > 0, "vel_rmax must be positive"
    assert 0 < tol_frac_change < 1, "tol_frac_change must be between 0 and 1"
    
    # Validate center arrays if provided
    if len(center_position) > 0:
        center_position = np.asarray(center_position)
        assert len(center_position) == 3, "center_position must be length 3"
    if len(center_velocity) > 0:
        center_velocity = np.asarray(center_velocity)
        assert len(center_velocity) == 3, "center_velocity must be length 3"
    
    if potential_compute_method == 'tree':
        try:
            import pyfalcon
        except ImportError:
            logger.info(f'falcON not available for tree method.')
            potential_compute_method = 'bfe'
    
    # Setup logger
    logger = logging.getLogger('boundness')
    if verbose and not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # Stack inputs for center determination
    if positions_star is not None and center_on == 'both':
        pos_for_center = np.vstack((positions_dark, positions_star))
        mass_for_center = np.concatenate((mass_dark, mass_star))
        vel_for_center = np.vstack((velocity_dark, velocity_star))
        
    elif positions_star is not None and center_on == 'star':
        pos_for_center = positions_star
        mass_for_center = mass_star
        vel_for_center = velocity_star
    else:
        pos_for_center = positions_dark
        mass_for_center = mass_dark
        vel_for_center = velocity_dark

    # Main function
    # ==================================

    # Determine center position
    if len(center_position) < 1:
        if center_vel_with_KDE:
            assert center_method != 'shrinking_sphere', "Centering method should be density_peak/KDE. or set center_vel_with_KDE = False"
            computed_central_loc = find_center_position(
                np.hstack((pos_for_center, vel_for_center)), mass_for_center,
                method=center_method, **(center_params or {}))
                
            center_position, center_velocity = computed_central_loc[:3], computed_central_loc[3:]
        else:
            center_position = find_center_position(
                pos_for_center, mass_for_center,
                method=center_method, **(center_params or {}))
        
    # Determine center velocity
    if len(center_velocity) < 1:
        ## base it on 10% dispersion... 
        vel_rmax = min(vel_rmax, 0.1 * np.std(np.linalg.norm(vel_for_center, axis=1)))
        dist2 = np.sum((pos_for_center - center_position)**2, axis=1)
        sel = dist2 < vel_rmax**2
        if np.any(sel):
            center_velocity = np.average(
                vel_for_center[sel], axis=0, weights=mass_for_center[sel])
        else:
            center_velocity = np.average(
                vel_for_center, axis=0, weights=mass_for_center)
    
    logger.info(f"Center position (method={center_method}): {np.around(center_position, decimals=2)}")
    logger.info(f"Center velocity: {np.around(center_velocity, decimals=2)}")

    # Prepare arrays for recursion
    if positions_star is not None:
        pos_all = np.vstack((positions_dark, positions_star))
        vel_all = np.vstack((velocity_dark, velocity_star))
        mass_all = np.concatenate((mass_dark, mass_star))
        n_dark = len(positions_dark)
    else:
        pos_all, vel_all, mass_all = positions_dark.copy(), velocity_dark.copy(), mass_dark.copy()
        n_dark = len(pos_all)

    # Recenter positions and velocities
    pos_rel = pos_all - center_position
    vel_rel = vel_all - center_velocity

    bound_history_dm = []
    bound_history_star = []
    mask_all = np.ones(len(pos_all), dtype=bool)

    # Define a minimum number of particles required to build a reliable potential
    min_particles_for_model = 5  # Example value, adjust as needed
    
    # Recursive energy cut
    for i in range(recursive_iter_converg):
         # --- New Check 1: Ensure enough particles are bound before iteration ---
        num_bound = np.sum(mask_all)
        if num_bound < min_particles_for_model:
            logger.info(f"Stopping: Only {num_bound} particles remaining, which is below the threshold of {min_particles_for_model}.")
            break
            
        if potential_compute_method == 'tree':
            # --- make unbound particles negligible mass so they don't source gravity.
            mass_all[~mask_all] = 0.01 # can not be zero, pos still required to find phi at those locations.
            
            _, phi = pyfalcon.gravity(
                pos_rel, mass_all*G, 
                eps=softening_tree, theta=0.4) # theta=0.4 has very high accuracy compared to regular codes.
        
        else:
            
            pot = agama.Potential(
                type='Multipole', particles=(pos_rel[mask_all], mass_all[mask_all]),
                symmetry='n', lmax=BFE_lmax)
            
            phi = pot.potential(pos_rel)
            
        
        kin = 0.5 * np.sum(vel_rel**2, axis=1)
        bound_mask = (phi + kin) < 0

        bound_history_dm.append(bound_mask[:n_dark].copy())
        if positions_star is not None:
            bound_history_star.append(bound_mask[n_dark:].copy())

        frac_change = np.mean(bound_mask != mask_all)
        logger.info(f"Iter {i}: Δ bound mask = {frac_change:.5f}")
        mask_all = bound_mask
        if frac_change < tol_frac_change:
            logger.info(f"Converged after {i+1} iterations.")
            break

    # Final masks
    bound_dark_final = mask_all[:n_dark].astype(int)
    results = [bound_dark_final]
    if positions_star is not None:
        results.append(mask_all[n_dark:].astype(int))
    if return_history:
        results.append(bound_history_dm)
        if positions_star is not None:
            results.append(bound_history_star)

    return tuple(results), center_position, center_velocity 

def find_center_position(
    positions: np.ndarray, masses: np.ndarray, 
    method: str = 'shrinking_sphere', **kwargs
) -> np.ndarray:
    """
    Find the center of a particle distribution using different algorithms.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Cartesian positions of particles.
    masses : np.ndarray, shape (N,)
        Masses of the particles.
    method : {'shrinking_sphere', 'density_peak'}, optional
        Center-finding algorithm to use. Options are:
        
        - 'shrinking_sphere': Iteratively shrink a sphere around the
          current center until convergence.
        - 'density_peak': Estimate density with a KDE + multipole expansion
          and locate the maximum-density region.
        
        Default is 'shrinking_sphere'.
    **kwargs : dict, optional
        Extra parameters passed to the chosen method:
        
        For ``shrinking_sphere``:
            - r_init : float, optional
                Initial search radius (default 30.0).
            - shrink_factor : float, optional
                Fraction by which to shrink the radius each iteration (default 0.9).
            - min_particles : int, optional
                Minimum number of particles required to continue shrinking (default 100).

        For ``density_peak``:
            - lmax : int, optional
                Maximum multipole order for density expansion in Agama (default 8).

    Returns
    -------
    np.ndarray, shape (3,)
        Estimated center position of the distribution.

    Notes
    -----
    - The 'density_peak' method is more computationally expensive but can be
      more accurate in systems with strong asymmetries.
    - The 'shrinking_sphere' method is robust for approximately spherical
      systems.
    """
    if method == 'shrinking_sphere':
        return _shrinking_sphere_center(positions, masses,
                                        r_init=kwargs.get('r_init', 30.0),
                                        shrink_factor=kwargs.get('shrink_factor', 0.9),
                                        min_particles=kwargs.get('min_particles', 100))

    # # does the KDE by default on either pos or posvel array. The pos will also be used to compute density_peak. 
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(positions.T, weights=masses)
    sample = positions[np.random.choice(len(positions), size=min(10_000, len(positions)), replace=False)]
    dens = kde(sample.T)
    Npick = max(10, int(len(dens)*0.01))
    idxs = np.argsort(dens)[-Npick:]
    centroid = np.average(sample[idxs], axis=0)
    
    if method == 'density_peak':
        print(f'Using the KDE center: {centroid}')
        pos_c = positions - centroid

        dens = agama.Potential(type='Multipole', 
                               particles=(pos_c, masses), 
                               symmetry='n', 
                               lmax=kwargs.get('lmax', 8)).density(pos_c[:, :3])
        
        ## pick max of 1% particles or 50.
        Npick = max(50, int(len(dens)*0.01))
        idxs = np.argsort(dens)[-Npick:]

        ## add the max dens loc to the original centroid
        centroid += np.average(pos_c[idxs], axis=0, weights=dens[idxs])
        print(f'Density peak found at: {centroid}')
    
    else:
        logger.info(f"Returning the KDE center.")
    
    return centroid

def _shrinking_sphere_center(
    positions: np.ndarray, masses: np.ndarray, 
    r_init: float = 30.0, shrink_factor: float = 0.9, 
    min_particles: int = 10
) -> np.ndarray:
    """
    Compute the center of a particle distribution via the shrinking-sphere algorithm.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Cartesian positions of particles.
    masses : np.ndarray, shape (N,)
        Masses of the particles.
    r_init : float, optional
        Initial radius of the enclosing sphere (default 30.0).
    shrink_factor : float, optional
        Factor by which the radius is reduced at each iteration (default 0.9).
    min_particles : int, optional
        Minimum number of particles required to continue shrinking (default 10).

    Returns
    -------
    np.ndarray, shape (3,)
        Estimated center position of the particle distribution.

    Notes
    -----
    The algorithm starts with the mass-weighted centroid and iteratively shrinks
    the enclosing sphere, recomputing the center at each step. Iteration stops
    when the number of particles inside the current sphere drops below
    ``min_particles``.
    """
    center = np.average(positions, axis=0, weights=masses)
    radius = r_init
    while True:
        dist2 = np.sum((positions - center)**2, axis=1)
        mask = dist2 < radius**2
        if np.sum(mask) < min_particles:
            break
        center = np.average(positions[mask], axis=0, weights=masses[mask])
        radius *= shrink_factor
    return center

## Fitting the multipole and cylspline potential files using Agama
def fitPotential(
    part,
    nsnap: int,
    sym: list[str] = ['n'],
    pole_l: list[int] = [4],
    rmax_sel: float = 600,
    rmax_ctr: float = 10,
    rmax_exp: float = 500,
    sim_dir: str = '',
    file_ext: str = 'DR',
    save_dir: str | None = None,
    subsample_factor: int = 1,
    save_coords: bool = True,
    host_num: int = 0,
    halo: str | None = None,
    spec_ind: dict | None = None,
    coord_props: dict | None = None,
    kind: str = 'whole',
):
    """
    Construct a combined Agama potential model from a simulation snapshot.

    Parameters
    ----------
    part : dict
        Dictionary of particle data from a GIZMO snapshot.
    nsnap : int
        Snapshot index.
    sym : list of {'n', 'a', 's', 't'}, optional
        Symmetry of the model:
        - 'n' : none (default)
        - 'a' : axisymmetric
        - 's' : spherical
        - 't' : triaxial
    pole_l : list of int, optional
        List of multipole expansion orders to include (default [4]).
    rmax_sel : float, optional
        Maximum radius for selecting particles (default 600).
    rmax_ctr : float, optional
        Maximum radius for particles used in center-finding (default 10).
    rmax_exp : float, optional
        Maximum radius for expansion fitting (default 500).
    sim_dir : str, optional
        Path to the simulation directory (default empty string).
    file_ext : str, optional
        File extension for saved potential files (default 'DR').
    save_dir : str or None, optional
        Directory to save outputs. If None, defaults to ``sim_dir`` (default None).
    subsample_factor : int, optional
        Factor by which to subsample particles (default 1 = no subsampling).
    save_coords : bool, optional
        Whether to save the coordinates used in the model construction (default True).
    host_num : int, optional
        Index of the host halo to use (default 0).
    halo : str or None, optional
        Name of the halo to model. If None, treats the entire simulation as the host
        (default None).
    spec_ind : dict or None, optional
        Dictionary mapping species → halo indices. If None, use all halos (default None).
    coord_props : dict or None, optional
        Coordinate properties to use. Expected keys: ``ctr`` (center), ``vctr`` (velocity center),
        ``rot`` (rotation matrix). If None, uses the principal axes of the host (default None).
    kind : {'whole', 'dark', 'bar'}, optional
        Type of potential to construct:
        - 'whole' : spherical + bar (default)
        - 'dark'  : dark matter only
        - 'bar'   : baryonic matter only

    Returns
    -------
    None
        This function saves potential models to disk, rather than returning them.

    Notes
    -----
    - Requires the Agama library for multipole expansions.
    - The function may write several files (potential models, coordinates) depending
      on ``save_coords`` and ``save_dir``.
    """

    # # list of custom package dependencies
    # import agama
    import gizmo_analysis as ga
    import utilities as ut

    ## sets the default units to: Msol, kpc, km/s, Remember Time is Length/vel.
    agama.setUnits(mass=1, length=1, velocity=1)

    symmlabel = {'a':'axi','s':'sph','t':'triax','n':'none'}

    mass_p = 100 #mass percent enclosed, set to 100 for all.
    
    #default centering and rotation to define aperture
    
    if halo:
        if spec_ind:
            print(f'Computing model for {halo}')
        else:
            print("Can't create a model without spec index! exitting.")
            exit()
    else:
        print('Fitting the model on entire simulation')
        spec_ind = dict()
        for spec in part.keys(): 
            spec_ind[spec] = range(part[spec]['mass'].shape[0])
    
    if coord_props:
        print('Using precomputed centering and rotation to define aperture from coord_props dict.')
    
    if coord_props is None:
        print('Using host centering and rotation')
        coord_props = dict()
        coord_props['ctr'] = part.host['position'][host_num]
        coord_props['vctr'] = part.host['velocity'][host_num]
        coord_props['rot'] = part.host['rotation'][host_num]
        
    new_ctr = coord_props['ctr']
    new_vctr = coord_props['vctr']
    new_rot = coord_props['rot']
    
    if spec_ind is None:
        spec_ind = dict()        
        for key in part.keys():
            spec_ind[key] = np.arange(0, part[key]['mass'].shape[0])
    
    dist=ut.particle.get_distances_wrt_center(part,
                                              species=['star','gas','dark'],
                                              part_indicess= [spec_ind['star'],
                                                              spec_ind['gas'],
                                                              spec_ind['dark']],
                                              center_position=new_ctr,
                                              rotation=new_rot,
                                              total_distance = True)
    
    dist_vectors = ut.particle.get_distances_wrt_center(part,
                                                        species=['star','gas','dark'],
                                                        part_indicess= [spec_ind['star'],
                                                                        spec_ind['gas'],
                                                                        spec_ind['dark']],
                                                        center_position=new_ctr,
                                                        rotation=new_rot)

    ##pick out gas and stars within the region that we want to supply to the model
    
    # select Gas
    m_gas_tot = part['gas']['mass'][spec_ind['gas']].sum() * subsample_factor
    pos_pa_gas = dist_vectors['gas'][dist['gas'] < rmax_sel]
    
    m_gas = part['gas']['mass'][spec_ind['gas']][dist['gas'] < rmax_sel] * subsample_factor
    print(f'{m_gas.sum():.2e} of {m_gas_tot:.2e} solar masses in gas selected')

    #select Star
    m_star_tot = part['star']['mass'][spec_ind['star']].sum() * subsample_factor

    pos_pa_star = dist_vectors['star'][dist['star'] < rmax_sel]
    m_star = part['star']['mass'][spec_ind['star']][dist['star'] < rmax_sel] * subsample_factor
    
    print(f'{m_star.sum():.2e} of {m_star_tot:.2e} solar masses in stars selected')

    ##separate cold gas in disk (modeled with cylspline) from hot gas in halo (modeled with multipole)

    tsel = (np.log10(part['gas']['temperature'][spec_ind['gas']]) < 4.5)
    rsel = (dist['gas'] < rmax_sel)

    pos_pa_gas_cold = dist_vectors['gas'][tsel & rsel]
    m_gas_cold = part['gas']['mass'][spec_ind['gas']][tsel & rsel] * subsample_factor
    print(f'{m_gas_cold.sum():.2e} of {m_gas.sum():.2e} solar masses are cold gas to be modeled with cylspline')

    pos_pa_gas_hot = dist_vectors['gas'][(~tsel) & rsel]
    m_gas_hot = part['gas']['mass'][spec_ind['gas']][(~tsel) & rsel] * subsample_factor
    print(f'{m_gas_hot.sum():.2e} of {m_gas.sum():.2e} solar masses are hot gas to be modeled with multipole')

    #combine components that will be fed to the cylspline part
    pos_pa_bar = np.vstack((pos_pa_star, pos_pa_gas_cold))
    m_bar = np.hstack((m_star, m_gas_cold))

    #pick out the dark matter
    m_dark_tot = part['dark']['mass'][spec_ind['dark']].sum() * subsample_factor

    rsel = dist['dark'] < rmax_sel
    pos_pa_dark = dist_vectors['dark'][rsel]
    m_dark = part['dark']['mass'][spec_ind['dark']][rsel] * subsample_factor
    print(f'{m_dark.sum():.2e} of {m_dark_tot:.2e} solar masses in dark matter selected')

    #stack with hot gas for multipole density
    pos_pa_dark = np.vstack((pos_pa_dark, pos_pa_gas_hot))
    m_dark = np.hstack((m_dark, m_gas_hot))

    if save_coords:
        #save the Hubble parameter to transform to comoving units
        hubble = part.info['hubble']
        scalefactor = part.info['scalefactor']
    
    # delete part because it's useless
    del part ## one can implenent garbage collector using gc if there are memmory issues.
    
    #right now, configured to save to a new directory in the simulation directory.
    #Recommended, since it's generally useful to have around
    output_stem = sim_dir + f'potential/{rmax_ctr:.0f}kpc/{nsnap:d}'
    
    if save_dir:
        output_stem = save_dir + f'potential/{rmax_ctr:.0f}kpc/{nsnap:d}'
        
    if host_num > 0:
        output_stem += '.host2'

    try:    
        # create the directory if it didn't exist
        print(f'creating directory {output_stem}')
        os.makedirs(os.path.dirname(output_stem))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    if save_coords:
        
        if halo:
            cname = f'{output_stem}_coords_{halo}_{file_ext}.txt'
        else:
            cname = f'{output_stem}_coords_{file_ext}.txt'
            
        print(f'Saving coordinate transformation to {cname}')
        
        with open(cname,'w') as f:
            f.write('# Hubble parameter and scale factor (to convert physical <-> comoving) \n')
            f.write(f'{hubble:.18g} {scalefactor:.18g}\n')
            f.write('# center position (kpc comoving)\n')
            np.savetxt(f, new_ctr)
            f.write('# center velocity (km/s physical)\n')
            np.savetxt(f, new_vctr)
            f.write('# rotation to principal-axis frame\n')
            try:
                np.savetxt(f, new_rot)
            except:
                np.savetxt(f, new_rot['rotation'])
    
    if not isinstance(pole_l, list):
        pole_l = [pole_l]
    if not isinstance(sym, list):
        sym = [sym]
        
    for symmetry in sym:
        for mult_l in pole_l: 
            # symmetry=sym
            print(f'Symmetry = {symmetry}, Multipole order = {mult_l} \n')
            
            print('Computing multipole expansion coefficients for dark matter/hot gas component')            
            p_dark=agama.Potential(type='Multipole',
                                   particles=(pos_pa_dark, m_dark),
                                   lmax=mult_l, symmetry=symmetry,
                                   # gridSizeR = 40,  ##change accordingly to your grid req.
                                   rmin=0.1, rmax=rmax_exp)
            
            if halo:
                p_dark.export(f'{output_stem}.dark.{symmlabel[symmetry]}_{mult_l}.{halo}.coef_mul_{file_ext}')
                print(f'halo BFE saved@ {output_stem}.dark.{symmlabel[symmetry]}_{mult_l}.{halo}.coef_mul_{file_ext}')
                
            else:
                p_dark.export(f'{output_stem}.dark.{symmlabel[symmetry]}_{mult_l}.coef_mul_{file_ext}')
                print(f'Sph harmonic BFE saved@ {output_stem}.dark.{symmlabel[symmetry]}_{mult_l}.coef_mul_{file_ext}')
                
                
            print('Computing azimuthal harmonic expansion coefficients for stellar/cold gas component')
            p_bar = agama.Potential(type='CylSpline',
                                    particles=(pos_pa_bar, m_bar),
                                    mmax=mult_l, symmetry=symmetry,
                                    #gridSizeR=40, gridSizeZ=40, ##change accordingly to your grid req.
                                    rmin=0.1, rmax=rmax_exp)
            if halo:
                p_bar.export(f'{output_stem}.bar.{symmlabel[symmetry]}_{mult_l}.{halo}.coef_cylsp_{file_ext}')
                print(f'CylSpl BFE saved@ {output_stem}.bar.{symmlabel[symmetry]}_{mult_l}.{halo}.coef_cylsp_{file_ext}')

            else:
                p_bar.export(f'{output_stem}.bar.{symmlabel[symmetry]}_{mult_l}.coef_cylsp_{file_ext}')
                print(f'CylSpl BFE saved@ {output_stem}.bar.{symmlabel[symmetry]}_{mult_l}.coef_cylsp_{file_ext}')

            print('done, enjoy your potentials!')


def make_uneven_grid(xmin: float, xmax: float | None = None, nbins: int = 10) -> np.ndarray:
    """
    Create a 1D grid with unequally spaced nodes, starting at 0,
    with second node at xmin and last node at xmax.
    
    Parameters
    ----------
    - xmin: float, location of the 1st nonzero node (> 0)
    - xmax: float, location of the last node (> xmin); if None, make uniform grid
    - nbins: int, total number of bins (>= 3)
    
    Returns
    -------
    - x: ndarray of shape (nbins,), containing the grid nodes
    """
    from scipy.optimize import root_scalar
    if nbins < 3:
        raise ValueError("bins must be at least 3.")
    if xmin <= 0:
        raise ValueError("xmin must be positive.")
    
    if xmax is None:
        # Uniform linear grid
        return np.linspace(0, xmin * (nbins - 1), nbins)

    if xmax <= xmin:
        raise ValueError("xmax must be greater than xmin.")

    N = nbins - 1  # number of intervals
    
    # **Check feasibility**:
    if xmax <= N * xmin:
        # no exponential grading possible → uniform
        return np.linspace(0, xmax, nbins)

    # Define equation to solve for Z:
    # f(Z) = (exp(Z * 1) - 1) / (exp(Z * N) - 1) - xmin/xmax = 0
    def f(Z):
        ez = np.exp(-Z)
        ezn = np.exp(-Z * N)
        return np.exp(Z * (1 - N)) * (1 - ez) / (1 - ezn) - (xmin / xmax)

    # Solve for Z
    sol = root_scalar(f, bracket=[1e-8, 100], method='brentq')
    if not sol.converged:
        raise RuntimeError("Failed to find solution for Z.")
    Z = sol.root

    # Now compute the grid
    k = np.arange(nbins)
    x = (np.exp(Z * k) - 1) / (np.exp(Z * N) - 1) * xmax
    return x

def empirical_density_profile(pos: np.ndarray, mass: np.ndarray | float, 
                              nbins: int = 50, 
                              rmin: float =0.1, 
                              rmax: float =600) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the number density radial profile. Assuming all the particles have the same mass.
    
    Parameters
    ----------
    pos (array-like): Array of particle positions or distances with shape (N, 3), or (N, ) where N is the number of particles.
    mass (scalar/array-like): scalar or Array of particle masses with shape (N, ), corresponding to the positions.
    nbins (int): Number of radial bins for the profile.
    rmin (float): Minimum radial grid node after 0.
    rmax (float): Maximum radial grid node.
        
    Returns
    -------
    radius (array-like): Array of radius values for the profile.
    density (array-like): Array of number density values as a function of radius.
    """
    pos = np.asarray(pos)
    
    # Compute radial distances
    if pos.ndim == 1:
        r_p = pos
    elif pos.ndim == 2 and pos.shape[1] == 3:
        r_p = np.linalg.norm(pos, axis=1)
    else:
        raise ValueError("pos must be shape (N,) or (N,3)")
    
    # Broadcast scalar mass if needed
    mass = np.asarray(mass)
    if mass.ndim == 0:
        mass = np.full(r_p.shape, mass, dtype=float)
    elif mass.shape[0] != r_p.shape[0]:
        raise ValueError("mass must be scalar or same length as pos")
    
    if not isinstance(nbins, int) or nbins <= 0:
        raise ValueError("nbins must be a positive integer")
    
    # Define the binning scheme
    bins = make_uneven_grid(rmin, rmax, nbins=nbins + 1)
    
    # Calculate the volumes of the spherical shells within each bin
    V_shells = 4/3 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    
    # Histogram the particle positions into the radial bins and compute the sum of masses within each bin
    density, _ = np.histogram(r_p, bins=bins, weights=mass)
    
    # Normalize the density by the volumes of the shells to obtain the number density
    density /= V_shells
    
    # Calculate the average radius within each bin
    radius = 0.5 * (bins[1:] + bins[:-1])
    
    return radius, density

def empirical_vel_circ_profile(pos: np.ndarray, mass: np.ndarray | float, 
                               nbins: int = 50, rmin: float = 0.1, rmax:float = 600, 
                               G: float = 4.300917270035e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the circular velocity profile v_circ(r) = sqrt(G * M_enclosed(<r) / r).
    Assuming particles have masses given in `mass` and positions `pos`.
    
    Parameters
    ----------
    pos (array-like): Array of particle positions or distances with shape (N, 3), or (N, ) where N is the number of particles.
    mass (scalar/array-like): scalar or Array of particle masses with shape (N, ), corresponding to the positions.
    nbins (int): Number of radial bins for the profile.
    rmin (float): Minimum radial grid node after 0.
    rmax (float): Maximum radial grid node.
    G (float): Gravitational constant in units such that v_circ is returned in desired units.
               Default is 4.300917270035e-6 (kpc * (km/s)^2 / Msun) — i.e. if `pos` in kpc and `mass` in Msun,
               v_circ will be in km/s.
        
    Returns
    -------
    radius (array-like): Array of radius values for the profile (bin centers).
    v_circ (array-like): Circular velocity values as a function of radius.
    """
    pos = np.asarray(pos)
    
    # Compute radial distances
    if pos.ndim == 1:
        r_p = pos
    elif pos.ndim == 2 and pos.shape[1] == 3:
        r_p = np.linalg.norm(pos, axis=1)
    else:
        raise ValueError("pos must be shape (N,) or (N,3)")
    
    # Broadcast scalar mass if needed
    mass = np.asarray(mass)
    if mass.ndim == 0:
        mass = np.full(r_p.shape, mass, dtype=float)
    elif mass.shape[0] != r_p.shape[0]:
        raise ValueError("mass must be scalar or same length as pos")
    
    if not isinstance(nbins, int) or nbins <= 0:
        raise ValueError("nbins must be a positive integer")
    
    # Define the binning scheme (nbins+1 edges to form nbins shells)
    bins = make_uneven_grid(rmin, rmax, nbins=nbins + 1)
    
    # Histogram the particle positions into the radial bins and compute the sum of masses within each bin
    mass_in_bins, _ = np.histogram(r_p, bins=bins, weights=mass)
    
    # Compute the cumulative (enclosed) mass at each shell (use cumulative sum over shells)
    M_enclosed_shells = np.cumsum(mass_in_bins)
    
    # Calculate the average radius within each bin (bin centers)
    radius = 0.5 * (bins[1:] + bins[:-1])
    
    # Avoid division by zero: compute v_circ only where radius > 0
    # v_circ = sqrt(G * M_enclosed / r)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_circ = np.sqrt(G * M_enclosed_shells / radius)
        # where radius is zero (shouldn't happen given rmin > 0), set v_circ to 0
        v_circ = np.where(radius > 0, v_circ, 0.0)
    
    return radius, v_circ

def empirical_vel_dispersion_profile(pos: np.ndarray, vel: np.ndarray, 
                                     nbins: int = 50, rmin: float = 0.1, rmax:float = 600) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the velocity dispersion profile as a function of radius.
    
    Parameters
    ----------
    pos (array-like): Array of particle positions with shape (N, 3), where N is the number of particles.
    vel (array-like): Array of particle velocities with shape (N, 3), corresponding to the positions.
    nbins (int): Number of radial bins for the profile.
    rmin (float): Minimum radial grid node after 0.
    rmax (float): Maximum radial grid node.

    Returns
    -------
    radius (array-like): Array of radius values for the profile.
    vel_disp (array-like): Array of velocity dispersion values as a function of radius.
    """
    from scipy.stats import binned_statistic
    
    if len(pos) != len(vel):
        raise ValueError("pos and vel arrays must have the same length")
    if not isinstance(nbins, int) or nbins <= 0:
        raise ValueError("nbins must be a positive integer")

    if len(pos.shape) == 1:
        r_p = pos
    elif len(pos.shape) == 2 and pos.shape[1] == 3:
        # Calculate the radial distances from the origin
        r_p = np.linalg.norm(pos, axis=1)
    else:
        raise ValueError("pos must be a (N, 3) array or a (N, ) array")

    if len(vel.shape) == 1:
        v_magnitude = vel
    elif len(vel.shape) == 2 and vel.shape[1] == 3:
        # Calculate the velocity magnitudes
        v_magnitude = np.linalg.norm(vel, axis=1)
    else:
        raise ValueError("vel must be a (N, 3) array or a (N, ) array")
            
    # Define the binning scheme
    bins = make_uneven_grid(rmin, rmax, nbins= nbins + 1)
    
    # Use binned_statistic to compute the standard deviation of the velocities in each bin
    vel_disp, _, _ = binned_statistic(r_p, v_magnitude, statistic='std', bins=bins)

    # Calculate the average radius within each bin
    radius = 0.5 * (bins[1:] + bins[:-1])
    
    return radius, vel_disp

def empirical_vel_rms_profile(pos: np.ndarray, vel: np.ndarray, 
                              nbins: int = 50, rmin: float = 0.1, rmax:float = 600) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the root mean square velocity radial profile as a function of radius.
    
    Parameters
    ----------
    pos (array-like): Array of particle positions with shape (N, 3), where N is the number of particles.
    vel (array-like): Array of particle velocities with shape (N, 3), corresponding to the positions.
    nbins (int): Number of radial bins for the profile.
    rmin (float): Minimum radial grid node after 0.
    rmax (float): Maximum radial grid node.

    Returns
    -------
    radius (array-like): Array of radius values for the profile.
    vel_rms (array-like): Array of root mean square velocity values as a function of radius.
    """
    if len(pos) != len(vel):
        raise ValueError("pos and vel arrays must have the same length")
    if not isinstance(nbins, int) or nbins <= 0:
        raise ValueError("nbins must be a positive integer")

    # Calculate the radial distances from the origin
    r = np.linalg.norm(pos, axis=1)
    
    # Calculate the velocity magnitudes
    v_rms = np.linalg.norm(vel, axis=1)

    # Define the binning scheme
    bins = make_uneven_grid(rmin, rmax, nbins= nbins + 1)

    # Histogram the squared velocity magnitudes into the radial bins
    binned_sum_v_rms, _ = np.histogram(r, bins=bins, weights=v_rms**2)
    binned_counts, _ = np.histogram(r, bins=bins)

    # Calculate the mean squared velocity within each bin
    mean_squared_v_rms = binned_sum_v_rms / binned_counts

    # Calculate the root mean square velocity as a function of radius
    vel_rms = np.sqrt(mean_squared_v_rms)

    # Calculate the average radius within each bin
    radius = 0.5 * (bins[1:] + bins[:-1])

    return radius, vel_rms

def measure_anisotropy(pos: np.ndarray, vel: np.ndarray,
                       nbins: int = 50,
                       rmin: float = 0.1, smooth=True):
    """
    Compute velocity anisotropy β(r) in radial bins.

    Parameters
    ----------
    pos : ndarray (N,3)
    vel : ndarray (N,3)
    nbins : int
        Number of radial bins (ignored if bin_mode='linear' and dr is given)
    rmin : float
        Minimum spacing in r.
    smooth : bool 
        Whether to do Univariate spline smoothing.

    Returns
    -------
    r_centers : ndarray
    beta_vals : ndarray
    """
    from scipy.interpolate import UnivariateSpline
    r = np.linalg.norm(pos, axis=1)
    vr = np.sum(pos * vel, axis=1) / r
    vt2 = np.sum(vel**2, axis=1) - vr**2

    rmax = np.percentile(r, 90)
    edges = make_uneven_grid(rmin, rmax, nbins= nbins + 1)
    print(f'No. in lower bin: {(r<rmin).sum()}')
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    vr2, _ = np.histogram(r, edges, weights=vr**2)
    vt2b, _ = np.histogram(r, edges, weights=vt2)
    N, _ = np.histogram(r, edges)
    N[N == 0] = 1
    sigma_r2 = vr2 / N
    sigma_t2 = vt2b / N
    beta_vals = 1 - sigma_t2 / (2 * sigma_r2)
    return r_centers, beta_vals
    # return r_centers, UnivariateSpline(r_centers, beta_vals, s=2, k=3)#(r_centers) if smooth else beta_vals 


def return_spherical_spiral_grid(rad: float = 1, proj: str = 'Cart') -> np.ndarray:
    """
    Generate a spherical grid.
    This function loads a pre-defined spherical grid and returns it based on the specified projection.
    
    Parameters
    ----------
    - rad: Radius of the spherical grid. Default is 1.
    - proj: Projection of the grid. Options are 'Cart' for Cartesian coordinates,
            'Sph' for spherical coordinates, and 'Cyl' for cylindrical coordinates.
            Default is 'Cart'.

    Returns
    -------
    - grid: Spherical grid points based on the specified projection.
    """

    # Validate the projection type
    assert proj in ['Cart', 'Sph', 'Cyl'], "Invalid projection type. Must be 'Cart', 'Sph', or 'Cyl'."
    assert rad > 0, "Radius must be greater than zero."
    
    # Load spherical grid from file
    XYZ_grid = rad * np.loadtxt('data/spherical_grid_unit.xyz')

    if proj == 'Cart':
        return XYZ_grid
    elif proj == 'Sph':
        # Convert Cartesian to spherical coordinates
        return Cart2Sph(XYZ_grid).values
    elif proj == 'Cyl':
        # Convert Cartesian to cylindrical coordinates
        return Cart2Cyl(XYZ_grid).values
        
def return_uni_sph_grid(rad: float = 1, num_pts: int =500) -> np.ndarray:
    """
    Generates a uniform spherical grid with the specified radius and number of points.
    The grid points are randomly distributed on the surface of the sphere.

    Parameters
    ----------
    - rad: Radius of the spherical grid. Default is 1.
    - num_pts: Number of points in the spherical grid. Default is 500.

    Returns
    -------
    - xyz_all: Spherical grid points in Cartesian coordinates.


    Example
    -------
    xyz_grid = return_uni_sph_grid(rad=2, num_pts=1000)
    """
    # Validate input parameters
    assert isinstance(rad, (int, float)), "Radius must be a numeric value."
    assert rad > 0, "Radius must be greater than zero."
    assert isinstance(num_pts, int) and num_pts > 0, "Number of points must be a positive integer."

    # Generate random spherical coordinates
    phi = np.random.uniform(0, 2*np.pi, num_pts)
    costheta = np.random.uniform(-1, 1, num_pts)
    theta = np.arccos(costheta)

    # Convert spherical coordinates to Cartesian coordinates
    x = rad * np.sin(theta) * np.cos(phi)
    y = rad * np.sin(theta) * np.sin(phi)
    z = rad * np.cos(theta)

    # Combine x, y, z coordinates
    xyz_all = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    return xyz_all
    
# Define the double power law density profile function
def double_power_law_density(
    mass: float,
    scaleradius: float,
    alpha: float,
    beta: float,
    gamma: float,
    rcut: float | None = None,
    cutoffstrength: float = 2.0,
) -> Callable[[float | np.ndarray], np.ndarray]:
    """
    Construct the double-power-law (Zhao 1996) density profile, normalized to total mass.

    The profile is defined as:

    .. math::

        \\rho(r) = \\rho_0 \\, r^{-\\gamma} 
        \\left[ 1 + \\left(\\frac{r}{a}\\right)^\\alpha \\right]^{(\\gamma - \\beta)/\\alpha}
        \\times f_{\\mathrm{cut}}(r),

    where

    - ``a`` is the scale radius,
    - ``α`` controls the transition steepness between inner and outer slopes,
    - ``β`` is the outer slope,
    - ``γ`` is the inner slope,
    - ``f_cut(r)`` is an optional exponential cutoff:

    .. math::

        f_{\\mathrm{cut}}(r) =
        \\begin{cases}
            \\exp\\!\\left[-\\left(\\tfrac{r}{r_{\\mathrm{cut}}}\\right)^{\\mathrm{cutoffstrength}}\\right], & r_{\\mathrm{cut}} \\neq None \\\\
            1, & r_{\\mathrm{cut}} = None
        \\end{cases}

    The normalization constant ``ρ₀`` is chosen so that the total mass integrates to ``mass``.

    Parameters
    ----------
    mass : float
        Total mass of the profile.
    scaleradius : float
        Scale radius ``a`` of the profile.
    alpha : float
        Transition sharpness between inner and outer slopes.
    beta : float
        Outer slope of the density profile.
    gamma : float
        Inner slope of the density profile.
    rcut : float or None, optional
        Cutoff radius. If None, no cutoff is applied (default None).
    cutoffstrength : float, optional
        Exponent controlling how steeply the cutoff is applied (default 2.0).

    Returns
    -------
    rho_fn : callable
        Function ``rho(r)`` that evaluates the density at scalar or array radii.

    Examples
    --------
    >>> rho_fn = double_power_law_density(1e12, 20.0, 1.0, 3.0, 1.0)
    >>> rho_fn(10.0)
    12345.678   # density at r = 10 (arbitrary units)
    >>> import numpy as np
    >>> r = np.logspace(-2, 2, 50)
    >>> rho = rho_fn(r)
    """
    from scipy.integrate import quad

    a = float(scaleradius)
    if (beta <= 3.0) and (rcut is None):
        raise ValueError("beta <= 3 requires a finite rcut to normalize total mass.")

    # base profile as function of x = r/a, with base_at_a analytic
    def _base(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(x > 0.0, x**(-gamma) * (1.0 + x**alpha)**(-(beta - gamma) / alpha), 0.0)
        return out

    base_at_a = (2.0**(-(beta - gamma) / alpha))

    # integrand for mass normalization (rho(a)==1 convention)
    def _integrand(r):
        x = r / a
        rho_unit = _base(x) / base_at_a
        if (rcut is not None) and (rcut > 0):
            rho_unit *= np.exp(- (r / rcut)**cutoffstrength)
        return r**2 * rho_unit

    # choose finite integration upper bound
    if (rcut is not None) and (rcut > 0):
        upper = float(rcut * 8.0)
    else:
        # use a very large but finite upper bound if no cutoff provided
        upper = max(a * 1e4, 1e3)

    I, _ = quad(_integrand, 0.0, upper, epsrel=1e-6, limit=200)
    mass_unit = 4.0 * np.pi * I
    if not np.isfinite(mass_unit) or mass_unit <= 0.0:
        raise RuntimeError("Normalization integral failed; try providing rcut or different slopes.")

    rho_a = float(mass) / mass_unit

    def rho(r):
        r_arr = np.asarray(r, dtype=float)
        x = r_arr / a
        rho_vals = rho_a * (_base(x) / base_at_a)
        if (rcut is not None) and (rcut > 0):
            rho_vals *= np.exp(- (r_arr / rcut)**cutoffstrength)
        return rho_vals

    return rho

def fit_double_spheroid_profile(
    r_centers: np.ndarray = np.array([]), rho_vals: np.ndarray = np.array([]),
    positions: np.ndarray = np.array([]), masses: np.ndarray|float =np.array([]), 
    bins: int = 20, axis_y: float = 1.0, axis_z: float = 1.0,  weighting: str|np.ndarray ='uniform', 
    plot_results: bool = False, return_profiles: bool = False,
    rcut: float|None = None, cutoff_strength: float = 2.0,
)-> Union[
    tuple[float, float, float, float, float],
    tuple[float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:   
    """
    Fit a spheroid (Zhao / generalized double-power-law) profile to particle data.

    Behaviour:
      - If r_centers and rho_vals are not provided (or mismatched), they are
        estimated from `positions` and `masses` using ellipsoidal radii
        r_tilde = sqrt(x^2 + (y/axis_y)^2 + (z/axis_z)^2).
      - If `agama` is available, the code will
        call agama.Potential(type='Spheroid', ...) to compute densities during
        the fit. Otherwise a pure-python density implementation is used.
      - We fit total mass (logM), scale radius (loga), alpha, beta, gamma. 
      Mass -> density normalization is done internally when using the pure-Python model.

    Parameters
    ----------
    r_centers : ndarray (Nbins,) or empty
        Radii where densities are measured (if empty, derived from positions).
    rho_vals : ndarray (Nbins,) or empty
        Densities at r_centers (if empty, computed from positions,masses).
    positions : ndarray (N,3)
        Particle positions (used if r_centers/rho_vals not provided).
    masses : ndarray (N,)
        Particle masses (used if r_centers/rho_vals not provided).
    bins : int
        Number of radial bins when estimating rho from particles.
    axis_y, axis_z : float
        axis ratios b/a and c/a for ellipsoidal radius and volume.
    weighting : 'uniform' | 'inner' | 'outer' | 'sqrt' | 'inverse_sqrt' | array_like
        Weights in the objective (applied to squared log residuals).
    plot_results : bool
        Show diagnostic plots.
    return_profiles : bool
        Whether to return raw profiles. [r_centers, rho_vals, rho_residuals, r2_rho_vals] 
    rcut : float or None
        Optional outer truncation radius for integral / normalization (physical units).
        If None the code integrates to a large radius (see code).
    cutoff_strength : float
        Controls steepness of exponential cutoff when rcut is given. The fallback
        multiplies density by exp[-(r/rcut)**cutoff_strength].

    Returns
    -------
    (M_fit, a_fit, alpha_fit, beta_fit, gamma_fit,
     r_centers, rho_vals, rho_residuals, r2_rho_vals if return_profiles)
    """
    
    AGAMA_AVAILABLE = False
    try:
        import agama
        agama.setUnits(mass=1, length=1, velocity=1)
        AGAMA_AVAILABLE = True
    except Exception:
        AGAMA_AVAILABLE = False

    # --- derive r_centers, rho_vals if not provided ---
    if (len(r_centers) != len(rho_vals)) or len(rho_vals) < 2:
        if positions is None or masses is None or len(positions) == 0:
            raise ValueError("Either supply r_centers & rho_vals, or positions & masses to estimate them.")
        positions = np.asarray(positions, dtype=float)

        # Check shape of positions
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"`positions` must be an (N,3) array, got shape {positions.shape}")        

        # Handle scalar vs array masses
        if np.isscalar(masses):
            masses = np.full(len(positions), masses, dtype=float)
        else:
            masses = np.asarray(masses, dtype=float)
            if masses.shape[0] != positions.shape[0]:
                raise ValueError(
                    f"Length of masses ({masses.shape[0]}) does not match number of positions ({positions.shape[0]})"
                )
               
        # Ellipsoidal radii
        x, y, z = positions.T
        r_tilde = np.sqrt(x**2 + (y/axis_y)**2 + (z/axis_z)**2)
        
        # Radial bins
        rmin, rmax = 0.1, np.percentile(r_tilde, 90)
        edges = make_uneven_grid(rmin, rmax, nbins=bins + 1)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        
        # Density calculation
        volumes = (4.0/3.0) * np.pi * axis_y * axis_z * (edges[1:]**3 - edges[:-1]**3)
        counts, _ = np.histogram(r_tilde, bins=edges, weights=masses)
        rho_vals = counts / np.maximum(volumes, 1e-18)
        M0 = masses.sum()

    else:
        # given: r_centers (log-spaced), rho_vals (same length)
        lnr = np.log(r_centers)
        y   = rho_vals * r_centers**3      # integrand in d ln r
        M0 = 4 * np.pi * np.trapezoid(y, x=lnr)
    
    # Setup weights
    if isinstance(weighting, str):
        weight_map = {
            'uniform': np.ones_like(r_centers),
            'inner': 1.0 / np.maximum(r_centers**2, 1e-18),
            'outer': r_centers**2,
            'sqrt': np.sqrt(np.maximum(r_centers, 1e-18)),
            'inverse_sqrt' : 1 / np.sqrt(np.maximum(r_centers, 1e-18))
        }
        weights = weight_map.get(weighting, np.ones_like(r_centers))
    else:
        if len(weighting) != len(r_centers): 
            raise ValueError("weighting array length must match r_centers|number of profile points|number of radial bins.") 
        weights = np.asarray(weighting)
            
    log_rho_data = np.log10(np.maximum(rho_vals, 1e-12))
  
    def objective(params):
        logM, loga, alpha, beta, gamma = params
        try:
            if AGAMA_AVAILABLE:
                pot = agama.Potential(type='Spheroid', mass=10**logM, scaleRadius=10**loga,
                                      alpha=alpha, beta=beta, gamma=gamma,
                                      axisRatioY=axis_y, axisRatioZ=axis_z, )
                coords = np.column_stack([r_centers, np.zeros(len(r_centers)), np.zeros(len(r_centers))])
                rho_model = pot.density(coords)

            else:
                pot = double_power_law_density(10**logM, 10**loga, 
                                               alpha, beta, gamma, 
                                               rcut=rcut, cutoffstrength=cutoffstrength)
                rho_model = pot.density(r_centers)
            
            log_rho_model = np.log10(np.maximum(rho_model, 1e-12))
            return np.sum(weights * (log_rho_model - log_rho_data)**2)
        except:
            return 1e10
    
    # Optimization
    p0 = [np.log10(M0), np.log10(5.0), 1.0, 3.0, 1.0]  # NFW-like
    bounds = [(np.log10(M0*0.8), np.log10(M0*1.2)), (np.log10(0.1), np.log10(r_centers[-1])),
              (0.1, np.inf), (1.0, np.inf), (0.0, np.inf)]
              # (0.1, 3.0), (1.0, 5.0), (0.0, 3.0)]
    
    res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds)
    logM_fit, loga_fit, alpha_fit, beta_fit, gamma_fit = res.x
    
    # Results
    M_fit, a_fit = 10**logM_fit, 10**loga_fit
    pot_fit = agama.Potential(type='Spheroid', mass=M_fit, scaleRadius=a_fit,
                             alpha=alpha_fit, beta=beta_fit, gamma=gamma_fit,
                             axisRatioY=axis_y, axisRatioZ=axis_z)
    
    coords = np.column_stack([r_centers, np.zeros(len(r_centers)), np.zeros(len(r_centers))])
    rho_model = pot_fit.density(coords)
    
    rho_residuals = rho_vals - rho_model     
    r2_rho_vals = r_centers**2 * rho_vals
  
    # Plotting
    if plot_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
        
        ax1.loglog(r_centers, rho_vals, c=plt.rcParams['text.color'], markersize=6, alpha=0.7, label='Data')
        ax1.loglog(r_centers, rho_model, 'r--', linewidth=2, label='Model')
        ax1.set_xlabel('r')
        ax1.set_ylabel(r'$\rho(r)$')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R^2
        ss_res = np.sum(rho_residuals**2)
        ss_tot = np.sum((rho_vals - np.mean(rho_vals))**2)
        r_squared = 1 - (ss_res / ss_tot)        
        
        ax1.text(0.95, 0.95, f'$R^2 = {r_squared:.2f}$', transform=ax1.transAxes, 
                ha='right', va='top', fontweight='bold', fontsize=12)
        
        r2_rho_model = r_centers**2 * rho_model
        ax2.loglog(r_centers, r2_rho_vals, c=plt.rcParams['text.color'], markersize=6, alpha=0.7, label='Data')
        ax2.loglog(r_centers, r2_rho_model, 'r--', linewidth=2, label='Model')
        ax2.set_xlabel('r')
        ax2.set_ylabel(r'$r^2 \rho(r)$')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # R^2
        r2_rho_residuals = r2_rho_vals - r2_rho_model
        ss_res = np.sum(r2_rho_residuals**2)
        ss_tot = np.sum((r2_rho_vals - np.mean(r2_rho_vals))**2)
        r_squared_2 = 1 - (ss_res / ss_tot)        
        
        ax2.text(0.95, 0.95, f'$R^2 = {r_squared_2:.2f}$', transform=ax2.transAxes,
                ha='right', va='top', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    if return_profiles:
        return (M_fit, a_fit, alpha_fit, beta_fit, gamma_fit), (r_centers, rho_vals, rho_residuals, r2_rho_vals) 
    return (M_fit, a_fit, alpha_fit, beta_fit, gamma_fit)

# Optional Numba import
from scipy import linalg
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(nopython=True, cache=True): # Dummy decorator
        def decorator(func):
            return func
        return decorator
    print("Numba not found. Running with pure NumPy/SciPy (might be slower for some operations).")

@jit(nopython=True, cache=True)
def _calculate_particle_distances_sq(xyz_coords: np.ndarray) -> np.ndarray:
    """
    Computes squared Euclidean distances from the origin for an array of coordinates.

    Parameters
    ----------
    xyz_coords : numpy.ndarray, shape (n, 3)
        Array of n particle coordinates (x, y, z).

    Returns
    -------
    numpy.ndarray, shape (n,)
        Array of squared distances from the origin for each particle.
    """
    return np.sum(xyz_coords**2, axis=1)


@jit(nopython=True, cache=True)
def _compute_weighted_structure_tensor(coords: np.ndarray, masses: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Computes the weighted structure tensor (also known as the second moment tensor).

    The structure tensor S_ij is calculated as:
    S_ij = sum_k (mass_k * weight_k * coord_i_k * coord_j_k) / sum_k (mass_k * weight_k)
    where k iterates over particles. Eigenvectors of this tensor give the principal axes
    of the particle distribution, and eigenvalues are proportional to the square of the
    lengths of these principal axes.

    Parameters
    ----------
    coords : numpy.ndarray, shape (m, 3)
        Coordinates of m selected particles.
    masses : numpy.ndarray, shape (m,)
        Masses of the m selected particles.
    weights : numpy.ndarray, shape (m,)
        Additional weights for each particle (e.g., 1/Rsph_k^2 for reduced tensor).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        The 3x3 weighted structure tensor. Returns a small diagonal matrix if no
        particles or zero total effective weight.
    """
    if coords.shape[0] == 0: # No particles
        return np.diag(np.array([1e-9, 1e-9, 1e-9], dtype=coords.dtype))

    effective_weights = masses * weights # Element-wise
    sum_effective_weights = np.sum(effective_weights)

    if sum_effective_weights == 0:
        return np.diag(np.array([1e-9, 1e-9, 1e-9], dtype=coords.dtype))

    structure = np.zeros((3, 3), dtype=coords.dtype)
    for k in range(coords.shape[0]):
        w_k = effective_weights[k]
        x_k = coords[k, 0]
        y_k = coords[k, 1]
        z_k = coords[k, 2]
        # Accumulating outer product x_k * x_k.T weighted by w_k
        structure[0, 0] += w_k * x_k * x_k
        structure[0, 1] += w_k * x_k * y_k
        structure[0, 2] += w_k * x_k * z_k
        # structure[1, 0] is structure[0, 1]
        structure[1, 1] += w_k * y_k * y_k
        structure[1, 2] += w_k * y_k * z_k
        # structure[2, 0] is structure[0, 2]
        # structure[2, 1] is structure[1, 2]
        structure[2, 2] += w_k * z_k * z_k
    
    # Symmetrize due to potential floating point asymmetries if calculated fully
    structure[1,0] = structure[0,1]
    structure[2,0] = structure[0,2]
    structure[2,1] = structure[1,2]
    
    return structure / sum_effective_weights


@jit(nopython=True, cache=True)
def _calculate_Rsphall_and_extract(
    all_coords: np.ndarray,
    transform_matrix_cols_bca: np.ndarray, 
    q_ratio: float, 
    s_ratio: float, 
    Rmax_scaled: float,
    Rmin_val: float, 
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates ellipsoidal radius (Rsph) for all particles and extracts those within bounds.

    Rsph is a measure of distance in a coordinate system scaled by the current ellipsoidal
    shape (defined by q_ratio = b/a, s_ratio = c/a).
    The transformation `transform_matrix_cols_bca` (with columns e_b, e_c, e_a) rotates
    particles into the principal axis frame. Coordinates are then scaled:
    X_scaled_b = (X_b / (b/a)), X_scaled_c = (X_c / (c/a)), X_scaled_a = (X_a / 1).
    Rsph = norm([X_scaled_b, X_scaled_c, X_scaled_a]).
    Particles are selected if their Rsph is less than `Rmax_scaled` (and greater than
    a similarly scaled `Rmin_val`).

    Parameters
    ----------
    all_coords : numpy.ndarray, shape (N, 3)
        Coordinates of all N particles in the system.
    transform_matrix_cols_bca : numpy.ndarray, shape (3, 3)
        Transformation matrix whose columns are the eigenvectors e_b, e_c, e_a
        (intermediate, minor, major axes of the ellipsoid from the previous iteration).
    q_ratio : float
        Axis ratio b/a from the previous iteration.
    s_ratio : float
        Axis ratio c/a from the previous iteration.
    Rmax_scaled : float
        The scaled maximum radius (aperture) for selection. This is typically
        Rmax_physical / ( (q_ratio * s_ratio)**(1/3) ).
    Rmin_val : float
        The physical minimum radius for shell selection.

    Returns
    -------
    Rsphall_values : numpy.ndarray, shape (N,)
        Ellipsoidal radius for all N particles.
    extract_mask : numpy.ndarray, dtype bool, shape (N,)
        Boolean mask indicating which particles are selected.
    Rsph_extracted : numpy.ndarray
        Ellipsoidal radii of the selected particles.
    """
    if all_coords.shape[0] == 0:
        return (
            np.empty(0, dtype=all_coords.dtype),
            np.empty(0, dtype=np.bool_), # Numba needs np.bool_ for boolean arrays
            np.empty(0, dtype=all_coords.dtype)
            )

    coords_in_eigenframe_bca = np.dot(all_coords, transform_matrix_cols_bca)
            
    # Ensure scales have the same dtype as coordinates for Numba compatibility
    scales = np.array([q_ratio, s_ratio, 1.0], dtype=all_coords.dtype) 
    epsilon = 1e-9 # Smallest allowed ratio to prevent division by zero
    if scales[0] < epsilon: scales[0] = epsilon
    if scales[1] < epsilon: scales[1] = epsilon

    scaled_coords_bca = np.empty_like(coords_in_eigenframe_bca)
    for i in range(coords_in_eigenframe_bca.shape[0]):
        scaled_coords_bca[i,0] = coords_in_eigenframe_bca[i,0] / scales[0]
        scaled_coords_bca[i,1] = coords_in_eigenframe_bca[i,1] / scales[1]
        scaled_coords_bca[i,2] = coords_in_eigenframe_bca[i,2] / scales[2]

    Rsphall_values_sq = np.sum(scaled_coords_bca**2, axis=1)
    Rsphall_values = np.sqrt(Rsphall_values_sq)

    extract_mask = Rsphall_values < Rmax_scaled
    if Rmin_val > 0:
        vol_scale_factor_inv = 1.0
        # Check (q_ratio * s_ratio) to avoid issues if it's zero or negative (shouldn't be)
        # Product can be zero if s_ratio or q_ratio is zero.
        prod_qs = q_ratio * s_ratio
        if prod_qs > 1e-9 : 
             vol_scale_factor_inv = prod_qs**(1./3.)
        
        Rmin_ap_scaled = Rmin_val / vol_scale_factor_inv if vol_scale_factor_inv > 1e-9 else Rmin_val * 1e9
        extract_mask &= (Rsphall_values >= Rmin_ap_scaled)

    Rsph_extracted = Rsphall_values[extract_mask]
    return Rsphall_values, extract_mask, Rsph_extracted

def compute_morphological_diagnostics(
    XYZ: np.ndarray,
    mass: np.ndarray | None = None,
    Vxyz: np.ndarray | None = None,
    Rmin: float = 0.0,
    Rmax: float = 1.0,
    reduced_structure: bool = True,
    orient_with_momentum: bool = True,
    tol: float = 1e-4,
    max_iter: int = 50,
    verbose: bool = False,
    return_ellip_triax: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float | None, float | None] :
    """
    Computes morphological diagnostics (axis lengths, principal axes, ellipticity,
    triaxiality) for a distribution of particles using an iterative approach based
    on the structure/inertia tensor.

    Method Overview:
    ----------------
        The function iteratively determines the shape of a particle distribution.
        1. Particle Selection: Initially, particles within a spherical shell (Rmin, Rmax)
           are selected. In subsequent iterations (if `reduced_structure` is True),
           particles are selected if they fall within an ellipsoid defined by the shape
           (axis ratios q=b/a, s=c/a) determined in the previous iteration. The
           ellipsoid's volume is typically kept comparable to the initial spherical volume.
        2. Structure Tensor: For the selected particles, a weighted structure tensor
           S_ij = sum(w_k * x_i_k * x_j_k) / sum(w_k) is computed.
           If `reduced_structure` is True, weights w_k are proportional to 1/Rsph_k^2,
           where Rsph_k is the particle's ellipsoidal radius normalized by the median
           ellipsoidal radius. This "reduces" the contribution of outer particles,
           focusing on the shape of the inner distribution.
        3. Diagonalization: The structure tensor is diagonalized to find its eigenvalues
           (proportional to squared principal axis lengths a^2, b^2, c^2) and
           eigenvectors (the principal axes e_a, e_b, e_c).
        4. Orientation (Optional): If `orient_with_momentum` is True and velocities
           (Vxyz) are provided, the minor principal axis (e_c) is aligned with the
           total angular momentum vector of the currently selected particles. The other
           axes are adjusted to maintain a right-handed orthonormal system.
        5. Convergence: New axis ratios (q_new = b/a, s_new = c/a) are calculated.
           The process repeats from step 1 (if `reduced_structure`) until the change
           in q and s between iterations is below the tolerance `tol`, or `max_iter`
           is reached.
        6. Output: The function returns the normalized principal semi-axis lengths (a,b,c
           such that a>=b>=c and a=1), the transformation matrix (rows are principal
           axes), and optionally ellipticity and triaxiality.

    Parameters
    ----------
    XYZ : numpy.ndarray, shape (n, 3)
        Particle coordinates (units of length L). XYZ[:,0]=X, XYZ[:,1]=Y, XYZ[:,2]=Z.
    mass : numpy.ndarray, optional, shape (n,)
        Particle masses (units of mass M). If None, assumes mass of 1.0 for all.
    Vxyz : numpy.ndarray, optional, shape (n, 3)
        Particle velocities (units of velocity V). Vxyz[:,0]=Vx, etc.
        Required if `orient_with_momentum` is True.
    Rmin : float, optional
        Minimum radius of the initial spherical shell for particle selection (L).
        For iterative reduced structure, this Rmin is effectively applied within
        the deformed ellipsoidal selection. Default is 0.0.
    Rmax : float, optional
        Maximum radius (aperture) for initial spherical particle selection (L).
        This also sets the scale for the iterative ellipsoidal selection. Default is 1.0.
    reduced_structure : bool, optional
        If True, adopts the iterative reduced form of the inertia tensor, where
        particle contributions are weighted by 1/Rsph^2 and the selection ellipsoid
        adapts. If False, a single pass is made on the initial spherical selection
        with uniform weights. Default is True.
    orient_with_momentum : bool, optional
        If True and `Vxyz` is provided, aligns the minor principal axis with the
        angular momentum vector of the selected particles *during each iteration*.
        This influences subsequent particle selection if `reduced_structure` is True.
        Default is True.
    tol : float, optional
        Tolerance for convergence of axis ratios (q=b/a, s=c/a) in the iterative
        process. Convergence is checked based on the maximum squared fractional
        change: max( (1 - q_new/q_old)^2, (1 - s_new/s_old)^2 ). Default is 1e-4.
    max_iter : int, optional
        Maximum number of iterations for convergence. Default is 50.
    verbose : bool, optional
        If True, print iteration progress and diagnostic information. Default is False.
    return_ellip_triax : bool, optional
        If True, return ellipticity and triaxiality in addition to `abc` and
        `Transform`. Default is False.

    Returns
    -------
    abc_normalized : numpy.ndarray, shape (3,)
        The principal semi-axis lengths (a,b,c) normalized such that a=1.0.
        Ordered as a >= b >= c.
    final_transform_matrix : numpy.ndarray, shape (3, 3)
        Orthogonal transformation matrix. Rows are the unit vectors of the
        principal axes (e_a, e_b, e_c) in the original coordinate system,
        corresponding to a_out, b_out, c_out.
        `Transform[0]` = major axis (e_a)
        `Transform[1]` = intermediate axis (e_b)
        `Transform[2]` = minor axis (e_c)
    ellip : float, optional
        Ellipticity (1 - c/a). Returned if `return_ellip_triax` is True.
        Value is between 0 (spherical) and 1 (infinitely thin disk/needle).
    triax : float, optional
        Triaxiality parameter ((a^2 - b^2) / (a^2 - c^2)), assuming a >= b >= c.
        Ranges from 0 (oblate, a=b>c) to 1 (prolate, a>b=c).
        Is NaN if a=c (spherical) and handled as 0.0 if a=b=c.
        Returned if `return_ellip_triax` is True.

    Guidance on `orient_with_momentum`:
    -----------------------------------
        - Iterative Alignment (this function's behavior if `orient_with_momentum=True`):
          Aligning the minor axis with the angular momentum of the *currently selected particles*
          in each iteration means the particle selection for the *next* iteration is influenced
          by this orientation. This can be useful for studying systems where shape and internal
          kinematics are expected to be tightly coupled at all scales considered by the iteration.
          It seeks a self-consistent shape-orientation state.
    
        - Final Alignment (Alternative Approach, not directly implemented here):
          An alternative is to compute the shape iteratively *without* momentum alignment during
          iterations (i.e., `orient_with_momentum=False`). After the shape converges, calculate the
          total angular momentum of the final particle set and then report the orientation of the
          derived ellipsoid with respect to this global angular momentum vector. This is simpler
          and often preferred if the goal is a single global shape and its relation to global spin.
    
        - For Dark Matter Halos:
          - If analyzing the intrinsic shape profile and how it correlates with the local angular
            momentum at different ellipsoidal radii, iterative alignment might be informative.
          - For a single, overall shape estimate (e.g., within R200), setting
            `orient_with_momentum=False` and then calculating the spin parameter or
            shape-spin alignment *after convergence* using the final particle set is common.
            This decouples the shape determination from a potentially noisy or radially varying
            momentum vector.
          - Iterative alignment can be sensitive if the angular momentum vector is weak or its
            direction changes significantly within the region being iterated upon.

    Warning for Shell Analysis with Iterative Momentum Orientation:
    -------------------------------------------------------------
        When using `orient_with_momentum=True` for a shell analysis (i.e., `Rmin > 0`),
        be aware that:
        - The angular momentum is calculated *only* for particles within that specific shell.
        - The iterative alignment attempts to align the minor axis of the *shell's inertia tensor*
          with the *shell's angular momentum vector*.
        - This orientation (and the subsequent shape refinement based on it) reflects the
          properties of that specific shell. It may not represent the global orientation or
          shape characteristics of the entire halo, as the shell's angular momentum and
          particle distribution can differ from the overall system or inner regions.
        - This approach is valid for studying the properties of discrete shells but exercise
          caution when extrapolating these shell-specific oriented shapes to the entire object.
    """
    # --- Input Validation ---
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise ValueError("XYZ must be a 2D array with shape (n, 3).")
    n_particles = XYZ.shape[0]

    if mass is None:
        mass_arr = np.ones(n_particles, dtype=float)
    else:
        mass_arr = np.asarray(mass, dtype=float)
        if mass_arr.ndim != 1 or mass_arr.shape[0] != n_particles:
            raise ValueError("mass must be a 1D array with shape (n,).")
        if np.any(mass_arr < 0):
            raise ValueError("Particle masses must be non-negative.")

    use_momentum_orientation_flag = False
    if Vxyz is not None:
        Vxyz_arr = np.asarray(Vxyz, dtype=float)
        if Vxyz_arr.ndim != 2 or Vxyz_arr.shape[1] != 3:
            raise ValueError("Vxyz must be a 2D array with shape (n, 3).")
        if Vxyz_arr.shape[0] != n_particles:
            raise ValueError("Vxyz must have the same number of particles as XYZ.")
        if orient_with_momentum:
            use_momentum_orientation_flag = True
    elif orient_with_momentum: # Vxyz is None but orient_with_momentum is True
        if verbose:
            print("Warning: `orient_with_momentum` is True, but `Vxyz` (velocities) not provided. Disabling momentum orientation.")
        # Vxyz_arr will be zeros, use_momentum_orientation_flag remains False
    
    # Ensure Vxyz_arr exists even if not used for momentum, for consistent typing later if accessed
    if Vxyz is None:
        Vxyz_arr = np.zeros_like(XYZ, dtype=float)


    if not (Rmin >= 0 and Rmax > 0 and Rmax > Rmin):
        raise ValueError("Rmin must be >= 0, Rmax > 0, and Rmax > Rmin.")

    XYZ_all = XYZ
    mass_all = mass_arr
    Vxyz_all = Vxyz_arr

    # q_iter, s_iter are b/a, c/a from the *previous* iteration. Start spherical.
    q_iter = 1.0 
    s_iter = 1.0  
    
    # Defines principal axes (columns e_b, e_c, e_a) for Rsphall calculation.
    # Initialized to align with Y, X, Z (b-like, c-like, a-like for initial q,s=1).
    current_transform_cols_bca = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]], dtype=float).T

    # Stores [b^2, c^2, a^2] from current iteration's structure tensor, used for q_new, s_new.
    eigval_loop_bca = np.array([1.0,1.0,1.0], dtype=float) 

    # Initial particle selection based on spherical shell
    initial_distances_sq_all = _calculate_particle_distances_sq(XYZ_all)
    extract_mask = (initial_distances_sq_all < Rmax**2) & (initial_distances_sq_all >= Rmin**2)

    # --- Iteration loop ---
    for iter_num in range(max_iter):
        if verbose:
            print(f"Iteration {iter_num + 1}/{max_iter}...")

        # 1. Select particles for this iteration
        if iter_num > 0 and reduced_structure:
            # Scale Rmax by volume factor (q*s)^(1/3) to keep enclosed volume ~constant
            # q_iter, s_iter are from the *previous* iteration's shape
            vol_scale_factor_inv = 1.0
            prod_qs_prev = q_iter * s_iter
            if prod_qs_prev > 1e-9: # Avoid issues with zero or tiny q/s
                vol_scale_factor_inv = prod_qs_prev**(1./3.)
            
            Rmax_scaled_aperture = Rmax / vol_scale_factor_inv if vol_scale_factor_inv > 1e-9 else Rmax * 1e9 # Effectively infinity if vol_scale is zero

            # `current_transform_cols_bca` and `q_iter, s_iter` are from the previous iteration
            _, extract_mask, Rsph_extracted = _calculate_Rsphall_and_extract(
                XYZ_all,
                current_transform_cols_bca, 
                q_iter, 
                s_iter, 
                Rmax_scaled_aperture,
                Rmin # Physical Rmin, gets scaled inside helper
            )
            
            if np.sum(extract_mask) < 10: # Need enough particles for stable tensor
                if verbose: print("Too few particles selected in reduced step. Using previous iteration's shape or stopping.")
                break # Exit loop, will use results from previous successful iteration
            
            # Calculate Rsph weights for structure tensor based on extracted particles
            Rsph_values_for_weighting = Rsph_extracted # These are for the currently extracted particles
            if Rsph_values_for_weighting.size > 0 :
                median_Rsph = np.median(Rsph_values_for_weighting)
                if median_Rsph < 1e-9: median_Rsph = 1.0 # Avoid division by zero or tiny median
                Rsph_factors = Rsph_values_for_weighting / median_Rsph
                # Prevent Rsph_factors from being zero if median_Rsph was tiny or Rsph_values_for_weighting had zeros
                Rsph_factors[Rsph_factors < 1e-6] = 1e-6 
                structure_tensor_weights = 1.0 / (Rsph_factors**2)
            else: # Should not happen if sum(extract_mask) >= 10
                num_extracted = np.sum(extract_mask)
                structure_tensor_weights = np.ones(num_extracted if num_extracted > 0 else 0, dtype=float)
        else: # First iteration or not using reduced_structure
            if iter_num == 0 : # Use pre-calculated initial spherical mask
                pass # extract_mask is already set
            else: # Not reduced_structure, but iter_num > 0 (loop should break after 1 iter if not reduced)
                 # This re-selects spherically if fixed_aperture and more than one iter (which it shouldn't do)
                 distances_sq_all = _calculate_particle_distances_sq(XYZ_all)
                 extract_mask = (distances_sq_all < Rmax**2) & (distances_sq_all >= Rmin**2)

            if np.sum(extract_mask) < 10:
                if verbose: print(f"Too few particles ({np.sum(extract_mask)}) initially or in fixed selection. Check Rmin/Rmax.")
                # If initial selection is bad, return NaNs
                nan_abc = np.full(3, np.nan)
                nan_transform = np.full((3,3), np.nan)
                if return_ellip_triax: return nan_abc, nan_transform, np.nan, np.nan
                return nan_abc, nan_transform
            
            # Uniform weighting for first iteration or non-reduced structure
            structure_tensor_weights = np.ones(np.sum(extract_mask), dtype=float)

        # Current particles for this iteration's tensor calculation
        current_XYZ = XYZ_all[extract_mask]
        current_mass = mass_all[extract_mask]
        
        # 2. Compute (weighted) structure tensor
        structure = _compute_weighted_structure_tensor(current_XYZ, current_mass, structure_tensor_weights)

        # 3. Diagonalize structure tensor
        # eigval_std has eigenvalues c_s^2, b_s^2, a_s^2 (ascending order)
        # eigvec_std columns are corresponding eigenvectors e_c, e_b, e_a
        try:
            eigval_std, eigvec_std = linalg.eigh(structure)
        except linalg.LinAlgError:
            if verbose: print("linalg.eigh failed during iteration. Using previous iteration's shape or stopping.")
            break 
        eigval_std = np.maximum(eigval_std, 1e-12) # Ensure eigenvalues are non-negative

        # Principal axis squared lengths (a^2 >= b^2 >= c^2)
        current_a_sq = eigval_std[2] # Corresponds to e_a = eigvec_std[:,2]
        current_b_sq = eigval_std[1] # Corresponds to e_b = eigvec_std[:,1]
        current_c_sq = eigval_std[0] # Corresponds to e_c = eigvec_std[:,0]

        # Principal eigenvectors (columns of eigvec_std are e_c, e_b, e_a)
        e_a_vec = eigvec_std[:, 2].copy()
        e_b_vec = eigvec_std[:, 1].copy()
        e_c_vec = eigvec_std[:, 0].copy()
        
        # 4. Orient axes: Ensure right-handed system (e_a, e_b, e_c)
        # Check determinant of [e_a, e_b, e_c]. If < 0, flip one axis (e.g., e_c).
        temp_evec_mat_abc = np.column_stack((e_a_vec, e_b_vec, e_c_vec))
        if np.linalg.det(temp_evec_mat_abc) < 0:
            e_c_vec *= -1.0 # Flipping e_c to make it right-handed with e_a, e_b

        # Optional: Align minor axis (e_c_vec) with angular momentum
        if use_momentum_orientation_flag and current_XYZ.shape[0] > 0: # Vxyz provided and particles exist
            current_Vxyz_selected = Vxyz_all[extract_mask]
            specific_momenta = np.cross(current_XYZ, current_Vxyz_selected)
            total_momentum_vec = np.sum(current_mass[:, np.newaxis] * specific_momenta, axis=0)

            if np.linalg.norm(total_momentum_vec) > 1e-9: # If there is non-negligible momentum
                # Align e_c_vec with total_momentum_vec
                if np.dot(e_c_vec, total_momentum_vec) < 0:
                    e_c_vec *= -1.0
                
                # Re-orthogonalize to maintain e_a, e_b, e_c as a right-handed system
                # with e_c now fixed by momentum.
                # We can preserve the direction of e_a (original major axis from S) as much as possible.
                e_a_original_dir = eigvec_std[:, 2].copy() # Original e_a from S tensor
                
                # If e_a_original_dir is not (anti)parallel to the new e_c_vec
                if np.abs(np.dot(e_a_original_dir, e_c_vec)) < 0.9999:
                    # e_b = normalize(e_c x e_a_original_dir)
                    e_b_vec = np.cross(e_c_vec, e_a_original_dir)
                    norm_eb = np.linalg.norm(e_b_vec)
                    if norm_eb > 1e-9: e_b_vec /= norm_eb
                    else: # Should not happen if e_a_orig and e_c are not parallel
                          # Fallback: if e_b is zero, choose arbitrary perpendicular
                        if abs(e_c_vec[0]) < 0.9: temp_perp = np.array([1.,0.,0.],dtype=float)
                        else: temp_perp = np.array([0.,1.,0.],dtype=float)
                        e_b_vec = np.cross(e_c_vec, temp_perp)
                        e_b_vec /= (np.linalg.norm(e_b_vec) + 1e-9) # Add epsilon

                    # e_a = normalize(e_b x e_c)
                    e_a_vec = np.cross(e_b_vec, e_c_vec)
                    # e_a_vec should already be unit norm if e_b, e_c are unit and ortho.
                    # For safety, re-normalize, though typically not needed if math is precise.
                    norm_ea = np.linalg.norm(e_a_vec)
                    if norm_ea > 1e-9: e_a_vec /= norm_ea
                    else: # e_b and e_c became parallel: Problem! Fallback.
                        if abs(e_b_vec[0]) < 0.9: temp_perp_ea = np.array([1.,0.,0.],dtype=float)
                        else: temp_perp_ea = np.array([0.,1.,0.],dtype=float)
                        e_a_vec = np.cross(e_b_vec, temp_perp_ea)
                        e_a_vec /= (np.linalg.norm(e_a_vec) + 1e-9)

                else: # e_a_original_dir was (anti)parallel to new e_c_vec (e.g., very prolate along L)
                      # Pick an arbitrary e_b perpendicular to e_c_vec
                    if abs(e_c_vec[0]) < 0.9: temp_perp_vec = np.array([1.,0.,0.],dtype=float)
                    else: temp_perp_vec = np.array([0.,1.,0.],dtype=float)
                    
                    e_b_vec = np.cross(e_c_vec, temp_perp_vec)
                    e_b_vec /= (np.linalg.norm(e_b_vec) + 1e-9) # Add epsilon
                    
                    e_a_vec = np.cross(e_b_vec, e_c_vec)
                    e_a_vec /= (np.linalg.norm(e_a_vec) + 1e-9) # Add epsilon


        # For next iteration's Rsphall calculation (needs e_b, e_c, e_a as columns)
        # and corresponding eigenvalues b^2, c^2, a^2 for q,s definition.
        current_transform_cols_bca = np.column_stack((e_b_vec, e_c_vec, e_a_vec))
        eigval_loop_bca = np.array([current_b_sq, current_c_sq, current_a_sq], dtype=float)

        # 5. Update q, s for the *next* iteration's Rsphall (or for convergence check)
        # q_new = b/a, s_new = c/a based on current iteration's shape.
        # Uses eigval_loop_bca which is [b^2, c^2, a^2]
        if eigval_loop_bca[2] > 1e-12: # If a^2 is not essentially zero
            q_new = np.sqrt(eigval_loop_bca[0] / eigval_loop_bca[2]) # sqrt(b^2/a^2)
            s_new = np.sqrt(eigval_loop_bca[1] / eigval_loop_bca[2]) # sqrt(c^2/a^2)
        else: # Degenerate case (a vanishes), treat as spherical to avoid NaNs
            q_new = 1.0
            s_new = 1.0
        
        # Sanitize q_new, s_new: ensure they are valid ratios (0 < s <= q <= 1)
        q_new = np.nan_to_num(q_new, nan=1.0, posinf=1.0, neginf=1.0) # Handle potential NaNs/Infs
        s_new = np.nan_to_num(s_new, nan=1.0, posinf=1.0, neginf=1.0)
        q_new = np.clip(q_new, 1e-3, 1.0) # b/a is between 0 (needle) and 1 (oblate/sphere)
        s_new = np.clip(s_new, 1e-3, q_new) # c/a is between 0 and q_new (c <= b)

        # 6. Convergence check (compare new q,s with q_iter,s_iter from PREVIOUS step)
        if iter_num > 0:
            # Use q_iter, s_iter (from previous iter) as q_old, s_old
            q_old_safe = q_iter if q_iter > 1e-6 else 1.0 # Avoid division by zero
            s_old_safe = s_iter if s_iter > 1e-6 else 1.0
            
            convergence_metric = np.max(np.array([
                (1.0 - q_new / q_old_safe)**2, 
                (1.0 - s_new / s_old_safe)**2
            ]))

            if convergence_metric < tol**2 : 
                if verbose: print(f"Converged at iteration {iter_num + 1} with metric {convergence_metric:.2e}.")
                break # Exit loop
        
        # Update q_iter, s_iter for the *next* iteration (or for output if loop finishes/breaks)
        q_iter = q_new
        s_iter = s_new

        if verbose:
            # current_a_sq, current_b_sq, current_c_sq are from sorted eigenvalues of S
            print(f"  sqrt(eigvals S): a_s={np.sqrt(current_a_sq):.3f}, b_s={np.sqrt(current_b_sq):.3f}, c_s={np.sqrt(current_c_sq):.3f}")
            print(f"  Current axis ratios for next step: q(b/a) = {q_iter:.4f}, s(c/a) = {s_iter:.4f}")
        
        # If not using reduced structure, only one iteration is needed.
        if not reduced_structure and iter_num == 0:
            if verbose: print("Not using reduced structure. Single pass results.")
            break
    else: # Loop finished due to max_iter
        if verbose: print(f"Reached max_iter ({max_iter}) without full convergence to tol={tol:.1e}.")

    # --- Prepare final results ---
    # eigval_loop_bca from the last successful iteration holds [b_k^2, c_k^2, a_k^2]
    # current_transform_cols_bca columns are [e_b_k, e_c_k, e_a_k]
    if np.any(np.isnan(eigval_loop_bca)) or np.sum(extract_mask) < 3:
        if verbose: print("Warning: Finalizing with potentially invalid eigenvalues or too few particles.")
        nan_abc = np.full(3, np.nan)
        nan_transform = np.full((3,3), np.nan)
        if return_ellip_triax: return nan_abc, nan_transform, np.nan, np.nan
        return nan_abc, nan_transform

    # Axis lengths from the last determined shape (a >= b >= c)
    # Based on eigval_loop_bca = [b^2, c^2, a^2]
    # So a_val = sqrt(eigval_loop_bca[2]), b_val = sqrt(eigval_loop_bca[0]), c_val = sqrt(eigval_loop_bca[1])
    a_val_final = np.sqrt(eigval_loop_bca[2]) if eigval_loop_bca[2] > 0 else 0.0
    b_val_final = np.sqrt(eigval_loop_bca[0]) if eigval_loop_bca[0] > 0 else 0.0
    c_val_final = np.sqrt(eigval_loop_bca[1]) if eigval_loop_bca[1] > 0 else 0.0
    
    # Ensure a >= b >= c order for output and normalize by a
    # The loop's q_iter, s_iter already reflect b/a and c/a from this ordering.
    # For output, ensure a_out >= b_out >= c_out
    # Create pairs of (value, vector) for sorting
    # Vectors are e_a_loop, e_b_loop, e_c_loop from current_transform_cols_bca
    e_a_loop_final = current_transform_cols_bca[:, 2] # Vector for a_val_final
    e_b_loop_final = current_transform_cols_bca[:, 0] # Vector for b_val_final
    e_c_loop_final = current_transform_cols_bca[:, 1] # Vector for c_val_final

    # Store as [(value, vector), ...] then sort by value descending for a, b, c output
    # These are the physical axis lengths before normalization
    axis_data_tuples = [
        (a_val_final, e_a_loop_final),
        (b_val_final, e_b_loop_final),
        (c_val_final, e_c_loop_final)
    ]
    # Sort by axis length, descending to get a, b, c
    axis_data_tuples.sort(key=lambda x: x[0], reverse=True)
    
    a_out = axis_data_tuples[0][0]
    b_out = axis_data_tuples[1][0]
    c_out = axis_data_tuples[2][0]

    e_a_out = axis_data_tuples[0][1]
    e_b_out = axis_data_tuples[1][1]
    e_c_out = axis_data_tuples[2][1]
    
    # Normalize abc so that a=1 (dimensionless ratios)
    if a_out > 1e-9: # Avoid division by zero if a_out is effectively zero
      abc_normalized = np.array([1.0, b_out / a_out, c_out / a_out])
    else: 
      abc_normalized = np.array([1.0, 1.0, 1.0]) # Fallback for degenerate case (point-like)
      if verbose: print("Warning: Major axis length a_out is near zero. Resulting shape is ill-defined.")
      # Make b/a and c/a also 1 if a is zero, to represent a sphere of zero size.
      # Or could return NaNs: np.full(3, np.nan) if more appropriate.

    # Final transformation matrix: rows are e_a, e_b, e_c for the sorted a,b,c
    final_transform_matrix = np.vstack((e_a_out, e_b_out, e_c_out))

    if not return_ellip_triax:
        return abc_normalized, final_transform_matrix

    # Ellipticity and Triaxiality using a_out, b_out, c_out
    ellip = 0.0
    if a_out > 1e-9: # Avoid division by zero
        ellip = 1.0 - (c_out / a_out)
    ellip = np.nan_to_num(ellip) # Should not be NaN if a_out > 0

    triax = 0.0 # Default for spherical or cases leading to NaN
    a_out_sq, b_out_sq, c_out_sq = a_out**2, b_out**2, c_out**2
    denominator_triax = a_out_sq - c_out_sq
    if denominator_triax > 1e-12: # Avoid division by zero if a approx c
        triax = (a_out_sq - b_out_sq) / denominator_triax
    elif np.abs(a_out_sq - b_out_sq) < 1e-12 : # Denominator is zero, check if numerator is also zero (sphere a=b=c)
        triax = 0.0 # Sphere (a=b=c)
    # else: triax is 0.0 (e.g. oblate where a=b, so num=0, or prolate where b=c, so T=1 if formula applied before check)
    # The (a^2-b^2)/(a^2-c^2) definition:
    # Sphere: a=b=c -> 0/0 (conventionally 0)
    # Oblate: a=b>c -> 0 / (a^2-c^2) -> 0
    # Prolate: a>b=c -> (a^2-c^2) / (a^2-c^2) -> 1
    # The above logic sets T=0 for oblate. For prolate (b=c), if denom!=0, it's (a^2-c^2)/(a^2-c^2)=1.
    if denominator_triax > 1e-12 and np.abs(b_out_sq - c_out_sq) < 1e-12: # Check for prolate (b=c, a>b)
        triax = 1.0

    triax = np.nan_to_num(triax) # Catch any other NaNs, though logic should prevent them.
    return abc_normalized, final_transform_matrix, ellip, triax


def compute_morphological_diagnostics_legacy(XYZ: np.ndarray, Vxyz: np.ndarray, mass: np.ndarray,
                                             aperture: float = 1.0, 
                                             reduced_structure: bool = True
                                            ) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    """
    Compute the morphological diagnostics through the (reduced or not) inertia tensor.

    Returns the morphological diagnostics for the input particles.

    Parameters
    ----------
    XYZ : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of length L) such that XYZ[:,0] = X,
        XYZ[:,1] = Y & XYZ[:,2] = Z
    mass : array_like of dtype float, shape (n, )
        Particles masses (in unit of mass M)
    Vxyz : array_like of dtype float, shape (n, 3)
        Particles coordinates (in unit of velocity V) such that Vxyz[:,0] = Vx,
        Vxyz[:,1] = Vy & Vxyz[:,2] = Vz
    aperture : float, optional
        Aperture (in unit of length L) for the computation. Default is 1.
    reduced_structure : bool, optional
        Boolean to allow the computation to adopt the iterative reduced form of the
        inertia tensor. Default to True

    Returns
    -------
    ellip : float
        The ellipticity parameter 1-c/a.
    triax : float
        The triaxiality parameter (a^2-b^2)/(a^2-c^2).
    Transform : array of dtype float, shape (3, 3)
        The orthogonal matrix representing the 3 axes as unit vectors: in real-world
        coordinates, Transform[0] = major, Transform[1] = inter, Transform[2] = minor. 
    abc : array of dtype float, shape (3, )
        The corresponding (a,b,c) lengths (in unit of length L).
    """
    import warnings
    from scipy import linalg
    
    # --- Add the deprecation warning here ---
    warnings.warn(
        "The function 'compute_morphological_diagnostics_legacy' is outdated and "
        "will be deprecated in a future version. Please use the newer "
        "'compute_morphological_diagnostics' or 'compute_morphological_diagnostics_numpy' "
        "functions, which offer more options and improvements.",
        DeprecationWarning,  # Use DeprecationWarning category
        stacklevel=2  # Points the warning to the caller of this function
    )
    # --- End of warning ---    

    particlesall = np.vstack([XYZ.T,mass,Vxyz.T]).T
    # Compute distances
    distancesall = np.linalg.norm(particlesall[:,:3],axis=1)
    # Restrict particles
    extract = (distancesall<aperture)
    particles = particlesall[extract].copy()
    distances = distancesall[extract].copy()
    Mass = np.sum(particles[:,3])

    # Compute momentum
    smomentums = np.cross(particlesall[:,:3],particlesall[:,4:7])
    momentum = np.sum(particles[:,3][:,np.newaxis]*smomentums[extract],axis=0)
    # Compute morphological diagnostics
    s = 1; q = 1; Rsphall = 1+reduced_structure*(distancesall-1); stop = False
    while not('structure' in locals()) or (reduced_structure and not(stop)):
        particles = particlesall[extract].copy()
        Rsph = Rsphall[extract]; Rsph/=np.median(Rsph)
        # Compute structure tensor
        structure = np.sum((particles[:,3]/Rsph**2)[:,np.newaxis,np.newaxis]*(np.matmul(particles[:,:3,np.newaxis],particles[:,np.newaxis,:3])),axis=0)/np.sum(particles[:,3]/Rsph**2)
        # Diagonalise structure tensor
        eigval,eigvec = linalg.eigh(structure)
        # Get structure direct oriented orthonormal base
        eigvec[:,2]*=np.round(np.sum(np.cross(eigvec[:,0],eigvec[:,1])*eigvec[:,2]))
        # Return minor axe
        structmainaxe = eigvec[:,np.argmin(eigval)].copy()
        # Permute base and align Y axis with minor axis in momentum direction
        sign = int(np.sign(np.sum(momentum*structmainaxe)+np.finfo(float).tiny))
        structmainaxe *= sign
        # print(sign, eigval, eigvec)
        temp = np.array([1,sign,1])*(eigvec[:,np.int64((np.argmin(eigval)+np.array([(3+sign)/2,0,(3-sign)/2]))%3)])
        eigval = eigval[np.int64((np.argmin(eigval)+np.array([(3+sign)/2,0,(3-sign)/2]))%3)]
        # Permute base to align Z axis with major axis
        foo = int((np.argmax(eigval)/2)*2)
        # return foo, temp
        temp = np.array(([(-1)**(1+foo/2),1,1])*(temp[:,[2-foo,1,foo]]))
        eigval = eigval[[2-foo,1,foo]]
        # Compute change of basis matrix
        transform = linalg.inv(temp)
        stop = (np.max((1-np.sqrt(eigval[:2]/eigval[2])/np.array([q,s]))**2)<1e-4)
        if (reduced_structure and not(stop)):
            q,s = np.sqrt(eigval[:2]/eigval[2])
            Rsphall = linalg.norm(np.matmul(transform,particlesall[:,:3,np.newaxis])[:,:,0]/np.array([q,s,1]),axis=1)
            extract = (Rsphall<aperture/(q*s)**(1/3.))
    Transform = transform.copy()
    ellip = 1-np.sqrt(eigval[1]/eigval[2])
    triax = (1-eigval[0]/eigval[2])/(1-eigval[1]/eigval[2])
    Transform = Transform[...,[2,0,1],:]#so that transform[0] = major, transform[1] = inter, transform[2] = minor
    abc = np.sqrt(eigval[[2,0,1]])
    # Return
    return abc/abc[0], Transform 

######################################################################################
################################Plotting Routines#####################################
######################################################################################

def gauss_filter_surf_dens(
    mass_particle_bin: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a 2D particle mass histogram and convert to surface density.

    This function smooths the binned particle mass distribution using
    ``scipy.ndimage.gaussian_filter`` and then divides by the bin surface area to obtain
    surface mass density.

    Parameters
    ----------
    mass_particle_bin : ndarray, shape (Nx, Ny)
        2D histogram of total particle masses in each bin.
    xedges : ndarray, shape (Nx+1,)
        Bin edges along the x-axis (e.g., projected coordinate).
    yedges : ndarray, shape (Ny+1,)
        Bin edges along the y-axis.
    **kwargs
        Additional keyword arguments passed to
        ``scipy.ndimage.gaussian_filter`` (e.g., ``sigma``).

    Returns
    -------
    surf_dens : ndarray, shape (Nx, Ny)
        Smoothed surface mass density array, in the same shape as ``mass_particle_bin``.

    Notes
    -----
    - Zero-valued bins after Gaussian smoothing are set to 1 before normalization,
      to avoid division-by-zero artifacts.
    - The surface density is computed as:

      .. math::

          \\Sigma(x, y) = \\frac{M_{\\mathrm{smoothed}}(x, y)}{\\Delta x \\; \\Delta y},

      where ``Δx`` and ``Δy`` are the uniform bin widths.

    Examples
    --------
        >>> H, xedges, yedges = np.histogram2d(x, y, bins=100, weights=masses)
        >>> surf_dens = gauss_filter_surf_dens(H, xedges, yedges, sigma=1.0)
        >>> surf_dens.shape
        (100, 100)
    """

    from scipy import ndimage
    # Apply Gaussian smoothing to the histogram data
    mass_particle_bin_gauss = ndimage.gaussian_filter(mass_particle_bin, **kwargs)
    # Calculate the surface area of each bin
    surface_area = np.diff(xedges)[0] * np.diff(yedges)[0] 
    # Compute the surface density of the dark matter particles
    return mass_particle_bin_gauss / surface_area

def _generate_ticks(vmin: float, vmax: float) -> np.ndarray:
    """
    Return a short array of 3-4 'nice' ticks within [vmin, vmax].

    Behavior:
      - Tries n=3 then n=4 ticks.
      - step = span / (n-1) is rounded to an integer if >=1, otherwise to nearest 0.5.
      - sequence anchored at vmin, shifted left by integer multiples of step if needed so the
        last tick doesn't exceed vmax.
      - guarantees ticks lie inside [vmin, vmax] (within floating tolerance).
    """
    span = vmax - vmin
    eps = 1e-12

    for n in (3, 4):
        step_raw = span / (n - 1)
        # simple rounding rule: integer steps if >=1, else 0.5 resolution
        if step_raw >= 1.0:
            step = float(round(step_raw))
            if step <= 0:
                step = 1.0
        else:
            step = round(step_raw * 2.0) / 2.0
            if step <= 0:
                step = step_raw  # fallback

        # anchor at vmin and build n ticks
        start = vmin
        ticks = start + np.arange(n) * step

        # if last tick > vmax, shift left by k*step so last <= vmax
        if ticks[-1] > vmax + eps:
            k = int(np.ceil((ticks[-1] - vmax) / step))
            start = start - k * step
            ticks = start + np.arange(n) * step

        # clamp to the interval and return if we have at least 3 ticks
        inside = ticks[(ticks >= vmin - eps) & (ticks <= vmax + eps)]
        if inside.size >= 3:
            # round to avoid tiny floating artifacts and return
            return np.unique(np.round(inside, 12))

    # fallback: evenly spaced 3 ticks
    return np.round(np.linspace(vmin, vmax, 3), 12)

def projected_density_image(
    part=None, spec: str ='dark', host_props: dict = dict(), 
    spec_ind: list = [], cosmo_box: bool = False, 
    grid_len: float = 100.0, no_bins: int = 2048, 
    xval: str = "X", yval: str = "Z",
    ax: matplotlib.axes.Axes | None = None,
    colorbar_ax: matplotlib.axes.Axes | bool | None = None,
    scale_size: float = 0,
    cmap: matplotlib.colors.Colormap | str | None = None,
    vmin: float | None = None, 
    vmax: float | None = None, 
    gauss_convol: bool = True,
    slice_width: float = 0.0, 
    slice_axis: str = None, 
    density_kind: str = "surface",  # 'surface' or 'volume' (when slice_width>0)
    return_dens: bool = False,  # if True -> return im_obj, surf_density ; else return None
    **kwargs: Any,
) -> None | tuple[matplotlib.image.AxesImage, np.ndarray]:

    """
    Generate a convolved image (imshow) of particle density.
    Can be used to directly pass the pos= (N, 3) arr, and mass= (N, ).

    Parameters
    ----------
    part : object, optional
        The Gizmo ParticleClass. If None, must specify `pos` and `mass` in kwargs. Default is None.
    spec : str, optional
        The type of particles to plot (e.g., 'dark', 'gas', 'star'). Default is 'dark'.
    host_props : dict, optional
        Host properties such as rotation matrix and position. Default is an empty dict.
    spec_ind : array-like, optional
        Indices of the particles to include in the plot. Default is an empty list.
    cosmo_box : bool, optional
        Whether to use a cosmological box for scaling. Default is False.    
    grid_len : float, optional
        The size of the grid in physical coordinates. Default is 100.
        Total box size across each axis will be 2*grid_len. 
    no_bins : int, optional
        The number of bins in the histogram. Default is 2048.
    xval, yval : str, optional
        The axes to plot. Default is 'X' and 'Z'.
    ax : matplotlib.Axes, optional
        The axes on which to plot. Default is None, which will create a new figure.
    colorbar_ax : matplotlib.Axes, optional
        The axes on which to plot the colorbar. Default is None.
    scale_size : float, optional
        The size of the scale bar in kpc. Default is 0 (no scale bar).
    cmap : str or matplotlib colormap, optional
        The colormap to use. Default is None, which will choose based on `spec`.
    vmin, vmax : float, optional
        The colorbar limits. Default is None, which will used optimized scales.
        Set to the same value vmin=vmax for mpl auto-scaling.
    gauss_convol : bool, optional
        Whether to apply Gaussian convolution to the surface density. Default is True.
    slice_width : float, optional
        selects particles within ±slice_width around the center (assumed host-centered). Default is 0 (all-particles).
    density_kind : str, optional
        computes either surface density or volume density (requires a non-zero slice_width). Default is surface.
    **kwargs : additional keyword arguments
        Other arguments for customizing the plot (e.g., scale bar properties).
        Can be used to directly pass the pos (N, 3) arr, and mass (N, ) or scalar. pos=pos, mass=mass.

    Returns (if return_dens True)
    -------
    im_obj : matplotlib.image.AxesImage
        The image object returned by imshow.
    proj_density : numpy.ndarray (2D array.)
        The density data: could be surface or volumetric.
    """

    # Convert string arguments to lowercase and perform basic checks.
    spec, xval, yval = spec.lower(), xval.lower(), yval.lower()
    density_kind = density_kind.lower()
    if slice_axis is not None:
        slice_axis = slice_axis.lower()
    if density_kind not in ("volume", "surface"):
        warnings.warn(f"density_kind '{density_kind}' is not a valid choice. forcing it to 'surface'.")
        density_kind = "surface"
    
    if density_kind == "volume" and slice_width == 0:
            warnings.warn(f"slice_width '{slice_width}' is not a valid choice for volumetric density. forcing it to 1 length units.")
            slice_width = 1.0
        
    # import mpl dependencies. 
    from matplotlib import offsetbox
    from matplotlib.lines import Line2D
    
    # enforce cosmo_box requirement
    if cosmo_box and part is None:
        warnings.warn("cosmo_box=True but no `part` provided; forcing cosmo_box=False.")
        cosmo_box = False
    
    # If part is None, require pos in kwargs
    if part is None: 
        if 'pos' not in kwargs:
            raise ValueError("Either `part` must be provided or kwargs must contain `pos` (N,3).")
        
        pos = kwargs["pos"]
        mass = kwargs.get("mass", np.ones((pos.shape[0],), dtype=float))    
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("kwargs['pos'] must have shape (N,3).")
        # Handle different mass formats
        if np.isscalar(mass):
            # Convert scalar to array of appropriate length
            mass = np.full(pos.shape[0], mass, dtype=float)
        elif mass.ndim == 2 and mass.shape[1] == 1:
            # Convert (N,1) array to (N,) array
            mass = mass.ravel()                
        if mass.shape[0] != pos.shape[0]:
            raise ValueError("mass length must match pos length.")
        bins = np.linspace(-grid_len, grid_len, no_bins + 1)
    
    # extract pos and mass from part & host_props. 
    else: 
        try:
            if "rot" not in host_props:
                host_props["rot"] = part.host["rotation"][0]
        except Exception as e:
            print(e)
            host_props["rot"] = np.identity(3)
    
        try:
            if "pos" not in host_props:
                host_props["pos"] = part.host["position"].flatten()
        except Exception as e:
            # fall back to zero center
            print(e)
            host_props["pos"] = np.zeros(3)
    
        # check whether spec_ind are selected.  
        if len(spec_ind) < 1:
            spec_ind = np.arange(part[spec]["mass"].shape[0])
    
        pos = np.dot(part[spec]["position"][spec_ind] - host_props["pos"], host_props["rot"].T)
        mass = part[spec]["mass"][spec_ind].ravel()
        del host_props # # no-longer required.
    
        if cosmo_box:
            scalef = part.info.get("scalefactor", 1.0)
            bins = np.linspace(-grid_len / scalef, grid_len / scalef, no_bins + 1)
        else:
            bins = np.linspace(-grid_len, grid_len, no_bins + 1)
    
    # axes mapping
    axes_kind = {"x": 0, "y": 1, "z": 2}
    if xval not in axes_kind or yval.lower() not in axes_kind:
        raise ValueError("xval and yval must be one of 'X','Y','Z'.")
     
    # determine slice_axis as leftover if not given
    if slice_axis is None:
        leftover = set(axes_kind.keys()) - {xval, yval}
        if len(leftover) != 1:
            # fallback default
            slice_axis = "Y"
        else:
            slice_axis = leftover.pop()
    
    if slice_axis not in axes_kind:
        raise ValueError("slice_axis must be one of 'X','Y','Z'.")
    
    # If slice_width > 0, select particles within the slice first
    if slice_width > 0:
        ax_idx = axes_kind[slice_axis]
        mask = np.abs(pos[:, ax_idx]) <= slice_width
        if mask.sum() == 0:
            warnings.warn("Slice selection contains zero particles; result will be empty.")
            pos = pos[mask]
            mass = mass[mask]
    
    # compute 2D histogram (mass-weighted) 
    mass_particle_bin, xedges, yedges = np.histogram2d(
        pos[:, axes_kind[xval]], pos[:, axes_kind[yval]],
        weights=mass, bins=bins
    )
    
    if gauss_convol:
        sigma = no_bins/4000 ## good choice! 
        calc_density = gauss_filter_surf_dens(mass_particle_bin, xedges, yedges, sigma=sigma)
    
    else:
        surface_area = np.diff(xedges)[0] * np.diff(yedges)[0] 
        # Compute the surface density of the dark matter particles
        calc_density = mass_particle_bin / surface_area
        
    # compute density
    if slice_width > 0 and density_kind == "volume":
        calc_density /= 2.0 * slice_width
    
    # let's convert to log-scale for plotting.
    if return_dens: dens_to_return = calc_density.copy()
    
    calc_density[calc_density==0] = 1
    calc_density = np.log10(calc_density)

    # colormap selection (keep your behavior)
    if cmap is None:
        try:
            import cmasher as cmr
            colormaps = {"star": cmr.ghostlight, "gas": cmr.gothic, "dark": cmr.eclipse}
            cmap = colormaps.get(spec, cmr.eclipse)
        except Exception:
            cmap = "cubehelix"

    # vmin/vmax logic
    vmins = {"star": 3.0, "gas": 4.0, "dark": 6.5}
    vmaxs = {"star": 10.0, "gas": 9.5, "dark": 9.5}
    vmin_use = vmins.get(spec, 5.0) if vmin is None else vmin
    vmax_use = vmaxs.get(spec, 9.5) if vmax is None else vmax
    if vmin_use == vmax_use:
        vmin_use, vmax_use = None, None

    # plotting the resultant 2D profiles . . . 
    if ax is None:
        fig = plt.figure(figsize=(3, 3), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("k")

    im_obj = ax.imshow(
        calc_density.T,
        interpolation="bilinear",
        cmap=cmap,
        origin="lower",
        vmin=vmin_use,
        vmax=vmax_use,
        aspect=1,
        extent=None if cosmo_box else [-grid_len, grid_len, -grid_len, grid_len],
    )

    # ensures that we use the vmin, vmax that mpl used.
    vmin_use, vmax_use = im_obj.get_clim()    
    
    # remove length scales:
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.tick_params(bottom=False,left=False)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.tick_params(bottom=False,left=False)
    
    if scale_size > 0:
        class AnchoredHScaleBar(offsetbox.AnchoredOffsetbox):
            """ 
            size: length of bar in data units
            extent : height of bar ends in axes units
            """
            def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                         pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None,
                         frameon=True, **kwargs):            
                if not ax:
                    ax = plt.gca()
                trans = ax.get_xaxis_transform()
                size_bar = offsetbox.AuxTransformBox(trans)
                line = Line2D([0,size],[0,0], **kwargs)
                vline1 = Line2D([0,0],[-extent/2.,extent/2.], **kwargs)
                vline2 = Line2D([size,size],[-extent/2.,extent/2.], **kwargs)
                size_bar.add_artist(line)
                size_bar.add_artist(vline1)
                size_bar.add_artist(vline2)
                txt = offsetbox.TextArea(label, textprops=dict(color='white',fontsize=8))
                self.vpac = offsetbox.VPacker(children=[size_bar,txt],
                                         align="center", pad=ppad, sep=sep)
                offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                         borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon)
        
        
        ob = AnchoredHScaleBar(size=scale_size/((grid_len*2/scalef/no_bins) if cosmo_box else 1), 
                               label=f'{{\\bf {int(scale_size)} kpc}}', loc=3, frameon=False,
                               pad=0.2, sep=0.3, borderpad=0.7, color="white", linewidth=2., ax=ax)
        
        ax.add_artist(ob)
        
    if colorbar_ax:
        if isinstance(colorbar_ax, plt.Axes):
            # If colorbar_ax is already an Axes object, continue
            pass
        elif colorbar_ax is True:
            # If colorbar_ax is True, create a new inset axes
            colorbar_ax = ax.inset_axes((0.35, 0.12, 0.6, 0.025))
        else:
            raise ValueError("Invalid input for colorbar_ax. Must be a matplotlib Axes object or True.")
            
        cbar = plt.colorbar(im_obj, cax=colorbar_ax, 
                            cmap=cmap,
                            orientation="horizontal", 
                            aspect=25, 
                            extend='both'
                           )
        cbar.ax.tick_params(size=3, width=0.5, labelsize=10, color="white", labelcolor='white') 
        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(0.12)
        
        # Generate ticks and labels
        ticks = _generate_ticks(vmin_use, vmax_use)
        new_labels = [rf'$\mathbf{{10^{{{int(tick)}}}}}$' for tick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(new_labels)
        
        filtered_kwargs = {key: value for key, value in kwargs.items() if key not in ['pos', 'mass']}
        
        cbar_label = rf'$\mathbf{{{{\Sigma_{{{spec}}}[M_{{\odot}}/kpc^2]}}}}$'
        if density_kind == "volume":
            cbar_label = rf'$\mathbf{{{{\rho_{{{spec}}}[M_{{\odot}}/kpc^3]}}}}$'
            
        ax.text(0.7, 0.2, cbar_label, ha='center', va='center', transform=ax.transAxes,
                color='w', fontsize=12, bbox=dict(facecolor='none', edgecolor='none'), **filtered_kwargs)
    
    
    if return_dens: return im_obj, dens_to_return
    
    return None


def _aggregate_data_chunk(chunk_data):
    """Helper function for parallel processing of pixel data."""
    import pandas as pd
    pixel_indices, weights = chunk_data
    df_chunk = pd.DataFrame({'pixel': pixel_indices, 'weight': weights})
    return df_chunk.groupby('pixel').sum()

def plot_mollweide_projection(
    pos: np.ndarray, weights: np.ndarray | None = None, 
    initial_nside: int = 60, 
    normalize: bool = False, 
    log_scale: bool = True,
    filter_radius: tuple[float, float] = (0, 0), 
    return_map: bool = False, 
    smooth_fwhm_deg: float | None = None,
    verbose: bool = False, 
    add_traj: np.ndarray | list =[], 
    add_end_pt: bool = False, add_traj_dist: bool = False,
    density_threshold: float = 1e5, cmap: str ='bone', **kwargs):
    """
    Generates a Mollweide projection of a 3D field using Healpix with dynamic binning and parallel processing.
    The Gaussian smoothing is dynamically calculated based on `nside` if not specified.

    Parameters
    ----------
    pos : np.ndarray
        3D Cartesian positions of points in the field (shape: [N, 3]).
    weights : np.ndarray, optional
        Weights for each point, such as masses (default is None, which assigns a weight of 1 to each point).
    initial_nside : int, optional
        Initial Healpix resolution parameter (default is 60).        
    normalize : bool, optional
        If True, computes the fractional variation (delta map) relative to the mean surface density.
    log_scale : bool, optional
        if True, convert the computed surface density to Log scale. 
    filter_radius : tuple of float, optional
        Range of distances to filter points, in the form (radius, tolerance) or (rmin, rmax). If both values are >0, only points within
        this radius (with a tolerance) are included.
    filter_radius : tuple of float, optional
        Range of distances to filter points, in the form (radius, tolerance) or (rmin, rmax). 
        - If both values are > 0, the points will be filtered to include only those within 
          a specific radius (with a tolerance). 
        - If both values are >= 0 and the second value is greater than the first, 
          the points will be filtered to be within the radial range [rmin, rmax].
        - The behavior depends on whether the first element is larger than the second 
          (radius and tolerance) or the values are a range (rmin, rmax).
    return_map : bool, optional
        Whether to return the Mollweide map data and surface density map (default is True).
    smooth_fwhm_deg : float, optional
        Full-width half-maximum (FWHM) in degrees for Gaussian smoothing. If None, FWHM is dynamically set based on `nside`.
    density_threshold : int, optional
        Number of points at which the function will automatically increase the `nside` value to improve map resolution.
    add_traj : np.array or list, optional
        Add a trajectory to the mollview map.
    **kwargs : optional
        Additional keyword arguments passed to `hp.mollview`.
        
        some example kwargs for reference:                            
            unit=r'$\\mathbf{Scatter}_\\mathrm{rate}$ $\\mathbf{[Gyr^{-1}]}$',
            xlabel=r"\\boldmath$\\mathbf{\\Phi}$",
            ylabel=r"\\boldmath$\\mathbf{\\theta}$",
            cb_orientation="horizontal",  
            xtick_label_color='w',
            ytick_label_color='k',
            override_plot_properties={'cbar_pad':0.1,
                                        'figure_width': 8, 
                                        'figure_size_ratio': 0.63},
            xtick_label_color='white',
            ytick_label_color='white',
            show_tickmarkers=True
            custom_xtick_labels=[r'$120^\\circ$', r'$60^\\circ$', 
                                    r'$0^\\circ$', r'$-60^\\circ$', 
                                    r'$-120^\\circ$'],
    

    Returns
    -------
    twd_map : np.ndarray, optional
        The projected Mollweide map if `return_map` is True.
    map_smooth : np.ndarray, optional
        The smoothed surface density map if `return_map` is True.
    """
    
    import healpy as hp
    
    # Check if vaex or pandas is available
    try:
        import vaex
        has_vaex = True
    except ImportError:
        import pandas as pd
        has_vaex = False
        from multiprocessing import cpu_count, Pool
        
    # Apply filter based on radial distance if specified
    # Case 1: Using radius and tolerance (radius, tolerance)
    if filter_radius[0] > 0 and filter_radius[1] > 0 and filter_radius[0] >= filter_radius[1]:
        distances = np.linalg.norm(pos, axis=1)
        within_radius = np.where(np.isclose(distances, filter_radius[0], atol=filter_radius[1]))[0]
        pos = pos[within_radius]
        if weights is not None:
            weights = weights[within_radius]
    # Case 2: Using radial range (rmin, rmax)
    elif filter_radius[0] >= 0 and filter_radius[1] > filter_radius[0]:
        distances = np.linalg.norm(pos, axis=1)
        within_range = (distances >= filter_radius[0]) & (distances <= filter_radius[1])
        pos = pos[within_range]
        if weights is not None:
            weights = weights[within_range]

    # Dynamically adjust nside based on data density
    nside = initial_nside
    
    if pos.shape[0] > density_threshold:
        if verbose: print('Dynamically adjusting resolution based on no. of particles')
        nside = min(512, int(initial_nside * (pos.shape[0] / density_threshold) ** 0.5))
        if verbose: print(f'New rez nside: {nside}')
            
    # Convert Cartesian coordinates to spherical coordinates (theta, phi)
    pos_spherical = hp.rotator.vec2dir(pos.T, lonlat=False).T  # Theta, Phi in radians
    
    # Convert spherical coordinates to Healpix pixel indices
    pixel_indices = hp.ang2pix(nside, pos_spherical[:, 0], pos_spherical[:, 1])
    num_pixels = hp.nside2npix(nside)
    
    # If no weights are provided, assign weight of 1 to each point
    if weights is None:
        weights = np.ones(pos.shape[0])

    # Parallelize the data aggregation process if using pandas or vaex
    if has_vaex:
        df = vaex.from_arrays(pixel=pixel_indices, weight=weights)
        pixel_data = df.groupby('pixel', agg={'weight': vaex.agg.sum('weight')})
        pixel_ids, total_mass_in_pixel = pixel_data['pixel'].values, pixel_data['weight'].values
    else:
        # For large datasets (1M+ particles), use parallelized approach
        if pos.shape[0] > 1_000_000:  # 1M threshold for parallel processing
            # Divide data for parallel processing with pandas
            num_chunks = max(4, min(cpu_count() - 1, 8))  # Cap at 8 processes
            
            # Create chunks more efficiently for large data
            chunk_size = len(pixel_indices) // num_chunks
            chunks = [(pixel_indices[i:i+chunk_size], weights[i:i+chunk_size]) 
                     for i in range(0, len(pixel_indices), chunk_size)]
            
            with Pool(num_chunks) as pool:
                chunk_results = pool.map(_aggregate_data_chunk, chunks)

            # Combine results from parallel processing
            df_combined = pd.concat(chunk_results).groupby('pixel').sum()
            pixel_ids = df_combined.index.values.astype(np.int64)  # Ensure integer type
            total_mass_in_pixel = df_combined['weight'].values
        else:
            # For smaller datasets, use numpy (still faster for < 1M particles)
            unique_pixels, inverse_indices = np.unique(pixel_indices, return_inverse=True)
            total_mass_in_pixel = np.bincount(inverse_indices, weights=weights)
            pixel_ids = unique_pixels.astype(np.int64)  # Ensure integer type
            
    # Calculate the surface density per pixel in units of mass per square degree
    area_per_pixel_deg2 = hp.nside2pixarea(nside, degrees=True)
    sky_map = np.zeros(num_pixels)
    sky_map[pixel_ids] = (total_mass_in_pixel / area_per_pixel_deg2)
    if log_scale:
        sky_map[pixel_ids] = np.log10(total_mass_in_pixel / area_per_pixel_deg2)
        
    # Apply normalization if required (fractional variation map)
    if normalize:
        mean_density = np.median(sky_map[sky_map > 0])  # Avoid dividing by zero
        sky_map = sky_map / mean_density - 1  # Fractional variation
    
    # Dynamic smoothing calculation if not provided
    if smooth_fwhm_deg is None:
        if verbose: print(f'Computing fwhm smoothing for pixels.')
        pixel_size_rad = np.sqrt(hp.nside2pixarea(nside))  # Approximate pixel size in radians
        smooth_fwhm_rad = 3 * pixel_size_rad  # Set FWHM to 3 times the pixel size
    else:
        smooth_fwhm_rad = np.radians(smooth_fwhm_deg)
    
    # Smooth the map using the dynamically calculated or specified FWHM
    map_smoothed = hp.smoothing(sky_map, fwhm=smooth_fwhm_rad)
    
    # Plot the map using a Mollweide projection with 'bone' colormap by default
    hp.newvisufunc.projview(map_smoothed,  coord=["G"], 
                            projection_type="mollweide",
                            extend='both',
                            flip='astro', 
                            cmap=cmap, 
                            ## some example kwargs use as you wish
                            # unit=r'$\\mathbf{Scatter}_\\mathrm{rate}$ $\\mathbf{[Gyr^{-1}]}$',
                            # xlabel=r"\boldmath$\mathbf{\Phi}$",
                            # ylabel=r"\boldmath$\mathbf{\theta}$",
                            # cb_orientation="horizontal",  
                            # xtick_label_color='w',
                            # ytick_label_color='k',
                            # override_plot_properties={'cbar_pad':0.1,
                            #                           'figure_width': 8, 
                            #                           'figure_size_ratio': 0.63},
                            # xtick_label_color='white',
                            # ytick_label_color='white',
                            # # show_tickmarkers=True
                            # custom_xtick_labels=[r'$120^\circ$', r'$60^\circ$', 
                            #                      r'$0^\circ$', r'$-60^\circ$', 
                            #                      r'$-120^\circ$'],

                            **kwargs, )
    
    if len(add_traj) > 0:
        theta_traj, phi_traj = hp.rotator.vec2dir(add_traj.T, lonlat=False)#theta,phi    
        hp.newvisufunc.newprojplot(theta_traj, phi_traj, #lw=1, 
                                   c='lime', ls='--')
        if add_end_pt:
            hp.newvisufunc.newprojplot(theta_traj[-1], phi_traj[-1], marker='*', 
                                       c='lime', 
                                       #markersize=10, 
                                       linewidth=0.25)
   
    # twd_map = hp.mollview(map_smoothed, title='', rot=0, cmap=cmap, return_projected_map=True, 
    #                       reuse_axes=False, **kwargs) 
                              
    return map_smoothed if return_map else None    
    
######################################################################################
################################IPython display#######################################
######################################################################################        

def play_gif(file_path: str):
    from IPython.display import Image
    return display(Image(data=open(file_path,'rb').read(), format='png'))  

def sound_noti(sound_file: str = f'data/beep.mp3'):
    from IPython.display import Audio
    return Audio(filename=sound_file, autoplay=True)

def play_video(file_name: str):
    from IPython.display import Video ##works for mp4 with codec avc1.
    return Video(filename=file_name, embed=True)

from IPython.core.magic import Magics, magics_class, line_cell_magic
from IPython import get_ipython

# ANSI Color Codes for Formatting
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"

def timeHuman_from_secs(seconds: float, return_time: bool = False) -> tuple[int, int, int]:
    """
    Convert seconds into a human-readable format and display with colors & progress bar.

    Parameters
    ----------
    seconds : float
        The time duration in seconds.
    return_time : bool, optional
        If True, returns the time in hours, minutes, and seconds. Default is False.

    Returns
    -------
    tuple or None
        If return_time is True, returns a tuple (hours, minutes, seconds). Otherwise, None.
    """
    if seconds < 1e-3:  # Microseconds
        time_str = f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:  # Milliseconds
        time_str = f"{seconds * 1e3:.2f} ms"
    else:  # Standard hh:mm:ss
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    # Apply colors only to the time duration, not the label
    if seconds < 1:
        color = GREEN  # Fast
    elif seconds < 10:
        color = YELLOW  # Medium
    else:
        color = RED  # Slow

    # Progress Bar Representation (max width = 30)
    bar_length = min(30, max(1, int(seconds * 3)))  # Scale dynamically
    bar = f"{CYAN}[{'█' * bar_length}{' ' * (30 - bar_length)}]{RESET}"

    # Properly color only the time value, not the label
    print(f"{BOLD}Time elapsed:{RESET} {color}{time_str}{RESET} {bar}")

    if return_time:
        return int(hours), int(minutes), seconds

@magics_class
class _TimeItMagics(Magics):
    """
    Jupyter magic for timing code execution.

    Usage:
    ------
    Line magic:
        %TimeIt <command>

    Cell magic:
        %%TimeIt
        <code block>
    """

    @line_cell_magic
    def TimeIt(self, line, cell=None):
        """A Jupyter magic that works for both line (`%TimeIt`) and cell (`%%TimeIt`) usage."""
        start_time = time.perf_counter()
        
        if cell is None:
            # Line magic case: single command
            exec(line, self.shell.user_ns)
        else:
            # Cell magic case: multiple lines
            exec(cell, self.shell.user_ns)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        timeHuman_from_secs(elapsed_time)

class TimeIt:
    """
    A utility class for timing code execution in various contexts.

    Usage:
    ------
    As a context manager:
        with cc.TimeIt():
            # Code to be timed

    As a decorator:
        @cc.TimeIt()
        def my_function():
            # Function to be timed

    As a Jupyter magic:
        %TimeIt <command>
        %%TimeIt
        <code block>
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Starts the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Stops the timer and prints elapsed time in a formatted way."""
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        timeHuman_from_secs(elapsed_time)

    def __call__(self, func):
        """Decorator to time functions."""
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            timeHuman_from_secs(elapsed_time)
            return result
        return wrapper

    @staticmethod
    def register_magics():
        """Register the Jupyter magic functions."""
        ipython = get_ipython()
        if ipython:
            ipython.register_magics(_TimeItMagics)

# Automatically register the Jupyter magics when imported
TimeIt.register_magics()

######################################################################################
################################KLD analysis tool#####################################
######################################################################################        
        
def kld_kde(dist0, dist1, bw=None, kernel = 'tophat', verbose=False):
    """
    Returns the kld between the two sampled distributions dist0 and dist1.
    Explicitly: D_KL(dist0|dist1), by Monte Carlo integration
    Uses a kernel density estimator to represent the probability distributions.

    Assumes that the shape of dist0 and dist1 is (nsamples, ndims)

    Optional arguments:
        bw: bandwidth for kernel density estimator.
            If "None", tries to estimate it based on the range of data in the two distributions.
        kernel: what shape of kernel to use (see list in sklearn.neighbors.KernelDensity)
        verbose: whether to print status updates
    """

    #if 1D arrays supplied, add an axis
    if len(dist0) ==0 or len(dist1) == 0:
        return 0
    
    if len(dist0.shape)<2:
        dist0=dist0[:,np.newaxis]
    if len(dist1.shape)<2:
        dist1=dist1[:,np.newaxis]

    #check that dist1 and dist2 have the same dimensions. If not return a flag value
    if dist0.shape[1]!=dist1.shape[1]:
        kld = 0
        if verbose: print('Error: distributions have different number of dimensions')
    if (np.where(dist0.shape==np.min(dist0.shape))[0]!=1) or (np.where(dist1.shape==np.min(dist1.shape))[0]!=1):
        kld = 0
        if verbose: print('Error: shape is not (nsamples,ndims)')
    else:
        if verbose: print('creating kernel densities for the two distributions')

        estimated_bandwidth = (bw is None)
        if estimated_bandwidth:

            #compute the range where the probability density needs compact support in each dimension
            data_ranges = np.zeros((2, dist0.shape[1]))

            for ax in range(dist0.shape[-1]):
                data_ranges[0,ax] = np.array([np.min(dist0[:,ax]),np.min(dist1[:,ax])]).min()
                data_ranges[1,ax] = np.max(np.array([np.max(dist0[:,ax]),np.max(dist1[:,ax])]))

            #estimate the bandwidth to be comparable to even interparticle spacing over this range
            bw_est_0 = np.min((data_ranges[1]-data_ranges[0])/np.sqrt(float(dist0.shape[0])))*3.0
            bw_est_1 = np.min((data_ranges[1]-data_ranges[0])/np.sqrt(float(dist1.shape[0])))*3.0

        if estimated_bandwidth: bw = bw_est_0
        if verbose: print('using bandwidth {0:0.3g} for dist 0'.format(bw))
        kde0 = KernelDensity(bandwidth = bw,kernel=kernel).fit(dist0)

        if estimated_bandwidth: bw = bw_est_1
        if verbose: print('using bandwidth {0:0.3g} for dist 1'.format(bw))
        kde1 = KernelDensity(bandwidth = bw,kernel=kernel).fit(dist1)


        if verbose: print('calculating kld')

        #the KDE returns the natural log of the normalized density
        logpdens = kde0.score_samples(dist0)
        logqdens = kde1.score_samples(dist0)

        #find any points where samples from q were undefined on p
        igood = ~(np.isinf(logqdens))
        if verbose: print('throwing out {0:d} bad points (if this number is large compared to n_samples, try increasing the bandwidth)'.format(np.sum(~igood)))

        kld = (logpdens[igood]-logqdens[igood]).sum()/igood.sum()


    if verbose: print('kld is: {0:0.3g}'.format(kld))

    return kld


def kld_single(dist0, dict1, bw=None, kernel = 'tophat', verbose=False):
    """
    Returns the kld between the two sampled distributions dist0 and dist1.
    Explicitly: D_KL(dist0|dist1), by Monte Carlo integration
    Uses a kernel density estimator to represent the probability distributions.

    Assumes that the shape of dist0 and dist1 is (nsamples, ndims)

    Optional arguments:
        bw: bandwidth for kernel density estimator.
            If "None", tries to estimate it based on the range of data in the two distributions.
        kernel: what shape of kernel to use (see list in sklearn.neighbors.KernelDensity)
        verbose: whether to print status updates
    """
    from sklearn.neighbors import KernelDensity
    #if 1D arrays supplied, add an axis

    if len(dist0)<2:
        return 0
    
    if verbose: print('creating kernel densities for the 1st distributions')

    estimated_bandwidth = (bw is None)
    if estimated_bandwidth:
        #compute the range where the probability density needs compact support in each dimension
        data_ranges = np.zeros((2, dist0.shape[1]))

        for ax in range(dist0.shape[-1]):
            data_ranges[0,ax] = np.array([np.min(dist0[:,ax])]).min()
            data_ranges[1,ax] = np.max(np.array([np.max(dist0[:,ax])]))

            #estimate the bandwidth to be comparable to even interparticle spacing over this range
        bw_est_0 = np.min((data_ranges[1]-data_ranges[0])/np.sqrt(float(dist0.shape[0])))*3.0

    if estimated_bandwidth: bw = bw_est_0
    if verbose: print('using bandwidth {0:0.3g} for dist 0'.format(bw))
    
    kde0 = KernelDensity(bandwidth = bw,kernel=kernel).fit(dist0)

    if verbose: print('calculating kld')

    #the KDE returns the natural log of the normalized density
    logpdens = kde0.score_samples(dist0)
        
    q = 1
    for key in dict1.keys():
        q *= 1/(dict1[key]['max']-dict1[key]['min'])    
        
    logqdens = np.log(q)

    #find any points where samples from q were undefined on p
        
    kld = (logpdens-logqdens).sum()/len(logpdens)


    if verbose: print('kld is: {0:0.3g}'.format(kld))

    return kld

def kld_single_gridbase(dist0, dict1, bw=None, kernel = 'tophat', verbose=False, cube=False):
    """
    Returns the kld between the two sampled distributions dist0 and dist1.
    Explicitly: D_KL(dist0|uniform_dict), by grid based integration
    Uses a kernel density estimator to represent the probability distributions.

    Assumes that the shape of dist0 and dist1 is (nsamples, ndims)

    Optional arguments:
        bw: bandwidth for kernel density estimator.
            If "None", tries to estimate it based on the range of data in the two distributions.
        kernel: what shape of kernel to use (see list in sklearn.neighbors.KernelDensity)
        verbose: whether to print status updates
    """
    
    #if 1D arrays supplied, add an axis
    from sklearn.neighbors import KernelDensity
    if len(dist0)<2:
        return 0
    
    if verbose: print('creating kernel densities for the 1st distributions')

    estimated_bandwidth = (bw is None)
    if estimated_bandwidth:
        #compute the range where the probability density needs compact support in each dimension
        data_ranges = np.zeros((2, dist0.shape[1]))

        for ax in range(dist0.shape[-1]):
            data_ranges[0,ax] = np.array([np.min(dist0[:,ax])]).min()
            data_ranges[1,ax] = np.max(np.array([np.max(dist0[:,ax])]))

            #estimate the bandwidth to be comparable to even interparticle spacing over this range
        bw_est_0 = np.min((data_ranges[1]-data_ranges[0])/np.sqrt(float(dist0.shape[0])))*3.0

    if estimated_bandwidth: bw = bw_est_0
    if verbose: print('using bandwidth {0:0.3g} for dist 0'.format(max(bw,600)))
    
    bw = max(bw,600)
    
    kde0 = KernelDensity(bandwidth = bw,kernel=kernel).fit(dist0)

    if verbose: print('calculating kld')

    #the KDE returns the natural log of the normalized density
    num_intervals_integral = 101
    
    if cube:
        min_, max_ = 0, 0
        for key in dict1.keys():
            min_ = min(min_,dict1[key]['min'])
            max_ = max(max_,dict1[key]['max'])

        pts_1d = np.linspace(min_,max_,num_intervals_integral)
        dv = (pts_1d[1]-pts_1d[0])**3
        grid_hyper= np.stack(np.meshgrid(pts_1d, pts_1d, pts_1d), axis=-1)

        
    else:
        
        pts_1d_0 = np.linspace(dict1['J_r']['min'],dict1['J_r']['max'],num_intervals_integral)
        pts_1d_1 = np.linspace(dict1['J_z']['min'],dict1['J_z']['max'],num_intervals_integral)
        pts_1d_2 = np.linspace(dict1['J_phi']['min'],dict1['J_phi']['max'],num_intervals_integral)
        dv = (pts_1d_0[1]-pts_1d_0[0])*(pts_1d_1[1]-pts_1d_1[0])*(pts_1d_2[1]-pts_1d_2[0])
        grid_hyper= np.stack(np.meshgrid(pts_1d_0, pts_1d_1, pts_1d_2), axis=-1)

    
    
    grid_pts = np.reshape(grid_hyper,(grid_hyper.shape[0]**3,3))

    loggrid_dens = (kde0.score_samples(grid_pts))
    
    norm_check = dv*(np.exp(loggrid_dens)).sum()
    if verbose: print(f'Integral evaluates to : {norm_check}')
    
#     if np.round(norm_check,decimals = 0) != 1:
#         print('integration out of tolerance')
#         return 8 ##Returns a fixed KLD when things look bad.

    tol = 0.2
    if (norm_check <= 1 - tol) | (norm_check >= 1 + tol): 
        print('integration out of tolerance')
        return 8 ##Returns a fixed KLD when things look bad.
    
    q = 1
    for key in dict1.keys():
        q *= 1/(dict1[key]['max']-dict1[key]['min'])    
        
    if cube:
        q = 1/((max_ - min_)**3)
    
    logqdens = np.log(q)
    #find any points where samples from q were undefined on p
    
#     print(f'nan found :{np.isnan(loggrid_dens).sum()}')
    
    kld = dv*np.nansum(loggrid_dens*np.exp(loggrid_dens)) - logqdens

    if verbose: print('kld is: {0:0.3g}'.format(kld))
    return kld

def set_default_mpl_style():
    import matplotlib.pyplot as plt
    # Update rcParams to turn off LaTeX and revert to default Matplotlib values
    plt.rcParams.update({
        'text.usetex': False,  # Turn off LaTeX
        'font.family': 'sans-serif',  # Default font family
        'font.sans-serif': ['DejaVu Sans'],  # Default sans-serif font
        'mathtext.fontset': 'dejavusans',  # Default math font
    })
        
## Functions for traversing the halo catalogs and merger trees:

#------------------------------------
# The halo tracking functions
#------------------------------------

def find_main(halo_tree, last_snap=600, host_no=0):
    '''
    return merger tree indices of the main halo (at z=0) across all snapshots
    
    halo_tree = halo merger tree
    host_no = 0 for isolated sims; 0 and 1 for paried sims
    '''
    if host_no == 0:
        # what is the tree index at the last snapshot
        main_tid_ls = np.where((halo_tree['snapshot'] == last_snap) & 
                               (halo_tree.prop('host.distance.total') == 0))[0][0]
        
    else:
        main_tid_ls = np.where((halo_tree['snapshot'] == last_snap) & 
                               (halo_tree.prop('host2.distance.total') == 0))[0][0]
    
    main_tids = np.flip(halo_tree.prop('progenitor.main.indices', main_tid_ls))
    
    return main_tids



def find_hal_ind_backward(halo_tree, halo_tid, last_snap=600):
    '''
    find merger tree indices of the progenitor of subhalo halo_tid
    in all previous snapshots
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo
    '''
    
    prog_halo_tids = np.flip(halo_tree.prop('progenitor.main.indices', halo_tid))
    
    return prog_halo_tids
    
def find_hal_ind_forward(halo_tree, halo_tid, last_snap=600, host_no=0):
    '''
    find merger tree indices of the descendant of subhalo halo_tid in all
    subsequent snapshots until merging with the host
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo
    '''
    
    desc_halo_tids = np.flip(np.setdiff1d(halo_tree.prop('descendant.indices', halo_tid), 
                                          find_main(halo_tree, last_snap, host_no)))
    
    return desc_halo_tids



def find_hal_ind_all(halo_tree, halo_tid, last_snap=600, host_no=0):
    '''find merger tree indicies of both the descendant and progenitor 
    of the subhalo halo_tid
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo
    '''
    
    ind_progs = find_hal_ind_backward(halo_tree, halo_tid, last_snap)
    ind_descs = find_hal_ind_forward(halo_tree, halo_tid, last_snap, host_no)
    
    return np.append(ind_progs, ind_descs[1:])



def find_hal_index_at_snap(halo_tree, index_start, end_snap):
    '''
    return halo merger tree index at snapshot end_snap
    
    halo_tree = halo merger tree
    index_start = merger tree index of the subhalo
    end_snap = snapshot where we want to find the merger tree index
               of the subhalo
    '''
    
    start_snap = halo_tree['snapshot'][index_start]
    ind = index_start
    
    if end_snap == start_snap:
        return index_start
    elif end_snap < start_snap:
        while start_snap != end_snap:
            ind = halo_tree['progenitor.main.index'][ind]
            start_snap = halo_tree['snapshot'][ind]
        return ind
    else:
        while start_snap != end_snap:
            ind = halo_tree['descendant.index'][ind]
            start_snap = halo_tree['snapshot'][ind]
        return ind



def find_infall_snapshots(halo_tree, halo_tid, last_snap=600, host_no=0):
    '''
    find infall snapshots for the subhalo with merger tree index halo_tid
    
    halo_tree = halo merger tree
    halo_tid = merger tree index of the subhalo of interest
    
    return
    1) a list of infalling snapshots.
    empty list = does not fall into the host
    length 1 = one infall
    length greater than 1 = multiple infalls

    2) corresponding indices of the subhalo at infalling snaps
    '''

    main = find_main(halo_tree, last_snap=last_snap, host_no=host_no) # indices of the main halo
    main_r = halo_tree['radius'][main]                                # virial radii of the main halo
    # pad zeros to the front so that the array index corresponds to snapshot
    to_pad = last_snap + 1 - len(main)
    main_r = np.append(np.zeros(to_pad, dtype=int), main_r)
    
    # finding all indices of the subhalo
    halo_tids = find_hal_ind_all(halo_tree, halo_tid, last_snap=last_snap, host_no=host_no)
    
    # differences between central distance of the subhalo and virial radius of the host
    # at all snapshots where the subhalo exists
    if host_no == 0:
        diff = halo_tree.prop('host.distance.total')[halo_tids] - main_r[halo_tree['snapshot'][halo_tids]]
    else:
        diff = halo_tree.prop('host2.distance.total')[halo_tids] - main_r[halo_tree['snapshot'][halo_tids]]

    # looping over to find infall snapshots
    infall_snaps = []
    infall_snaps_i = []
    for i in range(1, len(diff), 1):
        if (diff[i] < 0) and (diff[i-1] > 0):
            infall_snaps.append(halo_tree['snapshot'][halo_tids[i]])
            infall_snaps_i.append(halo_tids[i])

    return infall_snaps, infall_snaps_i

def MWPotential22() -> agama.Potential:
    """
    Returns
    -------
        MWPotential22 fit. 
    """
    import agama 
    # length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
    agama.setUnits(mass=1,length=1,velocity=1)

    components = [
        # Disk components approximated by Miyamoto-Nagai profiles
        {
            'type': 'MiyamotoNagai',
            'mass': 7.8723069987e9,
            'scaleRadius': 1.5259431976529216,
            'scaleHeight': 0.20663742603550295
        },
        {
            'type': 'MiyamotoNagai',
            'mass': -2.756252219443315e11,
            'scaleRadius': 6.782764436261113,
            'scaleHeight': 0.20663742603550295
        },
        {
            'type': 'MiyamotoNagai',
            'mass': 3.206184188979487e11,
            'scaleRadius': 5.894799616164217,
            'scaleHeight': 0.20663742603550295
        },
        # Bulge (Hernquist profile: Dehnen with gamma=1)
        {
            'type': 'Dehnen',
            'mass': 5e9,
            'scaleRadius': 1.0,
        },
        # Nucleus (Hernquist profile)
        {
            'type': 'Dehnen',
            'mass': 1.8142e9,
            'scaleRadius': 0.0688867,
        },
        # NFW Halo (computed densityNorm to match enclosed mass at 5.3 scale radii)
        {
            'type': 'NFW',
            'mass': 5.5427e11, 
            'scaleRadius': 15.626
        }
    ]
    return agama.Potential(*components)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def fit_dehnen_profile(positions, masses, axis_y=1.0, axis_z=1.0, bins=50):
    """
    Fit a triaxial Dehnen profile by mapping to ellipsoidal radius.

    Parameters
    ----------
    positions : ndarray, shape (N,3)
        Particle positions.
    masses : ndarray, shape (N,)
        Particle masses.
    axis_y, axis_z : float
        Flattening ratios b/a and c/a.
    bins : int
        Number of radial bins.

    Returns
    -------
    M_fit : float
        Fitted total mass.
    a_fit : float
        Scale radius.
    gamma_fit : float
        Inner slope.
    r_centers : ndarray
        Radial bin centers.
    rho_vals : ndarray
        Measured density in each bin.
    """
    x, y, z = positions.T
    r_tilde = np.sqrt(x**2 + (y/axis_y)**2 + (z/axis_z)**2)

    rmin, rmax = np.percentile(r_tilde, [0.1, 99.9])
    edges = np.logspace(np.log10(rmin), np.log10(rmax), bins + 1)
    r_centers = np.sqrt(edges[:-1] * edges[1:])

    counts, _ = np.histogram(r_tilde, edges, weights=masses)
    volumes = 4/3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho_vals = counts / volumes

    log_rho = np.log10(np.maximum(rho_vals, 1e-12))

    def log_model(r, logM, loga, gamma):
        M = 10**logM
        a = 10**loga
        pref = (3 - gamma) / (4 * np.pi) * M / a**3
        return np.log10(pref * (r/a)**(-gamma) * (1 + r/a)**(gamma - 4))

    p0 = [np.log10(masses.sum()), np.log10(np.median(r_tilde)), 1.0]
    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 3])

    popt, _ = curve_fit(log_model, r_centers, log_rho, p0=p0, bounds=bounds)
    M_fit = 10**popt[0]
    a_fit = 10**popt[1]
    gamma_fit = popt[2]

    return M_fit, a_fit, gamma_fit, r_centers, rho_vals

def fit_plummer_profile(positions, masses, bins=30):
    
    """
    Fit a spherical Plummer profile to particle data.

    Returns
    -------
    M_fit : float
        Fitted total mass.
    b_fit : float
        Scale radius.
    r_centers : ndarray
        Radial bin centers.
    rho_vals : ndarray
        Measured density in each bin.
    """
    r = np.linalg.norm(positions, axis=1)
    rmin, rmax = np.percentile(r, [0.1, 99.9])
    edges = np.logspace(np.log10(rmin), np.log10(rmax), bins + 1)
    r_centers = np.sqrt(edges[:-1] * edges[1:])

    counts, _ = np.histogram(r, edges, weights=masses)
    volumes = 4/3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho_vals = counts / volumes

    log_rho = np.log10(np.maximum(rho_vals, 1e-12))

    def log_model(r, logM, logb):
        M = 10**logM
        b = 10**logb
        return np.log10(3 * M / (4 * np.pi * b**3) * (1 + (r/b)**2)**(-2.5))

    p0 = [np.log10(masses.sum()), np.log10(np.median(r))]
    popt, _ = curve_fit(log_model, r_centers, log_rho, p0=p0)

    M_fit = 10**popt[0]
    b_fit = 10**popt[1]

    return M_fit, b_fit, r_centers, rho_vals


def measure_anisotropy(positions, velocities, bins=20):
    """
    Compute velocity anisotropy β(r) in radial bins.

    Returns
    -------
    r_centers : ndarray
    beta_vals : ndarray
    """
    r = np.linalg.norm(positions, axis=1)
    vr = np.sum(positions * velocities, axis=1) / r
    vt2 = np.sum(velocities**2, axis=1) - vr**2

    edges = np.logspace(
        np.log10(np.percentile(r, 1e-3)),
        np.log10(np.percentile(r, 99.9)),
        bins + 1
    )
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    vr2, _ = np.histogram(r, edges, weights=vr**2)
    vt2b, _ = np.histogram(r, edges, weights=vt2)
    N, _ = np.histogram(r, edges)
    N[N == 0] = 1

    sigma_r2 = vr2 / N
    sigma_t2 = vt2b / N
    beta_vals = 1 - sigma_t2 / (2 * sigma_r2)

    return r_centers, beta_vals

def _align_and_get_axes(pos: np.ndarray, masses: np.ndarray, Rmax: None | float = None) -> tuple[np.ndarray, float, float]:
    """
    Compute principal axes via reduced inertia tensor within Rmax or half-mass aperture.
    Wrapper around the main iterative reduced inertia tensor method. Assumes data is precentered.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Centered particle positions.
    masses : ndarray, shape (N,)
        Particle masses.

    Returns
    -------
    evecs : ndarray, shape (3, 3)
        Rotation matrix (eigenvectors) to align with principal axes.
        Each row is one of the principal axes [e_a, e_b, e_c].
    axis_y : float
        Axis ratio b/a.
    axis_z : float
        Axis ratio c/a.
    """

    if Rmax is None:
        # Calculate radius and half-mass radius
        r = np.linalg.norm(pos, axis=1)
        sorted_idx = np.argsort(r)
        m_sorted = masses[sorted_idx]
        m_cum = np.cumsum(m_sorted)
        r_half = r[sorted_idx][np.searchsorted(m_cum, 0.5 * m_cum[-1])]
    
        # Use Rmax = 0.5 × half-mass radius
        Rmax = 0.5 * r_half

    abc_normalized, final_transform_matrix = compute_morphological_diagnostics(
        pos, mass=masses, Rmax=Rmax,
        reduced_structure=True, orient_with_momentum=False,
        return_ellip_triax=False
    )

    return final_transform_matrix, abc_normalized[1], abc_normalized[2]
