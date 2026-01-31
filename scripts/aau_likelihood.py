#Likelihood
from scipy import stats
import numpy as np
from scipy.interpolate import make_splrep, interp1d
from stream_sim_funcs import create_stream_particle_spray, generate_stream_coords
from scipy.stats import binned_statistic
from astropy.coordinates import Galactocentric, ICRS
import astropy.units as u

from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import Galactocentric, ICRS, CartesianDifferential, CartesianRepresentation
from astropy import table
#from utils.coordinates_jet import icrs_to_jet, jet_to_icrs, get_phi12_from_stream, phi1_to_dist_jet, observed_to_simcart

def make_spline_old(x, y):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]    
    m = len(x_sorted)
    spline = make_splrep(x_sorted, y_sorted, k=3, s=(m - np.sqrt(2 * m)))
    return spline

def make_spline(x, y, binsize = 0.4):
    """
    Compute a 1D spline interpolation of binned data.

    This function sorts the input data by `x`, bins the data using a fixed `binsize`,
    computes the mean of `y` in each bin, and returns a spline function that
    interpolates these binned means.

    Parameters
    ----------
    x : array-like
        Independent variable values. Likely should be phi1
    y : array-like
        Dependent variable values corresponding to `x`.
    binsize : float, optional
        Width of bins used to group the data (default is 0.1).

    Returns
    -------
    spline : function
        A 1D interpolating function (`scipy.interpolate.interp1d`) that maps
        x-values to the mean binned y-values. Returns NaN for values outside the domain.
    
    Notes
    -----
    - Uses `scipy.stats.binned_statistic` to bin and average `y` values.
    - The resulting spline uses linear interpolation and does not extrapolate beyond data range.
    """
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices] 
    #binned statistic function
    #arrange with bin size, 0.1 in phi1
    #make a bunch of bins and statistics
    try:
        bins = np.arange(x_sorted.min(), x_sorted.max()+binsize, binsize)
    except:
        import pdb; pdb.set_trace()
    y_median, bin_edges, binnumber = binned_statistic(x_sorted, y_sorted, statistic='mean', bins=bins)
    bin_centers = (bins[:-1] + bins[1:])/2
    m = len(bin_centers)
    #spline = make_splrep(bin_centers, y_median, k=3, s=(m - np.sqrt(2 * m)))
    #spline = interp1d(bin_centers, y_median, bounds_error = False, fill_value = np.nan)
    spline = interp1d(bin_centers, y_median, bounds_error = False, fill_value = "extrapolate", kind="linear")
    return spline


def log_likelihood(
        prog_pars, 
        phi1_obs, 
        phi2_obs, 
        rv_obs, 
        rv_obs_errors, 
        dist_obs,
        dist_obs_errors,
        pmra_cosdec_obs, 
        pmra_cosdec_obs_errors, 
        pmdec_obs, 
        pmdec_obs_errors,
        pot,
        phi1_range=[-20,15],
        seed_num=69420,
):
    """
    Compute the log likelihood of the data given the model parameters.
    prog_pars: list of parameters for the stream progenitor
    phi1_obs: observed phi1 values
    phi2_obs: observed phi2 values
    rv_obs: observed radial velocities
    rv_obs_errors: errors on the radial velocities
    pmra_cosdec_obs: observed pmra values
    pmra_cosdec_obs_errors: errors on the pmra values
    pmdec_obs: observed pmdec values
    pmdec_obs_errors: errors on the pmdec values
    pot: potential object
    phi1_range: range of phi1 values to consider
    seed_num: seed number for the random number generator

    """
    #Deciding stream IC's

    #from MCMC
    prog_coords_today = prog_pars
    
    ra, dec, dist, pmra, pmdec, rv = prog_coords_today
    
    aau_c = coord.SkyCoord(
        ra=ra*u.degree, dec=dec*u.degree, distance=dist*u.kpc, 
        pm_ra_cosdec=pmra*u.mas/u.yr,
        pm_dec=pmdec*u.mas/u.yr,
        radial_velocity=rv*u.km/u.s
    )
    
    rep = aau_c.transform_to(coord.Galactocentric) # units here are kpc, km/s
    
    prog_wtoday = np.array(
        [rep.x.value, rep.y.value, rep.z.value,
         rep.v_x.value, rep.v_y.value, rep.v_z.value]
    ) # units here are kpc, km/s
    
    
    # # stream progenitor profile parameters
    prog_mass, prog_scaleradius =  20_000, 10/1_000 # Msun, kpc
    Age_stream_inGyr = 3 # Gyr --<
    
    # # num_particles for the spray model: 
    num_particles = 1_000 # # preferably a multiple of 2, leading+trailing arm
    
    # simulate a stream
    stream_unperturb = create_stream_particle_spray(pot_host=pot, 
    initmass=prog_mass, 
    scaleradius=prog_scaleradius, 
    prog_pot_kind='Plummer', 
    sat_cen_present=prog_wtoday, 
    num_particles=num_particles,
    time_end=0.0, 
    time_total=Age_stream_inGyr, save_rate=1,)


    #get it in correct coordinate frame
    phi1_model, phi2_model = generate_stream_coords(stream_unperturb['part_xv'], xv_prog=prog_wtoday, optimizer_fit=False,)
    
    # select only points in the phi1 range
    phi1_model_sel = (phi1_model > phi1_range[0]) & (phi1_model < phi1_range[1]) 
    phi1_obs_sel = (phi1_obs > phi1_range[0]) & (phi1_obs < phi1_range[1]) 
    if phi1_model_sel.sum() == 0:
        import pdb; pdb.set_trace()
    #phi1_model_sel = (phi1_model > phi1_range[0]) & (phi1_model < phi1_range[1]) 
    #Prior to the Prior
    if phi1_model.min() > phi1_range[0] or phi1_model.max() < phi1_range[1]:
        print("Bad model >:(")
        return -np.inf
    #check that the min phi1 model vals are >=min of phi1 observed ditto for max
    #phi1 from -15 to 15
    #if not return np.inf --> 
    #### on-sky track
    # generate a track spline
    phi2_spline = make_spline(phi1_model[phi1_model_sel], phi2_model[phi1_model_sel])
    # compute model scatter around track since there are no position errors
    phi2_std = np.nanstd(phi2_model[phi1_model_sel] - phi2_spline(phi1_model[phi1_model_sel]))
    phi2_vals = phi2_spline(phi1_obs[phi1_obs_sel])
    
    # compute likelihood
    lnlk_spatial = stats.norm.logpdf(phi2_obs[phi1_obs_sel], loc = phi2_vals, scale = np.sqrt(phi2_std**2+0.24**2))
   
    #### velocity track
    # generate a velocity spline
    rv_spline = make_spline(phi1_model[phi1_model_sel], rv_model[phi1_model_sel])
    rv_vals = rv_spline(phi1_obs[phi1_obs_sel])
    # compute likelihood
    lnlk_velocity = stats.norm.logpdf(rv_obs[phi1_obs_sel], loc = rv_vals, scale = np.sqrt((rv_obs_errors[phi1_obs_sel])**2+4.3**2))

    ### pmra track
    pmra_cosdec_spline = make_spline(phi1_model[phi1_model_sel], pmra_cosdec_model[phi1_model_sel])
    pmra_cosdec_vals = pmra_cosdec_spline(phi1_obs[phi1_obs_sel])
    lnlk_pmra_cosdec = stats.norm.logpdf(pmra_cosdec_obs[phi1_obs_sel], loc = pmra_cosdec_vals, scale = np.sqrt((pmra_cosdec_obs_errors[phi1_obs_sel])**2 +0.55**2 ))

    ### pmdec track
    pmdec_spline = make_spline(phi1_model[phi1_model_sel], pmdec_model[phi1_model_sel])
    pmdec_vals = pmdec_spline(phi1_obs[phi1_obs_sel])
    lnlk_pmdec = stats.norm.logpdf(pmdec_obs[phi1_obs_sel], loc = pmdec_vals, scale = np.sqrt((pmdec_obs_errors[phi1_obs_sel])**2 +0.55**2 ))

    lnlk_total = lnlk_spatial + lnlk_velocity + lnlk_pmra_cosdec + lnlk_pmdec
    #if np.isnan = True --> throw those out and find a new value
    #if np.isnan(np.sum(lnlk_total)):
        #import pdb; pdb.set_trace()
    
    ## dist
    dist_spline = make_spline(phi1_model[phi1_model_sel], dist_model[phi1_model_sel])
    dist_vals = dist_spline(phi1_obs[phi1_obs_sel])
    #print(dist_vals.shape)
    lnlk_dist = stats.norm.logpdf(dist_obs[phi1_obs_sel], loc = dist_vals, scale = dist_obs_errors[phi1_obs_sel])
    
    lnlk_total = lnlk_spatial + lnlk_velocity + lnlk_pmra_cosdec + lnlk_pmdec + lnlk_dist

    
    return np.sum(lnlk_total)


def log_prior(prog_pars): #specify some reasonable bounds with a prior function; that way we don't have to brute force a bunch of stuff
    ra,dec,dist, pm_ra, pm_dec,rv = prog_pars
    if 0 < ra < 100 and -30.0 < dec < -9.0 and 0.5 <dist < 40 and -2 < pm_ra < 2 and -2.50 < pm_dec < 1.0 and -100.0 < rv < 200.00:
        return 0.0
    return -np.inf

def log_probability(prog_pars,data_dict, pot):
    try:
        lp = log_prior(prog_pars)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(prog_pars,**data_dict,pot=pot)
    except Exception as e:
        print("Exception in log_probability:", e)
        print("prog_pars:", prog_pars)
        return -np.inf  # So MCMC doesn’t crash