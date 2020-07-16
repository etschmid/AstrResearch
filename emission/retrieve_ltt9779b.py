# To multithread, run this with mpiexec -n 4 python retrieve_HST.py
#
# General plan: 
#   (1) Set up eclipse observations (real or simulated)
#   (2) Run this script to retrieve atmo. parameters
#   (3) Edit 'summarize.py' as necessary, and run it to extract some
#        useful outputs & summary plots.
#   (4) Interpret.
#   (5) ... profit!
#
# HISTORY:
# 2020/06/18 IJMC: commented a bit and sent to Ethen Schmidt

import numpy as np
import sys
import emcee
import pickle as pickle
import time
import analysis as an
import tools
import pymultinest

from petitRADTRANS import Radtrans
import master_retrieval_model as rm
from petitRADTRANS import nat_cst as nc
import rebin_give_width as rgw
from scipy.interpolate import interp1d
from astropy.io import fits
import json

sys.stdout.flush()

################################################################################
################################################################################
### Hyperparameter setup, where to save things to / read things from
###   as well as how many MultiNest 'live points' and what 'species' and
###   chemistry mode to use, and individual planet parameters.
################################################################################
################################################################################


retrieval_name = 'niriss_Test_%s/niriss_'

absolute_path = '' # end with forward slash!
observation_files = {}
#observation_files['IRAC'] = './observations/toi193_spitzer-tess_flux_v2.dat'
#observation_files['TESS'] = './observations/toi193_tess_flux_v3.dat'
#observation_files['nirspec'] = './observations/ltt9779_hih2o_nirspecG395M_noiseless.txt'
observation_files['niriss'] = './observations/ltt9779_hih2o_jwst_niriss-2eclipses_4pRT.csv'
#observation_files['nirspec'] = './observations/ltt9779_hih2o_nirspecG395M_1tran.txt'
#observation_files['HSTWFC3'] = 'toi193_wfc3g141-sim_hiH2O_10bin.dat'
#observation_files['HSTWFC3'] = 'toi193_wfc3g141-sim_weaker_10bin.dat'

#species = ['H2O',  'CO2','TiO', 'VO','Na','K']
species = ['H2O',  'CO2', 'CO_all_iso']
chemMode = 'free' # abundance of each molecule/atom is a free param.
#chemMode = 'selfconsistent' # use "poor Man's chemistry" file as free params; 
                            # only free params are [Fe/H] and C/O. 
                            
n_live_points = 25   # for MultiNest; larger is more complete, but takes longer.

resume = True

if chemMode=='free':
    #species = species
    retrieval_name = retrieval_name % ('-'.join(species))
elif chemMode=='selfconsistent':
    species = ['CO_all_iso', 'H2O', \
                                  'CH4', 'NH3', 'CO2', 'H2S', \
                                  'Na', 'K', 'PH3', 'VO', \
                                  'TiO']
    retrieval_name = retrieval_name % 'SC'

plotting = True #False
if plotting:
    import pylab as plt

# Wavelength range of observations, fixed parameters that will not be retrieved
#WLEN = [2.8, 6.0] #[0.54, 5.1]
WLEN = [0.8, 2.9]
WLEN_plot = [0.5, 12]
LOG_G =  3.11 # logg of PLANET, per Jenkins
R_pl =   4.72*nc.r_earth # per Jenkins
R_star = 0.949*nc.r_sun # per TIC v8
# Get host star spectrum to calculate F_pl / F_star later.
T_star = 5443. #5374. # per TIC v8
x = nc.get_PHOENIX_spec(T_star)
fstar = interp1d(x[:,0], x[:,1])


nspecies = len(species)

####################################################################################
####################################################################################
### READ IN OBSERVATION
####################################################################################
####################################################################################

# Read in data, convert all to cgs!

data_wlen = {}
data_flux_nu = {}
data_flux_nu_error = {}
data_wlen_bins = {}

for name in observation_files.keys():
    dat_obs = np.genfromtxt(observation_files[name])
    if dat_obs.ndim==1: dat_obs = np.array([dat_obs])
    data_wlen[name] = dat_obs[:,0]*1e-4
    if dat_obs.shape[1]>3:
        data_wlen_bins[name] = (dat_obs[:,2] - dat_obs[:,1])*1e-4
        data_flux_nu[name] = dat_obs[:,3]
        data_flux_nu_error[name] = dat_obs[:,4]
    else:
        data_flux_nu[name] = dat_obs[:,1]
        data_flux_nu_error[name] = dat_obs[:,2]
        data_wlen_bins[name] = np.zeros_like(data_wlen[name])
        data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
        data_wlen_bins[name][-1] = data_wlen_bins[name][-2]

wobs = np.concatenate([data_wlen[key] for key in observation_files.keys()])
obs  = np.concatenate([data_flux_nu[key] for key in observation_files.keys()])
eobs = np.concatenate([data_flux_nu_error[key] for key in observation_files.keys()])
        
####################################################################################
####################################################################################
### MODEL SETUP
####################################################################################
####################################################################################

### Create and setup radiative transfer object
# Create random P-T profile to create RT arrays of the Radtrans object.
temp_params = {}
temp_params['log_delta'] = -6.
temp_params['log_gamma'] = np.log10(0.4)
temp_params['t_int'] = 750.
temp_params['t_equ'] = 1500.
temp_params['log_p_trans'] = -3.
temp_params['alpha'] = 0.
#p, t = nc.make_press_temp(temp_params)

# Create the Ratrans object here
#rt_object = Radtrans(line_species=['H2']+species, \
#                    rayleigh_species=['H2','He'], \
#                    continuum_opacities = ['H2-H2','H2-He'], \
#                    mode='c-k', \
#                    wlen_bords_micron=WLEN)
pRTobjects = {}
for instrument in data_wlen_bins.keys():
    pRTobjects[instrument] = Radtrans(line_species=['H2']+species, \
                    rayleigh_species=['H2','He'], \
                    continuum_opacities = ['H2-H2','H2-He'], \
                    mode='c-k', \
                             wlen_bords_micron = WLEN)

rt_object_plot = Radtrans(line_species=['H2'] + species, \
                    rayleigh_species=['H2','He'], \
                    continuum_opacities = ['H2-H2','H2-He'], \
                    mode='c-k', \
                     wlen_bords_micron=WLEN_plot)

# Create the RT arrays of appropriate lengths
pressures = np.logspace(-6, 2, 90)
rt_object_plot.setup_opa_structure(pressures)
for name in pRTobjects.keys():
    pRTobjects[name].setup_opa_structure(pressures)

################################################################################
################################################################################
###   PRIOR SETUP (For all priors, neglect the constant that would be added 
###         because of the normalization of the normal distribution)
################################################################################
################################################################################

# def b_range(x, b):
#     if x > b:
#         return -np.inf
#     else:
#         return 0.

# def a_b_range(x, a, b):
#     if x < a:
#         return -np.inf
#     elif x > b:
#         return -np.inf
#     else:
#         return 0.

# log_priors = {}
# log_priors['log_delta']      = lambda x: -((x-(-5.5))/2.5)**2./2.                           
# log_priors['log_gamma']      = lambda x: -((x-(-0.0))/2.)**2./2. 
# log_priors['t_int']          = lambda x: a_b_range(x, 0., 1000.)
# log_priors['t_equ']          = lambda x: a_b_range(x, 1000., 3500.)
# log_priors['log_p_trans']    = lambda x: -((x-(-3))/3.)**2./2.
# log_priors['alpha']          = lambda x: -((x-0.25)/0.4)**2./2.
# log_priors['log_g']          = lambda x: a_b_range(x, 2.0, 3.7) 
# log_priors['log_P0']         = lambda x: a_b_range(x, -4, 2.)

# # Priors for log mass fractions
# for speciesname in species:
#     log_priors[speciesname]     = lambda x: a_b_range(x, -10., 0.)

# if 'K'  in log_priors: log_priors['K']  = lambda x: a_b_range(x, -10., -2)
# if 'Na' in log_priors: log_priors['Na'] = lambda x: a_b_range(x, -10., -2)

############
# Prior setup
############

def prior(cube, ndim, nparams):
    log_delta   = -10 + 10*cube[0] #
    log_gamma   = -6 + 12*cube[1]
    t_int       = 0 + 1000*cube[2]
    t_equ       = 1000 + 2500*cube[3]
    log_p_trans = -9 + 15*cube[4]
    alpha       = -1 + 2 * cube[5]
    log_P0      = -4 + 5 * cube[6]
    log_g       = LOG_G
    
    cube[0]  = log_delta
    cube[1]  = log_gamma
    cube[2]  = t_int
    cube[3]  = t_equ
    cube[4]  = log_p_trans
    cube[5]  = alpha
#    cube[6]  = log_g
    cube[6]  = log_P0

    if chemMode=='selfconsistent':
        metallicity = -1.5+4.5*cube[7]
        CtoO = 0.1+1.5*cube[8]

        cube[7]  = metallicity
        cube[8]  = CtoO
    elif chemMode=='free':
        for ii in range(nspecies):
            cube[7+ii] = -10 + 10*cube[7+ii]

    
    return

################################################################################
################################################################################
### DEFINE LOG PROBABILITY
################################################################################
################################################################################

def loglike(cube, ndim, nparams, plotting=False, retbinspec=False, retfullspec=False):

    parameters = {}
    parameters['log_delta'] = cube[0]
    parameters['log_gamma'] = cube[1]
    parameters['t_int'] = cube[2]
    parameters['t_equ'] = cube[3]
    parameters['log_p_trans'] = cube[4]
    parameters['alpha'] = cube[5]
    parameters['log_P0'] = cube[7]
    parameters['log_Pquench'] = -10
    parameters['gravity'] = 1e1**LOG_G
    parameters['R_pl'] = R_pl
    #parameters['P_reference'] = 1e1**cube[1]
    #parameters['Pcloud'] = 1e1**cube[2]
    parameters['Rstar'] = R_star
    parameters['species'] = species
    if chemMode=='selfconsistent':
        parameters['metallicity'] = cube[7]
        parameters['CtoO'] = cube[8]
    elif chemMode=='free':
        for ii,speciesname in enumerate(species):
            parameters[speciesname] = cube[7+ii]

    log_likelihood = 0.
    for instrument in data_wlen.keys():
        # Calculate the forward model, this returns the wavelengths in
        # cm and the flux F_nu in erg/cm^2/s/Hz
        wlen, flux_nu = rm.retrieval_model_plain(pRTobjects[instrument], parameters)
        # Convert to observation for emission case
        flux_star = fstar(wlen)
        flux_sq   = (flux_nu/flux_star)*(R_pl/R_star)**2 
        
        # Or for transmission:
        #wlen_micron, transit_radius_cm = \
        #    rm.return_rad(pRTobjects[instrument], parameters)
        #r_planet_over_r_star = transit_radius_cm**2./parameters['Rstar']**2.

        obs = data_flux_nu[instrument]
        obs_err = data_flux_nu_error[instrument]

        if plotting:
            #plt.plot(wlen_micron, r_planet_over_r_star)
            print()


        flux_rebinned = rgw.rebin_give_width(wlen, \
                               flux_sq, \
                               data_wlen[instrument], \
                               data_wlen_bins[instrument])

        diff = (flux_rebinned - obs)
        
        log_likelihood += -np.sum((diff / obs_err)**2.)/2.

        if plotting:
            print()
            #plt.errorbar(data_wlen[instrument], \
                             #obs, \
                             #yerr = obs_err, \
                             #fmt = '+')
                             
            #plt.plot(data_wlen[instrument], rplrstar_rebinned, '.-')

    if plotting:
        print()
        #plt.show()

    print('log(L) = ', log_likelihood)
    ret = log_likelihood
    if retbinspec:
        ret = data_wlen[instrument], flux_rebinned
    elif retfullspec:
        ret = wlen, flux_sq
    return ret


if chemMode=='selfconsistent':
    n_params = 9
elif chemMode=='free':
    n_params = 7 + nspecies

cube = [.5]*n_params
prior(cube, 0,n_params)
loglike(cube, 0, n_params)

pymultinest.run(loglike, \
                    prior, \
                    n_params, \
                    outputfiles_basename=retrieval_name, \
                    resume = resume, \
                    verbose = True, \
                    n_live_points = n_live_points)

print('Well-hey-we-got-here----------------------------------------'*100)

parameter_names = ['log_delta', 'log_gamma', 't_int', 't_equ', 'log_p_trans', 'alpha', 'log_p0']
if chemMode=='selfconsistent':
    parameter_names += ['[Fe/H]', 'C/O']
elif chemMode=='free':
    parameter_names += list(species)

json.dump(parameter_names, open('%sparams.json' % retrieval_name, 'w'))
print('Finished with run %s' % retrieval_name)

stop
################################################################################
################################################################################
### Done.
################################################################################
################################################################################



parameter_names = {0: r"$\rm log(delta)$", \
              1: r"$\rm log(gamma)$", \
              2: r"$\rm T_{int}$", \
              3: r"$\rm T_{equ}$", \
              4: r"$\rm log(P_{tr})$", \
              5: r"$\rm alpha$", \
              6: r"$\rm log(g)$", \
              7: r"$\rm log(P_0)$"}
for ii,speciesname in enumerate(species):
    parameter_names[ii+8] = r"$\rm %s$" % speciesname


mcspec = np.array([calc_log_prob(sampler.flatchain[int(np.floor(sampler.flatchain.shape[0]/100))*ii], retconspec=True)[1] for ii in range(100)])
www, bestspec = calc_log_prob(best_position, retconspec=True)
www2, bestspec2 = calc_log_prob(best_position, retconspec=True, rt_object=rt_object_plot)
wbin, bestbin = calc_log_prob(best_position, retconbin=True, rt_object=rt_object_plot)
smcs = np.sort(mcspec, axis=0)


#wobs, obs, eobs = dat_obs.T

lt = [.8, 1, 1.7, 3, 4, 5, 10]

plt.figure()
plt.fill_between(www*1e4, smcs[16]*1e6, 1e6*smcs[84], color='c', alpha=0.5)
plt.plot(an.binarray(www2*1e4, 10)/10., an.binarray(bestspec2, 10)/10.*1e6, '-k', linewidth=2, alpha=0.5)
plt.plot(wbin*1e4, np.array(bestbin)*1e6, 'sk', mfc='gray')
plt.errorbar(wobs, obs*1e6, eobs*1e6, fmt='or', elinewidth=2, mew=2, ms=7)
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet/star contrast')
plt.xticks(lt,lt)
#plt.xlim(.7, 6)

from PT_envelopes import return_PT_envelopes
#samples_path = 'chain_pos_' + retrieval_name + '.pickle'
#f = open(samples_path,'rb')
#pos = pickle.load(f)
#prob = pickle.load(f)
#state = pickle.load(f)
#samples = pickle.load(f)
#f.close()
#chains = fits.getdata('chain_complete_' + retrieval_name + '.fits')
#lnprob = fits.getdata('chain_lnprob_' + retrieval_name + '.fits')

envelope_file = 'T_enve.pickle'
N_samples = 7000
#true_values = np.genfromtxt('input_params.dat')
envelopes = return_PT_envelopes(samples, \
                    envelope_file, \
                    N_samples = N_samples, \
                                read = False, true_values=best_position)

plt.figure()
plt.gcf().set_size_inches(10., 6.)
plt.gca().minorticks_on()
plt.gca().tick_params(axis='y',which='major',length=6.,width=1)
plt.gca().tick_params(axis='y',which='minor',length=3.,width=1)
plt.gca().tick_params(axis='x',which='major',length=6.,width=1)
plt.gca().tick_params(axis='x',which='minor',length=3.,width=1)

#plt.fill_betweenx(envelopes[0][0], envelopes[2][0], envelopes[3][0], \
#                      facecolor='lightgrey', label='5%$-$95%', zorder=-10)
plt.fill_betweenx(envelopes[0][0], envelopes[2][1], envelopes[3][1], \
                      facecolor='skyblue', label='68% interval', zorder=-10)
#plt.fill_betweenx(envelopes[0][0], envelopes[2][2], envelopes[3][2], \
#                      facecolor='dodgerblue', label='25%$-$75%', zorder=-10)
#plt.fill_betweenx(envelopes[0][0], envelopes[2][3], envelopes[3][3], \
#                      facecolor='b', label='35%$-$65%', zorder=-10)

plt.plot(envelopes[0][1], envelopes[0][0], color='red', linewidth=2, \
             zorder=10, label='Best-fit profile')

plt.xlabel('Temperature (K)', fontsize = 18, color = 'k')
plt.ylabel('Pressure (bar)', fontsize = 18, color = 'k')
plt.ylim([1e3,1e-6])
plt.xlim([500.,3000.])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yscale('log')
plt.legend(loc='best',frameon=False, fontsize=13)
plt.tight_layout()


tools.hparams(sampler.flatchain, 100, labs=parameter_names)


tools.printfigs('chain_pos_' + retrieval_name + '.pdf', pdfmode='gs')


stop

guess0 = [np.log10(val) for val in [0.01, 0.001, 1e-7, 1e-7, 1e-5, .74, 1e-5, 1e-6]]
guess = [-5.48, log10(.1), 200, 2800, -1, 0, 2.99, 0] + guess0
