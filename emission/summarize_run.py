import numpy as np

from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import master_retrieval_model as rm
import rebin_give_width as rgw
import pymultinest
import json
from scipy.interpolate import interp1d
from astropy.io import fits

import tools
import analysis as an
import mmr2vmr

plotting = True
save_obs = False

if plotting:
    import pylab as plt
observation_files = {}

nmc = 100

#LTT9779b:  
observation_files['nirspec'] = './observations/ltt9779_hih2o_nirspecG395M_noiseless.txt'
#observation_files['IRAC'] = 'observations/toi193_spitzer-tess_flux_v2.dat'
#observation_files['TESS'] = './observations/toi193_tess_flux_v3.dat'
runname_base = 'nirspec_250_LTT9779b_'
instrument_type = 'nirspec'




LOG_G =  3.11 
R_pl =   4.72*nc.r_earth # per Jenkins
R_star = 0.949*nc.r_sun # per TIC v8
T_star = 5443.
chemMode = 'selfconsistent' #'selfconsistent' # free
species = ['H2O',  'CO2','TiO', 'VO','Na','K'] #['CO_all_iso']

if chemMode=='free':
    print('free chem')
    runname = runname_base + '-'.join(species)
elif chemMode=='selfconsistent':
    species = ['CO_all_iso', 'H2O', \
                                  'CH4', 'NH3', 'CO2', 'H2S', \
                                  'Na', 'K', 'PH3', 'VO', \
                                  'TiO']
    runname = runname_base + 'SC'

x = nc.get_PHOENIX_spec(T_star)
fstar = interp1d(x[:,0], x[:,1])

wlen_bords_micron_data_take = [3, 5.1]# 1.065, 1.65]
wlen_bords_micron = [0.5, 12] #1.0, 1.75]

nspecies = len(species)

############
# Read data
############


data_wlen = {}
data_rplrstar = {}
data_rplrstar_error = {}
data_wlen_bins = {}
for name in observation_files.keys():
    dat_obs = np.genfromtxt(observation_files[name])
    if dat_obs.ndim==1: dat_obs = np.array([dat_obs])
    data_wlen[name] = dat_obs[:,0]*1e-4
    if dat_obs.shape[1]>3:
        data_wlen_bins[name] = (dat_obs[:,2] - dat_obs[:,1])*1e-4
        data_rplrstar[name] = dat_obs[:,3]
        data_rplrstar_error[name] = dat_obs[:,4]
    else:
        data_rplrstar[name] = dat_obs[:,1]
        data_rplrstar_error[name] = dat_obs[:,2]
        data_wlen_bins[name] = np.zeros_like(data_wlen[name])
        data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
        data_wlen_bins[name][-1] = data_wlen_bins[name][-2]


#for name in observation_files.keys():
#
#    dat_obs = np.genfromtxt(observation_files[name])
#    data_wlen[name] = dat_obs[:,0]
#    if dat_obs.shape[1]>3:
#        data_wlen_bins[name] = (dat_obs[:,2] - dat_obs[:,1])*1e-4
#        data_rplrstar[name] = dat_obs[:,3]
#        data_rplrstar_error[name] = dat_obs[:,4]
#    else:
#        data_rplrstar[name] = dat_obs[:,1]
#        data_rplrstar_error[name] = dat_obs[:,2]
#        data_wlen_bins[name] = np.zeros_like(data_wlen[name])
#        data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
#        data_wlen_bins[name][-1] = data_wlen_bins[name][-2]
#
#    
#    #data_wlen_bins[name] = np.zeros_like(data_wlen[name])
#   # data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
#  #  data_wlen_bins[name][-1] = data_wlen_bins[name][-2]
#
#    index = (data_wlen[name] >= wlen_bords_micron_data_take[0]) & \
#      (data_wlen[name] <= wlen_bords_micron_data_take[1])
#
#    data_wlen[name] = data_wlen[name][index]
#    data_rplrstar[name] = data_rplrstar[name][index]
#    data_rplrstar_error[name] = data_rplrstar_error[name][index]
#    data_wlen_bins[name] = data_wlen_bins[name][index]

wobs = np.concatenate([data_wlen[key] for key in observation_files.keys()])
obs  = np.concatenate([data_rplrstar[key] for key in observation_files.keys()])
eobs = np.concatenate([data_rplrstar_error[key] for key in observation_files.keys()])

############
# Make pRT object
############


pRTobject = Radtrans(line_species=species, \
                    rayleigh_species=['H2','He'], \
                    continuum_opacities = ['H2-H2','H2-He'], \
                    mode='c-k', \
                    wlen_bords_micron = wlen_bords_micron)

pressures = np.logspace(-6, 2, 90)
pRTobject.setup_opa_structure(pressures)


############
# Prior setup
############

def prior(cube, ndim, nparams):
    temp = TirrRange[0] + (TirrRange[1]-TirrRange[0])*cube[0]
    P_reference = -6.+8.*cube[1]
    P_cloud = -6.+8.*cube[2]
    
    cube[0]  = temp
    cube[1]  = P_reference
    cube[2]  = P_cloud

    if chemMode=='selfconsistent':
        metallicity = -1.5+4.5*cube[3]
        CtoO = 0.1+1.5*cube[4]

        cube[3]  = metallicity
        cube[4]  = CtoO
    elif chemMode=='free':
        for ii in range(nspecies):
            cube[3+ii] = -10 + 10*cube[3+ii]

    
    return

############
# Define log-likelihood
############

def loglike(cube, ndim, nparams, plotting=False, retbinspec=False, retfullspec=False, contribution=False):

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
    if chemMode=='selfconsistent':
        parameters['species'] = species
        parameters['metallicity'] = cube[7]
        parameters['CtoO'] = cube[8]
    elif chemMode=='free':
        parameters['species'] = species
        for ii,speciesname in enumerate(species):
            parameters[speciesname] = cube[7+ii]

    if contribution: # Contribution functions!
        objs = []
        for instrument in data_wlen.keys():
            objs.append(rm.retrieval_model_plain(pRTobject, parameters, contribution=contribution))
        return(objs)
            
    log_likelihood = 0.
    for instrument in data_wlen.keys():
        # Calculate the forward model, this returns the wavelengths in
        # cm and the flux F_nu in erg/cm^2/s/Hz
        wlen, flux_nu = rm.retrieval_model_plain(pRTobject, parameters)
        # Convert to observation for emission case
        flux_star = fstar(wlen)
        flux_sq   = (flux_nu/flux_star)*(R_pl/R_star)**2 
        #wlen *= 1e4 
        # Or for transmission:
        #wlen_micron, transit_radius_cm = \
        #    rm.return_rad(pRTobjects[instrument], parameters)
        #r_planet_over_r_star = transit_radius_cm**2./parameters['Rstar']**2.

        obs = data_rplrstar[instrument]
        obs_err = data_rplrstar_error[instrument]

        if plotting:
            #plt.plot(wlen_micron, r_planet_over_r_star)
            print()

        #print(wlen.mean(), data_wlen[instrument].mean())
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


# def prior(cube, ndim, nparams):

#     temp = 500.+1000.*cube[0]
#     P_reference = -6.+8.*cube[1]
#     P_cloud = -6.+8.*cube[2]
#     metallicity = -1.5+3.3*cube[3]
#     CtoO = 0.1+1.5*cube[4]

#     cube[0]  = temp
#     cube[1]  = P_reference
#     cube[2]  = P_cloud
#     cube[3]  = metallicity
#     cube[4]  = CtoO
    
#     return

# ############
# # Define log-likelihood
# ############

# def loglike(cube, ndim, nparams):

#     parameters = {}
#     parameters['CtoO'] = cube[4]
#     parameters['metallicity'] = cube[3]
#     parameters['temperature'] = cube[0]
#     parameters['log_Pquench'] = -10
#     parameters['gravity'] = 1e1**LOG_G
#     parameters['R_pl'] = R_pl
#     parameters['P_reference'] = 1e1**cube[1]
#     parameters['Pcloud'] = 1e1**cube[2]
#     parameters['Rstar'] = R_star

#     wlen_micron, transit_radius_cm = rm.return_rad(pRTobject, parameters)
#     r_planet_over_r_star = transit_radius_cm**2./parameters['Rstar']**2.

#     if plotting:
#         plt.plot(wlen_micron, r_planet_over_r_star, color = 'teal', alpha = 0.1, zorder = 0)

#     log_likelihood = 0.
#     for instrument in data_wlen.keys():

#         rplrstar_rebinned = rgw.rebin_give_width(wlen_micron, \
#                                r_planet_over_r_star, \
#                                data_wlen[instrument], \
#                                data_wlen_bins[instrument])

#         diff = (rplrstar_rebinned - data_rplrstar[instrument])

#         if save_obs:
#             np.savetxt('observations/test_obs.dat', \
#                        np.column_stack((data_wlen[instrument], \
#                                         rplrstar_rebinned, \
#                                         data_rplrstar_error[instrument])))
        
#         log_likelihood += -np.sum((diff/ \
#                 data_rplrstar_error[instrument])**2.)/2.

#         if plotting:

#             plt.errorbar(data_wlen[instrument], \
#                              data_rplrstar[instrument], \
#                              yerr = data_rplrstar_error[instrument], \
#                              fmt = '+', color='orange', zorder=1)


#     print('log(L) = ', log_likelihood)
#     return log_likelihood

def process_tp_from_samples(samps):
    temps = []
    for ii in range(len(samps)):
        temp_params = {}
        temp_params['log_delta'] = samps[ii,0]
        temp_params['log_gamma'] = samps[ii,1]
        temp_params['t_int'] = samps[ii,2]
        temp_params['t_equ'] = samps[ii,3]
        temp_params['log_p_trans'] =samps[ii,4]
        temp_params['alpha'] = samps[ii,5]
        p, t = nc.make_press_temp(temp_params)
        temps.append(t)

    temps = np.array(temps)
    sorttemps = np.sort(temps, axis=0)
    medtemps = np.median(temps, axis=0)
    onesiglo = sorttemps[int(.16*len(samps))]
    onesighi = sorttemps[int(.84*len(samps))]
    return p, medtemps, onesiglo, onesighi, temps 

import os
os.system('ls %s/*_post_equal_weights.dat > tmp' % runname )
f = open('tmp')
path = f.readlines()[0][:-1]
f.close()
os.system('rm tmp')
samples = np.genfromtxt(path)

best_ind = np.argmax(samples[:,-1])
best = samples[best_ind,:-1]
print('best log-like', samples[best_ind,-1])
print('Parameters, ', best)

w, spec = loglike(best, 0., 0., retfullspec=True)
binspec = np.concatenate([rgw.rebin_give_width(w, spec, data_wlen[instrument], data_wlen_bins[instrument]) for instrument in data_wlen.keys()])

mc_spec = np.zeros((nmc, spec.size), dtype=float)
mc_binspec = np.zeros((nmc, wobs.size), dtype=float)
for i in range(nmc):
    index = int(np.random.uniform()*len(samples))
    while (samples[index, -1]*-2)>(3*wobs.size): 
        index = int(np.random.uniform()*len(samples))
    sample = samples[index,:-1]
    print(i, ' T', sample[0])
    mc_spec[i] = loglike(sample, 0., 0., retfullspec=True)[1]
    mc_binspec[i] = np.concatenate([rgw.rebin_give_width(w, mc_spec[i], data_wlen[instrument], data_wlen_bins[instrument]) for instrument in data_wlen.keys()])


best_chisq = (((binspec - obs) / eobs)**2).sum()
mc_chisq = (((mc_binspec - obs) / eobs)**2).sum(1)
mc_spec0 = mc_spec.copy()
mc_spec.sort(axis=0)
all_chisq = np.concatenate(([best_chisq], mc_chisq))
all_prob = np.exp(-0.5*(all_chisq-all_chisq.min()))



all_spec = np.vstack((spec, mc_spec0))

weightedspec = an.wmean(all_spec, w=np.tile(all_prob.reshape(nmc+1, 1), (1, w.size)), axis=0).ravel()
weighted_binspec = np.concatenate([rgw.rebin_give_width(w, weightedspec, data_wlen[instrument], data_wlen_bins[instrument]) for instrument in data_wlen.keys()])
  


lo_conf_spec = mc_spec[int(.16*nmc)]
hi_conf_spec = mc_spec[int(.84*nmc)]
xx = np.concatenate((w, w[::-1]))
yy = np.concatenate((lo_conf_spec, hi_conf_spec[::-1]))

plt.figure()
tools.drawPolygon(np.vstack((1e4*xx,1e6*yy)).T, alpha=0.5)
#plt.plot(w*1e4, np.median(mc_spec, axis=0)*1e6, '-r', linewidth=1.5)
plt.plot(w*1e4, weightedspec*1e6, '-r', linewidth=1.5)
plt.plot(wobs*1e4, weighted_binspec*1e6, 'sr', ms=13)
for inst in observation_files.keys():
    plt.errorbar(data_wlen[inst]*1e4, 1e6*data_rplrstar[inst], 1e6*data_rplrstar_error[inst], fmt='ok', mew=2, elinewidth=2)

plt.xlabel('Wavelength [um]', fontsize=14)
plt.ylabel('Eclipse Depth [ppm]', fontsize=14)
plt.title('%s\n%s -- $\chi^2 = %1.1f$' % (runname, '-'.join(species), best_chisq))
plt.minorticks_on()
plt.ylim(0, 1000) #1e6*np.nanmax(obs+2*eobs))
plt.savefig('./%s/%s_spectrum_summary_plot.pdf' % (runname,instrument_type))

fits.writeto('./%s/%s_bestfit_spectrum.fits' % (runname,instrument_type), np.vstack((w, spec)), overwrite=True)
fits.writeto('./%s/%s_MCspectra_sorted.fits' % (runname,instrument_type), np.vstack((w, mc_spec)), overwrite=True)
fits.writeto('./%s/%s_MCspectra_unsorted.fits' % (runname,instrument_type), np.vstack((w, mc_spec0)), overwrite=True)

pp,tt, ttlo, tthi, temps = process_tp_from_samples(samples)

plt.figure()
plt.semilogy(tt, pp, 'r-', linewidth=2)
plt.fill_betweenx(pp, ttlo, tthi, alpha=0.5, color='r')
plt.ylim(plt.ylim()[::-1])
plt.xlabel('Temperature [K]', fontsize=16)
plt.ylabel('Pressure [bars]', fontsize=16)
#plt.show()


fig2=plt.figure()
ax3=plt.subplot(211, position=[.12, .1, .5, .8])
ax4=plt.subplot(212, position=[.64, .1, .3, .8])
ax3.semilogy(tt, pp, 'r-', linewidth=2)
ax3.fill_betweenx(pp, ttlo, tthi, alpha=0.5, color='r')
#ax2.set_ylim(plt.ylim()[::-1])
ax3.set_xlabel('Temperature [K]', fontsize=16)
ax3.set_ylabel('Pressure [bars]', fontsize=16)

objs4contribution = loglike(best, 0., 0., contribution=True)
fig1=plt.figure()
ax1=plt.subplot(111)
ax1.set_xlabel('Wavelength (microns)')
[ax.set_xlabel('Relative Contribution', fontsize=14) for ax in [ax4]]
[ax.set_ylabel('P (bar)') for ax in [ax1]]
[ax.set_title('Emission contribution function') for ax in [ax1]]
[ax.set_yscale('log') for ax in [ax1, ax3, ax4]]
ax1.set_xscale('log')
[ax.set_ylim([1e2,1e-6]) for ax in [ax1, ax3, ax4]]


for atmosphere in [objs4contribution[0]]:
    wlen_mu = nc.c/atmosphere.freq/1e-4
    X, Y = np.meshgrid(wlen_mu, pressures)
    ax1.contourf(X,Y,atmosphere.contr_em,30,cmap=plt.cm.bone_r)
    ax1.set_xlim([np.min(wlen_mu),np.max(wlen_mu)])
    for ii, instrument in enumerate(data_wlen.keys()):
        bandcenters = data_wlen[instrument]
        bandwidths = data_wlen_bins[instrument]
        for jj in range(len(bandcenters)):
            waveind = np.logical_and((wlen_mu/1e4) > (bandcenters - bandwidths/2.)[jj],
                                     (wlen_mu/1e4) <= (bandcenters + bandwidths/2.)[jj])
            contribution_fcn = atmosphere.contr_em[:,waveind].mean(1)
            ax4.semilogy(contribution_fcn/np.nanmax(contribution_fcn), pressures, label='%s (%1.1f um)' % (instrument, bandcenters[jj]*1e4), linewidth=1.5)
            
leg4=ax4.legend(fontsize=12, loc='upper center')
tools.legc(leg4)
ax4.set_yticklabels([])
fig1.savefig('./%s/%s_2d-contributions_summary_plot.pdf' % (runname,instrument_type))
fig2.savefig('./%s/%s_thermal_profile_summary_plot.pdf' % (runname,instrument_type))


# Posterior plots
##############################
prefix = runname + ( '/%s_' %instrument_type)
parameters = json.load(open(prefix + 'params.json'))
n_params = len(parameters)
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix)
s = a.get_stats()

evidence = s['global evidence']
u_evidence = s['global evidence error']

json.dump(s, open(prefix + 'stats.json', 'w'), indent=4)

paramstxt = ['  marginal likelihood:']
paramstxt.append('    ln Z = %.2f +- %.2f' % (evidence, u_evidence)) 
paramstxt.append('  parameters:')
for p, m in zip(parameters, s['marginals']):
	lo, hi = m['1sigma']
	med = m['median']
	sigma = (hi - lo) / 2
	if sigma == 0:
		i = 3
	else:
		i = max(0, int(-np.floor(np.log10(sigma))) + 1)
	fmt = '%%.%df' % i
	fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
	paramstxt.append(fmts % (p, med, sigma))

print('\n'.join(paramstxt))

print('creating marginal plot ...')
p = pymultinest.PlotMarginal(a)

values = a.get_equal_weighted_posterior()
assert n_params == len(s['marginals'])
modes = s['modes']

if chemMode == 'free' and 'metallicity' not in parameters:        ## added: if chemMode == 'free'
    ## Convert prt's default MASS mixing ratios to VOLUME m.r.'s
    mmr_array = values[:,-(1+nspecies):-1]
    mmr_dict = dict()
    for speciesno, speciesname in enumerate(species):
        mmr_dict[speciesname] = mmr_array[:,speciesno]
    
    vmr_dict = mmr2vmr.mmr2vmr(mmr_dict)
    vmr_array = np.zeros(mmr_array.shape, dtype=float)
    for speciesno, speciesname in enumerate(species):
        vmr_array[:,speciesno] = np.log10(vmr_dict[speciesname])
    
    values[:,-(1+nspecies):-1]     = vmr_array

tools.hparams(values[:,0:-1], 50, labs=parameters)
plt.savefig('./%s/%s_1d_marg.pdf' % (runname,instrument_type))

if 'Pcloud' in parameters:
    if chemMode=='selfconsistent':
        metalind = parameters.index('[Fe/H]')
    elif chemMode=='free':
        metalind = parameters.index('H2O')

    cloudind = parameters.index('Pcloud')
    xlab = 'log10(Metallicity/Solar)'
    ylab = 'Cloudtop Pressure [bars]'
    xmed, xerr = np.median(values[:,metalind]), an.dumbconf(values[:,metalind], .683, mid='median')[0]
    ymed, yerr = np.median(values[:,cloudind]), an.dumbconf(values[:,cloudind], .683, mid='median')[0]
    yupper = an.dumbconf(values[:,cloudind], .954, type='lower')[0]


    fig=figure()
    ax1 = subplot(111, position=[.12, .12, .55, .55])
    hist2d(values[:,metalind], values[:,cloudind], bins=30, cmap=cm.cubehelix_r)
    xlabel(xlab, fontsize=16)
    ylabel(ylab, fontsize=16)
    xlim(0, 2.5)
    ylim(0, -3.5)
    ax2 = subplot(111, position=[.7, .12, .23, .55])
    hist(values[:,cloudind], 50, orientation='horizontal', histtype='step', color='k', linewidth=2)
    xlabel('Probability', fontsize=14)
    xticks(xticks()[0], [])
    yticks(yticks()[0], [])
    ylim(ax1.get_ylim())
    ax3 = subplot(111, position=[.12, .7, .55, .23])
    hist(values[:,metalind], 50, histtype='step', color='k', linewidth=2)
    ylabel('Probability', fontsize=14)
    xticks(xticks()[0], [])
    yticks(yticks()[0], [])
    xlim(ax1.get_xlim())
    [ax.minorticks_on() for ax in [ax1, ax2, ax3]]
    ax1.errorbar(xmed, ymed, yerr, xerr, fmt='*r', ms=18, mew=2, elinewidth=2)
    ax2.plot(ax2.get_xlim(), [yupper, yupper], ':r', linewidth=2)
    ax2.text(.95, .65,  'P$_{cloud}$ <      \n%1.1f mb (2$\sigma$)' % (1000*10**yupper),  color='r', weight='bold', horizontalalignment='right', transform=ax2.transAxes, fontsize=13)
    fig.text(.72, .8, 'TOI 431', fontsize=24)
    ax3.plot([xmed, xmed], ax3.get_ylim(),  '--r', linewidth=1.5)
    ax3.text(.05, .6,'log([Fe/H]) =\n %1.1f $\pm$ %1.1f' % (xmed, xerr),color='r', weight='bold', horizontalalignment='left', transform=ax3.transAxes, fontsize=13)



os.system('python multinest_marginals.py %s/%s' % (runname,instrument_type))
#execstr = 'gs -q -sPAPERSIZE=letter -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=./%s/allfigs.pdf ./%s/*.pdf' % (runname, runname)   
execstr = 'tar -cvf ./%s/allfigs.tar ./%s/*.pdf' % (runname, runname) 
os.system(execstr)
print('Finished summarizing run %s' % runname)
