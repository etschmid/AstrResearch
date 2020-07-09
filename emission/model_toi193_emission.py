import numpy as np
import sys
import emcee
import pickle as pickle
import time
from emcee.utils import MPIPool

from petitRADTRANS import Radtrans
import master_retrieval_model as rm
from petitRADTRANS import nat_cst as nc
import rebin_give_width as rgw
from scipy.interpolate import interp1d
import pylab as plt

import tools

plt.ion()

depth = 1584. # transitdepth



wobs = [3.6, 4.5]
wedges = [3.2, 4], [4, 5.]
obs = [519, 383]
eobs = [ 42, 57]

wobs = [0.8,3.6, 4.5]
wedges = [.6, .9], [3.2, 4], [4, 5.]
obs = [60,519, 383]
eobs = [40, 42, 57]

kappa_IR = 0.01
gamma = 0.2
T_int = 200.
T_equ = 2800.



atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'Na', 'K'], \
      rayleigh_species = ['H2', 'He'], \
      continuum_opacities = ['H2-H2', 'H2-He'], \
      wlen_bords_micron = [0.3, 15])

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

R_pl = 3.96*nc.r_earth
gravity = 1412. # per Jenkins

for T_equ in [2000, 2400, 2800]:
    for kappa_IR in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        for gamma in [0.1, 0.2, 0.4, 0.8]:


allchis = []
allTequ = []
allkappa = []
allgamma = []

for T_equ in [1900, 1950, 2000, 2050, 2100]:
    for kappa_IR in [0.02, .05, 0.1, 0.2]:
        for gamma in [0.025, 0.05, 0.1, 0.15]:

            
            temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
            abundances = {}
            abundances['H2'] = 0.74 * np.ones_like(temperature)
            abundances['He'] = 0.24 * np.ones_like(temperature)
            abundances['H2O'] = 0.0003 * np.ones_like(temperature)
            abundances['CO_all_iso'] = 0.0005 * np.ones_like(temperature)
            abundances['CO2'] = 0.00001 * np.ones_like(temperature)
            abundances['CH4'] = 0.000001 * np.ones_like(temperature)
            abundances['Na'] = 0.00001 * np.ones_like(temperature)
            abundances['K'] = 0.000001 * np.ones_like(temperature)

            MMW = 2.33 * np.ones_like(temperature)

            atmosphere.calc_flux(temperature, abundances, gravity, MMW)

            stellar_spec = nc.get_PHOENIX_spec(5422)
            wlen_in_cm = stellar_spec[:,0]
            flux_star = stellar_spec[:,1]

            contrast = atmosphere.flux/np.interp(nc.c/atmosphere.freq/1e-4, wlen_in_cm/1e-4, flux_star) * depth

            lt = [.8, 1, 1.7, 3, 4, 5]

            atmotxt = '$\kappa_{IR}=%1.3f$' % kappa_IR, '$\gamma = %1.2f$' % gamma, '$T_{int} = %i K$' % T_int, '$T_{irr} = %i K$' % T_equ


            plt.figure(tools.nextfig(), [10, 5])
            plt.subplot(121)
            plt.plot(temperature, pressures)
            plt.yscale('log')
            plt.ylim([1e2, 1e-6])
            plt.xlabel('T (K)')
            plt.ylabel('P (bar)')
            plt.text(.9, .93, '\n'.join(atmotxt), horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=16)

            plt.subplot(122)
            plt.plot(nc.c/atmosphere.freq/1e-4, contrast)
            plt.xscale('log')
            plt.xlabel('Wavelength (microns)')
            plt.ylabel(r'Planet/star contrast')
            plt.xticks(lt,lt)
            plt.xlim(.7, 6)
            plt.errorbar(wobs, obs, eobs, fmt='or', elinewidth=2, mew=2, ms=7)

            chisq = 0.
            for ii in range(len(wobs)):
                edge = wedges[ii]
                ind = np.logical_and(nc.c/atmosphere.freq/1e-4 > edge[0], nc.c/atmosphere.freq/1e-4 < edge[1])
                avgband = contrast[ind].mean()
                plt.plot(wobs[ii], avgband, 's', ms=6, c='c', mew=2)
                chisq += ((obs[ii] - avgband) / eobs[ii])**2
            plt.text(.1, .9, '$\chi^2 = %1.1f$' % chisq, fontsize=20, horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)
            
            allchis.append(chisq)
            allTequ.append(T_equ)
            allkappa.append(kappa_IR)
            allgamma.append(gamma)

ntequ = np.unique(allTequ).size
nkappa = np.unique(allkappa).size
ngamma = np.unique(allgamma).size
Carray = np.array(allchis).reshape(ntequ, nkappa, ngamma)
Tarray =  np.array(allTequ).reshape(ntequ, nkappa, ngamma)
Karray =  np.array(allkappa).reshape(ntequ, nkappa, ngamma)
Garray =  np.array(allgamma).reshape(ntequ, nkappa, ngamma)


tind, kind, gind = (Carray==Carray.min()).nonzero()

xt, yt = Garray[tind][0,0], Karray[tind][0,:,0]
figure()
contourf(xt,yt, Carray[tind[0]])
colorbar()
title('Tequ = %i' % np.unique(Tarray[tind])[0])
xlabel('gamma')
ylabel('kappa_IR')
xticks(xt,xt)
yticks(yt,yt)

xt,yt = Garray[:,kind][0,0], Tarray[:,kind][:,0,0]
figure()
contourf(xt, yt, Carray[:,kind[0]])
colorbar()
title('kappa_IR = %1.4f' % Karray[tind,kind,gind])
xlabel('gamma')
ylabel('T_equ')
xticks(xt,xt)
yticks(yt,yt)

xt,yt = Karray[:,:,gind][0,:,0], Tarray[:,:,gind][:,0,0]
figure()
contourf(xt,yt, Carray[:,:,gind][:,:,0])
colorbar()
title('gamma = %1.4f' % Garray[tind,kind,gind])
xlabel('kappa_IR')
ylabel('T_equ')


