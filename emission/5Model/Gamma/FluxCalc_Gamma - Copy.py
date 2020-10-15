import numpy as np
import pandas as pd
import pylab as plt
from petitRADTRANS import nat_cst as nc

R_pl =   4.72*nc.r_earth # per Jenkins
R_star = 0.949*nc.r_sun # per TIC v8

star_Filename = 'LTT9779_fluxes_erg_s_cm2_Hz_sr.csv'
planet_Filename = 'Solar1_Profile.csv'

planet_Data = np.loadtxt(planet_Filename, delimiter = ',')
star_Data = pd.read_csv(star_Filename)

planet_Wavelength, planet_Flux = planet_Data[0], planet_Data[1]

planet_Flux_newGrid = np.interp(star_Data.wavelength, planet_Wavelength, planet_Flux)

TestRatio = planet_Flux_newGrid / star_Data.stellarSpectrum

Contrast = TestRatio * (R_pl/R_star)**2

plt.plot(star_Data.wavelength,Contrast)
plt.plot(3.6,482,'ro')
plt.plot(4.5,375,'ro')
error1 = 47
error2 = 62
plt.errorbar(3.6,482,yerr=error1, fmt = ' ')
plt.errorbar(4.5,375,yerr=error2, fmt = ' ')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'$F_{planet}$ / $F_*$')
plt.yscale('log')
plt.xscale('log')
plt.show()