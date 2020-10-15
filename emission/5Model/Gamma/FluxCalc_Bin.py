import numpy as np
import pandas as pd
import pylab as plt
import analysis as an
from petitRADTRANS import nat_cst as nc

R_pl =   4.72*nc.r_earth # per Jenkins
R_star = 0.949*nc.r_sun # per TIC v8
nbin = 20 #bin down by a factor of 20

star_Filename = 'LTT9779_fluxes_erg_s_cm2_Hz_sr.csv'
planet_Filename1 = 'Solar1_Profile.csv'
planet_Filename2 = 'Solar10.csv'
planet_Filename3 = 'Solar100.csv'
planet_Filename4 = 'Solar1000.csv'
planet_Filename5 = 'Solar10000.csv'

planet_Data1 = np.loadtxt(planet_Filename1, delimiter = ',')
planet_Data2 = np.loadtxt(planet_Filename2, delimiter = ',')
planet_Data3 = np.loadtxt(planet_Filename3, delimiter = ',')
planet_Data4 = np.loadtxt(planet_Filename4, delimiter = ',')
planet_Data5 = np.loadtxt(planet_Filename5, delimiter = ',')

star_Data = pd.read_csv(star_Filename)

planet_Wavelength1, planet_Flux1 = planet_Data1[0], planet_Data1[1]
planet_Wavelength2, planet_Flux2 = planet_Data2[0], planet_Data2[1]
planet_Wavelength3, planet_Flux3 = planet_Data3[0], planet_Data3[1]
planet_Wavelength4, planet_Flux4 = planet_Data4[0], planet_Data4[1]
planet_Wavelength5, planet_Flux5 = planet_Data5[0], planet_Data5[1]

#planet_Flux_newGrid = np.interp(star_Data.wavelength, planet_Wavelength, planet_Flux)

low_spectrum = an.binarray(star_Data.stellarSpectrum, nbin) / nbin
low_wavelength = an.binarray(star_Data.wavelength, nbin) / nbin

planet_Flux_newGrid1 = np.interp(low_wavelength, planet_Wavelength1, planet_Flux1)
planet_Flux_newGrid2 = np.interp(low_wavelength, planet_Wavelength2, planet_Flux2)
planet_Flux_newGrid3 = np.interp(low_wavelength, planet_Wavelength3, planet_Flux3)
planet_Flux_newGrid4 = np.interp(low_wavelength, planet_Wavelength4, planet_Flux4)
planet_Flux_newGrid5 = np.interp(low_wavelength, planet_Wavelength5, planet_Flux5)

#TestRatio = planet_Flux_newGrid / star_Data.stellarSpectrum
TestRatio1 = planet_Flux_newGrid1 / low_spectrum
TestRatio2 = planet_Flux_newGrid2 / low_spectrum
TestRatio3 = planet_Flux_newGrid3 / low_spectrum
TestRatio4 = planet_Flux_newGrid4 / low_spectrum
TestRatio5 = planet_Flux_newGrid5 / low_spectrum

Contrast1 = TestRatio1 * (R_pl/R_star)**2
Contrast2 = TestRatio2 * (R_pl/R_star)**2
Contrast3 = TestRatio3 * (R_pl/R_star)**2
Contrast4 = TestRatio4 * (R_pl/R_star)**2
Contrast5 = TestRatio5 * (R_pl/R_star)**2

plt.plot(low_wavelength,Contrast1,label='1x')
plt.plot(low_wavelength,Contrast2,label='10x')
plt.plot(low_wavelength,Contrast3,label='100x')
plt.plot(low_wavelength,Contrast4,label='1000x')
plt.plot(low_wavelength,Contrast5,label='10000x')
#plt.plot(low_wavelength,low_spectrum)
plt.ylim(1e0,1e3)
plt.xlim(1e0,1e1)
plt.plot(3.6,482,'ro')
plt.plot(4.5,375,'ro')
error1 = 47
error2 = 62
plt.errorbar(3.6,482,yerr=error1, fmt = ' ')
plt.errorbar(4.5,375,yerr=error2, fmt = ' ')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'$F_{planet}$ / $F_*$')
#plt.yscale('log')
plt.xscale('log')
plt.legend(title="Stellar Metalicity")
plt.show()