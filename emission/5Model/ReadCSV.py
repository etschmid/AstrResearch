import csv
import numpy as np
import pylab as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

with open('LTT9779_fluxes_erg_s_cm2_Hz_sr.csv') as csv_file: ##Read in csv file. 
    csv_reader = csv.reader(csv_file, delimiter=' ')
    wavelength = []
    stellar_spectrum = []
    stellar_blackbody = []
    planet_blackbody_1800 = []
    planet_blackbody_2000 = []
    
    for row in csv_reader:
        wave = row[0]
        stellar_s = row[1]
        stellar_b = row[2]
        planet_b1800 = row[3]
        planet_b2000 = row[4]
        
        wavelength.append(wave)
        stellar_spectrum.append(stellar_s)
        stellar_blackbody.append(stellar_b)
        planet_blackbody_1800.append(planet_b1800)
        planet_blackbody_2000.append(planet_b2000)


#for i in range(98900,388065,1000):
plt.plot(wavelength,stellar_spectrum)
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Stellar flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.clf()