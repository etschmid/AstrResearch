import csv
import pandas as pd
import numpy as np
import pylab as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc




col_list = ["wavelength", "stellarSpectrum", "stellarBlackbody", "planetBlackbody1800K", "planetBlackbody2000K"]
df = pd.read_csv("LTT9779_fluxes_erg_s_cm2_Hz_sr.csv", usecols=col_list)

plt.plot(df["wavelength"],df["stellarSpectrum"])
plt.xscale('log')
axes = plt.gca()
axes.set_xlim([0,10e2])
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Stellar flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.clf()