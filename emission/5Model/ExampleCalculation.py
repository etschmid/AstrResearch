import numpy as np
import pandas as pd

starFilename = 'ltt9779b_erg_s.csv' # several columns of data
planetFilename = 'Flux.csv' # 2 rows of data



planetModelData = np.loadtxt(planetFilename, delimiter=',')  # 2xN NumPy array
starData = pd.read_csv(starFilename)   # loads a Pandas data frame of length M
# M = 18915

planetWavelength, planetFlux = planetModelData[0], planetModelData[1]
# starData.wavelength, starData.stellarSpectrum, ... etc.

planetFlux_on_new_grid = np.interp(starData.wavelength, planetWavelength, planetFlux)
#   Given ES's model (x,y) and IC's wavelength grid x2, this projects y onto x2

someRatio = planetFlux_on_new_grid / starData.stellarSpectrum  # not quite 'contrast' yet