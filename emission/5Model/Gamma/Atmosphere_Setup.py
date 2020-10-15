import numpy as np
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'NH3', 'H2S'], \
      rayleigh_species = ['H2'], \
      #continuum_opacities = ['H2-H2', 'H2-He'], \
      wlen_bords_micron = [0.5, 20])