import csv
import numpy as np
import pylab as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'NH3', 'H2S'], \
      rayleigh_species = ['H2'], \
      #continuum_opacities = ['H2-H2', 'H2-He'], \
      wlen_bords_micron = [0.5, 20])


for i in np.linspace(-5,5,num=21):

    loggamma = i

    parameters = dict(gravity=10**3.11, log_delta=-6, log_gamma=loggamma, t_int=500, t_equ=1700, log_p_trans=-3, alpha=0.15)
    gravity = parameters['gravity'] 
    temp_params = {}
    temp_params['log_delta'] = parameters['log_delta']
    temp_params['log_gamma'] = parameters['log_gamma']
    temp_params['t_int'] = parameters['t_int']
    temp_params['t_equ'] = parameters['t_equ']
    temp_params['log_p_trans'] = parameters['log_p_trans']
    temp_params['alpha'] = parameters['alpha']

    press, temp = nc.make_press_temp(temp_params)

    atmosphere.setup_opa_structure(press)
    temperature = temp


    mass_fractions1 = {}
    mass_fractions1['H2'] = 0.98969613 * np.ones_like(temperature)
    mass_fractions1['H2O'] = 0.00281912 * np.ones_like(temperature)
    mass_fractions1['CO_all_iso'] = 0.00695022 * np.ones_like(temperature)
    mass_fractions1['CO2'] = 1.375e-6 * np.ones_like(temperature)
    mass_fractions1['CH4'] = 3.1547e-7 * np.ones_like(temperature)
    mass_fractions1['NH3'] = 3.3519e-7 * np.ones_like(temperature)
    mass_fractions1['H2S'] = 0.0005325 * np.ones_like(temperature)


    MMW = 2.33 * np.ones_like(temperature)
    R_pl = 4.72*nc.r_earth # per Jenkins
    P0 = 1e1**-1.6

    plot_select = [1,0,0,0,0]

    if(plot_select[0]==1):

        atmosphere.calc_flux(temperature, mass_fractions1, gravity, MMW)
        #plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, label='1x')
        
        plt.plot(temp,press, label='gamma = %d'% loggamma)
        #plt.plot(3.6,482,'ro')
        #plt.plot(4.5,375,'ro')
        
        
        
       # with open('Profile.csv', 'w') as csv_file: 
       #     csv_writer = csv.writer(csv_file, delimiter=',')
       #     csv_writer.writerow(nc.c/atmosphere.freq/1e-4)
       #     csv_writer.writerow(atmosphere.flux/1e-6)
            



plt.yscale('log')
#plt.xlabel('Wavelength (microns)')
#plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.ylabel('Pressure (Bar)')
plt.xlabel('Temp (K)')
plt.legend(title="Gamma")
#plt.xlim(1.0,20)
#plt.ylim(0,3)
plt.show()
plt.clf()