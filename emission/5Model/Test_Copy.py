import csv
import numpy as np
import pylab as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', 'CH4', 'CO2', 'NH3', 'H2S'], \
      rayleigh_species = ['H2'], \
      #continuum_opacities = ['H2-H2', 'H2-He'], \
      wlen_bords_micron = [0.5, 20])
      

parameters = dict(gravity=10**3.11, log_delta=-6, log_gamma=-3, t_int=500, t_equ=1700, log_p_trans=-3, alpha=0.15)
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

mass_fractions10 = {}
mass_fractions10['H2'] = 0.90583842 * np.ones_like(temperature)
mass_fractions10['H2O'] = 0.02599988 * np.ones_like(temperature)
mass_fractions10['CO_all_iso'] = 0.06409983 * np.ones_like(temperature)
mass_fractions10['CO2'] = 0.00015964 * np.ones_like(temperature)
mass_fractions10['CH4'] = 2.3111e-7 * np.ones_like(temperature)
mass_fractions10['NH3'] = 9.7757e-7 * np.ones_like(temperature)
mass_fractions10['H2S'] = 0.00390102 * np.ones_like(temperature)

mass_fractions100 = {}
mass_fractions100['H2'] = 0.50632723 * np.ones_like(temperature)
mass_fractions100['H2O'] = 0.15574354 * np.ones_like(temperature)
mass_fractions100['CO_all_iso'] = 0.30499699 * np.ones_like(temperature)
mass_fractions100['CO2'] = 0.00956291 * np.ones_like(temperature)
mass_fractions100['CH4'] = 1.3844e-7 * np.ones_like(temperature)
mass_fractions100['NH3'] = 1.4709e-6 * np.ones_like(temperature)
mass_fractions100['H2S'] = 0.02336773 * np.ones_like(temperature)

mass_fractions1000 = {}
mass_fractions1000['H2'] = 0.06097573 * np.ones_like(temperature)
mass_fractions1000['H2O'] = 0.18206009 * np.ones_like(temperature)
mass_fractions1000['CO_all_iso'] = 0.44884901 * np.ones_like(temperature)
mass_fractions1000['CO2'] = 0.28079858 * np.ones_like(temperature)
mass_fractions1000['CH4'] = 3.229e-10 * np.ones_like(temperature)
mass_fractions1000['NH3'] = 3.4308e-7 * np.ones_like(temperature)
mass_fractions1000['H2S'] = 0.02731625 * np.ones_like(temperature)

mass_fractions10000 = {}
mass_fractions10000['H2'] = 0.01829015 * np.ones_like(temperature)
mass_fractions10000['H2O'] = 0.06671905 * np.ones_like(temperature)
mass_fractions10000['CO_all_iso'] = 0.26069662 * np.ones_like(temperature)
mass_fractions10000['CO2'] = 0.64927704 * np.ones_like(temperature)
mass_fractions10000['CH4'] = 2.361e-12 * np.ones_like(temperature)
mass_fractions10000['NH3'] = 6.3012e-9 * np.ones_like(temperature)
mass_fractions10000['H2S'] = 0.00501714 * np.ones_like(temperature)

MMW = 2.33 * np.ones_like(temperature)
R_pl = 4.72*nc.r_earth # per Jenkins
P0 = 1e1**-1.6

plot_select = [1,1,1,1,1]

if(plot_select[0]==1):

    atmosphere.calc_flux(temperature, mass_fractions1, gravity, MMW)
    plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, label='1x')

    with open('Solar1.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(nc.c/atmosphere.freq/1e-4)
        csv_writer.writerow(atmosphere.flux/1e-6)
        
if(plot_select[1]==1):
    atmosphere.calc_flux(temperature, mass_fractions10, gravity, MMW)
    plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, label='10x')

    with open('Solar10.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(nc.c/atmosphere.freq/1e-4)
        csv_writer.writerow(atmosphere.flux/1e-6)
        
if(plot_select[2]==1):
    atmosphere.calc_flux(temperature, mass_fractions100, gravity, MMW)
    plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, label='100x')

    with open('Solar100.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(nc.c/atmosphere.freq/1e-4)
        csv_writer.writerow(atmosphere.flux/1e-6)
        
if(plot_select[3]==1):
    atmosphere.calc_flux(temperature, mass_fractions1000, gravity, MMW)
    plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, label='1000x')

    with open('Solar1000.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(nc.c/atmosphere.freq/1e-4)
        csv_writer.writerow(atmosphere.flux/1e-6)
        
if(plot_select[4]==1):
    atmosphere.calc_flux(temperature, mass_fractions10000, gravity, MMW)
    plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6, label='10000x')

    with open('Solar10000.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(nc.c/atmosphere.freq/1e-4)
        csv_writer.writerow(atmosphere.flux/1e-6)



plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.legend(title="Stellar Metalicity")
plt.xlim(1.0,20)
plt.ylim(0,3)
plt.show()
plt.clf()