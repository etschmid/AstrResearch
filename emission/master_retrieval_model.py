##----------------------------------------
## Master retrieval model -- for emission!
##----------------------------------------
#
# 2020-06-18 IJMC: A new version, sent to Ethen Schmidt.

import numpy as np
import sys
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from poor_mans_nonequ_chem_FeH import poor_mans_nonequ_chem as pm

def calc_MMW(abundances):

    MMWs = {}
    MMWs['e-'] = 0.
    MMWs['H'] = 1.
    MMWs['H-'] = 1.
    MMWs['H2'] = 2.
    MMWs['He'] = 4.
    MMWs['H2O'] = 18.
    MMWs['CH4'] = 16.
    MMWs['CO2'] = 44.
    MMWs['CO'] = 28.
    MMWs['Na'] = 23.
    MMWs['K'] = 39.
    MMWs['NH3'] = 17.
    MMWs['HCN'] = 27.
    MMWs['C2H2,acetylene'] = 26.
    MMWs['PH3'] = 34.
    MMWs['H2S'] = 34.
    MMWs['VO'] = 67.
    MMWs['TiO'] = 64.

    MMW = 0.
    for key in abundances.keys():
        if key == 'CO_all_iso':
            MMW += abundances[key]/MMWs['CO']
        else:
            MMW += abundances[key]/MMWs[key]
    
    return 1./MMW
    
####################################################################################
####################################################################################
####################################################################################

def retrieval_model_plain(rt_object, parameters, contribution=False):
    """contribution : bool
         Whether to return full RT object w/contribution fcns
    """
    gravity = parameters['gravity'] #1e1**log_g    
    temp_params = {}
    temp_params['log_delta'] = parameters['log_delta']
    temp_params['log_gamma'] = parameters['log_gamma']
    temp_params['t_int'] = parameters['t_int']
    temp_params['t_equ'] = parameters['t_equ']
    temp_params['log_p_trans'] = parameters['log_p_trans']
    temp_params['alpha'] = parameters['alpha']
    
    # Create temperature model
    press, temp = nc.make_press_temp(temp_params) # pressures from low to high
    # cgs to bar conversion:
    press_bar = rt_object.press/1e6

    if 'metallicity' in parameters and 'log_Pquench' in parameters:
        COs = parameters['CtoO'] * np.ones_like(rt_object.press)
        FeHs = parameters['metallicity'] * np.ones_like(rt_object.press)
        abundances = pm.interpol_abundances(COs, \
                FeHs, \
                temp, \
                press_bar, \
                Pquench_carbon = 1e1**parameters['log_Pquench'])
        for spec in rt_object.line_species:
            abundances[spec]  = abundances[spec.split('_')[0]]

        MMW = abundances['MMW']
    else:
        # Make dictionary for log 'metal' abundances
        ab_metals = {}
        for ii,speciesname in enumerate(parameters['species']):
            ab_metals[speciesname]     = parameters[speciesname] #params[-nspecies:][ii]

        abundances = {}
        metal_sum = 0.
        for name in ab_metals.keys():
            abundances[name] = np.ones_like(press)*1e1**ab_metals[name]
            metal_sum += 1e1**ab_metals[name]

        abH2He = 1. - metal_sum
        abundances['H2'] = np.ones_like(press)*abH2He*0.75
        abundances['He'] = np.ones_like(press)*abH2He*0.25

        MMW = calc_MMW(abundances)
        
    rt_object.calc_flux(temp, abundances, gravity, MMW)
    if contribution:
        rt_object.calc_flux(temp, abundances, gravity, MMW, contribution=contribution)
        ret = rt_object
    else:
        ret = nc.c/rt_object.freq, rt_object.flux
        
    return ret
