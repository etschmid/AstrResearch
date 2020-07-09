

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
MMWs['CO_all_iso'] = 28.
MMWs['Na'] = 23.
MMWs['K'] = 39.
MMWs['NH3'] = 17.
MMWs['HCN'] = 27.
MMWs['C2H2,acetylene'] = 26.
MMWs['PH3'] = 34.
MMWs['H2S'] = 34.
MMWs['VO'] = 67.
MMWs['TiO'] = 64.

def calc_MMW(abundances):
    MMW = 0.
    for key in abundances.keys():
        if key == 'CO_all_iso':
            MMW += abundances[key]/MMWs['CO']
        else:
            MMW += abundances[key]/MMWs[key]
    return 1./MMW

def mmr2vmr(ab_metals):
    MMR = {}
    metal_sum = 0.
    for name in ab_metals.keys():
        MMR[name] = 1e1**ab_metals[name]
        metal_sum += 1e1**ab_metals[name]
    if hasattr(metal_sum, '__iter__'):
        if (metal_sum>1).any(): metal_sum[metal_sum>1] = 1.
    elif metal_sum>1: metal_sum=1.

    abH2He = 1. - metal_sum
    MMR['H2'] = abH2He*0.75
    MMR['He'] = abH2He*0.25

    MMW = calc_MMW(MMR)
    VMR = dict()
    for species in MMR.keys():
        VMR[species] = 1.0 * (MMW/MMWs[species]) * MMR[species]

    return VMR
