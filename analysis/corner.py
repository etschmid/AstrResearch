import pylab as plt
import pickle
import numpy as np
import seaborn as sns
import pylab as plt

# nice_corner source can be found in the petitRADTRANS gitlab repository,
# requires seaborn
from nice_corner import nice_corner

samples_path = 'chain_pos.pickle'

# Load samples
f = open(samples_path,'rb')
pos = pickle.load(f)
prob = pickle.load(f)
state = pickle.load(f)
samples = pickle.load(f)
f.close()

parameter_names = {0: r"$\rm log(delta)$", \
              1: r"$\rm log(gamma)$", \
              2: r"$\rm T_{int}$", \
              3: r"$\rm T_{equ}$", \
              4: r"$\rm log(P_{tr})$", \
              5: r"$\rm alpha$", \
              6: r"$\rm log(g)$", \
              7: r"$\rm log(P_0)$", \
              8: r"$\rm CO$", \
              9: r"$\rm H_2O$", \
              10: r"$\rm CH_4$", \
              11: r"$\rm NH_3$", \
              12: r"$\rm CO_2$", \
              13: r"$\rm H_2S$", \
              14: r"$\rm Na$", \
              15: r"$\rm K$"}

output_file = 'test.pdf'

N_samples = 700000

parameter_ranges = {0: None, \
             1: None, \
             2: None, \
             3: None, \
             4: None, \
             5: None, \
             6: None, \
             7: None, \
             8: None, \
             9: None, \
             10: (-10., 0.), \
             11: (-10, 0.), \
             12: None, \
             13: (-10, 0.), \
             14: None, \
             15: None}
			 
parameter_plot_indices = [7, 8, 9, 10, 11, 12, 13, 14, 15]



max_val_ratio = 5.

nice_corner(samples, \
            parameter_names, \
            output_file, \
            N_samples = N_samples, \
            parameter_plot_indices = parameter_plot_indices, \
            true_values = true_values, \
            max_val_ratio = max_val_ratio, \
            parameter_ranges = parameter_ranges)