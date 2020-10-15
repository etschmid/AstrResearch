import csv
import pandas as pd
import numpy as np
import pylab as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc


R_pl = 4.72*nc.r_earth
R_star = 0.949*nc.r_sun

col_list = ["Star", "Planet"]
df = pd.read_csv("Flux.csv", usecols=col_list)

F = {}
df2 = pd.DataFrame(F)
temp = 0 
for i in range(0,3688,1):
    F[i] = (df["Planet"][i]/df["Star"][temp]) * (R_pl/R_star)**2
    temp = temp + 5


with open('Flux_R.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(F)

df2.to_csv('TESTTEST.csv')

print(F)

