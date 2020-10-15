import csv
import pandas as pd
import numpy as np
import pylab as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc


R_pl = 4.72*nc.r_earth
R_star = 0.949*nc.r_sun

# importing csv module 
import csv 
  
# csv file name 
filename = "Flux.csv"
  
# initializing the titles and rows list 
fields = [] 
rows = [] 
  
# reading csv file 
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = next(csvreader) 
  
    # extracting each data row one by one 
    for row in csvreader: 
        rows.append(row) 
  
    # get total number of rows 
    print("Total no. of rows: %d"%(csvreader.line_num)) 
  
# printing the field names 
print('Field names are:' + ', '.join(field for field in fields)) 
  
#  printing first 5 rows 
print('\nFirst 5 rows are:\n') 
for row in rows[:5]: 
    # parsing each column of a row 
    for col in row: 
        print("%10s"%col), 
    print('\n') 

F = {}
temp = 0 
for i in range(0,3688,1):
    F[i] = (float(rows[i][1])/float(rows[temp][0])) * (R_pl/R_star)**2
    temp = temp + 5


with open('Flux_R.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(F)


print(F)

