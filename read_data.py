import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data_folder = "/home/pol/Documents/covid/data/"
country = "italy"
infected = genfromtxt(data_folder+country+"_i.csv", delimiter=',')
recovered = genfromtxt(data_folder+country+"_r.csv", delimiter=',')
dead = genfromtxt(data_folder+country+"_d.csv", delimiter=',')

# Index 0 is country name, 1 latitude, 2 longitude
date_0 = 3
# 15/03/20, taken from dates.dat
date_max = 58
plt.plot(infected[date_0:],'o',label='I')
plt.plot(recovered[date_0:],'o',label='R')
plt.plot(dead[date_0:],'o',label='D')
plt.legend()
plt.show()
