"""Create file with all the data for a country
I = C - R - D"""
import sys
import numpy as np

country = sys.argv[1]

data_c = np.loadtxt(country + "_c.dat")
data_r = np.loadtxt(country + "_r.dat")
data_d = np.loadtxt(country + "_d.dat")
data_i = data_c - data_r - data_d

with open(country + ".dat", "w") as out_file:
    out_file.write("# infected, recovered, dead, cumulative\n")
    for day, value in enumerate(data_i):
        out_file.write(
            f"{value:8.0f}, {data_r[day]:8.0f}, {data_d[day]:8.0f}, {data_c[day]:8.0f}\n"
        )

with open(country + "_i.dat", "w") as out_file:
    for day, value in enumerate(data_i):
        out_file.write(f"{value:8.0f}\n")
