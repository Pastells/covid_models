"""Create file with all the data for a country
I = C - R - D"""
import sys
import numpy as np
import datetime

country = sys.argv[1]

data_c = np.genfromtxt(country + "_c.csv", delimiter=",", dtype=int)
data_r = np.genfromtxt(country + "_r.csv", delimiter=",", dtype=int)
data_d = np.genfromtxt(country + "_d.csv", delimiter=",", dtype=int)

# Works for China, may not work for other countries with more than one line
if data_c.ndim == 2:
    data_c = data_c[:, 3:].sum(axis=0)
    data_r = data_r[:, 3:].sum(axis=0)
    data_d = data_d[:, 3:].sum(axis=0)
else:
    data_c = data_c[3:]
    data_r = data_r[3:]
    data_d = data_d[3:]

data_i = data_c - data_r - data_d

day_0 = datetime.datetime.strptime("1/22/20", "%m/%d/%y")

dates = [day_0 + datetime.timedelta(days=day) for day, _ in enumerate(data_i)]

with open(country + ".dat", "w") as out_file:
    out_file.write("# date, infected, recovered, dead, cumulative\n")
    for day, value in enumerate(data_i):
        out_file.write(
            f"{value:8.0f}, {data_r[day]:8.0f}, {data_d[day]:8.0f}, {data_c[day]:8.0f}, {dates[day]:%m/%d/%y}\n"
        )

"""
with open(country + "_i.dat", "w") as out_file:
    for day, value in enumerate(data_i):
        out_file.write(f"{value:8.0f}\n")
"""
