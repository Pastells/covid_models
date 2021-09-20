import numpy
import pandas

confirmed = [6, 12, 19, 25, 31, 38, 44, 60, 80, 131, 131, 259, 467, 688, 776, 1776, 1460, 1739, 1984, 2101, 2590, 2827,
             3233, 3892, 3697, 3151, 3387, 2653, 2984, 2473, 2022, 1820, 1998, 1506, 1278, 2051, 1772, 1891, 399, 894,
             397, 650, 415, 518, 412, 439, 441, 435, 579, 206, 130, 120, 143, 146, 102, 46, 45, 20, 31, 26, 11, 18, 27,
             29, 39, 39]
n_days = len(confirmed)
recovered = numpy.full(shape=n_days, fill_value=-1)
dead = [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 8, 15, 15, 25, 26, 26, 38, 43, 46, 45, 57, 64, 66, 73, 73, 86, 89, 97, 108, 97,
        254, 121, 121, 142, 106, 106, 98, 115, 118, 109, 97, 150, 71, 52, 29, 44, 37, 35, 42, 31, 38, 31, 30, 28, 27,
        23, 17, 22, 11, 7, 14, 10, 14, 13, 13]
cummulative = numpy.full(shape=n_days, fill_value=-1)

dates = pandas.date_range("2020/01/04", "2020/03/09")

data = pandas.DataFrame({
    "#infected": confirmed,
    "recovered": recovered,
    "dead": dead,
    "cummulative": cummulative,
    "date": dates
})

print(data)

data.to_csv("wuhan.csv", index=False)
