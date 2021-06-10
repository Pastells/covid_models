import numpy as np
import argparse

# Cost days
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--alfa1", type=float)
parser.add_argument("--day_min", type=int, default=3)
parser.add_argument("--day_max ", type=int, default=46)
args = parser.parse_args()
# -------------------
day_min, day_max = args.day_min, args.day_max


# Read and average results
# -------------------
df = np.loadtxt("results.dat")
results = np.array([np.mean(df[i::4], axis=0) for i in range(4)])
check = df[0:4]
# -------------------


# Data
# -------------------
# fmt: off
Guariti = np.array([0, 0, 0, 1, 1, 1, 3, 45, 46, 50, 83, 149, 160, 276, 414, 523, 589, 622, 724, 1004, 1045, 1258, 1439, 1966, 2335, 2749, 2941, 4025, 4440, 5129, 6072, 7024, 7432, 8326, 9362, 10361, 10950, 12384, 13030, 14620, 15729, 16847, 18278, 19758, 20996, 21815])
Isolamento_domiciliare = np.array([49, 91, 162, 221, 284, 412, 543, 798, 927, 1000, 1065, 1155, 1060, 1843, 2180, 2936, 2599, 3724, 5036, 6201, 7860, 9268, 10197, 11108, 12090, 14935, 19185, 22116, 23783, 26522, 28697, 30920, 33648, 36653, 39533, 42588, 43752, 45420, 48134, 50456, 52579, 55270, 58320])
Ricoverati_sintomi = np.array([54, 99, 114, 128, 248, 345, 401, 639, 742, 1034, 1346, 1790, 2394, 2651, 3557, 4316, 5038, 5838, 6650, 7426, 8372, 9663, 11025, 12894, 14363, 15757, 16020, 17708, 19846, 20692, 21937, 23112, 24753, 26029, 26676, 27386, 27795, 28192, 28403, 28540, 28741, 29010, 28949])
Terapia_intensiva = np.array([26, 23, 35, 36, 56, 64, 105, 140, 166, 229, 295, 351, 462, 567, 650, 733, 877, 1028, 1153, 1328, 1518, 1672, 1851, 2060, 2257, 2498, 2655, 2857, 3009, 3204, 3396, 3489, 3612, 3732, 3856, 3906, 3981, 4023, 4035, 4053, 4068, 3994, 3977])
# fmt: on

start = (day_min <= 4) * 0 + (day_min > 4) * (day_min - 3)

data = [
    Guariti[day_min:day_max],
    Isolamento_domiciliare[start:day_max],
    Ricoverati_sintomi[start:day_max],
    Terapia_intensiva[start:day_max],
]
# -------------------


# Cost
# -------------------
cost = np.zeros(5)
cost_check = np.zeros(5)
for day in range(day_max - day_min):
    for comp in range(4):
        cost[comp] += (results[comp][day] - data[comp][day]) ** 2
        cost_check[comp] += (check[comp][day] - data[comp][day]) ** 2

cost[4] = np.sum(cost[:4], axis=0)
cost = cost / 1e6
print(cost)

cost_check[4] = np.sum(cost_check[:4], axis=0)
cost_check = cost_check / 1e6
print("Check first one")
print(cost_check)

with open("costs.dat", "a") as f:
    cost_str = " ".join("%.4f" % i for i in cost)
    f.write(f"# Average\n")
    f.write(f"{cost_str}\n")
# -------------------
