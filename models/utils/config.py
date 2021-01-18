""" Global default variables or parameters """

# ---------------------------------------
# can be changed via parser at runtime
# ---------------------------------------

# Initial conditions
# if set to None they are computed from the input data
E_0 = None
A_0 = None
I_0 = None
R_0 = 0
D_0 = 0

"""
# Italy Parameters
N = 41300
BETA = 7.9e-6 * N  # beta
DELTA = 0.0213 + 0.0163  # delta
THETA = 0.0163 / DELTA

# China* Parameters
N = 79200
I_0 = 999
R_0 = 10
D_0 = 17
BETA = 3.33e-6 * N
DELTA = 0.018 + 0.003
THETA = 0.003 / DELTA

"""
# China Parameters
N = 83000
I_0 = 430
R_0 = 10
D_0 = 15
BETA = 3.95e-6 * N
DELTA = 0.0353 + 0.0031
THETA = 0.0031 / DELTA


# Defaults
BETA_A = BETA / 2
DELTA_A = 0.01
ALPHA = 1
EPSILON = 1

K_INF = 1
K_REC = 1
K_LAT = 1


NETWORK = "ba"
NETWORK_PARAM = 5


# Sections
TRANSITION_DAYS = 4
POINTS_PER_DAY = 4

# Monte Carlo
MC_SEED0 = 1
MC_NSEED = 5

# Size of arrays
N_T_STEPS = int(1e7)

# Save
SAVE_FOLDER = "/home/pol/Documents/iiia_udl/programs/results/"

# Automatic configurator
SEED = 1
TIMEOUT = 1200

# Data
DATA = "/home/pol/Documents/iiia_udl/programs/data/italy.dat"
DAY_MIN = 30
DAY_MAX = 54
SECTIONS_DAYS = [0, DAY_MAX - DAY_MIN]
UNDIAGNOSED = 0  # 89.4, from Alex Arenas 2020 paper (Physical Review X, 10(4), 041055.)
METRICS = [
    "sq_diff",
    "sq_diff_weight",
    "sq_diff_scaled",
    "abs_diff",
    "abs_diff_scaled",
]
METRIC = "sq_diff_weight"

# ---------------------------------------
# can't be changed via parser, only here
# ---------------------------------------

# Work with new daily cases (if False) or cumulative data (if True)
CUMULATIVE = False
