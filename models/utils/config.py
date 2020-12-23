"""
Global deafult variables or parameters
"""

# Initial conditions
E_0 = 0
R_0 = 0

# Parameters
N = 81600
BETA = 0.89
DELTA = 0.64
BETA1 = 0.01
DELTA1 = 0.01
EPSILON = 1

K_INF = 1
K_REC = 1
K_LAT = 1

SECTIONS_DAYS = [0, 24]

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

# Data
DATA = "/home/pol/Documents/iiia_udl/programs/data/italy_i.csv"
DAY_MIN = 34
DAY_MAX = 58

# Save
SAVE_FOLDER = "/home/pol/Documents/iiia_udl/programs/models/results/"

# Automatic configurator
SEED = 1
TIMEOUT = 1200
