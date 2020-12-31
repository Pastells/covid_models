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

# Parameters
N = 81600
BETA = 0.89  # BETA_I
DELTA = 0.64  # DELTA_I
BETA_A = 0.01
DELTA_A = 0.01
ALPHA = 1
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

# Save
SAVE_FOLDER = "/home/pol/Documents/iiia_udl/programs/results/"

# Automatic configurator
SEED = 1
TIMEOUT = 1200

# Data
DATA = "/home/pol/Documents/iiia_udl/programs/data/italy_i.csv"
DAY_MIN = 33
DAY_MAX = 83
UNDIAGNOSED = 0  # 89.4, from Alex Arenas 2020 paper (Physical Review X, 10(4), 041055.)

# ---------------------------------------
# can't be changed via parser, only here
# ---------------------------------------

# Work with new daily cases (if False) or cumulative data (if True)
CUMULATIVE = True
