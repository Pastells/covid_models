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

# Defaults
N = 10000
BETA = 0.5
DELTA = 0.2
BETA_A = BETA / 2
DELTA_A = 0.01
THETA = 0.01
ALPHA = 1
EPSILON = 1

K_INF = 1
K_REC = 1
K_ASYM = 1


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

# Metaparameters for the automatic configurator
SEED = 1
TIMEOUT = 1200

# Data
DATA = "/home/pol/Documents/iiia_udl/programs/data/italy.dat"
DAY_MIN = 30
DAY_MAX = 54
SECTIONS_DAYS = [0, DAY_MAX - DAY_MIN]
UNDIAGNOSED = 0  # 89.4, from Alex Arenas 2020 paper (Physical Review X, 10(4), 041055.)

# Metrics
METRICS = [
    "sum_sq",
    "max_sq",
    "sum_sq_weight",
    "sum_sq_scaled",
]
METRICS_STR = "{" + ",".join(METRICS) + "}"

# Default metric

METRIC = "sum_sq"

# ---------------------------------------
# can't be changed via parser, only here
# ---------------------------------------

# Work with new daily cases (if False) or cumulative data (if True)
CUMULATIVE = False
