from functools import partial

import numpy
from scipy.optimize import minimize

from models.sir import sir
from models.utils import config

# CLI ARGS
dataset = "data/china.dat"
seed = 42

# COMMON ARGS
day_min = 0
day_max = 54
mc_nseed = 100

t_total = day_max - day_min
time_series = numpy.loadtxt(dataset, delimiter=",", dtype=int, usecols=(0, 1, 2, 3))
time_series = time_series[day_min:day_max]


# Optimization
def function(x):
    # Expand arguments
    n, initial_infected, initial_recovered, delta, beta = x
    cost = sir.sir(
        time_series=time_series,
        seed=seed,
        n_seeds=mc_nseed,
        t_total=t_total,
        n_t_steps=config.N_T_STEPS,
        metric="models.utils.utils.sum_sq",
        n=n,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        delta=delta,
        beta=beta
    )
    print(f"New cost is {cost}")
    return cost

x0 = numpy.array([70000, 10, 4, 0.2, 0.5])
# Add the range bounds
bounds = (
    (70000, 90000),
    (1, 1000),
    (0, 1000),
    (0.1, 1.0),
    (0.1, 1.0)
)
res = minimize(function, x0, bounds=bounds)

print(res.x)
print(res.message)
