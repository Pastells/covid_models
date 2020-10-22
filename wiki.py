import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Input parameters ####################

# int; total population
N = 350

# float; maximum elapsed time
T = 100.0

# float; start time
t = 0.0

# float; spatial parameter
V = 100.0

# float; rate of infection after contact
_alpha = 10.0

# float; rate of cure
_beta = 10.0

# int; initial infected population
n_I = 1

#########################################

# Compute susceptible population, set recovered to zero
n_S = N - n_I
n_R = 0

# Initialize results list
SIR_data = []
SIR_data.append((t, n_S, n_I, n_R))

# Main loop
while t < T:
    if n_I == 0:
        break

    w1 = _alpha * n_S * n_I / V
    w2 = _beta * n_I
    W = w1 + w2

    # generate exponentially distributed random variable dt
    # using inverse transform sampling
    dt = -math.log(1 - random.uniform(0.0, 1.0)) / W
    t = t + dt

    if random.uniform(0.0, 1.0) < w1 / W:
        n_S = n_S - 1
        n_I = n_I + 1
    else:
        n_I = n_I - 1
        n_R = n_R + 1

    SIR_data.append((t, n_S, n_I, n_R))

with open("SIR_data.txt", "w+") as fp:
    fp.write("\n".join("%f %i %i %i" % x for x in SIR_data))


from scipy.integrate import odeint
import numpy as np

# Numerical solution using an ordinary differential equation solver
ts = np.linspace(0, 2, num=200)
initial_S_I_R = (N - 1, 1, 0)


def differential_SIR(initial_S_I_R, t, _alpha, _beta, V):
    n_S, n_I, n_R = initial_S_I_R
    dS_dt = -_alpha * n_S / V * n_I
    dI_dt = (_alpha * n_S / V - _beta) * n_I
    dR_dt = _beta * n_I
    return dS_dt, dI_dt, dR_dt


solution = odeint(differential_SIR, initial_S_I_R, ts, args=(_alpha, _beta, V))
plt.plot(ts,solution)
plt.show()
