"""
Discrete mean-field SIRD model

Pol Pastells, 2020-2021

Equations of the deterministic system:

Delta_S(t) = - beta*I(t)*S(t)/(I(t)+S(t)) \n
Delta_I(t) =   beta/N*I(t)*S(t) -(delta+theta)* I(t) \n
Delta_R(t) =                   delta * I(t) \n
Delta_D(t) =                   theta * I(t)
"""

import sys
import numpy as np
import pandas as pd
from optilog.autocfg import ac, Int, Real


N = 59.3e6


@ac
def sird(
    data,
    day_min,
    day_max,
    plot,
    # n: Int(1000, int(1e8)) = int(59.3e6),
    q: Real(0, 1) = 0.1,
    w: Real(0, 1) = 0.9,
    delta: Real(0.01, 1.0) = 0.3,
    beta: Real(0.01, 1.0) = 0.5,
    theta: Real(0.001, 0.1) = 0.005,
):
    I, R, D = get_data(data, day_min, day_max)
    S = q * N - I - R - D
    data = np.array([S, I, R, D])

    if q < np.max(I + R + D) / N:
        raise ValueError("q is too small")

    cost = np.zeros(4)
    day_range = 1 + day_max - day_min
    left = np.zeros([4, day_range])
    right = np.zeros([4, day_range])
    for t in range(day_range):
        left[:, t] = data[:, t + 1] - data[:, t]
        r = beta * S[t] * I[t] / (S[t] + I[t])
        right[0, t] = -r
        right[1, t] = r - (delta + theta) * I[t]
        right[2, t] = delta * I[t]
        right[3, t] = theta * I[t]
        cost += w ** (day_range - t) * (left[:, t] - right[:, t]) ** 2

    cost *= 1 / (day_range * 1e3)
    print(cost)
    cost = cost.sum()
    sys.stdout.write(f"GGA SUCCESS {cost}\n")

    if plot:
        plots(left, right, day_range)


def plots(left, right, day_range):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    t = np.arange(day_range)
    titles = ["Susceptible", "Infected", "Recovered", "Dead"]
    for i, title in enumerate(titles):
        plt.title(title + " change")
        plt.plot(t, left[i], c="r")
        plt.plot(t, right[i], c="b")
        plt.show()


def get_data(data, day_min, day_max):
    df = pd.read_csv(data)
    df = df.loc[day_min : day_max + 1]
    I = df.totale_positivi.values
    R = df.dimessi_guariti.values
    D = df.deceduti.values
    return I, R, D


def main(args):
    # TODO: extend to work with other models
    # Renamed nu -> theta
    sird(
        args.data,
        args.day_min,
        args.day_max,
        args.plot,
        # n=args.n,
        q=args.q,
        w=args.w,
        delta=args.delta,
        beta=args.beta,
        theta=args.theta,
    )
