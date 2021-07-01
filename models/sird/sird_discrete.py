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
import traceback
import argparse
import numpy as np
import pandas as pd
from ..utils import config


def sird(args):
    I, R, D = get_data(args.data, args.day_min, args.day_max)
    S = args.q * args.n - I - R - D
    data = np.array([S, I, R, D])

    if args.plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()

    if args.q < np.max(I + R + D) / args.n:
        raise ValueError("q is too small")

    cost = np.zeros(4)
    day_range = 1 + args.day_max - args.day_min
    left = np.zeros([4, day_range])
    right = np.zeros([4, day_range])
    for t in range(day_range):
        left[:, t] = data[:, t + 1] - data[:, t]
        r = args.beta * S[t] * I[t] / (S[t] + I[t])
        right[0, t] = -r
        right[1, t] = r - (args.delta + args.theta) * I[t]
        right[2, t] = args.delta * I[t]
        right[3, t] = args.theta * I[t]
        cost += args.w ** (day_range - t) * (left[:, t] - right[:, t]) ** 2

    cost *= 1 / (day_range * 1e3)
    print(cost)
    cost = cost.sum()
    sys.stdout.write(f"GGA SUCCESS {cost}\n")

    if args.plot:
        t = np.arange(day_range)
        plt.title("Susceptible change")
        plt.plot(t, left[0], c="r")
        plt.plot(t, right[0], c="b")
        plt.show()
        plt.title("Infected change")
        plt.plot(t, left[1], c="r")
        plt.plot(t, right[1], c="b")
        plt.show()
        plt.title("Recovered change")
        plt.plot(t, left[2], c="r")
        plt.plot(t, right[2], c="b")
        plt.show()
        plt.title("Dead change")
        plt.plot(t, left[3], c="r")
        plt.plot(t, right[2], c="b")
        plt.show()


def get_data(data, day_min, day_max):
    df = pd.read_csv(data)
    df = df.loc[day_min : day_max + 1]
    I = df.totale_positivi.values
    R = df.dimessi_guariti.values
    D = df.deceduti.values
    return I, R, D


def main(args):
    # TODO: initial N is 59.3e6
    # TODO: extend to work with other models
    # TODO: add optilog integration
    # Renamed nu -> theta
    sird(args)
