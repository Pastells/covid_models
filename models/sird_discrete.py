"""
Discrete mean-field SIRD model

Pol Pastells, 2020-2021

Equations of the deterministic system:

Delta_S(t) = - beta*I(t)*S(t)/(I(t)+S(t)) \n
Delta_I(t) =   beta/N*I(t)*S(t) -(delta+nu)* I(t) \n
Delta_R(t) =                   delta * I(t) \n
Delta_D(t) =                   nu * I(t)
"""

import sys
import traceback
import argparse
import numpy as np
import pandas as pd
from utils import config


def main():
    args = parsing()
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
        right[1, t] = r - (args.delta + args.nu) * I[t]
        right[2, t] = args.delta * I[t]
        right[3, t] = args.nu * I[t]
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


def parsing():
    """input parameters"""

    parser = argparse.ArgumentParser(
        description="Discrete mean-field SIRD model",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=int(59.3e6),
        help="fixed number of (real) individuals for the country",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=config.DELTA,
        help="rate of recovery from infected phase (i->r) [0.05,1.0]",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=config.BETA,
        help="infectivity due to infected [0.05,1.0]",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=config.DELTA,
        help="death probability from infected phase (i->r) [0.05,1.0]",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.1,
        help="Ratio of the real population that is effective [0.002,1]",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=0.9,
        help="Forgetting factor, weight for cost sum (0,1]",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for the automatic configuration"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="timeout for the automatic configuration",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dpc-covid19-ita-andamento-nazionale.csv",
        help="file with time series",
    )
    parser.add_argument(
        "--day_min",
        type=int,
        default=0,
        help="first day to consider of the data series",
    )
    parser.add_argument(
        "--day_max",
        type=int,
        default=33,
        help="last day to consider of the data series",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc()
