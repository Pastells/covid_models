"""
Stochastic mean-field SIR model using the Gillespie algorithm
Pol Pastells, october 2020

equations of the deterministic system
s[t] = S[t-1] - beta*i[t-1]*s[t-1]
i[t] = I[t-1] + beta*i[t-1]*s[t-1] - delta * I[t-1]
r[t] = R[t-1] + delta * I[t-1]
"""

import random
import sys
import traceback
import numpy as np
import utils

# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def main():
    args = parsing()
    (
        I_0,
        R_0,
        n_t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        ratios,
    ) = parameters_init(args)

    # results per day and seed
    I_day, I_m = (
        np.zeros([mc_nseed, t_total]),
        np.zeros(t_total),
    )

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for seed in range(mc_seed0, mc_seed0 + mc_nseed):
        random.seed(seed)
        np.random.seed(seed)

        # -------------------------
        # initialization

        comp = Compartments(n_t_steps, args)

        I_day[mc_step, 0] = I_0
        # T = np.zeros(n_t_steps)
        # T[0]=0
        # S_day[mc_step,0]=s[0]
        t_steps, time, day = 0, 0, 1

        # Time loop
        while comp.I[t_steps] > 0.1 and day < t_total:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, comp.I[t_steps], I_day[mc_step]
            )
            t_steps, time = gillespie(t_steps, time, comp, ratios)
        # -------------------------

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "sir")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="stochastic mean-field SIR model using the Gillespie algorithm",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="parameter: fixed number of (effecitve) people [1000,1000000]",
    )
    parser.add_argument(
        "--I_0",
        type=int,
        default=50,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals [0,n]"
    )
    parser.add_argument(
        "--delta", type=float, default=0.2, help="parameter: ratio of recovery [1e-2,1]"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="parameter: ratio of infection [1e-2,1]"
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for the automatic configuration"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="timeout for the automatic configuration",
    )
    parser.add_argument(
        "--data", type=str, default="../data/italy_i.csv", help="file with time series"
    )
    parser.add_argument(
        "--day_min", type=int, default=33, help="first day to consider on data series"
    )
    parser.add_argument(
        "--day_max", type=int, default=58, help="last day to consider on data series"
    )

    parser.add_argument(
        "--mc_nseed",
        type=int,
        default=int(10),
        help="number of mc realizations",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="initial mc seed",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument("--save", action="store_true", help="specify for outputfile")

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    from numpy import genfromtxt

    I_0 = args.I_0
    R_0 = args.R_0
    n_t_steps = int(1e7)  # max simulation steps
    t_total = args.day_max - args.day_min  # max simulated days
    mc_nseed = args.mc_nseed  # MC realizations
    mc_seed0 = args.mc_seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    # print(infected_time_series)
    n = args.n
    ratios = {"beta": args.beta / n, "delta": args.delta}
    return (
        I_0,
        R_0,
        n_t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        ratios,
    )


# -------------------------


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, n_t_steps, args):
        """Initialization"""
        self.S = np.zeros(n_t_steps)
        self.I = np.zeros(n_t_steps)
        self.R = np.zeros(n_t_steps)
        self.I[0] = args.I_0
        self.R[0] = args.R_0
        self.S[0] = args.n - args.I_0 - args.R_0

    def infect(self, t_step):
        """Infection"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]

    def recover(self, t_step):
        """Recovery"""
        self.S[t_step] = self.S[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1


# -------------------------


def gillespie(t_steps, time, comp, ratios):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    lambda_sum = (ratios["delta"] + ratios["beta"] * comp.S[t_steps]) * comp.I[t_steps]
    prob_heal = ratios["delta"] * comp.I[t_steps] / lambda_sum

    t_steps += 1
    time += utils.time_dist(lambda_sum)
    # T[t_steps] = time

    gillespie_step(t_steps, comp, prob_heal)
    return t_steps, time


# -------------------------


def gillespie_step(t_steps, comp, prob_heal):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < prob_heal:
        comp.recover(t_steps)
    else:
        comp.infect(t_steps)


# ~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
