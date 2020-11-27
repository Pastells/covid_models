"""
Stochastic mean-field SIR model
Uses the Gillespie algorithm and erlang distribution transition times
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
        shapes,
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
        comp = Compartments(n_t_steps, shapes, args)

        I_day[mc_step, 0] = I_0
        t_step, time, day = 0, 0, 1

        # Time loop
        while comp.I[t_step, :-1].sum() > 0.1 and day < t_total:
            # Add individuals periodically
            # if time//add_n==1:
            # add_n += 30
            # S[t_step] += float(n)/2
            day, day_max = utils.day_data(
                time, t_total, day, day_max, comp.I[t_step, :-1].sum(), I_day[mc_step]
            )
            t_step, time = gillespie(t_step, time, comp, ratios, shapes)
        # -------------------------

        # final value for the rest of time, otherwise it contributes with a zero when averaged
        # S_day[mc_step,day:] = S_day[mc_step,day-1]
        I_day[mc_step, day:] = I_day[mc_step, day - 1]
        # R_day[mc_step,day:] = R_day[mc_step,day-1]

        # plot all trajectories
        # if plot:
        # plt.plot(I_day[mc_step,:])
        # plt.plot(T[:t_step],i[:t_step,:-1].sum(1),c='c')
        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "sir_erlang")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="stochastic mean-field SIR model using the Gillespie algorithm and Erlang \
            distribution transition times.",
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
        default=20,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals [0,n]"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help="parameter: mean ratio of recovery [1e-2,1]",
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="parameter: ratio of infection [1e-2,1]"
    )
    parser.add_argument(
        "--k_rec",
        type=int,
        default=1,
        help="parameter: k for the recovery time erlang distribution [1,5]",
    )
    parser.add_argument(
        "--k_inf",
        type=int,
        default=1,
        help="parameter: k for the infection time erlang distribution [1,5]",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for the automatic configuration"
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
        default=int(5),
        help="number of mc realizations, not really a parameter",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="initial mc seed, not really a parameter",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument("--save", action="store_true", help="specify for outputfile")

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """Initial parameters from argparse"""
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
    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec}
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
        shapes,
    )


# -------------------------


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, n_t_steps, shapes, args):
        """Initialization"""
        self.S = np.zeros([n_t_steps, shapes["k_inf"] + 1])
        self.I = np.zeros([n_t_steps, shapes["k_rec"] + 1])
        self.R = np.zeros(n_t_steps)

        # Used for both sir_erlang and sir_erlang sections, where args.n is a vector
        try:
            self.S[0, :-1] = (args.n - args.I_0 - args.R_0) / shapes["k_inf"]
        except TypeError:
            self.S[0, :-1] = (args.n[0] - args.I_0 - args.R_0) / shapes["k_inf"]

        self.S[0, -1] = self.I[0, :-1] = args.I_0 / shapes["k_rec"]
        self.I[0, -1] = self.R[0] = args.R_0

    def infect_adv_s(self, t_step, shapes, k):
        """Infect or advance in S
        S(k)-> S(k+1)/I(0)"""
        self.R[t_step] = self.R[t_step - 1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.S[t_step, k] = -1
        self.S[t_step, k + 1] = 1
        self.I[t_step, 0] += self.S[t_step, shapes["k_inf"]]
        self.S[t_step] += self.S[t_step - 1]

    def recover_adv_i(self, t_step, shapes, k):
        """Recover or advance in I
        I(k)-> I(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, k] = -1
        self.I[t_step, k + 1] = 1
        self.R[t_step] = self.R[t_step - 1] + self.I[t_step, shapes["k_rec"]]
        self.I[t_step] += self.I[t_step - 1]


# -------------------------


def gillespie(t_step, time, comp, ratios, shapes):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()

    lambda_sum = (ratios["delta"] + ratios["beta"] * stot) * itot
    prob_heal = ratios["delta"] * comp.I[t_step, :-1] / lambda_sum
    prob_infect = ratios["beta"] * comp.S[t_step, :-1] * itot / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    # T[t_step] = time

    gillespie_step(t_step, comp, prob_heal, prob_infect, shapes)
    return t_step, time


# -------------------------


def gillespie_step(t_step, comp, prob_heal, prob_infect, shapes):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    S and I have one extra dimension to temporally store the infected and recovered
    after k stages, due to the Erlang distribution
    """
    random = np.random.random()
    prob_heal_tot = prob_heal.sum()

    # I(k)-> I(k+1)/R"""
    if random < prob_heal_tot:
        for k in range(shapes["k_rec"]):
            if random < prob_heal[: k + 1].sum():
                comp.recover_adv_i(t_step, shapes, k)
                break

    # S(k)-> S(k+1)/I(0)
    else:
        for k in range(shapes["k_inf"]):
            if random < (prob_heal_tot + prob_infect[: k + 1].sum()):
                comp.infect_adv_s(t_step, shapes, k)
                break


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
