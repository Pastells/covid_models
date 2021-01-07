"""
Stochastic mean-field SIR model.
Uses the Gillespie algorithm and Erlang distribution transition times

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""

import random
import sys
import traceback
import numpy as np
from utils import utils, config


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def main():
    args = parsing()
    # print(args)
    t_total, time_series, rates, shapes = parameters_init(args)

    # results per day and seed
    I_day, I_m = (
        np.zeros([args.mc_nseed, t_total]).astype(int),
        np.zeros(t_total),
    )

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.mc_seed0, args.mc_seed0 + args.mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)

        # -------------------------
        # initialization
        comp = Compartments(shapes, args)

        I_day[mc_step, 0] = args.I_0
        t_step, time = 0, 0

        # Time loop
        while comp.I[t_step, :-1].sum() > 0 and time < t_total:
            t_step, time = gillespie(t_step, time, comp, rates, shapes)
        # -------------------------

        if config.CUMULATIVE is True:
            i_var = comp.I_cum
        else:
            i_var = comp.I[:, :-1].sum(axis=1)

        day_max = utils.day_data(comp.T[:t_step], i_var[:t_step], I_day[mc_step])

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    if config.CUMULATIVE is True:
        utils.cost_func(time_series[:, 3], I_m, I_std)
    else:
        utils.cost_func(time_series[:, 0], I_m, I_std)

    if args.save is not None:
        utils.saving(args, I_m, I_std, day_max)

    if args.plot:
        from utils import plots

        plots.plotting(args, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic mean-field SIR model using the Gillespie algorithm and Erlang \
            distribution transition times. \
            Dependencies: config.py, utils.py",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_params = parser.add_argument_group("parameters")

    parser_params.add_argument(
        "--n",
        type=int,
        default=config.N,
        help="fixed number of (effecitve) people [1000,1000000]",
    )
    parser_params.add_argument(
        "--delta",
        type=float,
        default=config.DELTA,
        help="rate of recovery [0.05,1]",
    )
    parser_params.add_argument(
        "--k_rec",
        type=int,
        default=config.K_REC,
        help="k for the recovery time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--beta",
        type=float,
        default=config.BETA,
        help="infectivity [0.05,1]",
    )
    parser_params.add_argument(
        "--k_inf",
        type=int,
        default=config.K_INF,
        help="k for the infection time erlang distribution [1,5]",
    )

    utils.parser_common(parser)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec}
    rates = {"beta": args.beta / args.n * args.k_inf, "delta": args.delta * args.k_rec}
    return t_total, time_series, rates, shapes


# -------------------------


class Compartments:
    """Compartments for the SIR Erlang model"""

    def __init__(self, shapes, args):
        """Initialization"""
        self.S = np.zeros([args.n_t_steps, shapes["k_inf"] + 1]).astype(int)
        self.I = np.zeros([args.n_t_steps, shapes["k_rec"] + 1]).astype(int)
        self.R = np.zeros(args.n_t_steps).astype(int)
        self.T = np.zeros(args.n_t_steps)

        # Used for both sir_erlang and sir_erlang sections, where args.n is a vector
        try:
            self.S[0, :-1] = (args.n - args.I_0 - args.R_0) / shapes["k_inf"]
        except TypeError:
            self.S[0, :-1] = (args.n[0] - args.I_0 - args.R_0) / shapes["k_inf"]

        self.S[0, -1] = self.I[0, :-1] = args.I_0 / shapes["k_rec"]
        self.I[0, -1] = self.R[0] = args.R_0
        self.T[0] = 0
        self.I_cum = np.zeros(args.n_t_steps).astype(int)
        self.I_cum[0] = args.I_0

    def infect_adv_s(self, t_step, k):
        """Infect or advance in S
        S(k)-> S(k+1)/I(0)"""
        self.R[t_step] = self.R[t_step - 1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.S[t_step, k] = -1
        self.S[t_step, k + 1] = 1
        self.I[t_step, 0] += self.S[t_step, -1]
        self.S[t_step] += self.S[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + self.S[t_step, -1]

    def recover_adv_i(self, t_step, k):
        """Recover or advance in I
        I(k)-> I(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, k] = -1
        self.I[t_step, k + 1] = 1
        self.R[t_step] = self.R[t_step - 1] + self.I[t_step, -1]
        self.I[t_step] += self.I[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]


# -------------------------


def gillespie(t_step, time, comp, rates, shapes):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()

    lambda_sum = (rates["delta"] + rates["beta"] * stot) * itot
    probs = {}
    probs["heal"] = rates["delta"] * comp.I[t_step, :-1] / lambda_sum
    probs["infect"] = rates["beta"] * comp.S[t_step, :-1] * itot / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time

    gillespie_step(t_step, comp, probs, shapes)
    return t_step, time


# -------------------------


def gillespie_step(t_step, comp, probs, shapes):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    S and I have one extra dimension to temporally store the infected and recovered
    after k stages, due to the Erlang distribution
    """
    random = np.random.random()
    prob_heal_tot = probs["heal"].sum()

    # I(k)-> I(k+1)/R
    if random < prob_heal_tot:
        for k in range(shapes["k_rec"]):
            if random < probs["heal"][: k + 1].sum():
                comp.recover_adv_i(t_step, k)
                return

    # S(k)-> S(k+1)/I(0)
    for k in range(shapes["k_inf"]):
        if random < (prob_heal_tot + probs["infect"][: k + 1].sum()):
            comp.infect_adv_s(t_step, k)
            return


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
