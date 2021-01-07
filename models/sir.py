"""
Stochastic mean-field SIR model using the Gillespie algorithm

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
    t_total, time_series, rates = parameters_init(args)
    # print(args)

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
        comp = Compartments(args)

        I_day[mc_step, 0] = args.I_0
        # S_day[mc_step,0]=s[0]
        t_step, time = 0, 0

        # Time loop
        while comp.I[t_step] > 0 and time < t_total:
            t_step, time = gillespie(t_step, time, comp, rates)
        # -------------------------

        if config.CUMULATIVE is True:
            i_var = comp.I_cum
        else:
            i_var = comp.I

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

        plots.plotting(args, I_day, day_max, I_m, I_std)  # , comp=comp, t_step=t_step)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic mean-field SIR model using the Gillespie algorithm. \
            Dependencies: config.py, utils.py",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--beta",
        type=float,
        default=config.BETA,
        help="infectivity [0.05,1]",
    )

    utils.parser_common(parser)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta / args.n, "delta": args.delta}
    return t_total, time_series, rates


# -------------------------


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, args):
        """Initialization"""
        self.S = np.zeros(args.n_t_steps).astype(int)
        self.I = np.zeros(args.n_t_steps).astype(int)
        self.R = np.zeros(args.n_t_steps).astype(int)
        self.T = np.zeros(args.n_t_steps)
        self.I[0] = args.I_0
        self.R[0] = args.R_0
        self.S[0] = args.n - args.I_0 - args.R_0
        self.T[0] = 0
        self.I_cum = np.zeros(args.n_t_steps).astype(int)
        self.I_cum[0] = args.I_0

    def infect(self, t_step):
        """Infection"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + 1

    def recover(self, t_step):
        """Recovery"""
        self.S[t_step] = self.S[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]


# -------------------------


def gillespie(t_step, time, comp, rates):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    lambda_sum = (rates["delta"] + rates["beta"] * comp.S[t_step]) * comp.I[t_step]
    prob_heal = rates["delta"] * comp.I[t_step] / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time

    gillespie_step(t_step, comp, prob_heal)
    return t_step, time


# -------------------------


def gillespie_step(t_step, comp, prob_heal):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < prob_heal:
        comp.recover(t_step)
    else:
        comp.infect(t_step)


# ~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
