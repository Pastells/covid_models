"""
Stochastic mean-field SAIR model using the Gillespie algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta_i/N*I(t)*S(t) \n
dA(t)/dt =   beta_a/N*A(t)*S(t) + beta_i/N*I(t)*S(t) -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta_i * I(t)                          + alpha*A(t)\n
dR(t)/dt =   delta_i * I(t)                          + delta_a * A(t)
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
    t_total, infected_time_series, rates = parameters_init(args)
    # print(args)

    # results per day and seed
    I_day, I_m = (
        np.zeros([args.mc_nseed, t_total]),
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

        # T = np.zeros(n_t_steps)
        # T[0]=0
        I_day[mc_step, 0] = args.I_0
        t_step, time, day = 0, 0, 1

        # -------------------------
        # Time loop
        # -------------------------
        while comp.I[t_step] > 0.1 and day < t_total:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, comp.I[t_step], I_day[mc_step]
            )
            t_step, time = gillespie(t_total, t_step, time, comp, rates)

        # -------------------------

        mc_step += 1
    # =========================
    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if args.save is not None:
        utils.saving(args, I_m, I_std, day_max)

    if args.plot:
        from utils import plots

        plots.plotting(args, I_day, day_max, I_m, I_std)


# -------------------------


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic mean-field SAIR model using the Gillespie algorithm. \
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
        "--delta_a",
        type=float,
        default=config.DELTA_A,
        help="rate of recovery from asymptomatic phase (a->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta_i",
        type=float,
        default=config.DELTA,
        help="rate of recovery from infected phase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--beta_a",
        type=float,
        default=config.BETA_A,
        help="infectivity due to asymptomatic [0.05,1]",
    )
    parser_params.add_argument(
        "--beta_i",
        type=float,
        default=config.BETA,
        help="infectivity due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--alpha",
        type=float,
        default=config.ALPHA,
        help="asymptomatic rate (a->i) [0.05,2]",
    )

    utils.parser_common(parser, True)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, infected_time_series = utils.parameters_init_common(args)

    rates = {
        "beta_a": args.beta_a / args.n,
        "beta_i": args.beta_i / args.n,
        "delta_a": args.delta_a,
        "delta_i": args.delta_i,
        "alpha": args.alpha,
    }

    return t_total, infected_time_series, rates


# -------------------------


class Compartments:
    """Compartments for SAIR model"""

    def __init__(self, args):
        """Initialization"""
        self.S = np.zeros(args.n_t_steps)
        self.A = np.zeros(args.n_t_steps)
        self.I = np.zeros(args.n_t_steps)
        self.R = np.zeros(args.n_t_steps)
        self.A[0] = args.A_0
        self.I[0] = args.I_0
        self.R[0] = args.R_0
        self.S[0] = args.n - args.I_0 - args.R_0 - args.A_0

    def turn_asymptomatic(self, t_step):
        """Turn asympomatic s->a"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.A[t_step] = self.A[t_step - 1] + 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]

    def turn_infectious(self, t_step):
        """Turn infectious a->i"""
        self.S[t_step] = self.S[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]

    def recover_a(self, t_step):
        """Recovery a->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1] + 1

    def recover_i(self, t_step):
        """Recovery i->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.A[t_step] = self.A[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1


# -------------------------


def gillespie(t_total, t_step, time, comp, rates):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (rates["alpha"] + rates["delta_a"]) * comp.A[t_step]
        + rates["delta_i"] * comp.I[t_step]
        + (rates["beta_a"] * comp.A[t_step] + rates["beta_i"] * comp.I[t_step])
        * comp.S[t_step]
    )

    probs = {}
    probs["heal_a"] = rates["delta_a"] * comp.A[t_step] / lambda_sum
    probs["heal_i"] = rates["delta_i"] * comp.I[t_step] / lambda_sum
    probs["asymptomatic"] = rates["alpha"] * comp.A[t_step] / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    # T[t_step] = time

    gillespie_step(t_step, comp, probs)
    return t_step, time


# -------------------------


def gillespie_step(t_step, comp, probs):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < probs["heal_a"]:
        comp.recover_a(t_step)
    elif random < (probs["heal_a"] + probs["heal_i"]):
        comp.recover_i(t_step)
    elif random < (probs["heal_a"] + probs["heal_i"] + probs["asymptomatic"]):
        comp.turn_infectious(t_step)
    else:
        comp.turn_asymptomatic(t_step)


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
