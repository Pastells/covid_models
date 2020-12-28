"""
Stochastic mean-field SEIR model using the Gillespie algorithm
Pol Pastells, october 2020

equations of the deterministic system
s[t] = S[t-1] - beta_e*e[t-1]*s[t-1] - beta_i*i[t-1]*s[t-1]
e[t] = E[t-1] + beta_e*e[t-1]*s[t-1] + beta_i*i[t-1]*s[t-1] - (epsilon+delta_e)*e[t-1]
i[t] = I[t-1] + epsilon*e[t-1] - delta_i * I[t-1]
r[t] = R[t-1] + delta_e *e[t-1] + delta_i * I[t-1]
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
    t_total, infected_time_series, ratios = parameters_init(args)

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
            t_step, time = gillespie(t_total, t_step, time, comp, ratios)

        # -------------------------

        mc_step += 1
    # =========================
    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if args.save is not None:
        utils.saving(args, I_m, I_std, day_max, "seir", args.save)

    if args.plot:
        from utils import plots

        plots.plotting(args, I_day, day_max, I_m, I_std)


# -------------------------


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic mean-field SEIR model using the Gillespie algorithm. \
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
        "--delta_e",
        type=float,
        default=config.DELTA_E,
        help="ratio of recovery from latent fase (e->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta_i",
        type=float,
        default=config.DELTA,
        help="ratio of recovery from infected fase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--beta_e",
        type=float,
        default=config.BETA_E,
        help="ratio of infection due to latent [0.05,1]",
    )
    parser_params.add_argument(
        "--beta_i",
        type=float,
        default=config.BETA,
        help="ratio of infection due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--epsilon",
        type=float,
        default=config.EPSILON,
        help="ratio of latency (e->i) [0.05,2]",
    )

    utils.parser_common(parser, True)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, infected_time_series = utils.parameters_init_common(args)

    ratios = {
        "beta_e": args.beta_e / args.n,
        "beta_i": args.beta_i / args.n,
        "delta_e": args.delta_e,
        "delta_i": args.delta_i,
        "epsilon": args.epsilon,
    }

    return t_total, infected_time_series, ratios


# -------------------------


class Compartments:
    """Compartments for SEIR model"""

    def __init__(self, args):
        """Initialization"""
        self.S = np.zeros(args.n_t_steps)
        self.E = np.zeros(args.n_t_steps)
        self.I = np.zeros(args.n_t_steps)
        self.R = np.zeros(args.n_t_steps)
        self.E[0] = args.E_0
        self.I[0] = args.I_0
        self.R[0] = args.R_0
        self.S[0] = args.n - args.I_0 - args.R_0 - args.E_0

    def turn_latent(self, t_step):
        """Infection s->e"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.E[t_step] = self.E[t_step - 1] + 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]

    def turn_infectious(self, t_step):
        """Turn infectious e->i"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]

    def recover1(self, t_step):
        """Recovery e->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1] + 1

    def recover2(self, t_step):
        """Recovery i->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1


# -------------------------


def gillespie(t_total, t_step, time, comp, ratios):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (ratios["epsilon"] + ratios["delta_e"]) * comp.E[t_step]
        + ratios["delta_i"] * comp.I[t_step]
        + (ratios["beta_e"] * comp.E[t_step] + ratios["beta_i"] * comp.I[t_step])
        * comp.S[t_step]
    )

    probs = {}
    probs["heal1"] = ratios["delta_e"] * comp.E[t_step] / lambda_sum
    probs["heal2"] = ratios["delta_i"] * comp.I[t_step] / lambda_sum
    probs["latent"] = ratios["epsilon"] * comp.E[t_step] / lambda_sum

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

    if random < probs["heal1"]:
        comp.recover1(t_step)
    elif random < (probs["heal1"] + probs["heal2"]):
        comp.recover2(t_step)
    elif random < (probs["heal1"] + probs["heal2"] + probs["latent"]):
        comp.turn_infectious(t_step)
    else:
        comp.turn_latent(t_step)


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
