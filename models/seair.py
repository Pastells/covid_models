"""
Stochastic mean-field SAIR model using the Gillespie algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta/N*I(t)*S(t) \n
dE(t)/dt =   beta_a/N*A(t)*S(t) + beta/N*I(t)*S(t) - epsilon*E(t)\n
dA(t)/dt =   epsilon * E(t)   -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta * I(t)  + alpha*A(t)\n
dR(t)/dt =   delta * I(t)  + delta_a * A(t)
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
    I_day = np.zeros([args.mc_nseed, t_total], dtype=int)

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
        t_step, time = 0, 0

        # -------------------------
        # Time loop
        # -------------------------
        while comp.I[t_step] > 0 and time < t_total:
            t_step, time = gillespie(t_total, t_step, time, comp, rates)
        # -------------------------

        if config.CUMULATIVE is True:
            i_var = comp.I_cum
        else:
            i_var = comp.I

        day_max = utils.day_data(
            comp.T[:t_step], i_var[:t_step], I_day[mc_step], day_max
        )

        mc_step += 1
    # =========================

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# -------------------------


def parsing():
    """input parameters"""
    description = "stochastic mean-field SAIR model using the Gillespie algorithm. \
        Dependencies: config.py, utils.py"

    parser = utils.ParserCommon(description)
    parser.n()
    parser.sir()
    parser.exposed()
    parser.asymptomatic()

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {
        "beta_a": args.beta_a / args.n,
        "beta": args.beta / args.n,
        "delta_a": args.delta_a,
        "delta": args.delta,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
    }

    return t_total, time_series, rates


# -------------------------


class Compartments:
    """Compartments for SAIR model"""

    def __init__(self, args):
        """Initialization"""
        self.S = np.zeros(args.n_t_steps, dtype=int)
        self.E = np.zeros(args.n_t_steps, dtype=int)
        self.A = np.zeros(args.n_t_steps, dtype=int)
        self.I = np.zeros(args.n_t_steps, dtype=int)
        self.R = np.zeros(args.n_t_steps, dtype=int)
        self.T = np.zeros(args.n_t_steps)
        self.E[0] = args.E_0
        self.A[0] = args.A_0
        self.I[0] = args.I_0
        self.R[0] = args.R_0
        self.S[0] = args.n - args.I_0 - args.R_0 - args.A_0 - args.E_0
        self.T[0] = 0
        self.I_cum = np.zeros(args.n_t_steps, dtype=int)
        self.I_cum[0] = args.I_0

    def turn_exposed(self, t_step):
        """Expose s->e"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.E[t_step] = self.E[t_step - 1] + 1
        self.A[t_step] = self.A[t_step - 1]
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def turn_asymptomatic(self, t_step):
        """Turn asymptomatic e->a"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1] - 1
        self.A[t_step] = self.A[t_step - 1] + 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def turn_infectious(self, t_step):
        """Turn infectious a->i"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + 1

    def recover_a(self, t_step):
        """Recovery a->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def recover_i(self, t_step):
        """Recovery i->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.A[t_step] = self.A[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]


# -------------------------


def gillespie(t_total, t_step, time, comp, rates):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (rates["alpha"] + rates["delta_a"]) * comp.A[t_step]
        + rates["epsilon"] * comp.E[t_step]
        + rates["delta"] * comp.I[t_step]
        + (rates["beta_a"] * comp.A[t_step] + rates["beta"] * comp.I[t_step])
        * comp.S[t_step]
    )

    probs = {}
    probs["heal_a"] = rates["delta_a"] * comp.A[t_step] / lambda_sum
    probs["heal_i"] = rates["delta"] * comp.I[t_step] / lambda_sum
    probs["asymptomatic"] = rates["alpha"] * comp.A[t_step] / lambda_sum
    probs["exposed"] = rates["epsilon"] * comp.E[t_step] / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time

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
    elif random < (
        probs["heal_a"] + probs["heal_i"] + probs["asymptomatic"] + probs["exposed"]
    ):
        comp.turn_asymptomatic(t_step)
    else:
        comp.turn_exposed(t_step)


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")