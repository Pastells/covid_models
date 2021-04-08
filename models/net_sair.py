"""

Stochastic SAIR model with a social network using the event-driven algorithm

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta/N*I(t)*S(t) \n
dA(t)/dt =   beta_a/N*A(t)*S(t) + beta/N*I(t)*S(t) -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta * I(t)                          + alpha*A(t)\n
dR(t)/dt =   delta * I(t)                          + delta_a * A(t)
"""

import random
import sys
import traceback
import numpy as np
from event_driven import fast_sair
from utils import utils, utils_net, config


def main():
    args = parsing()
    t_total, time_series, rates = parameters_init(args)
    # print(args)

    # results per day and seed
    I_day = np.zeros([args.mc_nseed, t_total], dtype=int)

    day_max = 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.seed, args.seed + args.mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)
        mc_step = mc_seed - args.seed

        G = utils_net.choose_network(args.n, args.network, args.network_param)
        t, S, A, I, R = fast_sair.fast_SAIR(
            G, rates, args.A_0, args.I_0, args.R_0, tmax=t_total - 0.95
        )

        import matplotlib.pyplot as plt

        plt.plot(t, S)
        plt.plot(t, A)
        plt.plot(t, I)
        plt.plot(t, R)

        I_day[mc_step, 0] = args.I_0

        if config.CUMULATIVE is True:
            i_var = I + R
        else:
            i_var = I

        day_max = utils.day_data(t, i_var, I_day[mc_step], day_max)

    # =========================

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    description = "stochastic SAIR model using the Gillespie algorithm. \
            Dependencies: config.py, utils.py, utils_net.py, fast_sair.py"

    parser = utils.ParserCommon(description)
    parser.n()
    parser.sir()
    parser.asymptomatic()
    parser.network()

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {
        "beta_a": args.beta_a,
        "beta": args.beta,
        "delta_a": args.delta_a,
        "delta": args.delta,
        "alpha": args.alpha,
    }

    return t_total, time_series, rates


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc(ex)
