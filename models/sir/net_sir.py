"""
Stochastic SIR model with a social network using the event-driven algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""

import random
import numpy as np

from . import fast_sir
from ..utils import utils, utils_net, config


def main(args):
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
        t, S, I, R = fast_sir.fast_SIR(
            G, rates, args.initial_infected, args.initial_recovered, tmax=t_total - 0.95
        )

        I_day[mc_step, 0] = args.initial_infected

        if config.CUMULATIVE is True:
            i_var = I + R
        else:
            i_var = I

        day_max = utils.day_data(t, i_var, I_day[mc_step], day_max)

    # =========================

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta, "delta": args.delta}
    return t_total, time_series, rates
