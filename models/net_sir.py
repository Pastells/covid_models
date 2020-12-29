"""
Stochastic SIR model with a social network using the event-driven algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta*I(t)*S(t) \n
dI(t)/dt =   beta*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                    delta * I(t)
"""

import random
import sys
import traceback
import numpy as np
import fast_sir
from utils import utils, utils_net, config


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

        G = utils_net.choose_network(args.n, args.network, args.network_param)
        t, S, I, R = fast_sir.fast_SIR(G, ratios, args.I_0, args.R_0)

        I_day[mc_step, 0] = args.I_0
        day = 1
        for t, time in enumerate(t):
            day, day_max = utils.day_data(
                time, t_total, day, day_max, I[t], I_day[mc_step]
            )

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if args.save is not None:
        utils.saving(args, I_m, I_std, day_max, "net_sir", args.save)

    if args.plot:
        from utils import plots

        plots.plotting(args, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic SIR model with a social network using the event-driven algorithm. \
                Dependencies: config.py, utils.py, utils_net.py, fast_sir.py",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_params = parser.add_argument_group("parameters")

    parser_params.add_argument(
        "--network",
        type=str,
        default=config.NETWORK,
        choices=["er", "ba"],
        help="Erdos-Renyi or Barabasi Albert {er,ba}",
    )
    parser_params.add_argument(
        "--network_param",
        type=int,
        default=config.NETWORK_PARAM,
        help="mean number of edges [1,50]",
    )

    parser_params.add_argument(
        "--n",
        type=int,
        default=config.N,
        help="fixed number of (effecitve) people [1000,1000000]",
    )

    parser_params.add_argument(
        "--delta", type=float, default=config.DELTA, help="ratio of recovery [0.05,1]"
    )
    parser_params.add_argument(
        "--beta", type=float, default=config.BETA, help="ratio of infection [0.05,1]"
    )

    utils.parser_common(parser)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, infected_time_series = utils.parameters_init_common(args)

    ratios = {"beta": args.beta, "delta": args.delta}
    return t_total, infected_time_series, ratios


# -------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
