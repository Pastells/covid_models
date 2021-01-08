"""
Stochastic SAIR model with a social network using the event-driven algorithm

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
from event_driven import fast_sair
from utils import utils, utils_net, config


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

        day_max = utils.day_data(t, i_var, I_day[mc_step])

        mc_step += 1
    # =========================

    I_m = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    if config.CUMULATIVE is True:
        utils.cost_func(time_series[:, 3], I_m)
    else:
        utils.cost_func(time_series[:, 0], I_m)

    if args.save is not None:
        utils.saving(args, I_m, day_max)

    if args.plot:
        from utils import plots

        plots.plotting(args, day_max, I_m)  # , comp=comp, t_step=t_step)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic SAIR model using the Gillespie algorithm. \
                Dependencies: config.py, utils.py, utils_net.py, fast_sair.py",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_params = parser.add_argument_group("parameters")

    parser_params.add_argument(
        "--network",
        type=str,
        choices=["er", "ba"],
        default=config.NETWORK,
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
    t_total, time_series = utils.parameters_init_common(args)

    rates = {
        "beta_a": args.beta_a,
        "beta_i": args.beta_i,
        "delta_a": args.delta_a,
        "delta_i": args.delta_i,
        "alpha": args.alpha,
    }

    return t_total, time_series, rates


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
