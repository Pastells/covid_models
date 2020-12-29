"""
Stochastic SEIR model with a social network using the event-driven algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta_e*E(t)*S(t) - beta_i*I(t)*S(t) \n
dE(t)/dt =   beta_e*E(t)*S(t) + beta_i*I(t)*S(t) -(epsilon+delta_e)*E(t)\n
dI(t)/dt = - delta_i * I(t)                      + epsilon*E(t)\n
dR(t)/dt =   delta_i * I(t)                      + delta_e * E(t)
"""

import random
import sys
import traceback
import numpy as np
import fast_seir
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
        t, S, E, I, R = fast_seir.fast_SEIR(
            G, ratios, args.E_0, args.I_0, args.R_0, tmax=t_total - 0.9
        )
        import matplotlib.pyplot as plt

        plt.plot(t, S)
        plt.plot(t, E)
        plt.plot(t, I)
        plt.plot(t, R)

        I_day[mc_step, 0] = args.I_0
        day = 1
        for t, time in enumerate(t):
            day, day_max = utils.day_data(
                time, t_total, day, day_max, I[t], I_day[mc_step]
            )
        day, day_max = utils.day_data(
            time, t_total, day, day_max, I[-1], I_day[mc_step], True
        )

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if args.save is not None:
        utils.saving(args, I_m, I_std, day_max, "net_seir", args.save)

    if args.plot:
        from utils import plots

        plots.plotting(args, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic SEIR model using the Gillespie algorithm. \
                Dependencies: config.py, utils.py, utils_net.py, fast_seir.py",
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
        "beta_e": args.beta_e,
        "beta_i": args.beta_i,
        "delta_e": args.delta_e,
        "delta_i": args.delta_i,
        "epsilon": args.epsilon,
    }

    return t_total, infected_time_series, ratios


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
