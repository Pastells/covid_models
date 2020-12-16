"""
Stochastic sir model with a social network using the event-driven algorithm
Pol Pastells, november 2020

equations of the deterministic system
s[t] = S[t-1] - beta*i[t-1]*s[t-1]
i[t] = I[t-1] + beta*i[t-1]*s[t-1] - delta * I[t-1]
r[t] = R[t-1] + delta * I[t-1]
"""

import random
import sys
import traceback
import numpy as np
import utils
import utils_net
import plots
import fast_seir


def main():
    args = parsing()
    (
        E_0,
        I_0,
        R_0,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        ratios,
        network,
        network_param,
    ) = parameters_init(args)

    # results per day and seed
    I_day, I_m = (
        np.zeros([mc_nseed, t_total]),
        np.zeros(t_total),
    )

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(mc_seed0, mc_seed0 + mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)

        G = utils_net.choose_network(n, network, network_param)
        t, S, E, I, R = fast_seir.fast_SEIR(
            G, ratios, E_0, I_0, R_0, tmax=t_total - 0.9
        )
        import matplotlib.pyplot as plt

        plt.plot(t, S)
        plt.plot(t, E)
        plt.plot(t, I)
        plt.plot(t, R)

        I_day[mc_step, 0] = I_0
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

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save is not None:
        utils.saving(args, I_m, I_std, day_max, "net_sir", save)

    if plot:
        plots.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic SEIR model using the Gillespie algorithm",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_init = parser.add_argument_group("initial conditions")
    parser_params = parser.add_argument_group("parameters")

    parser_init.add_argument(
        "--E_0", type=int, default=0, help="initial number of latent individuals"
    )
    parser_init.add_argument(
        "--I_0", type=int, default=20, help="initial number of infected individuals"
    )
    parser_init.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals"
    )

    parser_params.add_argument(
        "--network",
        type=str,
        default="er",
        choices=["er", "ba"],
        help="Erdos-Renyi or Barabasi Albert {er,ba}",
    )
    parser_params.add_argument(
        "--network_param",
        type=int,
        default=5,
        help="mean number of edges [1,50]",
    )

    parser_params.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="fixed number of (effecitve) people [1000,1000000]",
    )
    parser_params.add_argument(
        "--delta1",
        type=float,
        default=0.01,
        help="ratio of recovery from latent fase (e->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta2",
        type=float,
        default=0.2,
        help="ratio of recovery from infected fase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--beta1",
        type=float,
        default=0.01,
        help="ratio of infection due to latent [0.05,1]",
    )
    parser_params.add_argument(
        "--beta2",
        type=float,
        default=0.5,
        help="ratio of infection due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--epsilon",
        type=float,
        default=1,
        help="ratio of latency (e->i) [0.05,2]",
    )

    utils.parser_common(parser)

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    from numpy import genfromtxt

    E_0 = args.E_0
    I_0 = args.I_0
    R_0 = args.R_0
    t_total = args.day_max - args.day_min  # max simulated days
    mc_nseed = args.mc_nseed  # MC realizations
    mc_seed0 = args.mc_seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    # print(infected_time_series)
    n = args.n
    ratios = {
        "beta1": args.beta1,
        "beta2": args.beta2,
        "delta1": args.delta1,
        "delta2": args.delta2,
        "epsilon": args.epsilon,
    }
    network = args.network
    network_param = args.network_param

    return (
        E_0,
        I_0,
        R_0,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        ratios,
        network,
        network_param,
    )


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
