"""
Stochastic sir model with a social network using the event-driven algorithm
Pol Pastells, november 2020

equations of the deterministic system
s[t] = S[t-1] - beta*i[t-1]*s[t-1]
i[t] = I[t-1] + beta*i[t-1]*s[t-1] - delta * I[t-1]
r[t] = R[t-1] + delta * I[t-1]

to do:
    - add network parameters in parser
    - seed for network generation
"""

import random
import sys
import traceback
import numpy as np
import utils


def main():
    args = parsing()
    (
        E_0,
        I_0,
        R_0,
        t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        beta1,
        beta2,
        delta1,
        delta2,
        epsilon,
        network_type,
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
    for seed in range(mc_seed0, mc_seed0 + mc_nseed):
        random.seed(seed)
        np.random.seed(seed)

        G = utils.choose_network(n, network_type, network_param)
        t, S, E, I, R = fast_seir.fast_SEIR(
            G, beta1, beta2, delta1, delta2, epsilon, I_0, R_0, tmax=t_total - 0.9
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
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="stochastic SEIR model using the Gillespie algorithm",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="parameter: fixed number of (effecitve) people [1000,1000000]",
    )

    parser.add_argument(
        "--network_type",
        type=str,
        default="er",
        choices=["er", "ba"],
        help="parameter: Erdos-Renyi or Barabasi Albert supported right now ['er','ba']",
    )
    parser.add_argument(
        "--network_param",
        type=int,
        default=5,
        help="parameter: mean number of edges [1,50]",
    )

    parser.add_argument(
        "--e_0", type=int, default=0, help="initial number of latent individuals [1,n]"
    )
    parser.add_argument(
        "--i_0",
        type=int,
        default=20,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--r_0", type=int, default=0, help="initial number of inmune individuals [0,n]"
    )
    parser.add_argument(
        "--delta1",
        type=float,
        default=0.01,
        help="parameter: ratio of recovery from latent fase (e->r) [1e-2,1]",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.2,
        help="parameter: ratio of recovery from infected fase (i->r) [1e-2,1]",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.01,
        help="parameter: ratio of infection due to latent [1e-2,1]",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.5,
        help="parameter: ratio of infection due to infected [1e-2,1]",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1,
        help="parameter: ratio of latency (e->i) [1e-2,1]",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for the automatic configuration"
    )
    parser.add_argument(
        "--data", type=str, default="../data/italy_i.csv", help="file with time series"
    )
    parser.add_argument(
        "--day_min", type=int, default=33, help="first day to consider on data series"
    )
    parser.add_argument(
        "--day_max", type=int, default=58, help="last day to consider on data series"
    )

    parser.add_argument(
        "--mc_nseed",
        type=int,
        default=int(1e3),
        help="number of mc realizations",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="initial mc seed",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument(
        "--save", type=str, default=None, help="specify a name for outputfile"
    )

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    from numpy import genfromtxt

    E_0 = args.e_0
    I_0 = args.i_0
    R_0 = args.r_0
    t_steps = int(1e7)  # max simulation steps
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
    beta1 = args.beta1
    beta2 = args.beta2
    delta1 = args.delta1
    delta2 = args.delta2
    epsilon = args.epsilon
    network_type = args.network_type
    network_param = args.network_param

    return (
        E_0,
        I_0,
        R_0,
        t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        beta1,
        beta2,
        delta1,
        delta2,
        epsilon,
        network_type,
        network_param,
    )


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
