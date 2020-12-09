"""
Stochastic SIR model with a social network using the event-driven algorithm
It allows for different sections with different n, delta and beta
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
import fast_sir_sections


def main():
    args = parsing()
    (
        I_0,
        R_0,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        network_type,
        network_param,
        n_sections,
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

        # initialization
        section = 0
        (
            n,
            ratios,
            section_day,
            ratios_old,
            section_day_old,
        ) = parameters_section(args, section)

        G = utils.choose_network(n, network_type, network_param)

        t_all, S_all, I_all, R_all = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        I_day[mc_step, 0] = I_0

        # Sections
        while section < n_sections:
            t, S, I, R = fast_sir_sections.fast_SIR(
                G,
                ratios,
                ratios_old,
                section_day_old,
                I_0,
                R_0,
                tmin=section_day_old - 1,
                tmax=section_day,
            )
            t_all = np.append(t_all, t)
            S_all = np.append(S_all, S)
            I_all = np.append(I_all, I)
            R_all = np.append(R_all, R)
            section += 1
            if section < n_sections:
                (
                    n,
                    ratios,
                    section_day,
                    ratios_old,
                    section_day_old,
                ) = parameters_section(args, section, ratios, section_day)
                if section == n_sections - 1:
                    section_day -= 0.9
                G = utils.choose_network(n, network_type, network_param)
                I_0 = I[-1]
                print(R_0)
                R_0 += R[-1]
                print(R_0)

        day = 1
        for t_step, time in enumerate(t_all):
            day, day_max = utils.day_data(
                time, t_total, day, day_max, I_all[t_step], I_day[mc_step]
            )

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save is not None:
        utils.saving(args, I_m, I_std, day_max, "net_sir", save)

    import matplotlib.pyplot as plt

    plt.plot(t_all, S_all)
    plt.plot(t_all, I_all)
    plt.plot(t_all, R_all)
    plt.show()
    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="Stochastic SIR model with a social network using the event-driven algorithm \
            It allows for different sections with different n, delta and beta: \
            same number of arguments must be specified for all three, \
            and one more for section_days.",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=[int(1e4)],
        nargs="*",
        help="parameter: fixed number of (effecitve) people [1000,1000000]",
    )

    parser.add_argument(
        "--network_type",
        type=str,
        default="er",
        choices=["er", "ba"],
        help="parameter: Erdos-Renyi or Barabasi Albert supported right now [er,ba]",
    )
    parser.add_argument(
        "--network_param",
        type=int,
        default=5,
        help="parameter: mean number of edges [1,50]",
    )

    parser.add_argument(
        "--I_0",
        type=int,
        default=20,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals [0,n]"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=[0.2],
        nargs="*",
        help="parameter: mean ratio of recovery [0.05,1]",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=[0.5],
        nargs="*",
        help="parameter: ratio of infection [0.05,1]",
    )
    parser.add_argument(
        "--section_days",
        type=int,
        default=[0, 100],
        nargs="*",
        help="parameter: starting day for each section, firts one must be 0,\
                        and final day for last one",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for the automatic configuration"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="timeout for the automatic configuration",
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
    network_type = args.network_type
    network_param = args.network_param
    n_sections = len(args.section_days) - 1
    return (
        I_0,
        R_0,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        network_type,
        network_param,
        n_sections,
    )


# -------------------------


def parameters_section(args, section, ratios_old=None, section_day_old=0):
    """
    Section dependent parameters from argparse
    """
    n = sum(args.n[: section + 1])
    ratios = {"beta": args.beta[section], "delta": args.delta[section]}
    section_day = args.section_days[section + 1]
    return (
        n,
        ratios,
        section_day,
        ratios_old,
        section_day_old + 1,
    )


# -------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
