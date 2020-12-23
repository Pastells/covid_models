"""
Stochastic sir model with a social network using the event-driven algorithm
It allows for different sections with different n, delta and beta
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
import fast_seir_sections
from utils import utils, utils_net, config


def main():
    args = parsing()
    # print(args)
    t_total, infected_time_series, n_sections = parameters_init(args)

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

        # initialization
        section = 0
        (
            n,
            ratios,
            section_day,
            ratios_old,
            section_day_old,
        ) = parameters_section(args, section)

        G = utils_net.choose_network(n, args.network, args.network_param)

        t_all, S_all, E_all, I_all, R_all = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        I_day[mc_step, 0] = args.I_0

        # Sections
        while section < n_sections:
            t, S, E, I, R = fast_seir_sections.fast_SEIR(
                G,
                ratios,
                ratios_old,
                section_day_old,
                args.E_0,
                args.I_0,
                args.R_0,
                tmin=section_day_old - 1,
                tmax=section_day,
            )
            t_all = np.append(t_all, t)
            S_all = np.append(S_all, S)
            E_all = np.append(E_all, E)
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
                G = utils_net.choose_network(n, args.network, args.network_param)
                args.E_0 = E[-1]
                args.I_0 = I[-1]
                # R will have jumps, given that the n
                args.R_0 = R[-1]

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
        utils.saving(args, I_m, I_std, day_max, "net_seir_sections", args.save)

    import matplotlib.pyplot as plt

    sum_all = S_all + E_all + I_all + R_all
    plt.plot(t_all, S_all, label="S")
    plt.plot(t_all, E_all, label="E")
    plt.plot(t_all, I_all, label="I")
    plt.plot(t_all, R_all, label="R")
    plt.plot(t_all, sum_all, label="total")
    plt.legend()
    plt.show()
    if args.plot:
        from utils import plots

        plots.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic SEIR model using the Gillespie algorithm. \
                Dependencies: config.py, utils.py, utils_net.py, fast_seir_sections.py",
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
        default=[config.N],
        nargs="*",
        help="fixed number of (effecitve) people, initial and increments [1000,1000000]",
    )
    parser_params.add_argument(
        "--delta1",
        type=float,
        default=[config.DELTA1],
        nargs="*",
        help="ratio of recovery from latent fase (e->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta2",
        type=float,
        default=[config.DELTA],
        nargs="*",
        help="ratio of recovery from infected fase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--beta1",
        type=float,
        default=[config.BETA1],
        nargs="*",
        help="ratio of infection due to latent [0.05,1]",
    )
    parser_params.add_argument(
        "--beta2",
        type=float,
        default=[config.BETA],
        nargs="*",
        help="ratio of infection due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--epsilon",
        type=float,
        default=[config.EPSILON],
        nargs="*",
        help="ratio of latency (e->i) [0.05,2]",
    )
    parser_params.add_argument(
        "--section_days",
        type=int,
        default=config.SECTIONS_DAYS,
        nargs="*",
        help="starting day for each section, first one must be 0,\
                        and final day for last one",
    )

    utils.parser_common(parser, True)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, infected_time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days) - 1

    return t_total, infected_time_series, n_sections


# -------------------------


def parameters_section(args, section, ratios_old=None, section_day_old=0):
    """
    Section dependent parameters from argparse
    """
    n = sum(args.n[: section + 1])
    ratios = {
        "beta1": args.beta1[section],
        "beta2": args.beta2[section],
        "delta1": args.delta1[section],
        "delta2": args.delta2[section],
        "epsilon": args.epsilon[section],
    }
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
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
