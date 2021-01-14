"""
Stochastic SAIR model with a social network using the event-driven algorithm.
It allows for different sections with different n, delta and beta

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
from event_driven import fast_sair_sections
from utils import utils, utils_net, config


def main():
    args = parsing()
    t_total, time_series, n_sections = parameters_init(args)
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

        # initialization
        section = 0
        (
            n,
            rates,
            section_day,
            rates_old,
            section_day_old,
        ) = parameters_section(args, section)

        G = utils_net.choose_network(n, args.network, args.network_param)

        t_all, S_all, A_all, I_all, R_all = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        I_day[mc_step, 0] = args.I_0

        # Sections
        while section < n_sections:
            t, S, A, I, R = fast_sair_sections.fast_SAIR(
                G,
                rates,
                rates_old,
                section_day_old,
                args.A_0,
                args.I_0,
                args.R_0,
                tmin=section_day_old - 1,
                tmax=section_day,
            )
            t_all = np.append(t_all, t)
            S_all = np.append(S_all, S)
            A_all = np.append(A_all, A)
            I_all = np.append(I_all, I)
            R_all = np.append(R_all, R)
            section += 1
            if section < n_sections:
                (
                    n,
                    rates,
                    section_day,
                    rates_old,
                    section_day_old,
                ) = parameters_section(args, section, rates, section_day)
                if section == n_sections - 1:
                    section_day -= 0.9
                G = utils_net.choose_network(n, args.network, args.network_param)
                args.A_0 = A[-1]
                args.I_0 = I[-1]
                # R will have jumps
                args.R_0 = R[-1]

        if config.CUMULATIVE is True:
            i_var = I + R
        else:
            i_var = I

        day_max = utils.day_data(t, i_var, I_day[mc_step], day_max)

        mc_step += 1
    # =========================

    I_m = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    if config.CUMULATIVE is True:
        utils.cost_func(time_series[:, 3], I_m, args.metric)
    else:
        utils.cost_func(time_series[:, 0], I_m, args.metric)

    if args.save is not None:
        utils.saving(args, I_m, day_max)

    import matplotlib.pyplot as plt

    sum_all = S_all + A_all + I_all + R_all
    plt.plot(t_all, S_all, label="S")
    plt.plot(t_all, A_all, label="A")
    plt.plot(t_all, I_all, label="I")
    plt.plot(t_all, R_all, label="R")
    plt.plot(t_all, sum_all, label="total")
    plt.legend()
    plt.show()

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
                Dependencies: config.py, utils.py, utils_net.py, fast_sair_sections.py",
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
        "--delta_a",
        type=float,
        default=[config.DELTA_A],
        nargs="*",
        help="rate of recovery from asymptomatic phase (a->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta_i",
        type=float,
        default=[config.DELTA],
        nargs="*",
        help="rate of recovery from infected phase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--beta_a",
        type=float,
        default=[config.BETA_A],
        nargs="*",
        help="infectivity due to asymptomatic [0.05,1]",
    )
    parser_params.add_argument(
        "--beta_i",
        type=float,
        default=[config.BETA],
        nargs="*",
        help="infectivity due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--alpha",
        type=float,
        default=[config.ALPHA],
        nargs="*",
        help="asymptomatic rate (a->i) [0.05,2]",
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
    t_total, time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days) - 1

    return t_total, time_series, n_sections


# -------------------------


def parameters_section(args, section, rates_old=None, section_day_old=0):
    """
    Section dependent parameters from argparse
    """
    n = sum(args.n[: section + 1])
    rates = {
        "beta_a": args.beta_a[section],
        "beta_i": args.beta_i[section],
        "delta_a": args.delta_a[section],
        "delta_i": args.delta_i[section],
        "alpha": args.alpha[section],
    }
    section_day = args.section_days[section + 1]
    return (
        n,
        rates,
        section_day,
        rates_old,
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
