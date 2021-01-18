"""
Stochastic SIR model with a social network using the event-driven algorithm.
It allows for different sections with different n, delta and beta

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""

import random
import sys
import traceback
import numpy as np
from event_driven import fast_sir_sections
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

        t_all, S_all, I_all, R_all = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
        I_day[mc_step, 0] = args.I_0

        # Sections
        while section < n_sections:
            t, S, I, R = fast_sir_sections.fast_SIR(
                G,
                rates,
                rates_old,
                section_day_old,
                args.I_0,
                args.R_0,
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
                    rates,
                    section_day,
                    rates_old,
                    section_day_old,
                ) = parameters_section(args, section, rates, section_day)
                if section == n_sections - 1:
                    section_day -= 0.9
                G = utils_net.choose_network(n, args.network, args.network_param)
                args.I_0 = I[-1]
                # R will have jumps, given that the n
                args.R_0 = R[-1]

        if config.CUMULATIVE is True:
            i_var = I + R
        else:
            i_var = I

        day_max = utils.day_data(t, i_var, I_day[mc_step], day_max)

        mc_step += 1
    # =========================

    import matplotlib.pyplot as plt

    sum_all = S_all + I_all + R_all
    plt.plot(t_all, S_all, label="S")
    plt.plot(t_all, I_all, label="I")
    plt.plot(t_all, R_all, label="R")
    plt.plot(t_all, sum_all, label="total")
    plt.legend()
    plt.show()

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""

    description = "Stochastic SIR model with a social network using the event-driven algorithm \
        It allows for different sections with different n, delta and beta: \
        same number of arguments must be specified for all three, \
        and one more for section_days. \
            Dependencies: config.py, utils.py, utils_net.py, fast_sir_sections.py"

    parser = utils.parser_common(description)
    parser.n_sections()
    parser.sir_sections()
    parser.network()

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
    rates = {"beta": args.beta[section], "delta": args.delta[section]}
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
