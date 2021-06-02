"""
Stochastic SAIR model with a social network using the event-driven algorithm.
It allows for different sections with different n, delta and beta

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta/N*I(t)*S(t) \n
dA(t)/dt =   beta_a/N*A(t)*S(t) + beta/N*I(t)*S(t) -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta * I(t)                          + alpha*A(t)\n
dR(t)/dt =   delta * I(t)                          + delta_a * A(t)
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

    day_max = 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.seed, args.seed + args.mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)
        mc_step = mc_seed - args.seed

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

        I_day[mc_step, 0] = args.initial_infected

        # Sections
        while section < n_sections:
            t, A, I, R = fast_sair_sections.fast_SAIR(
                G,
                rates,
                rates_old,
                section_day_old,
                args.initial_asymptomatic,
                args.initial_infected,
                args.initial_recovered,
                tmin=section_day_old - 1,
                tmax=section_day,
            )
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
                args.initial_asymptomatic = A[-1]
                args.initial_infected = I[-1]
                # R will have jumps
                args.initial_recovered = R[-1]

        day_max = utils.day_data(t, I, I_day[mc_step], day_max)
        del t, I, G

    # =========================

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""

    description = "Stochastic SAIR model with a social network using the event-driven algorithm \
        It allows for different sections with different n, delta, delta_a, beta, beta_a and alpha: \
        same number of arguments must be specified for all 5, and section_days. \
            Dependencies: config.py, utils.py, utils_net.py, fast_sair_sections.py"

    parser = utils.ParserCommon(description)
    parser.n_sections()
    parser.sir_sections()
    parser.asymptomatic_sections()
    parser.network()

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days)
    args.section_days.insert(0, 0)

    if (
        not len(args.beta)
        == len(args.delta)
        == len(args.beta_a)
        == len(args.delta_a)
        == len(args.alpha)
        == n_sections
        >= len(args.n)
    ):
        raise ValueError("All rates and n must have same dimension")

    return t_total, time_series, n_sections


# -------------------------


def parameters_section(args, section, rates_old=None, section_day_old=0):
    """
    Section dependent parameters from argparse
    """
    n = sum(args.n[: section + 1])
    rates = {
        "beta_a": args.beta_a[section],
        "beta": args.beta[section],
        "delta_a": args.delta_a[section],
        "delta": args.delta[section],
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
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc(ex)
