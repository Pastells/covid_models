"""
Stochastic mean-field SIR model \n
Uses the Gillespie algorithm and Erlang distribution transition times.
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
import sir_erlang
from utils import utils, config


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    args = parsing()
    # print(args)
    t_total, time_series, n_sections = parameters_init(args)

    # results per day and seed
    I_day = np.zeros([args.mc_nseed, t_total], dtype=int)

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.mc_seed0, args.mc_seed0 + args.mc_nseed):
        # print("seed", mc_seed)
        random.seed(mc_seed)
        np.random.seed(mc_seed)

        # initialization
        section = 0
        (
            n,
            rates,
            shapes,
            section_day,
            rates_old,
            section_day_old,
            n_ind,
        ) = parameters_section(args, section)

        comp = sir_erlang.Compartments(shapes, args)

        t_step, time = 0, 0
        I_day[mc_step, 0] = args.I_0
        index_n = 1  # just to avoid pylint complaining

        # Sections
        while section < n_sections:
            # print("section", section)
            # Time loop
            while comp.I[t_step, :-1].sum() > 0 and time < section_day:
                # print(time, day, section_day)

                # add individuals
                if (n_ind is not None) and (time // n_ind[index_n, 1] == 1):
                    comp.S[t_step] += n_ind[index_n, 0] / shapes["k_inf"]
                    if index_n < (len(n_ind) - 1):
                        index_n += 1
                    else:
                        n_ind = None

                t_step, time = gillespie(
                    t_step,
                    time,
                    section_day_old,
                    comp,
                    rates,
                    rates_old,
                    shapes,
                )

            section += 1
            if section < n_sections:
                (
                    n,
                    rates,
                    shapes,
                    section_day,
                    rates_old,
                    section_day_old,
                    n_ind,
                ) = parameters_section(args, section, rates, section_day, n)
                if n_ind is not None:
                    comp.S[t_step, :-1] += n_ind[0, 0] / shapes["k_inf"]
                index_n = 1
        # -------------------------

        if config.CUMULATIVE is True:
            i_var = comp.I_cum
        else:
            i_var = comp.I[:, :-1].sum(axis=1)

        day_max = utils.day_data(comp.T[:t_step], i_var[:t_step], I_day[mc_step], day_max)

        mc_step += 1
    # =========================

    I_m = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    if config.CUMULATIVE is True:
        utils.cost_func(time_series[:, 3], I_m, args.metric)
    else:
        utils.cost_func(time_series[:, 0], I_m, args.metric)

    if args.save is not None:
        utils.saving(args, I_m, day_max)

    if args.plot:
        from utils import plots

        plots.plotting(args, day_max, I_m)  # , comp=comp, t_step=t_step)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


# -------------------------
def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-field SIR model using the Gillespie algorithm and Erlang \
            distribution transition times. It allows for different sections with different \
            n, delta and beta: same number of arguments must be specified for all three, \
            and one more for section_days. \
            Dependencies: config.py, utils.py, sir_erlang.py",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_params = parser.add_argument_group("parameters")

    parser_params.add_argument(
        "--n",
        type=int,
        default=[config.N],
        nargs="*",
        help="fixed number of (effecitve) people, initial and increments [1000,1000000]",
    )
    parser_params.add_argument(
        "--delta",
        type=float,
        default=[config.DELTA],
        nargs="*",
        help="rate of recovery [0.05,1]",
    )
    parser_params.add_argument(
        "--k_rec",
        type=int,
        default=config.K_REC,
        help="k for the recovery time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--beta",
        type=float,
        default=[config.BETA],
        nargs="*",
        help="infectivity [0.05,1]",
    )
    parser_params.add_argument(
        "--k_inf",
        type=int,
        default=config.K_INF,
        help="k for the infection time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--section_days",
        type=int,
        default=config.SECTIONS_DAYS,
        nargs="*",
        help="starting day for each section, first one must be 0,\
                        and final day for last one",
    )
    parser_params.add_argument(
        "--transition_days",
        type=int,
        default=config.TRANSITION_DAYS,
        help="days it takes to transition from one number of individuals to the next [1,10]",
    )

    utils.parser_common(parser)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days) - 1
    return t_total, time_series, n_sections


# -------------------------


def parameters_section(args, section, rates_old=None, section_day_old=0, n_old=None):
    """
    Section dependent parameters from argparse
    """
    n = sum(args.n[: section + 1])
    n_ind = utils.n_individuals(n, n_old, section_day_old, args.transition_days)
    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec}
    rates = {"beta": args.beta[section] / n, "delta": args.delta[section]}
    section_day = args.section_days[section + 1]

    # should return section_day_old , given that the input will be section_day.
    # but I want to add 2 to it for the tanh in rates_sir/n_individuals
    return (n, rates, shapes, section_day, rates_old, section_day_old + 1, n_ind)


# -------------------------


def gillespie(t_step, time, section_day_old, comp, rates, rates_old, shapes):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()

    rates_eval = utils.rates_sir(time, rates, rates_old, section_day_old)

    lambda_sum = (rates_eval["delta"] + rates_eval["beta"] * stot) * itot
    probs = {}
    probs["heal"] = rates_eval["delta"] * comp.I[t_step, :-1] / lambda_sum
    probs["infect"] = rates_eval["beta"] * comp.S[t_step, :-1] * itot / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time

    sir_erlang.gillespie_step(t_step, comp, probs, shapes)
    return t_step, time


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc()
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
