"""
Stochastic mean-field SAIR model.
Uses the Gillespie algorithm and Erlang distribution transition times.
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
import sair_erlang
from utils import utils, config


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
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
            rates,
            shapes,
            section_day,
            rates_old,
            section_day_old,
            n_ind,
        ) = parameters_section(args, section)

        comp = sair_erlang.Compartments(shapes, args)

        t_step, time, day = 0, 0, 1
        I_day[mc_step, 0] = args.I_0
        index_n = 1  # just to avoid pylint complaining

        # Sections
        while section < n_sections:
            # Time loop
            while comp.I[t_step, :-1].sum() > 0 and day < section_day:

                # add individuals
                if (n_ind is not None) and (time // n_ind[index_n, 1] == 1):
                    comp.S[t_step] += n_ind[index_n, 0] / shapes["k_inf"]
                    if index_n < (len(n_ind) - 1):
                        index_n += 1
                    else:
                        n_ind = None

                day, day_max = utils.day_data(
                    time,
                    t_total,
                    day,
                    day_max,
                    comp.I[t_step, :-1].sum(),
                    I_day[mc_step],
                )
                t_step, time = gillespie(
                    t_total,
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

        # plot all trajectories
        # if plot:
        # plt.plot(I_day[mc_step,:])
        # plt.plot(T[:t_step],i[:t_step,:-1].sum(1),c='c')
        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if args.save is not None:
        utils.saving(args, I_m, I_std, day_max, "sair_erlang_sections", args.save)

    if args.plot:
        from utils import plots

        plots.plotting(args, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


# -------------------------
def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-field SAIR model using the Gillespie algorithm and Erlang \
            distribution transition times. It allows for different sections with different \
            n, delta and beta: same number of arguments must be specified for all three, \
            and one more for section_days. \
            Dependencies: config.py, utils.py, sair_erlang.py",
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
        "--k_rec",
        type=int,
        default=config.K_REC,
        help="k for the recovery time erlang distribution [1,5]",
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
        "--k_inf",
        type=int,
        default=config.K_INF,
        help="k for the infection time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--alpha",
        type=float,
        default=[config.ALPHA],
        nargs="*",
        help="asymptomatic rate (a->i) [0.05,1]",
    )
    parser_params.add_argument(
        "--k_lat",
        type=int,
        default=config.K_LAT,
        help="k for the asymptomatic time erlang distribution [1,5]",
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

    utils.parser_common(parser, True)

    return parser.parse_args()


# -------------------------
# Parameters
# -------------------------


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, infected_time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days) - 1
    # print(infected_time_series)
    return t_total, infected_time_series, n_sections


# -------------------------


def parameters_section(args, section, rates_old=None, section_day_old=0, n_old=None):
    """
    Section dependent parameters from argparse
    """
    n = sum(args.n[: section + 1])
    n_ind = utils.n_individuals(n, n_old, section_day_old, args.transition_days)
    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec, "k_lat": args.k_lat}
    rates = {
        "beta_a": args.beta_a[section] / n * args.k_inf,
        "beta_i": args.beta_i[section] / n * args.k_inf,
        "delta_a": args.delta_a[section] * args.k_rec,
        "delta_i": args.delta_i[section] * args.k_rec,
        "alpha": args.alpha[section] * args.k_lat,
    }
    section_day = args.section_days[section + 1]

    # should return section_day_old , given that the input will be section_day.
    # but I want to add 2 to it for the tanh in rates_sir/n_individuals
    return (n, rates, shapes, section_day, rates_old, section_day_old + 1, n_ind)


# -------------------------


def gillespie(
    t_total,
    t_step,
    time,
    section_day_old,
    comp,
    rates,
    rates_old,
    shapes,
):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()
    etot_rec = comp.A[t_step, :-1, 0].sum()
    etot_inf = comp.A[t_step, :-1, 1].sum()
    etot = etot_inf + etot_rec - comp.A[t_step, 0, 0]

    rates_eval = utils.rates_sair(time, rates, rates_old, section_day_old)
    lambda_sum = (
        rates_eval["alpha"] * etot_inf
        + rates_eval["delta_a"] * etot_rec
        + rates_eval["delta_i"] * itot
        + (rates_eval["beta_a"] * etot + rates_eval["beta_i"] * itot) * stot
    )

    probs = {}
    probs["heal_a"] = rates_eval["delta_a"] * comp.A[t_step, :-1, 0] / lambda_sum
    probs["heal_i"] = rates_eval["delta_i"] * comp.I[t_step, :-1] / lambda_sum
    probs["asymptomatic"] = rates_eval["alpha"] * comp.A[t_step, :-1, 1] / lambda_sum
    probs["infect"] = (
        (rates_eval["beta_a"] * etot + rates_eval["beta_i"] * itot)
        * comp.S[t_step, :-1]
        / lambda_sum
    )

    t_step += 1
    time += utils.time_dist(lambda_sum)
    if time > t_total:
        return t_step, True  # rare,  but sometimes long times may appear

    sair_erlang.gillespie_step(t_step, comp, probs, shapes)
    return t_step, time


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
