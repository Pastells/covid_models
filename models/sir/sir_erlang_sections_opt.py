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
from collections import namedtuple

import numpy as np
from optilog.autocfg import ac, Int, Real

from ..utils import utils, config
from . import sir_erlang

Result = namedtuple("Result", "infected day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series: np.ndarray, infected, t_total, day_max, n_seeds, metric):
    var_m = utils.mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], var_m, metric)


@ac
def sir_erlang_sections(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    n_t_steps: int,
    metric: str,
    n: Int(70000, 90000) = 70000,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
    delta: Real(0.1, 1.0) = 0.2,
    beta: Real(0.1, 1.0) = 0.5,
    k_rec: Int(1, 5) = 1,
    k_inf: Int(1, 5) = 1,
):

    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=int)

    day_max = 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.seed, args.seed + args.mc_nseed):
        # print("seed", mc_seed)
        random.seed(mc_seed)
        np.random.seed(mc_seed)
        mc_step = mc_seed - args.seed

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

        comp = sir_erlang.Compartments(
            n, n_t_steps, initial_infected, initial_recovered, shapes
        )

        t_step, time = 0, 0
        I_day[mc_step, 0] = args.initial_infected
        index_n = 1  # just to avoid pylint complaining

        # Sections
        while section < n_sections:
            # print("section", section, time, section_day)
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

        day_max = utils.day_data(
            comp.T[:t_step], i_var[:t_step], I_day[mc_step], day_max
        )

    # =========================

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


# -------------------------
# -------------------------
# Parameters


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
    # TODO: aclarar aquest comentari ^^
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


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days) - 1
    return t_total, time_series, n_sections


def main(args):
    t_total, time_series, n_sections = parameters_init(args)
    sir_erlang_sections(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.n_t_steps,
        args.metric,
        n=args.n,  # due to a bug, naming the configurable parameters is mandatory
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
    )
