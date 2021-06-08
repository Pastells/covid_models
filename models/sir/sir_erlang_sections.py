"""
Stochastic mean-field SIR model \n
Uses the Gillespie algorithm and Erlang distribution transition times.
It allows for different sections with different n, delta and beta

Pol Pastells, 2020-2021

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
    n_sections: int,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
    k_rec: Int(1, 5) = 1,
    k_inf: Int(1, 5) = 1,
    section_day1: Int(0, 1000) = 10,
    section_day2: Int(0, 1000) = 0,
    section_day3: Int(0, 1000) = 0,
    section_day4: Int(0, 1000) = 0,
    section_day5: Int(0, 1000) = 0,
    n1: Int(70000, 90000) = 70000,
    n2: Int(0, 90000) = 7000,
    n3: Int(0, 90000) = 0,
    n4: Int(0, 90000) = 0,
    n5: Int(0, 90000) = 0,
    delta1: Real(0.1, 1.0) = 0.2,
    delta2: Real(0.1, 1.0) = 0.2,
    delta3: Real(0.1, 1.0) = 0.2,
    delta4: Real(0.1, 1.0) = 0.2,
    delta5: Real(0.1, 1.0) = 0.2,
    beta1: Real(0.1, 1.0) = 0.5,
    beta2: Real(0.1, 1.0) = 0.7,
    beta3: Real(0.1, 1.0) = 0.5,
    beta4: Real(0.1, 1.0) = 0.3,
    beta5: Real(0.1, 1.0) = 0.5,
):
    # "reconstruct" arrays and shapes dictionary
    section_days = [
        0,
        section_day1,
        section_day2,
        section_day3,
        section_day4,
        section_day5,
    ]

    n_vect = [n1, n2, n3, n4, n5]
    beta_vect = np.array([beta1, beta2, beta3, beta4, beta5]) * k_inf
    delta_vect = np.array([delta1, delta2, delta3, delta4, delta5]) * k_rec
    shapes = {"k_inf": k_inf, "k_rec": k_rec}

    mc_step = 0
    day_max = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    results = list()

    while mc_step < n_seeds:
        current_seed += 1
        result = gillespie_simulation(
            current_seed,
            n_vect,
            beta_vect,
            delta_vect,
            section_days,
            n_sections,
            n_t_steps,
            initial_infected,
            initial_recovered,
            t_total,
            shapes,
            day_max,
        )
        day_max = result.day_max

        if check_successful_simulation(result, t_total):
            mc_step += 1
            results.append(result)

    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=int)

    for mc_step, result in enumerate(results):
        infected[mc_step] = result.infected

    cost = get_cost(time_series, infected, t_total, day_max, n_seeds, metric)
    print(f"GGA SUCCESS {cost}")
    return cost


def gillespie_simulation(
    seed: int,
    n_vect: list,
    beta_vect: list,
    delta_vect: list,
    section_days: list,
    n_sections: int,
    n_t_steps: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    shapes: dict,
    day_max: int,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    # initialization
    section = 0
    n, rates, section_day, rates_old, section_day_old, n_ind = parameters_section(
        n_vect,
        beta_vect,
        delta_vect,
        section_days,
        section,
    )

    comp = sir_erlang.Compartments(
        n, n_t_steps, initial_infected, initial_recovered, shapes
    )

    infected = np.zeros(t_total, dtype=int)
    infected[0] = initial_infected

    t_step, time = 0, 0
    index_n = 1  # just to avoid pylint complaining

    # Sections
    while section < n_sections:
        # Time loop
        while comp.I[t_step, :-1].sum() > 0 and time < section_day:

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
                section_day,
                rates_old,
                section_day_old,
                n_ind,
            ) = parameters_section(
                n_vect,
                beta_vect,
                delta_vect,
                section_days,
                section,
                rates,
                section_day,
                n,
            )
            if n_ind is not None:
                comp.S[t_step, :-1] += n_ind[0, 0] / shapes["k_inf"]
            index_n = 1

    day_max = utils.day_data(
        comp.T[:t_step], comp.I[:t_step, :-1].sum(axis=1), infected, day_max
    )

    return Result(infected, day_max)


def parameters_section(
    n_vect,
    beta_vect,
    delta_vect,
    section_days,
    section,
    rates_old=None,
    section_day_old=0,
    n_old=None,
):
    """
    Section dependent parameters
    """
    n = sum(n_vect[: section + 1])
    n_ind = utils.n_individuals(n, n_old, section_day_old)
    rates = {"beta": beta_vect[section] / n, "delta": delta_vect[section]}
    section_day = section_days[section + 1]

    # should return section_day_old , given that the input will be section_day.
    # but I want to add 2 to it for the tanh in rates_sir/n_individuals
    # TODO: aclarar aquest comentari ^^
    return (n, rates, section_day, rates_old, section_day_old + 1, n_ind)


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


def pad(var, length=5):
    return np.pad(var, (0, length - len(var)))


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days)

    if not len(args.beta) == len(args.delta) == n_sections >= len(args.n):
        raise ValueError("All rates and n must have same dimension")

    args.section_days = pad(args.section_days)
    args.beta = pad(args.beta)
    args.delta = pad(args.delta)
    args.n = pad(args.n)

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
        n_sections,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        k_rec=args.k_rec,
        k_inf=args.k_inf,
        section_day1=args.section_days[0],
        section_day2=args.section_days[1],
        section_day3=args.section_days[2],
        section_day4=args.section_days[3],
        section_day5=args.section_days[4],
        n1=args.n[0],
        n2=args.n[1],
        n3=args.n[2],
        n4=args.n[3],
        n5=args.n[4],
        delta1=args.delta[0],
        delta2=args.delta[1],
        delta3=args.delta[2],
        delta4=args.delta[3],
        delta5=args.delta[4],
        beta1=args.beta[0],
        beta2=args.beta[1],
        beta3=args.beta[2],
        beta4=args.beta[3],
        beta5=args.beta[4],
    )
