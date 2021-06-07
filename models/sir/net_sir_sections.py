"""
Stochastic SIR model with a social network using the event-driven algorithm.
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

from . import fast_sir_sections
from ..utils import utils, utils_net, config

Result = namedtuple("Result", "infected day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series: np.ndarray, infected, t_total, day_max, n_seeds, metric):
    var_m = utils.mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], var_m, metric)


@ac
def net_sir_sections(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    metric: str,
    n_sections: int,
    network: str,
    network_param: int,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
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
    beta_vect = [beta1, beta2, beta3, beta4, beta5]
    delta_vect = [delta1, delta2, delta3, delta4, delta5]

    mc_step = 0
    day_max = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    results = list()

    while mc_step < n_seeds:
        current_seed += 1
        result = event_driven_simulation(
            current_seed,
            n_vect,
            beta_vect,
            delta_vect,
            section_days,
            n_sections,
            network,
            network_param,
            initial_infected,
            initial_recovered,
            t_total,
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


def event_driven_simulation(
    seed: int,
    n_vect: list,
    beta_vect: list,
    delta_vect: list,
    section_days: list,
    n_sections: int,
    network: str,
    network_param: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    day_max: int,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    # initialization
    section = 0
    (
        n,
        rates,
        section_day,
        rates_old,
        section_day_old,
    ) = parameters_section(n_vect, beta_vect, delta_vect, section_days, section)

    G = utils_net.choose_network(n, network, network_param)

    infected = np.zeros(t_total, dtype=int)
    infected[0] = initial_infected

    # Sections
    while section < n_sections:
        t, I, R = fast_sir_sections.fast_SIR(
            G,
            rates,
            rates_old,
            section_day_old,
            initial_infected,
            initial_recovered,
            tmin=section_day_old - 1,
            tmax=section_day,
        )
        section += 1
        if section < n_sections:
            (n, rates, section_day, rates_old, section_day_old,) = parameters_section(
                n_vect, beta_vect, delta_vect, section_days, section, rates, section_day
            )
            if section == n_sections - 1:
                section_day -= 0.9
            G = utils_net.choose_network(n, network, network_param)
            initial_infected = I[-1]
            # R will have jumps, given that the n
            initial_recovered = R[-1]

    day_max = utils.day_data(t, I, infected, day_max)
    del t, I, G
    return Result(infected, day_max)


def parameters_section(
    n_vect,
    beta_vect,
    delta_vect,
    section_days,
    section,
    rates_old=None,
    section_day_old=0,
):
    """
    Section dependent parameters from argparse
    """
    n = sum(n_vect[: section + 1])
    rates = {"beta": beta_vect[section], "delta": delta_vect[section]}
    section_day = section_days[section + 1]
    return (
        n,
        rates,
        section_day,
        rates_old,
        section_day_old + 1,
    )


def pad(var, length=5):
    return np.pad(var, (0, length - len(var)))


def parameters_init(args):
    """initial parameters from argparse"""
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
    net_sir_sections(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.metric,
        n_sections,
        args.network,
        args.network_param,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
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
