"""
Stochastic SIR model with a social network using the event-driven algorithm.
It allows for different sections with different n, delta and beta

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""
import functools
import random

import numpy as np
from optilog.autocfg import ac, Int, Real, Categorical

from .sir import simulate_evolution, Result, get_cost
from . import fast_sir_sections
from ..utils import utils, utils_net


@ac
def net_sir_sections(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    metric: str,
    n_sections: int,
    network: Categorical("er", "ba") = "ba",
    network_param: Int(1, 50) = 5,
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
    beta_vect = np.array([beta1, beta2, beta3, beta4, beta5]) / network_param
    delta_vect = [delta1, delta2, delta3, delta4, delta5]

    func = functools.partial(
        event_driven_simulation,
        n_vect=n_vect,
        beta_vect=beta_vect,
        delta_vect=delta_vect,
        section_days=section_days,
        n_sections=n_sections,
        network=network,
        network_param=network_param,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        t_total=t_total,
    )

    evolution_df, day_max = simulate_evolution(func, n_seeds, seed, t_total)

    cost = get_cost(time_series, evolution_df.infected, t_total, day_max, metric)
    print(f"GGA SUCCESS {cost}")
    return cost, evolution_df


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
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    # initialization
    section = 0
    (n, rates, section_day, rates_old, section_day_old,) = parameters_section(
        n_vect, beta_vect, delta_vect, section_days, section, network_param
    )

    G = utils_net.choose_network(n, network, network_param)

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
                n_vect,
                beta_vect,
                delta_vect,
                section_days,
                section,
                network_param,
                rates,
                section_day,
            )
            if section == n_sections - 1:
                section_day -= 0.9
            G = utils_net.choose_network(n, network, network_param)
            initial_infected = I[-1]
            # R will have jumps, given that the n
            initial_recovered = R[-1]

    _, susceptible = utils.day_data(t, n - I - R, t_total)
    day_max, infected = utils.day_data(t, I, t_total)
    _, recovered = utils.day_data(t, R, t_total)
    del t, I, R, G
    return Result(susceptible, infected, recovered, day_max)


def parameters_section(
    n_vect,
    beta_vect,
    delta_vect,
    section_days,
    section,
    network_param,
    rates_old=None,
    section_day_old=0,
):
    """
    Section dependent parameters from argparse
    """
    n = sum(n_vect[: section + 1])
    rates = {"beta": beta_vect[section] / network_param, "delta": delta_vect[section]}
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
    return net_sir_sections(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.metric,
        n_sections,
        network=args.network,
        network_param=args.network_param,
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
