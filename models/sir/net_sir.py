"""
Stochastic SIR model with a social network using the event-driven algorithm

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

from . import fast_sir
from ..utils import utils, utils_net

Result = namedtuple("Result", "infected day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series: np.ndarray, infected, t_total, day_max, n_seeds, metric):
    var_m = utils.mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], var_m, metric)


@ac
def net_sir(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    metric: str,
    network: str,
    network_param: int,
    n: Int(70000, 90000) = 70000,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
    delta: Real(0.1, 1.0) = 0.2,
    beta: Real(0.1, 1.0) = 0.5,
):
    # Normalize beta for the number of individuals
    rates = {"beta": beta / n, "delta": delta}
    beta = beta / n

    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=np.uint32)

    mc_step = 0
    day_max = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    results = list()

    while mc_step < n_seeds:
        current_seed += 1
        result = event_driven_simulation(
            current_seed,
            n,
            network,
            network_param,
            initial_infected,
            initial_recovered,
            t_total,
            rates,
            day_max,
        )
        day_max = result.day_max

        if check_successful_simulation(result, t_total):
            mc_step += 1
            results.append(result)

    for mc_step, result in enumerate(results):
        infected[mc_step] = result.infected

    cost = get_cost(time_series, infected, t_total, day_max, n_seeds, metric)
    print(f"GGA SUCCESS {cost}")
    return cost


def event_driven_simulation(
    seed: int,
    n: int,
    network: str,
    network_param: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    rates: dict,
    day_max: int,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    infected = np.zeros(t_total, dtype=int)
    infected[0] = initial_infected

    G = utils_net.choose_network(n, network, network_param)
    t, I = fast_sir.fast_SIR(
        G, rates, initial_infected, initial_recovered, tmax=t_total - 0.95
    )

    day_max = utils.day_data(t, I, infected, day_max)
    del t, I, G
    return Result(infected, day_max)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta, "delta": args.delta}
    return t_total, time_series, rates


def main(args):
    t_total, time_series, rates = parameters_init(args)
    net_sir(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.metric,
        args.network,
        args.network_param,
        n=args.n,  # due to a bug, naming the configurable parameters is mandatory
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        delta=rates["delta"],
        beta=rates["beta"],
    )
