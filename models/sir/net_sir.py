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
from typing import Tuple

import numpy as np
import pandas
from optilog.autocfg import ac, Int, Real, Categorical

from . import fast_sir
from ..utils import utils, utils_net

Result = namedtuple("Result", "susceptible infected recovered day_max")


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
    network: Categorical("er", "ba") = "ba",
    network_param: Int(1, 50) = 5,
    n: Int(70000, 90000) = 70000,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
    delta: Real(0.1, 1.0) = 0.2,
    beta: Real(0.1, 1.0) = 0.5,
) -> Tuple[float, pandas.DataFrame]:
    # Normalize beta for the number of individuals
    rates = {"beta": beta / network_param, "delta": delta}

    mc_step = 0
    day_max = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    seeds = list()
    evolution = np.zeros([3, n_seeds, t_total])

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
            seeds.append(current_seed)
            evolution[0, mc_step, :] = result.susceptible
            evolution[1, mc_step, :] = result.infected
            evolution[2, mc_step, :] = result.recovered
            mc_step += 1

    cost = get_cost(time_series, evolution[1], t_total, day_max, n_seeds, metric)
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

    susceptible = np.zeros(t_total, dtype=int)
    infected = np.zeros(t_total, dtype=int)
    recovered = np.zeros(t_total, dtype=int)

    G = utils_net.choose_network(n, network, network_param)
    t, S, I, R = fast_sir.fast_SIR(
        G, rates, initial_infected, initial_recovered, tmax=t_total - 0.95
    )

    day_max = utils.day_data(t, I, infected, day_max)
    utils.day_data(t, S, susceptible, day_max)
    utils.day_data(t, R, recovered, day_max)
    del t, R, I, S, G
    return Result(susceptible, infected, recovered, day_max)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    net_sir(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.metric,
        network=args.network,
        network_param=args.network_param,
        n=args.n,  # due to a bug, naming the configurable parameters is mandatory
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        delta=args.delta,
        beta=args.beta,
    )
