"""
Stochastic SIR model with a social network using the event-driven algorithm

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""
import functools
import random
from typing import Tuple

import numpy as np
import pandas
from optilog.tuning import ac, Int, Real, Categorical

from . import fast_sir
from .sir import get_cost, simulate_evolution, Result
from ..utils import utils, utils_net


@ac
def net_sir(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    metric: str,
    network: Categorical("er", "ba") = "ba",
    network_param: Int(1, 50) = 5,
    n: Int(70_000, 500_000) = 70_000,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
    delta: Real(0.1, 1.0) = 0.2,
    beta: Real(0.1, 1.0) = 0.5,
) -> Tuple[float, pandas.DataFrame]:
    # Normalize beta for the number of individuals
    rates = {"beta": beta / network_param, "delta": delta}

    func = functools.partial(
        event_driven_simulation,
        n=n,
        network=network,
        network_param=network_param,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        t_total=t_total,
        rates=rates,
    )

    evolution_df, day_max = simulate_evolution(func, n_seeds, seed, t_total)

    cost = get_cost(time_series, evolution_df.infected, t_total, day_max, metric)
    print(f"GGA SUCCESS {cost}")
    return cost, evolution_df


def event_driven_simulation(
    seed: int,
    n: int,
    network: str,
    network_param: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    rates: dict,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    G = utils_net.choose_network(n, network, network_param)
    t, S, I, R = fast_sir.fast_SIR(
        G, rates, initial_infected, initial_recovered, tmax=t_total - 0.95
    )

    _, susceptible = utils.day_data(t, S, t_total)
    day_max, infected = utils.day_data(t, I, t_total)
    _, recovered = utils.day_data(t, R, t_total)
    del t, R, I, S, G
    return Result(susceptible, infected, recovered, day_max)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    return net_sir(
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
