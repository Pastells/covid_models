"""

Stochastic SAIR model with a social network using the event-driven algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta/N*I(t)*S(t) \n
dA(t)/dt =   beta_a/N*A(t)*S(t) + beta/N*I(t)*S(t) -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta * I(t)                          + alpha*A(t)\n
dR(t)/dt =   delta * I(t)                          + delta_a * A(t)
"""
import functools
import random

import numpy as np
from optilog.tuning import ac, Int, Real, Categorical

from . import fast_sair
from .sair import simulate_evolution, get_cost, Result
from ..utils import utils, utils_net


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


@ac
def net_sair(
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
    initial_asymptomatic: Int(0, 1000) = 0,
    alpha: Real(0.05, 2.0) = 0.05,
    delta_a: Real(0.05, 1.0) = 0.05,
    delta: Real(0.03, 0.06) = 0.03,
    beta_a: Real(0.05, 1.0) = 0.05,
    beta: Real(0.3, 0.4) = 0.3,
):
    rates = {
        "beta_a": beta_a / network_param,
        "beta": beta / network_param,
        "delta_a": delta_a,
        "delta": delta,
        "alpha": alpha,
    }

    func = functools.partial(
        event_driven_simulation,
        n=n,
        network=network,
        network_param=network_param,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        initial_asymptomatic=initial_asymptomatic,
        t_total=t_total,
        rates=rates,
    )

    # results per day and seed
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
    initial_asymptomatic: int,
    t_total: int,
    rates: dict,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    G = utils_net.choose_network(n, network, network_param)
    t, I = fast_sair.fast_SAIR(
        G,
        rates,
        initial_asymptomatic,
        initial_infected,
        initial_recovered,
        tmax=t_total - 0.95,
    )

    day_max, infected = utils.day_data(t, I, t_total)
    del t, I, G
    # TODO compute Susceptible, Asymptomatic, and Recovered
    return Result(None, None, infected, None, day_max)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    return net_sair(
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
        initial_asymptomatic=args.initial_asymptomatic,
        alpha=args.alpha,
        delta_a=args.delta_a,
        delta=args.delta,
        beta_a=args.beta_a,
        beta=args.beta,
    )
