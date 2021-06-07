"""
Stochastic mean-field SIR model using the Gillespie algorithm

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

from ..utils import utils

Result = namedtuple("Result", "infected recovered day_max")

# TODO: maybe put this in the utils or a common file
def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series: np.ndarray, infected, t_total, day_max, n_seeds, metric):
    var_m = utils.mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], var_m, metric)


@ac
def sir(
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
):
    # Normalize beta for the number of individuals
    beta = beta / n

    mc_step = 0
    day_max = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    results = list()

    while mc_step < n_seeds:
        current_seed += 1
        result = gillespie_simulation(
            current_seed,
            n,
            n_t_steps,
            initial_infected,
            initial_recovered,
            t_total,
            beta,
            delta,
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
    n: int,
    n_t_steps: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    beta: float,
    delta: float,
    day_max: int,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    # initialization
    comp = Compartments(n, n_t_steps, initial_infected, initial_recovered)

    infected = np.zeros(t_total, dtype=int)
    recovered = np.zeros(t_total, dtype=int)

    infected[0] = initial_infected
    recovered[0] = initial_recovered

    t_step, time = 0, 0

    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(
            t_step,
            time,
            comp,
            beta=beta,
            delta=delta,
        )

    day_max = utils.day_data(comp.T[:t_step], comp.R[:t_step], recovered, day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.I[:t_step], infected, day_max)

    return Result(infected, recovered, day_max)


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, n, n_t_steps, initial_infected, initial_recovered):
        """Initialization"""
        self.S = np.zeros(n_t_steps, dtype=int)
        self.I = np.zeros(n_t_steps, dtype=int)
        self.R = np.zeros(n_t_steps, dtype=int)
        self.T = np.zeros(n_t_steps)
        self.I[0] = initial_infected
        self.R[0] = initial_recovered
        self.S[0] = n - initial_infected - initial_recovered
        self.T[0] = 0
        self.I_cum = np.zeros(n_t_steps, dtype=int)
        self.I_cum[0] = initial_infected

    def infect(self, t_step):
        """Infection"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + 1

    def recover(self, t_step):
        """Recovery"""
        self.S[t_step] = self.S[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]


def gillespie(t_step, time, comp, beta, delta):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    lambda_sum = (delta + beta * comp.S[t_step]) * comp.I[t_step]
    prob_heal = delta * comp.I[t_step] / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time

    gillespie_step(t_step, comp, prob_heal)
    return t_step, time


def gillespie_step(t_step, comp, prob_heal):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < prob_heal:
        comp.recover(t_step)
    else:
        comp.infect(t_step)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta, "delta": args.delta}
    return t_total, time_series, rates


def main(args):
    t_total, time_series, rates = parameters_init(args)
    sir(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.n_t_steps,
        args.metric,
        n=args.n,  # due to a bug, naming the configurable parameters is mandatory
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        delta=rates["delta"],
        beta=rates["beta"],
    )
