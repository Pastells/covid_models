"""
Stochastic mean-field SIRD model using the Gillespie algorithm

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =   delta*(1-theta) * I(t)
dD(t)/dt =   delta*theta * I(t)
"""

import random
import sys
from collections import namedtuple
from typing import Tuple

import numpy as np
import pandas
from optilog.tuning import ac, Int, Real

from ..utils import utils

Result = namedtuple("Result", "infected recovered dead day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series, infected, recovered, dead, metric):
    mean_infected = utils.mean_std(infected)
    mean_recovered = utils.mean_std(recovered)
    mean_dead = utils.mean_std(dead)

    infected_cost = utils.cost_return(time_series[:, 0], mean_infected, metric)
    print(f"cost_infected = {infected_cost}", file=sys.stderr)
    recovered_cost = utils.cost_return(time_series[:, 1], mean_recovered, metric)
    print(f"cost_recovered = {recovered_cost}", file=sys.stderr)
    dead_cost = utils.cost_return(time_series[:, 2], mean_dead, metric)
    print(f"cost_dead = {dead_cost}", file=sys.stderr)

    return infected_cost + recovered_cost + dead_cost


@ac
def sird(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    n_t_steps: int,
    metric,
    scale_cost: bool,
    n: Int(70_000, 500_000) = 70_000,
    initial_infected: Int(410, 440) = 410,
    initial_recovered: Int(4, 6) = 4,
    initial_dead: Int(1, 100) = 1,
    delta: Real(0.03, 0.06) = 0.03,
    beta: Real(0.3, 0.4) = 0.3,
    theta: Real(0.004, 0.008) = 0.004,
) -> Tuple[float, pandas.DataFrame]:
    # Normalize beta for the number of individuals
    beta = beta / n

    day_max = 0
    mc_step = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    seeds = list()
    evolution = np.zeros([4, n_seeds, t_total])

    while mc_step < n_seeds:
        current_seed += 1
        result = gillespie_simulation(
            current_seed,
            n,
            n_t_steps,
            initial_infected,
            initial_recovered,
            initial_dead,
            t_total,
            delta,
            beta,
            theta,
        )

        day_max = max(day_max, result.day_max)

        if check_successful_simulation(result, t_total):
            seeds.append(current_seed)
            susceptible = n - result.infected - result.recovered - result.dead
            evolution[0, mc_step, :] = susceptible
            evolution[1, mc_step, :] = result.infected
            evolution[2, mc_step, :] = result.recovered
            evolution[3, mc_step, :] = result.dead
            mc_step += 1

    evolution_df = pandas.DataFrame(
        [],
        columns=pandas.MultiIndex.from_product(
            [["susceptible", "infected", "recovered", "dead"], seeds]
        ),
    )
    for step, seed in enumerate(seeds):
        evolution_df[("susceptible", seed)] = evolution[0, step]
        evolution_df[("infected", seed)] = evolution[1, step]
        evolution_df[("recovered", seed)] = evolution[2, step]
        evolution_df[("dead", seed)] = evolution[3, step]

    cost = get_cost(time_series, evolution[1], evolution[2], evolution[3], metric)
    if scale_cost:
        cost = cost / len(time_series) * 100
    # Report to optilog the cost
    print(f"GGA SUCCESS {cost}")

    return cost, evolution_df


def gillespie_simulation(
    seed: int,
    n: int,
    n_t_steps: int,
    initial_infected: int,
    initial_recovered: int,
    initial_dead: int,
    t_total: int,
    delta: float,
    beta: float,
    theta: float,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    comp = Compartments(n, n_t_steps, initial_infected, initial_recovered, initial_dead)

    t_step, time = 0, 0

    # Time loop
    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(t_step, time, comp, delta, beta, theta)
    # -------------------------

    day_max, infected = utils.day_data(comp.T[:t_step], comp.I[:t_step], t_total)
    _, recovered = utils.day_data(comp.T[:t_step], comp.R[:t_step], t_total)
    _, dead = utils.day_data(comp.T[:t_step], comp.D[:t_step], t_total)

    return Result(infected, recovered, dead, day_max)


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, n, n_t_steps, initial_infected, initial_recovered, initial_dead):
        """Initialization"""
        self.S = np.zeros(n_t_steps, dtype=int)
        self.I = np.zeros(n_t_steps, dtype=int)
        self.R = np.zeros(n_t_steps, dtype=int)
        self.D = np.zeros(n_t_steps, dtype=int)
        self.T = np.zeros(n_t_steps)
        self.I[0] = initial_infected
        self.R[0] = initial_recovered
        self.D[0] = initial_dead
        self.S[0] = n - initial_infected - initial_recovered - initial_dead
        self.T[0] = 0

    def infect(self, t_step):
        """Infection"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]
        self.D[t_step] = self.D[t_step - 1]

    def recover(self, t_step):
        """Recovery"""
        self.S[t_step] = self.S[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1
        self.D[t_step] = self.D[t_step - 1]

    def die(self, t_step):
        """Death"""
        self.S[t_step] = self.S[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1]
        self.D[t_step] = self.D[t_step - 1] + 1


def gillespie(t_step, time, comp, delta, beta, theta):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    lambda_sum = (delta + beta * comp.S[t_step]) * comp.I[t_step]
    probs = {}
    probs["heal"] = delta * comp.I[t_step] / lambda_sum
    probs["die"] = theta

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time

    gillespie_step(t_step, comp, probs)
    return t_step, time


def gillespie_step(t_step, comp, probs):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < probs["heal"]:
        random = np.random.random()
        if random < probs["die"]:
            comp.die(t_step)
        else:
            comp.recover(t_step)
    else:
        comp.infect(t_step)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta, "delta": args.delta, "theta": args.theta}
    return t_total, time_series, rates


def main(args):
    t_total, time_series, rates = parameters_init(args)
    print(f"r = {rates['beta'] / args.n}", file=sys.stderr)
    print(f"a = {rates['delta']*(1-rates['theta'])}", file=sys.stderr)
    print(f"d = {rates['delta']*rates['theta']}", file=sys.stderr)

    return sird(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.n_t_steps,
        args.metric,
        args.scale_cost,
        n=args.n,  # due to a bug, naming the configurable parameters is mandatory
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        initial_dead=args.initial_dead,
        delta=rates["delta"],
        beta=rates["beta"],
        theta=rates["theta"],
    )
