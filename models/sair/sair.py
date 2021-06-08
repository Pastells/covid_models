"""
Stochastic mean-field SAIR model using the Gillespie algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta/N*I(t)*S(t) \n
dA(t)/dt =   beta_a/N*A(t)*S(t) + beta/N*I(t)*S(t) -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta * I(t)                          + alpha*A(t)\n
dR(t)/dt =   delta * I(t)                          + delta_a * A(t)
"""

import random
from collections import namedtuple

import numpy as np
from optilog.autocfg import ac, Int, Real

from ..utils import utils, config


Result = namedtuple("Result", "infected day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series: np.ndarray, infected, t_total, day_max, n_seeds, metric):
    var_m = utils.mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], var_m, metric)


@ac
def sair(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    n_t_steps: int,
    metric,
    n: Int(70000, 90000) = 70000,
    initial_infected: Int(410, 440) = 410,
    initial_recovered: Int(4, 6) = 4,
    initial_asymptomatic: Int(0, 100) = 0,
    alpha: Real(0.05, 2.0) = 0.05,
    delta_a: Real(0.05, 1.0) = 0.05,
    delta: Real(0.03, 0.06) = 0.03,
    beta_a: Real(0.05, 1.0) = 0.05,
    beta: Real(0.3, 0.4) = 0.3,
):
    # Normalize beta and beta_a for the number of individuals
    rates = {
        "beta_a": beta_a / n,
        "beta": beta / n,
        "delta_a": delta_a,
        "delta": delta,
        "alpha": alpha,
    }

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
            initial_asymptomatic,
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
    initial_asymptomatic: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    rates: dict,
    day_max: int,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    # initialization
    comp = Compartments(
        n, n_t_steps, initial_asymptomatic, initial_infected, initial_recovered
    )

    infected = np.zeros(t_total, dtype=int)
    infected[0] = initial_infected

    t_step, time = 0, 0

    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(
            t_step,
            time,
            comp,
            rates=rates,
        )

    day_max = utils.day_data(comp.T[:t_step], comp.I[:t_step], infected, day_max)

    return Result(infected, day_max)


class Compartments:
    """Compartments for SAIR model"""

    def __init__(
        self, n, n_t_steps, initial_asymptomatic, initial_infected, initial_recovered
    ):
        """Initialization"""
        self.S = np.zeros(n_t_steps, dtype=int)
        self.A = np.zeros(n_t_steps, dtype=int)
        self.I = np.zeros(n_t_steps, dtype=int)
        self.R = np.zeros(n_t_steps, dtype=int)
        self.T = np.zeros(n_t_steps)
        self.A[0] = initial_asymptomatic
        self.I[0] = initial_infected
        self.R[0] = initial_recovered
        self.S[0] = n - initial_infected - initial_recovered - initial_asymptomatic
        self.T[0] = 0
        self.I_cum = np.zeros(n_t_steps, dtype=int)
        self.I_cum[0] = initial_infected

    def turn_asymptomatic(self, t_step):
        """Turn asymptomatic s->a"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.A[t_step] = self.A[t_step - 1] + 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def turn_infectious(self, t_step):
        """Turn infectious a->i"""
        self.S[t_step] = self.S[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + 1

    def recover_a(self, t_step):
        """Recovery a->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def recover_i(self, t_step):
        """Recovery i->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.A[t_step] = self.A[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]


def gillespie(t_step, time, comp, rates):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (rates["alpha"] + rates["delta_a"]) * comp.A[t_step]
        + rates["delta"] * comp.I[t_step]
        + (rates["beta_a"] * comp.A[t_step] + rates["beta"] * comp.I[t_step])
        * comp.S[t_step]
    )

    probabilities = {
        "heal_a": rates["delta_a"] * comp.A[t_step] / lambda_sum,
        "heal_i": rates["delta"] * comp.I[t_step] / lambda_sum,
        "asymptomatic": rates["alpha"] * comp.A[t_step] / lambda_sum,
    }

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time

    gillespie_step(t_step, comp, probabilities)
    return t_step, time


def gillespie_step(t_step, comp, probabilities):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random_value = np.random.random()

    if random_value < probabilities["heal_a"]:
        comp.recover_a(t_step)
    elif random_value < (probabilities["heal_a"] + probabilities["heal_i"]):
        comp.recover_i(t_step)
    elif random_value < (
        probabilities["heal_a"]
        + probabilities["heal_i"]
        + probabilities["asymptomatic"]
    ):
        comp.turn_infectious(t_step)
    else:
        comp.turn_asymptomatic(t_step)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    sair(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.n_t_steps,
        args.metric,
        n=args.n,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        initial_asymptomatic=args.initial_asymptomatic,
        alpha=args.alpha,
        delta_a=args.delta_a,
        delta=args.delta,
        beta_a=args.beta_a,
        beta=args.beta,
    )
