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

from utils.utils import mean_alive
from ..utils import utils, config


Result = namedtuple("Result", "infected asymptomatic recovered day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series: np.ndarray, infected, t_total, day_max, n_seeds, metric):
    var_m = mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], var_m, metric)


@ac
def sair(time_series: np.ndarray,
         seed: int,
         n_seeds: int,
         t_total: int,
         n_t_steps: int,
         metric,
         n: Int(70000, 90000) = config.N,
         initial_infected: Int(410, 440) = config.I_0,
         initial_recovered: Int(4, 6) = config.R_0,
         initial_asymptomatic: Int(0, 100) = config.A_0,
         alpha: Real(0.05, 2.0) = config.ALPHA,
         delta_a: Real(0.05, 1.0) = config.DELTA_A,
         delta: Real(0.03, 0.06) = config.DELTA,
         beta_a: Real(0.05, 1.0) = config.BETA_A,
         beta: Real(0.3, 0.4) = config.BETA):
    # Normalize beta and beta_a for the number of individuals
    beta = beta / n
    beta_a = beta_a / n

    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=int)

    mc_step = 0
    day_max = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    results = list()

    while mc_step < n_seeds:
        current_seed += 1
        result = gillespie_simulation(
            current_seed,
            n, n_t_steps,
            initial_asymptomatic, initial_infected, initial_recovered,
            t_total,
            alpha, delta_a, delta, beta_a, beta,
            day_max
        )
        day_max = result.day_max

        if check_successful_simulation(result, t_total):
            mc_step += 1
            results.append(result)

    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=int)
    recovered = np.zeros([n_seeds, t_total], dtype=int)
    asymptomatic = np.zeros([n_seeds, t_total], dtype=int)

    for mc_step, result in enumerate(results):
        infected[mc_step] = result.infected
        recovered[mc_step] = result.recovered
        asymptomatic[mc_step] = result.asymptomatic

    cost = get_cost(time_series, infected, t_total, day_max, n_seeds, metric)
    print(f"GGA SUCCESS {cost}")
    return cost


def gillespie_simulation(seed,
                         n,
                         n_t_steps,
                         initial_asymptomatic,
                         initial_infected,
                         initial_recovered,
                         t_total,
                         alpha,
                         delta_a,
                         delta,
                         beta_a,
                         beta,
                         day_max) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    # initialization
    comp = Compartments(n, n_t_steps, initial_asymptomatic,
                        initial_infected, initial_recovered)

    infected = np.zeros(t_total, dtype=int)
    asymptomatic = np.zeros(t_total, dtype=int)
    recovered = np.zeros(t_total, dtype=int)

    infected[0] = initial_infected
    asymptomatic[0] = initial_asymptomatic
    recovered[0] = initial_recovered

    t_step, time = 0, 0

    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(t_step, time, comp, alpha=alpha, beta=beta,
                                 delta_a=delta_a, delta=delta, beta_a=beta_a)

    day_max = utils.day_data(comp.T[:t_step], comp.A[:t_step], asymptomatic,
                             day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.R[:t_step], recovered,
                             day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.I[:t_step], infected,
                             day_max)

    return Result(infected, asymptomatic, recovered, day_max)


class Compartments:
    """Compartments for SAIR model"""

    def __init__(self, n, n_t_steps,
                 initial_asymptomatic, initial_infected, initial_recovered):
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


def gillespie(t_step, time, comp, alpha, delta_a, delta, beta_a, beta):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (alpha + delta_a) * comp.A[t_step]
        + delta * comp.I[t_step]
        + (beta_a * comp.A[t_step] + beta * comp.I[t_step])
        * comp.S[t_step]
    )

    probabilities = {
        "heal_a": delta_a * comp.A[t_step] / lambda_sum,
        "heal_i": delta * comp.I[t_step] / lambda_sum,
        "asymptomatic": alpha * comp.A[t_step] / lambda_sum
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
    elif random_value < (probabilities["heal_a"] + probabilities["heal_i"]
                         + probabilities["asymptomatic"]):
        comp.turn_infectious(t_step)
    else:
        comp.turn_asymptomatic(t_step)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {
        "beta_a": args.beta_a,
        "beta": args.beta,
        "delta_a": args.delta_a,
        "delta": args.delta,
        "alpha": args.alpha,
    }

    return t_total, time_series, rates


def main(args):
    t_total, time_series, rates = parameters_init(args)
    sair(time_series, args.seed, args.mc_nseed, t_total, args.n_t_steps, args.metric,
         alpha=rates["alpha"], delta_a=rates["delta_a"], delta=rates["delta"],
         beta_a=rates["beta_a"], beta=rates["beta"])
