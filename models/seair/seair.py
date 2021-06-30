"""
Stochastic mean-field SAIR model using the Gillespie algorithm

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta/N*I(t)*S(t) \n
dE(t)/dt =   beta_a/N*A(t)*S(t) + beta/N*I(t)*S(t) - epsilon*E(t)\n
dA(t)/dt =   epsilon * E(t)   -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta * I(t)  + alpha*A(t)\n
dR(t)/dt =   delta * I(t)  + delta_a * A(t)
"""

import random
from collections import namedtuple

import numpy as np
from optilog.autocfg import ac, Int, Real

from ..utils import utils


Result = namedtuple("Result", "infected day_max")


def get_cost(time_series, infected, t_total, day_max, n_seeds, metric):
    mean_infected = utils.mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], mean_infected, metric)


@ac
def seair(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    n_t_steps: int,
    metric,
    n: Int(70000, 90000) = 70000,
    initial_exposed: Int(0, 1) = 1,  # TODO
    initial_asymptomatic: Int(0, 1) = 1,  # TODO
    initial_infected: Int(410, 440) = 410,
    initial_recovered: Int(4, 6) = 4,
    alpha: Real(0.0, 1.0) = 0.1,  # TODO
    delta_a: Real(0.0, 1.0) = 0.1,  # TODO
    epsilon: Real(0.0, 1.0) = 0.1,  # TODO
    delta: Real(0.0, 1.0) = 0.1,  # TODO
    beta_a: Real(0.0, 1.0) = 0.1,  # TODO
    beta: Real(0.0, 1.0) = 0.1,  # TODO
):
    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=int)

    day_max = 0
    # =========================
    # MC loop
    # =========================

    for mc_seed in range(seed, seed + n_seeds):
        random.seed(mc_seed)
        np.random.seed(mc_seed)
        mc_step = mc_seed - seed
        result = gillespie_simulation(
            n,
            n_t_steps,
            initial_exposed,
            initial_asymptomatic,
            initial_infected,
            initial_recovered,
            t_total,
            alpha,
            delta_a,
            epsilon,
            delta,
            beta_a,
            beta,
            day_max,
        )
        day_max = result.day_max
        infected[mc_step] = result.infected

    cost = get_cost(time_series, infected, t_total, day_max, n_seeds, metric)
    print(f"GGA SUCCESS {cost}")
    return cost


def gillespie_simulation(
    n: int,
    n_t_steps: int,
    initial_exposed,
    initial_asymptomatic,
    initial_infected,
    initial_recovered,
    t_total: int,
    alpha,
    delta_a,
    epsilon,
    delta,
    beta_a,
    beta,
    day_max: int,
):
    comp = Compartments(
        n=n,
        n_t_steps=n_t_steps,
        initial_exposed=initial_exposed,
        initial_asymptomatic=initial_asymptomatic,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
    )

    infected = np.zeros(t_total, dtype=int)
    t_step, time = 0, 0

    # Time loop
    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(
            t_step, time, comp, alpha, delta_a, epsilon, delta, beta_a, beta
        )

    day_max = utils.day_data(comp.T[:t_step], comp.I[:t_step], infected, day_max)

    return Result(infected, day_max)


class Compartments:
    """Compartments for SAIR model"""

    def __init__(
        self,
        n,
        n_t_steps,
        initial_exposed,
        initial_asymptomatic,
        initial_infected,
        initial_recovered,
    ):
        """Initialization"""
        self.S = np.zeros(n_t_steps, dtype=int)
        self.E = np.zeros(n_t_steps, dtype=int)
        self.A = np.zeros(n_t_steps, dtype=int)
        self.I = np.zeros(n_t_steps, dtype=int)
        self.R = np.zeros(n_t_steps, dtype=int)
        self.T = np.zeros(n_t_steps)
        self.E[0] = initial_exposed
        self.A[0] = initial_asymptomatic
        self.I[0] = initial_infected
        self.R[0] = initial_recovered
        self.S[0] = (
            n
            - initial_infected
            - initial_recovered
            - initial_asymptomatic
            - initial_exposed
        )
        self.T[0] = 0
        self.I_cum = np.zeros(n_t_steps, dtype=int)
        self.I_cum[0] = initial_infected

    def turn_exposed(self, t_step):
        """Expose s->e"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.E[t_step] = self.E[t_step - 1] + 1
        self.A[t_step] = self.A[t_step - 1]
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def turn_asymptomatic(self, t_step):
        """Turn asymptomatic e->a"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1] - 1
        self.A[t_step] = self.A[t_step - 1] + 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def turn_infectious(self, t_step):
        """Turn infectious a->i"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + 1

    def recover_a(self, t_step):
        """Recovery a->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.A[t_step] = self.A[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def recover_i(self, t_step):
        """Recovery i->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.A[t_step] = self.A[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]


def gillespie(t_step, time, comp, alpha, delta_a, epsilon, delta, beta_a, beta):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (alpha + delta_a) * comp.A[t_step]
        + epsilon * comp.E[t_step]
        + delta * comp.I[t_step]
        + (beta_a * comp.A[t_step] + beta * comp.I[t_step]) * comp.S[t_step]
    )

    probs = {}
    probs["heal_a"] = delta_a * comp.A[t_step] / lambda_sum
    probs["heal_i"] = delta * comp.I[t_step] / lambda_sum
    probs["asymptomatic"] = alpha * comp.A[t_step] / lambda_sum
    probs["exposed"] = epsilon * comp.E[t_step] / lambda_sum

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

    if random < probs["heal_a"]:
        comp.recover_a(t_step)
    elif random < (probs["heal_a"] + probs["heal_i"]):
        comp.recover_i(t_step)
    elif random < (probs["heal_a"] + probs["heal_i"] + probs["asymptomatic"]):
        comp.turn_infectious(t_step)
    elif random < (
        probs["heal_a"] + probs["heal_i"] + probs["asymptomatic"] + probs["exposed"]
    ):
        comp.turn_asymptomatic(t_step)
    else:
        comp.turn_exposed(t_step)


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {
        "beta_a": args.beta_a / args.n,
        "beta": args.beta / args.n,
        "delta_a": args.delta_a,
        "delta": args.delta,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
    }

    return t_total, time_series, rates


def main(args):
    t_total, time_series, rates = parameters_init(args)

    seair(
        time_series=time_series,
        seed=args.seed,
        n_seeds=args.mc_nseed,
        t_total=t_total,
        n_t_steps=args.n_t_steps,
        metric=args.metric,
        n=args.n,
        initial_exposed=args.initial_exposed,
        initial_asymptomatic=args.initial_asymptomatic,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        alpha=rates["alpha"],
        delta_a=rates["delta_a"],
        epsilon=rates["epsilon"],
        delta=rates["delta"],
        beta_a=rates["beta_a"],
        beta=rates["beta"],
    )
