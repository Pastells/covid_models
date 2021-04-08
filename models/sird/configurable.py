"""
Stochastic mean-field SIRD model using the Gillespie algorithm

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                   delta*(1-theta) * I(t)
dD(t)/dt =                   delta*theta * I(t)
"""

import random
import sys
import traceback
import os
from collections import namedtuple

import numpy as np
from optilog.autocfg import ac, Int, Real

# this is required as running > if __name__ == "__main__"
# from inside the module itself is an antipattern and we
# must force the path to the project top-level module
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from models.utils import utils
from models.sird.sird import gillespie_step, parsing


# Renamed:
# - main -> sird
# - main_loop -> gillespie_simulation

Result = namedtuple("Result", "infected recovered dead day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series, infected, recovered, dead, metric):
    mean_infected = utils.mean_std(infected)
    mean_recovered = utils.mean_std(recovered)
    mean_dead = utils.mean_std(dead)

    infected_cost = utils.cost_return(time_series[:, 0], mean_infected, metric)
    print(f"cost_infected = {infected_cost}")
    recovered_cost = utils.cost_return(time_series[:, 1], mean_recovered, metric)
    print(f"cost_recovered = {recovered_cost}")
    dead_cost = utils.cost_return(time_series[:, 2], mean_dead, metric)
    print(f"cost_dead = {dead_cost}")

    return infected_cost + recovered_cost + dead_cost


@ac
def sird(time_series: np.ndarray,
         seed: int,
         n_seeds: int,
         t_total: int,
         n_t_steps: int,
         metric,
         n: Int(70000, 90000) = 70000,
         initial_infected: Int(410, 440) = 410,
         initial_recovered: Int(4, 6) = 4,
         initial_dead: Int(1, 100) = 1,
         delta: Real(0.03, 0.06) = 0.03,
         beta: Real(0.3, 0.4) = 0.3,
         theta: Real(0.004, 0.008) = 0.004):
    random.seed(seed)
    np.random.seed(seed)

    # Normalize beta for the number of individuals
    beta = beta / n

    day_max = 0
    mc_step = 0

    results = []
    while mc_step < n_seeds:
        result = gillespie_simulation(
            n, n_t_steps,
            initial_infected, initial_recovered, initial_dead,
            t_total,
            delta, beta, theta,
            day_max
        )

        day_max = result.day_max

        if check_successful_simulation(result, t_total):
            mc_step += 1
            results.append(result)
    # =========================

    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=int)
    recovered = np.zeros([n_seeds, t_total], dtype=int)
    dead = np.zeros([n_seeds, t_total], dtype=int)
    for mc_step, result in enumerate(results):
        infected[mc_step] = result.infected
        recovered[mc_step] = result.recovered
        dead[mc_step] = result.dead

    cost = get_cost(time_series, infected, recovered, dead, metric)
    print(f"GGA SUCCESS {cost}")

    return cost


def gillespie_simulation(n: int,
                         n_t_steps: int,
                         initial_infected: int,
                         initial_recovered: int,
                         initial_dead: int,
                         t_total,  # TODO
                         delta: float,
                         beta: float,
                         theta: float,
                         day_max: int) -> Result:
    comp = Compartments(n, n_t_steps,
                        initial_infected, initial_recovered, initial_dead)

    infected = np.zeros(t_total, dtype=int)
    recovered = np.zeros(t_total, dtype=int)
    dead = np.zeros(t_total, dtype=int)

    infected[0] = initial_infected
    recovered[0] = initial_recovered
    dead[0] = initial_dead

    t_step, time = 0, 0

    # Time loop
    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(t_step, time, comp, delta, beta, theta)
    # -------------------------

    day_max = utils.day_data(comp.T[:t_step], comp.I[:t_step], infected, day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.R[:t_step], recovered, day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.D[:t_step], dead, day_max)

    return Result(infected, recovered, dead, day_max)


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, n, n_t_steps,
                 initial_infected, initial_recovered, initial_dead):
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


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta, "delta": args.delta, "theta": args.theta}
    return t_total, time_series, rates


def main():
    args = parsing()
    t_total, time_series, rates = parameters_init(args)
    print(f"r = {rates['beta']}")
    print(f"a = {rates['delta']*(1-rates['theta'])}")
    print(f"d = {rates['delta']*rates['theta']}")
    sird(time_series, args.seed, args.mc_nseed, t_total, args.n_t_steps, args.metric,
         n=args.n,  # due to a bug, naming the configurable parameters is mandatory
         initial_infected=args.I_0,
         initial_recovered=args.R_0,
         initial_dead=args.D_0,
         delta=rates["delta"],
         beta=rates["beta"],
         theta=rates["theta"])


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"{repr(ex)}")
        print(f"GGA CRASHED {1e20}")
        traceback.print_exc(ex)