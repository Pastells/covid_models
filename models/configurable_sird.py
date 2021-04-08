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

import numpy as np
from optilog.autocfg import ac, Int, Real

from utils import utils, config


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
         theta: Real(0.004, 0.008) = 0.004
        ):
    # Normalize beta for the number of individuals
    beta = beta / n

    # results per day and seed
    I_day = np.zeros([n_seeds, t_total], dtype=int)
    R_day = np.zeros([n_seeds, t_total], dtype=int)
    D_day = np.zeros([n_seeds, t_total], dtype=int)

    day_max = 0
    check_realization_alive = t_total - 1
    random.seed(seed)
    np.random.seed(seed)
    # =========================
    # MC loop
    # =========================
    mc_step = 0
    while mc_step < n_seeds:
        # print(mc_step)
        I_day[mc_step], R_day[mc_step], D_day[mc_step], day_max = gillespie_simulation(
            n, n_t_steps,
            initial_infected, initial_recovered, initial_dead,
            t_total,
            delta, beta, theta,
            day_max
        )
        if I_day[mc_step, check_realization_alive] != 0:
            mc_step += 1
        # =========================

    I_m = utils.mean_std(I_day)
    R_m = utils.mean_std(R_day)
    D_m = utils.mean_std(D_day)

    if config.CUMULATIVE is True:
        utils.cost_func(time_series[:, 3], I_m, metric)
    else:
        # utils.cost_func(time_series[:, 0], I_m)
        cost = utils.cost_return(time_series[:, 0], I_m, metric)
        sys.stdout.write(f"cost_I = {cost}\n")

    _cost = utils.cost_return(time_series[:, 1], R_m, metric)
    sys.stdout.write(f"cost_R = {_cost}\n")
    cost += _cost
    _cost = utils.cost_return(time_series[:, 2], D_m, metric)
    sys.stdout.write(f"cost_D = {_cost}\n")
    cost += _cost
    sys.stdout.write(f"GGA SUCCESS {cost}\n")

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
                         day_max: int):
    comp = Compartments(n, n_t_steps, initial_infected, initial_recovered, initial_dead)

    I_day = np.zeros(t_total, dtype=int)
    R_day = np.zeros(t_total, dtype=int)
    D_day = np.zeros(t_total, dtype=int)
    I_day[0] = initial_infected
    R_day[0] = initial_recovered
    D_day[0] = initial_dead
    t_step, time = 0, 0

    # Time loop
    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(t_step, time, comp, delta, beta, theta)
    # -------------------------

    if config.CUMULATIVE is True:
        i_var = comp.I_cum
    else:
        i_var = comp.I

    day_max = utils.day_data(comp.T[:t_step], i_var[:t_step], I_day, day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.R[:t_step], R_day, day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.D[:t_step], D_day, day_max)

    return [I_day, R_day, D_day, day_max]


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, n, n_t_steps, initial_infected, initial_recovered, initial_dead):
        """Initialization"""
        self.S = np.zeros(n_t_steps, dtype=int)
        self.I = np.zeros(n_t_steps, dtype=int)
        self.R = np.zeros(n_t_steps, dtype=int)
        self.D = np.zeros(n_t_steps, dtype=int)
        self.T = np.zeros(n_t_steps)
        self.I[0] = initial_infected  # I_0
        self.R[0] = initial_recovered  # R_0
        self.D[0] = initial_dead  # D_0
        self.S[0] = n - initial_infected - initial_recovered - initial_dead
        self.T[0] = 0
        self.I_cum = np.zeros(n_t_steps, dtype=int)
        self.I_cum[0] = initial_infected

    def infect(self, t_step):
        """Infection"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]
        self.D[t_step] = self.D[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + 1

    def recover(self, t_step):
        """Recovery"""
        self.S[t_step] = self.S[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1
        self.D[t_step] = self.D[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def die(self, t_step):
        """Death"""
        self.S[t_step] = self.S[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1]
        self.D[t_step] = self.D[t_step - 1] + 1
        self.I_cum[t_step] = self.I_cum[t_step - 1]


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


def parsing():
    """input parameters"""

    description = "stochastic mean-field SIRD model using the Gillespie algorithm. \
            Dependencies: config.py, utils.py"

    parser = utils.ParserCommon(description)
    parser.n()
    parser.sir()
    parser.dead()

    return parser.parse_args()


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta, "delta": args.delta, "theta": args.theta}
    return t_total, time_series, rates


def main():
    args = parsing()
    t_total, time_series, rates = parameters_init(args)
    sys.stdout.write(f"r = {rates['beta']}\n")
    sys.stdout.write(f"a = {rates['delta']*(1-rates['theta'])}\n")
    sys.stdout.write(f"d = {rates['delta']*rates['theta']}\n")
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
        sys.stdout.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc(ex)