"""
Stochastic mean-field SIRD model using the Gillespie algorithm
runs in parallel using joblib

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                   delta*(1-theta) * I(t)
dD(t)/dt =                   delta*theta * I(t)
"""

import random
import sys
from typing import Tuple

import numpy as np
import pandas
from joblib import Parallel, delayed
from ..utils import utils, config
from .sird import sird as sird_sequential


def sird(args) -> Tuple[float, pandas.DataFrame]:
    if args.sequential:
        return sird_sequential(args)

    t_total, time_series, rates = parameters_init(args)

    sys.stdout.write(f"r = {rates['beta']}\n")
    sys.stdout.write(f"a = {rates['delta']*(1-rates['theta'])}\n")
    sys.stdout.write(f"d = {rates['delta']*rates['theta']}\n")

    # results per day and seed
    I_day = np.zeros([args.mc_nseed, t_total], dtype=int)
    R_day = np.zeros([args.mc_nseed, t_total], dtype=int)
    D_day = np.zeros([args.mc_nseed, t_total], dtype=int)

    random.seed(args.seed)
    np.random.seed(args.seed)
    check_realization_alive = t_total - 1

    # =========================
    # MC loop
    # =========================

    # Obtain number of cores from machine (doesn't check if they are available)
    # num_cores = multiprocessing.cpu_count()
    num_cores = 6

    # Reuse pool of workers in batches with size a multiple of num_cores
    BATCH_SIZE = 2
    # Threshold to stop
    BAD_REALIZATIONS_THRES = BATCH_SIZE * num_cores // 2 * 3
    with Parallel(n_jobs=num_cores) as parallel:
        accum = 0
        results = []
        while (1 + accum) * num_cores + 1 < args.mc_nseed:
            # print(accum)
            ran = range(
                args.seed + accum * BATCH_SIZE * num_cores,
                min(
                    args.seed + (accum + 1) * BATCH_SIZE * num_cores,
                    args.seed + args.mc_nseed,
                ),
            )

            _results = parallel(
                delayed(main_loop)(args, t_total, rates, time_series[-1, 0])
                for mc_seed in ran
            )
            results.extend(_results)

            bad_realizations = 0
            for mc_seed, result in enumerate(_results):
                bad_realizations += result[4]
            if bad_realizations >= BAD_REALIZATIONS_THRES:
                sys.stdout.write(
                    f"{bad_realizations} bad realizations out of {BATCH_SIZE * num_cores}\n"
                )
                break
                # raise ValueError(
                # "Bad realizations: (I_day_max < I_data / 2) or (I_day_max > I_data x 2)"
                # )

            accum += 1

    # get daily data from results list
    days_max = []
    for mc_seed, result in enumerate(results):
        I_day[mc_seed] = result[0]
        R_day[mc_seed] = result[1]
        D_day[mc_seed] = result[2]
        days_max.append(result[3])
    # =========================

    day_max = max(days_max)

    I_m, R_m, D_m = utils.mean_alive_rd(
        I_day, t_total, day_max, args.mc_nseed, R_day, D_day
    )

    # Compute and print cost functions for I, R and D

    cost = 0
    if config.CUMULATIVE is True:
        utils.cost_func(time_series[:, 3], I_m, args.metric)
    else:
        # utils.cost_func(time_series[:, 0], I_m)
        _cost = utils.cost_return(time_series[:, 0], I_m, args.metric)
        cost += _cost
        # sys.stdout.write(f"cost_I = {cost}\n")

    # _cost = utils.cost_return(np.diff(time_series[:, 0]), np.diff(I_m), args.metric)
    # sys.stdout.write(f"cost_I_der = {_cost}\n")
    # cost += _cost

    _cost = utils.cost_return(time_series[:, 1], R_m, args.metric)
    # sys.stdout.write(f"cost_R = {_cost}\n")
    cost += _cost

    _cost = utils.cost_return(time_series[:, 2], D_m, args.metric)
    # sys.stdout.write(f"cost_D = {_cost}\n")
    cost += _cost

    sys.stdout.write(f"GGA SUCCESS {cost}\n")

    with open("costs.dat", "a") as f:
        f.write(f"{cost}\n")

    if args.cost_day is not None:
        print(I_m[args.cost_day - args.day_min :, 0])
        print(R_m[args.cost_day - args.day_min :, 0])
        print(D_m[args.cost_day - args.day_min :, 0])

    if args.save is not None:
        save = args.save
        args.save = save + "R"
        utils.saving(args, R_m, day_max, var="R")
        args.save = save + "D"
        utils.saving(args, D_m, day_max, var="D")
        args.save = save
        utils.saving(args, I_m, day_max)

    if args.plot:
        from ..utils import plots

        plots.plotting(args, day_max, I_m, R_m, D_m)


def main_loop(args, t_total, rates, last_day_inf):
    """Function to be run in parallel"""

    # -------------------------
    # initialization

    comp = Compartments(args)

    # results per day and seed
    I_day = np.zeros(t_total, dtype=int)
    R_day = np.zeros(t_total, dtype=int)
    D_day = np.zeros(t_total, dtype=int)
    I_day[0] = args.initial_infected
    R_day[0] = args.initial_recovered
    D_day[0] = args.initial_dead
    t_step, time = 0, 0

    # Time loop
    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(t_step, time, comp, rates)
    # -------------------------

    if config.CUMULATIVE is True:
        i_var = comp.I_cum
    else:
        i_var = comp.I

    day_max, infected = utils.day_data(comp.T[:t_step], i_var[:t_step], t_total)
    _, recovered = utils.day_data(comp.T[:t_step], comp.R[:t_step], t_total)
    _, dead = utils.day_data(comp.T[:t_step], comp.D[:t_step], t_total)

    bad_realization = 0
    if infected[-1] < last_day_inf / 2 or infected[-1] > last_day_inf * 2:
        bad_realization = 1

    return [I_day, R_day, D_day, day_max, bad_realization]


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta / args.n, "delta": args.delta, "theta": args.theta}
    return t_total, time_series, rates


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, args):
        """Initialization"""
        self.S = np.zeros(args.n_t_steps, dtype=int)
        self.I = np.zeros(args.n_t_steps, dtype=int)
        self.R = np.zeros(args.n_t_steps, dtype=int)
        self.D = np.zeros(args.n_t_steps, dtype=int)
        self.T = np.zeros(args.n_t_steps)
        self.I[0] = args.initial_infected
        self.R[0] = args.initial_recovered
        self.D[0] = args.initial_dead
        self.S[0] = (
            args.n - args.initial_infected - args.initial_recovered - args.initial_dead
        )
        self.T[0] = 0
        self.I_cum = np.zeros(args.n_t_steps, dtype=int)
        self.I_cum[0] = args.initial_infected

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


def gillespie(t_step, time, comp, rates):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    lambda_sum = (rates["delta"] + rates["beta"] * comp.S[t_step]) * comp.I[t_step]
    probs = {}
    probs["heal"] = rates["delta"] * comp.I[t_step] / lambda_sum
    probs["die"] = rates["theta"]

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


def main(args):
    sird(args)
