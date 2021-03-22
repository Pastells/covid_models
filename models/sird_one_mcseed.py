"""
Stochastic mean-field SIRD model using the Gillespie algorithm

Pol Pastells, 2020

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
from utils import utils, config

# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def main():
    args = parsing()
    t_total, time_series, rates = parameters_init(args)
    # print(args)
    sys.stdout.write(f"r = {rates['beta']}\n")
    sys.stdout.write(f"a = {rates['delta']*(1-rates['theta'])}\n")
    sys.stdout.write(f"d = {rates['delta']*rates['theta']}\n")

    # results per day
    I_day = np.zeros([t_total], dtype=int)
    R_day = np.zeros([t_total], dtype=int)
    D_day = np.zeros([t_total], dtype=int)
    I_m = np.zeros([t_total, 2], dtype=int)
    R_m = np.zeros([t_total, 2], dtype=int)
    D_m = np.zeros([t_total, 2], dtype=int)

    day_max = 0
    random.seed(args.seed)
    np.random.seed(args.seed)

    # -------------------------
    # initialization

    comp = Compartments(args)

    I_day[0] = args.I_0
    R_day[0] = args.R_0
    D_day[0] = args.D_0
    # S_day[0]=s[0]
    t_step, time = 0, 0

    # Time loop
    while comp.I[t_step] > 0 and time < t_total:
        t_step, time = gillespie(t_step, time, comp, rates)
    # -------------------------

    if config.CUMULATIVE is True:
        i_var = comp.I_cum
    else:
        i_var = comp.I

    day_max = utils.day_data(comp.T[:t_step], i_var[:t_step], I_day, day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.R[:t_step], R_day, day_max)
    day_max = utils.day_data(comp.T[:t_step], comp.D[:t_step], D_day, day_max)

    I_m[:, 0] = I_day
    R_m[:, 0] = R_day
    D_m[:, 0] = D_day

    # =========================

    check_realization_alive = t_total - 1
    if I_day[check_realization_alive] == 0:
        cost = -1
    else:
        if config.CUMULATIVE is True:
            utils.cost_func(time_series[:, 3], I_m, args.metric)
        else:
            # utils.cost_func(time_series[:, 0], I_m)
            cost = utils.cost_return(time_series[:, 0], I_m, args.metric)
            sys.stdout.write(f"cost_I = {cost}\n")

        _cost = utils.cost_return(time_series[:, 1], R_m, args.metric)
        sys.stdout.write(f"cost_R = {_cost}\n")
        cost += _cost
        _cost = utils.cost_return(time_series[:, 2], D_m, args.metric)
        sys.stdout.write(f"cost_D = {_cost}\n")
        cost += _cost

    sys.stdout.write(f"GGA SUCCESS {cost}\n")

    if args.save is not None:
        save = args.save
        args.save = save + "R"
        utils.saving(args, R_m, day_max, var="R")
        args.save = save + "D"
        utils.saving(args, D_m, day_max, var="D")
        args.save = save
        utils.saving(args, I_m, day_max)

    if args.plot:
        from utils import plots

        plots.plotting(args, day_max, I_m, R_m, D_m)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""

    description = "stochastic mean-field SIRD model using the Gillespie algorithm. \
        Dependencies: config.py, utils.py"

    parser = utils.ParserCommon(description)
    parser.n()
    parser.sir()
    parser.dead()

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta / args.n, "delta": args.delta, "theta": args.theta}
    return t_total, time_series, rates


# -------------------------


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, args):
        """Initialization"""
        self.S = np.zeros(args.n_t_steps, dtype=int)
        self.I = np.zeros(args.n_t_steps, dtype=int)
        self.R = np.zeros(args.n_t_steps, dtype=int)
        self.D = np.zeros(args.n_t_steps, dtype=int)
        self.T = np.zeros(args.n_t_steps)
        self.I[0] = args.I_0
        self.R[0] = args.R_0
        self.D[0] = args.D_0
        self.S[0] = args.n - args.I_0 - args.R_0 - args.D_0
        self.T[0] = 0
        self.I_cum = np.zeros(args.n_t_steps, dtype=int)
        self.I_cum[0] = args.I_0

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


# -------------------------


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


# -------------------------


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


# ~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc(ex)