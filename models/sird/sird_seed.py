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


from .sird import parsing, parameters_init, Compartments, gillespie

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


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc()
