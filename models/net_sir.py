"""
Stochastic SIR model with a social network using the event-driven algorithm

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""

import random
import sys
import traceback
import resource
import numpy as np
from event_driven import fast_sir
from utils import utils, utils_net, config


def main():
    args = parsing()
    t_total, time_series, rates = parameters_init(args)
    print(args)

    # results per day and seed
    I_day = np.zeros([args.mc_nseed, t_total], dtype=np.uint32)

    day_max = 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.seed, args.seed + args.mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)
        mc_step = mc_seed - args.seed

        G = utils_net.choose_network(args.n, args.network, args.network_param)
        t, I = fast_sir.fast_SIR(
            G, rates, args.initial_infected, args.initial_recovered, tmax=t_total - 0.95
        )
        I_day[mc_step, 0] = args.initial_infected

        day_max = utils.day_data(t, I, I_day[mc_step], day_max)
        del t, I, G

    # =========================

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    description = "stochastic SIR model with a social network using the event-driven algorithm. \
            Dependencies: config.py, utils.py, utils_net.py, fast_sir.py"

    parser = utils.ParserCommon(description)
    parser.n()
    parser.sir()
    parser.network()

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    rates = {"beta": args.beta / args.network_param, "delta": args.delta}
    return t_total, time_series, rates


# -------------------------

if __name__ == "__main__":
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (int(1024 ** 3 * 5.5), hard))
    try:
        main()
    except MemoryError as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        sys.stdout.write("MemoryError in python\n")
        sys.stdout.write(f"GGA MEMOUT {1e20}\n")
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc()
