"""
Stochastic mean-field SEIR model using the Gillespie algorithm
Pol Pastells, october 2020

equations of the deterministic system
s[t] = S[t-1] - beta1*e[t-1]*s[t-1] - beta2*i[t-1]*s[t-1]
e[t] = E[t-1] + beta1*e[t-1]*s[t-1] + beta2*i[t-1]*s[t-1] - (epsilon+delta1)*e[t-1]
i[t] = I[t-1] + epsilon*e[t-1] - delta2 * I[t-1]
r[t] = R[t-1] + delta1 *e[t-1] + delta2 * I[t-1]
"""

import random
import sys
import traceback
import numpy as np
import utils


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    args = parsing()
    (
        E_0,
        I_0,
        R_0,
        n_t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        ratios,
    ) = parameters_init(args)

    # results per day and seed
    I_day, I_m = (
        np.zeros([mc_nseed, t_total]),
        np.zeros(t_total),
    )

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for seed in range(mc_seed0, mc_seed0 + mc_nseed):
        random.seed(seed)
        np.random.seed(seed)

        # -------------------------
        # initialization
        comp = Compartments(n_t_steps, args)

        # T = np.zeros(n_t_steps)
        # T[0]=0
        I_day[mc_step, 0] = I_0
        t_step, time, day = 0, 0, 1

        # -------------------------
        # Time loop
        # -------------------------
        while comp.I[t_step] > 0.1 and day < t_total:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, comp.I[t_step], I_day[mc_step]
            )
            t_step, time = gillespie(t_total, t_step, time, comp, ratios)
            # print(t_step, time, day, t_total, n_t_steps)

        # -------------------------

        mc_step += 1
    # =========================
    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "seir")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# -------------------------


def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="stochastic mean-field SEIR model using the Gillespie algorithm",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="parameter: fixed number of (effecitve) people [1000,1000000]",
    )
    parser.add_argument(
        "--E_0", type=int, default=0, help="initial number of latent individuals [1,n]"
    )
    parser.add_argument(
        "--I_0",
        type=int,
        default=20,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals [1,n]"
    )
    parser.add_argument(
        "--delta1",
        type=float,
        default=0.01,
        help="parameter: ratio of recovery from latent fase (e->r) [1e-2,1]",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.2,
        help="parameter: ratio of recovery from infected fase (i->r) [1e-2,1]",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.01,
        help="parameter: ratio of infection due to latent [1e-2,1]",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.5,
        help="parameter: ratio of infection due to infected [1e-2,1]",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1,
        help="parameter: ratio of latency (e->i) [1e-2,1]",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for the automatic configuration"
    )
    parser.add_argument(
        "--data", type=str, default="../data/italy_i.csv", help="file with time series"
    )
    parser.add_argument(
        "--day_min", type=int, default=33, help="first day to consider on data series"
    )
    parser.add_argument(
        "--day_max", type=int, default=58, help="last day to consider on data series"
    )

    parser.add_argument(
        "--mc_nseed",
        type=int,
        default=int(5),
        help="number of mc realizations",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="initial mc seed",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument("--save", action="store_true", help="specify for outputfile")
    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    from numpy import genfromtxt

    E_0 = args.E_0
    I_0 = args.I_0
    R_0 = args.R_0
    n_t_steps = int(1e7)  # max simulation steps
    t_total = args.day_max - args.day_min  # max simulated days
    mc_nseed = args.mc_nseed  # MC realizations
    mc_seed0 = args.mc_seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    # print(infected_time_series)
    n = args.n
    ratios = {
        "beta1": args.beta1 / n,
        "beta2": args.beta2 / n,
        "delta1": args.delta1,
        "delta2": args.delta2,
        "epsilon": args.epsilon,
    }

    return (
        E_0,
        I_0,
        R_0,
        n_t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        ratios,
    )


# -------------------------


class Compartments:
    """Compartments for SIR model"""

    def __init__(self, n_t_steps, args):
        """Initialization"""
        self.S = np.zeros(n_t_steps)
        self.E = np.zeros(n_t_steps)
        self.I = np.zeros(n_t_steps)
        self.R = np.zeros(n_t_steps)
        self.E[0] = args.E_0
        self.I[0] = args.I_0
        self.R[0] = args.R_0
        self.S[0] = args.n - args.I_0 - args.R_0 - args.E_0

    def turn_latent(self, t_step):
        """Infection s->e"""
        self.S[t_step] = self.S[t_step - 1] - 1
        self.E[t_step] = self.E[t_step - 1] + 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1]

    def turn_infectious(self, t_step):
        """Turn infectious e->i"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1] + 1
        self.R[t_step] = self.R[t_step - 1]

    def recover1(self, t_step):
        """Recovery e->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1] - 1
        self.I[t_step] = self.I[t_step - 1]
        self.R[t_step] = self.R[t_step - 1] + 1

    def recover2(self, t_step):
        """Recovery i->r"""
        self.S[t_step] = self.S[t_step - 1]
        self.E[t_step] = self.E[t_step - 1]
        self.I[t_step] = self.I[t_step - 1] - 1
        self.R[t_step] = self.R[t_step - 1] + 1


# -------------------------


def gillespie(t_total, t_step, time, comp, ratios):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (ratios["epsilon"] + ratios["delta1"]) * comp.E[t_step]
        + ratios["delta2"] * comp.I[t_step]
        + (ratios["beta1"] * comp.E[t_step] + ratios["beta2"] * comp.I[t_step])
        * comp.S[t_step]
    )

    prob_heal1 = ratios["delta1"] * comp.E[t_step] / lambda_sum
    prob_heal2 = ratios["delta2"] * comp.I[t_step] / lambda_sum
    prob_latent = ratios["epsilon"] * comp.E[t_step] / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
    # T[t_step] = time

    gillespie_step(t_step, comp, prob_heal1, prob_heal2, prob_latent)
    return t_step, time


# -------------------------


def gillespie_step(t_step, comp, prob_heal1, prob_heal2, prob_latent):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < prob_heal1:
        comp.recover1(t_step)
    elif random < (prob_heal1 + prob_heal2):
        comp.recover2(t_step)
    elif random < (prob_heal1 + prob_heal2 + prob_latent):
        comp.turn_infectious(t_step)
    else:
        comp.turn_latent(t_step)


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
