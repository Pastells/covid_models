"""
Stochastic mean-field SEIR model
using the Gillespie algorithm and Erlang distribution transition times
Pol Pastells,  october 2020

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
        shapes,
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
    for mc_seed in range(mc_seed0, mc_seed0 + mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)

        # -------------------------
        # initialization
        comp = Compartments(n_t_steps, shapes, args)

        I_day[mc_step, 0] = I_0
        # T = np.zeros(n_t_steps)
        # T[0]=0
        t_step, time, day = 0, 0, 1

        # Time loop
        while comp.I[t_step, :-1].sum() > 0 and day < t_total:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, comp.I[t_step, :-1].sum(), I_day[mc_step]
            )
            t_step, time = gillespie(
                t_total,
                t_step,
                time,
                comp,
                ratios,
                shapes,
            )
            if time is True:
                break

        # -------------------------

        """
        if plot:
            plt.plot(T[:t_step], S[:t_step, :-1].sum(1), c='r')
            plt.plot(T[:t_step], E[:t_step, :-1, 0].sum(1), c='g')
            plt.plot(T[:t_step], E[:t_step, :-1, 1].sum(1), c='b')
            plt.plot(T[:t_step], I[:t_step, :-1].sum(1), c='c')
            plt.plot(T[:t_step], R[:t_step], c='m')
        """

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save is not None:
        utils.saving(args, I_m, I_std, day_max, "net_sir", save)

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# -------------------------
def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-field SEIR model using the Gillespie algorithm and Erlang \
            distribution transition times.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_init = parser.add_argument_group("initial conditions")
    parser_params = parser.add_argument_group("parameters")

    parser_init.add_argument(
        "--E_0", type=int, default=0, help="initial number of latent individuals"
    )
    parser_init.add_argument(
        "--I_0", type=int, default=20, help="initial number of infected individuals"
    )
    parser_init.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals"
    )

    parser_params.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="fixed number of (effecitve) people [1000,1000000]",
    )
    parser_params.add_argument(
        "--delta1",
        type=float,
        default=0.01,
        help="ratio of recovery from latent fase (e->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta2",
        type=float,
        default=0.2,
        help="ratio of recovery from infected fase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--k_rec",
        type=int,
        default=1,
        help="k for the recovery time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--beta1",
        type=float,
        default=0.01,
        help="ratio of infection due to latent [0.05,1]",
    )
    parser_params.add_argument(
        "--beta2",
        type=float,
        default=0.5,
        help="ratio of infection due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--k_inf",
        type=int,
        default=1,
        help="k for the infection time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--epsilon",
        type=float,
        default=1,
        help="ratio of latency (e->i) [0.05,2]",
    )
    parser_params.add_argument(
        "--k_lat",
        type=int,
        default=1,
        help="k for the latent time erlang distribution [1,5]",
    )

    utils.parser_common(parser)
    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters
def parameters_init(args):
    """Initial parameters from argparse"""
    from numpy import genfromtxt

    E_0 = args.E_0
    I_0 = args.I_0
    R_0 = args.R_0
    n_t_steps = args.n_t_steps  # max simulation steps
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
    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec, "k_lat": args.k_lat}
    ratios = {
        "beta1": args.beta1 / n * args.k_inf,
        "beta2": args.beta2 / n * args.k_inf,
        "delta1": args.delta1 * args.k_rec,
        "delta2": args.delta2 * args.k_rec,
        "epsilon": args.epsilon * args.k_lat,
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
        shapes,
    )


# -------------------------


class Compartments:
    """Compartments for the SEIR Erlang model"""

    def __init__(self, n_t_steps, shapes, args):
        """Initialization"""
        self.S = np.zeros([n_t_steps, shapes["k_inf"] + 1])
        self.E = np.zeros([n_t_steps, shapes["k_lat"] + 1, 2])
        self.I = np.zeros([n_t_steps, shapes["k_rec"] + 1])
        self.R = np.zeros(n_t_steps)

        # Used for both seir_erlang and seir_erlang sections, where args.n is a vector
        try:
            self.S[0, :-1] = (args.n - args.I_0 - args.R_0) / shapes["k_inf"]
        except TypeError:
            self.S[0, :-1] = (args.n[0] - args.I_0 - args.R_0) / shapes["k_inf"]

        self.S[0, -1] = self.E[0, :-1] = args.E_0 / shapes["k_lat"]
        self.E[0, -1] = self.I[0, :-1] = args.I_0 / shapes["k_rec"]
        self.I[0, -1] = self.R[0] = args.R_0

    def latent_adv_s(self, t_step, k):
        """Turn latent or advance in S
        S(k)-> S(k+1)/E(0)"""
        self.E[t_step, :-1] = self.E[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.R[t_step] = self.R[t_step - 1]
        self.S[t_step, k] = -1
        self.S[t_step, k + 1] = 1
        self.E[t_step, 0] += self.S[t_step, -1]
        self.S[t_step] += self.S[t_step - 1]

    def infect_adv_e(self, t_step, k):
        """Turn infectious or advance in E
        E(k)-> E(k+1)/I(0)"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.R[t_step] = self.R[t_step - 1]
        self.E[t_step, k, 1] = -1
        self.E[t_step, k + 1, 1] = 1
        self.I[t_step, 0] += self.E[t_step, -1, 1]
        self.E[t_step, 0, 0] = self.E[t_step, 0, 1]
        self.E[t_step] += self.E[t_step - 1]

    def recover_adv_e(self, t_step, k):
        """Recover or advance in E
        E(k)-> E(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.E[t_step, k, 0] = -1
        self.E[t_step, k + 1, 0] = 1
        self.R[t_step] = self.R[t_step - 1] + self.E[t_step, -1, 0]
        self.E[t_step, 0, 1] = self.E[t_step, 0, 0]
        self.E[t_step] += self.E[t_step - 1]

    def recover_adv_i(self, t_step, k):
        """Recover or advance in I
        I(k)-> I(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.E[t_step, :-1] = self.E[t_step - 1, :-1]
        self.I[t_step, k] = -1
        self.I[t_step, k + 1] = 1
        self.R[t_step] = self.R[t_step - 1] + self.I[t_step, -1]
        self.I[t_step] += self.I[t_step - 1]


# -------------------------


def gillespie(
    t_total,
    t_step,
    time,
    comp,
    ratios,
    shapes,
):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()
    etot_rec = comp.E[t_step, :-1, 0].sum()
    etot_inf = comp.E[t_step, :-1, 1].sum()
    etot = etot_inf + etot_rec - comp.E[t_step, 0, 0]

    lambda_sum = (
        ratios["epsilon"] * etot_inf
        + ratios["delta1"] * etot_rec
        + ratios["delta2"] * itot
        + (ratios["beta1"] * etot + ratios["beta2"] * itot) * stot
    )

    prob_heal1 = ratios["delta1"] * comp.E[t_step, :-1, 0] / lambda_sum
    prob_heal2 = ratios["delta2"] * comp.I[t_step, :-1] / lambda_sum
    prob_latent = ratios["epsilon"] * comp.E[t_step, :-1, 1] / lambda_sum
    prob_infect = (
        (ratios["beta1"] * etot + ratios["beta2"] * itot)
        * comp.S[t_step, :-1]
        / lambda_sum
    )

    t_step += 1
    time += utils.time_dist(lambda_sum)
    if time > t_total:
        return t_step, True  # rare,  but sometimes long times may appear
    # T[t_step] = time

    gillespie_step(
        t_step,
        comp,
        prob_heal1,
        prob_heal2,
        prob_latent,
        prob_infect,
        shapes,
    )
    return t_step, time


# -------------------------


def gillespie_step(
    t_step, comp, prob_heal1, prob_heal2, prob_latent, prob_infect, shapes
):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    S and I have one extra dimension to temporally store the infected and recovered
    after k stages, due to the Erlang distribution
    """
    random = np.random.random()
    prob_heal1_tot = prob_heal1.sum()
    prob_heal2_tot = prob_heal2.sum()
    prob_latent_tot = prob_latent.sum()

    # E(k)-> E(k+1)/R
    if random < prob_heal1_tot:
        for k in range(shapes["k_lat"]):
            if random < prob_heal1[: k + 1].sum():
                comp.recover_adv_e(t_step, k)
                break

    # I(k)-> I(k+1)/R
    elif random < (prob_heal1_tot + prob_heal2_tot):
        random -= prob_heal1_tot
        for k in range(shapes["k_rec"]):
            if random < prob_heal2[: k + 1].sum():
                comp.recover_adv_i(t_step, k)
                break

    # E(k)-> E(k+1)/I(0)
    elif random < (prob_heal1_tot + prob_heal2_tot + prob_latent_tot):
        random -= prob_heal1_tot + prob_heal2_tot
        for k in range(shapes["k_lat"]):
            if random < prob_latent[: k + 1].sum():
                comp.infect_adv_e(t_step, k)
                break

    # S(k)-> S(k+1)/E(0)
    else:
        random -= prob_heal1_tot + prob_heal2_tot + prob_latent_tot
        for k in range(shapes["k_inf"]):
            if random < prob_infect[: k + 1].sum():
                comp.latent_adv_s(t_step, k)
                break


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
