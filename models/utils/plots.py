""" Generate plots """

import numpy as np
import matplotlib.pyplot as plt
from . import config


def plotting(args, I_day, day_max, I_m, I_std):
    """ If --plot is added makes some plots"""
    from numpy import genfromtxt

    infected_cumulative = (
        genfromtxt(args.data, delimiter=",")[args.day_min : args.day_max]
        * 100
        / (100 - args.undiagnosed)
    )

    infected_daily = np.copy(infected_cumulative)

    for day in range(len(infected_daily) - 1, 0, -1):
        infected_daily[day] = infected_daily[day] - infected_daily[day - 1]

    # S_m = S_day.mean(0)
    # I_m = I_day.mean(0)
    # I_std = I_day.std(0)
    # R_m = R_day.mean(0)
    # S_std = S_day.std(0)
    # R_std = R_day.std(0)
    # print(r_m[day_max],"recovered individuals")

    plt.errorbar(
        np.arange(day_max),
        I_m[:day_max],
        yerr=I_std[:day_max],
        marker="o",
        ls="",
        label="Daily infected cases",
    )
    plt.plot(infected_daily, "o", label="data")
    plt.legend()
    if args.save is not None:
        plt.savefig(args.save + "_daily.png")
    plt.show()

    # I_m = np.median(i_day,0)

    # alpha = 0.70
    # p_l = ((1.0-alpha)/2.0) * 100
    # p_u = (alpha+((1.0-alpha)/2.0)) * 100
    # I_95[:,0] = np.percentile(i_day, p_l,0)
    # I_95[:,1] = np.percentile(i_day, p_u,0)

    # plt.plot(i_m,'o',c='orange',label='i median')
    # plt.plot(i_95[:,0],c='orange')
    # plt.plot(i_95[:,1],c='orange')

    I_cum = np.zeros(day_max)
    I_cum_std = np.zeros(day_max)
    I_cum[0] = I_m[0]
    I_cum_std[0] = I_std[0]
    for u in range(1, day_max):
        I_cum[u] = I_cum[u - 1] + I_m[u]
        I_cum_std[u] = I_cum_std[u - 1] + I_std[u]

    plt.errorbar(
        np.arange(day_max),
        I_cum,
        yerr=I_cum_std,
        marker="o",
        ls="",
        label="Daily infected cases (cumulative)",
    )

    plt.plot(infected_cumulative, "o", label="data")
    plt.legend()
    if args.save is not None:
        plt.savefig(args.save + "_cumulative.png")
    plt.show()
