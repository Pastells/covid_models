""" Generate plots """

import numpy as np
import matplotlib.pyplot as plt
from . import config, utils


def error_plot(var_m, day_max, label):
    """Generate errorbar plot for given variable"""
    plt.errorbar(
        np.arange(day_max),
        var_m[:day_max, 0],
        yerr=var_m[:day_max, 1],
        marker="o",
        ls="",
        label=label,
    )


def show_save(save=None, name=None):
    """Add legend, save if wanted and show"""
    plt.legend()
    if save is not None:
        plt.savefig(save + name)
    plt.show()


def plotting(
    args,
    day_max,
    I_m,
    R_m=None,
    D_m=None,
    comp=None,
    t_step=None,
):
    """ If --plot is added makes some plots"""

    time_series = utils.get_time_series(args)

    if comp is not None:
        plt.plot(comp.T[:t_step], comp.S[:t_step], label="S single realization")
        # plt.plot(comp.T[:t_step], comp.E[:t_step], label="E single realization")
        # plt.plot(comp.T[:t_step], comp.A[:t_step], label="A single realization")
        plt.plot(comp.T[:t_step], comp.I[:t_step], label="I single realization")
        plt.plot(comp.T[:t_step], comp.I_cum[:t_step], label="C single realization")
        plt.plot(comp.T[:t_step], comp.R[:t_step], label="R single realization")
        suma = comp.I + comp.R  # + comp.E + comp.A
        plt.plot(comp.T[:t_step], suma[:t_step], label="total")

    if R_m is not None:
        error_plot(R_m, day_max, "Recoverd cases")

    if D_m is not None:
        error_plot(D_m, day_max, "Death cases")

    if config.CUMULATIVE is True:
        i_label = "Cumulative infected cases"
    else:
        i_label = "Daily infected cases"

    error_plot(I_m, day_max, i_label)
    show_save()

    if config.CUMULATIVE is True:
        I_daily_m = np.copy(I_m)
        for day in range(len(I_daily_m) - 1, 0, -1):
            I_daily_m[day] = I_daily_m[day] - I_daily_m[day - 1]

        plt.plot(
            np.arange(day_max),
            I_daily_m[:day_max],
            marker="o",
            ls="",
            label="Daily infected cases",
        )
    else:
        error_plot(I_m, day_max, "Daily infected cases")

    plt.plot(time_series[:, 0], "o", label="data")

    show_save(args.save, "_daily.png")

    if config.CUMULATIVE is True:
        error_plot(I_m, day_max, "Cumulative infected cases")

    elif comp is not None:
        plt.plot(
            comp.T[:t_step], comp.I_cum[:t_step], label="Cumulative infected cases"
        )

    plt.plot(time_series[:, 3], "o", label="data")

    show_save(args.save, "_cumulative.png")

    # S_m = S_day.mean(0)
    # I_m = I_day.mean(0)
    # I_std = I_day.std(0)
    # R_m = R_day.mean(0)
    # S_std = S_day.std(0)
    # R_std = R_day.std(0)
    # print(r_m[day_max],"recovered individuals")

    # I_m = np.median(i_day,0)

    # alpha = 0.70
    # p_l = ((1.0-alpha)/2.0) * 100
    # p_u = (alpha+((1.0-alpha)/2.0)) * 100
    # I_95[:,0] = np.percentile(i_day, p_l,0)
    # I_95[:,1] = np.percentile(i_day, p_u,0)

    # plt.plot(i_m,'o',c='orange',label='i median')
    # plt.plot(i_95[:,0],c='orange')
    # plt.plot(i_95[:,1],c='orange')
