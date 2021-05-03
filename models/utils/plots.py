""" Generate plots """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from . import config
from .utils import get_time_series

sns.set()


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


def show_save(save=None, name=None, metric="sum of squares"):
    """Add legend, save if wanted and show"""
    metric = metric.split(".")[-1]
    plt.ylabel(f"Cost ({metric})")
    plt.xlabel("Days")
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

    time_series = get_time_series(args)

    if comp is not None:
        every = 100
        plt.plot(
            comp.T[:t_step:every], comp.S[:t_step:every], label="S single realization"
        )
        # plt.plot(comp.T[:t_step:every], comp.E[:t_step:every], label="E single realization")
        # plt.plot(comp.T[:t_step:every], comp.A[:t_step:every], label="A single realization")
        plt.plot(
            comp.T[:t_step:every], comp.I[:t_step:every], label="I single realization"
        )
        plt.plot(
            comp.T[:t_step:every],
            comp.I_cum[:t_step:every],
            label="C single realization",
        )
        plt.plot(
            comp.T[:t_step:every], comp.R[:t_step:every], label="R single realization"
        )
        suma = comp.I + comp.R  # + comp.E + comp.A
        plt.plot(comp.T[:t_step:every], suma[:t_step:every], label="total")

    if R_m is not None:
        error_plot(R_m, day_max, "Recoverd cases")

    if D_m is not None:
        error_plot(D_m, day_max, "Death cases")

    if config.CUMULATIVE is True:
        i_label = "Cumulative infected cases"
    else:
        i_label = "Daily infected cases"

    error_plot(I_m, day_max, i_label)
    show_save(metric=args.metric)
    # show_save(args.save, "_trajectories")

    if config.CUMULATIVE is True:
        daily_m = np.copy(I_m)
        for day in range(len(daily_m) - 1, 0, -1):
            daily_m[day] = daily_m[day] - daily_m[day - 1]

        plt.plot(
            np.arange(day_max),
            daily_m[:day_max],
            marker="o",
            ls="",
            label="Daily infected cases",
        )
    else:
        error_plot(I_m, day_max, "Daily infected cases")

    plt.plot(time_series[:, 0], "o", label="data I")

    if R_m is not None:
        error_plot(R_m, day_max, "Recoverd cases")
        plt.plot(time_series[:, 1], "o", label="data R")

    if D_m is not None:
        error_plot(D_m, day_max, "Death cases")
        plt.plot(time_series[:, 2], "o", label="data D")

    show_save(args.save, "_daily.png", metric=args.metric)

    """
    if config.CUMULATIVE is True:
        error_plot(I_m, day_max, "Cumulative infected cases")

    elif comp is not None:
        plt.plot(
            comp.T[:t_step], comp.I_cum[:t_step], label="Cumulative infected cases"
        )

    plt.plot(time_series[:, 3], "o", label="data")

    show_save(args.save, "_cumulative.png")
    """

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
