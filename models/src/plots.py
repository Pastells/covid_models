"""
Generate plots
"""

import numpy as np
import matplotlib.pyplot as plt


def plotting(infected_time_series, I_day, day_max, I_m, I_std):
    """ If --plot is added makes some plots"""

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
        label="i mean",
    )
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

    plt.errorbar(
        np.arange(day_max),
        I_m[:day_max],
        yerr=I_std[:day_max],
        marker="o",
        ls="",
        label="i mean",
    )
    plt.plot(infected_time_series, "o", label="data")
    plt.legend()
    plt.show()
