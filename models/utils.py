"""
Common functions for all models
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------


def beta_func(beta, t):
    """
    returns beta as a function of time
    """
    # t_conf = 20 # day of confinement
    # alpha = 0.5
    # delta_t = 5
    # if t<t_conf:
    return beta
    # else:
    # return beta*alpha + beta*(1-alpha)*np.exp(-(t-t_conf)/delta_t)


# -------------------------


def time_dist(lambd):
    """
    Time intervals of a Poisson process follow an exponential distribution
    """
    return -np.log(1 - np.random.random()) / lambd


# -------------------------


def mean_alive(i_day, t_total, day_max, nseed):
    """
    Given that we already have a pandemic to study,
    we average only the alive realizations after an
    (arbitrary) time equal to half the time of the longest
    realization
    The running variance is computed according to:
        https://www.johndcook.com/blog/standard_deviation/
    """
    check_realization_alive = day_max // 2

    for i in range(nseed):
        if i_day[i, check_realization_alive] != 0:
            x_var = i_day[i]
            alive_realizations = 1
            s_var = np.zeros(t_total)
            i_m = x_var
            break

    for j in range(i + 1, nseed):
        if i_day[j, check_realization_alive] != 0:
            alive_realizations += 1
            x_var = i_day[j]
            i_m_1 = i_m
            i_m = i_m_1 + (x_var - i_m_1) / alive_realizations
            s_var = s_var + (x_var - i_m_1) * (x_var - i_m)

    i_std = np.sqrt(s_var / (alive_realizations - 1))

    if nseed - alive_realizations > nseed * 0.1:
        print("The initial number of infected may be too low")
        print(
            f"Alive realizations after {check_realization_alive} days = {alive_realizations},\
              out of {nseed}"
        )
    return i_m, i_std


# -------------------------


def plotting(infected_time_series, i_day, day_max, i_m, i_std):
    """
    If --plot is added makes some plots
    """

    # s_m = s_day.mean(0)
    # i_m = i_day.mean(0)
    # i_std = i_day.std(0)
    # r_m = r_day.mean(0)
    # s_std = s_day.std(0)
    # r_std = r_day.std(0)
    # print(r_m[day_max],"recovered individuals")

    plt.errorbar(
        np.arange(day_max),
        i_m[:day_max],
        yerr=i_std[:day_max],
        marker="o",
        ls="",
        label="i mean",
    )
    plt.show()

    # i_m = np.median(i_day,0)

    # alpha = 0.70
    # p_l = ((1.0-alpha)/2.0) * 100
    # p_u = (alpha+((1.0-alpha)/2.0)) * 100
    # i_95[:,0] = np.percentile(i_day, p_l,0)
    # i_95[:,1] = np.percentile(i_day, p_u,0)

    # plt.plot(i_m,'o',c='orange',label='i median')
    # plt.plot(i_95[:,0],c='orange')
    # plt.plot(i_95[:,1],c='orange')

    plt.errorbar(
        np.arange(day_max),
        i_m[:day_max],
        yerr=i_std[:day_max],
        marker="o",
        ls="",
        label="i mean",
    )
    plt.plot(infected_time_series, "o", label="data")
    plt.legend()
    plt.show()


def saving(args, i_m, i_std, day_max):
    """
    If --save is added creates an output file wicreates an output file with date and time
    """
    import time

    filename = "results/sir" + time.strftime("%d%m_%H%M%S") + ".dat"
    with open(filename, "w") as out_file:
        out_file.write(f"#{args}\n")
        for day in range(day_max):
            out_file.write(f"{i_m[day]}, {i_std[day]}\n")


# ~~~~~~~~~~~~~~~~~~~
# Output
# ~~~~~~~~~~~~~~~~~~~
def cost_func(infected_time_series, i_m, i_std):
    """
    compute cost function with a weighted mean squared error
    """
    import sys

    pad = len(infected_time_series) - len(i_m)

    if pad > 0:
        i_m = np.pad(i_m, (0, pad), "constant")
        i_std = np.pad(i_std, (0, pad), "constant")

    cost = 0
    for i, _ in enumerate(infected_time_series):
        cost += (i_m[i] - infected_time_series[i]) ** 2 / (1 + i_std[i])
    cost = np.sqrt(cost)
    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# -------------------------

"""
Functions with two versions
"""


def day_data(mc_step, t, time, t_total, day, day_max, i, i_day, last_event=False):
    """
    Write number of infected per day instead of event
    Also tracks day_max
    """
    if last_event:
        # final value for rest of time, otherwise contributes with zero when averaged
        days_jumped = int(time - day)
        max_days = min(day + days_jumped + 1, t_total)
        i_day[mc_step, day:max_days] = i[t]
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    if time // day == 1:
        days_jumped = int(time - day)
        max_days = min(day + days_jumped + 1, t_total)
        i_day[mc_step, day:max_days] = i[t]
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    return day, day_max


# -------------------------


def day_data_k(mc_step, t, time, t_total, day, day_max, i, i_day, last_event=False):
    """
    Write number of infected per day instead of event
    Also tracks day_max
    """
    if last_event:
        days_jumped = int(time - day)
        max_days = min(day + days_jumped + 1, t_total)
        i_day[mc_step, day:max_days] = i[t, :-1].sum()
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    if time // day == 1:
        days_jumped = int(time - day)
        max_days = min(day + days_jumped + 1, t_total)
        i_day[mc_step, day:max_days] = i[t, :-1].sum()
        # s_day[mc_step,day:day+days_jumped+1]=s[t:-1].sum()
        # r_day[mc_step,day:day+days_jumped+1]=r[t]
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    return day, day_max
