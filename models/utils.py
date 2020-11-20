"""
Common functions for all models
"""

import argparse
import random
import heapq
import numpy as np
import matplotlib.pyplot as plt

# -------------------------


def beta_func(beta, t):
    """ returns beta as a function of time"""
    # t_conf = 20 # day of confinement
    # alpha = 0.5
    # delta_t = 5
    # if t<t_conf:
    return beta
    # else:
    # return beta*alpha + beta*(1-alpha)*np.exp(-(t-t_conf)/delta_t)


# -------------------------


def time_dist(lambd):
    """ Time intervals of a Poisson process follow an exponential distribution"""
    return random.expovariate(lambd)


# -------------------------


def _truncated_exponential_(lambd, T):
    """returns a number between 0 and T from an
    exponential distribution conditional on the outcome being between 0 and T"""
    t = random.expovariate(lambd)
    L = int(t / T)
    return t - L * T


# -------------------------


def day_data(time, t_total, day, day_max, I, I_day, last_event=False):
    """
    Write number of infected per day instead of event
    Also tracks day_max

    I :  is the the number of infected for the last event, i.e. I[t]
    I_day : is the number of infected when time is a day multiple for the current seed,
            i.e. I_day[mc_seed]

    :Returns:

    day : adds one, or one + days_jumped if an event was slow
    day_max : updates if day is bigger than the resulting from previous seeds

    :Modifies:

    I_day : adds the infected number for day (before updating), if several days have
            elapsed they all get the same value as the previous

    """
    if last_event:
        # final value for rest of time, otherwise contributes with zero when averaged
        _days_jumped = int(time - day)
        max_days = min(day + _days_jumped + 1, t_total)
        I_day[day : max_days - 1] = I_day[day - 1]
        I_day[max_days - 1] = I
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    if time // day == 1:
        _days_jumped = int(time - day)
        max_days = min(day + _days_jumped + 1, t_total)
        I_day[day : max_days - 1] = I_day[day - 1]
        I_day[max_days - 1] = I
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    return day, day_max


# -------------------------


def mean_alive(I_day, t_total, day_max, nseed):
    """
    Given that we already have a pandemic to study,
    we average only the alive realizations after an
    (arbitrary) time equal to half the time of the longest
    realization
    The running variance is computed according to:
        https://www.johndcook.com/blog/standard_deviation/

    :Returns:

    I_m :mean
    I_std : standard deviation

    :Modifies:

    I_day : adds the infected number for day (before updating), if several days have
            elapsed they all get the same value as the previous
    """
    check_realization_alive = day_max // 2

    # first seed alive
    for u in range(nseed):
        if I_day[u, check_realization_alive] != 0:
            _x_var = I_day[u]
            alive_realizations = 1
            _s_var = np.zeros(t_total)
            I_m = _x_var
            break

    for j in range(u + 1, nseed):
        if I_day[j, check_realization_alive] != 0:
            alive_realizations += 1
            _x_var = I_day[j]
            _I_m_1 = I_m
            I_m = _I_m_1 + (_x_var - _I_m_1) / alive_realizations
            _s_var = _s_var + (_x_var - _I_m_1) * (_x_var - I_m)

    I_std = np.sqrt(_s_var / (alive_realizations - 1))

    if nseed - alive_realizations > nseed * 0.1:
        print("The initial number of infected may be too low")
        print(
            f"Alive realizations after {check_realization_alive} days = {alive_realizations},\
              out of {nseed}"
        )
    return I_m, I_std


# -------------------------


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


# -------------------------


def saving(args, I_m, I_std, day_max, program_name):
    """ If --save is added creates an output file with date and time"""
    import time

    filename = f"results/{program_name}" + time.strftime("%d%m_%H%M%S") + ".dat"
    with open(filename, "w") as out_file:
        out_file.write(f"#{args}\n")
        for day in range(day_max):
            out_file.write(f"{I_m[day]}, {I_std[day]}\n")


# ~~~~~~~~~~~~~~~~~~~
# Output
# ~~~~~~~~~~~~~~~~~~~
def cost_func(infected_time_series, I_m, I_std):
    """ compute cost function with a weighted mean squared error"""
    import sys

    pad = len(infected_time_series) - len(I_m)

    if pad > 0:
        I_m = np.pad(I_m, (0, pad), "constant")
        I_std = np.pad(I_std, (0, pad), "constant")

    cost = 0
    for u, _ in enumerate(infected_time_series):
        cost += (I_m[u] - infected_time_series[u]) ** 2 / (1 + I_std[u])
    cost = np.sqrt(cost)
    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# -------------------------


class ArgumentParser(argparse.ArgumentParser):
    def add_argument(self, *args, help=None, default=None, **kwargs):
        if help is not None:
            kwargs["help"] = help
        if default is not None and args[0] != "-h":
            kwargs["default"] = default
            if help is not None:
                kwargs["help"] += f" [{default}]"
        super().add_argument(*args, **kwargs)


# -------------------------------------
# Used in models with a social network:
# -------------------------------------


class myQueue(object):
    r"""
    This class is used to store and act on a priority queue of events for
    event-driven simulations.  It is based on heapq.

    Each queue is given a tmax (default is infinity) so that any event at later
    time is ignored.

    This is a priority queue of 4-tuples of the form
        ``(t, counter, function, function_arguments)``

    The ``counter`` is present just to break ties, which generally only occur when
    multiple events are put in place for the initial condition, but could also
    occur in cases where events tend to happen at discrete times.

    note that the function is understood to have its first argument be t, and
    the tuple ``function_arguments`` does not include this first t.

    So function is called as
        ``function(t, *function_arguments)``
    """

    def __init__(self, tmax=float("Inf")):
        self._Q_ = []
        self.tmax = tmax
        self.counter = 0  # tie-breaker for putting things in priority queue

    def add(self, time, function, args=()):
        r"""time is the time of the event.  args are the arguments of the
        function not including the first argument which must be time"""
        if time < self.tmax:
            heapq.heappush(self._Q_, (time, self.counter, function, args))
            self.counter += 1

    def pop_and_run(self):
        r"""Pops the next event off the queue and performs the function"""
        t, counter, function, args = heapq.heappop(self._Q_)
        function(t, *args)

    def __len__(self):
        r"""this will allow us to use commands like ``while Q:`` """
        return len(self._Q_)


# -------------------------


def _get_rate_functions_(
    G, beta, delta, transmission_weight=None, recovery_weight=None
):
    r"""
    Arguments :
        G : networkx Graph
            the graph disease spreads on

        beta : number
            disease parameter giving edge transmission rate (subject to edge scaling)

        delta : number (default None)
            disease parameter giving typical recovery rate,

        transmission_weight : string (default None)
            The attribute name under which transmission rates are saved.
            `G.adj[u][v][transmission_weight]` scales up or down the recovery rate.
            (note this is G.edge[u][v][..] in networkx 1.x and
            G.edges[u,v][..] in networkx 2.x.
            The backwards compatible version is G.adj[u][v]
            https://networkx.github.io/documentation/stable/release/migration_guide_from_1.x_to_2.0.html)

        recovery_weight : string       (default None)
            a label for a weight given to the nodes to scale their
            recovery rates
                `delta_i = G.node[i][recovery_weight]*delta`
    Returns :
        : trans_rate_fxn, rec_rate_fxn
            Two functions such that
            - `trans_rate_fxn(u,v)` is the transmission rate from u to v and
            - `rec_rate_fxn(u)` is the recovery rate of u."""
    if transmission_weight is None:
        trans_rate_fxn = lambda x, y: beta
    else:
        try:
            trans_rate_fxn = lambda x, y: beta * G.adj[x][y][transmission_weight]
        except AttributeError:  # apparently you have networkx v1.x not v2.x
            trans_rate_fxn = lambda x, y: beta * G.edge[x][y][transmission_weight]

    if recovery_weight is None:
        rec_rate_fxn = lambda x: delta
    else:
        rec_rate_fxn = lambda x: delta * G.nodes[x][recovery_weight]

    return trans_rate_fxn, rec_rate_fxn
