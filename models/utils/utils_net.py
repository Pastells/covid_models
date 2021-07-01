""" Common functions for network models """

import heapq
import random
import numpy as np
import networkx as nx


def _truncated_exponential_(lambd, T):
    """returns a number between 0 and T from an
    exponential distribution, conditional on the outcome being between 0 and T"""
    time = random.expovariate(lambd)
    L = int(time / T)
    return time - L * T


# -------------------------


def choose_network(n, network_type, network_param, seed=None):
    """select network type to create a graph using networkx"""
    if network_type == "er":
        G = nx.erdos_renyi_graph(n, network_param / n, seed)
    elif network_type == "ba":
        G = nx.barabasi_albert_graph(n, network_param, seed)
    else:
        raise ValueError(f"Network type '{network_type}' not avaliable")
    return G


# -------------------------


class MyQueue:
    r"""
    This class is used to store and act on a priority queue of events for
    event-driven simulations.  It is based on heapq.

    Each queue is given a tmax (default is infinity) so that any event at later
    time is ignored.

    This is a priority queue of 4-tuples of the form
        ``(time, counter, function, function_arguments)``

    The ``counter`` is present just to break ties, which generally only occur when
    multiple events are put in place for the initial condition, but could also
    occur in cases where events tend to happen at discrete times.

    note that the function is understood to have its first argument be time, and
    the tuple ``function_arguments`` does not include this first time.

    So function is called as
        ``function(time, *function_arguments)``
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
        time, counter, function, args = heapq.heappop(self._Q_)
        function(time, *args)

    def __len__(self):
        r"""this will allow us to use commands like ``while Q:``"""
        return len(self._Q_)


# -------------------------


def markovian_times(sus_neighbors, beta, delta, alpha=None):
    """Cycle through, find infection times and check it it is less than recovery time"""

    # Handle delta_a = 0 in SAIR model
    if delta != 0:
        duration = random.expovariate(delta)
    else:
        duration = 1e20

    if alpha is not None:
        duration2 = random.expovariate(alpha)
        if duration < duration2:
            pass
        else:
            duration = duration2

    trans_prob = 1 - np.exp(-beta * duration)
    number_to_infect = np.random.binomial(len(sus_neighbors), trans_prob)
    transmission_recipients = random.sample(sus_neighbors, number_to_infect)
    trans_delay = {}
    for recipient in transmission_recipients:
        trans_delay[recipient] = _truncated_exponential_(beta, duration)

    if alpha is not None:
        if duration < duration2:
            return trans_delay, duration, "recover"
        return trans_delay, duration, "infect"
    return trans_delay, duration
