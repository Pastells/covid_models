""" Common functions for network models """
import heapq
import random
import numpy as np
import networkx as nx


def _truncated_exponential_(lambd, T):
    """returns a number between 0 and T from an
    exponential distribution conditional on the outcome being between 0 and T"""
    t = random.expovariate(lambd)
    L = int(t / T)
    return t - L * T


# -------------------------


def choose_network(n, network_type, network_param, seed=None):
    """ select network type to create a graph using networkx"""
    if network_type == "er":
        G = nx.erdos_renyi_graph(n, network_param / n, seed)
    elif network_type == "ba":
        G = nx.barabasi_albert_graph(n, network_param, seed)
    else:
        raise ValueError(f"Network type '{network_type}' not avaliable")
    return G


# -------------------------


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


def Markovian_times(node, sus_neighbors, beta, delta, alpha=None):
    """Cycle through, find infection times and check it it is less than recovery time"""

    duraten = random.expovariate(delta)

    if alpha is not None:
        duraten2 = random.expovariate(alpha)
        if duraten < duraten2:
            pass
        else:
            duraten = duraten2

    trans_prob = 1 - np.exp(-beta * duraten)
    number_to_infect = np.random.binomial(len(sus_neighbors), trans_prob)
    # print(len(suscep_neighbors),number_to_infect,trans_prob, beta, duraten)
    transmission_recipients = random.sample(sus_neighbors, number_to_infect)
    trans_delay = {}
    for v in transmission_recipients:
        trans_delay[v] = _truncated_exponential_(beta, duraten)

    if alpha is not None:
        if duraten < duraten2:
            return (trans_delay, duraten, "recover")
        else:
            return (trans_delay, duraten, "infect")
    else:
        return trans_delay, duraten
