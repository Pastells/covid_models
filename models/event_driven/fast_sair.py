import random
from collections import defaultdict
import numpy as np
from utils import utils_net


def _process_trans_SAIR_(
    time,
    G,
    target,
    times,
    S,
    A,
    I,
    R,
    Q,
    status,
    rec_time,
    pred_inf_time,
    rates,
    e_or_i,
):
    r"""
        From figure A.4 of Kiss, Miller, & Simon.  Please cite the book if
        using this algorithm.

        :Arguments:

        time : number
            time of transmission
    **G**  networkx Graph
        node : node
            node receiving transmission.
        times : list
            list of times at which events have happened
        S, A, I, R : lists
            lists of numbers of nodes of each status at each time
        Q : MyQueue
            the queue of events
        status : dict
            dictionary giving status of each node
        rec_time : dict
            dictionary giving recovery time of each node
        pred_inf_time : dict
            dictionary giving predicted infeciton time of nodes
        rates (beta_a/2,delta_a/2, alpha):
            rates of infection, recovery and latency

        :Returns:

        Nothing

        :MODIFIES:

        status : updates status of newly infected node
        rec_time : adds recovery time for node
        times : appends time of event
        S : appends new S (reduced by 1 from last)
        I : appends new I (increased by 1)
        R : appends new R (same as last)
        Q : adds recovery and transmission events for newly infected node.
        pred_inf_time : updated for nodes that will receive transmission

    """

    if (e_or_i == "A" and status[target] == "S") or (
        e_or_i == "I" and status[target] == "A"
    ):  # nothing happens if already infected.

        times.append(time)
        R.append(R[-1])

        suscep_neighbors = [v for v in G.neighbors(target) if status[v] == "S"]

        if e_or_i == "A":
            status[target] = "A"
            S.append(S[-1] - 1)
            A.append(A[-1] + 1)
            I.append(I[-1])
            trans_delay, rec_delay, recover_or_infect = utils_net.markovian_times(
                suscep_neighbors,
                rates["beta_a"],
                rates["delta_a"],
                rates["alpha"],
            )
        else:
            status[target] = "I"
            S.append(S[-1])
            A.append(A[-1] - 1)
            I.append(I[-1] + 1)
            trans_delay, rec_delay = utils_net.markovian_times(
                suscep_neighbors,
                rates["beta"],
                rates["delta"],
            )
            recover_or_infect = "recover"

        rec_time[target] = time + rec_delay
        if rec_time[target] <= Q.tmax:
            if recover_or_infect == "recover":
                Q.add(
                    rec_time[target],
                    _process_rec_SAIR_,
                    args=(target, times, S, A, I, R, status, e_or_i),
                )
            else:
                Q.add(
                    rec_time[target],
                    _process_trans_SAIR_,
                    args=(
                        G,
                        target,
                        times,
                        S,
                        A,
                        I,
                        R,
                        Q,
                        status,
                        rec_time,
                        pred_inf_time,
                        rates,
                        "I",
                    ),
                )

        for v in trans_delay:
            inf_time = time + trans_delay[v]
            if (
                inf_time <= rec_time[target]
                and inf_time < pred_inf_time[v]
                and inf_time <= Q.tmax
            ):
                Q.add(
                    inf_time,
                    _process_trans_SAIR_,
                    args=(
                        G,
                        v,
                        times,
                        S,
                        A,
                        I,
                        R,
                        Q,
                        status,
                        rec_time,
                        pred_inf_time,
                        rates,
                        "A",
                    ),
                )
                pred_inf_time[v] = inf_time


# -------------------------


def _process_rec_SAIR_(time, node, times, S, A, I, R, status, e_or_i):
    r"""

    :Arguments:

        event : event
            has details on node and time
        times : list
            list of times at which events have happened
        S, A, I, R : lists
            lists of numbers of nodes of each status at each time
        status : dict
            dictionary giving status of each node


    :Returns:

    Nothing

    MODIFIES
    ----------
    status : updates status of newly recovered node
    times : appends time of event
    S : appends new S (same as last)
    A : appends new A (same as last)
    I : appends new I (decreased by 1)
    R : appends new R (increased by 1)
    """

    times.append(time)
    S.append(S[-1])
    R.append(R[-1] + 1)

    if e_or_i == "A":
        A.append(A[-1] - 1)
        I.append(I[-1])
    else:
        A.append(A[-1])
        I.append(I[-1] - 1)

    status[node] = "R"


# -------------------------


def _process_inf_SAIR_(time, node, times, S, A, I, R, status):
    r"""

    :Arguments:

        event : event
            has details on node and time
        times : list
            list of times at which events have happened
        S, A, I, R : lists
            lists of numbers of nodes of each status at each time
        status : dict
            dictionary giving status of each node


    :Returns:

    Nothing

    MODIFIES
    ----------
    status : updates status of newly recovered node
    times : appends time of event
    S : appends new S (same as last)
    A : appends new A (same as last)
    I : appends new I (decreased by 1)
    R : appends new R (increased by 1)
    """
    times.append(time)
    S.append(S[-1])
    A.append(A[-1] - 1)
    I.append(I[-1] + 1)
    R.append(R[-1])
    status[node] = "I"


# -------------------------


def fast_SAIR(
    G,
    rates,
    E_0=0,
    I_0=0,
    R_0=0,
    tmin=0,
    tmax=float("Inf"),
):
    r"""
    fast SAIR simulation for exponentially distributed infection and
    recovery times

    :Arguments:

    **G** networkx Graph
        The underlying network

    **beta** number
        transmission rate per edge

    **delta** number
        recovery rate per node

    **E_0** number
        initially exposed nodes (NOT IMPLEMENTED)

    **I_0** number
        initially infected nodes

    **R_0** number
        initially recovered nodes

    **tmin** number (default 0)
        starting time

    **tmax** number  (default Infinity)
        maximum time after which simulation will stop.
        the default of running to infinity is okay for SAIR,
        but not for SIS.


    :Returns:

    **times, S, A, I, R** numpy arrays

    """

    # initial setup.
    status = defaultdict(lambda: "S")  # node status defaults to 'S'
    rec_time = defaultdict(lambda: tmin - 1)  # node recovery time defaults to -1

    # simply remove initially recovered nodes
    if R_0 != 0:
        R_0_nodes = random.sample(G.nodes(), R_0)
        G.remove_nodes_from(R_0_nodes)

    """
    if R_0 is not None:
        for node in R_0:
            status[node] = "R"
            rec_time[node] = (
                tmin - 1
            )  # default value ensures that the recovered nodes appear with a time
    """
    pred_inf_time = defaultdict(lambda: float("Inf"))
    # infection time defaults to \infty  --- this could be set to tmax,
    # probably with a slight improvement to performance.

    Q = utils_net.MyQueue(tmax)

    """
    if I_0 is None:  # create initial infecteds list if not given
        initial_number = 1
        I_0 = random.sample(G.nodes(), initial_number)
    elif G.has_node(I_0):
        I_0 = [I_0]
    # else it is assumed to be a list of nodes.
    """
    # Just one sample, so there's no possible overlap
    I_0 = random.sample(G.nodes(), I_0 + E_0)

    times, S, A, I, R = (
        [tmin],
        [G.order() - len(I_0[E_0:])],
        [len(I_0[E_0:])],
        [0],
        [0],
    )

    for u in I_0[:E_0]:
        status[u] = "S"
        pred_inf_time[u] = tmin
        Q.add(
            tmin,
            _process_trans_SAIR_,
            args=(
                G,
                u,
                times,
                S,
                A,
                I,
                R,
                Q,
                status,
                rec_time,
                pred_inf_time,
                rates,
                "A",
            ),
        )
    for u in I_0[E_0:]:
        status[u] = "A"
        pred_inf_time[u] = tmin
        Q.add(
            tmin,
            _process_trans_SAIR_,
            args=(
                G,
                u,
                times,
                S,
                A,
                I,
                R,
                Q,
                status,
                rec_time,
                pred_inf_time,
                rates,
                "I",
            ),
        )

    while Q:  # all the work is done in this while loop.
        Q.pop_and_run()

    # the initial infections were treated as ordinary infection events at
    # time 0.
    # So each initial infection added an entry at time 0 to lists.
    # We'd like to get rid these excess events.
    times = times[len(I_0) :]
    S = S[len(I_0) :]
    A = A[len(I_0) :]
    I = I[len(I_0) :]
    R = R[len(I_0) :]

    return np.array(times), np.array(S), np.array(A), np.array(I), np.array(R) + R_0
