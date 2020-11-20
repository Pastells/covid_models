import random
from collections import defaultdict
import numpy as np
import utils


def _process_trans_SIR_(
    time,
    G,
    source,
    target,
    times,
    S,
    I,
    R,
    Q,
    status,
    rec_time,
    pred_inf_time,
    transmissions,
    beta,
    delta,
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
        S, I, R : lists
            lists of numbers of nodes of each status at each time
        Q : myQueue
            the queue of events
        status : dict
            dictionary giving status of each node
        rec_time : dict
            dictionary giving recovery time of each node
        pred_inf_time : dict
            dictionary giving predicted infeciton time of nodes
        beta,delta:
            rates of infection and recovery

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

    if status[target] == "S":  # nothing happens if already infected.
        status[target] = "I"
        times.append(time)
        transmissions.append((time, source, target))
        S.append(S[-1] - 1)  # one less susceptible
        I.append(I[-1] + 1)  # one more infected
        R.append(R[-1])  # no change to recovered

        suscep_neighbors = [v for v in G.neighbors(target) if status[v] == "S"]

        trans_delay, rec_delay = Markovian_times(target, suscep_neighbors, beta, delta)

        rec_time[target] = time + rec_delay
        if rec_time[target] <= Q.tmax:
            Q.add(
                rec_time[target],
                _process_rec_SIR_,
                args=(target, times, S, I, R, status),
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
                    _process_trans_SIR_,
                    args=(
                        G,
                        target,
                        v,
                        times,
                        S,
                        I,
                        R,
                        Q,
                        status,
                        rec_time,
                        pred_inf_time,
                        transmissions,
                        beta,
                        delta,
                    ),
                )
                pred_inf_time[v] = inf_time


def _process_rec_SIR_(time, node, times, S, I, R, status):
    r"""From figure A.3 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.

    :Arguments:

        event : event
            has details on node and time
        times : list
            list of times at which events have happened
        S, I, R : lists
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
    I : appends new I (decreased by 1)
    R : appends new R (increased by 1)
    """
    times.append(time)
    S.append(S[-1])  # no change to number susceptible
    I.append(I[-1] - 1)  # one less infected
    R.append(R[-1] + 1)  # one more recovered
    status[node] = "R"


def Markovian_times(node, sus_neighbors, beta, delta):
    """Cycle through, find infection times and check it it is less than recovery time"""

    duration = random.expovariate(delta)

    trans_prob = 1 - np.exp(-beta * duration)
    number_to_infect = np.random.binomial(len(sus_neighbors), trans_prob)
    # print(len(suscep_neighbors),number_to_infect,trans_prob, beta, duration)
    transmission_recipients = random.sample(sus_neighbors, number_to_infect)
    trans_delay = {}
    for v in transmission_recipients:
        trans_delay[v] = utils._truncated_exponential_(beta, duration)
    return trans_delay, duration


def fast_SIR(
    G,
    beta,
    delta,
    initial_infecteds=None,
    initial_recovereds=None,
    tmin=0,
    tmax=float("Inf"),
):
    r"""
    fast SIR simulation for exponentially distributed infection and
    recovery times

    :Arguments:

    **G** networkx Graph
        The underlying network

    **beta** number
        transmission rate per edge

    **delta** number
        recovery rate per node

    **initial_infecteds** node or iterable of nodes
        if a single node, then this node is initially infected

        if an iterable, then whole set is initially infected

        if None, then choose randomly based on rho.

        If rho is also None, a random single node is chosen.

        If both initial_infecteds and rho are assigned, then there
        is an error.

    **initial_recovereds** iterable of nodes (default None)
        this whole collection is made recovered.
        Currently there is no test for consistency with initial_infecteds.
        Understood that everyone who isn't infected or recovered initially
        is initially susceptible.

    **tmin** number (default 0)
        starting time

    **tmax** number  (default Infinity)
        maximum time after which simulation will stop.
        the default of running to infinity is okay for SIR,
        but not for SIS.

    :Returns:

    **times, S, I, R** numpy arrays

    """

    # initial setup.
    status = defaultdict(lambda: "S")  # node status defaults to 'S'
    rec_time = defaultdict(lambda: tmin - 1)  # node recovery time defaults to -1
    if initial_recovereds is not None:
        for node in initial_recovereds:
            status[node] = "R"
            rec_time[node] = (
                tmin - 1
            )  # default value ensures that the recovered nodes appear with a time
    pred_inf_time = defaultdict(lambda: float("Inf"))
    # infection time defaults to \infty  --- this could be set to tmax,
    # probably with a slight improvement to performance.

    Q = utils.myQueue(tmax)

    if initial_infecteds is None:  # create initial infecteds list if not given
        initial_number = 1
        initial_infecteds = random.sample(G.nodes(), initial_number)
    elif G.has_node(initial_infecteds):
        initial_infecteds = [initial_infecteds]
    # else it is assumed to be a list of nodes.

    times, S, I, R = ([tmin], [G.order()], [0], [0])
    # usefull for full_data case
    transmissions = []

    for u in initial_infecteds:
        pred_inf_time[u] = tmin
        Q.add(
            tmin,
            _process_trans_SIR_,
            args=(
                G,
                None,
                u,
                times,
                S,
                I,
                R,
                Q,
                status,
                rec_time,
                pred_inf_time,
                transmissions,
                beta,
                delta,
            ),
        )

    # Note that when finally infected, pred_inf_time is correct
    # and rec_time is correct.
    # So if return_full_data is true, these are correct

    while Q:  # all the work is done in this while loop.
        Q.pop_and_run()

    # the initial infections were treated as ordinary infection events at
    # time 0.
    # So each initial infection added an entry at time 0 to lists.
    # We'd like to get rid these excess events.
    times = times[len(initial_infecteds) :]
    S = S[len(initial_infecteds) :]
    I = I[len(initial_infecteds) :]
    R = R[len(initial_infecteds) :]

    return np.array(times), np.array(S), np.array(I), np.array(R)