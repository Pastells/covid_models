import networkx as nx
import fast_sir
import EoN
import matplotlib.pyplot as plt

size = 3000
beta = 0.5
delta = 0.2


def fast_sim(k):
    G = nx.erdos_renyi_graph(size, k / size)
    t, S, I, R = fast_sir.fast_SIR(G, beta, delta, initial_infecteds=50)
    plt.plot(t, I, label=f"<k> = {k}")


for k in range(1, 8):
    fast_sim(k)
plt.legend()
plt.show()
