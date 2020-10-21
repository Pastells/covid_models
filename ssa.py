# Salp Swarm Algorithm

import numpy as np
import matplotlib.pyplot as plt

D = 2 # dimensions
F = np.array([0,0]) # target. unknown --> fitness
ub = np.array([10,10]) # upper bound
lb = np.array([-10,-10]) # lower bound
N = 20 # swarm size
L = 8 # max iterations

def fitness(pos):
    return - (pos[0]**2 + pos[1]**2) + 200

# initialization
x = np.random.randint(-10,10,[N,D])

# lowest fitness --> target
Fitness = -np.inf
for pos in x[:]:
    fit = fitness(pos)
    if fit>Fitness:
        F=pos
        Fitness = fit
print(Fitness,F)

plt.scatter(x[1:,0],x[1:,1],c='r')
plt.scatter(x[0,0],x[0,1],c='b')
plt.show()

for l in range(1,L):
    c1 = 2*np.exp(-(4*l/L)**2)
    c2, c3 = np.random.random([2,D])

    # leader
    for dim in range(D):
        if (c3[dim]>=0):
            x[0,dim] = F[dim] + c1*( (ub[dim]-lb[dim])*c2[dim]+lb[dim] )
        else:
            x[0,dim] = F[dim] - c1*( (ub[dim]-lb[dim])*c2[dim]+lb[dim] )
            print("<0")

    # followers
    for i in range(1,N):
        x[i] = 0.5 * (x[i]+x[i-1])

    # move inside boundaries
    for i in range(N):
        for dim in range(D):
            if x[i,dim]>ub[dim]:
                x[i,dim]=ub[dim]
            elif x[i,dim]<lb[dim]:
                x[i,dim]=lb[dim]


    plt.scatter(x[1:,0],x[1:,1],c='r')
    plt.scatter(x[0,0],x[0,1],c='b')
    plt.show()

    for pos in x[:]:
        fit = fitness(pos)
        if fit>Fitness:
            F=pos
            Fitness = fit
    print(Fitness,F)
