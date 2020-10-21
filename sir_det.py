# Deterministic mean-fiel SIR model
# Pol Pastells, october 2020

import numpy as np
import matplotlib.pyplot as plt
import argparse

# -------------------------
# parser
parser = argparse.ArgumentParser(description='Stochastic mean-fiel SIR model using the Gillespie algorithm')
parser.add_argument('--N',type=int,default=int(1e4),help="Fixed number of people")
parser.add_argument('--I0',type=int,default=10,help="Initial number of infected individuals")
parser.add_argument('--R0',type=int,default=0,help="Initial number of inmune individuals")
parser.add_argument('--delta',type=float,default=1,help="Ratio of recovery")
parser.add_argument('--beta',type=float,default=1.5,help="Ratio of infection")
args = parser.parse_args()

# -------------------------
# Parameters
N = args.N
beta = args.beta/N
print(beta)
delta = args.delta
I0 = args.I0
R0 = args.R0
t_total = 100 # max simulated days
t_conf = 20 # day of confinement

alpha = 0.5
delta_t = 5

def beta_func(t):
    #if t<t_conf:
        return beta
    #else:
        #return beta*alpha + beta*(1-alpha)*np.exp(-(t-t_conf)/delta_t)

def poisson_time(x):
    # The time intervals of a Poisson process follow an exponential distribution
    return -np.log(1-np.random.random())/x



# Initialization
T_steps = int(1e4)
S,I,R,T = np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps)
I[0] = I0
R[0] = R0
S[0] = N-I[0]-R[0]
T[0]=0

t=0
day_max = 50
h=float(day_max)/T_steps
while (I[t]>0.5 and t<T_steps-1):
    t+=1
    S[t] = S[t-1] - h*beta_func(t)*I[t-1]*S[t-1]
    I[t] = I[t-1] + h*beta_func(t)*I[t-1]*S[t-1] - h*delta * I[t-1]
    R[t] = R[t-1] + h*delta * I[t-1]
    T[t] = T[t-1] + h

plt.plot(T[:t],S[:t],label='S')
plt.plot(T[:t],I[:t],label='I')
plt.plot(T[:t],R[:t],label='R')
plt.legend()
plt.show()
