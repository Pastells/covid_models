# Stochastic mean-fiel SIR model using the Gillespie algorithm
# Pol Pastells, october 2020

# Equations of the system
#S[t] = S[t-1] - beta_func(t)*I[t-1]*S[t-1]
#I[t] = I[t-1] + beta_func(t)*I[t-1]*S[t-1] - delta * I[t-1]
#R[t] = R[t-1] + delta * I[t-1]

import numpy as np
import matplotlib.pyplot as plt
import argparse

# -------------------------
# parser
parser = argparse.ArgumentParser(description='Stochastic mean-fiel SIR model using the Gillespie algorithm')
parser.add_argument('--N',type=int,default=int(1e4),help="Fixed number of people")
parser.add_argument('--I0',type=int,default=10,help="Initial number of infected")
parser.add_argument('--delta',type=float,default=1,help="Ratio of recovery")
parser.add_argument('--beta',type=float,default=1.5,help="Ratio of infection")
parser.add_argument('--nseed',type=int,default=100,help="Number of realizations")
parser.add_argument('--seed0',type=int,default=1,help="Initial seed")
args = parser.parse_args()

# -------------------------
# Parameters
N = args.N
beta = args.beta
delta = args.delta
I0 = args.I0
T_steps = int(1e6) # max simulation steps
t_total = 100 # max simulated days
nseed = args.nseed # MC realizations
seed0 = args.seed0
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


# results per day and seed
S_day,I_day,R_day = np.zeros([nseed,t_total]),np.zeros([nseed,t_total]),np.zeros([nseed,t_total])
S_m,I_m,R_m = np.zeros(t_total),np.zeros(t_total),np.zeros(t_total)
S_std,I_std,R_std = np.zeros(t_total),np.zeros(t_total),np.zeros(t_total)

mc_step = 0
day_max = 0
# =========================
# MC loop
# =========================
for seed in range(seed0,seed0+nseed):
    np.random.seed(seed)

    # -------------------------
    # Initialization
    S,I,R,T = np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps)
    I[0] = I0
    S[0] = N-I[0]
    T[0]=0

    # -------------------------
    # Time loop
    # -------------------------
    t,time,day=0,0,1
    while (I[t]>0):
        if(time//day==1):
            day += 1
            day_max = max(day_max,day)
            S_day[mc_step,day-1]=S[t]
            I_day[mc_step,day-1]=I[t]
            R_day[mc_step,day-1]=R[t]
        lambda_sum = delta*I[t]+beta_func(t)*I[t]*S[t]/N
        prob_heal = delta*I[t]/lambda_sum
        t+=1
        time += poisson_time(lambda_sum)
        T[t] = time
        if(np.random.random()<prob_heal):
            # heal
            S[t] = S[t-1]
            I[t] = I[t-1] - 1
            R[t] = R[t-1] + 1
        else:
            # infect
            S[t] = S[t-1] - 1
            I[t] = I[t-1] + 1
            R[t] = R[t-1]
    # -------------------------

    # final value for the rest of time, otherwise it contributes with a zero when averaged
    S_day[mc_step,day:] = S_day[mc_step,day-1]
    I_day[mc_step,day:] = I_day[mc_step,day-1]
    R_day[mc_step,day:] = R_day[mc_step,day-1]

    mc_step += 1
# =========================

# plot all occurrences for all time without averaging
    #plt.plot(T[:t],I[:t])
#plt.show()

S_m = S_day.mean(0)
I_m = I_day.mean(0)
R_m = R_day.mean(0)
S_std = S_day.std(0)
I_std = I_day.std(0)
R_std = R_day.std(0)

# -------------------------
# Plot
#plt.errorbar(np.arange(day_max),S_m[:day_max],yerr=S_std[:day_max])
plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='')
#plt.errorbar(np.arange(day_max),R_m[:day_max],yerr=R_std[:day_max])
plt.legend()
plt.grid()
plt.show()
