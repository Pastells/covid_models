# Stochastic mean-field SIR model using the Gillespie algorithm
# Pol Pastells, october 2020

# Equations of the deterministic system
#S[t] = S[t-1] - beta*I[t-1]*S[t-1]
#I[t] = I[t-1] + beta*I[t-1]*S[t-1] - delta * I[t-1]
#R[t] = R[t-1] + delta * I[t-1]

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import argparse
import sys

# -------------------------
# parser
parser = argparse.ArgumentParser(description='Stochastic mean-fiel SIR model using the Gillespie algorithm',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--N',type=int,default=int(1e4),help="Fixed number of (effecitve) people [1000,1000000]")
parser.add_argument('--I0',type=int,default=20,help="Initial number of infected individuals [1,N]")
parser.add_argument('--R0',type=int,default=0,help="Initial number of inmune individuals [0,N]")
parser.add_argument('--delta',type=float,default=0.2,help="Ratio of recovery [1e-2,1]")
parser.add_argument('--beta',type=float,default=0.5,help="Ratio of infection [1e-2,1]")
parser.add_argument('--erlang_k',type=int,default=1,help="k parameter for the Erlang distribution, if set to 1 is an exponential distribution")

parser.add_argument('--llavor',type=int,default=1,help="Llavor from the automatic configuration")
parser.add_argument('--data',type=str,default="../data/italy_i.csv",help="File with time series")
parser.add_argument('--day_min',type=int,default=33,help="First day to consider on data series")
parser.add_argument('--day_max',type=int,default=58,help="Last day to consider on data series")

parser.add_argument('--nseed',type=int,default=int(1e2),help="Number of realizations, not really a parameter")
parser.add_argument('--seed0',type=int,default=1,help="Initial seed, not really a parameter")

args = parser.parse_args()
# -------------------------
# Parameters
N = args.N
beta = args.beta/N
delta = args.delta
I0 = args.I0
R0 = args.R0
erlang_k = args.erlang_k
T_steps = int(1e7) # max simulation steps
t_total = 100 # max simulated days
nseed = args.nseed # MC realizations
seed0 = args.seed0
infected_time_series = genfromtxt(args.data, delimiter=',')[args.day_min:args.day_max]

print(args)
print(infected_time_series)

def beta_func(t):
    #t_conf = 20 # day of confinement
    #alpha = 0.5
    #delta_t = 5
    #if t<t_conf:
        return beta
    #else:
        #return beta*alpha + beta*(1-alpha)*np.exp(-(t-t_conf)/delta_t)

def time_dist(x,k):
    # Time intervals of a Poisson process follow an exponential distribution
    random,erlang = np.random.random(k),1
    for i in random:
        erlang *= i

    return -np.log(1-erlang)/x


# results per day and seed
#S_day,S_m,S_95 = np.zeros([nseed,t_total]),np.zeros(t_total),np.zeros([t_total,2])
I_day,I_m,I_95 = np.zeros([nseed,t_total]),np.zeros(t_total),np.zeros([t_total,2])
R_day,R_m,R_95 = np.zeros([nseed,t_total]),np.zeros(t_total),np.zeros([t_total,2])

mc_step,day_max = 0,0
# =========================
# MC loop
# =========================
for seed in range(seed0,seed0+nseed):
    np.random.seed(seed)

    # -------------------------
    # Initialization
    S,I,R = np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps)
    I[0] = I0
    R[0] = R0
    S[0] = N-I[0]-R[0]
    I_day[mc_step,0]=I[0]
    R_day[mc_step,0]=R[0]
    #T = np.zeros(T_steps)
    #T[0]=0
    #S_day[mc_step,0]=S[0]

    # -------------------------
    # Time loop
    # -------------------------
    t,time,day,add_n=0,0,1,20
    while (I[t]>0 and day<t_total-1):
        # Add individuals periodically
        #if(time//add_n==1):
            #add_n += 30
            #S[t] += float(N)/2
        if(time//day==1):
            day_max = max(day_max,day)
            #S_day[mc_step,day]=S[t]
            I_day[mc_step,day]=I[t]
            R_day[mc_step,day]=R[t]
            day += 1
        lambda_sum = delta*I[t]+beta_func(t)*I[t]*S[t]
        prob_heal = delta*I[t]/lambda_sum
        t+=1
        time += time_dist(lambda_sum,erlang_k)
        #T[t] = time
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
    #S_day[mc_step,day:] = S_day[mc_step,day-1]
    I_day[mc_step,day:] = I_day[mc_step,day-1]
    R_day[mc_step,day:] = R_day[mc_step,day-1]

    plt.plot(I_day[mc_step,:])

    mc_step += 1
# =========================

plt.show()




# ~~~~~~~~~~~~~~~~~~~
#Plots
# ~~~~~~~~~~~~~~~~~~~

#S_m = S_day.mean(0)
I_m = I_day.mean(0)
R_m = R_day.mean(0)
#S_std = S_day.std(0)
I_std = I_day.std(0)
R_std = R_day.std(0)
print(R_m[day_max],"Recovered individuals")
plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='',label='I mean')

I_m = np.median(I_day,0)

alpha = 0.70
p_l = ((1.0-alpha)/2.0) * 100
p_u = (alpha+((1.0-alpha)/2.0)) * 100
I_95[:,0] = np.percentile(I_day, p_l,0)
I_95[:,1] = np.percentile(I_day, p_u,0)

plt.plot(I_m,'o',c='orange',label='I median')
plt.plot(I_95[:,0],c='orange')
plt.plot(I_95[:,1],c='orange')

plt.plot(infected_time_series,'o',label='data')
plt.legend()
plt.show()

# ~~~~~~~~~~~~~~~~~~~
# Output
# ~~~~~~~~~~~~~~~~~~~
cost = 0
for i in range(len(infected_time_series)):
    cost += (I_m[i]-infected_time_series[i])**2/(1+I_std[i])
cost = np.sqrt(cost)
print("GGA SUCCESS {}".format(cost))
