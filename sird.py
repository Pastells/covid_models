# Stochastic mean-fiel SIR model using the Gillespie algorithm
# Pol Pastells, october 2020

# Equations of the deterministic system
#S[t] = S[t-1] - beta*I[t-1]*S[t-1]
#I[t] = I[t-1] + beta*I[t-1]*S[t-1] - (delta+delta_d) * I[t-1]
#R[t] = R[t-1] + delta * I[t-1]
#D[t] = D[t-1] + delta_d * I[t-1]

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

# -------------------------
# parser
parser = argparse.ArgumentParser(description='Stochastic mean-fiel SIR model using the Gillespie algorithm')
parser.add_argument('--N',type=int,default=int(1e4),help="Fixed number of people")
parser.add_argument('--I0',type=int,default=10,help="Initial number of infected individuals")
parser.add_argument('--R0',type=int,default=0,help="Initial number of inmune individuals")
parser.add_argument('--delta',type=float,default=1,help="Ratio of recovery")
parser.add_argument('--delta_d',type=float,default=0,help="Ratio of death")
parser.add_argument('--beta',type=float,default=1.5,help="Ratio of infection")
parser.add_argument('--nseed',type=int,default=100,help="Number of realizations")
parser.add_argument('--seed0',type=int,default=1,help="Initial seed")
args = parser.parse_args()


# -------------------------
# Parameters
N = args.N
beta = args.beta/N
delta = args.delta
delta_d = args.delta_d
I0 = args.I0
R0 = args.R0
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
S_day,I_day,R_day,D_day = np.zeros([nseed,t_total]),np.zeros([nseed,t_total]),np.zeros([nseed,t_total]),np.zeros([nseed,t_total])
S_m,I_m,R_m,D_m = np.zeros(t_total),np.zeros(t_total),np.zeros(t_total),np.zeros(t_total)
S_std,I_std,R_std,D_std = np.zeros(t_total),np.zeros(t_total),np.zeros(t_total),np.zeros(t_total)

mc_step = 0
day_max = 0
# =========================
# MC loop
# =========================
for seed in range(seed0,seed0+nseed):
    np.random.seed(seed)

    # -------------------------
    # Initialization
    S,I,R,D,T = np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps)
    I[0] = I0
    R[0] = R0
    D[0] = 0
    S[0] = N-I[0]-R[0]
    T[0]=0

    # -------------------------
    # Time loop
    # -------------------------
    t,time,day=0,0,1
    while (I[t]>0 and day<t_total-1):
        if(time//day==1):
            day += 1
            day_max = max(day_max,day)
            S_day[mc_step,day-1]=S[t]
            I_day[mc_step,day-1]=I[t]
            R_day[mc_step,day-1]=R[t]
            D_day[mc_step,day-1]=D[t]
        lambda_sum = (delta+delta_d)*I[t]+beta_func(t)*I[t]*S[t]
        prob_heal = delta*I[t]/lambda_sum
        prob_die = delta_d*I[t]/lambda_sum
        t+=1
        time += poisson_time(lambda_sum)
        T[t] = time
        random=np.random.random()
        if(random<prob_heal):
            # heal
            S[t] = S[t-1]
            I[t] = I[t-1] - 1
            R[t] = R[t-1] + 1
            D[t] = D[t-1]
        elif(random<(prob_heal+prob_die)):
            # die
            S[t] = S[t-1]
            I[t] = I[t-1] - 1
            R[t] = R[t-1]
            D[t] = D[t-1] + 1
        else:
            # infect
            S[t] = S[t-1] - 1
            I[t] = I[t-1] + 1
            R[t] = R[t-1]
            D[t] = D[t-1]
    # -------------------------

    # final value for the rest of time, otherwise it contributes with a zero when averaged
    S_day[mc_step,day:] = S_day[mc_step,day-1]
    I_day[mc_step,day:] = I_day[mc_step,day-1]
    R_day[mc_step,day:] = R_day[mc_step,day-1]
    D_day[mc_step,day:] = D_day[mc_step,day-1]

    mc_step += 1
# =========================

# plot all occurrences for all time without averaging
    #plt.plot(T[:t],I[:t])
#plt.show()

S_m = S_day.mean(0)
I_m = I_day.mean(0)
R_m = R_day.mean(0)
D_m = D_day.mean(0)
S_std = S_day.std(0)
I_std = I_day.std(0)
R_std = R_day.std(0)
D_std = D_day.std(0)

print(R_m[day_max])

# -------------------------
# Plot
plt.errorbar(np.arange(day_max),S_m[:day_max],yerr=S_std[:day_max],marker='o',ls='',label='S')
plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='',label='I')
plt.errorbar(np.arange(day_max),R_m[:day_max],yerr=R_std[:day_max],marker='o',ls='',label='R')
plt.errorbar(np.arange(day_max),D_m[:day_max],yerr=D_std[:day_max],marker='o',ls='',label='D')
plt.legend()
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Comparison with deterministic model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialization
T_steps = int(1e5)
S,I,R,D,T = np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps)
I[0] = I0
R[0] = R0
D[0] = 0
S[0] = N-I[0]-R[0]
T[0]=0

t=0
h=float(day_max)/T_steps
while (I[t]>0.01 and t<T_steps-1):
    t+=1
    S[t] = S[t-1] - h*beta_func(t)*I[t-1]*S[t-1]
    I[t] = I[t-1] + h*beta_func(t)*I[t-1]*S[t-1] - h*(delta+delta_d) * I[t-1]
    R[t] = R[t-1] + h*delta * I[t-1]
    D[t] = D[t-1] + h*delta_d * I[t-1]
    T[t] = T[t-1] + h

plt.plot(T[:t],S[:t],label='S_det')
plt.plot(T[:t],I[:t],label='I_det')
plt.plot(T[:t],R[:t],label='R_det')
plt.plot(T[:t],D[:t],label='D_det')
plt.errorbar(np.arange(day_max),S_m[:day_max],yerr=S_std[:day_max],marker='o',ls='',label='S')
plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='',label='I')
plt.errorbar(np.arange(day_max),R_m[:day_max],yerr=R_std[:day_max],marker='o',ls='',label='R')
plt.errorbar(np.arange(day_max),D_m[:day_max],yerr=D_std[:day_max],marker='o',ls='',label='D')
plt.legend()
plt.show()

#"N_{N}_I0_{I0}_R0_{R0}_delta_{delta}_delta_d_{delta_d}"
print args
time = time.localtime(time.time())
filename = "{}{}{}{}" % (sird_time.tm_year,time.tm_mon,time.tm_mday,time.tm_hour)
#with open(filename,'w') as f:
    #for i in range(day_max):
        #f.write("{i},{S[i]},{I[i]},{R[i]},{D[i]}"




