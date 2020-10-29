# Stochastic mean-fiel SEIR model using the Gillespie algorithm
# Pol Pastells, october 2020

# Equations of the deterministic system
#S[t] = S[t-1] - beta1*E[t-1]*S[t-1] - beta2*I[t-1]*S[t-1]
#E[t] = E[t-1] + beta1*E[t-1]*S[t-1] + beta2*I[t-1]*S[t-1] - (epsilon+delta1)*E[t-1]
#I[t] = I[t-1] + epsilon*E[t-1] - delta2 * I[t-1]
#R[t] = R[t-1] + delta1 *E[t-1] + delta2 * I[t-1]

import numpy as np
import matplotlib.pyplot as plt
import argparse

# -------------------------
# parser
parser = argparse.ArgumentParser(description='Stochastic mean-fiel SIR model using the Gillespie algorithm')
parser.add_argument('--N',type=int,default=int(1e4),help="Fixed number of people")
parser.add_argument('--E0',type=int,default=0,help="Initial number of latent individuals")
parser.add_argument('--I0',type=int,default=10,help="Initial number of infected individuals")
parser.add_argument('--R0',type=int,default=0,help="Initial number of inmune individuals")
parser.add_argument('--delta1',type=float,default=0,help="Ratio of recovery from latent fase (E->R)")
parser.add_argument('--delta2',type=float,default=1,help="Ratio of recovery from infected fase (I->R)")
parser.add_argument('--beta1',type=float,default=0,help="Ratio of infection due to latent")
parser.add_argument('--beta2',type=float,default=1.5,help="Ratio of infection due to infected")
parser.add_argument('--epsilon',type=float,default=1,help="Ratio of latency (E->I)")
parser.add_argument('--nseed',type=int,default=100,help="Number of realizations")
parser.add_argument('--seed0',type=int,default=1,help="Initial seed")
args = parser.parse_args()

# -------------------------
# Parameters
N = args.N
beta1 = args.beta1/N
beta2 = args.beta2/N
delta1 = args.delta1
delta2 = args.delta2
epsilon = args.epsilon
I0 = args.I0
E0 = args.E0
R0 = args.R0
nseed = args.nseed # MC realizations
seed0 = args.seed0

# If it breaks is probably due to this 2 being too low:
T_steps = int(1e6) # max simulation steps
t_total = 1000 # max simulated days

# Parameters for varible beta with time due to confinement
t_conf = 20 # day of confinement
alpha = 0.5 # weight
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
S_day,E_day,I_day,R_day = np.zeros([nseed,t_total]),np.zeros([nseed,t_total]),np.zeros([nseed,t_total]),np.zeros([nseed,t_total])
S_m,E_m,I_m,R_m = np.zeros(t_total),np.zeros(t_total),np.zeros(t_total),np.zeros(t_total)
S_std,E_std,I_std,R_std = np.zeros(t_total),np.zeros(t_total),np.zeros(t_total),np.zeros(t_total)

mc_step = 0
day_max = 0
# =========================
# MC loop
# =========================
for seed in range(seed0,seed0+nseed):
    np.random.seed(seed)

    # -------------------------
    # Initialization
    S,E,I,R,T = np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps),np.zeros(T_steps)
    E[0] = E0
    I[0] = I0
    R[0] = R0
    S[0] = N-I[0]-R[0]-E[0]
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
            E_day[mc_step,day-1]=E[t]
            I_day[mc_step,day-1]=I[t]
            R_day[mc_step,day-1]=R[t]

        lambda_sum = (epsilon+delta1)*E[t] + delta2*I[t]+ \
                     (beta1*E[t] + beta2*I[t])*S[t]

        #print(t,S[t],E[t],I[t],R[t],lambda_sum)
        prob_heal1 = delta1*E[t]/lambda_sum
        prob_heal2 = delta2*I[t]/lambda_sum
        prob_latent = epsilon*E[t]/lambda_sum

        t+=1
        time += poisson_time(lambda_sum)
        T[t] = time
        random = np.random.random()
        if(random<prob_heal1):
            # E->R
            S[t] = S[t-1]
            E[t] = E[t-1] - 1
            I[t] = I[t-1]
            R[t] = R[t-1] + 1
        elif(random<(prob_heal1+prob_heal2)):
            # I->R
            S[t] = S[t-1]
            E[t] = E[t-1]
            I[t] = I[t-1] - 1
            R[t] = R[t-1] + 1
        elif(random<(prob_heal1+prob_heal2+prob_latent)):
            # E->I
            S[t] = S[t-1]
            E[t] = E[t-1] - 1
            I[t] = I[t-1] + 1
            R[t] = R[t-1]
        else:
            # S->E
            S[t] = S[t-1] - 1
            E[t] = E[t-1] + 1
            I[t] = I[t-1]
            R[t] = R[t-1]
    # -------------------------

    # final value for the rest of time, otherwise it contributes with a zero when averaged
    S_day[mc_step,day:] = S_day[mc_step,day-1]
    E_day[mc_step,day:] = E_day[mc_step,day-1]
    I_day[mc_step,day:] = I_day[mc_step,day-1]
    R_day[mc_step,day:] = R_day[mc_step,day-1]

    mc_step += 1
# =========================

# plot all occurrences for all time without averaging
    #plt.plot(T[:t],I[:t])
#plt.show()

S_m = S_day.mean(0)
E_m = E_day.mean(0)
I_m = I_day.mean(0)
R_m = R_day.mean(0)
S_std = S_day.std(0)
E_std = E_day.std(0)
I_std = I_day.std(0)
R_std = R_day.std(0)

out_file = open("seir.dat","w")
for day in range(day_max):
    out_file.write("%s\n" % S_m[day])
out_file.close()


print(R_m[day_max])

# -------------------------
# Plot
#plt.errorbar(np.arange(day_max),S_m[:day_max],yerr=S_std[:day_max])
#plt.errorbar(np.arange(day_max),E_m[:day_max],yerr=E_std[:day_max])
plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='')
#plt.errorbar(np.arange(day_max),R_m[:day_max],yerr=R_std[:day_max])
plt.legend()
plt.show()
