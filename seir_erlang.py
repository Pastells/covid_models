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
parser = argparse.ArgumentParser(description='Stochastic mean-fiel SEIR model using the Gillespie algorithm')
parser.add_argument('--N',type=int,default=int(1e4),help="Fixed number of people")
parser.add_argument('--E0',type=int,default=0,help="Initial number of latent individuals")
parser.add_argument('--I0',type=int,default=10,help="Initial number of infected individuals")
parser.add_argument('--R0',type=int,default=0,help="Initial number of inmune individuals")
parser.add_argument('--delta1',type=float,default=0,help="Ratio of recovery from latent fase (E->R)")
parser.add_argument('--delta2',type=float,default=1,help="Ratio of recovery from infected fase (I->R)")
parser.add_argument('--k_rec',type=int,default=1,help="k parameter for the recovery time Erlang distribution, if set to 1 is an exponential distribution")
parser.add_argument('--beta1',type=float,default=0,help="Ratio of infection due to latent")
parser.add_argument('--beta2',type=float,default=1.5,help="Ratio of infection due to infected")
parser.add_argument('--k_inf',type=int,default=1,help="k parameter for the infection times (1/beta1, 1/beta2) Erlang distribution, if set to 1 is an exponential distribution")
parser.add_argument('--epsilon',type=float,default=1,help="Ratio of latency (E->I)")
parser.add_argument('--k_lat',type=int,default=1,help="k parameter for the latent times (1/delta1, 1/epsilon) Erlang distribution, if set to 1 is an exponential distribution")

parser.add_argument('--llavor',type=int,default=1,help="Llavor from the automatic configuration")
parser.add_argument('--data',type=str,default="italy_i.csv",help="File with time series")
parser.add_argument('--day_min',type=int,default=33,help="First day to consider on data series")
parser.add_argument('--day_max',type=int,default=58,help="Last day to consider on data series")

parser.add_argument('--nseed',type=int,default=100,help="Number of realizations")
parser.add_argument('--seed0',type=int,default=1,help="Initial seed")
args = parser.parse_args()

# -------------------------
# Parameters
N = args.N
k_inf=args.k_inf
k_rec=args.k_rec
k_lat=args.k_lat
beta1 = args.beta1/N*k_inf
beta2 = args.beta2/N*k_inf
delta1 = args.delta1*k_lat
epsilon = args.epsilon*k_lat
delta2 = args.delta2*k_rec
I0 = args.I0
E0 = args.E0
R0 = args.R0
T_steps = int(1e7) # max simulation steps
t_total = (args.day_max-args.day_min)*2 # max simulated days
nseed = args.nseed # MC realizations
seed0 = args.seed0
infected_time_series = genfromtxt(args.data, delimiter=',')[args.day_min:args.day_max]

print(args)


def beta_func(t):
    #t_conf = 20 # day of confinement
    #alpha = 0.5
    #delta_t = 5
    #if t<t_conf:
        return beta
    #else:
        #return beta*alpha + beta*(1-alpha)*np.exp(-(t-t_conf)/delta_t)

def time_dist(x):
    # Time intervals of a Poisson process follow an exponential distribution
    return -np.log(1-np.random.random())/x

# results per day and seed
#S_day,S_m,S_95 = np.zeros([nseed,t_total]),np.zeros(t_total),np.zeros([t_total,2])
#E_day,E_m,E_95 = np.zeros([nseed,t_total]),np.zeros(t_total),np.zeros([t_total,2])
I_day,I_m,I_95 = np.zeros([nseed,t_total]),np.zeros(t_total),np.zeros([t_total,2])
#R_day,R_m,R_95 = np.zeros([nseed,t_total]),np.zeros(t_total),np.zeros([t_total,2])

mc_step,day_max = 0,0
# =========================
# MC loop
# =========================
for seed in range(seed0,seed0+nseed):
    np.random.seed(seed)

    # -------------------------
    # Initialization
    S,E,I,R = np.zeros([T_steps,k_inf]),np.zeros([T_steps,k_lat]),np.zeros([T_steps,k_rec]),np.zeros(T_steps)
    E[0] = E0/k_lat
    I[0] = I0/k_rec
    R[0] = R0
    S[0] = (N-I0-R0-E0)/k_inf

    # -------------------------
    # Time loop
    # -------------------------
    t,time,day=0,0,1
    while (I[t]>0):
        if(time//day==1):
            day_max = max(day_max,day)
            #S_day[mc_step,day]=S[t]
            #E_day[mc_step,day]=E[t]
            I_day[mc_step,day]=I[t]
            #R_day[mc_step,day]=R[t]
            day += 1

        Stot = S[t].sum()
        Itot = I[t].sum()
        Etot = E[t].sum()
        lambda_sum = (epsilon+delta1)*Etot + delta2*Itot+ \
                     (beta1*Etot + beta2*Itot)*Stot

        prob_heal1 = delta1*E[t]/lambda_sum
        prob_heal2 = delta2*I[t]/lambda_sum
        prob_latent = epsilon*E[t]/lambda_sum
        prob_infect = (beta1*Etot + beta2*Itot)*S[t]/lambda_sum

        t+=1
        time += time_dist(lambda_sum)
        random = np.random.random()
        end=False
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
