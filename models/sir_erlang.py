# Stochastic mean-field SIR model using the Gillespie algorithm and Erlang distribution transition times
# Pol Pastells, october 2020

# Equations of the deterministic system
#S[t] = S[t-1] - beta*I[t-1]*S[t-1]
#I[t] = I[t-1] + beta*I[t-1]*S[t-1] - delta * I[t-1]
#R[t] = R[t-1] + delta * I[t-1]

import numpy as np
import matplotlib.pyplot as plt;

# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    args = parsing()
    I0,R0,T_steps,t_total,nseed,seed0,plot,infected_time_series,N,k_inf,k_rec,beta,delta = parameters_init(args)


# results per day and seed
    #S_day,S_m,S_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    I_day,I_m,I_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    R_day,R_m,R_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])

    mc_step,day_max = 0,0
# =========================
# MC loop
# =========================
    for seed in range(seed0,seed0+nseed):
        np.random.seed(seed)

        # -------------------------
        # Initialization
        S,I,R = np.zeros([T_steps,k_inf+1]),np.zeros([T_steps,k_rec+1]),np.zeros(T_steps)
        S[0,:-1] = (N-I0-R0)/k_inf
        S[0,-1],I[0,:-1] = I0/k_rec,I0/k_rec
        I[0,-1],R[0] = R0,R0

        #S_day[mc_step,0]=S[0]
        I_day[mc_step,0]=I0
        #R_day[mc_step,0]=R0
        #T = np.zeros(T_steps)
        #T[0]=0

        # -------------------------
        # Time loop
        # -------------------------
        t,time,day,add_n=0,0,1,20
        while (I[t,:-1].sum()>0.1 and day<t_total-1):
            # Add individuals periodically
            #if(time//add_n==1):
                #add_n += 30
                #S[t] += float(N)/2
            if(time//day==1):
                days_jumped = int(time-day)
                #S_day[mc_step,day:day+days_jumped+1]=S[t:-1].sum()
                I_day[mc_step,day:day+days_jumped+1]=I[t,:-1].sum()
                #R_day[mc_step,day:day+days_jumped+1]=R[t]
                day += days_jumped
                day_max = max(day_max,day)
                day += 1

            Stot = S[t,:-1].sum()
            Itot = I[t,:-1].sum()

            lambda_sum = (delta+beta_func(beta,t)*Stot)*Itot
            prob_heal = delta*I[t,:-1]/lambda_sum
            prob_infect = beta_func(beta,t)*S[t,:-1]*Itot/lambda_sum

            #print(I[t],R[t])
            t+=1
            time += time_dist(lambda_sum)
            #T[t] = time

            gillespie_step(t,S,I,R,prob_heal,prob_infect,k_rec,k_inf)
            #print(I[t],R[t])
            #print(I[t,:-1].sum())
        # -------------------------
        if(time//day==1):
            days_jumped = int(time-day)
            I_day[mc_step,day:day+days_jumped+1]=I[t,:-1].sum()
            day += days_jumped
            day_max = max(day_max,day)
            day += 1
        else:
            I_day[mc_step,day]=I[t,:-1].sum()
            day_max = max(day_max,day)
            day += 1


        # final value for the rest of time, otherwise it contributes with a zero when averaged
        #S_day[mc_step,day:] = S_day[mc_step,day-1]
        I_day[mc_step,day:] = I_day[mc_step,day-1]
        #R_day[mc_step,day:] = R_day[mc_step,day-1]

        if(plot):
            plt.plot(I_day[mc_step,:])
            #plt.plot(T[:t],I[:t,:-1].sum(1),c='c')
        mc_step += 1
# =========================

    check_realization_alive=day_max//2

    for i in range(nseed):
        if(I_day[i,check_realization_alive]!=0):
            x_var = I_day[i]
            alive_realizations = 1
            S_var = np.zeros(t_total+10)
            I_m = x_var
            break

    for j in range(i+1,nseed):
        if(I_day[j,check_realization_alive]!=0):
            alive_realizations += 1
            x_var = I_day[j]
            I_m_1 = I_m
            I_m = I_m_1 + (x_var-I_m_1)/alive_realizations
            S_var = S_var + (x_var-I_m_1)*(x_var-I_m)

    I_std = np.sqrt(S_var/(alive_realizations-1))

    if(nseed-alive_realizations>nseed*0.1):
        print('The initial number of infected may be too low')
        print(f'Alive realizations after {check_realization_alive} days = {alive_realizations}, out of {nseed}')


    plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='',label='I mean')
    plt.show();

    #if(plot): plotting(infected_time_series,I_day,day_max,I_95,I_m,I_std);

    cost_func(infected_time_series,I_m,I_std)
# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%

# -------------------------
def parsing():
    import argparse
    parser = argparse.ArgumentParser(description='Stochastic mean-field SIR model using the Gillespie algorithm and Erlang distribution transition times',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--N',type=int,default=int(1e4),help="Fixed number of (effecitve) people [1000,1000000]")
    parser.add_argument('--I0',type=int,default=20,help="Initial number of infected individuals [1,N]")
    parser.add_argument('--R0',type=int,default=0,help="Initial number of inmune individuals [0,N]")
    parser.add_argument('--delta',type=float,default=0.2,help="Mean ratio of recovery [1e-2,1]")
    parser.add_argument('--beta',type=float,default=0.5,help="Ratio of infection [1e-2,1]")
    parser.add_argument('--k_rec',type=int,default=1,help="k parameter for the recovery time Erlang distribution, if set to 1 is an exponential distribution")
    parser.add_argument('--k_inf',type=int,default=1,help="k parameter for the infection time Erlang distribution, if set to 1 is an exponential distribution")

    parser.add_argument('--llavor',type=int,default=1,help="Llavor from the automatic configuration")
    parser.add_argument('--data',type=str,default="../data/italy_i.csv",help="File with time series")
    parser.add_argument('--day_min',type=int,default=33,help="First day to consider on data series")
    parser.add_argument('--day_max',type=int,default=58,help="Last day to consider on data series")

    parser.add_argument('--nseed',type=int,default=int(1e2),help="Number of realizations, not really a parameter")
    parser.add_argument('--seed0',type=int,default=1,help="Initial seed, not really a parameter")
    parser.add_argument('--plot',action='store_true',help="Specify for plots")

    args = parser.parse_args()
    print(args)
    return args

# -------------------------
# Parameters

def parameters_init(args):
    from numpy import genfromtxt
    I0 = args.I0
    R0 = args.R0
    T_steps = int(1e7) # max simulation steps
    t_total = 100 #(args.day_max-args.day_min)*2 # max simulated days
    nseed = args.nseed # MC realizations
    seed0 = args.seed0
    plot = args.plot
    infected_time_series = genfromtxt(args.data, delimiter=',')[args.day_min:args.day_max]
    #print(infected_time_series)
    N = args.N
    k_inf=args.k_inf
    k_rec=args.k_rec
    beta = args.beta/N*k_inf
    delta = args.delta*k_rec
    return I0,R0,T_steps,t_total,nseed,seed0,plot,infected_time_series,N,k_inf,k_rec,beta,delta

# -------------------------

def beta_func(beta,t):
    #t_conf = 20 # day of confinement
    #alpha = 0.5
    #delta_t = 5
    #if t<t_conf:
        return beta
    #else:
        #return beta*alpha + beta*(1-alpha)*np.exp(-(t-t_conf)/delta_t)

# Time intervals of a Poisson process follow an exponential distribution
def time_dist(x):
    return -np.log(1-np.random.random())/x
# -------------------------

def gillespie_step(t,S,I,R,prob_heal,prob_infect,k_rec,k_inf):
# S and I have one extra dimension to temporally store the infected and recovered after k stages, due to the Erlang distribution
    random = np.random.random()
    prob_heal_tot = prob_heal.sum()

    # I(k)-> I(k+1)/R
    if(random<prob_heal_tot):
        for k in range(k_rec):
            if(random<prob_heal[:k+1].sum()):
                S[t,:-1] = S[t-1,:-1]
                I[t,k]   = -1
                I[t,k+1] = 1
                R[t]     = R[t-1] + I[t,k_rec]
                I[t]    += I[t-1]
                break

    # S(k)-> S(k+1)/I(0)
    else:
        for k in range(k_inf):
            if(random<(prob_heal_tot+prob_infect[:k+1].sum())):
                R[t]     = R[t-1]
                I[t,:-1] = I[t-1,:-1]
                S[t,k]   = -1
                S[t,k+1] = 1
                I[t,0]  += S[t,k_inf]
                S[t]    += S[t-1]
                break
# -------------------------

def plotting(infected_time_series,I_day,day_max,I_95,I_m,I_std):
    #S_m = S_day.mean(0)
    I_m = I_day.mean(0)
    I_std = I_day.std(0)
    #R_m = R_day.mean(0)
    #S_std = S_day.std(0)
    #R_std = R_day.std(0)
    #print(R_m[day_max],"Recovered individuals")
    plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='',label='I mean')
    plt.show();
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
def cost_func(infected_time_series,I_m,I_std):
    cost = 0
    for i in range(len(infected_time_series)):
        cost += (I_m[i]-infected_time_series[i])**2/(1+I_std[i])
    cost = np.sqrt(cost)
    print(f"GGA SUCCESS {cost}")
# ~~~~~~~~~~~~~~~~~~~



if __name__ == "__main__":
    #import traceback
    try:
        main()
    except:
        print(f"GGA CRASHED {1e20}")
        traceback.print.exc()
