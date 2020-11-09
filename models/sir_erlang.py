"""
stochastic mean-field sir model
Uses the Gillespie algorithm and erlang distribution transition times
Pol Pastells, october 2020

equations of the deterministic system
s[t] = s[t-1] - beta*i[t-1]*s[t-1]
i[t] = i[t-1] + beta*i[t-1]*s[t-1] - delta * i[t-1]
r[t] = r[t-1] + delta * i[t-1]
"""

import numpy as np
import matplotlib.pyplot as plt

# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    args = parsing()
    i_0,r_0,t_steps,t_total,nseed,seed0,plot,infected_time_series,n, \
    k_inf,k_rec,beta,delta = parameters_init(args)


# results per day and seed
    days_gap = 5
    #s_day,s_m,s_95 = np.zeros([nseed,t_total+days_gap]),np.zeros(t_total+days_gap),np.zeros([t_total+days_gap,2])
    i_day,i_m,i_95 = np.zeros([nseed,t_total+days_gap]),np.zeros(t_total+days_gap),np.zeros([t_total+days_gap,2])
    #r_day,r_m,r_95 = np.zeros([nseed,t_total+days_gap]),np.zeros(t_total+days_gap),np.zeros([t_total+days_gap,2])

    mc_step,day_max = 0,0
# =========================
# MC loop
# =========================
    for seed in range(seed0,seed0+nseed):
        np.random.seed(seed)

        # -------------------------
        # initialization
        s,i,r = np.zeros([t_steps,k_inf+1]),np.zeros([t_steps,k_rec+1]),np.zeros(t_steps)
        s[0,:-1] = (n-i_0-r_0)/k_inf
        s[0,-1],i[0,:-1] = i_0/k_rec,i_0/k_rec
        i[0,-1],r[0] = r_0,r_0

        #s_day[mc_step,0]=s[0]
        i_day[mc_step,0]=i_0
        #r_day[mc_step,0]=r_0
        #T = np.zeros(t_steps)
        #T[0]=0
        t,time,day=0,0,1

        # Time loop
        while (i[t,:-1].sum()>0.1 and day<t_total-1):
            # Add individuals periodically
            #if time//add_n==1:
                #add_n += 30
                #s[t] += float(n)/2
            day = day_data(mc_step,t,time,day,day_max,i,i_day)
            t,time = gillespie(t,time,s,i,r,beta,delta,k_rec,k_inf)
        # -------------------------
        day = day_data(mc_step,t,time,day,day_max,i,i_day,True)


        # final value for the rest of time, otherwise it contributes with a zero when averaged
        #s_day[mc_step,day:] = s_day[mc_step,day-1]
        i_day[mc_step,day:] = i_day[mc_step,day-1]
        #r_day[mc_step,day:] = r_day[mc_step,day-1]

        if plot:
            plt.plot(i_day[mc_step,:])
            #plt.plot(T[:t],i[:t,:-1].sum(1),c='c')
        mc_step += 1
# =========================

    check_realization_alive=day_max//2

    for i in range(nseed):
        if i_day[i,check_realization_alive]!=0:
            x_var = i_day[i]
            alive_realizations = 1
            s_var = np.zeros(t_total+days_gap)
            i_m = x_var
            break

    for j in range(i+1,nseed):
        if i_day[j,check_realization_alive]!=0:
            alive_realizations += 1
            x_var = i_day[j]
            i_m_1 = i_m
            i_m = i_m_1 + (x_var-i_m_1)/alive_realizations
            s_var = s_var + (x_var-i_m_1)*(x_var-i_m)

    i_std = np.sqrt(s_var/(alive_realizations-1))

    if nseed-alive_realizations>nseed*0.1:
        print('The initial number of infected may be too low')
        print(f'Alive realizations after {check_realization_alive} days = {alive_realizations},\
              out of {nseed}')


    plt.errorbar(np.arange(day_max),i_m[:day_max],yerr=i_std[:day_max],\
                 marker='o',ls='',label='i mean')
    plt.show()

    #if plot:
        #plotting(infected_time_series,i_day,day_max,i_95,i_m,i_std)

    cost_func(infected_time_series,i_m,i_std)
# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%

# -------------------------
def parsing():
    """
    input parameters
    """
    import argparse
    parser = argparse.ArgumentParser(description='stochastic mean-field sir model using the \
                                     Gillespie algorithm and erlang distribution transition times',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n',type=int,default=int(1e4),
                        help="Fixed number of (effecitve) people [1000,1000000]")
    parser.add_argument('--i_0',type=int,default=20,
                        help="initial number of infected individuals [1,n]")
    parser.add_argument('--r_0',type=int,default=0,
                        help="initial number of inmune individuals [0,n]")
    parser.add_argument('--delta',type=float,default=0.2,
                        help="Mean ratio of recovery [1e-2,1]")
    parser.add_argument('--beta',type=float,default=0.5,
                        help="ratio of infection [1e-2,1]")
    parser.add_argument('--k_rec',type=int,default=1,
                        help="k parameter for the recovery time erlang distribution [1,5]")
    parser.add_argument('--k_inf',type=int,default=1,
                        help="k parameter for the infection time erlang distribution [1,5]")

    parser.add_argument('--llavor',type=int,default=1,
                        help="Llavor from the automatic configuration")
    parser.add_argument('--data',type=str,default="../data/italy_i.csv",
                        help="File with time series")
    parser.add_argument('--day_min',type=int,default=33,
                        help="First day to consider on data series")
    parser.add_argument('--day_max',type=int,default=58,
                        help="Last day to consider on data series")

    parser.add_argument('--nseed',type=int,default=int(3),
                        help="number of realizations, not really a parameter")
    parser.add_argument('--seed0',type=int,default=1,
                        help="initial seed, not really a parameter")
    parser.add_argument('--plot',action='store_true',
                        help="specify for plots")

    args = parser.parse_args()
    print(args)
    return args

# -------------------------
# Parameters

def parameters_init(args):
    from numpy import genfromtxt
    i_0 = args.i_0
    r_0 = args.r_0
    t_steps = int(1e7) # max simulation steps
    t_total = 100 #(args.day_max-args.day_min)*2 # max simulated days
    nseed = args.nseed # MC realizations
    seed0 = args.seed0
    plot = args.plot
    infected_time_series = genfromtxt(args.data, delimiter=',')[args.day_min:args.day_max]
    #print(infected_time_series)
    n = args.n
    k_inf=args.k_inf
    k_rec=args.k_rec
    beta = args.beta/n*k_inf
    delta = args.delta*k_rec
    return i_0,r_0,t_steps,t_total,nseed,seed0,plot,infected_time_series,n,k_inf,k_rec,beta,delta

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

def day_data(mc_step,t,time,day,day_max,i,i_day,last_event=False):
    """
    Write number of infected per day instead of event
    Also tracks day_max
    """
    if time//day == 1:
        days_jumped = int(time-day)
        #s_day[mc_step,day:day+days_jumped+1]=s[t:-1].sum()
        #r_day[mc_step,day:day+days_jumped+1]=r[t]
        i_day[mc_step,day:day+days_jumped+1]=i[t,:-1].sum()
        day += days_jumped
        day_max = max(day_max,day)
        day += 1
        return day
    if last_event:
        i_day[mc_step,day]=i[t,:-1].sum()
        day_max = max(day_max,day)
        day += 1
        return day
    return day
# -------------------------

def gillespie(t,time,s,i,r,beta,delta,k_rec,k_inf):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = s[t,:-1].sum()
    itot = i[t,:-1].sum()

    lambda_sum = (delta+beta_func(beta,t)*stot)*itot
    prob_heal = delta*i[t,:-1]/lambda_sum
    prob_infect = beta_func(beta,t)*s[t,:-1]*itot/lambda_sum

    t+=1
    time += time_dist(lambda_sum)
    #T[t] = time

    gillespie_step(t,s,i,r,prob_heal,prob_infect,k_rec,k_inf)
    return t,time
# -------------------------

def gillespie_step(t,s,i,r,prob_heal,prob_infect,k_rec,k_inf):
# s and i have one extra dimension to temporally store the infected and recovered after k stages, due to the erlang distribution
    random = np.random.random()
    prob_heal_tot = prob_heal.sum()

    # i(k)-> i(k+1)/r
    if random<prob_heal_tot:
        for k in range(k_rec):
            if random<prob_heal[:k+1].sum():
                s[t,:-1] = s[t-1,:-1]
                i[t,k]   = -1
                i[t,k+1] = 1
                r[t]     = r[t-1] + i[t,k_rec]
                i[t]    += i[t-1]
                break

    # s(k)-> s(k+1)/i(0)
    else:
        for k in range(k_inf):
            if random<(prob_heal_tot+prob_infect[:k+1].sum()):
                r[t]     = r[t-1]
                i[t,:-1] = i[t-1,:-1]
                s[t,k]   = -1
                s[t,k+1] = 1
                i[t,0]  += s[t,k_inf]
                s[t]    += s[t-1]
                break
# -------------------------

def plotting(infected_time_series,i_day,day_max,i_95,i_m,i_std):
    #s_m = s_day.mean(0)
    i_m = i_day.mean(0)
    i_std = i_day.std(0)
    #r_m = r_day.mean(0)
    #s_std = s_day.std(0)
    #r_std = r_day.std(0)
    #print(r_m[day_max],"recovered individuals")
    plt.errorbar(np.arange(day_max),i_m[:day_max],yerr=i_std[:day_max],\
                 marker='o',ls='',label='i mean')
    plt.show()
    plt.errorbar(np.arange(day_max),i_m[:day_max],yerr=i_std[:day_max],\
                 marker='o',ls='',label='i mean')

    i_m = np.median(i_day,0)

    alpha = 0.70
    p_l = ((1.0-alpha)/2.0) * 100
    p_u = (alpha+((1.0-alpha)/2.0)) * 100
    i_95[:,0] = np.percentile(i_day, p_l,0)
    i_95[:,1] = np.percentile(i_day, p_u,0)

    plt.plot(i_m,'o',c='orange',label='i median')
    plt.plot(i_95[:,0],c='orange')
    plt.plot(i_95[:,1],c='orange')

    plt.plot(infected_time_series,'o',label='data')
    plt.legend()
    plt.show()

# ~~~~~~~~~~~~~~~~~~~
# Output
# ~~~~~~~~~~~~~~~~~~~
def cost_func(infected_time_series,i_m,i_std):
    cost = 0
    for i,_ in enumerate(infected_time_series):
        cost += (i_m[i]-infected_time_series[i])**2/(1+i_std[i])
    cost = np.sqrt(cost)
    print(f"GGA SUCCESS {cost}")
# ~~~~~~~~~~~~~~~~~~~



if __name__ == "__main__":
    import traceback
    try:
        main()
    # handle error when running with --help
    except SystemExit as error:
        print(f"GGA CRASHED {1e20}")
        print(repr(error))
    except:
        print(f"GGA CRASHED {1e20}")
        traceback.print.exc()
