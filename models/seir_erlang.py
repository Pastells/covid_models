"""
stochastic mean-field seir model using the Gillespie algorithm
Pol Pastells, october 2020

equations of the deterministic system
s[t] = s[t-1] - beta1*e[t-1]*s[t-1] - beta2*i[t-1]*s[t-1]
e[t] = e[t-1] + beta1*e[t-1]*s[t-1] + beta2*i[t-1]*s[t-1] - (epsilon+delta1)*e[t-1]
i[t] = i[t-1] + epsilon*e[t-1] - delta2 * i[t-1]
r[t] = r[t-1] + delta1 *e[t-1] + delta2 * i[t-1]
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    args = parsing()
    e_0,i_0,r_0,t_steps,t_total,nseed,seed0,plot,infected_time_series,n,\
    k_inf,k_rec,k_lat,beta1,beta2,delta1,epsilon,delta2 = parameters_init(args)

# results per day and seed
    #s_day,s_m,s_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    #e_day,e_m,e_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    i_day,i_m,i_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    #r_day,r_m,r_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])

    mc_step,day_max = 0,0
# =========================
# MC loop
# =========================
    for seed in range(seed0,seed0+nseed):
        np.random.seed(seed)

        # -------------------------
        # initialization
        s,e,i,r = np.zeros([t_steps,k_inf+1]),np.zeros([t_steps,k_lat+1,2]),np.zeros([t_steps,k_rec+1]),np.zeros(t_steps)
        s[0,:-1] = (n-i_0-r_0)/k_inf
        s[0,-1],e[0,:-1] = e_0/k_lat,e_0/k_lat
        e[0,-1],i[0,:-1] = i_0/k_rec,i_0/k_rec
        i[0,-1],r[0] = r_0,r_0

        #s_day[mc_step,0]=s[0]
        #e_day[mc_step,0]=e_0
        i_day[mc_step,0]=i_0
        #r_day[mc_step,0]=r_0
        T = np.zeros(t_steps)
        T[0]=0

        # -------------------------
        # Time loop
        # -------------------------
        t,time,day,add_n=0,0,1,20
        while (i[t,:-1].sum()>0 and day<t_total-1):
            #if time//add_n==1:
                #add_n += 30
                #s[t] += float(n)/2
            if time//day==1:
                days_jumped = int(time-day)
                #s_day[mc_step,day:day+days_jumped+1]=s[t:-1].sum()
                #e_day[mc_step,day:day+days_jumped+1]=e[t:-1].sum()
                i_day[mc_step,day:day+days_jumped+1]=i[t,:-1].sum()
                #r_day[mc_step,day:day+days_jumped+1]=r[t]
                day += days_jumped
                day_max = max(day_max,day)
                day += 1

            stot = s[t,:-1].sum()
            itot = i[t,:-1].sum()
            etot_rec = e[t,:-1,0].sum()
            etot_inf = e[t,:-1,1].sum()
            etot = etot_inf + etot_rec - e[t,0,0]

            lambda_sum = epsilon*etot_inf+ delta1*etot_rec + delta2*itot+ \
                         (beta1*etot + beta2*itot)*stot

            prob_heal1 = delta1*e[t,:-1,0]/lambda_sum
            prob_heal2 = delta2*i[t,:-1]/lambda_sum
            prob_latent = epsilon*e[t,:-1,1]/lambda_sum
            prob_infect = (beta1*etot + beta2*itot)*s[t,:-1]/lambda_sum


            t+=1
            time += time_dist(lambda_sum)
            if time > t_total:
                break # rare, but sometimes long times may appear
            T[t] = time

            gillespie_step(t,s,e,i,r,prob_heal1,prob_heal2,prob_latent,prob_infect,k_rec,k_lat,k_inf)
        # -------------------------
        # final value for the rest of time, otherwise it contributes with a zero when averaged
        #s_day[mc_step,day:] = s_day[mc_step,day-1]
        #e_day[mc_step,day:] = e_day[mc_step,day-1]
        i_day[mc_step,day:] = i_day[mc_step,day-1]
        #r_day[mc_step,day:] = r_day[mc_step,day-1]

        if plot:
            spacing = 100
            plt.plot(T[:t],s[:t,:-1].sum(1),c='r')
            plt.plot(T[:t],e[:t,:-1,0].sum(1),c='g')
            plt.plot(T[:t],e[:t,:-1,1].sum(1),c='b')
            plt.plot(T[:t],i[:t,:-1].sum(1),c='c')
            plt.plot(T[:t],r[:t],c='m')

        mc_step += 1
# =========================

    i_std = i_day.std(0)
    i_m = i_day.mean(0)
    if plot:
        plotting(infected_time_series,i_day,day_max,i_95)

    cost_func(infected_time_series,i_m,i_std)

# -------------------------
def parsing():
    """
    input parameters
    """
    import argparse
    parser = argparse.ArgumentParser(description='stochastic mean-field seir model using the \
                                     Gillespie algorithm and erlang distribution transition times',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--n',type=int,default=int(1e4),
                        help="Fixed number of people")
    parser.add_argument('--e_0',type=int,default=0,
                        help="initial number of latent individuals")
    parser.add_argument('--i_0',type=int,default=10,
                        help="initial number of infected individuals")
    parser.add_argument('--r_0',type=int,default=0,
                        help="initial number of inmune individuals")
    parser.add_argument('--delta1',type=float,default=0,
                        help="ratio of recovery from latent fase (e->r)")
    parser.add_argument('--delta2',type=float,default=1,
                        help="ratio of recovery from infected fase (i->r)")
    parser.add_argument('--k_rec',type=int,default=1,
                        help="k parameter for the recovery time erlang distribution, if set to 1 is an exponential distribution")
    parser.add_argument('--beta1',type=float,default=0,
                        help="ratio of infection due to latent")
    parser.add_argument('--beta2',type=float,default=1.5,
                        help="ratio of infection due to infected")
    parser.add_argument('--k_inf',type=int,default=1,
                        help="k parameter for the infection times (1/beta1, 1/beta2) erlang distribution")
    parser.add_argument('--epsilon',type=float,default=1,
                        help="ratio of latency (e->i)")
    parser.add_argument('--k_lat',type=int,default=1,
                        help="k parameter for the latent times (1/delta1, 1/epsilon) erlang distribution")

    parser.add_argument('--llavor',type=int,default=1,
                        help="Llavor from the automatic configuration")
    parser.add_argument('--data',type=str,default="../data/italy_i.csv",
                        help="File with time series")
    parser.add_argument('--day_min',type=int,default=33,
                        help="First day to consider on data series")
    parser.add_argument('--day_max',type=int,default=58,
                        help="Last day to consider on data series")

    parser.add_argument('--nseed',type=int,default=100,
                        help="number of realizations")
    parser.add_argument('--seed0',type=int,default=1,
                        help="initial seed")
    parser.add_argument('--plot',action='store_true',
                        help="specify for plots")

    args = parser.parse_args()
    print(args)
    return args

# -------------------------
# Parameters
def parameters_init(args):
    from numpy import genfromtxt
    e_0 = args.e_0
    i_0 = args.i_0
    r_0 = args.r_0
    t_steps = int(1e7) # max simulation steps
    t_total = (args.day_max-args.day_min)*2 # max simulated days
    nseed = args.nseed # MC realizations
    seed0 = args.seed0
    plot = args.plot
    infected_time_series = genfromtxt(args.data, delimiter=',')[args.day_min:args.day_max]
    #print(infected_time_series)
    n = args.n
    k_inf=args.k_inf
    k_rec=args.k_rec
    k_lat=args.k_lat
    beta1 = args.beta1/n*k_inf
    beta2 = args.beta2/n*k_inf
    delta1 = args.delta1*k_lat
    epsilon = args.epsilon*k_lat
    delta2 = args.delta2*k_rec
    return e_0,i_0,r_0,t_steps,t_total,nseed,seed0,plot,infected_time_series,n,k_inf,k_rec,k_lat,beta1,beta2,delta1,epsilon,delta2

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

def gillespie_step(t,s,e,i,r,prob_heal1,prob_heal2,prob_latent,prob_infect,k_rec,k_lat,k_inf):
    random = np.random.random()
    prob_heal1_tot = prob_heal1.sum()
    prob_heal2_tot = prob_heal2.sum()
    prob_latent_tot = prob_latent.sum()
    prob_infect_tot = prob_infect.sum()

    # e(k)-> e(k+1)/r
    if random<prob_heal1_tot:
        for k in range(k_lat):
            if random<prob_heal1[:k+1].sum():
                s[t,:-1]   = s[t-1,:-1]
                i[t,:-1]   = i[t-1,:-1]
                e[t,k,0]   = -1
                e[t,k+1,0] = 1
                r[t]       = r[t-1] + e[t,k_lat,0]
                e[t,0,1]   = e[t,0,0]
                e[t]      += e[t-1]
                break

    # i(k)-> i(k+1)/r
    elif random<(prob_heal1_tot+prob_heal2_tot):
        random -= prob_heal1_tot
        for k in range(k_rec):
            if random<prob_heal2[:k+1].sum():
                s[t,:-1] = s[t-1,:-1]
                e[t,:-1] = e[t-1,:-1]
                i[t,k]   = -1
                i[t,k+1] = 1
                r[t]     = r[t-1] + i[t,k_rec]
                i[t]    += i[t-1]
                break

    # e(k)-> e(k+1)/i(0)
    elif random<(prob_heal1_tot+prob_heal2_tot+prob_latent_tot):
        random -= (prob_heal1_tot+prob_heal2_tot)
        for k in range(k_lat):
            if random<prob_latent[:k+1].sum():
                s[t,:-1]   = s[t-1,:-1]
                i[t,:-1]   = i[t-1,:-1]
                r[t]       = r[t-1]
                e[t,k,1]   = -1
                e[t,k+1,1] = 1
                i[t,0]    += e[t,k_lat,1]
                e[t,0,0]   = e[t,0,1]
                e[t]      += e[t-1]
                break

    # s(k)-> s(k+1)/e(0)
    else:
        random -= (prob_heal1_tot+prob_heal2_tot+prob_latent_tot)
        for k in range(k_inf):
            if random<prob_infect[:k+1].sum():
                e[t,:-1] = e[t-1,:-1]
                i[t,:-1] = i[t-1,:-1]
                r[t]     = r[t-1]
                s[t,k]   = -1
                s[t,k+1] = 1
                e[t,0]  += s[t,k_inf]
                s[t]    += s[t-1]
                break
# -------------------------

def plotting(infected_time_series,i_day,day_max,i_95):
    spacing = 100
    #plt.plot(T[:t:spacing],s[:t:spacing,:-1].sum(1),label='s',c='r')
    #plt.plot(T[:t:spacing],e[:t:spacing,:-1,0].sum(1),label='e_rec',c='g')
    #plt.plot(T[:t:spacing],e[:t:spacing,:-1,1].sum(1),label='e_inf',c='b')
    #plt.plot(T[:t:spacing],i[:t:spacing,:-1].sum(1),label='i',c='c')
    #plt.plot(T[:t:spacing],r[:t:spacing],label='r',c='m')
    #plt.legend()
    plt.show()

    #s_m = s_day.mean(0)
    #e_m = e_day.mean(0)
    i_m = i_day.mean(0)
    #r_m = r_day.mean(0)
    #s_std = s_day.std(0)
    #e_std = e_day.std(0)
    i_std = i_day.std(0)
    #r_std = r_day.std(0)

    #out_file = open("seir.dat","w")
    #for day in range(day_max):
        #out_file.write("%s\n" % s_m[day])
    #out_file.close()

    #plt.errorbar(np.arange(day_max),s_m[:day_max],yerr=s_std[:day_max])
    #plt.errorbar(np.arange(day_max),e_m[:day_max],yerr=e_std[:day_max])
    plt.errorbar(np.arange(day_max),i_m[:day_max],yerr=i_std[:day_max],marker='o',ls='',label='i mean')
    #plt.errorbar(np.arange(day_max),r_m[:day_max],yerr=r_std[:day_max])

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
    for i in range(len(infected_time_series)):
        cost += (i_m[i]-infected_time_series[i])**2/(1+i_std[i])
    cost = np.sqrt(cost)
    print(f"GGA sUCCess {cost}")
# ~~~~~~~~~~~~~~~~~~~



if __name__ == "__main__":
    #import traceback
    try:
        main()
    except:
        print(f"GGA CrAsHeD {1e20}")
        traceback.print.exc()



