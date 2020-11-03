# Stochastic mean-field SEIR model using the Gillespie algorithm
# Pol Pastells, october 2020

# Equations of the deterministic system
#S[t] = S[t-1] - beta1*E[t-1]*S[t-1] - beta2*I[t-1]*S[t-1]
#E[t] = E[t-1] + beta1*E[t-1]*S[t-1] + beta2*I[t-1]*S[t-1] - (epsilon+delta1)*E[t-1]
#I[t] = I[t-1] + epsilon*E[t-1] - delta2 * I[t-1]
#R[t] = R[t-1] + delta1 *E[t-1] + delta2 * I[t-1]

import numpy as np
import matplotlib.pyplot as plt

def main():
    args = parsing()
    E0,I0,R0,T_steps,t_total,nseed,seed0,plot,infected_time_series,N,k_inf,k_rec,k_lat,beta1,beta2,delta1,epsilon,delta2 = parameters_init(args)

# results per day and seed
    #S_day,S_m,S_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    #E_day,E_m,E_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    I_day,I_m,I_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])
    #R_day,R_m,R_95 = np.zeros([nseed,t_total+10]),np.zeros(t_total+10),np.zeros([t_total+10,2])

    mc_step,day_max = 0,0
# =========================
# MC loop
# =========================
    for seed in range(seed0,seed0+nseed):
        np.random.seed(seed)

        # -------------------------
        # Initialization
        S,E,I,R = np.zeros([T_steps,k_inf+1]),np.zeros([T_steps,k_lat+1,2]),np.zeros([T_steps,k_rec+1]),np.zeros(T_steps)
        S[0,:-1] = (N-I0-R0)/k_inf
        S[0,-1],E[0,:-1] = E0/k_lat,E0/k_lat
        E[0,-1],I[0,:-1] = I0/k_rec,I0/k_rec
        I[0,-1],R[0] = R0,R0

        #S_day[mc_step,0]=S[0]
        #E_day[mc_step,0]=E0
        I_day[mc_step,0]=I0
        #R_day[mc_step,0]=R0
        T = np.zeros(T_steps)
        T[0]=0

        # -------------------------
        # Time loop
        # -------------------------
        t,time,day,add_n=0,0,1,20
        while (I[t,:-1].sum()>0 and day<t_total-1):
            #if(time//add_n==1):
                #add_n += 30
                #S[t] += float(N)/2
            if(time//day==1):
                days_jumped = int(time-day)
                #S_day[mc_step,day:day+days_jumped+1]=S[t:-1].sum()
                #E_day[mc_step,day:day+days_jumped+1]=E[t:-1].sum()
                I_day[mc_step,day:day+days_jumped+1]=I[t,:-1].sum()
                #R_day[mc_step,day:day+days_jumped+1]=R[t]
                day += days_jumped
                day_max = max(day_max,day)
                day += 1

            Stot = S[t,:-1].sum()
            Itot = I[t,:-1].sum()
            Etot_rec = E[t,:-1,0].sum()
            Etot_inf = E[t,:-1,1].sum()
            Etot = Etot_inf + Etot_rec - E[t,0,0]

            lambda_sum = epsilon*Etot_inf+ delta1*Etot_rec + delta2*Itot+ \
                         (beta1*Etot + beta2*Itot)*Stot

            prob_heal1 = delta1*E[t,:-1,0]/lambda_sum
            prob_heal2 = delta2*I[t,:-1]/lambda_sum
            prob_latent = epsilon*E[t,:-1,1]/lambda_sum
            prob_infect = (beta1*Etot + beta2*Itot)*S[t,:-1]/lambda_sum


            t+=1
            time += time_dist(lambda_sum)
            if (time > t_total): break; # rare, but sometimes long times may appear
            T[t] = time

            gillespie_step(t,S,E,I,R,prob_heal1,prob_heal2,prob_latent,prob_infect,k_rec,k_lat,k_inf)
        # -------------------------
        # final value for the rest of time, otherwise it contributes with a zero when averaged
        #S_day[mc_step,day:] = S_day[mc_step,day-1]
        #E_day[mc_step,day:] = E_day[mc_step,day-1]
        I_day[mc_step,day:] = I_day[mc_step,day-1]
        #R_day[mc_step,day:] = R_day[mc_step,day-1]

        if(plot):
            spacing = 100
            plt.plot(T[:t],S[:t,:-1].sum(1),c='r')
            plt.plot(T[:t],E[:t,:-1,0].sum(1),c='g')
            plt.plot(T[:t],E[:t,:-1,1].sum(1),c='b')
            plt.plot(T[:t],I[:t,:-1].sum(1),c='c')
            plt.plot(T[:t],R[:t],c='m')

        mc_step += 1
# =========================

    I_std = I_day.std(0)
    I_m = I_day.mean(0)
    if(plot): plotting(infected_time_series,I_day,day_max,I_95);

    cost_func(infected_time_series,I_m,I_std)

# -------------------------
def parsing():
    import argparse
    parser = argparse.ArgumentParser(description='Stochastic mean-field SEIR model using the Gillespie algorithm')
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
    parser.add_argument('--data',type=str,default="../data/italy_i.csv",help="File with time series")
    parser.add_argument('--day_min',type=int,default=33,help="First day to consider on data series")
    parser.add_argument('--day_max',type=int,default=58,help="Last day to consider on data series")

    parser.add_argument('--nseed',type=int,default=100,help="Number of realizations")
    parser.add_argument('--seed0',type=int,default=1,help="Initial seed")
    parser.add_argument('--plot',action='store_true',help="Specify for plots")

    args = parser.parse_args()
    print(args)
    return args

# -------------------------
# Parameters
def parameters_init(args):
    from numpy import genfromtxt
    E0 = args.E0
    I0 = args.I0
    R0 = args.R0
    T_steps = int(1e7) # max simulation steps
    t_total = (args.day_max-args.day_min)*2 # max simulated days
    nseed = args.nseed # MC realizations
    seed0 = args.seed0
    plot = args.plot
    infected_time_series = genfromtxt(args.data, delimiter=',')[args.day_min:args.day_max]
    #print(infected_time_series)
    N = args.N
    k_inf=args.k_inf
    k_rec=args.k_rec
    k_lat=args.k_lat
    beta1 = args.beta1/N*k_inf
    beta2 = args.beta2/N*k_inf
    delta1 = args.delta1*k_lat
    epsilon = args.epsilon*k_lat
    delta2 = args.delta2*k_rec
    return E0,I0,R0,T_steps,t_total,nseed,seed0,plot,infected_time_series,N,k_inf,k_rec,k_lat,beta1,beta2,delta1,epsilon,delta2

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

def gillespie_step(t,S,E,I,R,prob_heal1,prob_heal2,prob_latent,prob_infect,k_rec,k_lat,k_inf):
    random = np.random.random()
    prob_heal1_tot = prob_heal1.sum()
    prob_heal2_tot = prob_heal2.sum()
    prob_latent_tot = prob_latent.sum()
    prob_infect_tot = prob_infect.sum()

    # E(k)-> E(k+1)/R
    if(random<prob_heal1_tot):
        for k in range(k_lat):
            if(random<prob_heal1[:k+1].sum()):
                S[t,:-1]   = S[t-1,:-1]
                I[t,:-1]   = I[t-1,:-1]
                E[t,k,0]   = -1
                E[t,k+1,0] = 1
                R[t]       = R[t-1] + E[t,k_lat,0]
                E[t,0,1]   = E[t,0,0]
                E[t]      += E[t-1]
                break

    # I(k)-> I(k+1)/R
    elif(random<(prob_heal1_tot+prob_heal2_tot)):
        random -= prob_heal1_tot
        for k in range(k_rec):
            if(random<prob_heal2[:k+1].sum()):
                S[t,:-1] = S[t-1,:-1]
                E[t,:-1] = E[t-1,:-1]
                I[t,k]   = -1
                I[t,k+1] = 1
                R[t]     = R[t-1] + I[t,k_rec]
                I[t]    += I[t-1]
                break

    # E(k)-> E(k+1)/I(0)
    elif(random<(prob_heal1_tot+prob_heal2_tot+prob_latent_tot)):
        random -= (prob_heal1_tot+prob_heal2_tot)
        for k in range(k_lat):
            if(random<prob_latent[:k+1].sum()):
                S[t,:-1]   = S[t-1,:-1]
                I[t,:-1]   = I[t-1,:-1]
                R[t]       = R[t-1]
                E[t,k,1]   = -1
                E[t,k+1,1] = 1
                I[t,0]    += E[t,k_lat,1]
                E[t,0,0]   = E[t,0,1]
                E[t]      += E[t-1]
                break

    # S(k)-> S(k+1)/E(0)
    else:
        random -= (prob_heal1_tot+prob_heal2_tot+prob_latent_tot)
        for k in range(k_inf):
            if(random<prob_infect[:k+1].sum()):
                E[t,:-1] = E[t-1,:-1]
                I[t,:-1] = I[t-1,:-1]
                R[t]     = R[t-1]
                S[t,k]   = -1
                S[t,k+1] = 1
                E[t,0]  += S[t,k_inf]
                S[t]    += S[t-1]
                break
# -------------------------

def plotting(infected_time_series,I_day,day_max,I_95):
    spacing = 100
    #plt.plot(T[:t:spacing],S[:t:spacing,:-1].sum(1),label='S',c='r')
    #plt.plot(T[:t:spacing],E[:t:spacing,:-1,0].sum(1),label='E_rec',c='g')
    #plt.plot(T[:t:spacing],E[:t:spacing,:-1,1].sum(1),label='E_inf',c='b')
    #plt.plot(T[:t:spacing],I[:t:spacing,:-1].sum(1),label='I',c='c')
    #plt.plot(T[:t:spacing],R[:t:spacing],label='R',c='m')
    #plt.legend()
    plt.show()

    #S_m = S_day.mean(0)
    #E_m = E_day.mean(0)
    I_m = I_day.mean(0)
    #R_m = R_day.mean(0)
    #S_std = S_day.std(0)
    #E_std = E_day.std(0)
    I_std = I_day.std(0)
    #R_std = R_day.std(0)

    #out_file = open("seir.dat","w")
    #for day in range(day_max):
        #out_file.write("%s\n" % S_m[day])
    #out_file.close()

    #plt.errorbar(np.arange(day_max),S_m[:day_max],yerr=S_std[:day_max])
    #plt.errorbar(np.arange(day_max),E_m[:day_max],yerr=E_std[:day_max])
    plt.errorbar(np.arange(day_max),I_m[:day_max],yerr=I_std[:day_max],marker='o',ls='',label='I mean')
    #plt.errorbar(np.arange(day_max),R_m[:day_max],yerr=R_std[:day_max])

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



