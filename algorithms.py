import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

#  Stochastic Bandits 

# ETC
# a: the arm we choose to pull
# mu2: the true value of the mean for arm 2
# a: the arm we choose to pull
# mu2: the true value of the mean for arm 2
def pullGaussian1(a,mu2):
        if a == 1:
            return np.random.normal(0,1)
        return np.random.normal(mu2,1)
    
def pullBernoulli1(a,p):
        if a == 1:
            p = 0.5
            return np.random.binomial(1,p)
        return np.random.binomial(1,p)

# m is the number of times we explore arm a
# n is the horizons or the number of times we play
# mu2 is the mean of arm bandit 2
def ETC(m,n,mu2,comment=False,gaussian=1):
    arm_means = [0,0]
    true_mean = [0,mu2]
    arm_pulls = [0,0]
    if gaussian == 1:
        mu1 = 0
        optimal = mu2 if mu2 > mu1 else mu1
    else:
        mu1 = 0.5
        optimal = mu2 if mu2 > mu1 else mu1
    
    # exploration phase
    exploration_regret = (optimal - mu2)*m + (optimal - mu1)*m
    
    if gaussian == 1:
        reward_1 = [pullGaussian1(1,mu2) for a in range(m)]
        reward_2 = [pullGaussian1(2,mu2) for a in range(m)]
    else:
        reward_1 = [pullBernoulli1(1,mu2) for a in range(m)]
        reward_2 = [pullBernoulli1(2,mu2) for a in range(m)]
    empirical_mean_1 = np.mean(reward_1)
    empirical_mean_2 = np.mean(reward_2)
    # exploitation phase
    best_mean = mu2 if empirical_mean_1 < empirical_mean_2 else mu1
    best_arm = 1 if empirical_mean_1 < empirical_mean_2 else 0
    if comment:
        print("arm1 mean:" + str(mu1))
        print("arm2 mean:" + str(mu2))
        print("best arm:" + str(best_arm))
        print("optimal arm:" + str(optimal))
    
    #reward_exploit = [pullGaussian(best_arm,mu2) for i in range(n - 2*m)]
    #reward = pullGaussian(best_arm,mu2)
    exploitation_regret = (optimal - best_mean)*(n-2*m)
    
        
    total_regret = exploitation_regret + exploration_regret
    
    if comment:
        print("exploration regret:" + str(exploration_regret))
        print("exploitation regret:" + str(exploitation_regret))
        print("total regret:" + str(total_regret))
        print("best arm true mean:" + str(true_mean[best_arm-1]))
        print("\n")
    
    return total_regret

# simulating ETC's N parameter
# Function takes in lists for m, n (horizon), and mu2 and
# performs a grid search of each set of parameters.
# Will ignore permutations where n <= m.
def simulationN_ETC(mu2,m,n=1000,num_sim=1000,gaussian=1):
    df = pd.DataFrame()
    df['mu2'] = mu2
    det = [determine_m(a) for a in mu2]
    for j in tqdm(m):            
        point_lst = []
        err_lst = []
        for i in tqdm(mu2):
            if j != 1000:
                simulation = [ETC(m=j,n=n,mu2=i,gaussian=gaussian) for a in range(num_sim)]
            else:
                simulation = [ETC(m=determine_m(i),n=n,mu2=i,gaussian=gaussian) for a in range(num_sim)]
            point = np.mean(simulation)
            err = np.var(simulation)
            point_lst += [point]
            err_lst += [err]
        
        df[str(j) + "point"] = point_lst
        df[str(j) + "error"] = err_lst
    return df

def determine_m(mu2):
    return int(max(1,np.ceil(4*np.log(250*mu2**2)/mu2**2)))


# UCB

def pullGaussian(mu):
    return np.random.normal(mu,1)
    
def pullBernoulli(p):
    return np.random.binomial(1,p)

def simulationN_standard(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_standard(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def simulationN_asymptotic(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_asymptotic(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def simulationN_moss(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_moss(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def simulationN_KL(mu2,n=1000,num_sim=1000,gaussian=True):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for m in tqdm(mu2):
        simulation = [UCB_KL(n=n,mu2=m,gaussian=1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

# UCB Algorithms

def UCB_standard(n,mu2,gaussian=1):
    if mu2 == 0:
        return 0
    reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0 else 1
    
    while(t < n):
        ucb = [np.mean(rewards1) + np.sqrt(2*np.log(n**2)/ti[0]),np.mean(rewards2) + np.sqrt(2*np.log(n**2)/ti[1])]
        argmax = np.argmax(ucb)
        reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else mu2
    
    return regret

def UCB_asymptotic(n,mu2,gaussian=1):
    if mu2 == 0:
        return 0
    reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0 else 1
    
    while(t < n):
        ft = np.log(1 + t*np.log(np.log(t)))
        ucb = [np.mean(rewards1) + np.sqrt(2*ft/ti[0]),np.mean(rewards2) + np.sqrt(2*ft/ti[1])]
        argmax = np.argmax(ucb)
        reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else mu2
    
    return regret

def log_plus(n,t):
    x = n / (2  *t)
    return max(np.log(1),np.log(x))

def UCB_moss(n,mu2,gaussian=1):
    if mu2 == 0: 
        return 0
    reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0 else 1
    
    while(t < n):
        if t != 1:
            ft = 1 + t*np.log(t)*np.log(t)
            
        ucb = [np.mean(rewards1) + np.sqrt((4/ti[0])*log_plus(n,ti[0])),np.mean(rewards2) + np.sqrt((4/ti[1])*log_plus(n,ti[1]))]
        argmax = np.argmax(ucb)
        reward = [pullGaussian(0),pullGaussian(mu2)] if gaussian else [pullBernoulli(0),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else mu2
    
    return regret

def d(p,q):
    if (p == 0):
        if (q < 1 and q > 0):
            return np.log(1/(1-q))
        else:
            return 0
    if (p == 1):
        if (q < 1 and q > 0):
            return np.log(1/q)
        else:
            return 1
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def calculate_ucb(p,t,ti):
    ft = 1 + t*(np.log(np.log(t)))
    upper_bound = np.log(ft) / ti
    bounds = [p,1]
    for i in range(10):
       
        half = (sum(bounds)) / 2
        if bounds[1]-bounds[0] < 1e-5:
             #early stopping
            break
        
        entropy = d(p,half)

        if entropy < upper_bound:
            bounds[0] = half
        else:
            bounds[1] = half

    return half

def UCB_KL(n,mu2,gaussian=1):
    if mu2 == 0.5:
        return 0
    reward = [pullBernoulli(0.5),pullBernoulli(mu2)]
    rewards1 = [reward[0]]
    rewards2 = [reward[1]]
    ti = [1,1]
    t = 2
    regret = 0
    optimal = 0 if mu2 < 0.5 else 1
    
    while(t < n): 
        #ucb = calculate_ucb(np.array([np.mean(rewards1),np.mean(rewards2)]),t,ti[0])
        ucb = [calculate_ucb(np.mean(rewards1),t,ti[0]),calculate_ucb(np.mean(rewards2),t,ti[1])]
        argmax = np.argmax(ucb)
        
        reward = [pullBernoulli(0.5),pullBernoulli(mu2)]
        
        if argmax == 0:
            rewards1 += [reward[0]]
        else:
            rewards2 += [reward[1]]
        ti[argmax] += 1
        t+=1
        regret += 0 if optimal == argmax else abs(mu2-0.5)
    
    return regret

# Thompson Sampling

def simulationN_TS1(mu2,p1,p2,n=1000,num_sim=1000):
    point_lst = []
    var_lst = []
    for i in tqdm(mu2):
        simulation = [thompson_sampling_gaussian(n=n,mu2=i,p1=p1,p2=p2) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df = pd.DataFrame()
    df['mu2'] = mu2
    df['regret'] = point_lst
    df['var'] = var_lst
    return df

def simulation_TS2(mu2,p1,p2,n=1000,num_sim=1000):
    point_lst = []
    var_lst = []
    for i in tqdm(mu2):
        simulation = [thompson_sampling_bernoulli(n=n,mu2=i,p1=p1,p2=p2) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df = pd.DataFrame()
    df['mu2'] = mu2
    df['regret'] = point_lst
    df['var'] = var_lst
    return df

def calculate_regret(n,opt_mu,mus):
    rn = n*opt_mu - sum(mus)
    
def calculate_posterior_value_gaussian(x,sigma,mup,sigmap,n):
    # x is the observed sample mean
    # sigma is the signal std
    # mup is the mean of the prior
    # sigmap is the std of the prior
    # n is the number of simulated values to get
    mean = ((mup / (sigmap)**2) + (np.mean(x)/(sigma)**2)) / ((1/(sigmap)**2) + (1/(sigma)**2))
    std = ((1/(sigmap)**2) + (1/(sigma)**2))**(-1)
    #std = 1/((1/(sigmap)**2) + n)
    #mean = std*sum(x)
    return [mean,std**(0.5)]
    
def thompson_sampling_gaussian(n,mu2,p1,p2):
    # n is the horizon (number of iterations)
    # mu2 is the true mean of bandit 2
    # p1 is the prior distribution for bandit 1
    # p2 is the prior distriubtion for bandit 2
    t = 1
    regret = 0
    rewards1 = []
    rewards2 = []
    visits = [0,0]
    dist1 = p1
    dist2 = p2
    if mu2 > 0: 
        opt_mu = mu2
    else:
        opt_mu = 0
    
    while(t < n):
        
        # Sampling v_{t}
        sim1 = np.random.normal(dist1[0],dist1[1])
        sim2 = np.random.normal(dist2[0],dist2[1])

        # Choosing At
        arm = np.argmax([sim1,sim2])
        
        # pull reward and update distribution
        if arm == 0: # Choose arm 1
            reward = pullGaussian(0)
            rewards1 += [reward]
            regret += abs((opt_mu - 0))
            visits[0] += 1
            dist1 = calculate_posterior_value_gaussian(rewards1,1,dist1[0],dist1[1],visits[0])
        else: # Choose arm 2
            reward = pullGaussian(mu2)
            rewards2 += [reward]
            regret += abs(opt_mu-mu2)
            visits[1] += 1
            dist2 = calculate_posterior_value_gaussian(rewards2,1,dist2[0],dist2[1],visits[1])
        t += 1
    return regret

def calculate_posterior_value_bernoulli(x,alpha,beta):
    # x is the observed sample mean
    # sigma is the signal std
    # mup is the mean of the prior
    # sigmap is the std of the prior
    # n is the number of simulated values to get
    return [alpha+np.mean(x),beta+1-np.mean(x)]
    
def thompson_sampling_bernoulli(n,mu2,p1,p2):
    # n is the horizon (number of iterations)
    # mu2 is the true mean of bandit 2
    # p1 is the prior distribution for bandit 1
    # p2 is the prior distriubtion for bandit 2
    t = 1
    regret = 0
    dist1 = p1
    dist2 = p2
    rewards1 = []
    rewards2 = []
    visits = [0,0]
    if mu2 > 0.5: 
        opt_mu = mu2
    else:
        opt_mu = 0.5
    
    while(t < n):
        
        # Sampling v_{t}
        sim1 = np.random.beta(dist1[0],dist1[1])
        sim2 = np.random.beta(dist2[0],dist2[1])

        # Choosing At
        arm = np.argmax([sim1,sim2])
        
        
        # pull reward and update distribution
        if arm == 0: # Choose arm 1
            reward = pullBernoulli(dist1[0] / (dist1[0] + dist1[1]))
            regret += abs((opt_mu - 0.5))
            rewards1 += [reward]
            dist1 = calculate_posterior_value_bernoulli(rewards1,dist1[0],dist1[1])
        else: # Choose arm 2
            reward = pullBernoulli(dist2[0] / (dist2[0] + dist2[1]))
            regret += abs(opt_mu-mu2)
            rewards2 += [reward]
            dist1 = calculate_posterior_value_bernoulli(rewards2,dist2[0],dist2[1])
        t += 1
    return regret

# Linear Bandits

def UCB_Linear(a,n,theta,lamb=0.1):
    delta = 1/n
    mu_hat = 0
    theta_hat = 0
    ti = [0,0]
    rewards1 = []
    rewards2 = []
    t = 1
    regret = 0
    V = lamb
    at = [a[0]*theta,a[1]*theta]
    optimal = a[np.argmax(at)]
    while(t < n): 
        beta = np.sqrt(lamb) + np.sqrt(2*np.log(1/delta) + np.log((1+(t-1))/delta))
        a_hat = [0,0]
        for i in range(2):
            a_hat[i] = a[i]*theta_hat + beta*np.sqrt(a[i]**(2)/lamb)
        argmax = np.argmax(a_hat)
        reward = theta*a[argmax] + np.random.normal(0,1)
        V = V + a[argmax]**2
        
        mu_hat = mu_hat + reward*a[argmax]
        theta_hat = mu_hat / V
        ti[argmax] += 1
        t+=1
        regret_t = 0 if a[argmax] == optimal else (optimal-a[argmax])*theta
        regret+= regret_t
    
    return regret

def simulationN_LinUCB(a,theta,n=1000,num_sim=1000):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for t in tqdm(theta):
        simulation = [UCB_Linear(a,n,t) for b in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def simulationN_TSLin(a,theta,n=1000,num_sim=1000):
    point_lst = []
    var_lst = []
    df = pd.DataFrame()
    for t in tqdm(theta):
        simulation = [thompsonSampling_linear(a,n,t) for b in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df['point'] = point_lst
    df['var'] = var_lst
    return df

def thompsonSampling_linear(a,n,theta):
    mu_t = 0
    sigma_t = 1
    regret = 0
    at = [a[0]*theta,a[1]*theta]
    optimal = a[np.argmax(at)]
    choose = [0,0]
    
    for i in range(n):
        theta_hat = np.random.normal(mu_t, sigma_t)
        X_t = [a[0]*theta_hat,a[1]*theta_hat]
        action = np.argmax(X_t)
        regret_t = 0 if a[action] == optimal else (a[action]-optimal)*theta
        regret += regret_t
        choose[action]+=1
        
        reward = theta*a[action] + np.random.normal(0,1) # noise
        sigma_t1 = sigma_t / ((X_t[action]**(2))*sigma_t + 1)
        mu_t1 = sigma_t1*(mu_t/sigma_t + X_t[action]*reward)
        mu_t = mu_t1
        sigma_t = sigma_t1
    return regret

def calculate_posterior_value_bernoulli(x,alpha,beta):
    # x is the observed sample mean
    # sigma is the signal std
    # mup is the mean of the prior
    # sigmap is the std of the prior
    # n is the number of simulated values to get
    return [alpha+np.mean(x),beta+1-np.mean(x)]

def psq(alpha,beta,s,q):
    return (alpha+s) / (alpha + beta + q)

def wt(alpha,beta,s,q,t,n):
    if t == n:
        return psq(alpha,beta,s,q)
        
    return psq(alpha,beta,s,q)*(1+wt(alpha,beta,s+1,q+1,t+1,n)) + (1-psq(alpha,beta,s,q))*wt(alpha,beta,s,q+1,t+1,n)

def wt2(t,n):
    if t == n:
        return 0.5
    return 0.5+wt2(t+1,n)

# Bayesian optimal Polciy
def bayesian_optimal_policy(n,mu1,p1):
    # n is the horizon (number of iterations)
    # mu2 is the true mean of bandit 2
    # p1 is the prior distribution for bandit 1
    # p2 is the prior distriubtion for bandit 2
    wt1 = [0]
    rewards1 = []
    regret = 0
    rewards2 = []
    visits = [0,0]
    dist1 = p1
    opt_mu = mu1 if mu1 > 0.5 else 0.5
    for t in range(n,0,-1): 
        w1 = 0
        w2 = 0
        for i in range(t):
            w1 = wt(dist1[0],dist1[1],i,i,t,n)
            w2 = wt2(t,n)
        
            arm = np.argmax([w1,w2])

            if arm == 0:
                reward = pullBernoulli(mu1)
                rewards1 += [reward]
                regret += abs((opt_mu-mu1))
                visits[0] += 1
                dist1 = calculate_posterior_value_bernoulli(rewards1,dist1[0],dist1[1])

            else:
                reward = 0.5
                rewards2 += [reward]
                regret += abs(opt_mu-0.5)
                visits[1] += 1
    
        
    return regret

def simulationN_BOP(mu1,p1,n=20,num_sim=1000):
    point_lst = []
    var_lst = []
    for i in tqdm(mu1):
        simulation = [bayesian_optimal_policy(n=n,mu1=i,p1=p1) for a in range(num_sim)]
        point = np.mean(simulation)
        var = np.var(simulation)
        point_lst += [point]
        var_lst += [var]
    df = pd.DataFrame()
    df['mu1'] = mu1
    df['regret'] = point_lst
    df['var'] = var_lst
    return df

