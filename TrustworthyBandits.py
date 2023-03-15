import numpy as np

def first_best_policy(p0,rho,c,n,r=0,lamb=0):
    """
    Policy for trustworthy recommender system implemented by
    Bayesian Optimal Policy (BOP)

    Args:
        p0: Prior for recieving good news (constant)
        rho: Background learning (constant)
        c: cost of consuming product
        n: Horizon
    """ 
    qi = np.zeros(n+1)
    gi = np.zeros(n+1)
    wi = np.zeros([2,n+1])
    pi = np.zeros(n+1)
    alphai = np.ones(n+1)
    pi[0] = ((1-rho)*p0 )/ ((1-rho)*p0 + 1 - p0)
    for i in range(1,n+1):
        pi[i] = (pi[i-1]*(1-rho*alphai[i-1])) / (pi[i-1]*(1-rho*alphai[i-1])+(1-pi[i-1]))
    
    for i in range(n+1):
        gi[i] = (p0 - pi[i])/(1-pi[i])
        qi[i] = (gi[i] + (1-gi[i])*alphai[i]*pi[i]) / ((gi[i]) + (1-gi[i])*alphai[i])
        if qi[i] < c:
            alphai[i] = 0
        
    for i in range(n-1,-1,-1):
        wi[0][i] = (1-c)*(n-i+1)
        w = (1-c)*pi[i] + (-c)*(1-pi[i]) + rho*pi[i]*wi[0][i+1]-rho*pi[i]*wi[1][i+1]
        #print(qi)
        if w > 0:
            wi[1][i] = w*alphai[i] + wi[1][i+1]
        else:
            alphai[i] = 0
            wi[1][i] = wi[1][i+1]
    return pi,alphai,wi[1][:],qi

def second_best_policy(p0,rho,c,n):
    """
    Policy for trustworthy recommender system implemented by
    Bayesian Optimal Policy (BOP)

    Args:
        p0: Prior for recieving good news (constant)
        rho: Background learning (constant)
        c: cost of consuming product
        n: Horizon
    """ 
    qi = np.zeros(n+1)
    gi = np.zeros(n+1)
    wi = np.zeros([2,n+1])
    pi = np.zeros(n+1)
    alphai = np.zeros(n+1)
    pi[0] = ((1-rho)*p0 )/ ((1-rho)*p0 + 1 - p0)
    for i in range(1,n+1):
        alphai[i-1] = ((1-c)*(p0-pi[i-1]))/((1-p0)*(c-pi[i-1]))
        pi[i] = (pi[i-1]*(1-rho*alphai[i-1])) / (pi[i-1]*(1-rho*alphai[i-1])+(1-pi[i-1]))
    
    for i in range(n+1):
        gi[i] = (p0 - pi[i])/(1-pi[i])
        qi[i] = (gi[i] + (1-gi[i])*alphai[i]*pi[i]) / ((gi[i]) + (1-gi[i])*alphai[i])
        
    for i in range(n-1,-1,-1):
        wi[0][i] = (1-c)*(n-i+1)
        w = (1-c)*pi[i] + (-c)*(1-pi[i]) + rho*pi[i]*wi[0][i+1]-rho*pi[i]*wi[1][i+1]
        #print(qi)
        if w > 0:
            wi[1][i] = w*alphai[i] + wi[1][i+1]
        else:
            wi[1][i] = wi[1][i+1]
    return pi,alphai,wi[1][:],qi