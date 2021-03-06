import sys
import numpy as np
from numba import jit
from scipy.stats import dirichlet
from scipy.stats import halfcauchy

@jit
def Support(rc):
    if rc <= 0 : return -np.inf

@jit
def Number(r,rc,Nstr,Rmax):
    return Nstr*(CDF(r,rc)/CDF(Rmax,rc))

@jit
def CDF(r,rc): 
    return r**2/(rc**2+r**2)

@jit
def logPriors(rc,hyper):
    lp_rc = halfcauchy.logpdf(rc,loc=0,scale=hyper[0])
    return lp_rc


@jit
def logLikeStar(r,p,rc,Rmax):
    return np.log(p*LikeCluster(r,rc,Rmax))


@jit
def LikeCluster(r,rc,Rmax):
    x  = 1+(r/rc)**2
    z  = 2/(rc*x)**2
    return z/CDF(Rmax,rc)

@jit
def LikeField(r,rm):
    return 2/rm**2


def LogPosterior(params,r,pro,Rmax,hyper):
    rc  = params[0]
    #----- Checks if parameters' values are in the ranges
    supp = Support(rc)
    if supp == -np.inf : 
        return -np.inf

    # ----- Computes Priors ---------
    lprior = logPriors(rc,hyper)
    # ----- Computes Likelihoods ---------
    llike  = map(lambda x,y:logLikeStar(x,y,rc,Rmax),r,pro)

    lpos   = np.sum(llike)+lprior
    # print lpos
    return lpos



