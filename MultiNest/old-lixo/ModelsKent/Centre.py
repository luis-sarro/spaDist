import sys
import numpy as np
from numba import jit
from kent import *

D2R   = np.pi/180.
cntr  = [56.65,24.13]
pi    = np.pi

@jit
def Support(params):
    if params[0] <= -pi/2.0 or params[0] > pi/2.0          : return False
    if params[1] <= -pi     or params[1] > pi              : return False
    if params[2] <= -pi/2.0 or params[2] > pi/2.0          : return False
    if params[3] <= 0.0     or params[3] > 1e6             : return False
    if params[4] <= 0.0     or params[4] > 1.0             : return False
    return True

@jit
def Number(r,params,Rmax):
    return np.ones_like(r)

@jit
def Density(r,params,Rmax):
    return np.ones_like(r)
@jit
def logrho(x,G,kappa,beta):
    g1x  = G[:,0].dot(x)
    g2x  = G[:,1].dot(x)
    g3x  = G[:,2].dot(x)
    f    = kappa*(g1x + 0.5*beta*(g2x**2 - g3x**2))
    return f

@jit
def logLikeStar(x,p,G,kappa,beta,cte):
    """
    Uses log sum exp to avoid overflow
    """
    A = np.log(p)+logrho(x,G,kappa,beta)-cte
    B = np.log(1.0-p) - np.log(4.0*np.pi)
    return np.logaddexp(A,B)



class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,Rmax,trans_lim,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.cdts       = cdts
        self.Rmax       = Rmax
        self.t          = trans_lim
        self.Dist       = Dist
        self.N          = float(len(cdts))
        self.pro        = cdts[:,2]

        ###### Transform cooridnates into x,y,z in the sphere. Sytem of heaton et al 2013 envirometrics: env2251
        # alpha = phi, -pi<phi<pi
        # delta = theta, -0.5pi<delta<0.5pi
        x     = np.cos(self.cdts[:,1]*D2R)*np.cos(self.cdts[:,0]*D2R)
        y     = np.cos(self.cdts[:,1]*D2R)*np.sin(self.cdts[:,0]*D2R)
        z     = np.sin(self.cdts[:,1]*D2R)
        self.data  = np.stack((x,y,z),axis=-1)
        
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #---------- Uniform Priors ------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50
        #============== Obtains radii and pa ===================
        kent  = KentDistribution(params[0], params[1], params[2],params[3],params[4])
        G,k,b = Gamma(params[0],params[1],params[2]),params[3],params[4]
        
        #------ Computes Likelihood -----------
        cte        = kent.log_normconst(k,b)
        llike      = np.sum(map(lambda x,p:logLikeStar(x,p,G,k,b,cte),self.data,self.pro))
        # print(llike)
        return llike



