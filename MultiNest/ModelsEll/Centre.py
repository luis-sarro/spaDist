import sys
import numpy as np
from numba import jit
from scipy.stats import norm
from scipy.special import ellipk

D2R   = np.pi/180.
cntr  = [56.65,24.13]

@jit
def Support(params):
    rc = params[4]
    if rc <= 0.0 : return False
    if params[2] < 0.0 : return False
    if params[2] > 1.0 : return False
    return True

@jit
def Number(r,params,Rmax):
    return CDF(r,params)#/CDF(Rmax,rc))

@jit
def Density(r,params,Rmax):
    rc = params[2]
    x = (r/rc)**2
    z = np.exp(-x/2)/(rc**2)
    return z/CDF(Rmax,rc)

@jit
def CDF(r,params):
    e = params[2]
    rc = 1.0
    # x  = (r/rc)**2
    # frac = (1-np.exp(-x/2))
    return 2*np.pi*rc*(1-e)

@jit
def LikeRadius(r,params,Rmax):
    rc =1.0
    # x = (r/rc)**2
    # z = np.exp(-x/2)/(rc**2)
    z = np.exp(-r/rc)
    return z/CDF(Rmax,params)

@jit
def logLikeStar(x,params,Rmax):#*LikePA(x[2],x[3],0.01)
    return np.log(x[0]*(x[1]*LikeRadius(x[1],params,Rmax)) + (1.-x[0])*LikeField(x[1],Rmax))


@jit
def LikePA(cdf,theta,sg):
    return norm.pdf(cdf,loc=(theta+np.pi)/(2.*np.pi),scale=sg)+ 1e-100

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

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
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #---------- Uniform Priors ------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50

        a0,d0,epsi,delta   = params[0],params[1],params[2],params[3]
        a0=0.0
        d0=0.0
        #============== Obtains radii and pa ===================
        # Equation 2 from Kacharov 2014.
        # x     = np.sin((self.cdts[:,0]-(cntr[0]-a0))*D2R)*np.cos(self.cdts[:,1]*D2R)*self.Dist
        # y     = (np.cos((cntr[1]-d0)*D2R)*np.sin(self.cdts[:,1]*D2R)-
        #         np.sin((cntr[1]-d0)*D2R)*np.cos(self.cdts[:,1]*D2R)*np.cos((self.cdts[:,0]-(cntr[0]-a0))*D2R))*self.Dist
        
        x     = ((self.cdts[:,0]-(cntr[0]+a0))*D2R)*self.Dist
        y     = ((self.cdts[:,1]-(cntr[1]+d0))*D2R)*self.Dist
        xn    = (x*np.sin(delta) + y*np.cos(delta))
        yn    = (x*np.cos(delta) - y*np.sin(delta))
        theta = np.arctan2(yn,xn)
        radii = np.sqrt(xn**2 + (yn/(1-epsi))**2)

        # radii = np.sqrt(x_new**2 + y_new**2)*(1.0-epsi)
        # theta = np.arctan2(np.sin((self.cdts[:,0]-(cntr[0]-a0))*D2R),
        #                  np.cos((cntr[1]-d0)*D2R)*np.tan(self.cdts[:,1]*D2R)-
        #                  np.sin((cntr[1]-d0)*D2R)*np.cos((self.cdts[:,0]-(cntr[0]-a0))*D2R))
        # 
        # theta = (theta + 2*np.pi)%(2*np.pi)
        idx   = np.arange(len(theta))#np.argsort(theta)
        cdf   = (1./len(theta))*np.arange(len(theta))
        data  = np.vstack([self.cdts[:,2][idx],radii[idx],cdf,theta[idx]]).T

        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda x:logLikeStar(x,params,self.Rmax),data))
        # print(llike)
        return llike



