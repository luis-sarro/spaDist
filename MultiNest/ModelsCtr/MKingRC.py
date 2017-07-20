import sys
import numpy as np
from numba import jit
from scipy.stats import norm
from scipy.stats import poisson
import pandas as pd

from scipy.special import gamma
import scipy.integrate as integrate
from scipy.interpolate import interp1d

D2R   = np.pi/180.
a     = 1e-5

@jit
def Support(params):
    rc  = params[3]
    rt  = params[4]
    if rc <= 0.0 : return False
    if rt <= rc : return False
    return True

def rho(r,params):
    rc = params[3]
    rt = params[4]
    a  = 0.4
    b  = 1.2
    x  = (1.0 +  (r/rc)**(1./a))**-a
    y  = (1.0 + (rt/rc)**(1./a))**-a
    return r*(x-y)**b


def Number(r,params,Rmax):
    cte = integrate.quad(lambda x:LikeRadious(x,params),a,Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    Num = np.vectorize(lambda y: integrate.quad(lambda x:LikeRadious(x,params)/cte,a,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Num(r)


def Density(r,params,Rmax):
    cte = integrate.quad(lambda x:LikeRadious(x,params),a,Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
    Den = np.vectorize(lambda x:LikeRadious(x,params)/(x*cte))
    return Den(r)

def LikeRadious(r,params):
    return np.piecewise(r,[r>params[4],r<=params[4]],[0.0,lambda x: rho(x,params)])

@jit
def LikePA(cdf,theta,sg):
    return norm.pdf(cdf,loc=theta/(2.*np.pi),scale=sg)+ 1e-100

@jit
def LogLikeStar(x,params,Rmax,sg,cte):
    return np.log(x[0]*(LikeRadious(x[1],params)/cte) + (1.-x[0])*LikeField(x[1],Rmax))

@jit
def LikeField(r,rm):
    return 2.*r/rm**2


class Module:
    """
    Chain for computing the likelihood 
    """
    def __init__(self,cdts,pro,Rmax,trans_lim,Dist):
        """
        Constructor of the logposteriorModule
        """
        self.cdts       = cdts
        self.pro        = pro
        self.Rmax       = Rmax
        self.t          = trans_lim
        self.Dist       = Dist
        self.sg         = 0.01
        self.llike_p    = np.sum(np.log(pro))
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #---------- Uniform Priors ------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):

        cntr  = params[1:3]
        phi   = params[0]

        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50

        #============== Obtains radii and pa ===================

        radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(self.cdts[:,1]*D2R)+
                np.cos(cntr[1]*D2R)*np.cos(self.cdts[:,1]*D2R)*
                np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist + 1e-20 # avoids zeros
        theta = np.arctan2(np.sin((self.cdts[:,0]-cntr[0])*D2R),
                         np.cos(cntr[1]*D2R)*np.tan(self.cdts[:,1]*D2R)-
                         np.sin(cntr[1]*D2R)*np.cos((self.cdts[:,0]-cntr[0])*D2R))

        theta   = (theta + 2*np.pi)%(2*np.pi)
        data  = np.vstack([self.cdts[:,2],radii,theta]).T

        quadrants = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        quarter = pd.cut(theta,bins=quadrants,include_lowest=True)
        counts = pd.value_counts(quarter)
        Like = poisson.pmf(counts,len(theta)/4.0)
        totLike = reduce(lambda x, y: x*y, Like)

        cte = integrate.quad(lambda x:LikeRadious(x,params),a,self.Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]

        #------ Computes Likelihood -----------
        llike = np.sum(map(lambda x:LogLikeStar(x,params,self.Rmax,self.sg,cte),data))+np.log(totLike)
        np.set_printoptions(threshold=np.nan)
        #print(llike)
#        print(datatmp[0:2,[0,1,2]])
        return llike
    



