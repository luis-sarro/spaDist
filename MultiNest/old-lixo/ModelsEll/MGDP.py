import sys
import numpy as np
from numba import jit
from scipy.stats import norm
from scipy.special import hyp2f1
import scipy.integrate as integrate
from scipy.stats import poisson
import pandas as pd

D2R   = np.pi/180.
cntr  = [56.65,24.13]

@jit
def Support(params):
    if params[3] <= 0 : return False
    if params[4] <= 0 : return False
    # if params[3] >= 3 : return False
    return True

@jit
def Density(r,params,Rmax):
    rc = params[4]
    a  = params[5]
    b  = params[6]
    v1 = (1.0 + (r/rc)**(1./a))**(-a*b) 
    w  = 2.*a
    x  = a*b
    y  = 1. + 2.*a
    z  = -((rc/Rmax)**(-1./a))
    v2 = (Rmax**2)*hyp2f1(w,x,y,z)
    return 2*v1/v2


def Number(r,params,Rmax):
    Num = np.array(map(lambda y: integrate.quad(lambda x:Density(x,params,Rmax)*x,1e-5,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000),r))
    return Num

@jit
def LogLikeStar(x,params,Rmax):
    return np.log(x[0]*(x[1]*Density(x[1],params,Rmax)) + (1.-x[0])*LikeField(x[1],Rmax))

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

        a0, d0, b, delta, a, expa,expb = params[0],params[1],params[2],params[3],params[4], params[5], params[6]
        local_params = [params[0],params[1],params[2],params[3],params[4],params[5],params[6]]

        #============== Obtains radii and pa ===================


        x     = ((self.cdts[:,0]-(cntr[0]+a0))*D2R)*self.Dist
        y     = ((self.cdts[:,1]-(cntr[1]+d0))*D2R)*self.Dist

        xn = x*np.sin(delta) + y*np.cos(delta)
        yn = x*np.cos(delta) - y*np.sin(delta)
        r     = np.sqrt(xn**2 + yn**2)
        theta   = np.arctan2(xn,yn)
        theta   = (theta + 2*np.pi)%(2*np.pi)

        rcTheta   = (params[4]*params[2])/np.sqrt((params[2]*np.cos(theta))**2+(params[4]*np.sin(theta))**2)

        # That is: a*b/sqrt((b*cos(theta))^2+(a*sin(theta))^2)
        mock_params = np.tile(local_params,(len(x),1))

        mock_params[:,4] = rcTheta
        data  = np.vstack([self.cdts[:,2],r,theta]).T

        quadrants = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        quarter = pd.cut(theta,bins=quadrants,include_lowest=True)
        counts = pd.value_counts(quarter)
        Like = poisson.pmf(counts,len(theta)/4.0)
        totLike = reduce(lambda x, y: x*y, Like)
        
        #------ Computes Likelihood -----------
        llike = np.sum(map(lambda x:LogLikeStar(x,params,self.Rmax),data))+np.log(totLike)
        # print(llike)
        return llike



