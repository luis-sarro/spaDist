import sys
import numpy as np
from numba import jit
from scipy.stats import norm
from scipy.stats import poisson
import pandas as pd
from scipy.special import gamma
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from functools import reduce

cntr  = [56.65,24.13]
D2R   = np.pi/180.
a     = 1e-5

@jit
def Support(params):
    b  = params[2]
    rc = params[4]
    rt = params[5]
    rtb = params[6]
    if rc <= 0.0 : return False
    if rt <= rc : return False
    if b <= 0.0 : return False
    if rtb <= 0 : return False
    if rtb <= b : return False    
    return True

def rho(r,params):
    rc = params[4]
    rt = params[5]
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
    return np.piecewise(r,[r>params[5],r<=params[5]],[0.0,lambda x: rho(x,params)])

@jit
def logLikeStar(x,params,Rmax,cte):
    return np.log(x[0]*(LikeRadious(x[1],params)/cte) + (1.-x[0])*LikeField(x[1],Rmax))

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
        #============== Obtains radii and pa ===================

        a0, d0, b, delta, a, rta, rtb = params[0],params[1],params[2],params[3],params[4], params[5], params[6]
        local_params = [params[0],params[1],params[2],params[3],params[4],params[5],params[6]]

        x     = ((self.cdts[:,0]-(cntr[0]+a0))*D2R)*self.Dist
        y     = ((self.cdts[:,1]-(cntr[1]+d0))*D2R)*self.Dist
        xn = x*np.sin(delta) + y*np.cos(delta)
        yn = x*np.cos(delta) - y*np.sin(delta)
        r     = np.sqrt(xn**2 + yn**2)
        theta   = np.arctan2(xn,yn)
        theta   = (theta + 2*np.pi)%(2*np.pi)
        rcTheta   = (params[4]*params[2])/np.sqrt((params[2]*np.cos(theta))**2+(params[4]*np.sin(theta))**2)
        rtTheta   = (params[5]*params[6])/np.sqrt((params[6]*np.cos(theta))**2+(params[5]*np.sin(theta))**2)
        # That is: a*b/sqrt((b*cos(theta))^2+(a*sin(theta))^2)
        mock_params = np.tile(local_params,(len(r),1))
        mock_params[:,4] = rcTheta
        #mock_params[:,5] = rtTheta
        #mock_params[:,4] = params[2]
        mock_params[:,5] = params[6]

        data  = np.vstack([self.cdts[:,2],r,theta]).T

        quadrants = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        quarter = pd.cut(theta,bins=quadrants,include_lowest=True)
        counts = pd.value_counts(quarter)
        Like = poisson.pmf(counts,len(theta)/4.0)
        totLike = reduce(lambda x, y: x*y, Like)

        cte = integrate.quad(lambda x:LikeRadious(x,mock_params[0,:]),10E-5,self.Rmax,epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0]
#        print(cte,totLike)
        
        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda x,thetaparams:logLikeStar(x,thetaparams,self.Rmax,cte),data,mock_params))+np.log(totLike)

#        np.set_printoptions(threshold=np.nan)
#        print(mock_params[0,0],mock_params[0,1],mock_params[0,4],mock_params[0,5],params[4],params[5],llike,np.sum(r))

        return llike



