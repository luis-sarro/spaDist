import sys
import math
import numpy as np
from numba import jit
from scipy.stats import norm
from scipy.stats import poisson
import pandas as pd

D2R   = np.pi/180.

@jit
def Support(params):
    rc = params[3]
    rt = params[4]
    if rc <= 0.0 : return False
    if rt <= rc : return False
    return True

@jit
def CDFv(r,rc,rt):
    w = r**2 + rc**2
    x = 1 + (rt/rc)**2
    y = 1 + (r/rc)**2
    z = rc**2 + rt**2
    a = (r**2)/z +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) -2*np.log(rc)
    b = -4 + (rt**2)/z + 4*rc/np.sqrt(z)         + np.log(z) -2*np.log(rc)
    NK  = a/b
    result = np.zeros(len(r))
    idBad  = np.where(r>rt)[0]
    idOK   = np.where(r<=rt)[0]
    result[idOK] = NK[idOK]
    result[idBad] = 1
    return result

@jit
def CDFs(r,rc,rt):
    if r > rt:
        return 1
    w = r**2 + rc**2
    x = 1 + (rt/rc)**2
    y = 1 + (r/rc)**2
    z = rc**2 + rt**2
    a = (r**2)/z +  4*(rc-np.sqrt(w))/np.sqrt(z) + np.log(w) -2*np.log(rc)
    b = -4 + (rt**2)/z + 4*rc/np.sqrt(z)         + np.log(z) -2*np.log(rc)
    NK  = a/b
    return NK

@jit
def Number(r,params,Rmax):
    return (CDFv(r,params[3],params[4])/CDFs(Rmax,params[3],params[4]))

def Density(r,params,Rmax):
    return np.piecewise(r,[r>params[4],r<=params[4]],[0.0,
        lambda x: King(x,params[3],params[4])/CDFs(Rmax,params[3],params[4])])

@jit
def King(r,rc,rt):
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    z = rc**2 + rt**2
    k   = 2*((x**(-0.5))-(y**-0.5))**2
    norm= (rc**2)*(-4 + (rt**2)/z + 4*rc/np.sqrt(z) + np.log(z) -2*np.log(rc))
    lik = k/norm
    return lik

@jit
def LikePA(cdf,theta,sg):
    return norm.pdf(cdf,loc=theta/(2.*np.pi),scale=sg)+ 1e-100

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

@jit
def LogLikeStar(x,params,Rmax,sg):
    return np.log(x[0]*(x[1]*Density(x[1],params,Rmax)) + (1.-x[0])*LikeField(x[1],Rmax))

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
                np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist + 1e-20
        theta = np.arctan2(np.sin((self.cdts[:,0]-cntr[0])*D2R),
                         np.cos(cntr[1]*D2R)*np.tan(self.cdts[:,1]*D2R)-
                         np.sin(cntr[1]*D2R)*np.cos((self.cdts[:,0]-cntr[0])*D2R))
        # print(min(theta),max(theta))
        theta   = (theta + 2*np.pi)%(2*np.pi)
        
        local_params = [params[0],params[1],params[2],params[3],params[4],params[5]]
        mock_params = np.tile(local_params,(len(radii),1))
    
        rcThetaJ = params[3] + params[5] * (self.cdts[:,3]-13.5)
        tmp1 = np.isnan(rcThetaJ)
        rcThetaJ[tmp1]=params[5]
        mock_params[:,3] = rcThetaJ
#        for i in range(len(tmp1)):
#            print(mock_params[i])
        
        data  = np.vstack([self.cdts[:,2],radii,theta]).T
        quadrants = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        quarter = pd.cut(theta,bins=quadrants,include_lowest=True)
        counts = pd.value_counts(quarter)
        Like = poisson.pmf(counts,len(theta)/4.0)
        totLike = reduce(lambda x, y: x*y, Like)
        
        #------ Computes Likelihood -----------
        llike = np.sum(map(lambda x,thetaparams:LogLikeStar(x,thetaparams,self.Rmax,self.sg),data,mock_params))+np.log(totLike)
        if (np.isnan(llike)): llike = -9999999.99
        return llike







