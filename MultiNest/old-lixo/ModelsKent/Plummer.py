import sys
import numpy as np
from numba import jit
from kent import *

D2R   = np.pi/180.
cntr  = [56.65,24.13]
pi    = np.pi

@jit
def Support(params):
    if params[0] <= -pi/2.0 or params[0] > pi/2.0 : return False
    if params[1] <= -pi     or params[1] > pi     : return False
    if params[2] <= -pi/2.0 or params[2] > pi/2.0 : return False
    if params[3] <= 0.0     or params[3] > 1e6    : return False
    if params[4] <= 0.0     or params[4] > 1.0    : return False
    if params[5] <= 0.0                           : return False
    return True

@jit
def CDF(params,r):
    e  = params[4]
    rc = params[5]
    return (r**2)*(rc**2)/((rc**2+r**2))

@jit
def Number(r,params,Rmax):
    return CDF(params,r)/CDF(params,Rmax)
@jit
def Density(r,params,Rmax):
    rc= params[5]
    x  = 1.0+(r/(rc))**2
    z  = 2.0/(x**2)
    return z/CDF(params,Rmax)

@jit
def logLikeStar(x,r,p,params,G,kappa,beta,cte,Rmax):
    A = np.log(p)     +logrho(x,G,kappa,beta)-cte + np.log(Density(r,params,Rmax))+np.log(r)
    # B = np.log(1.0-p) - np.log(4.0*np.pi) + np.log(2.0) - 2.0*np.log(Rmax)+np.log(r)
    # print(A,B,np.logaddexp(A,B))
    return A
    # return np.logaddexp(A,B)

@jit
def logrho(x,G,kappa,beta):
    g1x  = G[:,0].dot(x)
    g2x  = G[:,1].dot(x)
    g3x  = G[:,2].dot(x)
    f    = kappa*(g1x + 0.5*beta*(g2x**2 - g3x**2))
    return f

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
        self.pro        = self.cdts[:,2]

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
        theta,phi,psi,epsilon  = params[0],params[1],params[2],params[4]
        #============== Obtains radii and pa ===================
        # Equation 2 from Kacharov 2014.
        x     = np.sin(self.cdts[:,0]*D2R -phi)*np.cos(self.cdts[:,1]*D2R)*self.Dist
        y     = (np.cos(theta)*np.sin(self.cdts[:,1]*D2R)-
                np.sin(theta)*np.cos(self.cdts[:,1]*D2R)*np.cos(self.cdts[:,0]*D2R-phi))*self.Dist
        
        # x     = (self.cdts[:,0]*D2R - phi  )*self.Dist
        # y     = (self.cdts[:,1]*D2R - theta)*self.Dist
        xn    = x*np.cos(psi) - y*np.sin(psi)
        yn    = x*np.sin(psi) + y*np.cos(psi)
        t     = np.arctan2(yn,xn)
        radii = np.sqrt(xn**2 + yn**2)
        r2a   = np.abs(np.cos(t))*np.sqrt(1.0+(np.tan(t)/np.sqrt(1.0-epsilon**2))**2)
        aes   = radii*r2a

        kent   = KentDistribution(params[0], params[1], params[2],params[3],params[4])
        G,k,b  = Gamma(params[0],params[1],params[2]),params[3],params[4]
        
        cte    = kent.log_normconst(k,b)
        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda r,x,p:logLikeStar(x,r,p,params,G,k,b,cte,self.Rmax),aes,self.data,self.pro))
        # print(llike)
        return llike+np.sum(np.log(r2a))



