import sys
import numpy as np
from numba import jit
from kent import *
import scipy.integrate as integrate

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
    if params[6] <= params[5]                     : return False 
    return True


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

def Number(r,params,Rmax):
    Num = np.vectorize(lambda y: integrate.quad(lambda x:Density(x,params,Rmax)*x,1e-5,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Num(r)

def Density(r,params,Rmax):
    return np.piecewise(r,[r>params[6],r<=params[6]],[0.0,
        lambda x: King(x,params[5],params[6])/CDFs(Rmax,params[5],params[6])])

def rhoR(r,params,Rmax):
    return np.piecewise(r,[r>params[6],r<=params[6]],[0.0,lambda x: King(x,params[5],params[6])])

@jit
def King(r,rc,rt):
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    z = rc**2 + rt**2
    k = 2*((x**(-0.5))-(y**-0.5))**2
    norm= (rc**2)*(-4 + (rt**2)/z + 4*rc/np.sqrt(z) + np.log(z) -2*np.log(rc))
    lik = k/norm
    return lik

@jit
def logLikeStarR(r,p,params,Rmax,cte):
    return np.log(p*(rhoR(r,params,Rmax)/cte) + (1.-p)*(2./Rmax**2))+np.log(r)

@jit
def rhoK(x,G,kappa,beta,cte,A):
    g1x  = G[:,0].dot(x)
    g2x  = G[:,1].dot(x)
    g3x  = G[:,2].dot(x)
    f    = np.exp(kappa*(g1x + 0.5*beta*(g2x**2 - g3x**2))-cte+A)
    return f

@jit
def logLikeStarK(x,p,G,kappa,beta,cte,A):
    """
    Uses constant A to avoid overflow.
    """
    return np.log(p*rhoK(x,G,kappa,beta,cte,A) + ((1.0-p)*np.exp(A)/(4.0*np.pi)))-A

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
        self.A          = 700.0
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

        kent       = KentDistribution(params[0], params[1], params[2],params[3],params[4])
        G,k,b      = Gamma(params[0],params[1],params[2]),params[3],params[4]
        
        cte        = kent.log_normconst(k,b)
        #------ Computes Likelihood -----------
        llike_ang  = np.sum(map(lambda xyz,p:logLikeStarK(xyz,p,G,k,b,cte,self.A),self.data,self.pro))
        llike_rad  = np.sum(map(lambda r,p:logLikeStarR(r,p,params,self.Rmax,cte),aes,self.pro))
        # print(llike_ang,llike_rad)
        return llike_ang+llike_rad+np.sum(np.log(r2a))



