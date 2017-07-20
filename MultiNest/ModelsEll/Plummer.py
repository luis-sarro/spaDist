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
    b = params[3]
    if b <= 0.0 : return False
    if b > rc : return False
    return True

@jit
def Number(r,params,Rmax):
    rc = params[4]
    return CDF(rc,r)/CDF(rc,Rmax)
@jit
def Density(r,params,Rmax):
    rc = params[4]
    x  = 1.0+(r/(rc))**2
    z  = 2.0*x**-2
    return z/CDF(rc,Rmax)

@jit
def CDF(rc,r):
    return (r**2)*(rc**2)/(rc**2+r**2)

@jit
def LikePA(cdf,theta,sg):
    return norm.pdf(cdf,loc=theta/(2.*np.pi),scale=sg)+ 1e-100

@jit
def logLikeStar(rad,pro,params,Rmax,cdf,theta,sg):
    return np.log(pro*(rad*Density(rad,params,Rmax))*LikePA(cdf,theta,sg) + (1.-pro)*LikeField(rad,Rmax))

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
        self.cdts       = cdts # Coordinates and probabilities
        self.Rmax       = Rmax # Maximum radius from command line
        self.t          = trans_lim # Limits of the parameter space explored
        self.Dist       = Dist # Distance to the cluster in parsecs
        self.sg         = 0.01 # For std deviation of departures from uniform
        print("Module Initialized")

    def Priors(self,params, ndim, nparams):
        #---------- Uniform Priors ------
        for i in range(ndim):
            params[i] = (params[i])*(self.t[i,1]-self.t[i,0])+self.t[i,0]

    def LogLike(self,params,ndim,nparams):
        #----- Checks if parameters' values are in the ranges
        if not Support(params):
            return -1e50

        #JO a0,d0,epsi,delta   = params[0],params[1],params[2],params[3]
        # LSB:
        a0, d0, b, delta, a = params[0],params[1],params[2],params[3],params[4]
        local_params = [params[0],params[1],params[2],params[3],params[4]]
        
        #============== Obtains radii and pa ===================
        # Equation 2 from Kacharov 2014.
        # x     = np.sin((self.cdts[:,0]-(cntr[0]-a0))*D2R)*np.cos(self.cdts[:,1]*D2R)*self.Dist
        # y     = (np.cos((cntr[1]-d0)*D2R)*np.sin(self.cdts[:,1]*D2R)-
        #         np.sin((cntr[1]-d0)*D2R)*np.cos(self.cdts[:,1]*D2R)*np.cos((self.cdts[:,0]-(cntr[0]-a0))*D2R))*self.Dist
        # The code infers a shift with respect to the centre taken from the literature.
        # D2R Degrees to radians
        # delta is the angle of the semimajor axis of the ellipse with respect to the RA axis
        # epsi is the ellipticity from 0 to 1

        # LSB: 
        #delta = np.pi/4.0
        #a0=1
        #d0=1
        #epsi=0.99
        #params[2] = 2.5
        #params[4] = 2.5
        
        x     = ((self.cdts[:,0]-(cntr[0]+a0))*D2R)*self.Dist
        y     = ((self.cdts[:,1]-(cntr[1]+d0))*D2R)*self.Dist
        xn    = (x*np.sin(delta) + y*np.cos(delta))
        yn    = (x*np.cos(delta) - y*np.sin(delta))
        theta = np.arctan2(yn,xn)
        r     = np.sqrt(xn**2 + yn**2)
        # JO: r2a   = np.abs(np.cos(theta))*np.sqrt(1.0+(np.tan(theta)/(1.0-epsi))**2)
        # JO: aes = r*r2a
        # LSB: to compute the rc for each theta.
        rcTheta   = (params[4]*params[2])/np.sqrt((params[2]*np.cos(theta))**2+(params[4]*np.sin(theta))**2)
        # That is: a*b/sqrt((b*cos(theta))^2+(a*sin(theta))^2)
        mock_params = np.tile(local_params,(len(x),1))
        mock_params[:,4] = rcTheta

        theta   = (theta + 2*np.pi)%(2*np.pi)
        idx   = np.argsort(theta)
        cdf   = (1./len(theta))*np.arange(len(theta))
        cdf = cdf[idx]
        
        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda r,pro,thetapars,cdf,theta:logLikeStar(r,pro,thetapars,self.Rmax,cdf,theta,self.sg),r,self.cdts[:,2],mock_params,cdf,theta))
        # print(params[0:5],llike)
        # JO: return llike+np.sum(np.log(r2a))
        # LSB:
        return llike



