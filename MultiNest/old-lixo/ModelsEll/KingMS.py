import sys
import numpy as np
from numba import jit
from scipy.stats import norm
from scipy.stats import poisson
import pandas as pd
import scipy.integrate as integrate

D2R   = np.pi/180.
cntr  = [56.65,24.13]


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

def Number(r,params,Rmax):
    Num = np.vectorize(lambda y: integrate.quad(lambda x:Density(x,params,Rmax)*x,1e-5,y,
                epsabs=1.49e-03, epsrel=1.49e-03,limit=1000)[0])
    return Num(r)

def Density(r,params,Rmax):
    return np.piecewise(r,[r>params[5],r<=params[5]],[0.0,
        lambda x: King(x,params[4],params[5])/CDFs(Rmax,params[4],params[5])])

@jit
def King(r,rc,rt):
    x = 1 + (r/rc)**2
    y = 1 + (rt/rc)**2
    z = rc**2 + rt**2
    k   = 2*((x**(-0.5))-(y**-0.5))**2
    norm= (rc**2)*(-4 + (rt**2)/z + 4*rc/np.sqrt(z) + np.log(z) -2*np.log(rc))
    lik = k/norm
    return lik

#@jit
#def LikePA(theta):

#    quadrants = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
#    quarter = pd.cut(theta,bins=quadrants,include_lowest=True)
#    counts = pd.value_counts(quarter)
#    Like = poisson.pmf(counts,len(theta)/4.0)
#    totLike = reduce(lambda x, y: x*y, Like)
#    return totLike

@jit
def LikeField(r,rm):
    return 2.*r/rm**2

@jit
def logLikeStar(x,params,Rmax):
#    return np.log(x[0]*(x[1]*Density(x[1],params,Rmax))*LikePA(x[2],0.01) + (1.-x[0])*LikeField(x[1],Rmax))
    return np.log(x[0]*(x[1]*Density(x[1],params,Rmax)) + (1.-x[0])*LikeField(x[1],Rmax))

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

        #JO: a0,d0,epsi,delta   = params[0],params[1],params[2],params[3]
        # LSB:
        a0, d0, b, delta, a, rta, rtb = params[0],params[1],params[2],params[3],params[4], params[5], params[6]
        local_params = [params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]]

        #============== Obtains radii and pa ===================
        # Equation 2 from Kacharov 2014. Since we also infer the centre positon x0=y0=0
        #x     = np.sin((self.cdts[:,0]-(cntr[0]-a0))*D2R)*np.cos(self.cdts[:,1]*D2R)
        #y     = (np.cos((cntr[1]-d0)*D2R)*np.sin(self.cdts[:,1]*D2R)-
        #        np.sin((cntr[1]-d0)*D2R)*np.cos(self.cdts[:,1]*D2R)*np.cos((self.cdts[:,0]-(cntr[0]-a0))*D2R))

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
        mock_params = np.tile(local_params,(len(x),1))

        # Include mass segregation
        rcThetaJ = rcTheta + params[7] * (self.cdts[:,3]-13.5)
        tmp1 = np.isnan(rcThetaJ)
        rcThetaJ[tmp1]=rcTheta[tmp1]

        mock_params[:,4] = rcThetaJ
        mock_params[:,5] = rtTheta

        # radii = np.arccos(np.sin(cntr[1]*D2R)*np.sin(self.cdts[:,1]*D2R)+
        #         np.cos(cntr[1]*D2R)*np.cos(self.cdts[:,1]*D2R)*
        #         np.cos((cntr[0]-self.cdts[:,0])*D2R))*self.Dist + 1e-20
        # theta = np.arctan2(np.sin((self.cdts[:,0]-cntr[0])*D2R),
        #                  np.cos(cntr[1]*D2R)*np.tan(self.cdts[:,1]*D2R)-
        #                  np.sin(cntr[1]*D2R)*np.cos((self.cdts[:,0]-cntr[0])*D2R))

        # To be removed: in an ellipse, not all angles are equally probable
        #idx   = np.argsort(theta)
        #cdf   = (1./len(theta))*np.arange(len(theta))
        #data  = np.vstack([self.cdts[:,2][idx],r[idx],cdf,theta[idx]]).T
        data  = np.vstack([self.cdts[:,2],r,theta]).T

        quadrants = [0,np.pi/2.0,np.pi,3.0*np.pi/2.0,2.0*np.pi]
        quarter = pd.cut(theta,bins=quadrants,include_lowest=True)
        counts = pd.value_counts(quarter)
        Like = poisson.pmf(counts,len(theta)/4.0)
        totLike = reduce(lambda x, y: x*y, Like)
        
        #------ Computes Likelihood -----------
        llike  = np.sum(map(lambda x,thetaparams:logLikeStar(x,thetaparams,self.Rmax),data,mock_params))+np.log(totLike)
        if (np.isnan(llike)): llike = -9999999.99
        return llike











