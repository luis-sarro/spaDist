import sys
from numpy import *
from math import factorial
from scipy.optimize import bisect

def Rpsi(psi):
    return array([
      [1.0, 0.0,      0.0      ],
      [0.0, cos(psi), -sin(psi)],
      [0.0, sin(psi), cos(psi) ]
    ])  

def Rphi(phi):
    return array([
      [cos(phi), -sin(phi), 0.0],
      [sin(phi),  cos(phi), 0.0],
      [0.0     ,       0.0, 1.0]
    ])  

def Rtheta(theta):
    return array([
      [cos(theta),0.0,-sin(theta)],
      [0.0       ,1.0,        0.0],
      [sin(theta),0.0, cos(theta)]
    ])  


def Gamma(theta, phi, psi):
    Rth   = Rtheta(theta)
    Rph   = Rphi(phi)
    Rps   = Rpsi(psi)
    return Rph.dot(Rth.dot(Rps))


def spherical_2_gammas(theta, phi, psi):
      G      = Gamma(theta, phi, psi)
      gamma1 = G[:,0]
      gamma2 = G[:,1]
      gamma3 = G[:,2]    
      return gamma1, gamma2, gamma3

def Spherics(G):
      # G          = stack([gamma1,gamma2,gamma3],axis=-1)
      theta, phi = arcsin(G[:,0][2]),arctan2(G[:,0][1],G[:,0][0])
      u          = ((Rtheta(theta).T.dot(Rphi(phi).T)).dot(G))[:,1]
      psi        = arctan2(u[2], u[1])
      return theta, phi, psi


class KentDistribution(object):
  """
  The kent distribution is similar to a multivariate gaussian but defined on the sphear. (i.e. all vectors have norm=1)
  This version uses the coordiante system of Heat2014, Envirnometrics, Wiley online library
  and the normalisation constant is approximated by the saddle algorithm of Kume and Wood 2005. See below.
  Notice that beta is defined in the interval [0,1) thus this beta = original_beta * kappa/2
  """
  def __init__(self, theta,phi,psi, kappa, beta):
    self.gamma1,self.gamma2,self.gamma3 = spherical_2_gammas(theta, phi, psi)
    self.kappa = float(kappa)
    self.beta = float(beta)

    assert abs(inner(self.gamma1, self.gamma2)) < 1E-10
    assert abs(inner(self.gamma2, self.gamma3)) < 1E-10
    assert abs(inner(self.gamma3, self.gamma1)) < 1E-10
    assert self.kappa >= 0.0
    assert self.beta  >= 0.0 and self.beta <= 1.0

  def log_pdf(self, xs):
    """
    Returns the log(pdf) of the kent distribution without normalisation.
    If you want it normalised substract log_normconst()
    """
    axis = len(shape(xs))-1
    g1x = sum(self.gamma1*xs, axis)
    g2x = sum(self.gamma2*xs, axis)
    g3x = sum(self.gamma3*xs, axis)
    f = self.kappa*(g1x + 0.5*self.beta*(g2x**2 - g3x**2))
    return f

  def log_normconst(self,k,b):
    """
    Returns the logarithm of the normalisation constant. C1,C2 and C3
    It uses the sadle point approximation of Kume and Wood 2005, Biometrika, 92,2,465-476 
    """

    def kfb(j, gam, lam, ta):
      if (j == 1):
        kd = sum(0.5/(lam - ta) + 0.25 * (gam**2/(lam - ta)**2))
        
      if (j > 1):
        kd = sum(0.5 * factorial(j - 1)/(lam - ta)**j + 0.25*factorial(j) * gam**2/(lam - ta)**(j + 1))
      return kd

    beta= 0.5*b*k
    gam = array([0,k,0])
    lam = array([0,-beta,beta])
    p   = 3.0

    lam  = sort(lam)
    mina = min(lam)
    if mina <= 0 :
        aaa = abs(mina) + 1
        lam = lam + aaa

    low  = lam[0] - 0.25*p - 0.5*sqrt(0.25 * p**2 + p*max(gam)**2)
    up   = lam[0] - 0.25   - 0.5*sqrt(0.25 + min(gam)**2)
    tau  = bisect(lambda ta:sum(0.5/(lam - ta) + 0.25 * (gam**2/(lam - ta)**2)) - 1,a=low, b=up)

    rho3 = kfb(3, gam, lam, tau)/kfb(2, gam, lam, tau)**1.5
    rho4 = kfb(4, gam, lam, tau)/kfb(2, gam, lam, tau)**2
    Ta   = rho4/8 - 5/24 * rho3**2
    c1   = 0.5 * log(2) + 0.5 * (p - 1) * log(pi) - 0.5 * log(kfb(2, gam, lam, tau)) - 0.5 * sum(log(lam - tau)) - tau + 0.25 * sum(gam**2/(lam - tau))
    c2   = c1 + log1p(Ta)
    c3   = c1 + Ta
    if (mina <= 0):
        c1 = c1 + aaa
        c2 = c2 + aaa
        c3 = c3 + aaa
    return c1





  
  