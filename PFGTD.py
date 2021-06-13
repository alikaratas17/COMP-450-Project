import numpy as np
from math import log,exp
import random
"""
Algorithm taken from:
A. Jacobsen and A. Chan, "Parameter-free Gradient Temporal Difference Learning", arXiv, preprint, 10 May 2021, arXiv:2105.04129
"""

class PFGTD:
  def __init__(self,g,W0 = 1,dimension = 8,B="DEFAULT"):
    self.W = W0
    if B=="DEFAULT":
        self.B = magnitude(g,dimension)/W0
    elif B=="ZERO":
        self.B = 0
    self.dim = dimension
    self.sum_m_squared = 0
    self.u = np.array([random.uniform(-1,1) for i in range(dimension)]).reshape((dimension,1))
    self.sum_g_mag = magnitude_squared(g,self.dim)
  def play(self):
    self.v = self.B * self.W
    w = self.v * self.u
    return w
  def update(self,g,h):
    s = np.matmul(g.reshape((1,self.dim)),self.u.reshape((self.dim,1)))
    self.W = self.W - s * self.v
    m = s/(1 - self.B * s)
    self.sum_m_squared += m ** 2
    temp1 = 2/(2 - log(3)) * m / (1+self.sum_m_squared)
    B_hat = self.B - temp1
    self.B = max(min(B_hat, 1/(2*h+ + 1e-20)),-1/(2*h+ + 1e-20) )
    self.sum_g_mag += magnitude_squared(g,self.dim)
    self.u = logistic_func(self.u - g * ((2 ** 0.5)/2) / (self.sum_g_mag ** 0.5 + 1e-20),self.dim)
      

class ConstrainedClipping:
  def __init__(self,g,A,M):
    self.hint = magnitude(g,A.dim)
    self.A = A
    self.M = M
  
  def update(self,g):
    gtrunc = np.zeros((self.A.dim,1))
    scaled = self.M(g,self.A.dim)
    if scaled > self.hint:
      gtrunc += self.hint*g/(scaled+1e-20)
    else:
      gtrunc += g
    h = max(self.hint, scaled)
    self.hint=h
    return gtrunc,self.hint

def scaler(x,dim):
  return magnitude(x,dim)

class Algo:
  def __init__(self,g,d=8):
    self.A = PFGTD(g,dimension=d)
    self.C = ConstrainedClipping(g,self.A,scaler)
  def getWeight(self):
    return self.A.play()
  def update(self,g):
    g,h = self.C.update(g)
    self.A.update(g,h)

def magnitude(g,d):
  s =0
  for i in range(d):
    s+= g[i]**2
  return s**0.5

def magnitude_squared(g,d):
  s =0
  for i in range(d):
    s+= g[i]**2
  return s

def logistic_func(e,d):
  return np.array([1/(1+exp(-e[i])) for i in range(d)]).reshape((d,1))
