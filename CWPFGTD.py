import numpy as np
from math import log

"""
Algorithm taken from:
A. Jacobsen and A. Chan, "Parameter-free Gradient Temporal Difference Learning", arXiv, preprint, 10 May 2021, arXiv:2105.04129
"""

class CWPFGTD:
  def __init__(self,g,W0 = 1,dimension = 8,B="DEFAULT"):
    self.W = [np.zeros((dimension,1))+W0]
    if B=="DEFAULT":
        self.B = [g/W0]
    else:
        self.B = [np.zeros((dimension,1))]
    self.dim = dimension
    self.sum_m = np.zeros((dimension,1))
  def play(self):
    return self.B[-1]*self.W[-1]
  def update(self,g,h):
    W = np.zeros((self.dim,1))
    for i in range(self.dim):
      W[i] = self.W[-1][i]-g[i]*self.B[-1][i]
    self.W.append(W)
    m = np.zeros((self.dim,1))
    for i in range(self.dim):
      m[i] = g[i]/(1-self.B[-1][i]*g[i])
    self.sum_m +=m**2
    B_hat = np.zeros((self.dim,1))
    for i in range(self.dim):
      B_hat[i]=self.B[-1][i]-2*m[i]/((2-log(3))*(1+self.sum_m[i]))
    B = np.zeros((self.dim,1))
    for i in range(self.dim):
      B[i]= max(min(B_hat[i],1/(2*h[i]+1e-15)),-1/(2*h[i]+1e-15))
    self.B.append(B)
      

class ConstrainedClipping:
  def __init__(self,g,A,M):
    self.hint = g
    self.A = A
    self.M = M
  
  def update(self,g):
    gtrunc = np.zeros((self.A.dim,1))
    for i in range(self.A.dim):
      if self.M(g[i]) > self.hint[i]:
        gtrunc[i] +=self.hint[i]*g[i]/(self.M(g[i])+1e-15)
      else:
        gtrunc[i] +=g[i]
    h = np.zeros((self.A.dim,1))
    for i in range(self.A.dim):
      h[i] = max(self.hint[i],self.M(g)[i])
    self.hint=h
    return gtrunc,self.hint

def scaler(x):
  return abs(x)

class Algo:
  def __init__(self,g,d=8):
    self.A = CWPFGTD(g,dimension=d)
    self.C = ConstrainedClipping(g,self.A,scaler)
  def getWeight(self):
    return self.A.play()
  def update(self,g):
    g,h = self.C.update(g)
    self.A.update(g,h)

