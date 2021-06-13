import numpy as np
import random
"""
Both update formulas taken from:
R.S. Sutton and A.G. Barto, “Reinforcement Learning An Introduction”, MIT Press, 2020.
"""

class UpdateMethod:
  def __init__(self):
    pass
  def update(self):
    pass

class OffPolicyTD(UpdateMethod):
  def __init__(self,alpha,discount,dim=8):
    self.alpha = alpha
    self.discount = discount
    self.dim = dim
  def update(self, reward,weights,oldFeatures,newFeatures,ratio,debug =False):
    a1 = np.matmul(newFeatures.reshape((1,self.dim)),weights)
    a2 = np.matmul(oldFeatures.reshape((1,self.dim)),weights)
    delta_t = reward + self.discount *  a1- a2
    if debug:
        print(np.transpose(a1))
        print(np.transpose(a2))
        print(delta_t)
        print(np.transpose(self.alpha * ratio * delta_t * oldFeatures.reshape(self.dim,1)))
    return weights.reshape(self.dim,1) + self.alpha * ratio * delta_t * oldFeatures.reshape(self.dim,1)

class GTD2(UpdateMethod):
  def __init__(self,alpha,betha,discount,dim=8):
    self.alpha = alpha
    self.betha = betha
    self.discount = discount
    self.v_vector = np.zeros((dim,1))
    self.dim = dim
    for j in range(8):
      self.v_vector[j] = random.uniform(-1,1)
  def update(self,oldFeatures,newFeatures,reward,weights,ratio):
    a1 = reward+self.discount*np.matmul(newFeatures.reshape(1,self.dim),weights.reshape((self.dim,1)))
    a2 = np.matmul(oldFeatures.reshape(1,self.dim),weights.reshape((self.dim,1)))
    delta = a1-a2

    b1 =oldFeatures-self.discount*newFeatures
    b2 = np.matmul(oldFeatures.reshape((1,self.dim)),self.v_vector)
    b3 = self.alpha*ratio*b1*b2
    newWeights =  weights.reshape((self.dim,1))+ b3
    c1 = (delta-np.matmul(oldFeatures.reshape(1,self.dim),self.v_vector))*oldFeatures
    self.v_vector = self.v_vector + self.betha * ratio * c1
    return newWeights

