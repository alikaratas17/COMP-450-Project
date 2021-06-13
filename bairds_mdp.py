import random
import numpy as np
"""
Example taken from:
R.S. Sutton and A.G. Barto, “Reinforcement Learning An Introduction”, MIT Press, 2020.
"""

class MDP:
  def __init__(self):
    self.discount = 0.99
  
  def nextState(self,state):
    next_state = random.randint(1,7)
    if state == 7 and next_state == 7:
      ratio = 7.0
    elif state == 7:
      ratio = 0.0
    elif next_state == 7:
      ratio = 7.0
    else:
      ratio = 0.0
    return next_state,ratio
  
  def getStateValue(self,state,weights):
    return np.matmul(weights.reshape((1,8)),self.getFeatures(state))
  
  def getDiscountFactor(self):
    return self.discount
  
  def getFeatures(self,state):
    features = np.zeros((8,1))
    if state == 7:
      features[6]=1
      features[7]=2
    else:
      features[7]=1
      features[state-1]=2
    return features

