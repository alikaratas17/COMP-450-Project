import random
"""
Example taken from:
R.S. Sutton and A.G. Barto, “Reinforcement Learning An Introduction”, MIT Press, 2020.
"""

class randomWalk:
  def getStartState(self):
    return 500
  def getTransition(self,state):
    next_state = self.__nextState(state)
    if next_state ==0:
      return next_state,-1
    if next_state ==1001:
      return next_state,+1
    return next_state,0
  def __nextState(self,state):
    if random.uniform(0,1) >0.5:
      if state <= 100:
        return random.randint(0,state-1)
      else:
        return random.randint(state-100,state-1)
    if state >= 901:
      return random.randint(state+1,1001)
    else:
      return random.randint(state+1,state+100)
