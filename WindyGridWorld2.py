"""
Example taken from:
R.S. Sutton and A.G. Barto, “Reinforcement Learning An Introduction”, MIT Press, 2020.
"""

class mdp:
  def __init__(self):
    pass
  def getStartState(self):
    return (0,4)
  def getTransition(self,state,action):
    wind = self.getWind(state[0])
    move = self.getMove(action)
    newState = [state[0]+move[0],state[1]-wind+move[1]]
    if newState[0]<0:
      newState[0]=0
    elif newState[0]>9:
      newState[0]=9
    if newState[1]<0:
      newState[1]=0
    elif newState[1]>7:
      newState[1]=7
    newState= tuple(newState)
    if newState==(7,4):
      return newState,1
    return newState,0
  def getMove(self,a):
    if a=="UP":
      return (0,-1)
    if a=="DOWN":
      return (0,1)
    if a=="RIGHT":
      return (1,0)
    if a=="LEFT":
      return (-1,0)
    print("Unrecognized action",a)
  def getWind(self,x):
    if x==6 or x==7:
      return 2
    if x >2 and x<9:
      return 1
    return 0