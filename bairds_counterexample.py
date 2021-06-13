import numpy as np
import random
import matplotlib.pyplot as plt
import updaters
import bairds_mdp
import CWPFGTD

def mainTD():
  mdp = bairds_mdp.MDP()
  updater = updaters.OffPolicyTD(0.01,mdp.getDiscountFactor())
  stepCount = 10000
  weights = np.zeros((8,stepCount+1))
  values = np.zeros((stepCount+1))
  for j in range(8):
    weights[j,0] = random.uniform(-1,1)
  current = random.randint(1,7)
  newFeatures = np.array(mdp.getFeatures(current))
  values[0] = np.matmul(newFeatures.reshape((1,8)),weights[:,0])
  reward = 0
  for i in range(stepCount):
    prev = current
    oldFeatures = newFeatures
    current,ratio = mdp.nextState(current)
    newFeatures = np.array(mdp.getFeatures(current))
    
    values[i+1] = np.matmul(newFeatures.reshape((1,8)),weights[:,i])
    newWeights = updater.update(reward,weights[:,i],oldFeatures,newFeatures,ratio)
    for j in range(8):
      weights[j,i+1]=newWeights[j]
  for j in range(8):
    plt.figure()
    plt.title("Weight:"+str(j+1))
    plt.plot(range(stepCount+1),weights[j,:])
  plt.figure()
  plt.title("Values")
  plt.plot(range(stepCount+1),values)
  plt.show()



def mainGTD2():
  mdp = bairds_mdp.MDP()
  updater = updaters.GTD2(0.001,0.001,mdp.getDiscountFactor())
  stepCount = 10000000
  weights = np.zeros((8,stepCount+1))
  values = np.zeros((stepCount+1))
  for j in range(8):
    weights[j,0] = random.uniform(-1,1)
  current = random.randint(1,7)
  newFeatures = np.array(mdp.getFeatures(current))
  values[0] = np.matmul(newFeatures.reshape((1,8)),weights[:,0])
  reward = 0
  for i in range(stepCount):
    prev = current
    oldFeatures = newFeatures
    current,ratio = mdp.nextState(current)
    newFeatures = np.array(mdp.getFeatures(current))
    
    values[i+1] = np.matmul(newFeatures.reshape((1,8)),weights[:,i])
    newWeights = updater.update(oldFeatures,newFeatures,reward,weights[:,i],ratio)
    for j in range(8):
      weights[j,i+1]=newWeights[j]
  for j in range(8):
    plt.figure()
    plt.title("Weight:"+str(j+1))
    plt.plot(range(stepCount+1),weights[j,:])
  plt.figure()
  plt.title("Values")
  plt.plot(range(stepCount+1),values)
  plt.show()




def mainCWPFGTD():
  mdp = bairds_mdp.MDP()
  stepCount = 1000
  current = random.randint(1,7)
  newFeatures = np.array(mdp.getFeatures(current))
  weights = np.zeros((8,stepCount+1))
  values = np.zeros((stepCount+1))
  values[0] = np.matmul(newFeatures.reshape((1,8)),weights[:,0])
  A = CWPFGTD.CWPFGTD()
  C = CWPFGTD.ConstrainedClipping(np.array([random.uniform(-1,1) for i in range(8)]).reshape((8,1)),A,CWPFGTD.scaler)
  weight = A.run(C.run())
  for j in range(8):
    weights[j,0] = weight[j]
  for i in range(stepCount):
    prev = current
    oldFeatures = newFeatures
    current,ratio = mdp.nextState(current)
    newFeatures = np.array(mdp.getFeatures(current))
    values[i+1] = np.matmul(newFeatures.reshape((1,8)),weights[:,i])
    g,h = C.update(oldFeatures)
    A.update(g,h)
    newWeights = A.play()
    for j in range(8):
      weights[j,i+1]=newWeights[j]
  for j in range(8):
    plt.figure()
    plt.title("Weight:"+str(j+1))
    plt.plot(range(stepCount+1),weights[j,:])
  plt.figure()
  plt.title("Values")
  plt.plot(range(stepCount+1),values)
  plt.show()




if __name__=="__main__":
  mainCWPFGTD()