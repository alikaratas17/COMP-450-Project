import PFGTD,CWPFGTD
"""
Algorithm taken from:
A. Jacobsen and A. Chan, "Parameter-free Gradient Temporal Difference Learning", arXiv, preprint, 10 May 2021, arXiv:2105.04129
"""

class Algo:
  def __init__(self,g,d=8):
    self.A1 = CWPFGTD.CWPFGTD(g,W0=1/2,dimension=d)
    self.C1 = CWPFGTD.ConstrainedClipping(g,self.A1,CWPFGTD.scaler)
    self.A2 = PFGTD.PFGTD(g,W0=d/2,dimension=d)
    self.C2 = PFGTD.ConstrainedClipping(g,self.A2,PFGTD.scaler)
  def getWeight(self):
    return self.A1.play() + self.A2.play()
  def update(self,g):
    g1,h1 = self.C1.update(g)
    self.A1.update(g1,h1)
    g2,h2 = self.C2.update(g)
    self.A2.update(g2,h2)
