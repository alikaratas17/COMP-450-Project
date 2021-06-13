import PFGTD,CWPFGTD
"""
Algorithm taken from:
A. Jacobsen and A. Chan, "Parameter-free Gradient Temporal Difference Learning", arXiv, preprint, 10 May 2021, arXiv:2105.04129
"""

class Algo:
  def __init__(self,g,d=8,B="DEFAULT"):
    self.C = CWPFGTD.ConstrainedClipping(g,self,CWPFGTD.scaler)
    self.A1 = CWPFGTD.CWPFGTD(g,W0=1/2,dimension=d,B=B)
    self.A2 = PFGTD.PFGTD(g,W0=d/2,dimension=d,B=B)
    self.dim = d
  def getWeight(self):
    return self.A1.play() + self.A2.play()
  def update(self,g):
    g1,h = self.C.update(g)
    self.A1.update(g1,h)
    self.A2.update(g1,PFGTD.magnitude(h,self.dim))
