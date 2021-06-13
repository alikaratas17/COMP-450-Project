class Writer:
  def write(self,l,filename):
    with open(filename,"w") as f:
      for e in l:
        f.write(str(e)+";")
class Reader:
  def read(self,ty,filename):
    with open(filename,"r") as f:
      lines = f.readlines()
      elems = []
      for line in lines:
        sep = line.split(";")
        for e in sep:
          elems.append(ty(e))
    return elems