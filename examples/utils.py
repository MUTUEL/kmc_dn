import sys

def parseArgs():
      i = 1
      paras = {}
      while len(sys.argv) > i+1:
            if sys.argv[i][0]=='-':
                  paras[sys.argv[i][1:]] = sys.argv[i+1]
                  i+=1
            i+=1
      return paras