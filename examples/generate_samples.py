from voltage_search_tests import getRandomDn

def main():
    dops = [30, 30, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10]
    if "i" in args:
        startIndex = int(args["i"])
    else:
        startIndex = 0
    if "t" in args:
        times = int(args["t"])
    else:
        times = 20
    for i in range(startIndex, startIndex+times:
        dn = getRandomDn(dops[i], round(dops[i]/10.0))
        dn.saveSelf("sample%d.kmc"%(i+1))



if __name__== "__main__":
  main()