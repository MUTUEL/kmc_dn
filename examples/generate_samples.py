from voltage_search_tests import getRandomDn

def main():
    dops = [30, 30, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10]

    for i in range(len(dops)):
        dn = getRandomDn(dops[i], round(dops[i]/10.0))
        dn.saveSelf("sample%d.kmc"%(i+1))



if __name__== "__main__":
  main()