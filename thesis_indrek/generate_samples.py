from voltage_search_tests import getRandomDn

# This script is used to generate .kmc files with random placements. Good when in cluster
# You need to use the same placement from multiple parallel scripts.

def main():
    dops = [30, 30, 30, 30, 20, 20, 20, 20, 10, 10, 10, 10]

    for i in range(len(dops)):
        dn = getRandomDn(dops[i], round(dops[i]/10.0))
        dn.saveSelf("sample%d.kmc"%(i+1))



if __name__== "__main__":
  main()