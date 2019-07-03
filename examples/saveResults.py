from AnimateSwipe import getSwipeResults
from voltage_search_tests import getRandomDn

def main():
    #for index in range(4000, 4070):
    #    animateExample(index, False, dmp_name="resultDump")
    for index in [1]:
        rel_path = "resultDump%d.kmc"%(index)
        dn = getRandomDn(30, 3)
        dn.loadSelf(rel_path, True)
        getSwipeResults(dn, 40, 5000000, 40)
        dn.saveSelf(rel_path, True)
        

if __name__== "__main__":
    main()