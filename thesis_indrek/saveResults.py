from voltage_search_tests import getRandomDn
from utils import parseArgs, getSwipeResults

# Used to generate swipe_results for .kmc files. This uses 
# parameters so it can be used in cluster.
# -i: start index of the first file.
# -t: Number of files to be processed after the first.
# -f: folder of the files. Relative to the script calling location

def main():
    args = parseArgs()
    if "i" in args:
        startIndex = int(args["i"])
    else:
        startIndex = 0
    if "t" in args:
        times = int(args["t"])
    else:
        times = 20
    if "f" in args:
        folder = args["f"] + "/"
    else:
        folder = ""
    for index in range(startIndex, startIndex+times):
        rel_path = "%sresultDump%d.kmc"%(folder, index)
        dn = getRandomDn(30, 3)
        try:
            dn.loadSelf(rel_path, True)
            print("%d: %d"%(index, dn.N))
        except:
            continue        
        getSwipeResults(dn, 40, 5000000, 40)
        dn.saveSelf(rel_path, True)
        

if __name__== "__main__":
    main()