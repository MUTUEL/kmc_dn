#!/usr/bin/python
import time
from utils import parseArgs
from voltage_search_tests import getRandomDn


# This file is used in the cluster to count finished scripts. 
# By giving it the right parameters you can be sure that this 
# doesn't finish before everything else.
# You can use parameters.
# Use either -s int -t int, or -i int. But not both
# -s: start index of the file required.
# -t: number of files required after start index
# -i: index of one of the premade lists. For complex lists you can define one here, 
#   and append to arr.
# -r: Weather swipe_results are required. If it is false, we only check for file existance.
#       However, if it is true, then we don't require file existance and only require
#       swipe_results existance for files that do exist.
#

def main():
    arr = [i for i in range(1, 15)]
    arr.extend(range(16, 46))
    for i in range(1, 11):
        arr.extend(range(i*100+1, i*100+15))
        arr.extend(range(i*100+16, i*100+46))
    allArrs = [arr]

    arr = [i for i in range(0, 700)]
    allArrs.append(arr)
    args = parseArgs()


    if "s" in args and "t" in args:
        arr = [i for i in range(int(args["s"]), int(args["s"])+int(args["t"]))]
    elif "i" in args:
        arr = allArrs[int(args["i"])]
    else:
        arr = allArrs[0]
    if "r" in args:
        requires_results = True
    else:
        requires_results = False
    if "f" in args:
        folder = args["f"] + "/"
    else:
        folder = ""
    i = 0
    while i < len(arr):
        rel_path = "%sresultDump%d.kmc"%(folder, arr[i])
        dn = getRandomDn(30, 3)
        try:
            dn.loadSelf(rel_path, True)
            if not requires_results:
                i+=1
                print("incremented")
            elif hasattr(dn, "swipe_results"):
                i+=1
                print("incremented")
            else:
                print("sleeping")
                time.sleep(100)
        except:
            if requires_results:
                i+=1
                print("incremented")
            else:
                print("sleeping")
                time.sleep(100)
        

if __name__== "__main__":
    main()
        