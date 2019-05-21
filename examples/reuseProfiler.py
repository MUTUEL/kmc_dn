import sys
sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import time, random, json


#%% Parameters

def profile(N = 30, M=3, hops= 1100000, tests = 100):
    xdim = 1  # Length along x dimension
    ydim = 1  # Length along y dimension
    zdim = 0  # Length along z dimension
    #res = 1  # Resolution of laplace grid


    
    #%% Initialize simulation object

    for i in range(0, tests):
        # Define electrodes
        electrodes = np.zeros((8, 4))
        voltage_range = 600
        electrodes[0] = [0, ydim/4, 0, (random.random()-0.5)*voltage_range]
        electrodes[1] = [0, 3*ydim/4, 0, (random.random()-0.5)*voltage_range]
        electrodes[2] = [xdim, ydim/4, 0, (random.random()-0.5)*voltage_range]
        electrodes[3] = [xdim, 3*ydim/4, 0, (random.random()-0.5)*voltage_range]
        electrodes[4] = [xdim/4, 0, 0, (random.random()-0.5)*voltage_range]
        electrodes[5] = [3*xdim/4, 0, 0, (random.random()-0.5)*voltage_range]
        electrodes[6] = [xdim/4, ydim, 0, (random.random()-0.5)*voltage_range]
        electrodes[7] = [3*xdim/4, ydim, 0, 0]
        kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes = electrodes)
        print ("i:%d\n"%(i))
        kmc.go_simulation(hops=hops,goSpecificFunction="wrapperSimulateRecord")

def compile(splits, hops):
    with open('reusing.log') as f:
        data = json.load(f)
        print (data)

        pdata = []
        keys = []
        key = 2
        i = 0
        while key < hops:
            ru_sum = [0]*len(splits)
            s_sum = [0]*len(splits)
            j = 0
            cur = 0
            for result in data:
                if j >= splits[cur][0]:
                    j = 0
                    cur+=1
                ru_sum[cur]+= result['Reuses'][i]
                s_sum[cur]+= result['States'][i]
                j+=1
            arr = [str(key)]
            for j in range(len(splits)):
                arr.append("%.1f"%(ru_sum[j]/splits[j][0]))
                arr.append("%.1f"%(s_sum[j]/splits[j][0]))
            pdata.append(arr)
            key*=2
            i+=1
        print (pdata)
        ptable = """\n\\begin{center}\n\\begin{tabular}{ %s }\n\\hline\nhops"""%("|c|"+"c|c|"*len(splits))
        for split in splits:
            ptable+=" & %s reuses & %s states"%(split[1], split[1])
        ptable += "\\\\\n\\hline\n"
        for row in pdata:
            ptable+="%s"%(row[0])
            j = 1
            while j < len(row):
                ptable+=" & %s & %s"%(row[j], row[j+1])
                j+=2
            ptable+="\\\\\n"
       
        ptable+="""\\hline\\end{tabular}\n\\end{center}\n"""
        print (ptable)

hops = 8500000
tests = 100
#profile(10, 1, hops, tests)
#profile(20, 2, hops, tests)
l = [60, 120, 240]
l2 = [3, 6, 12]
profile(l[0], l2[0], hops, tests)
profile(l[1], l2[1], hops, tests)
profile(l[2], l2[2], hops, tests)

compile([(tests, "%dD "%(l[0])), (tests, "%dD "%(l[1])), (tests, "%dD "%(l[2]))], hops)