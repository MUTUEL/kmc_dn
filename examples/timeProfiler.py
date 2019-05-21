import sys
sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import time


#%% Parameters

N = 30  # Number of acceptors
M = 4  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
hops = int(22E5)
#res = 1  # Resolution of laplace grid

# Define electrodes
electrodes = np.zeros((8, 4))
electrodes[0] = [0, ydim/4, 0, 10]
electrodes[1] = [0, 3*ydim/4, 0, 0]
electrodes[2] = [xdim, ydim/4, 0, 10]
electrodes[3] = [xdim, 3*ydim/4, 0, 0]
electrodes[4] = [xdim/4, 0, 0, 10]
electrodes[5] = [3*xdim/4, 0, 0, 0]
electrodes[6] = [xdim/4, ydim, 0, 10]
electrodes[7] = [3*xdim/4, ydim, 0, 0]

 
#%% Initialize simulation object
times = {"python":[], "go":[], "goRecord":[]}
hopSteps = [5000, 25000, 100000, 250000, 500000, 2000000]
for i in range(0, 10):
    kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes = electrodes)
    timeRow = {"python":[], "go":[], "goRecord":[]}
    print ("i:%d\n"%(i))

    for j in range(len(hopSteps)):
        print ("j:%d\n"%(j))
        start = time.time()
        #kmc.python_simulation(hops=hopSteps[j])
        end = time.time()
        timeRow["python"].append(end-start)
        start = time.time()
        #kmc.go_simulation(hops=hopSteps[j],goSpecificFunction="wrapperSimulate")
        end = time.time()
        timeRow["go"].append(end-start)
        start = time.time()
        kmc.go_simulation(hops=hopSteps[j],goSpecificFunction="wrapperSimulateRecord")
        end = time.time()
        timeRow["goRecord"].append(end-start)
    for key in timeRow:
        times[key].append(timeRow[key])

averages = {}
for key in times:
    averageList = []
    for i in range(len(hopSteps)):
        sum = 0
        for j in range(len(times[key])):
            sum+=times[key][j][i]
        averageList.append(sum*1.0/len(times[key]))
    averages[key] = averageList

#print (times)
print (averages)
#%% Profile code

# Prerun discrete simulation for honest timing
# kmc.simulate_discrete(hops=hops)
#kmc.python_simulation(hops=hops)
#print (kmc.current)
#kmc.go_simulation(hops=hops)

#a = cProfile.run('kmc.simulate_discrete(hops = hops)')

