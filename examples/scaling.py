'''
Perform scaling experiment on the kmc simulate function.
'''

import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import time
#%% Parameters

N = np.linspace(1, 100, num=20, dtype = int)  # Acceptor list
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
hops = int(1E4)
res = 0.1

# Define electrodes
electrodes = np.zeros((2, 4))  # Electrodes with their voltage
electrodes[0] = [0, ydim/2, 0, 10]  # Left electrode
electrodes[1] = [xdim, ydim/2, 0, -10] # Right electrode

# Initialize other parameters
times = np.zeros(len(N))
 
#%% Run simulation loops

for i in range(len(N)):
    # Initialize new object
    kmc = kmc_dn.kmc_dn(N[i], M, xdim, ydim, zdim, electrodes = electrodes, res = res)
    
    # Simulate hops and time
    tic = time.time()
    kmc.simulate(hops = hops)
    times[i] = time.time() - tic
    

#%% Visualize()

plt.plot(N, times, '.')
plt.title('$10^4$ hops')
plt.ylabel('Time (s)')
plt.xlabel('Amount of acceptors')
plt.show()