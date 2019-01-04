'''
This code performs a Kinetic Monte Carlo simulation of a 2D variable range
hopping system. The code is inspired by the work of Jeroen van Gelder.

Pseudo-code (algorithm):
    # Initialization
    1. Donor placement (N acceptors and M < N donors)
    2. Place charges (N-M)
    3. Solve electrostatic potential from gates (relaxation method?)
    4. Solve compensation energy terms
    # Loop
    5. Calculate Coulomb energy terms
    6. Calculate hopping rates
    7. Hopping event
    8. Current converged?
        No: return to 5.
        Yes: simulation done.

@author: Bram de Wilde (b.dewilde-1@student.utwente.nl)
'''

import sys
sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import cProfile


#%% Parameters

N = 30  # Number of acceptors
M = 3  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
hops = int(2E5)
#res = 1  # Resolution of laplace grid

# Define electrodes
electrodes = np.zeros((2, 4))  # Electrodes with their voltage
electrodes[0] = [0, ydim/2, 0, 10]  # Left electrode
electrodes[1] = [xdim, ydim/2, 0, -10] # Right electrode

 
#%% Initialize simulation object

kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes = electrodes)

#%% Profile code

# Prerun discrete simulation for honest timing
#kmc.simulate_discrete(hops=hops)

a = cProfile.run('kmc.simulate_discrete(hops = hops)')

