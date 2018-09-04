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

import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import cProfile


#%% Parameters

N = 5  # Number of acceptors
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
#res = 1  # Resolution of laplace grid

# Define electrodes
electrodes = np.zeros((2, 5))  # Electrodes with their voltage
electrodes[0] = [0, ydim/2, 0, 10, 0]  # Left electrode
electrodes[1] = [xdim, ydim/2, 0, 0, 0] # Right electrode

 
#%% Initialize simulation object

kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes)

# Place acceptors on a line
for i in range(N):
    kmc.acceptors[i] = [(i+1) * xdim/(N+1), ydim/2, 0, 0]
#kmc.acceptors[0, 3] = 1

# Re initialize
kmc.calc_distances()
kmc.constant_energy()


#%% Simulate

kmc.simulate()



