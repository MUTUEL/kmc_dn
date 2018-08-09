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

#%% Parameters

N = 20  # Number of acceptors
M = 10  # Number of donors
xdim = 20  # Length along x dimension
ydim = 20  # Lenght along y dimension
res = 1  # Resolution of laplace grid

# Define electrodes
electrodes = np.empty((2, 3))  # Electrodes with their voltage
electrodes[0] = [0, ydim/2, 10]  # Left electrode
electrodes[1] = [xdim, ydim/2, 0] # Right electrode


#%% Dopant (1.) and charge (2.) placement, and potential (3.) and compensation (4.).

kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, electrodes, res)


#%% Update transition matrix (5. and 6.)

kmc.update_transition_matrix()


#%% Plotting

print(kmc.E_constant)

## Plot potential profile
plt.imshow(kmc.V.transpose(), interpolation='bicubic')

# Plot impurity configuration
plt.plot(kmc.acceptors[:, 0], kmc.acceptors[:, 1], 'x')
plt.plot(kmc.donors[:, 0], kmc.donors[:, 1], 'o')


