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

N = 1  # Number of acceptors
M = 0  # Number of donors
xdim = 20  # Length along x dimension
ydim = 20  # Length along y dimension
zdim = 20  # Length along z dimension
res = 1  # Resolution of laplace grid

# Define electrodes
electrodes = np.zeros((2, 5))  # Electrodes with their voltage
electrodes[0] = [0, 0, 0, 10, 0]  # Left electrode
electrodes[1] = [xdim, 0, 0, -10, 0] # Right electrode


#%% Dopant (1.) and charge (2.) placement, and potential (3.) and compensation (4.).

kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes, res)

#%% Create test scenario

# kmc.acceptors = np.array([[xdim/2, ydim/2, 0, 1]])
#%% Update transition matrix (5. and 6.)

#%% Pick hopping event (7.)

# kmc.simulate(interval=100)

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(111)

## Plot potential profile
ax.imshow(kmc.V[:, :, 1].transpose(), interpolation='bicubic', origin='lower')

# Plot impurity configuration (red = 2, orange = 1, black = 0 holes)
colors = ['red' if i==2
          else 'orange' if i==1
          else 'black' for i in kmc.acceptors[:, 3]]
ax.scatter(kmc.acceptors[:, 0], kmc.acceptors[:, 1], c = colors, marker='o')

ax.scatter(kmc.donors[:, 0], kmc.donors[:, 1], marker='x')

plt.show()
