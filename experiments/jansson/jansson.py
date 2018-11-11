'''
This experiment attempts to recreate the results shown in Jansson's paper:
    Negative differential conductivity in the hopping transport model.
The following parameters are used in the reproduction plot:
    kT = 1
    I_0 = 30kT
    U = [0, -250]
    a_b = 0.05R seems to give a good correspondence
Also not that since I use holes instead of electrons (and the whole (1-n) vs just n)
my simulation should give similar results for negative voltage.
'''
#%% Imports
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt

#%% System setup
xdim = 1
ydim = 1
acceptors = np.zeros((10, 3))
acceptors[0] = [xdim/10, ydim/2, 0]
acceptors[1] = [2*xdim/10, ydim/2, 0]
acceptors[2] = [3*xdim/10, ydim/2, 0]
acceptors[3] = [4*xdim/10, ydim/2, 0]
acceptors[4] = [5*xdim/10, ydim/2, 0]
acceptors[5] = [6*xdim/10, ydim/2-ydim/10, 0]
acceptors[6] = [6*xdim/10, ydim/2+ydim/10, 0]
acceptors[7] = [7*xdim/10, ydim/2+ydim/10, 0]
acceptors[8] = [8*xdim/10, ydim/2+ydim/10, 0]
acceptors[9] = [9*xdim/10, ydim/2+ydim/10, 0]
electrodes = np.zeros((2, 4))
electrodes[0] = [0, ydim/2, 0, 10]
electrodes[1] = [xdim, ydim/2, 0, 0]
kT = 1
I_0 = 100*kT
ab_R = 0.05

#%% Initialize system
kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes)
kmc.load_acceptors(acceptors)

# Set constants
kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R
kmc.initialize(placement = False)

#%% Simulate IV curve
V_high = -1000
points = 100
bias = np.linspace(V_high, -V_high, points)
current = kmc_dn_utils.IV(kmc, 0, bias, hops = 10000) 

#%% Visualize
domain = kmc_dn_utils.visualize_basic(kmc)

plt.figure()
plt.plot(bias, current[:, 0])
plt.show()
