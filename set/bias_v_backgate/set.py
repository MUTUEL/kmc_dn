'''Investigate single acceptor situation and see if we get set-like dynamics.'''


import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt

#%% Parameters

N = 1  # Number of acceptors
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 0  # Length along y dimension
zdim = 0  # Length along z dimension

electrodes = np.zeros((2,5))
electrodes[0] = [0, 0, 0, 10, 0]
electrodes[1] = [xdim, 0, 0, 0, 0]
 
#%% Initialize simulation object

kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes)

kmc.acceptors[0] = [xdim/2, 0, 0, 0]

# Fix dimensionless constants
N_sites = kmc.acceptors.shape[0]/(xdim)  # Site concentration
R = N_sites**(-1/2)
kmc.ab = 10000*R  #Fix ab/R = 10000
kmc.eps = 0.5*kmc.e**2/(4*np.pi*R*kmc.k*kmc.T)  # Fix I_0/kT = 1
V_0 = kmc.k*kmc.T/kmc.e  # Fix eV_0/kT = 1

# Set temperature
kmc.T = 0.01 * kmc.e*V_0/kmc.k

# Temporary potential fix
kmc.V[:, 0, 0] = np.linspace(electrodes[0, 3], electrodes[1, 3], num=102)
kmc.calc_distances() 
kmc.constant_energy()

#%% Current simulation with backgate

hops = int(1E3)

# Voltages
bias = np.linspace(-10*V_0, 10*V_0, num = 20)
#bias = np.zeros(1)
#bias[0] = 0.5
backgate = np.linspace(-10*V_0, 10*V_0, num = 20)
#backgate = np.zeros(1)
#backgate[0] = -V_0
current = np.zeros((backgate.shape[0], bias.shape[0]))


for i in range(backgate.shape[0]):
    for j in range(bias.shape[0]):
        kmc.electrodes[0, 3] = bias[j]
        kmc.V[:, 0, 0] = np.linspace(kmc.electrodes[0, 3], electrodes[1, 3], num=102)  # Set bias
        kmc.V[40:60, 0, 0] = backgate[i]  # Set backgate
        kmc.constant_energy()

        kmc.simulate_discrete(hops)
        current[i, j] = -kmc.current[0]
    
#%% Visualize
plt.figure()

#plt.plot(bias, current[0])

plt.matshow(current, origin='lower', extent=(bias[0], bias[-1], backgate[0], backgate[-1]))
cbar = plt.colorbar()
cbar.set_label('Current')
plt.xlabel('$V_{SD}$')
plt.ylabel('$V_{G}$')

plt.show()
