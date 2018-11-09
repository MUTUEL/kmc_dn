'''
This script performs a 'single atom transisitor' experiment, for validation.
In other words, a system with two electrodes and one acceptor exactly in between.
Theoretical treatment can be found in ?my thesis?
#TODO: make nice bias vs. backgate and temperature maps.
'''
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_ut
import numpy as np
import matplotlib.pyplot as plt

#%% Parameters
N = 1  # Number of acceptors
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
mu = 0  # Chemical potential
hops = int(1E4)  # Hops per validation run

electrodes = np.zeros((2, 4))
electrodes[0] = [0, ydim/2, 0, 10]
electrodes[1] = [xdim, ydim/2, 0, 0]

#%% Initialize simulation object
kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, mu = mu, electrodes = electrodes)

# Get V_0 for which eV_0 = kT
V_0 = kmc.kT/kmc.e

# Set ab >> R, such that distance does not matter
kmc.ab = 1000*kmc.R

# Place single atom in middle of domain
acceptors = np.array([[xdim/2, ydim/2, 0]])
kmc.load_acceptors(acceptors)



#%% Current simulation with backgate

# Voltages
bias = np.linspace(-100*V_0, 100*V_0, num = 101)

# IV measurement
currentlist = kmc_dn_ut.IV(kmc, 0, bias)


#%% Visualize

plt.plot(bias, currentlist[:, -1])
#import matplotlib.colors as colors
#import matplotlib.cm as cmx
#plt.figure()
#
#jet = plt.get_cmap('plasma')
#cNorm = colors.LogNorm(vmin=temperature[0], vmax=temperature[-1])
#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#
#for i in range(temperature.shape[0]):
#    colorVal = scalarMap.to_rgba(temperature[i])
#    plt.plot(bias, current[i], color=colorVal)
#
#plt.vlines(V_0, np.min(current), np.max(current))
#
#plt.xlabel('$V_{SD}$')
#plt.ylabel('Current ($v_0$)')
#
#scalarMap.set_array(temperature)
#cbar = plt.colorbar(scalarMap)
#cbar.set_label('Temperature ($eV_0$)')
#
plt.show()
