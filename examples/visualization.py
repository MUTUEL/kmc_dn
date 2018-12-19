import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt

#%% System setup
xdim = 1
ydim = 1
N = 30
M = 3

# Define 8 electrodes
electrodes = np.zeros((8, 4))
electrodes[0] = [0, ydim/4, 0, 0]
electrodes[1] = [0, 3*ydim/4, 0, 0]
electrodes[2] = [xdim, ydim/4, 0, 10]
electrodes[3] = [xdim, 3*ydim/4, 0, 0]
electrodes[4] = [xdim/4, 0, 0, 0]
electrodes[5] = [3*xdim/4, 0, 0, 0]
electrodes[6] = [xdim/4, ydim, 0, 0]
electrodes[7] = [3*xdim/4, ydim, 0, 0]

kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, 0, electrodes=electrodes, 
                    callback_traffic=True)

# Parameters
gene = [0.3730247,  0.94274381, 0.49073662, 0.89030952, 0.30259412]
output_electrode = 2
kT = 1
I_0 = 50*kT
V_high = 500*kT
ab_R = 0.25
prehops = 10000
hops = 100000

kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R

# Define input signals
controls = [3, 4, 5, 6, 7]
kmc.electrodes[output_electrode, 3] = 0

for index, control in enumerate(controls):
    kmc.electrodes[control, 3] = -V_high + 2*V_high*gene[index]

# Obtain device response
kmc.electrodes[0, 3] = V_high
kmc.electrodes[1, 3] = V_high
kmc.update_V()
kmc.simulate_discrete(prehops = prehops)
kmc.simulate_discrete(hops = hops)

kmc_dn_utils.visualize_current_density(kmc, res=0.33)
plt.show()

