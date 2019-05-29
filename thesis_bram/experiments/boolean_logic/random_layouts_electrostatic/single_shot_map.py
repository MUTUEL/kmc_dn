import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_boolean_logic as config

cf = config.experiment_config()
#%% System setup
xdim = 1
ydim = 1
layout = 0
# Load layouts
acceptor_layouts = np.load('acceptor_layouts.npy')
donor_layouts = np.load('donor_layouts.npy')

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

kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes)
kmc.load_acceptors(acceptor_layouts[layout])
kmc.load_donors(donor_layouts[layout])

# Parameters
gene = [0.5884281 , 0.49808138, 0.55724021, 0.68831492, 0.56113931]
avg = 1
output_electrode = 2
kT = 1
I_0 = 50*kT
V_high = 500*kT
ab_R = 0.25
prehops = 1000
hops = 10000
res = 10

kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R

# Define input signals
P = [0, 1, 0, 1]
Q = [0, 0, 1, 1]
w = [1, 1, 1, 1]
kmc.electrodes[output_electrode] = 0
P_V = np.linspace(-2*V_high, 2*V_high, res)
Q_V = np.linspace(-2*V_high, 2*V_high, res)
outputmap = np.zeros((res, res))


for index, control in enumerate(cf.controls):
    kmc.electrodes[control, 3] = -V_high + 2*V_high*gene[index]

# Obtain device response
output = np.zeros(4*avg)
for mapi in range(res):
    for mapj in range(res):

        kmc.electrodes[cf.P] = P_V[mapi]
        kmc.electrodes[cf.Q] = Q_V[mapj]
        kmc.update_V()
        for l in range(avg):
            kmc.simulate_discrete(prehops = prehops)
            kmc.simulate_discrete(hops = hops)
            output = kmc.current[cf.output]
        outputmap[mapi, mapj] = output

kmc_dn_utils.visualize_basic(kmc)
plt.figure()
plt.imshow(outputmap)
plt.show()

