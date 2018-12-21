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
layout = 4
# Load layouts
acceptor_layouts = np.load('acceptor_layouts.npy')
donor_layouts = np.load('donor_layouts.npy')

# Define 8 electrodes
electrodes = np.zeros((8, 4))
electrodes[0] = [0, ydim/4, 0, 0]
electrodes[1] = [0, 3*ydim/4, 0, 0]
electrodes[2] = [xdim, ydim/4, 0, 0]
electrodes[3] = [xdim, 3*ydim/4, 0, 0]
electrodes[4] = [xdim/4, 0, 0, 0]
electrodes[5] = [3*xdim/4, 0, 0, 0]
electrodes[6] = [xdim/4, ydim, 0, 0]
electrodes[7] = [3*xdim/4, ydim, 0, 0]

kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes)
kmc.load_acceptors(acceptor_layouts[layout])
kmc.load_donors(donor_layouts[layout])

# Parameters
gene = [0.75945677, 1.,         0.5860348,  0.99585631, 0.74258337]
avg = 1
output_electrode = 2
kT = 1
I_0 = cf.I_0
V_high = 1000*kT
ab_R = cf.ab_R
prehops = 10000
hops = 1000000

kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R

# Define input signals
P = [0, 1, 0, 1]
Q = [0, 0, 1, 1]
w = [1, 1, 1, 1]
kmc.electrodes[output_electrode, 3] = 0


for index, control in enumerate(cf.controls):
    kmc.electrodes[control, 3] = -V_high + 2*V_high*gene[index]

# Obtain device response
output = np.zeros(4*avg)
for k in range(4):
    kmc.electrodes[cf.P, 3] = P[k]*V_high
    kmc.electrodes[cf.Q, 3] = Q[k]*V_high
    kmc.update_V()
    kmc.simulate_discrete(prehops = prehops)
    for l in range(avg):
        kmc.simulate_discrete(hops = hops)
        output[k*avg + l] = kmc.current[cf.output]

plt.figure()
plt.plot(output)
plt.show()

