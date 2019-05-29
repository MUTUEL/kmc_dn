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
electrodes = np.zeros((3, 4))
electrodes[0] = [0, ydim/4, 0, 0]
electrodes[1] = [0, 3*ydim/4, 0, 0]
electrodes[2] = [xdim, ydim/4, 0, 0]

static_electrodes = np.zeros((5, 4))
static_electrodes[0] = [xdim, 3*ydim/4, 0, 0]
static_electrodes[1] = [xdim/4, 0, 0, 0]
static_electrodes[2] = [3*xdim/4, 0, 0, 0]
static_electrodes[3] = [xdim/4, ydim, 0, 0]
static_electrodes[4] = [3*xdim/4, ydim, 0, 0]

kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes, static_electrodes=static_electrodes)
kmc.load_acceptors(acceptor_layouts[layout])
kmc.load_donors(donor_layouts[layout])

# Parameters
gene = [0.47507716, 0.3606721,  0.47602667, 0.25694824, 0.46620197, 0.07907845]
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
    kmc.static_electrodes[control, 3] = cf.controlrange[0]*(1 - gene[index]) + gene[index]*cf.controlrange[1]

# Obtain device response
output = np.zeros(4*avg)
for k in range(4):
    kmc.electrodes[cf.P, 3] = P[k]*(cf.inputrange[0]*(1 - gene[-1]) + gene[-1]*cf.inputrange[1])
    kmc.electrodes[cf.Q, 3] = Q[k]*(cf.inputrange[0]*(1 - gene[-1]) + gene[-1]*cf.inputrange[1])
    kmc.update_V()
    kmc.simulate_discrete(prehops = prehops)
    for l in range(avg):
        kmc.simulate_discrete(hops = hops)
        output[k*avg + l] = kmc.current[cf.output]

plt.figure()
plt.plot(output)
plt.show()

