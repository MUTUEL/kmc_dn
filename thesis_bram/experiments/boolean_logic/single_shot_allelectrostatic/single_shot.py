import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_single_shot as config


cf = config.experiment_config()

saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

#%% System setup
xdim = 1
ydim = 1
layout = cf.layout

# Load layouts
acceptor_layouts = np.load('acceptor_layouts.npy')
donor_layouts = np.load('donor_layouts.npy')

# Define 8 electrodes
electrodes = np.zeros((1 + len(cf.controls), 4))
electrodes[0] = cf.electrode_positions[cf.output]
for ii in range(len(cf.controls)):
    electrodes[1 + ii] = cf.electrode_positions[cf.controls[ii]]

static_electrodes = np.zeros((2 + len(cf.static_controls), 4))
static_electrodes[0] = cf.electrode_positions[cf.P]
static_electrodes[1] = cf.electrode_positions[cf.Q]
for ii in range(len(cf.static_controls)):
    static_electrodes[2 + ii] = cf.electrode_positions[cf.static_controls[ii]]

kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, 
        electrodes=electrodes, static_electrodes=static_electrodes)
kmc.load_acceptors(acceptor_layouts[layout])
kmc.load_donors(donor_layouts[layout])

# Set parameters
kmc.kT = cf.kT
kmc.I_0 = cf.I_0
kmc.ab = cf.ab_R*kmc.R

# Update constant quantities that depend on I_0 and ab
kmc.calc_E_constant()
kmc.calc_transitions_constant()

# Define input signals
P = [0, 1, 0, 1]
Q = [0, 0, 1, 1]
w = [1, 1, 1, 1]


# Set the Voltages
# first genomes are for injecting controls, then static controls,
# last is for inputscaling
for k in range(len(cf.controls)):
    kmc.electrodes[1 + k, 3] = cf.controlrange[0]*(1 - cf.gene[k]) \
                               + cf.gene[k]*cf.controlrange[1]
for k in range(len(cf.static_controls)):
    kmc.static_electrodes[2 + k, 3] = cf.controlrange[0]*(1 - cf.gene[k + len(cf.controls)]) \
                               + cf.gene[k + len(cf.controls)]*cf.controlrange[1]

# Obtain device response
output = np.zeros((2, 4*cf.avg))
for k in range(4):
    if(cf.evolve_input):
        kmc.static_electrodes[0, 3] = P[k]*(cf.inputrange[0]*(1 - cf.gene[-1]) + cf.gene[-1]*cf.inputrange[1])
        kmc.static_electrodes[1, 3] = Q[k]*(cf.inputrange[0]*(1 - cf.gene[-1]) + cf.gene[-1]*cf.inputrange[1])
    else:
        kmc.static_electrodes[0, 3] = P[k]*cf.inputrange
        kmc.static_electrodes[1, 3] = Q[k]*cf.inputrange

    kmc.update_V()
    if(cf.use_go):
        kmc.go_simulation(hops=0, prehops = cf.prehops,
                          goSpecificFunction='wrapperSimulateRecord')
    else:
        kmc.python_simulation(prehops = cf.prehops)

    for l in range(cf.avg):
        if(cf.use_go):
            kmc.go_simulation(hops=cf.hops,
                              goSpecificFunction='wrapperSimulateRecord')
        else:
            kmc.python_simulation(hops = cf.hops)
        output[:, k*cf.avg + l] = kmc.current

# Save experiment
SaveLib.saveExperiment(saveDirectory, 
                       output = output)

print('All done!')
