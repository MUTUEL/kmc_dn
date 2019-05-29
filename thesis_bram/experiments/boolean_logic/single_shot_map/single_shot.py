import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_single_shot as config
import time

cf = config.experiment_config()

saveDirectory = SaveLib.createSaveDirectory(cf.filepath, cf.name)

#%% System setup
xdim = 1
ydim = 1

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
kmc.load_acceptors(acceptor_layouts[cf.layout])
kmc.load_donors(donor_layouts[cf.layout])

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
kmc.electrodes[cf.output] = 0

for index, control in enumerate(cf.controls):
    kmc.electrodes[control, 3] = (1-cf.gene[index])*cf.controlrange[0] \
                                 + cf.gene[index]*cf.controlrange[1]

# Obtain device response
output = np.zeros((cf.n_voltage, cf.n_voltage, 8, cf.avg))
for ii in range(cf.n_voltage):
    for jj in range(cf.n_voltage):
        # Time estimate
        if(ii == 0 and jj == 0):
            starttime = time.time()
        if(ii == 0 and jj == 1):
            eta = (time.time() - starttime)*cf.avg*cf.n_voltage**2
            print(f'Estimated time remaining: {eta} s, or {eta/3600} h')
        # Set input voltages
        kmc.electrodes[cf.P] = cf.inputaxis[ii]
        kmc.electrodes[cf.Q] = cf.inputaxis[jj]
        kmc.update_V()

        # Prestabilize the system
        kmc.python_simulation(prehops = cf.prehops)

        for kk in range(cf.avg):
            kmc.python_simulation(hops = cf.hops)
            output[ii, jj, :, kk] = kmc.current

SaveLib.saveExperiment(saveDirectory, 
                       output = output,
                       inputaxis = cf.inputaxis)

print('All done!')
