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
electrodes[0] = [0, ydim/2, 0, 0]
electrodes[1] = [1, ydim/2, 0, 0]

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

# Obtain device response
output = np.zeros((len(cf.kT), 2, len(cf.voltages), cf.avg))
for ii in range(len(cf.kT)):
    # Timing
    if(ii==0):
        starttime = time.time()
    if(ii==1):
        dt = time.time() - starttime
        print(f'Estimated remaining time: {dt*len(cf.kT)} s, or \
                 {dt*len(cf.kT)/3600} h'}
    # Set the temperature
    kmc.kT = cf.kT[ii]

    # Update constant quantities that depend on I_0 and ab
    kmc.calc_E_constant()
    kmc.calc_transitions_constant()

    for jj in range(len(cf.voltages)):
        # Set voltage
        kmc.electrodes[0] = cf.voltages[jj]
        kmc.update_V()

        # Prestabilize the system
        kmc.python_simulation(prehops = cf.prehops)

        # Get current
        for kk in range(cf.avg):
            kmc.python_simulation(hops = cf.hops)
            output[ii, :, jj, kk] = kmc.current

SaveLib.saveExperiment(saveDirectory, 
                       kT = cf.kT,
                       output = output,
                       voltages = cf.voltages)

print('All done!')
