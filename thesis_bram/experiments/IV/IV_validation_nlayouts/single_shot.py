import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_single_shot as config
import time
import os

cf = config.experiment_config()

genePool = Evolution.GenePool(cf)

#%% System setup
xdim = 1
ydim = 1

# Initialize data array
outputs = np.zeros((len(cf.layouts), len(cf.voltagelist)))

# Load layouts and target
acceptor_layouts = np.load('acceptor_layouts.npy')
donor_layouts = np.load('donor_layouts.npy')
voltagelist = cf.voltagelist

# Define 2 electrodes
electrodes = np.zeros((2, 4))
electrodes[0] = [0, ydim/2, 0, 0]
electrodes[1] = [1, ydim/2, 0, 0]

# Initialize arrays to save experiment
filepath = SaveLib.createSaveDirectory(cf.filepath, cf.name)
os.makedirs(filepath + '/kmc_objects')

for ii in range(len(cf.layouts)):
    layout = cf.layouts[ii]
    # Initialize kmc
    if(cf.use_random_layout):
        kmc = kmc_dn.kmc_dn(cf.layoutsize, 
            cf.layoutsize//10, xdim, ydim, 0, electrodes=electrodes)
    else:
        kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes)
        kmc.load_acceptors(acceptor_layouts[layout])
        kmc.load_donors(donor_layouts[layout])

    # Parameters
    kT = cf.kT
    prehops = cf.prehops
    hops = cf.hops
    kmc.kT = kT
    kmc.I_0 = cf.I_0
    kmc.ab = cf.ab_R*kmc.R

    # Update constant quantities that depend on I_0 and ab
    kmc.calc_E_constant()
    kmc.calc_transitions_constant()

    # Initialize starting occupation
    kmc.place_charges_random()
    start_occupation = kmc.occupation.copy()

    # Obtain device response
    currents = kmc_dn_utils.IV(kmc, cf.bias_electrode, voltagelist,
                                    prehops = prehops,
                                    hops = hops,
                                    printETA = True)
    output = currents[1].copy()

    # Save output and kmc object
    outputs[ii] = output
    kmc.saveSelf(filepath + '/kmc_objects/' + '{0:03}'.format(ii) + '.kmc')

# Save experiment
SaveLib.saveExperiment(filepath,
                   outputs = outputs,
                   voltagelist = voltagelist,
                   )


print('All done!')

