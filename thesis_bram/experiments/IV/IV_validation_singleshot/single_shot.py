import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_single_shot as config
import time

cf = config.experiment_config()

genePool = Evolution.GenePool(cf)

#%% System setup
xdim = 1
ydim = 1
layout = cf.layout

# Load layouts and target
acceptor_layouts = np.load('acceptor_layouts.npy')
donor_layouts = np.load('donor_layouts.npy')
IV_data = np.load('IV_fit.npz')
voltagelist = IV_data['input_kmc']
target = IV_data['output_kmc']

# Optionally use voltagelist defined in config
if(cf.useTarget == False):
    voltagelist = cf.voltagelist

# Define 2 electrodes
electrodes = np.zeros((2, 4))
electrodes[0] = [0, ydim/2, 0, 0]
electrodes[1] = [1, ydim/2, 0, 0]

# Initialize kmc
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

# Initialize arrays to save experiment
filepath = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Initialize starting occupation
kmc.place_charges_random()
start_occupation = kmc.occupation.copy()

# seed random generator
np.random.seed(1)

# Obtain device response
currents = kmc_dn_utils.IV(kmc, 0, voltagelist,
                                prehops = prehops,
                                hops = hops,
                                printETA = True)
output = currents[1].copy()

# Find approximate nu
nu = target[0]/output[0]
print(nu)

# Scale output by nu
output = cf.nu*output

# Save experiment
SaveLib.saveArrays(filepath,
                   output = output,
                   voltagelist = voltagelist,
                   target = target,
                   )

# Plot IV curve
plt.figure()
plt.plot(voltagelist, output, 'x', label='1')
plt.plot(voltagelist, target, '-x', label='1')
plt.show()

print('All done!')

