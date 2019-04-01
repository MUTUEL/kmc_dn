import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_boolean_logic as config
import time


cf = config.experiment_config()

genePool = Evolution.GenePool(cf)

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

kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes, static_electrodes=static_electrodes)
kmc.load_acceptors(acceptor_layouts[layout])
kmc.load_donors(donor_layouts[layout])

# Parameters
kT = cf.kT
I_0 = cf.I_0
ab_R = cf.ab_R
prehops = cf.prehops
hops = cf.hops

kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R

# Update constant quantities that depend on I_0 and ab
kmc.calc_E_constant()
kmc.calc_transitions_constant()

# Define input signals
P = [0, 1, 0, 1]
Q = [0, 0, 1, 1]
w = [1, 1, 1, 1]
kmc.electrodes[0, 3] = 0

# Initialize arrays to save experiment
geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
outputArray = np.zeros((cf.generations, cf.genomes, 4*cf.avg))
fitnessArray = np.zeros((cf.generations, cf.genomes))
filepath = SaveLib.createSaveDirectory(cf.filepath, cf.name)
# Save experiment (also copies files)
SaveLib.saveExperiment(filepath,
                       geneArray = geneArray,
                       fitnessArray = fitnessArray,
                       outputArray = outputArray,
                       target = cf.target,
                       P = P,
                       Q = Q)

for i in range(cf.generations):
    geneArray[i] = genePool.pool
    for j in range(cf.genomes):
        # Time estimate
        if(i == 0 and j == 1):
            tic = time.time()
        if(i == 0 and j == 2):
            print(f'Estimated time: {(time.time()-tic)*cf.generations*cf.genomes/3600} h')

        # Set the Voltages
        # first genomes are for injecting controls, then static controls,
        # last is for inputscaling
        for k in range(len(cf.controls)):
            kmc.electrodes[1 + k, 3] = cf.controlrange[0]*(1 - genePool.pool[j, k]) \
                                       + genePool.pool[j, k]*cf.controlrange[1]
        for k in range(len(cf.static_controls)):
            kmc.static_electrodes[2 + k, 3] = cf.controlrange[0]*(1 - genePool.pool[j, k + len(cf.controls)]) \
                                       + genePool.pool[j, k + len(cf.controls)]*cf.controlrange[1]

        fitness_list = []
        # Obtain device response
        output = np.zeros(4*cf.avg)
        for k in range(4):
            if(cf.evolve_input):
                kmc.static_electrodes[0, 3] = P[k]*(cf.inputrange[0]*(1 - genePool.pool[j, -1]) + genePool.pool[j, -1]*cf.inputrange[1])
                kmc.static_electrodes[1, 3] = Q[k]*(cf.inputrange[0]*(1 - genePool.pool[j, -1]) + genePool.pool[j, -1]*cf.inputrange[1])
            else:
                kmc.static_electrodes[0, 3] = P[k]*cf.inputrange
                kmc.static_electrodes[1, 3] = Q[k]*cf.inputrange

            kmc.update_V()
            kmc.python_simulation(prehops = prehops)
            for l in range(cf.avg):
                kmc.python_simulation(hops = hops)
                output[k*cf.avg + l] = kmc.current[0]
        outputArray[i, j] = output
        fitness_list.append(cf.Fitness(output, cf.target))

        genePool.fitness[j] = min(fitness_list)
        fitnessArray[i, j] = genePool.fitness[j]
    # Status print
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # Evolve to the next generation
    genePool.NextGen()

    # Save experiment
    SaveLib.saveArrays(filepath,
                       geneArray = geneArray,
                       fitnessArray = fitnessArray,
                       outputArray = outputArray,
                       target = cf.target,
                       P = P,
                       Q = Q)

print('All done!')
