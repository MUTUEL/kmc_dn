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

# Load layouts and target
acceptor_layouts = np.load('acceptor_layouts.npy')
donor_layouts = np.load('donor_layouts.npy')
IV_data = np.load('IV_fit.npz')
voltagelist = IV_data['input_kmc']
target = IV_data['output_kmc']

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

# Initialize arrays to save experiment
geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
outputArray = np.zeros((cf.generations, cf.genomes, len(voltagelist)))
fitnessArray = np.zeros((cf.generations, cf.genomes))
filepath = SaveLib.createSaveDirectory(cf.filepath, cf.name)

# Save experiment (also copies files)
SaveLib.saveExperiment(filepath,
                       geneArray = geneArray,
                       fitnessArray = fitnessArray,
                       outputArray = outputArray,
                       voltagelist = voltagelist,
                       target = target,
                       )

for i in range(cf.generations):
    geneArray[i] = genePool.pool
    for j in range(cf.genomes):
        # Time estimation
        if(i == 0 and j == 1):
            tic = time.time()
        if(i == 0 and j == 2):
            print(f'Estimated time: {(time.time()-tic)*cf.generations*cf.genomes/3600} h')

        # Set I_0 and ab
        kmc.I_0 = cf.generange[0][0] + genePool.pool[j, 0] \
                  * (cf.generange[0][1] - cf.generange[0][0])
        kmc.ab = cf.generange[1][0] + genePool.pool[j, 1] \
                  * (cf.generange[1][1] - cf.generange[1][0])
        kmc.ab = kmc.ab*kmc.R

        # Update constant quantities that depend on I_0 and ab
        kmc.calc_E_constant()
        kmc.calc_transitions_constant()

        fitness_list = []
        # Obtain device response
        for l in range(cf.avg):
            current_array = kmc_dn_utils.IV(kmc, 0, voltagelist,
                                            prehops = prehops,
                                            hops = hops)
            output = current_array[1]
        outputArray[i, j] = output
        fitness_list.append(cf.Fitness(output, target))

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
                       voltagelist = voltagelist,
                       target = target,
                       )

print('All done!')

