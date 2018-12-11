import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_boolean_logic as config


cf = config.experiment_config()

genePool = Evolution.GenePool(cf)

avg = 5
#%% System setup
xdim = 1
ydim = 1
# Make acceptor grid
acceptors = np.zeros((16, 3))
for i in range(4):
    for j in range(4):
        acceptors[4*i + j] = [(i+1)*xdim/5, (j+1)*ydim/5, 0]

# Define 8 electrodes
electrodes = np.zeros((8, 4))
electrodes[0] = [0, ydim/4, 0, 10]
electrodes[1] = [0, 3*ydim/4, 0, 0]
electrodes[2] = [xdim, ydim/4, 0, 10]
electrodes[3] = [xdim, 3*ydim/4, 0, 0]
electrodes[4] = [xdim/4, 0, 0, 10]
electrodes[5] = [3*xdim/4, 0, 0, 0]
electrodes[6] = [xdim/4, ydim, 0, 10]
electrodes[7] = [3*xdim/4, ydim, 0, 0]

kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes)
kmc.load_acceptors(acceptors)

# Parameters
kT = 1
I_0 = 50*kT
V_high = 100*kT
ab_R = 0.25

kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R

# Define target
P = [0, 1, 0, 1]
Q = [0, 0, 1, 1]
w = [1, 1, 1, 1]
kmc.electrodes[cf.output] = 0

# Initialize arrays to save experiment
geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
outputArray = np.zeros((cf.generations, cf.genomes, 4))
fitnessArray = np.zeros((cf.generations, cf.genomes))

for i in range(cf.generations):
    geneArray[i] = genePool.pool
    for j in range(cf.genomes):
        # Set the Voltages
        for k in cf.controls:
            kmc.electrodes[k, 3] = -V_high + 2*genePool.pool[j, k-3]*V_high
        
        fitness = 0
        for k in range(avg):
            # Obtain device response
            output = np.zeros(4)
            for k in range(4):
                kmc.electrodes[cf.P] = P[k]*V_high
                kmc.electrodes[cf.Q] = Q[k]*V_high
                kmc.update_V()
                kmc.simulate_discrete(prehops = 500, hops = 5000)
                output[k] = kmc.current[cf.output]
            outputArray[i, j] = output
            fitness += cf.Fitness(output, cf.target)

        genePool.fitness[j] = fitness/avg
        fitnessArray[i, j] = genePool.fitness[j]
    # Status print
    print("Generation nr. " + str(i + 1) + " completed")
    print("Highest fitness: " + str(max(genePool.fitness)))

    # Evolve to the next generation
    genePool.NextGen()

# Save experiment
filepath = SaveLib.createSaveDirectory(cf.filepath, cf.name)
SaveLib.saveExperiment(filepath, 
                       geneArray = geneArray,
                       fitnessArray = fitnessArray,
                       outputArray = outputArray,
                       target = cf.target,
                       P = P,
                       Q = Q)

domain = kmc_dn_utils.visualize_basic(kmc)
plt.figure()
plt.plot(output)
plt.show()
