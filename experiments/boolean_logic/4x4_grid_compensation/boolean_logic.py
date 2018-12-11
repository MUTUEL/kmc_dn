import SkyNEt.modules.Evolution as Evolution
import SkyNEt.modules.SaveLib as SaveLib
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import config_boolean_logic as config


cf = config.experiment_config()

genePool = Evolution.GenePool(cf)

#%% System setup
xdim = 1
ydim = 1
# Make acceptor grid
acceptors = np.zeros((16, 3))
for i in range(4):
    for j in range(4):
        acceptors[4*i + j] = [(i+1)*xdim/5, (j+1)*ydim/5, 0]
donors = np.zeros((2, 3))
donors[0] = [0.25, 0.5, 0]
donors[1] = [0.75, 0.5, 0]

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
kmc.load_donors(donors)

# Parameters
kT = 1
I_0 = 30*kT
V_high = 500*kT
ab_R = 0.30
prehops = 1000
hops = 10000

kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R

# Define input signals
P = [0, 1, 0, 1]
Q = [0, 0, 1, 1]
w = [1, 1, 1, 1]
kmc.electrodes[cf.output] = 0

# Initialize arrays to save experiment
geneArray = np.zeros((cf.generations, cf.genomes, cf.genes))
outputArray = np.zeros((cf.generations, cf.genomes, 4*cf.avg))
fitnessArray = np.zeros((cf.generations, cf.genomes))
filepath = SaveLib.createSaveDirectory(cf.filepath, cf.name)

for i in range(cf.generations):
    geneArray[i] = genePool.pool
    for j in range(cf.genomes):
        # Set the Voltages
        for k in cf.controls:
            kmc.electrodes[k, 3] = -V_high + 2*genePool.pool[j, k-3]*V_high
        
        fitness_list = []
        # Obtain device response
        output = np.zeros(4*cf.avg)
        for k in range(4):
            kmc.electrodes[cf.P] = P[k]*V_high*2
            kmc.electrodes[cf.Q] = Q[k]*V_high*2
            kmc.update_V()
            kmc.simulate_discrete(prehops = prehops)
            for l in range(cf.avg):
                kmc.simulate_discrete(hops = hops)
                output[k*cf.avg + l] = kmc.current[cf.output]
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
