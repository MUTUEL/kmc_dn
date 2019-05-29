'''
This file performs an extended version of the boltzmann validation experiment.
To quickly check this, use the kmc_dn_utils method validate_boltzmann with
keyword argument standalone = True.
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils

#%% Parameters
N = 10  # Number of acceptors
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
mu = 1  # Chemical potential
n = 5  # Amount of carriers
hops = int(1E4)  # Hops per validation run
points = 1000  # Amount of points for plotting convergence
avg = 10  # Amount of validation runs

#%% Initialize simulation object
kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, mu = mu)

#%% Run validation

# Prerun validation function once for compilation
(E_microstates,
 p_theory,
 hops_array,
 p_sim_interval) = kmc_dn_utils.validate_boltzmann(kmc,
                                                   hops = 10,
                                                   n = n,
                                                   points = 1,
                                                   mu = mu,
                                                   standalone = False)
p_sim = np.zeros((avg, p_theory.shape[0], points))
# Actual validation loop
for i in range(avg):
    tic = time.time()
    (E_microstates,
     p_theory,
     hops_array,
     p_sim[i]) = kmc_dn_utils.validate_boltzmann(kmc,
                                                 hops = hops,
                                                 n = n,
                                                 points = points,
                                                 mu = mu,
                                                 standalone = False)
    toc = time.time()
    print('Elapsed time for ' + str(hops) + ': ' + str(toc-tic) + ' seconds')

#%% Calculations 
sigma = np.std(p_sim[:, :, -1], axis = 0)
p_sim = np.average(p_sim[:, :, -1], axis = 0)

# Check how many are within 2*sigma
within_error = 0
for i in range(p_sim.shape[0]):
    if(p_theory[i] - 2*sigma[i] < p_sim[i] < p_theory[i] + 2*sigma[i]):
        within_error += 1
within_error /= p_sim.shape[0]


#%% Plotting

# Simulated probabilities with error
indices = np.argsort(E_microstates)
plt.figure()
plt.errorbar(E_microstates, p_sim, yerr=2*sigma, fmt='o')
plt.plot(E_microstates[indices], p_theory[indices], 'r-')
plt.title(f'{within_error*100:.4}% within 2*$\sigma$.')

plt.show()
