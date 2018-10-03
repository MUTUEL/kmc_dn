# Make a nice convergence plot for Boltzmann validation
#%% imports

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import kmc_dopant_networks

#%% Import object and visualize

input = open('boltzmann_validation_obj.pkl', 'rb')
kmc = pickle.load(input)

#%% Do validation for n = 5

V_0 =  kmc.k*kmc.T/kmc.e  # Fix eV_0/kT = 1
n = 5
hops = int(1E6)
points = 1000
avg = 10


for i in range(avg):
    tic = time.time()
    h, p = kmc.validate_boltzmann(hops=hops, n=n, points = points, V_0 = V_0)
    toc = time.time()
    print('Elapsed time for ' + str(hops) + ': ' + str(toc-tic) + ' seconds')
    np.save('harst' + str(i) + '.npy', h)
    np.save('pstrs' + str(i) + '.npy', p)

#%% Calculations on p
#s = np.zeros((p.shape[0], p.shape[1]-window))
#for i in range(window, len(h)):
#    # Calculate std on current window
#    s[:, i-window] = np.std(p[:, i-window:i], axis=1)
    
#s_sim = np.zeros(p.shape[0])
#p_sim = np.zeros(p.shape[0])
#for i in range(s_sim.shape[0]):
#    s_sim[i] = np.std(p[i])
#    p_sim[i] = np.mean(p[i])
#    
#norm = np.zeros(p.shape[1])
#for i in range(norm.shape[0]):
#    norm[i] = np.linalg.norm(kmc.p - p[:, i])/np.linalg.norm(kmc.p) 
#    
# Calculate sigma on final p
sigma = np.std(p[:, :, -1], axis = 0)
p_sim = np.average(p[:, :, -1], axis = 0)
#%% Various plots
    
#i = 1
#plt.figure()
#plt.plot(h, p[i])
#plt.fill_between(h[window:], p[i, window:] + 2*s[i], p[i, window:] - 2*s[i], alpha=0.5, color = 'r')
#plt.hlines(kmc.p[i], h[0], h[-1], linestyle = '--')

# Theoretical probabilities
#plt.figure()
#plt.plot(kmc.E_microstates, kmc.p, '.')
#plt.xlabel('Microstate energy (kT)')
#plt.ylabel('Probability')

# Simulated probabilities with error 
indices = np.argsort(kmc.E_microstates)
plt.figure()
plt.errorbar(kmc.E_microstates, p_sim, yerr=2*sigma, fmt='o')
plt.plot(kmc.E_microstates[indices], kmc.p[indices], 'r-')

# Full norm plot
#plt.figure()
#plt.loglog(h, norm)
#
plt.show()