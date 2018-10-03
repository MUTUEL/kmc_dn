# Make a nice convergence plot for Boltzmann validation

import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% Parameters

N = 10  # Number of acceptors
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
init = True
 
#%% Initialize simulation object

if(init):
    kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim)
    
    # Fix parameters such that all dimensionless quantities are 1
    N_sites = kmc.acceptors.shape[0]/(xdim*ydim)  # Site concentration
    R = N_sites**(-1/2)
    kmc.ab = R  #Fix ab/R = 1
    kmc.eps = 0.5*kmc.e**2/(4*np.pi*R*kmc.k*kmc.T)  # Fix I_0/kT = 1
    V_0 = kmc.k*kmc.T/kmc.e  # Fix eV_0/kT = 1
    
    # Save object 
    output = open('boltzmann_validation_obj.pkl', 'wb')
    pickle.dump(kmc, output, -1)

