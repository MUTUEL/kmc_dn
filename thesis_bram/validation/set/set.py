'''
This script performs a 'single atom transisitor' experiment, for validation.
In other words, a system with two electrodes and one acceptor exactly in between.
Theoretical treatment can be found in my thesis
#TODO: make nice bias vs. backgate and temperature maps.
'''

import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_ut
import numpy as np
import matplotlib.pyplot as plt

def exp_tresh(x):
    '''
    Exp tresholded at 1
    '''
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= 0:
            y[i] = np.exp(x[i])
        else:
            y[i] = 1
    return y

def analytical(U_S, U_G, U_D, kT, nu_0 = 1):
    sg_forward = nu_0 * exp_tresh((U_S - U_G)/kT) 
    sg_backward = nu_0 * exp_tresh((U_G - U_S)/kT) 
    gd_forward = nu_0 * exp_tresh((U_G - U_D)/kT) 
    gd_backward = nu_0 * exp_tresh((U_D - U_G)/kT) 
    return sg_forward*gd_forward/(sg_forward + gd_forward)- sg_backward*gd_backward/(sg_backward + gd_backward)

#%% Parameters
N = 1  # Number of acceptors
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 0  # Length along y dimension
zdim = 0  # Length along z dimension
mu = 0  # Chemical potential
hops = int(1E5)  # Hops per validation run
avg = 10


electrodes = np.zeros((2, 4))
electrodes[0] = [0, 0, 0, 10]
electrodes[1] = [xdim, 0, 0, 0]

#%% Initialize simulation object
kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, mu = mu, electrodes = electrodes)

# Set U_0 = kmc.kT
U_0 = 0.1*kmc.kT


# Place single atom in middle of domain
acceptors = np.array([[xdim/2, 0, 0]])
kmc.load_acceptors(acceptors)

# Set ab >> R, such that distance does not matter
kmc.ab = 100000*kmc.R
kmc.initialize(V=False, dopant_placement = False)

#%% Current simulation with backgate

# Voltages
bias = np.linspace(-100*U_0, 100*U_0, num = 101)
current = np.zeros((avg, 2, len(bias)))

# IV measurement
for i in range(avg):
    current[i] = kmc_dn_ut.IV(kmc, 0, bias, hops = hops)

# Save
np.savez('data', bias = bias, current = current)
