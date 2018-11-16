'''
This file is a collection of utilities to be used together with the class
kmc_dn defined in kmc_dopant_networks.

#TODO: documentation
@author: Bram de Wilde (b.dewilde-1@student.utwente.nl)
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
import fenics as fn
import time

plt.ioff()

#%% Visualization functions

def visualize_basic(kmc_dn, show_occupancy = True, show_V = True):
    '''
    Returns a figure which shows the domain with potential profile. It
    also show all dopants with acceptor occupancy.
    Note: only 2D is supported
    '''
    if(kmc_dn.dim == 2):
        # Initialize figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(right=kmc_dn.xdim)
        ax.set_ylim(top=kmc_dn.ydim)

        # Extract potential profile from V
        x = np.arange(0, kmc_dn.xdim, kmc_dn.res)
        y = np.arange(0, kmc_dn.ydim, kmc_dn.res)
        V_plot = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                V_plot[i, j] = kmc_dn.V(x[i], y[j])


        # Plot potential profile
        if(show_V):
            V_profile = ax.imshow(V_plot.transpose(),
                                  interpolation='bicubic',
                                  origin='lower',
                                  extent=(0, kmc_dn.xdim, 0, kmc_dn.ydim))
            cbar = fig.colorbar(V_profile)


        if(show_occupancy):
            # Plot impurity configuration (red = 2, orange = 1, black = 0 holes)
            colors = ['red' if i==2
                      else 'orange' if i==1
                      else 'black' for i in kmc_dn.occupation]
            ax.scatter(kmc_dn.acceptors[:, 0], kmc_dn.acceptors[:, 1], c = colors, marker='o')

            ax.scatter(kmc_dn.donors[:, 0], kmc_dn.donors[:, 1], marker='x')
        else:
            ax.scatter(kmc_dn.acceptors[:, 0], kmc_dn.acceptors[:, 1], color = 'black', marker='o')

            ax.scatter(kmc_dn.donors[:, 0], kmc_dn.donors[:, 1], marker='x')

        ax.set_xlabel('x (a.u.)')
        ax.set_ylabel('y (a.u.)')
        if(show_occupancy):
            ax.set_title('Yellow = occupied, black = unoccupied')
        cbar.set_label('Chemical potential (V)')

    return fig

def visualize_current(kmc_dn):
    '''Returns a figure which shows the domain with potential profile and
    occupancy. Plot vectors which correlate to the net hop direction.'''

    fig = visualize_basic(kmc_dn, show_occupancy = False)
    # Calculate current_vectors from traffic
    current_vectors = np.zeros((kmc_dn.transitions.shape[0], 3))
    for i in range(kmc_dn.N):
        for j in range(kmc_dn.traffic.shape[0]):
            if(j is not i):
                # Arriving hops
                current_vectors[i] += kmc_dn.traffic[j, i] * kmc_dn.vectors[j, i]

                # Departing hops
                current_vectors[i] += kmc_dn.traffic[i, j] * kmc_dn.vectors[i, j]

    x = kmc_dn.acceptors[:, 0]
    y = kmc_dn.acceptors[:, 1]
    norm = np.linalg.norm(current_vectors[:kmc_dn.N], axis = 1)
    u = current_vectors[:kmc_dn.N, 0]/norm
    v = current_vectors[:kmc_dn.N, 1]/norm

    quiv = fig.axes[0].quiver(x, y, u, v, norm, cmap=plt.cm.inferno)
    fig.colorbar(quiv)

    return fig

def visualize_current_density(kmc_dn, res = None):
    '''
    Returns a figure of the domain where the colors indicate the current
    density and arrow the current direction at dopant sites. It uses only
    the during simulation tracked traffic array.
    res is the resolution in which the domain is split up.
    '''
    # Set resolution to V grid resolution if unspecified
    if(res == None):
        res = kmc_dn.res

    # Set up grid
    x = np.linspace(0, kmc_dn.xdim, kmc_dn.xdim/res + 1)
    y = np.linspace(0, kmc_dn.xdim, kmc_dn.xdim/res + 1)
    current_map = np.zeros((len(x) - 1, len(y) - 1))

    # Calculate current_vectors from traffic
    current_vectors = np.zeros((kmc_dn.transitions.shape[0], 3))
    for i in range(kmc_dn.N):
        for j in range(kmc_dn.traffic.shape[0]):
            if(j is not i):
                # Arriving hops
                current_vectors[i] += kmc_dn.traffic[j, i] * kmc_dn.vectors[j, i]

                # Departing hops
                current_vectors[i] += kmc_dn.traffic[i, j] * kmc_dn.vectors[i, j]

    norm = np.linalg.norm(current_vectors[:kmc_dn.N], axis = 1)  # Vector lengths

    for i in range(len(x)-1):
        for j in range(len(y)-1):
            # For each square in domain, loop over all acceptors
            indices_in_square = []
            for k in range(kmc_dn.N):
                if( x[i] <= kmc_dn.acceptors[k, 0] <= x[i+1]
                    and y[j] <= kmc_dn.acceptors[k, 1] <= y[j+1]):
                    current_map[i, j] += norm[k]

    # Plot current density
    fig = plt.figure()
    plt.axis('scaled')
    ax = fig.add_subplot(111)
    ax.set_xlim(right=kmc_dn.xdim)
    ax.set_ylim(top=kmc_dn.ydim)
    ax.imshow(current_map.transpose(), interpolation = 'none',
           origin='lower', extent=(0, kmc_dn.xdim, 0, kmc_dn.ydim), cmap=plt.cm.plasma)
    # Overlay dopants
    ax.scatter(kmc_dn.acceptors[:, 0], kmc_dn.acceptors[:, 1], color = 'black', marker='o')
    # Overlay dopant vectors
    x_dopants = kmc_dn.acceptors[:, 0]
    y_dopants = kmc_dn.acceptors[:, 1]
    u = current_vectors[:kmc_dn.N, 0]/norm
    v = current_vectors[:kmc_dn.N, 1]/norm
    ax.quiver(x_dopants, y_dopants, u, v, norm, cmap=plt.cm.inferno)
    #ax.quiver(x_dopants, y_dopants, u, v)

    return fig, current_map


def validate_boltzmann(kmc_dn, hops = 1000, n = 2, points = 100, mu = 1,
                       standalone = True):
    '''
    Perform validation of a simulation algorithm by means of checking
    whether it obeys boltzmann statistics. Hops is the total amount of hops
    performed and n equals the (constant!) number of carriers in the system.
    points; the amount of points in convergence array
    V_0; chemical potential
    '''
    kmc_dn.reset()
    # Initialize
    hops_array = np.zeros(points)
    interval = hops/points

    # Prepare system
    kmc_dn.electrodes = np.zeros((0, 4))  # Remove electrodes from system
    kmc_dn.P = 0

    # Initialize other attributes
    kmc_dn.transitions = np.zeros((kmc_dn.N + kmc_dn.electrodes.shape[0],
                                 kmc_dn.N + kmc_dn.electrodes.shape[0]))
    kmc_dn.transitions_constant = np.zeros((kmc_dn.N + kmc_dn.electrodes.shape[0],
                                 kmc_dn.N + kmc_dn.electrodes.shape[0]))
    kmc_dn.distances = np.zeros((kmc_dn.N + kmc_dn.electrodes.shape[0],
                               kmc_dn.N + kmc_dn.electrodes.shape[0]))
    kmc_dn.vectors = np.zeros((kmc_dn.N + kmc_dn.electrodes.shape[0],
                             kmc_dn.N + kmc_dn.electrodes.shape[0], 3))
    kmc_dn.site_energies = np.zeros((kmc_dn.N + kmc_dn.electrodes.shape[0],))
    kmc_dn.problist = np.zeros((kmc_dn.N+kmc_dn.P)**2)
    kmc_dn.electrode_occupation = np.zeros(kmc_dn.P, dtype=int)

    # Set chemical potential and initialize
    kmc_dn.mu = mu
    kmc_dn.initialize(placement = False)

    # Make microstate array
    perm_array = np.zeros((kmc_dn.N), dtype = bool)
    perm_array[:n] = True
    perms = list(itertools.permutations(perm_array))  # All possible permutations
    microstates = np.asarray(list(set(perms)))  # Remove all duplicate microstates

    # Assign energies to microstates
    E_microstates = np.zeros(microstates.shape[0])
    for i in range(E_microstates.shape[0]):
        kmc_dn.occupation = microstates[i]
        E_microstates[i] = kmc_dn.total_energy()

    # Calculate theoretical probabilities
    p_theory = np.zeros(microstates.shape[0])
    for i in range(p_theory.shape[0]):
        p_theory[i] = np.exp(-E_microstates[i]/(kmc_dn.kT))
    p_theory = p_theory/np.sum(p_theory)  # Normalize

    # Prepare simulation of probabilities
    kmc_dn.time = 0  # Reset simulation time
    previous_microstate = np.random.randint(microstates.shape[0])  # Random first microstate
    kmc_dn.occupation = microstates[previous_microstate].copy()
    p_sim = np.zeros(microstates.shape[0])
    p_sim_interval = np.zeros((microstates.shape[0], points))

    # Simulation loop
    interval_counter = 0
    for i in range(hops):
        # Hopping event
        kmc_dn.simulate_discrete(hops = 1, reset = False)

        # Save time spent in previous microstate
        p_sim[previous_microstate] += kmc_dn.hop_time

        # Find index of current microstate
        for j in range(microstates.shape[0]):
            if(np.array_equal(kmc_dn.occupation, microstates[j])):
               previous_microstate = j
               break

        # Save probabilities each interval
        if(i >= (interval_counter+1)* interval - 1):
            p_sim_interval[:, interval_counter] = p_sim/kmc_dn.time
            hops_array[interval_counter] = i + 1
            interval_counter += 1

    # Normalize final probability
    p_sim = p_sim/kmc_dn.time

    # Calculate norm
    convergence = np.linalg.norm(p_sim - p_theory)/np.linalg.norm(p_theory)

    print('Norm of difference: ' + str(convergence))

    if(not standalone):
        return E_microstates, p_theory, hops_array, p_sim_interval


def IV(kmc_dn, electrode, voltagelist,
       tol = 1E-2, interval = 1000, hops = 0, prehops = 0):
    '''
    Performs a simple IV curve measurement, by supplying the
    voltages in voltagelist on electrode electrode.
    If hops is unspecified, simulation method will be convergence based.
    '''
    if(hops == 0):
        discrete = False
    else:
        discrete = True

    voltages = len(voltagelist)
    currentlist = np.zeros((voltages, kmc_dn.P))
    
    # Check if any other electrode is non-zero
    zero = sum(kmc_dn.electrodes[:, 3]) - kmc_dn.electrodes[electrode, 3]
    zero = (zero == 0)

    # If all other electrodes are zero, calculate V profile only once!
    if(zero):
        V0 = voltagelist[0]
        kmc_dn.electrodes[electrode, 3] = voltagelist[0]
        kmc_dn.update_V()
        eVref = kmc_dn.eV_constant.copy()

    for i in range(voltages):
        # Measure time for second voltage to estimate total time
        if(i == 1):
            tic = time.time()

        # If all electrodes zero, only explicitly update E_constant
        if(zero):
            kmc_dn.eV_constant = eVref * voltagelist[i]/V0
            kmc_dn.electrodes[electrode, 3] = voltagelist[i]
            kmc_dn.E_constant = kmc_dn.eV_constant + kmc_dn.comp_constant
            kmc_dn.site_energies[kmc_dn.N + electrode] = kmc_dn.e*voltagelist[i]
        # Otherwise recalculate V
        else:
            kmc_dn.electrodes[electrode, 3] = voltagelist[i]
            kmc_dn.update_V()

        if(discrete):
            kmc_dn.simulate_discrete(hops = hops, prehops = prehops)
        else:
            kmc_dn.simulate(tol = tol, interval = interval, prehops = prehops)

        currentlist[i] = kmc_dn.current
        
        # Print estimated remaining time
        if(i == 1):
            print(f'Estimated time for IV curve: {(time.time()-tic)*voltages} seconds')
    return currentlist
