'''
This file is a collection of utilities to be used together with the class
kmc_dn defined in kmc_dopant_networks.

#TODO: documentation
@author: Bram de Wilde (b.dewilde-1@student.utwente.nl)
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools

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
        plt.axis('scaled')
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
            fig.colorbar(V_profile)


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

    return fig

def visualize_current(kmc_dn):
    '''Returns a figure which shows the domain with potential profile and
    occupancy. Plot vectors which correlate to the net hop direction.'''
    fig = visualize_basic(show_occupancy = False)

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
