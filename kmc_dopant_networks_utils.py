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
import math

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






def visualize_current_density(kmc_dn, res = None, title = None,
                              normalize = True):
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
    y = np.linspace(0, kmc_dn.ydim, kmc_dn.ydim/res + 1)
    current_map = np.zeros((len(x) - 1, len(y) - 1))

    # Calculate current_vectors from traffic
    current_vectors = np.zeros((kmc_dn.transitions.shape[0], 3))
    for i in range(kmc_dn.N + kmc_dn.P):
        for j in range(kmc_dn.traffic.shape[0]):
            if(j is not i):
                # Arriving hops
                current_vectors[i] += kmc_dn.traffic[j, i] * kmc_dn.vectors[j, i]

                # Departing hops
                current_vectors[i] += kmc_dn.traffic[i, j] * kmc_dn.vectors[i, j]

    norm = np.linalg.norm(current_vectors, axis = 1)  # Vector lengths


    # Loop over all squares in the domain
    # Square index is bottom left corner
    for i in range(len(x) -1):
        for j in range(len(y) - 1):
            acc_in_box = []  # indices of acceptors in box
            for k in range(kmc_dn.N):
                if( (x[i] <= kmc_dn.acceptors[k, 0] <= x[i+1])
                    and (y[j] <= kmc_dn.acceptors[k, 1] <= y[j+1])):
                    acc_in_box.append(k)

            net_box_current = 0
            for k in acc_in_box:
                net_box_current += current_vectors[k]
            
            current_map[i, j] = np.linalg.norm(net_box_current)

    if(normalize):
        current_map = current_map/np.max(current_map)

    # Normalize electrode currents
    electrode_currents = kmc_dn.electrode_occupation/np.max(kmc_dn.electrode_occupation)

    interp = 'none'
    interp = 'gaussian'
    # Plot current density
    fig = plt.figure()
    ax = fig.add_subplot(111)
    margins = 0.1
    ax.set_xlim(left=-margins*kmc_dn.xdim, right=(1+margins)*kmc_dn.xdim)
    ax.set_ylim(bottom=-margins*kmc_dn.ydim, top=(1+margins)*kmc_dn.ydim)
    ax.set_aspect('equal')
    im = ax.imshow(current_map.transpose(), interpolation = interp,
           origin='lower', extent=(0, kmc_dn.xdim, 0, kmc_dn.ydim), cmap=plt.cm.plasma)
    # Overlay dopants
    ax.scatter(kmc_dn.acceptors[:, 0], kmc_dn.acceptors[:, 1], color = 'black', marker='o')
    # Overlay dopant vectors
    x_dopants = kmc_dn.acceptors[:, 0]
    y_dopants = kmc_dn.acceptors[:, 1]
    u = current_vectors[:, 0]/norm
    v = current_vectors[:, 1]/norm
    #ax.quiver(x_dopants, y_dopants, u, v, norm, cmap=plt.cm.inferno)
    ax.quiver(x_dopants, y_dopants, u[:kmc_dn.N], v[:kmc_dn.N])
    # Overlay electrodes as dots
    ax.quiver(kmc_dn.electrodes[:, 0], kmc_dn.electrodes[:, 1],
              current_vectors[kmc_dn.N:, 0], current_vectors[kmc_dn.N:, 1], pivot='mid', color='red',
              alpha=0.8)

    # Add colorbar
    fig.colorbar(im, label = 'Current density (a.u.)')

    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('y (a.u.)')
    if(title != None):
        ax.set_title(title)

    return fig, current_map

def visualize_dwelltime(kmc_dn, show_V = True):
    '''
    Returns a figure which shows the domain with potential profile.
    It shows dwelltime (relative hole occupancy) as the size of
    the dopants.
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


        ax.scatter(kmc_dn.acceptors[:, 0], kmc_dn.acceptors[:, 1], s = kmc_dn.dwelltime/np.max(kmc_dn.dwelltime) * 110,marker='o')

        ax.scatter(kmc_dn.donors[:, 0], kmc_dn.donors[:, 1], marker='x')

        ax.set_xlabel('x (a.u.)')
        ax.set_ylabel('y (a.u.)')
        ax.set_title('Yellow = occupied, black = unoccupied')
        cbar.set_label('Chemical potential (V)')

    return fig

def visualize_traffic(kmc_dn, pos=111, title="", figure=None):
    if figure:
        fig = figure
    else:
        fig = plt.figure()

    ax = fig.add_subplot(pos)
    ax.set_xlim(right=max(1, kmc_dn.xdim))
    ax.set_ylim(top=max(1, kmc_dn.ydim))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if len(title) > 0:
        ax.set_title(title)

    acceptorColors = []
    for i in range(len(kmc_dn.acceptors)):
        h = hex(math.floor((1-kmc_dn.average_occupation[i])*255))
        h = h[2:]
        if len(h)==1:
            h = "0%s"%(h)
        colorStr = "#%s%s%s"%(h, h, h)
        acceptorColors.append(colorStr)

    ax.scatter(kmc_dn.acceptors[:, 0], kmc_dn.acceptors[:, 1], c = 'k', marker='o', s=64)
    ax.scatter(kmc_dn.acceptors[:, 0], kmc_dn.acceptors[:, 1], c = acceptorColors, marker='o', s=48)
    ax.scatter(kmc_dn.electrodes[:,0], kmc_dn.electrodes[:,1], c = 'r', marker='o')
    ax.scatter(kmc_dn.donors[:, 0], kmc_dn.donors[:, 1], marker='x')
    largest = 0
    for row in kmc_dn.traffic:
        for ele in row:
            if largest < ele:
                largest =  ele
    
    NSites = len(kmc_dn.acceptors)
    NElectrodes = len(kmc_dn.electrodes)
    N = NSites + NElectrodes
    arrowLength = kmc_dn.xdim/20
    for i in range(N):
        for j in range(N):
            traffic = kmc_dn.traffic[i][j]
            if traffic <= 0:
                continue
            
            intensity = traffic / 1.0 / largest
            if intensity < 0.01:
                continue
            #print ("i: %d, j: %d, intensity: %.3f, largest: %d"%(i, j, intensity, largest))
            startPos = getPosition(kmc_dn, i)
            endPos = getPosition(kmc_dn, j)
            distance = getDistance(startPos, endPos)
            arrows = max(1, math.floor(distance / (arrowLength*1.5)))
            arrowVector = ((endPos[0]-startPos[0])/distance*arrowLength, 
                (endPos[1]-startPos[1])/distance*arrowLength)
            for a in range(arrows):
                x = startPos[0]+a*(arrowVector[0])*1.5+arrowVector[0]*0.25
                y = startPos[1]+a*arrowVector[1]*1.5+arrowVector[1]*0.25
                width = 0.004
                ax.arrow(x, y, arrowVector[0], arrowVector[1], length_includes_head=True, width=width, head_width=3*width, head_length=arrowLength/4, alpha=math.sqrt(intensity))
    center = (kmc_dn.xdim/2, kmc_dn.ydim/2)
    for i in range(NElectrodes):
        ele = kmc_dn.electrodes[i]
        x = (ele[0] - center[0])*0.1 + ele[0]
        y = (ele[1] - center[1])*0.1 + ele[1]
        ax.text(x, y, "V:%.2f\nCurrent: %.3f"%(ele[3], kmc_dn.current[i]))
    return fig


def visualize_traffic_substraction(kmc1, kmc2, pos=111, title="", figure=None):
    if figure:
        fig = figure
    else:
        fig = plt.figure()

    ax = fig.add_subplot(pos)
    ax.set_xlim(right=max(1, kmc1.xdim))
    ax.set_ylim(top=max(1, kmc1.ydim))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if len(title) > 0:
        ax.set_title(title)

    acceptorColors = []
    for i in range(len(kmc1.acceptors)):
        print (kmc1.average_occupation[i])
        print (kmc2.average_occupation[i])
        diff = kmc1.average_occupation[i] - kmc2.average_occupation[i]
        print (diff)
        r, g, b = 255, 255, 255
        if diff < 0:
            g, b = 255+(diff*255), 255+(diff*255)
        else :
            r, b = 255-(diff*255), 255+(diff*255)
        colorStr = getColorHex(r, g, b)
        print (colorStr)
        acceptorColors.append(colorStr)

    ax.scatter(kmc1.acceptors[:, 0], kmc1.acceptors[:, 1], c = 'k', marker='o', s=64)
    ax.scatter(kmc1.acceptors[:, 0], kmc1.acceptors[:, 1], c = acceptorColors, marker='o', s=48)
    ax.scatter(kmc1.electrodes[:,0], kmc1.electrodes[:,1], c = 'r', marker='o')
    ax.scatter(kmc1.donors[:, 0], kmc1.donors[:, 1], marker='x')
    largest = 0
    for i in range(len(kmc1.traffic)):
        for j in range(len(kmc1.traffic[i])):
            curr = kmc1.traffic[i][j]/kmc1.time
            
            if largest < curr:
                largest =  curr
    
    NSites = len(kmc1.acceptors)
    NElectrodes = len(kmc1.electrodes)
    N = NSites + NElectrodes
    arrowLength = kmc1.xdim/20
    for i in range(N):
        for j in range(N):
            traffic = kmc2.traffic[i][j]/kmc2.time -kmc1.traffic[i][j]/kmc1.time
            if traffic <= 0:
                continue
            
            intensity = traffic / largest
            if intensity < 0.01:
                continue
            #print ("i: %d, j: %d, intensity: %.3f, largest: %d"%(i, j, intensity, largest))
            startPos = getPosition(kmc1, i)
            endPos = getPosition(kmc1, j)
            distance = getDistance(startPos, endPos)
            arrows = max(1, math.floor(distance / (arrowLength*1.5)))
            arrowVector = ((endPos[0]-startPos[0])/distance*arrowLength, 
                (endPos[1]-startPos[1])/distance*arrowLength)
            for a in range(arrows):
                x = startPos[0]+a*(arrowVector[0])*1.5+arrowVector[0]*0.25
                y = startPos[1]+a*arrowVector[1]*1.5+arrowVector[1]*0.25
                width = 0.004
                ax.arrow(x, y, arrowVector[0], arrowVector[1], length_includes_head=True, width=width, head_width=3*width, head_length=arrowLength/4, alpha=math.sqrt(intensity))
    center = (kmc1.xdim/2, kmc1.ydim/2)
    for i in range(NElectrodes):
        ele = kmc1.electrodes[i]
        x = (ele[0] - center[0])*0.06 + ele[0]
        y = (ele[1] - center[1])*0.06 + ele[1]
        ax.text(x, y, "V:%.2f\nCurrent difference: %.3f"%(ele[3], kmc2.current[i] - kmc1.current[i]))
    return fig

def plot_swipe(data, pos=111, figure=None, title=""):
    if figure:
        fig = figure
    else:
        fig = plt.figure()

    ax = fig.add_subplot(pos)
    voltages = []
    currents = []
    for vol, curr in data:
        voltages.append(vol)
        currents.append(curr)
    ax.plot(voltages, currents)

    if len(title) > 0:
        ax.set_title(title)

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
    kmc_dn.initialize(dopant_placement = False)

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
        kmc_dn.simulate_discrete_callback(hops = 1, reset = False)

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
    currentlist = np.zeros((kmc_dn.P, voltages))

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
            kmc_dn.site_energies[kmc_dn.N + electrode] = voltagelist[i]
        # Otherwise recalculate V
        else:
            kmc_dn.electrodes[electrode, 3] = voltagelist[i]
            kmc_dn.update_V()

        if(discrete):
            kmc_dn.python_simulation(hops = hops, prehops = prehops)
        else:
            kmc_dn.python_simulation(tol = tol, interval = interval, prehops = prehops)

        currentlist[:, i] = kmc_dn.current

        # Print estimated remaining time
        if(i == 1):
            print(f'Estimated time for IV curve: {(time.time()-tic)*voltages} seconds')
    return currentlist

def getPosition(kmc_dn, index):
    if index < len(kmc_dn.acceptors):
        return (kmc_dn.acceptors[index][0], kmc_dn.acceptors[index][1])
    else:
        return (kmc_dn.electrodes[index-len(kmc_dn.acceptors)][0], 
            kmc_dn.electrodes[index-len(kmc_dn.acceptors)][1])

def getDistance(coord1, coord2):
    x = coord2[0] - coord1[0]
    y = coord2[1] - coord1[1]
    return math.sqrt((x*x)+(y*y))

def getColorHex(r, g, b):
    colorStr = "#%s%s%s"%(singleHex(r), singleHex(g), singleHex(b))
    return colorStr

def singleHex(val):
    if val > 255:
        val = 255
    if val < 0:
        val = 0
    r = hex(math.floor(val))[2:]
    if len(r)==1:
        r = "0%s"%(r)
    return r
