'''
Class definition file for the kinetic monte carlo simulations on the
dopant network system. This class incorporates the following algorithm.

Pseudo-code (algorithm):
    # Initialization
    1. Donor placement (N acceptors and M < N donors)
    2. Place charges (N-M)
    3. Solve electrostatic potential from gates (relaxation method?)
    4. Solve compensation energy terms
    # Loop
    5. Calculate Coulomb energy terms
    6. Calculate hopping rates
    7. Hopping event
    8. Current converged?
        No: return to 5.
        Yes: simulation done.

(Average) quantities that are tracked:
    Current through electrodes
    ?Current through domain?
    Number of particles in the system
    Average energy of the system
    ?Average hopping distance?

Quantities that can be calculated:
    Mobility?
    Conductance?

TODO: Implement Tsigankov mixed algorithm and perform validation
TODO: Replace electrodes.shape[0] with self.P and replace self.P by smth like
    problist.



@author: Bram de Wilde (b.dewilde-1@student.utwente.nl)
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
from numba import jit
import fenics as fn

plt.ioff()

# Method definitions outside class (for numba support)

@jit
def _full_event_loop(N, P, nu, kT, I_0, R, time, occupation, distances, E_constant, site_energies,
                      transitions_constant, transitions, problist, electrode_occupation):
    '''
    This functions does everything to perform one event, which means:
    - Calculate site energies; it does this including chemical potential and
        acceptor interaction.
    - Calculate transition matrix; uses site_energies and transitions_constant
        and implement the MA rate.
    - Pick and perform event; uses the transition matrix (a standard step)
    '''
    # calc_site_energies_acc
    for i in range(N):
        acceptor_interaction = 0
        site_energies[i] = E_constant[i]
        for j in range(N):
            if j is not i:
                acceptor_interaction += (1 - occupation[j])/distances[i, j]

        site_energies[i] += -I_0*R*acceptor_interaction

    # calc_transitions
    for i in range(N+P):
        for j in range(N+P):
            if(not _transition_possible(i, j, N, occupation)):
                transitions[i, j] = 0
            else:
                if(i < N and j < N):
                    dE = (site_energies[j] - site_energies[i]
                          - I_0*R/distances[i, j])
                else:
                    dE = site_energies[j] - site_energies[i]

                if(dE > 0):
                    t = nu*np.exp(-dE/kT)  # Calculate MA rate
                else:
                    t = 1

                transitions[i, j] = t
    transitions = transitions_constant*transitions

    # pick_event
    # Transform transitions matrix into cumulative sum
    for i in range(N+P):
        for j in range(N+P):
            if(i == 0 and j == 0):
                problist[0] = transitions[i, j]
            else:
                problist[(N+P)*i + j] = transitions[i, j] + problist[(N+P)*i + j-1]

    # Calculate hopping time
    hop_time = np.random.exponential(scale=1/problist[-1])

    # Normalization
    problist = problist/problist[-1]

    # Find transition index of random event
    event = np.random.rand()
    for i in range((N+P)**2):
        if(problist[i] >= event):
            event = i
            break
    #TODO: implement binary tree search algorithm

    # Convert to acceptor/electrode indices
    transition = [int(event/(N+P)), int(event%(N+P))]

    # Perform hop
    if(transition[0] < N):  # Hop from acceptor
        occupation[transition[0]] = False
    else:  # Hop from electrode
        electrode_occupation[transition[0] - N] -= 1
    if(transition[1] < N):  # Hop to acceptor
        occupation[transition[1]] = True
    else:  # Hop to electrode
        electrode_occupation[transition[1] - N] += 1

    # Increment time
    time += hop_time

    return hop_time, time, transition, occupation, electrode_occupation


@jit
def _calc_site_energies_acc(N, I_0, R, occupation, distances, E_constant, site_energies):
    '''
    Calculates the potential energy of each acceptor site by evaluating
    the Coulomb interactions between all other acceptors and by adding
    the constant energy terms.
    The energies are stored in the (N+P)x1 array site_energies
    '''
    for i in range(N):
        acceptor_interaction = 0
        site_energies[i] = E_constant[i]
        for j in range(N):
            if j is not i:
                acceptor_interaction += (1 - occupation[j])/distances[i, j]

        site_energies[i] += -I_0*R*acceptor_interaction

    return site_energies

@jit
def _transition_possible(i, j, N, occupation):
    '''
    Returns false if:
     - i has occupation 0
     - j has occupation 1
     - i and j are electrodes
    '''
    if(i >= N and j >= N):
        return False
    elif(i >= N):
        if(occupation[j] == True):
            return False
        else:
            return True
    elif(j >= N):
        if(occupation[i] == False):
            return False
        else:
            return True
    elif(i == j):
        return False
    else:
        if(occupation[i] == False or occupation[j] == True):
            return False
        else:
            return True

@jit
def _calc_transitions_MA(N, P, nu, kT, I_0, R, occupation, transitions_constant, transitions,
                         distances, site_energies):
    '''
    Calculates the transition matrix transitions using the Miller-Abrahams rate.
    '''
    for i in range(N+P):
        for j in range(N+P):
            if(not _transition_possible(i, j, N, occupation)):
                transitions[i, j] = 0
            else:
                if(i < N and j < N):
                    dE = (site_energies[j] - site_energies[i]
                          - I_0*R/distances[i, j])
                else:
                    dE = site_energies[j] - site_energies[i]

                t = nu*np.exp(-dE/kT)  # Calculate MA rate
                # Threshold value if larger than 1
                if(t > 1):
                    t = 1

                transitions[i, j] = t

    return transitions_constant*transitions

@jit
def _pick_event(N, P, time, problist, transitions, occupation, electrodes):
    '''
    Picks event and performs the transition
    '''
    # Transform transitions matrix into cumulative sum
    for i in range(N+P):
        for j in range(N+P):
            if(i == 0 and j == 0):
                problist[0] = transitions[i, j]
            else:
                problist[(N+P)*i + j] = transitions[i, j] + problist[(N+P)*i + j-1]

    # Calculate hopping time
    hop_time = np.random.exponential(scale=1/problist[-1])

    # Normalization
    problist = problist/problist[-1]

    # Find transition index of random event
    event = np.random.rand()
    for i in range((N+P)**2):
        if(problist[i] >= event):
            event = i
            break
    #TODO: implement binary tree search algorithm

    # Convert to acceptor/electrode indices
    transition = [int(event/(N+P)), int(event%(N+P))]

    # Perform hop
    if(transition[0] < N):  # Hop from acceptor
        occupation[transition[0]] = False
    else:  # Hop from electrode
        electrodes[transition[0] - N, 4] -= 1
    if(transition[1] < N):  # Hop to acceptor
        occupation[transition[1]] = True
    else:  # Hop to electrode
        electrodes[transition[1] - N, 4] += 1

    # Increment time
    time += hop_time

    return hop_time, time, transition, occupation, electrodes

class kmc_dn():
    '''This class is a wrapper for all functions and variables needed to perform
    a kinetic monte carlo simulation of a variable range hopping system'''

    def __init__(self, N, M, xdim, ydim, zdim, mu = 0, **kwargs):
        '''Upon initialization of this class, the impurities and charges are placed.
        They are placed inside a rectangle of xdim by ydim. There are N acceptors
        and M donors.

        ------------------------------------------------------------------------
        Input arguments
        ------------------------------------------------------------------------
        N; number of acceptors
        M; number of donors
        xdim; x dimension size of domain
        ydim; y dimension size of domain
        zdim; z dimension size of domain

        possible kwargs:
        electrodes; electrode configuration, an Px4 np.array, where
            P is the number of electrodes, the first three columns correspond
            to the x, y and coordinates of the electrode, respectively,
            the fourth column holds the electrode voltage.
            default: np.zeros((0, 4))
        res; resolution used for potential landscape calculation
            default: min[xdim, ydim, zdim]/100
        place_dopants_charges; choice of the place_dopants_charges method.
            Possible options are:
            'place_dopants_charges_random'; random dopant and charge placement
            Default: place_dopants_charges_random
        calc_E_constant; choice of the calc_E_constant method.
            Possible options are:
            'calc_E_constant_V'; include only local chemical potential
            'calc_E_constant_V_comp'; include local chemical potential and
                compensation sites.
            Default: calc_E_constant_V_comp
        calc_site_energies; choice of the calc_site_energies method
            Possible options are:
            'calc_site_energies_acc'; Site energy depends on acc-acc interaction
                and E_constant
            Default: calc_site_energies_acc
        calc_rate; choice of the calc_rate method
            Possible options are:
            'calc_rate_MA'; Miller-Abrahams rate
            Default: calc_rate_MA
        calc_transitions; choice of the calc_transitions method
            Possible options are:
            'calc_transitions_MA'; use the Miller-Abrahams rate.
            Default: calc_transitions_MA
        pick_event; choice of the pick_event method
            Possible options are:
            'pick_event_standard'; pick event following standard algorithm, i.e.
                purely based on the transitions matrix.
            Default: pick_event_standard
        perform_event; choice of the perform_event method
            Possible options are:
            'perform_event_standard'; perfom event following standard algorithm,
                i.e. change relevant occupation numbers and increment time.
            Default: perform_event_standard
        stopping_criterion; choice of the stopping_criterion method
            Possible options are:
            'stopping_criterion_discrete'; stop when a predetermined amount of
                hops is reached.
            'stopping_criterion_convergence'; stop when the current converged
                the predefined tolerance tol.
            Default: stopping_criterion_discrete
        callback; choice of the callback method
            Possible options are:
            'callback_standard'; track avg_carriers and current vectors
            'callback_avg_carriers'; track avg_carriers
            'callback_current_vectors'; track current vectors
            'callback_traffic'; tracks traffic in array
            Default: callback_standard

        ------------------------------------------------------------------------
        Class attributes
        ------------------------------------------------------------------------
        Constants:
        e; elementary charge
        eps; relative permittivity
        nu; attempt frequency for transitions
        k; boltzmann constant
        T; temperature
        ab; bohr radius/localization radius
        U; interaction energy for double occupancy (EXPERIMENTAL/UNIMPLEMENTED)

        Other attributes:
        time; simulation time
        counter; simulation event counter
        acceptors; Nx3 array, where first three columns are x, y, z position
        donors; Mx3 array, where the columns represent x, y, z position of donors.
            Donors are assumed to be always occupied.
        occupation; (N,) bool array, indicating hole occupancy of each acceptor
        electrodes; electrode configuration, an Px5 np.array, where
            P is the number of electrodes, the first three columns correspond
            to the x, y and coordinates of the electrode, respectively,
            the fourth column holds the electrode voltage and the last column
            tracks the amount of carriers sourced/sinked in the electrode.
        electrodes_grid; same array as electrodes, but here x, y, z are integers
            matched to the V matrix. So x, y, z are the indices such that
            V[x, y, z] is the local chemical potential at that electrode.
        distances; (N+P)x(N+P) array, with the distance for each transition
        transitions; (N+P)x(N+P) array, with the transition rate for each transition
        transitions_constant; (N+P)x(N+P) array, with the constant part
            (i.e. position dependent) of the transitions array
        transitions_possible; (N+P)x(N+P) boolean array, indicating for each
            hop i -> j if it is possible.
        vectors; (N+P)x(N+P)x3 array, where vectors[i, j] is the unit vector
            point from site i to site j
        current_vectors; (N+P)x3 array, where vectors[i] is the vector which
        points in the average direction of transport through the acceptor i.
        V; 3D array, the chemical potential profile
        E_constant; Nx1 array, energy contributions that are constant throughout one simulation
            run, is equal to the sum of:
            eV_constant; Nx1 array, contribution of chemical potential to site energy
            comp_constant; Nx1 array, contribution of compensation charges to site energy
        site_energies; Nx1 array, contains potential energy of each acceptor site.
        P; cumulative probability list for possible transitions
        hop_time; time spent for the last hop
        transition; [i, j] for the last hop from site i -> site j
        current; Px1 array, holds the current, i.e. electrodes[:, 4]/time, for
            each electrode
        old_current; Px1 array, holds the current for the previous interval, used
            to check for convergence.
        avg_current; tracks the current each interval
        avg_carriers; tracks the average number of carriers in the system.

        ------------------------------------------------------------------------
        Class methods
        ------------------------------------------------------------------------
        Individual methods:
        place_dopants_charges; places dopants and charges on the domain
        calc_distances; calculates matrices distances and vectors
        calc_V; calculates matrix V
        calc_E_constant; calculates potential energy terms that are constant
            through a single simulation.
        calc_site_energies; calculate the vector site_energies
        calc_transitions; calculate the matrix transitions
            depends on
            - calc_rate; hopping rate equation
        callback; arbitrary function that is executed each kmc iteration. Used to
            track various quantities, visualization etc.
        pick_event; pick event based on transitions
        perform_event; perform event picked by pick_event and increment time.
        stopping_criterion; function that checks if stopping criterion is met.

        Wrapper methods:
        The above methods are all specifically performing one step of a
        simulation routine. However, each method can be replaced by a different
        one. E.g. calc_site_energies will depend on the type of Hamiltonian you
        wish to simulate, but its output will always be of the same type.
        There are two main wrapper methods that allow you to easily run a
        simulation in an external script.
        initialize; performs everything that is constant for kmc simulation
            - place_dopants_charges
            - calc_distances
            - calc_V
            - calc_E_constant
        simulate; runs a kmc simulation based on the initialized situation
            while(not stopping_criterion)
            - calc_site_energies
            - calc_transitions
            - pick_event
            - perform_event
            - callback
        '''
        # Constants
        self.e = 1  #  1.602E-19 Coulomb
        self.eps = 1  # Relative permittivity (11.68 for boron doped Si)
        self.nu = 1
        self.k = 1
        self.T = 1
        self.ab = 100 # Bohr radius (or localization radius)
        self.U = 100  # 5/8 * 1/self.ab   # J
        self.time = 0  # s
        self.mu = mu  # Equilibrium chemical potential

        # Initialize variables
        self.N = N
        self.M = M
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim


        # Check dimensionality
        if(self.ydim == 0 and self.zdim == 0):
            self.dim = 1
            self.R = (self.N/self.xdim)**(-1)
        elif(self.zdim == 0):
            self.dim = 2
            self.R = (self.N/(self.xdim*self.ydim))**(-1/2)
        else:
            self.dim = 3
            self.R = (self.N/(self.xdim*self.ydim*self.zdim))**(-1/3)

        # Initialize shorthand constants
        self.I_0 = self.e**2/(4*np.pi*self.eps*self.R)
        self.kT = self.k*self.T

        # Set dimensonless variables to 1
        self.ab = self.R
        self.I_0 = self.kT


        # Initialize parameter kwargs
        if('electrodes' in kwargs):
            self.electrodes = kwargs['electrodes'].copy()
            self.P = self.electrodes.shape[0]
        else:
            self.electrodes = np.zeros((0, 4))
            self.P = 0

        if('res' in kwargs):
            self.res = kwargs['res']
        else:
            if(self.dim == 1):
                self.res = self.xdim/100
            if(self.dim == 2):
                self.res =  min([self.xdim, self.ydim])/100
            if(self.dim == 3):
                self.res = min([self.xdim, self.ydim, self.zdim])/100

        # Initialize method kwargs

        if('calc_E_constant' in kwargs):
            if(kwargs['calc_E_constant'] == 'calc_E_constant_V'):
                self.calc_E_constant = self.calc_E_constant_V
            if(kwargs['calc_E_constant'] == 'calc_E_constant_V_comp'):
                self.calc_E_constant = self.calc_E_constant_V_comp
        else:
            self.calc_E_constant = self.calc_E_constant_V_comp

        if('calc_site_energies' in kwargs):
            if(kwargs['calc_site_energies'] == 'calc_site_energies_acc'):
                self.calc_site_energies = self.calc_site_energies_acc
                self.coulomb_interactions = True
        else:
            self.calc_site_energies = self.calc_site_energies_acc
            self.coulomb_interactions = True

        if('calc_transitions' in kwargs):
            if(kwargs['calc_transitions'] == 'calc_transitions_MA'):
                self.calc_transitions = self.calc_transitions_MA
        else:
            self.calc_transitions = self.calc_transitions_MA


        if('pick_event' in kwargs):
            if(kwargs['pick_event'] == 'pick_event_standard'):
                self.pick_event = self.pick_event_standard
        else:
            self.pick_event = self.pick_event_standard

        if('callback' in kwargs):
            if(kwargs['callback'] == 'callback_standard'):
                self.callback = self.callback_standard
            if(kwargs['callback'] == 'callback_avg_carriers'):
                self.callback = self.callback_avg_carriers
            if(kwargs['callback'] == 'callback_current_vectors'):
                self.callback = self.callback_current_vectors
            if(kwargs['callback'] == 'callback_traffic'):
                self.callback = self.callback_traffic
            if(kwargs['callback'] == 'callback_dwelltime'):
                self.callback = self.callback_dwelltime
            if(kwargs['callback'] == 'none'):
                self.callback = self.callback_none
        else:
            self.callback = self.callback_standard

        # Initialize sim object
        self.initialize()

    def initialize(self, dopant_placement = True, charge_placement = True, distances = True,
                   V = True, E_constant = True):
        '''
        Wrapper function which:
        - places acceptors/donors
        - calculates distances
        - calculates V (electrostatic potential profile)
        - calculates E_constant
        These are all methods that determine the starting position of a kmc
        simulation.
        All methods are toggleable by boolean values.
        '''
        # Initialize other attributes
        self.transitions = np.zeros((self.N + self.P,
                                     self.N + self.P))
        self.transitions_constant = np.zeros((self.N + self.P,
                                     self.N + self.P))
        self.distances = np.zeros((self.N + self.P,
                                   self.N + self.P))
        self.vectors = np.zeros((self.N + self.P,
                                 self.N + self.P, 3))
        self.site_energies = np.zeros((self.N + self.P,))
        self.problist = np.zeros((self.N+self.P)**2)
        self.occupation = np.zeros(self.N, dtype=bool)
        self.electrode_occupation = np.zeros(self.P, dtype=int)

        if(dopant_placement):
            self.place_dopants_random()

        if(charge_placement):
            self.place_charges_random()

        if(distances):
            self.calc_distances()

            self.calc_transitions_constant()

        if(V):
            self.init_V()

        if(E_constant):
            self.calc_E_constant()


    def simulate_discrete(self, hops = 1000, reset = True, prehops = 0):
        '''
        Wrapper function that perform a simulation loop for a predetermined
        amount of hops.
        hops has to be an integer.
        Reset is optional, such that this function can be used to go through
        a simulation step by step (useful for e.g. boltzmann validation)
        prehops can be used to perform hops before measuring any current, 
        and can be used to bring the system in 'equilibrium' first.
        '''
        if(reset):
            self.reset()  # Reset all relevant trackers before running a simulation

        if(prehops != 0):
            for i in range(prehops):
                # Perform a single event
                (self.hop_time,
                self.time,
                self.transition,
                self.occupation,
                self.electrode_occupation) = _full_event_loop(self.N,
                                                            self.P,
                                                            self.nu,
                                                            self.kT,
                                                            self.I_0,
                                                            self.R,
                                                            self.time,
                                                           self.occupation,
                                                           self.distances,
                                                           self.E_constant,
                                                           self.site_energies,
                                                           self.transitions_constant,
                                                           self.transitions,
                                                           self.problist,
                                                           self.electrode_occupation)

                self.reset()

        for i in range(hops):
            # Perform a single event
            (self.hop_time,
             self.time,
             self.transition,
             self.occupation,
             self.electrode_occupation) = _full_event_loop(self.N,
                                                           self.P,
                                                           self.nu,
                                                           self.kT,
                                                           self.I_0,
                                                           self.R,
                                                           self.time,
                                                           self.occupation,
                                                           self.distances,
                                                           self.E_constant,
                                                           self.site_energies,
                                                           self.transitions_constant,
                                                           self.transitions,
                                                           self.problist,
                                                           self.electrode_occupation)

            self.callback()

            self.counter += 1

        self.current = self.electrode_occupation/self.time

    def simulate(self, tol = 1E-2, interval = 1000, prehops = 0):
        '''
        Wrapper function that performs a simulation until the all electrode
        currents have converged with tolerance tol. The function checks
        for convergence every interval hops.
        prehops indicates the amount of hops performed before calculating
        currents for convergence. This can be used to first bring the system
        into 'equilibrium'.
        '''
        self.reset()  # Reset all relevant trackers before running a simulation


        for i in range(prehops):
            # Perform a single event
            (self.hop_time,
             self.time,
             self.transition,
             self.occupation,
             self.electrode_occupation) = _full_event_loop(self.N,
                                                           self.P,
                                                           self.nu,
                                                           self.kT,
                                                           self.I_0,
                                                           self.R,
                                                           self.time,
                                                           self.occupation,
                                                           self.distances,
                                                           self.E_constant,
                                                           self.site_energies,
                                                           self.transitions_constant,
                                                           self.transitions,
                                                           self.problist,
                                                           self.electrode_occupation)

        self.reset()
        self.converged = False
        while(not self.converged):
            # Perform a single event
            (self.hop_time,
             self.time,
             self.transition,
             self.occupation,
             self.electrode_occupation) = _full_event_loop(self.N,
                                                           self.P,
                                                           self.nu,
                                                           self.kT,
                                                           self.I_0,
                                                           self.R,
                                                           self.time,
                                                           self.occupation,
                                                           self.distances,
                                                           self.E_constant,
                                                           self.site_energies,
                                                           self.transitions_constant,
                                                           self.transitions,
                                                           self.problist,
                                                           self.electrode_occupation)

            self.callback()

            # Check for convergence
            if(self.counter != 0 and self.counter%interval == 0):
                self.current = self.electrode_occupation/self.time

                # Check convergence
                if(np.linalg.norm(self.current, 2) == 0):
                    if(np.linalg.norm(self.old_current - self.current, 2) == 0):
                        self.converged = True
                    else:
                        self.converged = False
                elif((np.linalg.norm(self.old_current - self.current, 2)
                      /np.linalg.norm(self.current,2)) < tol):
                    self.converged = True
                else:
                    self.old_current = self.current.copy()  # Store current
                    self.converged = False

            self.counter += 1

        print(f'Converged in {self.counter} hops')

    def reset(self):
        '''
        Resets all relevant trackers before running a simulation
        '''
        self.time = 0
        self.old_current = 0
        self.counter = 0
        self.electrode_occupation[:] = 0  # Reset current

        # Callback quantities
        self.avg_carriers_prenorm = 0
        self.avg_carriers = 0
        self.current_vectors = np.zeros((self.transitions.shape[0], 3))
        self.traffic = np.zeros(self.transitions.shape)
        self.dwelltime = np.zeros(self.N)
        self.previous_occupation = self.occupation

    def place_dopants_random(self):
        '''
        Place dopants and charges on a 3D hyperrectangular domain (xdim, ydim, zdim).
        Place N acceptors and M donors. Place N-M charges.
        Returns acceptors (Nx4 array) and donors (Mx3 array). The first three columns
        of each represent the x, y and z coordinates, respectively, of the acceptors
        and donors. The fourth column of acceptors denotes charge occupancy, with
        0 being an unoccupied acceptor and 1 being an occupied acceptor
        '''
        # Initialization
        self.acceptors = np.random.rand(self.N, 3)
        self.donors = np.random.rand(self.M, 3)
        self.occupation = np.zeros(self.N, dtype=bool)

        # Place dopants
        self.acceptors[:, 0] *= self.xdim
        self.acceptors[:, 1] *= self.ydim
        self.acceptors[:, 2] *= self.zdim
        self.donors[:, 0] *= self.xdim
        self.donors[:, 1] *= self.ydim
        self.donors[:, 2] *= self.zdim

    def place_charges_random(self):
        '''
        Places N-M holes
        '''
        # Place charges
        charges_placed = 0
        while(charges_placed < self.N-self.M):
            trial = np.random.randint(self.N)  # Try a random acceptor
            if(self.occupation[trial] == False):
                self.occupation[trial] = True  # Place charge
                charges_placed += 1

    def calc_distances(self):
        '''
        Calculates the distances between each hopping sites and stores them
        in the matrix distances.
        Also stores the unit vector in the hop direction i->j in the matrix vectors.
        '''
        for i in range(self.N+self.P):
            for j in range(self.N+self.P):
                if(i is not j):
                    # Distance electrode -> electrode
                    if(i >= self.N and j >= self.N):
                        self.distances[i, j] = self.dist(self.electrodes[i - self.N, :3],
                                                          self.electrodes[j - self.N, :3])
                        self.vectors[i, j] = ((self.electrodes[j - self.N, :3]
                                              - self.electrodes[i - self.N, :3])
                                              /self.distances[i, j])

                    # Distance electrodes -> acceptor
                    elif(i >= self.N and j < self.N):
                        self.distances[i, j] = self.dist(self.electrodes[i - self.N, :3],
                                                          self.acceptors[j])
                        self.vectors[i, j] = ((self.acceptors[j]
                                              - self.electrodes[i - self.N, :3])
                                              /self.distances[i, j])
                    # Distance acceptor -> electrode
                    elif(i < self.N and j >= self.N):
                        self.distances[i, j] = self.dist(self.acceptors[i],
                                                          self.electrodes[j - self.N, :3])
                        self.vectors[i, j] = ((self.electrodes[j - self.N, :3]
                                              - self.acceptors[i])
                                              /self.distances[i, j])
                    # Distance acceptor -> acceptor
                    elif(i < self.N and j < self.N):
                        self.distances[i, j] = self.dist(self.acceptors[i],
                                                          self.acceptors[j])
                        self.vectors[i, j] = ((self.acceptors[j]
                                              - self.acceptors[i])
                                              /self.distances[i, j])

    @staticmethod
    def fn_onboundary(x, on_boundary):
        return on_boundary

    def init_V(self):
        '''
        This function sets up various parameters for the calculation of
        the chemical potential profile using fenics.
        It is generally assumed that during the simulation of a 'sample'
        the following are unchanged:
        - dopant positions
        - electrode positions/number
        Note: only 2D support for now
        '''
        # Put electrode positions and values in a dict
        self.fn_electrodes = {}
        for i in range(self.P):
            self.fn_electrodes[f'e{i}_x'] = self.electrodes[i, 0]
            if(self.dim > 1):
                self.fn_electrodes[f'e{i}_y'] = self.electrodes[i, 1]
            self.fn_electrodes[f'e{i}'] = self.electrodes[i, 3]

        # Define boundary expression string
        self.fn_expression = ''
        if(self.dim == 1):
            for i in range(self.P):
                self.fn_expression += (f'x[0] == e{i}_x ? e{i} : ')

        if(self.dim == 2):
            surplus = self.xdim/10  # Electrode modelled as point +/- surplus
            for i in range(self.P):
                if(self.electrodes[i, 0] == 0 or self.electrodes[i, 0] == self.xdim):
                    self.fn_expression += (f'x[0] == e{i}_x && '
                                           f'x[1] >= e{i}_y - {surplus} && '
                                           f'x[1] <= e{i}_y + {surplus} ? e{i} : ')
                else:
                    self.fn_expression += (f'x[0] >= e{i}_x - {surplus} && '
                                           f'x[0] <= e{i}_x + {surplus} && '
                                           f'x[1] == e{i}_y ? e{i} : ')
        self.fn_expression += f'{self.mu}'  # Add constant chemical potential

        # Define boundary expression
        self.fn_boundary = fn.Expression(self.fn_expression,
                                         degree = 1,
                                         **self.fn_electrodes)

        # Define FEM mesh (res should be small enough, otherwise solver may break)
        if(self.dim == 1):
            self.fn_mesh = fn.IntervalMesh(int(self.xdim//self.res), 0, self.xdim)
        if(self.dim == 2):
            self.fn_mesh = fn.RectangleMesh(fn.Point(0, 0),
                                            fn.Point(self.xdim, self.ydim),
                                            int(self.xdim//self.res),
                                            int(self.ydim//self.res))

        # Define function space
        self.fn_functionspace = fn.FunctionSpace(self.fn_mesh, 'P', 1)

        # Define fenics boundary condition
        self.fn_bc = fn.DirichletBC(self.fn_functionspace,
                                    self.fn_boundary,
                                    self.fn_onboundary)

        # Write problem as fn_a == fn_L
        self.V = fn.TrialFunction(self.fn_functionspace)
        self.fn_v = fn.TestFunction(self.fn_functionspace)
        self.fn_a = fn.dot(fn.grad(self.V), fn.grad(self.fn_v)) * fn.dx
        self.fn_f = fn.Constant(0)
        self.fn_L = self.fn_f*self.fn_v*fn.dx

        # Solve V
        self.V = fn.Function(self.fn_functionspace)
        fn.solve(self.fn_a == self.fn_L, self.V, self.fn_bc)

    def update_V(self):
        '''
        This function updates/recalculates V using fenics.
        Should be called after changing electrode voltages.
        '''
        # Update electrode values in fn_electrodes
        for i in range(self.P):
            self.fn_electrodes[f'e{i}'] = self.electrodes[i, 3]

        # Update boundary condition
        self.fn_boundary = fn.Expression(self.fn_expression,
                                         degree = 1,
                                         **self.fn_electrodes)
        self.fn_bc = fn.DirichletBC(self.fn_functionspace,
                                    self.fn_boundary,
                                    self.fn_onboundary)

        # Solve V
        fn.solve(self.fn_a == self.fn_L, self.V, self.fn_bc)

        # Update constant energy
        self.calc_E_constant()


    def calc_transitions_constant(self):
        '''
        Calculates the constant (position dependent part) of the MA rate.
        This function puts the constant rate to 0 for transitions i -> j
        '''
        self.transitions_constant = self.nu*np.exp(-2 * self.distances/self.ab)
        self.transitions_constant -= np.eye(self.transitions.shape[0])


    def calc_E_constant_V(self):
        '''
        Solve the constant energy terms for each acceptor site.
        This method includes only energy contributions of the local chemical
        potential V.
        Also fixes the electrode energies in site_energies.
        '''
        # Initialization
        self.eV_constant = np.zeros((self.N,))
        self.comp_constant = np.zeros((self.N,))

        for i in range(self.N):
            # Add electrostatic potential
            if(self.dim == 1):
                self.eV_constant[i] += self.e*self.V(self.acceptors[i, 0])

            if(self.dim == 2):
                self.eV_constant[i] += self.e*self.V(self.acceptors[i, 0], self.acceptors[i, 1])

            if(self.dim == 3):
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 3) + 1
                z = self.acceptors[i, 2]/self.zdim * (self.V.shape[2] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)),
                                                    int(round(y)),
                                                    int(round(z))]

        self.E_constant = self.eV_constant

        # Calculate electrode energies
        self.site_energies[self.N:] = self.e*self.electrodes[:, 3]

    def calc_E_constant_V_comp(self):
        '''
        Solve the constant energy terms for each acceptor site.
        This method includes energy contributions of the local chemical potential
        V and of Coulomb interaction between the site and compensation charges.
        Also fixes the electrode energies in site_energies.
        '''
        # Initialization
        self.eV_constant = np.zeros((self.N,))
        self.comp_constant = np.zeros((self.N,))

        for i in range(self.N):
            # Add electrostatic potential
            if(self.dim == 1):
                self.eV_constant[i] += self.e*self.V(self.acceptors[i, 0])

            if(self.dim == 2):
                self.eV_constant[i] += self.e*self.V(self.acceptors[i, 0], self.acceptors[i, 1])

            if(self.dim == 3):
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 3) + 1
                z = self.acceptors[i, 2]/self.zdim * (self.V.shape[2] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)),
                                                    int(round(y)),
                                                    int(round(z))]

            # Add compensation
            self.comp_constant[i] += self.I_0*self.R* sum(
                    1/self.dist(self.acceptors[i], self.donors[k]) for k in range(self.M))

        self.E_constant = self.eV_constant + self.comp_constant

        # Calculate electrode energies
        self.site_energies[self.N:] = self.e*self.electrodes[:, 3]

    def calc_site_energies_acc(self):
        '''
        Calculates the potential energy of each acceptor site by evaluating
        the Coulomb interactions between all other acceptors and by adding
        the constant energy terms.
        The energies are stored in the (N+P)x1 array site_energies
        '''
        self.site_energies = _calc_site_energies_acc(self.N,
                                                     self.I_0,
                                                     self.R,
                                                     self.occupation,
                                                     self.distances,
                                                     self.E_constant,
                                                     self.site_energies)

    def calc_transitions_MA(self):
        '''
        Calculates the matrix transitions, following the MA hopping rate.
        It uses the method calc_energy_differences().
        transitions_constant = nu*exp(-2r_ij/a)
        transitions_constant also makes sure any transition rate to the same
        element is zero
        '''
        self.transitions = _calc_transitions_MA(self.N,
                                                self.P,
                                                self.nu,
                                                self.kT,
                                                self.I_0,
                                                self.R,
                                                self.occupation,
                                                self.transitions_constant,
                                                self.transitions,
                                                self.distances,
                                                self.site_energies)

        #TODO: check for fixed points

    def callback_none(self):
        pass

    def callback_standard(self):
        '''
        This is the standard callback function, tracking only traffic
        '''
        self.callback_traffic()
        self.callback_current_vectors()

    def callback_avg_carriers(self):
        '''
        Tracks the average number of carriers in the system
        '''
        self.avg_carriers_prenorm += self.hop_time * sum(self.occupation)
        self.avg_carriers = self.avg_carriers_prenorm/self.time


    def callback_current_vectors(self):
        '''
        Adds the appropriate unit vectors of the last transition to the matrix
        current_vectors.
        '''
        self.current_vectors[self.transition[1]] += self.vectors[self.transition[0],
                                                                 self.transition[1]] # Arrival
        self.current_vectors[self.transition[0]] += self.vectors[self.transition[0],
                                                                 self.transition[1]] # Departure

    def callback_traffic(self):
        '''
        Tracks the amount of hops i -> j in the matrix traffic.
        '''
        self.traffic[self.transition[0], self.transition[1]] += 1

    def callback_dwelltime(self):
        '''
        Tracks the time every acceptor is occupied
        '''
        self.dwelltime += self.hop_time * self.previous_occupation
        self.previous_occupation = self.occupation

    def pick_event_standard(self):
        '''
        Based on the transition matrix self.transitions, pick a hopping event.
        '''
        (self.hop_time,
         self.time,
         self.transition,
         self.occupation,
         self.electrodes) = _pick_event(self.N,
                                        self.P,
                                        self.time,
                                        self.problist,
                                        self.transitions,
                                        self.occupation,
                                        self.electrodes)


    def pick_event_tsigankov(self):
        '''Pick a hopping event based on t_dist and accept/reject it based on
        the energy dependent rate'''
        # Randomly determine event
        event = np.random.rand()

        # Find transition index
        event = min(np.where(self.P >= event)[0])

        # Convert to acceptor/electrode indices
        self.transition = [int(np.floor(event/self.t_dist.shape[0])),
                           int(event%self.t_dist.shape[0])]

        if(self.transition_possible(self.transition[0],
                                    self.transition[1])):
            # Calculate hopping probability
            eij = self.energy_difference(self.transition[0],
                                         self.transition[1])
            prob = 1/(1 + np.exp(eij/(self.k*self.T)))
            if(np.random.rand() < prob):
                # Perform hop
                if(self.transition[0] < self.N):  # Hop from acceptor
                    self.acceptors[self.transition[0], 3] -= 1
                else:  # Hop from electrode
                    self.electrodes[self.transition[0] - self.N, 4] -= 1
                if(self.transition[1] < self.N):  # Hop to acceptor
                    self.acceptors[self.transition[1], 3] += 1
                else:  # Hop to electrode
                    self.electrodes[self.transition[1] - self.N, 4] += 1

        # Increment time
        self.time += self.timestep

    #%% Load methods
    def load_acceptors(self, acceptors):
        '''
        This function allows loading an acceptor layout.
        '''
        # Overwrite acceptors array
        self.acceptors = acceptors
        self.N = self.acceptors.shape[0]

        # Re-initialize dimensionless constants (R may have changed)
        if(self.ydim == 0 and self.zdim == 0):
            self.R = (self.N/self.xdim)**(-1)
        elif(self.zdim == 0):
            self.R = (self.N/(self.xdim*self.ydim))**(-1/2)
        else:
            self.R = (self.N/(self.xdim*self.ydim*self.zdim))**(-1/3)

        # Initialize shorthand constants
        self.I_0 = self.e**2/(4*np.pi*self.eps*self.R)
        self.kT = self.k*self.T

        # Set dimensonless variables to 1
        self.ab = self.R
        self.I_0 = self.kT

        # Re-initialize everything but placement and V
        self.initialize(V = False, dopant_placement = False)

    def load_donors(self, donors):
        '''
        This function loads a donor layout.
        '''
        # Overwrite donors array
        self.donors = donors
        self.M = self.donors.shape[0]

        # Re-initialize everything but placement and V
        self.initialize(V=False, dopant_placement=False)

    #%% Miscellaneous methods
    def calc_t_dist(self):
        '''Calculates the transition rate matrix t_dist, which is based only
        on the distances between sites (as defined in Tsigankov2003)'''
        # Initialization
        self.t_dist = np.zeros((self.N + P,
                                self.N + P))
        self.P = np.zeros((self.transitions.shape[0]**2))  # Probability list

        # Loop over possible transitions site i -> site j
        for i in range(self.t_dist.shape[0]):
            for j in range(self.t_dist.shape[0]):
                self.t_dist[i, j] = self.rate(i, j, 0)

        # Calculate cumulative transition rate (partial sums)
        for i in range(self.t_dist.shape[0]):
            for j in range(self.t_dist.shape[0]):
                if(i == 0 and j == 0):
                    self.P[i*self.t_dist.shape[0] + j] = self.t_dist[i, j]
                else:
                    self.P[i*self.t_dist.shape[0] + j] = self.P[i*self.t_dist.shape[0] + j - 1] + self.t_dist[i, j]

        # Pre-calculate constant timestep
        self.timestep = 1/self.P[-1]

        # Normalization
        self.P = self.P/self.P[-1]


    def total_energy(self):
        '''Calculates the hamiltonian for the full system.'''
        H = 0  # Initialize

        # Coulomb interaction sum
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                H += ((1 - self.occupation[i]) * (1 - self.occupation[j])
                      /self.distances[i, j])
        H *= self.I_0 * self.R

        # Add electrostatic contribution
        for i in range(self.N):
            H = H - (1 - self.occupation[i]) * self.eV_constant[i]

        return H




    @staticmethod
    def dist(ri, rj):
        '''Calculate cartesian distance between 3D vectors ri and rj'''
        return np.sqrt((ri[0] - rj[0])**2 + (ri[1] - rj[1])**2 + (ri[2] - rj[2])**2)
