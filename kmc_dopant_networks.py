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

plt.ioff()

class kmc_dn():
    '''This class is a wrapper for all functions and variables needed to perform
    a kinetic monte carlo simulation of a variable range hopping system'''

    def __init__(self, N, M, xdim, ydim, zdim, **kwargs):
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
        electrodes; electrode configuration, an Px5 np.array, where
            P is the number of electrodes, the first three columns correspond
            to the x, y and coordinates of the electrode, respectively,
            the fourth column holds the electrode voltage and the last column
            tracks the amount of carriers sourced/sinked in the electrode.
            default: np.zeros((0, 5))
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
        acceptors; Nx4 array, where first three columns are x, y, z position and
            the last column is the hole occupancy
        donors; Mx3 array, where the columns represent x, y, z position of donors.
            Donors are assumed to be always occupied.
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

        # Initialize variables
        self.N = N
        self.M = M
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim


        # Check dimensionality
        if(self.ydim == 0 and self.zdim == 0):
            self.dim = 1
        elif(self.zdim == 0):
            self.dim = 2
        else:
            self.dim = 3

        # Initialize parameter kwargs
        if('electrodes' in kwargs):
            self.electrodes = kwargs['electrodes'].copy()
        else:
            self.electrodes = np.zeros((0, 5))

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
        if('place_dopants_charges' in kwargs):
            if(kwargs['place_dopants_charges'] == 'place_dopants_charges_random'):
                self.place_dopants_charges = self.place_dopants_charges_random
        else:
            self.place_dopants_charges = self.place_dopants_charges_random

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

        if('calc_rate' in kwargs):
            if(kwargs['calc_rate'] == 'calc_rate_MA'):
                self.calc_rate = self.calc_rate_MA
        else:
            self.calc_rate = self.calc_rate_MA

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

        if('perform_event' in kwargs):
            if(kwargs['perform_event'] == 'perform_event_standard'):
                self.perform_event = self.perform_event_standard
        else:
            self.perform_event = self.perform_event_standard

        if('stopping_criterion' in kwargs):
            if(kwargs['stopping_criterion'] == 'stopping_criterion_discrete'):
                self.stopping_criterion = self.stopping_criterion_discrete
            if(kwargs['stopping_criterion'] == 'stopping_criterion_convergence'):
                self.stopping_criterion = self.stopping_criterion_convergence
        else:
            self.stopping_criterion = self.stopping_criterion_discrete

        if('callback' in kwargs):
            if(kwargs['callback'] == 'callback_standard'):
                self.callback = self.callback_standard
            if(kwargs['callback'] == 'callback_avg_carriers'):
                self.callback = self.callback_avg_carriers
            if(kwargs['callback'] == 'callback_current_vectors'):
                self.callback = self.callback_current_vectors
        else:
            self.callback = self.callback_standard

        # Initialize other attributes
        self.transitions = np.zeros((N + self.electrodes.shape[0],
                                     N + self.electrodes.shape[0]))
        self.transitions_constant = np.zeros((N + self.electrodes.shape[0],
                                     N + self.electrodes.shape[0]))
        self.transitions_possible = np.zeros((N + self.electrodes.shape[0],
                                     N + self.electrodes.shape[0]))
        self.transitions_possible = self.transitions_possible.astype(bool)
        self.distances = np.zeros((N + self.electrodes.shape[0],
                                   N + self.electrodes.shape[0]))
        self.vectors = np.zeros((N + self.electrodes.shape[0],
                                 N + self.electrodes.shape[0], 3))
        self.site_energies = np.zeros((N + self.electrodes.shape[0],))
        self.energy_differences = np.zeros((N + self.electrodes.shape[0],
                                            N + self.electrodes.shape[0]))



        #TODO: callback

        # Initialize sim object
        self.initialize()

    def initialize(self):
        '''
        Wrapper function which:
        - places acceptors/donors
        - calculates distances
        - calculates V (electrostatic potential profile)
        - calculates E_constant
        These are all methods that determine the starting position of a kmc
        simulation.'''
        self.place_dopants_charges()

        self.calc_distances()

        self.calc_transitions_constant()

        self.calc_V()

        self.calc_E_constant()

    def simulate(self, **kwargs):
        '''
        Wrapper function which performs the following simulation loop:
        while(not stopping_criterion)
        - calc_site_energies
        - calc_transitions
        - pick_event
        - perform_event
        - callback
        Any simulation you perform follows this basic loop. Choose a different
        type of simulation by specifying the specific method
        TODO: e.g.

        Possible kwargs:

        '''
        self.reset()  # Reset all relevant trackers before running a simulation

        while(not self.stopping_criterion(**kwargs)):
            self.calc_site_energies()

            self.calc_transitions()

            self.pick_event()

            self.perform_event()

            self.callback()

            self.counter += 1

    def reset(self):
        '''
        Resets all relevant trackers before running a simulation
        '''
        self.time = 0
        self.old_current = 0
        self.counter = 0
        for i in range(self.electrodes.shape[0]):
            self.electrodes[i, 4] = 0  # Reset current
        self.avg_carriers_prenorm = 0
        self.avg_carriers = 0
        self.current_vectors = np.zeros((self.transitions.shape[0], 3))
        self.traffic = np.zeros(self.transitions.shape)

    def place_dopants_charges_random(self):
        '''
        Place dopants and charges on a 3D hyperrectangular domain (xdim, ydim, zdim).
        Place N acceptors and M donors. Place N-M charges.
        Returns acceptors (Nx4 array) and donors (Mx3 array). The first three columns
        of each represent the x, y and z coordinates, respectively, of the acceptors
        and donors. The fourth column of acceptors denotes charge occupancy, with
        0 being an unoccupied acceptor and 1 being an occupied acceptor
        '''
        # Initialization
        self.acceptors = np.random.rand(self.N, 4)
        self.donors = np.random.rand(self.M, 3)

        # Place dopants
        self.acceptors[:, 0] *= self.xdim
        self.acceptors[:, 1] *= self.ydim
        self.acceptors[:, 2] *= self.zdim
        self.donors[:, 0] *= self.xdim
        self.donors[:, 1] *= self.ydim
        self.donors[:, 2] *= self.zdim

        # Place charges
        self.acceptors[:, 3] = 0  # Set occupancy to 0
        charges_placed = 0
        while(charges_placed < self.N-self.M):
            trial = np.random.randint(self.N)  # Try a random acceptor
            if(self.acceptors[trial, 3] < 1):
                self.acceptors[trial, 3] += 1  # Place charge
                charges_placed += 1

    def calc_distances(self):
        '''
        Calculates the distances between each hopping sites and stores them
        in the matrix distances.
        Also stores the unit vector in the hop direction i->j in the matrix vectors.
        '''
        for i in range(self.distances.shape[0]):
            for j in range(self.distances.shape[0]):
                if(i >= self.N and j >= self.N):
                    self.distances[i, j] = self.dist(self.electrodes[i - self.N, :3],
                                                      self.electrodes[j - self.N, :3])  # Distance electrode -> electrode
                    self.vectors[i, j] = ((self.electrodes[j - self.N, :3]
                                          - self.electrodes[i - self.N, :3])
                                          /self.distances[i, j])

                elif(i >= self.N and j < self.N):
                    self.distances[i, j] = self.dist(self.electrodes[i - self.N, :3],
                                                      self.acceptors[j, :3])  # Distance electrode -> acceptor
                    self.vectors[i, j] = ((self.acceptors[j, :3]
                                          - self.electrodes[i - self.N, :3])
                                          /self.distances[i, j])
                elif(i < self.N and j >= self.N):
                    self.distances[i, j] = self.dist(self.acceptors[i, :3],
                                                      self.electrodes[j - self.N, :3])  # Distance acceptor -> electrode
                    self.vectors[i, j] = ((self.electrodes[j - self.N, :3]
                                          - self.acceptors[i, :3])
                                          /self.distances[i, j])
                elif(i < self.N and j < self.N):
                    self.distances[i, j] = self.dist(self.acceptors[i, :3],
                                                      self.acceptors[j, :3])  # Distance acceptor -> acceptor
                    self.vectors[i, j] = ((self.acceptors[j, :3]
                                          - self.acceptors[i, :3])
                                          /self.distances[i, j])
        self.dist_plus_inf = self.distances + np.diag(np.ones(self.distances.shape[0])*np.inf)  # For vectorization later

    def calc_V(self):
        '''Numerically solve Laplace with relaxation method'''

        # Grid initialization (for now initialized in 3D for method compatibility)
        self.V = np.zeros((int(self.xdim/self.res) + 2,
                           int(self.ydim/self.res) + 2,
                           int(self.zdim/self.res) + 2))  # +2 for boundaries

        # Convert electrode coordinates to grid coordinates
        self.electrodes_grid = self.electrodes.astype(int)
        for i in range(self.electrodes.shape[0]):
            if(self.dim == 1):
                x = self.electrodes[i, 0]/self.xdim * (self.V.shape[0] - 1)
                self.electrodes_grid[i, :3] = [round(x), 0, 0]
            if(self.dim == 2):
                x = self.electrodes[i, 0]/self.xdim * (self.V.shape[0] - 1)
                y = self.electrodes[i, 1]/self.ydim * (self.V.shape[1] - 1)
                self.electrodes_grid[i, :3] = [round(x), round(y), 0]

            if(self.dim == 3):
                x = self.electrodes[i, 0]/self.xdim * (self.V.shape[0] - 1)
                y = self.electrodes[i, 1]/self.ydim * (self.V.shape[1] - 1)
                z = self.electrodes[i, 2]/self.zdim * (self.V.shape[2] - 1)
                self.electrodes_grid[i, :3] = [round(x),
                                               round(y),
                                               round(z)]

        if(self.dim == 1):
            # Boundary conditions (i.e. electrodes)
            for i in range(self.electrodes.shape[0]):
                self.V[tuple(self.electrodes_grid[i, :3])] = self.electrodes[i, 3]

            # 1D relaxation
            self.V[:, 0, 0] = self.relaxation(self.V[:, 0, 0],
                                              fixedpoints = self.electrodes_grid[:, 0])

        if(self.dim == 2):
            # Boundary conditions (i.e. electrodes)
            for i in range(self.electrodes.shape[0]):
                self.V[tuple(self.electrodes_grid[i, :3])] = self.electrodes[i, 3]

            # 1D boundary relaxation
            points = [val == 0 for val in self.electrodes_grid[:, 1]]  # Find electrodes that lie on border [:, 0, 0]
            self.V[:, 0, 0] = self.relaxation(self.V[:, 0, 0],
                                              fixedpoints = self.electrodes_grid[points, 0])
            points = [val == self.V.shape[1]-1 for val in self.electrodes_grid[:, 1]]
            self.V[:, -1, 0] = self.relaxation(self.V[:, -1, 0],
                                              fixedpoints = self.electrodes_grid[points, 0])
            points = [val == 0 for val in self.electrodes_grid[:, 0]]
            self.V[0, :, 0] = self.relaxation(self.V[0, :, 0],
                                              fixedpoints = self.electrodes_grid[points, 1])
            points = [val == self.V.shape[0]-1 for val in self.electrodes_grid[:, 0]]
            self.V[-1, :, 0] = self.relaxation(self.V[-1, :, 0],
                                              fixedpoints = self.electrodes_grid[points, 1])

            # 2D relaxation
            self.V[:, :, 0] = self.relaxation(self.V[:, :, 0],
                                              fixedpoints = self.electrodes_grid[:, :2])

        if(self.dim == 3):
            # Boundary conditions (i.e. electrodes)
            for i in range(self.electrodes.shape[0]):
                self.V[tuple(self.electrodes_grid[i, :3])] = self.electrodes[i, 3]

            # 1D boundary relaxation
            # x-y plane (z=0)
            # Border [:, 0, 0]
            points = (np.asarray([val == 0 for val in self.electrodes_grid[:, 1]])
                   &  np.asarray([val == 0 for val in self.electrodes_grid[:, 2]]))
            self.V[:, 0, 0] = self.relaxation(self.V[:, 0, 0],
                                              fixedpoints = self.electrodes_grid[points, 0])
            # Border [:, -1, 0]
            points = (np.asarray([val == self.V.shape[1]-1 for val in self.electrodes_grid[:, 1]])
                &     np.asarray([val == 0 for val in self.electrodes_grid[:, 2]]))
            self.V[:, -1, 0] = self.relaxation(self.V[:, -1, 0],
                                              fixedpoints = self.electrodes_grid[points, 0])
            # Border [0, :, 0]
            points = (np.asarray([val == 0 for val in self.electrodes_grid[:, 0]])
                &     np.asarray([val == 0 for val in self.electrodes_grid[:, 2]]))
            self.V[0, :, 0] = self.relaxation(self.V[0, :, 0],
                                              fixedpoints = self.electrodes_grid[points, 1])

            # Border [-1, :, 0]
            points = (np.asarray([val == self.V.shape[0]-1 for val in self.electrodes_grid[:, 0]])
                &     np.asarray([val == 0 for val in self.electrodes_grid[:, 2]]))
            self.V[-1, :, 0] = self.relaxation(self.V[-1, :, 0],
                                              fixedpoints = self.electrodes_grid[points, 1])

            # x-y plane (z=-1)
            # Border [:, 0, -1]
            points = (np.asarray([val == 0 for val in self.electrodes_grid[:, 1]])
                &     np.asarray([val == self.V.shape[2]-1 for val in self.electrodes_grid[:, 2]]))
            self.V[:, 0, -1] = self.relaxation(self.V[:, 0, -1],
                                              fixedpoints = self.electrodes_grid[points, 0])
            # Border [:, -1, -1]
            points = (np.asarray([val == self.V.shape[1]-1 for val in self.electrodes_grid[:, 1]])
                &     np.asarray([val == self.V.shape[2]-1 for val in self.electrodes_grid[:, 2]]))
            self.V[:, -1, -1] = self.relaxation(self.V[:, -1, -1],
                                              fixedpoints = self.electrodes_grid[points, 0])
            # Border [0, :, -1]
            points = (np.asarray([val == 0 for val in self.electrodes_grid[:, 0]] )
                &     np.asarray([val == self.V.shape[2]-1 for val in self.electrodes_grid[:, 2]]))
            self.V[0, :, -1] = self.relaxation(self.V[0, :, -1],
                                              fixedpoints = self.electrodes_grid[points, 1])
            # Border [-1, :, -1]
            points = (np.asarray([val == self.V.shape[0]-1 for val in self.electrodes_grid[:, 0]])
                &     np.asarray([val == self.V.shape[2]-1 for val in self.electrodes_grid[:, 2]]))
            self.V[-1, :, -1] = self.relaxation(self.V[-1, :, -1],
                                              fixedpoints = self.electrodes_grid[points, 1])

            # Remaining four borders (z != 0 or -1)
            # Border [0, 0, :]
            points = (np.asarray([val == 0 for val in self.electrodes_grid[:, 0]])
                &     np.asarray([val == 0 for val in self.electrodes_grid[:, 1]]))
            self.V[0, 0, :] = self.relaxation(self.V[0, 0, :],
                                              fixedpoints = self.electrodes_grid[points, 2])
            # Border [0, -1, :]
            points = (np.asarray([val == 0 for val in self.electrodes_grid[:, 0]])
                &     np.asarray([val == self.V.shape[1]-1 for val in self.electrodes_grid[:, 1]]))
            self.V[0, -1, :] = self.relaxation(self.V[0, -1, :],
                                              fixedpoints = self.electrodes_grid[points, 2])
            # Border [-1, 0, :]
            points = (np.asarray([val == self.V.shape[0]-1 for val in self.electrodes_grid[:, 0]])
                &     np.asarray([val == 0 for val in self.electrodes_grid[:, 1]]))
            self.V[-1, 0, :] = self.relaxation(self.V[-1, 0, :],
                                              fixedpoints = self.electrodes_grid[points, 2])
            # Border [-1, -1, :]
            points = (np.asarray([val == self.V.shape[0]-1 for val in self.electrodes_grid[:, 0]])
                &     np.asarray([val == self.V.shape[1]-1 for val in self.electrodes_grid[:, 1]]))
            self.V[-1, -1, :] = self.relaxation(self.V[-1, -1, :],
                                              fixedpoints = self.electrodes_grid[points, 2])

            #2D boundary relaxation
            # Plane [:, :, 0]
            points = [val == 0 for val in self.electrodes_grid[:, 2]]
            self.V[:, :, 0] = self.relaxation(self.V[:, :, 0],
                                              fixedpoints = self.electrodes_grid[points, :2])
            # Plane [:, :, -1]
            points = [val == self.V.shape[2]-1 for val in self.electrodes_grid[:, 2]]
            self.V[:, :, -1] = self.relaxation(self.V[:, :, -1],
                                              fixedpoints = self.electrodes_grid[points, :2])
            # Plane [0, :, :]
            points = [val == 0 for val in self.electrodes_grid[:, 0]]
            self.V[0, :, :] = self.relaxation(self.V[0, :, :],
                                              fixedpoints = self.electrodes_grid[points, 1:3])
            # Plane [-1, :, :]
            points = [val == self.V.shape[0]-1 for val in self.electrodes_grid[:, 0]]
            self.V[-1, :, :] = self.relaxation(self.V[-1, :, :],
                                              fixedpoints = self.electrodes_grid[points, 1:3])
            # Plane [:, 0, :]
            points = [val == 0 for val in self.electrodes_grid[:, 1]]
            fixedpoints = np.delete(self.electrodes_grid[points, :3], 1, 1)
            self.V[:, 0, :] = self.relaxation(self.V[:, 0, :], fixedpoints = fixedpoints)
            # Plane [:, -1, :]
            points = [val == self.V.shape[1]-1 for val in self.electrodes_grid[:, 1]]
            fixedpoints = np.delete(self.electrodes_grid[points, :3], 1, 1)
            self.V[:, -1, :] = self.relaxation(self.V[:, -1, :], fixedpoints = fixedpoints)

            #3D relaxation
            self.V = self.relaxation(self.V, fixedpoints = self.electrodes_grid[:, :3])

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
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)), 0, 0]

            if(self.dim == 2):
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)),
                                                    int(round(y)), 0]

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
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)), 0, 0]

            if(self.dim == 2):
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)),
                                                    int(round(y)), 0]

            if(self.dim == 3):
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 3) + 1
                z = self.acceptors[i, 2]/self.zdim * (self.V.shape[2] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)),
                                                    int(round(y)),
                                                    int(round(z))]

            # Add compensation
            self.comp_constant[i] += -self.e**2/(4 * np.pi * self.eps) * sum(
                    1/self.dist(self.acceptors[i, :3], self.donors[k, :3]) for k in range(self.donors.shape[0]))

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
        self.occupation_repeat = np.repeat(self.acceptors[:, 3].reshape((1, self.N)), self.N, 0)

        # Vectorized calculation of acceptor site energies
        presum = (1 - self.occupation_repeat)/self.dist_plus_inf[:self.N, :self.N]
        self.site_energies[:self.N] = (-self.e**2/(4 * np.pi * self.eps) *np.sum(presum, axis = 1)
                                    + self.E_constant[:self.N])


    def calc_transitions(self):
        '''
        Calculates the matrix transitions by calling the calc_rate method.
        Caution: run calc_site_energies before this method, otherwise it will
        calculate the transition based on old site energies.
        '''
        # Loop over possible hops from site i -> site j
        for i in range(self.transitions.shape[0]):
            for j in range(self.transitions.shape[0]):
                if(not self.transition_possible(i, j)):
                    self.transitions[i, j] = 0
                else:
                    dE = self.energy_difference(i, j)
                    self.transitions[i, j] = self.calc_rate(i, j, dE)

        # Raise flag if a fixed point (i.e. transitions = 0) is reached
        if(not np.any(self.transitions)):
            return 1
        else:
            return 0

    def calc_transitions_MA(self):
        '''
        Calculates the matrix transitions, following the MA hopping rate.
        It uses the method calc_energy_differences().
        transitions_constant = nu*exp(-2r_ij/a)
        transitions_constant also makes sure any transition rate to the same
        element is zero
        '''
        # Calculate energy differences
        self.calc_energy_differences()

        # Calculate MA rates
        rates = np.exp(-self.energy_differences/self.k*self.T)

        # Get a treshold mask for all values >= 1
        treshold_mask = rates >= 1

        # Calculate masked rates
        rates = (1 - treshold_mask)*rates + treshold_mask

        # Calculate possible transitions
        self.calc_transitions_possible()

        # Calculate final transitions by adding constant and MA rate, masked by
        # transitions possible
        self.transitions = self.transitions_possible*self.transitions_constant*rates

    def callback_standard(self):
        '''
        This is the standard callback function, tracking average carriers
        and current vectors.
        '''
        self.callback_avg_carriers()
        self.callback_current_vectors()

    def callback_avg_carriers(self):
        '''
        Tracks the average number of carriers in the system
        '''
        self.avg_carriers_prenorm += self.hop_time * sum(self.acceptors[:, 3])
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


    def pick_event_standard(self):
        '''
        Based on the transition matrix self.transitions, pick a hopping event.
        '''
        # Get the cumulative sum of flattened array transitions
        self.problist = np.cumsum(self.transitions.flatten())

        # Calculate hopping time
        self.hop_time = np.random.exponential(scale=1/self.problist[-1])

        # Normalization
        self.problist = self.problist/self.problist[-1]

        # Find transition index of random event
        event = min(np.where(self.problist >= np.random.rand())[0])

        # Convert to acceptor/electrode indices
        self.transition = [int(event/self.transitions.shape[0]),
                           event%self.transitions.shape[0]]

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

    def perform_event_standard(self):
        '''
        Performs the event transition (with the standard algorithm).
        '''
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
        self.time += self.hop_time

    def stopping_criterion_discrete(self, **kwargs):
        '''
        Stop when reaching the amount of hops specified
        Possible kwargs:
        hops; the amount of hops to perform
            default: 1000
        '''
        #TODO: maybe put this somewhere else, now it is run every time
        if('hops' in kwargs):
            hops = kwargs['hops']
        else:
            hops = 1000

        if(self.counter < hops):
            return False
        else:
            return True

    def stopping_criterion_convergence(self, **kwargs):
        '''
        Stop when the current through all electrodes has converged to some
        tolerance
        Possible kwargs:
        tol; tolerance for convergence
        interval; the amount of hopping events before rechecking the current
        '''
        #TODO: maybe put this somewhere else, now it is run every time
        if('tol' in kwargs):
            tol = kwargs['tol']
        else:
            tol = 1E-2
        if('interval' in kwargs):
            interval = kwargs['interval']
        else:
            interval = 1000

        # Calculate currents
        if(self.time != 0 and self.counter%interval == 0):
            self.current = self.electrodes[:, 4]/self.time

            # Check convergence
            if(np.linalg.norm(self.current, 2) == 0):
                if(np.linalg.norm(self.old_current - self.current, 2) == 0):
                    return True
                else:
                    return False
            elif(np.linalg.norm(self.old_current - self.current, 2)/np.linalg.norm(self.current,2) < tol):
                return True
            else:
                self.old_current = self.current.copy()  # Store current
                return False
        else:
            return False


    def simulate_tsigankov(self, interval = 500, tol = 1E-3):
        '''Perform a kmc simulation with the Tsigankov algorithm (from
        Tsigankov2003)'''
        # Initialization
        self.old_current = np.ones((self.electrodes.shape[0]))
        self.current = np.zeros((self.electrodes.shape[0]))
        self.time = 0  # Reset simulation time
        self.avg_carriers = []  # Tracks average number of carriers per interval
        self.avg_current = []  # Tracks average current per interval
        for i in range(self.electrodes.shape[0]):
            self.electrodes[i, 4] = 0  # Reset current
        self.calc_t_dist()


        # Simulation loop
        converged = False
        counter = 0  # Counts the amount of intervals needed for convergence
        self.old_current *= np.inf
        self.prev_time = self.time  # Timestamp of previous interval

        while(not converged):
            self.avg_carriers.append(0)  # Add entry to average carrier tracker
            self.avg_current.append(0)  # Add entry to average current tracker
            for i in range(interval):
                # Hopping event
                self.pick_event_tsigankov()

                # Update number of particles
                self.avg_carriers[counter] += self.timestep * sum(self.acceptors[:, 3])

            # Update average trackers
            self.avg_carriers[counter] /= (self.time - self.prev_time)
            self.avg_current[counter] /= (self.time - self.prev_time)

            # Calculate currents
            self.current = self.electrodes[:, 4]/self.time

            # Check convergence
            if(np.linalg.norm(self.current, 2) == 0
               and np.linalg.norm(self.old_current - self.current, 2) == 0):
                converged = True
            elif(np.linalg.norm(self.old_current - self.current, 2)/np.linalg.norm(self.current,2) < tol):
                converged = True
            else:
                self.old_current = self.current.copy()  # Store current

            counter += 1
            self.prev_time = self.time

        print('Converged in '
              + str(counter)
              + ' intervals of '
              + str(interval)
              + ' hops ('
              + str(counter*interval)
              + ' total hops)'
              )

    #%% Miscellaneous methods
    def calc_transitions_possible(self):
        '''
        Calculates the boolean matrix transitions_possible.
        if transitions_possible[i, j] is True a transition is possible.
        '''
        # Re-initialize transitions_possible as True
        self.transitions_possible[:, :] = True

        # Set electrode -> electrode hops to False
        self.transitions_possible[self.N:, self.N:] = False

        for i in range(self.N):
            # Set unoccied -> anything False
            if(self.acceptors[i, 3] == 0):
                self.transitions_possible[i, :] = False
            # Set anything -> occupied False
            if(self.acceptors[i, 3] == 1):
                self.transitions_possible[:, i] = False

    def transition_possible(self, i, j):
        '''Check if a hop from i -> j is possible. Returns True if transition is
        allowed, otherwise returns False'''
        possible = True
        if(i >= self.N and j >= self.N):
            possible = False  # No transition electrode -> electrode
        elif(i >= self.N and j < self.N):
            if(self.acceptors[j, 3] == 1):
                possible = False  # No transition electrode -> occupied
        elif(i < self.N and j >= self.N):
            if(self.acceptors[i, 3] == 0):
                possible = False  # No transition empty -> electrode
        elif(i == j):
            possible = False  # No transition to same acceptor
        elif(i < self.N and j < self.N):
            if(self.acceptors[i, 3] == 0
               or self.acceptors[j, 3] == 1):
                possible = False  # No transition empty -> or -> occupied
        return possible

    def calc_energy_differences(self):
        '''Calculate the energy difference for a hop i -> j based on the array
        site_energies. The class attribute coulomb_interactions indicates whether
        the method for calculating the site energies incorporates acceptor-acceptor
        interaction.
        Stores the energy differences in the array energy_differences'''
        se = np.tile(self.site_energies.reshape((1, self.N + self.electrodes.shape[0])),
                     (self.N + self.electrodes.shape[0], 1))
        self.energy_differences = se - se.transpose()

        if(self.coulomb_interactions):
            self.energy_differences[:self.N, :self.N] -= self.e**2/(4 * np.pi * self.eps*self.dist_plus_inf[:self.N, :self.N])

    def calc_t_dist(self):
        '''Calculates the transition rate matrix t_dist, which is based only
        on the distances between sites (as defined in Tsigankov2003)'''
        # Initialization
        self.t_dist = np.zeros((self.N + self.electrodes.shape[0],
                                self.N + self.electrodes.shape[0]))
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


    def calc_rate_MA(self, i, j, dE):
        '''Calculate the transition rate for hop i->j with the Miller-Abrahams
        rate, using the energy difference dE.'''
        if(dE > 0):
            transition_rate = self.nu*np.exp(-2*self.distances[i, j]/self.ab
                                                    - dE/(self.k*self.T))
        else:
            transition_rate = self.nu*np.exp(-2*self.distances[i, j]/self.ab)

        return transition_rate

    def total_energy(self):
        '''Calculates the hamiltonian for the full system.'''
        H = 0  # Initialize

        # Coulomb interaction sum
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                H += ((1 - self.acceptors[i, 3]) * (1 - self.acceptors[j, 3])
                      /self.distances[i, j])
        H *= self.e**2/(4*np.pi*self.eps)

        # Add electrostatic contribution
        for i in range(self.N):
            H = H - (1 - self.acceptors[i, 3]) * self.eV_constant[i]

        return H


    def validate_boltzmann(self, hops = 1000, n = 2, points = 100, V_0 = 1):
        '''Perform validation of a simulation algorithm by means of checking
        whether it obeys boltzmann statistics. Hops is the total amount of hops
        performed and n equals the (constant!) number of carriers in the system.
        points; the amount of points in convergence array
        V_0; chemical potential'''
        # Initialize
        hops_array = np.zeros(points)
        interval = hops/points

        # Prepare system
        self.electrodes = np.zeros((0, 5))  # Remove electrodes from system
        self.site_energies = np.zeros((self.N + self.electrodes.shape[0],))  # Reset site energies parameter
        self.V[:, :, :] = V_0 # Set chemical potential
        self.calc_E_constant()  # Update electrostatic energy contribution
        self.transitions = np.zeros((self.N, self.N))

        # Make microstate array
        perm_array = np.zeros((self.N))
        perm_array[:n] = 1
        perms = list(itertools.permutations(perm_array))  # All possible permutations
        self.microstates = np.asarray(list(set(perms)))  # Remove all duplicate microstates

        # Assign energies to microstates
        self.E_microstates = np.zeros(self.microstates.shape[0])
        for i in range(self.E_microstates.shape[0]):
            self.acceptors[:, 3] = self.microstates[i]  # Create microstate
            self.E_microstates[i] = self.total_energy()

        # Calculate theoretical probabilities
        self.p_theory = np.zeros(self.microstates.shape[0])
        self.Z = 0  # Partition function
        for i in range(self.p_theory.shape[0]):
            self.p_theory[i] = np.exp(-self.E_microstates[i]/(self.k*self.T))
            self.Z += self.p_theory[i]
        self.p_theory = self.p_theory/self.Z  # Normalize

        # Simulate probabilities
        self.time = 0  # Reset simulation time
        previous_microstate = np.random.randint(self.microstates.shape[0])  # Random first microstate
        self.acceptors[:, 3] = self.microstates[previous_microstate]
        self.p_sim = np.zeros(self.microstates.shape[0])
        p_sim_interval = np.zeros((self.microstates.shape[0], points))

        # Simulation loop
        interval_counter = 0
        for i in range(hops):
            # Hopping event
            self.calc_site_energies()
            self.calc_transitions()
            self.pick_event()
            self.perform_event()

            # Save time spent in previous microstate
            self.p_sim[previous_microstate] += self.hop_time

            # Find index of current microstate
            for j in range(self.microstates.shape[0]):
                if(np.array_equal(self.acceptors[:, 3], self.microstates[j])):
                    previous_microstate = j
                    break

            # Save probabilities each interval
            if(i >= (interval_counter+1)* interval - 1):
                p_sim_interval[:, interval_counter] = self.p_sim/self.time
                hops_array[interval_counter] = i + 1
                interval_counter += 1

        self.p_sim /= self.time  # Normalize end probability

        # Calculate norm
        convergence = np.linalg.norm(self.p_sim - self.p_theory)/np.linalg.norm(self.p_theory)
        print('Norm of difference: ' + str(convergence))

        return hops_array, p_sim_interval

    def visualize(self, show_occupancy = True, show_V = True):
        '''Returns a figure which shows the domain with potential profile. It
        also show all dopants with acceptor occupancy.'''
        if(self.dim == 2):
            # Initialize figure
            fig = plt.figure()
            plt.axis('scaled')
            ax = fig.add_subplot(111)
            ax.set_xlim(right=self.xdim)
            ax.set_ylim(top=self.ydim)

            # Plot potential profile
            if(show_V):
                V_profile = ax.imshow(self.V[:, :, 0].transpose(),
                                      interpolation='bicubic',
                                      origin='lower',
                                      extent=(0, self.xdim, 0, self.ydim))
                fig.colorbar(V_profile)


            if(show_occupancy):
                # Plot impurity configuration (red = 2, orange = 1, black = 0 holes)
                colors = ['red' if i==2
                          else 'orange' if i==1
                          else 'black' for i in self.acceptors[:, 3]]
                ax.scatter(self.acceptors[:, 0], self.acceptors[:, 1], c = colors, marker='o')

                ax.scatter(self.donors[:, 0], self.donors[:, 1], marker='x')
            else:
                ax.scatter(self.acceptors[:, 0], self.acceptors[:, 1], color = 'black', marker='o')

                ax.scatter(self.donors[:, 0], self.donors[:, 1], marker='x')

            ax.set_xlabel('x (a.u.)')
            ax.set_ylabel('y (a.u.)')


        return fig

    def visualize_current(self):
        '''Returns a figure which shows the domain with potential profile and
        occupancy. Plot vectors which correlate to the net hop direction.'''
        fig = self.visualize(show_occupancy = False)

        x = self.acceptors[:, 0]
        y = self.acceptors[:, 1]
        norm = np.linalg.norm(self.current_vectors[:self.N], axis = 1)
        u = self.current_vectors[:self.N, 0]/norm
        v = self.current_vectors[:self.N, 1]/norm

        quiv = fig.axes[0].quiver(x, y, u, v, norm, cmap=plt.cm.inferno)
        fig.colorbar(quiv)

        return fig

    def energy_difference(self, i, j):
        '''Calculate the energy difference for a hop i -> j based on the matrix
        site_energies. The class attribute coulomb_interactions indicates whether
        the method for calculating the site energies incorporates acceptor-acceptor
        interaction.'''
        if(i >= self.N or j >= self.N):
            dE = self.site_energies[j] - self.site_energies[i]  # No Coulomb interaction for electrode hops
        elif(self.coulomb_interactions):
            dE = (self.site_energies[j] - self.site_energies[i]
                  - self.e**2/(4 * np.pi * self.eps*self.distances[i, j]))
        else:
            dE = ej - ei
        return dE



    @staticmethod
    def dist(ri, rj):
        '''Calculate cartesian distance between 3D vectors ri and rj'''
        return np.sqrt((ri[0] - rj[0])**2 + (ri[1] - rj[1])**2 + (ri[2] - rj[2])**2)

    @staticmethod
    def relaxation(A, alpha=1, tol=1E-3, fixedpoints = np.asarray([])):
        '''Perform relaxation method on 1, 2 or 3D vector A. alpha is the
        relaxation parameter and tol the tolerance for convergence.
        fixedpoints is an N x dim np.ndarray with coordinates that will never be
        updated.'''
        # Check if fixedpoints is a numpy array
        if(type(fixedpoints) != np.ndarray):
            print('fixedpoints must be a numpy array!')
            return
        dim = len(A.shape)
        B = A.copy()  # Otherwise method changes object A

        if(dim == 1):
            # Initialization
            B_old = B + 1

            # Relaxation loop
            while((np.linalg.norm(B - B_old)
                    /np.linalg.norm(B)) > tol):
                B_old = B.copy()  # Store previous V

                # Loop over internal elements
                for i in range(1, A.shape[0]-1):
                    B[i] = alpha * 1/2 * (B_old[i-1] + B_old[i+1])

                # Loop over fixed points
                for i in range(fixedpoints.shape[0]):
                    B[int(fixedpoints[i])] = B_old[int(fixedpoints[i])]

        if(dim == 2):
            # Initialization
            B_old = B + 1

            # Relaxation loop
            while((np.linalg.norm(B - B_old)
                    /np.linalg.norm(B)) > tol):
                B_old = B.copy()  # Store previous V

                # Loop over internal elements
                for i in range(1, A.shape[0]-1):
                    for j in range(1, A.shape[1]-1):
                        B[i, j] = alpha * 1/4 * (B_old[i-1, j] + B_old[i+1, j]
                                                 + B_old[i, j-1] + B_old[i, j+1])

                # Loop over fixed points
                for i in range(fixedpoints.shape[0]):
                    B[tuple(fixedpoints[i])] = B_old[tuple(fixedpoints[i])]

        if(dim == 3):
            # Initialization
            B_old = B + 1

            # Relaxation loop
            while((np.linalg.norm(B - B_old)
                    /np.linalg.norm(B)) > tol):
                B_old = B.copy()  # Store previous V

                # Loop over internal elements
                for i in range(1, A.shape[0]-1):
                    for j in range(1, A.shape[1]-1):
                        for k in range(1, A.shape[2]-1):
                            B[i, j, k] = alpha * 1/6 * (B_old[i-1, j, k]
                                                        + B_old[i+1, j, k]
                                                        + B_old[i, j-1, k]
                                                        + B_old[i, j+1, k]
                                                        + B_old[i, j, k-1]
                                                        + B_old[i, j, k+1])

                # Loop over fixed points
                for i in range(fixedpoints.shape[0]):
                    B[tuple(fixedpoints[i])] = B_old[tuple(fixedpoints[i])]

        return B
