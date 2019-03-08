'''
================================================
Kinetic Monte Carlo for dopant networks (kmc_dn)
================================================
This file defines a class, kmc_dn, that is an implementation of
a kinetic Monte Carlo simulation algorithm for dopant networks.
In particular, simulations on an acceptor-doped material with
compensating donors can be performed on a domain surrounded by an
arbitrary number of electrodes. Documentation is available both
in docstrings in this file, as well as on GitHub
(https://github.com/brambozz/kmc_dn).

TODO list
=========
#TODO: Implement Tsigankov mixed algorithm and perform validation

@author: Bram de Wilde (b.dewilde-1@student.utwente.nl)
@author: Indrek Klanberg ((i.klanberg@student.utwente.nl)
'''

# Imports
import sys
sys.path.insert(0,'./goSimulation')
from goSimulation.pythonBind import callGoSimulation
import numpy as np
from numba import jit
import fenics as fn
import logging
import pickle


@jit(nopython=True, cache=True)
def _simulate_discrete_record(N_acceptors, N_electrodes, nu, kT, I_0, R, 
                              time, occupation, distances, E_constant, 
                              site_energies, transitions_constant,
                              transitions, problist, electrode_occupation,
                              hops, record=False):
    '''
    This function performs hops hopping events, meaning:
    - Calculate site energies; this is done by adding the acceptor-
        acceptor interaction to the constant energy per site E_constant.
    - Calculate transition matrix; uses site_energies and
        transitions_constant and implements the MA rate.
    - Pick and perform event; uses the transition matrix
        and repeat this hops times.

    Input arguments
    ---------------

    Returns
    -------

    '''
    # Initialize current and traffic array
    N_sites = N_acceptors+N_electrodes
    traffic = np.zeros(transitions_constant.shape)
    occupations_in_time = np.zeros(len(occupation))
    time = 0

    for _ in range(hops):
        # Calculate site_energies
        for i in range(N_acceptors):
            acceptor_interaction = 0
            site_energies[i] = E_constant[i]
            for j in range(N_acceptors):
                if j is not i:
                    acceptor_interaction += (1 - occupation[j])/distances[i, j]

            site_energies[i] += -I_0*R*acceptor_interaction

        # Calculate transitions
        for i in range(N_sites):
            for j in range(N_sites):
                if(not _transition_possible(i, j, N_acceptors, occupation)):
                    transitions[i, j] = 0
                else:
                    if(i < N_acceptors and j < N_acceptors):
                        dE = site_energies[j] - site_energies[i] \
                             - I_0*R/distances[i, j]
                    else:
                        dE = site_energies[j] - site_energies[i]

                    # Calculate MA rate
                    if(dE > 0):
                        transitions[i, j] = nu*np.exp(-dE/kT)
                    else:
                        transitions[i, j] = nu

        # Add constant part of transitions
        transitions = transitions_constant*transitions

        # pick_event
        # Transform transitions matrix into cumulative sum
        for i in range(N_sites):
            for j in range(N_sites):
                if(i == 0 and j == 0):
                    problist[0] = transitions[i, j]
                else:
                    problist[(N_sites)*i + j] = transitions[i, j] \
                                                + problist[(N_sites)*i + j-1]

        # Calculate hopping time
        hop_time = np.random.exponential(scale=1/problist[-1])

        # Normalization of probability list
        problist = problist/problist[-1]

        # Find transition index of random event
        event = np.random.rand()
        for i in range((N_sites)**2):
            if(problist[i] >= event):
                event = i
                break

        # Convert event to acceptor/electrode indices
        transition = [int(event/(N_sites)), int(event%(N_sites))]

        # Perform hop
        if(transition[0] < N_acceptors):  # Hop from acceptor
            occupation[transition[0]] = False
        else:  # Hop from electrode
            electrode_occupation[transition[0] - N_acceptors] -= 1
        if(transition[1] < N_acceptors):  # Hop to acceptor
            occupation[transition[1]] = True
        else:  # Hop to electrode
            electrode_occupation[transition[1] - N_acceptors] += 1

        # Update record
        if record: 
            traffic[transition[0], transition[1]] += 1
            for i in range(len(occupation)):
                if occupation[i]:
                    occupations_in_time[i]+=hop_time

        # Increment time
        time += hop_time

    return time, occupation, electrode_occupation, traffic, occupations_in_time

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

class kmc_dn():
    def __init__(self, N, M, xdim, ydim, zdim, mu = 0, **kwargs):
        '''
        =======================
        kmc_dn simulation class
        =======================
        This class is capable of performing kinetic Monte Carlo
        simulations on a 1D/2D/3D domain of acceptor sites, surrounded
        by an arbitary number of electrodes. The physical modelling is
        based on the Miller-Abrahams formalism of variable range hopping
        conduction and the simulation algorithm is a rather standard
        one. A more detailed description of the model may be found
        in my (Bram de Wilde) master thesis, which will be published
        on the GitHub page for this project.

        Input arguments
        ===============
        N; int
            number of acceptors.
        M; int
            number of donors.
        xdim; float
            x dimension of domain.
        ydim; float
            y dimension of domain, if ydim = 0 then domain is 1D.
        zdim; float
            z dimension of domain, if zdim = 0 then domain is 2D.
        mu; float
            chemical potential along those parts of the domain border
            where no electrodes are defined.

        Possible keyword arguments
        ==========================
        electrodes; Px4 float np.array
            Represents the electrode layout, where P is the number of
            electrodes, the first three columns correspond to the x, y
            and z coordinates of the electrode, respectively.
            The fourth column holds the electrode voltage.
            default: np.zeros((0, 4))
        static_electrodes; Px4 float np.array
            Represents the electrode layout, where P is the number of
            electrodes, the first three columns correspond to the x, y
            and z coordinates of the electrode, respectively.
            The fourth column holds the electrode voltage.
            These electrodes do not allow any current transport and
            only influence the potential profile V.
            default: np.zeros((0, 4))
        res; float
            Resolution used for solving the chemical potential profile
            with fenics. E.g., if (xdim, ydim) = (1, 1) and res = 0.1,
            the domain would be split into hundred elements.
            default: min[xdim, ydim, zdim]/100
        calc_E_constant; string
            Determines the choice of the calc_E_constant method.
            Possible options are:
            'calc_E_constant_V'; include only local chemical potential
            'calc_E_constant_V_comp'; include local chemical potential
                and compensation sites.
            default: calc_E_constant_V_comp

        Class attributes
        ================
        Here follows a list of all class attributes that were not
        previously mentioned, but are used in simulation.

        Constants
        ~~~~~~~~~
        dim; int
            Represents the dimensionality of the system (either 1, 2
            or 3).
        nu; float
            Attempt frequency for hopping events
        kT; float
            Energy corresponding to the temperature (i.e. Boltzmann 
            constant multiplied by temperature)
        I_0; float
            The interaction energy for two acceptors separated by a 
            distance R, i.e. I_0 = e**2/(4*pi*eps) * 1/R
        U; interaction energy for double occupancy (EXPERIMENTAL/UNIMPLEMENTED)
        ab; float
            Bohr radius/localization radius
        R; float
            The average distance between acceptors. This is evaluated
            by R = N^(-1/dim), where N is the acceptor density

        Chemical potential related attributes
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        V; function
            The chemical potential in the domain as a callable function.
            E.g. in a (1, 1) domain V(0.5, 0.5) would give the chemical
            potential in the center of the domain.
        fn_electrodes; dict
            A dictionary holding strings to define fn_expression.
        fn_expression; string
            A string containing the expression that defines the fenics
            boundary condition.
        fn_boundary; fenics Expression
        fn_mesh; fenics Mesh
        fn_functionspace; fenics FunctionSpace
            Represents the functionspace in which fenics finds a
            solution
        fn_bc; fenics BC
        fn_v; fenics TestFunction
        fn_a;
        fn_f;
        fn_L;

        Other attributes
        ~~~~~~~~~~~~~~~~
        time; float
            Simulation time.
        counter; int
            Counts the number of performed hops.
        acceptors; Nx3 float np.array
            Represents the acceptor layout. The columns correspond to
            the x, y and z coordinates, respectively.
        donors; Mx3 float np.array
            Represents the donor layout. The columns correspond to
            the x, y and z coordinates, respectively.
        occupation; (N,) bool np.array
            Indicates hole occupancy of each acceptor.
        electrode_occupation; (P,) int np.array
            Indicates the amount of holes sourced/sinked from each
            electrode. A negative value means sourcing, i.e. holes
            leaving the electrode into the system.
        distances; (N+P)x(N+P) float np.array
            Contains the distance between all sites, i.e. both acceptors
            and electrodes.
        transitions; (N+P)x(N+P) float np.array
            Contains the transition rate for each pair of sites. I.e.
            transitions[i, j] is the transition rate for hop i->j.
        transitions_constant; (N+P)x(N+P) float np.array
            Contains the constant (i.e. position dependent) contribution
            to the transitions array.
        vectors; (N+P)x(N+P)x3 float np.array
            vectors[i, j] is the unit vector pointing from site i to
            site j.
        E_constant; (N,) float np.array
            Contains energy contributions that are constant throughout
            one simulation run and is (by default) equal to the sum of
            eV_constant and comp_constant.
        site_energies; (N+P,) float np.array
            Contains the potential energy of each site.
        hop_time; float
            Time increment of the last hop.
        transition; list
            [i, j], where i and j correspond to the sites for the hop
            i->j.
        current; (P,) float np.array
            Contains the current, i.e. electrode_occupation/time, for
            each electrode.
        traffic; (N+P)x(N+P) int np.array
            Contains for each possible transition i->j the amount of
            hops that occured during simulation in traffic[i, j]
        average_occupation; (N,) float np.array
            Contains for each hopping site the average occupation as a
            fraction of the simulation time

        Class methods
        =============
        Below follows a short description of each method in this class.
        For detailed documentation, refer to the individual docstrings
        of each method.
        #TODO
        '''
        # Constants
        self.nu = 1  # Hop attempt frequency (1/s)
        self.kT = 1  # Temperature energy
        self.I_0 = self.kT  # Interaction energy
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

        # Set dimensonless variables to 1
        self.ab = self.R

        # Initialize parameter kwargs
        if('electrodes' in kwargs):
            self.electrodes = kwargs['electrodes'].copy()
            self.P = self.electrodes.shape[0]
        else:
            self.electrodes = np.zeros((0, 4))
            self.P = 0

        if 'acceptors' in kwargs:
            self.acceptors = kwargs['acceptors'].copy()
        if 'donors' in kwargs:
            self.donors = kwargs['donors'].copy()

        

        if('static_electrodes' in kwargs):
            self.static_electrodes = kwargs['static_electrodes'].copy()
        else:
            self.static_electrodes = np.zeros((0, 4))


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
            
        # Initialize sim object
        self.initialize(dopant_placement=(not hasattr(self, 'acceptors')), charge_placement=(not hasattr(self, 'donors')))


    def initialize(self, dopant_placement = True, charge_placement = True, 
                   distances = True, V = True, E_constant = True):
        '''
        Wrapper function which:
        - Initializes various attributes
        - Places acceptors/donors
        - Places charges
        - Calculates distances
        - Calculates constant part of transitions
        - Calculates V (electrostatic potential profile)
        - Calculates E_constant
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

    def reset(self):
        '''
        Resets all relevant trackers before running a simulation.
        In particular it resets:
        - Simulation time and hop counter
        - Simulation current, i.e. electrode_occupation

        Importantly, this function does NOT reset the occupation of
        acceptors. This is important if you want to run e.g. 10 
        simulations of the same system, but want them to be physically
        sequential. It would not make sense to randomize the placement
        then, since the equilibrium amount of charge carriers could
        be different.
        '''
        self.time = 0
        self.old_current = 0
        self.counter = 0
        self.electrode_occupation = np.zeros(self.P, dtype=int)

    def go_simulation(self, hops = 1E5, prehops = 0, 
                      goSpecificFunction="wrapperSimulateRecord", 
                      record=False):
        '''
        Perform a simulation with the go implementation.
        
        Input arguments
        ---------------
        prehops; int
            The amount of hops performed before tracking the current.
            This can be used to bring the system closer to equilibrium
            before actual simulation.
        hops; int
            The amount of hops performed to simulate.
        goSpecificFunction; string
            Specify the specific go function that is used to simulate.
        record; bool
            If True, the simulation will also keep track of traffic
            and the time each site is occupied.

        Output arguments (see main docstring for definition)
        ----------------
        kmc_dn.time
        kmc_dn.occupation
        kmc_dn.electrode_occupation
        kmc_dn.current
        if(record):
            kmc_dn.traffic
            kmc_dn.average_occupation
        '''
        self.makeSimulation(simulateFunction = callGoSimulation, 
                            preHopFunction = callGoSimulation, 
                            hops = hops, prehops = prehops, 
                            goSpecificFunction=goSpecificFunction, 
                            record=record)
    
    def python_simulation(self, hops = 1E5, prehops = 0, 
                          record = False):
        '''
        Perform a simulation with the python implementation.
        
        Input arguments
        ---------------
        prehops; int
            The amount of hops performed before tracking the current.
            This can be used to bring the system closer to equilibrium
            before actual simulation.
        hops; int
            The amount of hops performed to simulate.
        record; bool
            If True, the simulation will also keep track of traffic
            and the time each site is occupied.

        Output arguments (see main docstring for definition)
        ----------------
        kmc_dn.time
        kmc_dn.occupation
        kmc_dn.electrode_occupation
        kmc_dn.current
        if(record):
            kmc_dn.traffic
            kmc_dn.average_occupation
        '''
        self.makeSimulation(simulateFunction=_simulate_discrete_record, 
                            preHopFunction=_simulate_discrete_record, 
                            hops=hops, prehops=prehops, record=record)

    def makeSimulation(self, simulateFunction = None, preHopFunction = None, 
                       prehops = 0, hops = 1E5, record = False, 
                       goSpecificFunction = None):
        '''
        A wrapper function that allows the user to simulate with either
        python or go. Examples can be found in self.python_simulation()
        and self.go_simulation()
        '''
        # Initialize simulation methods
        if simulateFunction != None:
            self.simulate_func = simulateFunction
        if preHopFunction != None:
            self.simulate_prehop = preHopFunction

        
        # Reset current and time
        self.reset()     

        # Initialize simulation arguments
        args = {"N_acceptors":self.N, "N_electrodes":self.P, 
                "nu":self.nu, "kT":self.kT, "I_0":self.I_0, 
                "R":self.R, "time":self.time, "occupation":self.occupation, 
                "distances":self.distances,
                "E_constant":self.E_constant, 
                "site_energies":self.site_energies, 
                "transitions_constant":self.transitions_constant,
                "transitions":self.transitions, "problist":self.problist, 
                "electrode_occupation":self.electrode_occupation, 
                "record":False}

        if goSpecificFunction != None:
            args["goSpecificFunction"] = goSpecificFunction
    

        # Simulate prehops
        if(prehops != 0):
            args["hops"] = prehops
            self.simulate_prehop(**args)
            self.reset()
            args['electrode_occupation']=self.electrode_occupation
        args["hops"] = hops

        # Simulate hops
        if record:
            args["record"] = True

            # Simulate
            (self.time, self.occupation, 
             self.electrode_occupation, 
             self.traffic, occupations_in_time) = self.simulate_func(**args)

            # Calculate quantities
            self.average_occupation= [x / self.time for x in occupations_in_time]
            self.current = self.electrode_occupation/self.time
        else:
            if(self.simulate_func == _simulate_discrete_record):
                (self.time, 
                 self.occupation, 
                 self.electrode_occupation,
                 _,
                 _) = self.simulate_func(**args)
            else:
                (self.time, 
                 self.occupation, 
                 self.electrode_occupation) = self.simulate_func(**args)

            # Calculate quantities
            self.current = self.electrode_occupation/self.time
        

    def place_dopants_random(self):
        '''
        Place dopants and charges on a 3D hyperrectangular domain (xdim, ydim, zdim).
        Place N acceptors and M donors. 
        Returns acceptors (Nx4 array) and donors (Mx3 array). The first three columns
        of each represent the x, y and z coordinates, respectively, of the acceptors
        and donors. 
        '''
        # Initialization
        self.acceptors = np.random.rand(self.N, 3)
        self.donors = np.random.rand(self.M, 3)

        # Place dopants
        self.acceptors[:, 0] *= self.xdim
        self.acceptors[:, 1] *= self.ydim
        self.acceptors[:, 2] *= self.zdim
        self.donors[:, 0] *= self.xdim
        self.donors[:, 1] *= self.ydim
        self.donors[:, 2] *= self.zdim

    def place_charges_random(self):
        '''
        Places N-M holes on the acceptor sites.
        This is done purely randomly.
        '''
        # Empty charges
        self.occupation = np.zeros(self.N, dtype=bool)

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
        # Turn off log messages
        fn.set_log_level(logging.WARNING)

        # Put electrode positions and values in a dict
        self.fn_electrodes = {}
        for i in range(self.P):
            self.fn_electrodes[f'e{i}_x'] = self.electrodes[i, 0]
            if(self.dim > 1):
                self.fn_electrodes[f'e{i}_y'] = self.electrodes[i, 1]
            self.fn_electrodes[f'e{i}'] = self.electrodes[i, 3]
        for i in range(self.static_electrodes.shape[0]):
            self.fn_electrodes[f'es{i}_x'] = self.static_electrodes[i, 0]
            if(self.dim > 1):
                self.fn_electrodes[f'es{i}_y'] = self.static_electrodes[i, 1]
            self.fn_electrodes[f'es{i}'] = self.static_electrodes[i, 3]


        # Define boundary expression string
        self.fn_expression = ''
        if(self.dim == 1):
            for i in range(self.P):
                self.fn_expression += (f'x[0] == e{i}_x ? e{i} : ')
            for i in range(self.static_electrodes.shape[0]):
                self.fn_expression += (f'x[0] == es{i}_x ? es{i} : ')

        if(self.dim == 2):
            surplus = self.xdim/10  # Electrode modelled as point +/- surplus
            #TODO: Make this not hardcoded
            for i in range(self.P):
                if(self.electrodes[i, 0] == 0 or self.electrodes[i, 0] == self.xdim):
                    self.fn_expression += (f'x[0] == e{i}_x && '
                                           f'x[1] >= e{i}_y - {surplus} && '
                                           f'x[1] <= e{i}_y + {surplus} ? e{i} : ')
                else:
                    self.fn_expression += (f'x[0] >= e{i}_x - {surplus} && '
                                           f'x[0] <= e{i}_x + {surplus} && '
                                           f'x[1] == e{i}_y ? e{i} : ')
            for i in range(self.static_electrodes.shape[0]):
                if(self.static_electrodes[i, 0] == 0 or self.static_electrodes[i, 0] == self.xdim):
                    self.fn_expression += (f'x[0] == es{i}_x && '
                                           f'x[1] >= es{i}_y - {surplus} && '
                                           f'x[1] <= es{i}_y + {surplus} ? es{i} : ')
                else:
                    self.fn_expression += (f'x[0] >= es{i}_x - {surplus} && '
                                           f'x[0] <= es{i}_x + {surplus} && '
                                           f'x[1] == es{i}_y ? es{i} : ')

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
        for i in range(self.static_electrodes.shape[0]):
            self.fn_electrodes[f'es{i}'] = self.static_electrodes[i, 3]

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
                self.eV_constant[i] += self.V(self.acceptors[i, 0])

            if(self.dim == 2):
                self.eV_constant[i] += self.V(self.acceptors[i, 0], self.acceptors[i, 1])

            if(self.dim == 3):
                x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
                y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 3) + 1
                z = self.acceptors[i, 2]/self.zdim * (self.V.shape[2] - 3) + 1
                self.eV_constant[i] += self.e*self.V[int(round(x)),
                                                    int(round(y)),
                                                    int(round(z))]

        self.E_constant = self.eV_constant

        # Calculate electrode energies
        self.site_energies[self.N:] = self.electrodes[:, 3]

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
                self.eV_constant[i] += self.V(self.acceptors[i, 0])

            if(self.dim == 2):
                self.eV_constant[i] += self.V(self.acceptors[i, 0], self.acceptors[i, 1])

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
        self.site_energies[self.N:] = self.electrodes[:, 3]


    #%% Load methods
    def load_acceptors(self, acceptors):
        '''
        This function loads an acceptor layout.
        It also recalculates R, as the number of acceptors might have 
        changed and it sets ab/R to 1. This is important, because you
        might have to re set it afterwards.
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

        # Set dimensonless variables to 1
        self.ab = self.R

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

    def saveSelf(self, fileName):
        '''
        Save the entire class object as a .kmc file. Also saves
        the current value as the attribute expected_current.
        This function can be used to generate test cases, or simply
        to save a simulation object for future reference.
        '''
        setattr(self, "expected_current", self.current)
        with open(fileName, "wb") as f:
            d = {}
            for key in dir(self):
                attr = getattr(self, key)
                if isinstance(attr, (list, tuple, int, float, np.ndarray)):
                    d[key] = getattr(self, key)
            pickle.dump(d, f)
    
    def loadSelf(self, fileName):
        '''
        Load a .kmc object save with self.saveSelf()
        '''
        with open(fileName, "rb") as f:
            d = pickle.load(f)
            for key in d:
                setattr(self, key, d[key])

    #%% Miscellaneous methods

    def total_energy(self):
        '''
        #TODO: Include donor-acceptor terms.

        Calculates the hamiltonian for the full system.
        '''
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



    #%% Unimplemented/unsupported section
    def calc_t_dist(self):
        '''
        UNIMPLEMENTED: This is previous work on the algorithm described
        by Tsikgankov. Might be revisited at some point, but is left
        here for refecence.

        Calculates the transition rate matrix t_dist, which is based only
        on the distances between sites (as defined in Tsigankov2003)
        '''
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

    def pick_event_tsigankov(self):
        '''
        UNIMPLEMENTED: This is previous work on the algorithm described
        by Tsikgankov. Might be revisited at some point, but is left
        here for refecence.

        Pick a hopping event based on t_dist and accept/reject it based on
        the energy dependent rate
        '''
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
                
                
