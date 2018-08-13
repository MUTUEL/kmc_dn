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

@author: Bram de Wilde (b.dewilde-1@student.utwente.nl)
'''

import numpy as np

class kmc_dn():
    '''This class is a wrapper for all functions and variables needed to perform
    a kinetic monte carlo simulation of a variable range hopping system'''

    def __init__(self, N, M, xdim, ydim,
                 electrodes, res = 'unspecified'):
        '''Upon initialization of this class, the impurities and charges are placed.
        They are placed inside a rectangle of xdim by ydim. There are N acceptors
        and M donors.

        ---Input arguments---
        N; number of acceptors
        M; number of donors
        xdim; x dimension length of domain
        ydim; y dimension length of domain
        electrodes; electrode configuration, an Px4 np.array, where
            P is the number of electrodes, the first and second column correspond
            to the x and y coordinate of the electrode, the third column holds
            the electrode voltage and the last column track the amount of
            carriers sourced/sinked in the electrode.
        res; resolution used for potential landscape calculation

        ---Class attributes---
        
        '''

        # Constants
        self.e = 1.602E-19  # Coulomb
        self.eps = 11.68  # Relative permittivity
        self.nu = 1
        self.k = 1
        self.T = 300
        self.ab = 1
        self.U = 5/8 * 1/self.ab   # J
        self.time = 0  # s

        # Initialize variables
        self.N = N
        self.M = M
        self.xdim = xdim
        self.ydim = ydim
        if(res == 'unspecified'):
            self.res = min([xdim, ydim])/100
        else:
            self.res = res
        self.transitions = np.zeros((N + electrodes.shape[0],
                                     N + electrodes.shape[0]))

        # Place acceptors and donors
        self.place_dopants_charges()

        # Place electrodes
        self.electrodes = electrodes

        # Calculate electrostatic potential profile
        self.electrostatic_landscape()

        # Calculate constant energy terms (potential and compensation)
        self.constant_energy()

    def place_dopants_charges(self):
        '''
        Place dopants and charges on a rectangular domain (xdim, ydim).
        Place N acceptors and M donors. Place N-M charges.
        Returns acceptors (Nx3 array) and donors (Nx2 array). The first two columns
        of each represent the x and y coordinate, respectively, of the acceptors
        and donors. The third column of acceptors denotes charge occupancy, with
        0 being an unoccupied acceptor and 1 being an occupied acceptor
        '''
        # Initialization
        self.acceptors = np.random.rand(self.N, 3)
        self.donors = np.random.rand(self.M, 2)

        # Place dopants
        self.acceptors[:, 0] *= self.xdim
        self.acceptors[:, 1] *= self.ydim
        self.donors[:, 0] *= self.xdim
        self.donors[:, 1] *= self.ydim

        # Place charges
        self.acceptors[:, 2] = 0  # Set occupancy to 0
        charges_placed = 0
        while(charges_placed < self.N-self.M):
            trial = np.random.randint(self.N)  # Try a random acceptor
            if(self.acceptors[trial, 2] < 2):
                self.acceptors[trial, 2] += 1  # Place charge
                charges_placed += 1


    def electrostatic_landscape(self):
        '''Numerically solve Laplace with relaxation method'''
        # Parameters
        alpha = 1  # Relaxation parameter (between 1 and 2, 1.2-1.5 works best)
        # Grid initialization
        self.V = np.zeros((int(self.xdim/self.res) + 2,
                           int(self.ydim/self.res) + 2))  # +2 for boundaries
        V_old = np.ones((int(self.xdim/self.res) + 2,
                         int(self.ydim/self.res) + 2))

        # Boundary conditions (i.e. electrodes)
        for i in range(self.electrodes.shape[0]):
            x = self.electrodes[i, 0]/self.xdim * (self.V.shape[0] - 1)
            y = self.electrodes[i, 1]/self.ydim * (self.V.shape[1] - 1)
            self.V[int(round(x)), int(round(y))] = self.electrodes[i, 2]

        # Relaxation loop
        while(np.linalg.norm(self.V - V_old) > 0.01):
            V_old = self.V.copy()
            # Loop over internal elements
            for i in range(1, self.V.shape[0]-1):
                for j in range(1, self.V.shape[1]-1):
                    self.V[i, j] = alpha *1/4 * (self.V[i-1, j] + self.V[i+1, j] +
                                          self.V[i, j-1] + self.V[i, j+1])

    def constant_energy(self):
        '''Solve the constant energy terms for each acceptor site. These
        are terms due to the electrostatic landscape and the donor atoms.'''
        # Initialization
        self.E_constant = np.zeros((self.N, 1))

        for i in range(self.acceptors.shape[0]):
            # Add electrostatic potential
            x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 3) + 1
            y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 3) + 1
            self.E_constant[i] += self.e*self.V[int(round(x)), int(round(y))]

            # Add compensation
            self.E_constant[i] += -self.e**2/(4 * np.pi * self.eps) * sum(
                    1/self.dist(self.acceptors[i, :2], self.donors[k, :2]) for k in range(self.donors.shape[0]))


    def update_transition_matrix(self):
        '''Updates the transition matrix based on the current occupancy of
        acceptors'''
        # Loop over possible hops from site i -> site j
        for i in range(self.transitions.shape[0]):
            for j in range(self.transitions.shape[0]):

                # Check if transition is possible
                possible = True
                if(i >= self.N and j >= self.N):
                    possible = False  # No transition electrode -> electrode
                elif(i >= self.N and j < self.N):
                    if(self.acceptors[j, 2] == 2):
                        possible = False  # No transition electrode -> occupied
                elif(i < self.N and j >= self.N):
                    if(self.acceptors[i, 2] == 0):
                        possible = False  # No transition empty -> electrode
                elif(i == j):
                    possible = False  # No transition to same acceptor
                elif(i < self.N and j < self.N):
                    if(self.acceptors[i, 2] == 0
                       or self.acceptors[j, 2] == 2):
                        possible = False  # No transition empty -> or -> occupied



                if(not possible):
                    self.transitions[i, j] = 0
                else:
                    # Calculate ei
                    if(i >= self.N):
                        ei = 0  # Hop from electrode into system
                    else:
                        # Acceptor interaction loop
                        acc_int = 0
                        for k in range(self.acceptors.shape[0]):
                            if(k != i):
                                acc_int += ((1 - self.acceptors[k, 2])
                                            /self.dist(self.acceptors[i, :2],
                                                       self.acceptors[k, :2]))
                        ei = (self.e**2/(4 * np.pi * self.eps) * acc_int
                            + self.E_constant[i])
                        if(self.acceptors[i, 2] == 2):
                            ei += self.U

                    # Calculate ej
                    if(j >= self.N):
                        ej = 0  # Hop to electrode
                    else:
                        # Acceptor interaction loop
                        acc_int = 0
                        for k in range(self.acceptors.shape[0]):
                            if(k != j):
                                acc_int += ((1 - self.acceptors[k, 2])
                                        /self.dist(self.acceptors[j, :2],
                                                   self.acceptors[k, :2]))
                        ej = (self.e**2/(4 * np.pi * self.eps) * acc_int
                            + self.E_constant[j])
                        if(self.acceptors[j, 2] == 1):
                            ej += self.U

                    # Calculate energy difference
                    if(i >= self.N or j >= self.N):
                        eij = ej - ei  # No Coulomb interaction for electrode hops
                    else:
                        eij = ej - ei + (self.e**2/(4 * np.pi * self.eps)
                                    /self.dist(self.acceptors[i, :2],
                                               self.acceptors[j, :2]))

                    # Calculate hopping distance
                    if(i >= self.N):  # Hop from electrode
                        hop_dist = self.dist(self.electrodes[i-self.N, :2],
                                             self.acceptors[j, :2])
                    elif(j >= self.N):  # Hop to electrode
                        hop_dist = self.dist(self.acceptors[i, :2],
                                             self.electrodes[j-self.N, :2])
                    else:  # Hop acceptor -> acceptor
                        hop_dist = self.dist(self.acceptors[i, :2],
                                             self.acceptors[j, :2])

                    # Calculate transition rate
                    if(eij < 0):
                        self.transitions[i, j] = self.nu*np.exp(-2*hop_dist/self.ab
                                                                - eij/(self.k*self.T))
                    else:
                        self.transitions[i, j] = self.nu*np.exp(-2*hop_dist/self.ab)


    def pick_event(self):
        '''Based in the transition matrix self.transitions, pick a hopping event'''
        # Initialization
        self.P = np.zeros((self.transitions.shape[0]**2))  # Probability list

        # Calculate cumulative transition rate (partial sums)
        for i in range(self.transitions.shape[0]):
            for j in range(self.transitions.shape[0]):
                if(i == 0 and j == 0):
                    self.P[i*self.transitions.shape[0] + j] = self.transitions[i, j]
                else:
                    self.P[i*self.transitions.shape[0] + j] = self.P[i*self.transitions.shape[0] + j - 1] + self.transitions[i, j]

        # Normalization
        self.P = self.P/self.P[-1]

        # Randomly determine event
        event = np.random.rand()

        # Find transition index
        event = min(np.where(self.P >= event)[0])

        # Convert to acceptor/electrode indices
        self.transition = [int(np.floor(event/self.transitions.shape[0])),
                           int(event%self.transitions.shape[0])]

        # Perform hop
        if(self.transition[0] < self.N):  # Hop from acceptor
            self.acceptors[self.transition[0], 2] -= 1
        else:  # Hop from electrode
            self.electrodes[self.transition[0] - self.N, 3] -= 1
        if(self.transition[1] < self.N):  # Hop to acceptor
            self.acceptors[self.transition[1], 2] += 1
        else:
            self.electrodes[self.transition[1] - self.N, 3] += 1

        # Increment time
        self.time += 1/self.transitions[self.transition[0], self.transition[1]]


    def simulate(self, interval = 500, tol = 1E-3):
        '''Performs a kmc simulation and checks current convergence after [interval]
        hop events.'''
        # Initialization
        self.old_current = np.ones((self.electrodes.shape[0]))
        self.current = np.zeros((self.electrodes.shape[0]))
        self.time = 0  # Reset simulation time
        for i in range(self.electrodes.shape[0]):
            self.electrodes[i, 3] = 0  # Reset current


        # Simulation loop
        converged = False
        self.old_current *= np.inf
        while(not converged):
            for i in range(interval):
                # Hopping event
                self.update_transition_matrix()
                self.pick_event()

            # Calculate currents
            self.current = self.electrodes[:, 3]/self.time

            # Check convergence
            if(np.linalg.norm(self.old_current - self.current, 2)/np.linalg.norm(self.current,2) < tol):
                converged = True
            else:
                self.old_current = self.current.copy()  # Store current
            print(self.current)




    @staticmethod
    def dist(ri, rj):
        '''Calculate cartesian distance between 2D vectors ri and rj'''
        return np.sqrt((ri[0] - rj[0])**2 + (ri[1] - rj[1])**2)
