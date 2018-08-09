'''
Class definition file for the kinetic monte carlo simulations on the 
dopant network system.
'''

import numpy as np

class kmc_dn():
    '''This class is a wrapper for all functions and variables needed to perform 
    a kinetic monte carlo simulation of a variable range hopping system'''
    
    def __init__(self, N, M, xdim, ydim, 
                 electrodes, res = 'unspecified'):
        '''Upon initialization of this class, the impurities and charges are placed.
        They are placed inside a rectangle of xdim by ydim. There are N acceptors
        and M donors.'''
        
        # Constants
        self.e = 1.602E-19  # Coulomb
        self.eps = 11.68  # Relative permittivity
        self.U = 1  # J
        self.nu = 1
        self.k = 1
        self.T = 300
        
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
            x = self.acceptors[i, 0]/self.xdim * (self.V.shape[0] - 2) + 1
            y = self.acceptors[i, 1]/self.ydim * (self.V.shape[1] - 2) + 1
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
                
                if(self.acceptors[i, 2] != 0 
                   and self.acceptors[j, 2] != 2
                   and not(i > N and j > N)):
                    # Calculate ei
                    if(i > N):
                        ei = 0  # Hop from electrode into system
                    else:
                        # Acceptor interaction loop
                        acc_int = 0
                        for k in range(self.acceptors.shape[0]):
                            if(k != i):
                                acc_int += (1 - self.acceptors[k, 2])
                                        /self.dist(self.acceptors[i, :2], 
                                                   self.acceptors[k, :2])           
                        ei = self.e**2/(4 * np.pi * self.eps) * acc_int 
                            + self.E_constant[i]
                        if(acceptors[i, 2] == 2):
                            ei += U
                    
                    # Calculate ej
                    if(j > N):
                        ej = 0  # Hop to electrode
                    else:
                        # Acceptor interaction loop
                        acc_int = 0
                        for k in range(self.acceptors.shape[0]):
                            if(k != j):
                                acc_int += (1 - self.acceptors[k, 2])
                                        /self.dist(self.acceptors[j, :2], 
                                                   self.acceptors[k, :2])         
                        ej = self.e**2/(4 * np.pi * self.eps) * acc_int
                            + self.E_constant[j]
                        if(acceptors[j, 2] == 1):
                            ej += U
                    
                    # Calculate transition rate
                    eij = ej - ei + self.e**2/(4 * np.pi * self.eps)
                                    /self.dist(self.acceptors[j, :2], 
                                               self.acceptors[k, :2])
                    if(eij < 0):
                        self.transitions[i, j] = self.nu*np.exp(-2*self.dist(self.acceptors[j, :2], 
                                                                                self.acceptors[k, :2])
                                                                /self.ab
                                                                -eij/(self.k*self.T))
                    else:
                        self.transitions[i, j] = self.nu*np.exp(-2*self.dist(self.acceptors[j, :2], 
                                                                                self.acceptors[k, :2])
                                                                /self.ab)
                else:
                    self.transitions[i, j] = 0
                    
        
    @staticmethod        
    def dist(ri, rj):
        '''Calculate cartesian distance between 2D vectors ri and rj'''
        return np.sqrt((ri[0] - rj[0])**2 + (ri[1] - rj[1])**2)

