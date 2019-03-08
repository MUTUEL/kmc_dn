import math
import numpy as np
import kmc_dopant_networks as kmc_dn
import matplotlib.pyplot as plt
import kmc_dopant_networks_utils as kmc_utils
import dn_animation as anim
import copy
import random

class dn_search():
    def __init__(self, initial_dn, tests, xdim, ydim, x_start_resolution, y_start_resolution):
        '''
        =======================
        dn_search dopant network placement search class
        ======================= 
        This class is made for performing a search to find dopant_network (dn)
        placement that matches the test data.

        Input arguments
        ===============
        initial_dn; kmc_dn
            Instance of a kmc_dn class with initial dopant placements, which is used to create
            neighbouring kmc_dn instances to perform search.
        tests; list
            This is a list of tests, each test is a tuple, which contains electrode configuration
            and expected current(s)
        '''

        self.minimum_resolution = 0.01
        self.donor_dist = 10
        self.xdim = xdim
        self.ydim = ydim
        self.x_resolution = x_start_resolution
        self.y_resolution = y_start_resolution

        self.dn = initial_dn
        self.tests = tests
        sum = 0
        for t in tests:
            for _ in t[1]:
                sum+=1
        self.N_tests = sum
        self.simulation_strategy = [
            {'func':"go_simulation",
             'args':{'hops':1000, 'goSpecificFunction':"wrapperSimulateProbability"},
             'expected_error':0.04,
             'threshold_per_test':0.005},
            {'func':"go_simulation",
             'args':{'hops':5000},
             'expected_error':0.025,
             'threshold_per_test':0.005},
            {'func':"go_simulation",
             'args':{'hops':50000},
             'expected_error':0.01,
             'threshold_per_test':0.0},
        ]
        self.setStrategy(0)
        self.error_func = self.cumulative_error
        self.initRandomPlacement()

    def setStrategy(self, index):
        self.current_strategy = index
        self.simulation_func = self.simulation_strategy[index]['func']
        self.simulation_args = self.simulation_strategy[index]['args']
        self.expected_error =  self.simulation_strategy[index]['expected_error']
        self.threshold_per_test =  self.simulation_strategy[index]['threshold_per_test']

    def validate_error(self, dn):
        for test in self.tests:
            electrodes = test[0]
            execpted_currents = test[1]
            diffs = []
            for i in range(len(electrodes)):
                dn.electrodes[i][3] = electrodes[i]
            dn.update_V()
            getattr(dn, self.simulation_func)(**self.simulation_args)
            for index, current in execpted_currents:
                diff = math.fabs(dn.current[index]-current)
                diffs.append(diff)
        return self.error_func(diffs)
        
    def cumulative_error(self, diffs):
        error = 0
        for diff in diffs:
            if diff > self.expected_error:
                error+=diff-self.expected_error
        return error

    def initRandomPlacement(self):
        #Inits random placement but within the given resolution.
        xInt = self.xdim / self.x_resolution
        yInt = self.ydim / self.y_resolution
        self.dn.xCoords = []
        self.dn.yCoords = []
        for i in range(self.dn.N):
            found = True
            x = 0
            y = 0
            while found:
                found = False
                x = np.random.randint(xInt)
                y = np.random.randint(yInt)
                for j in range(len(self.dn.xCoords)):
                    if self.dn.xCoords[j] == x and self.dn.yCoords[j] == y:
                        found = True
            self.dn.acceptors[i][0] = x * self.x_resolution 
            self.dn.acceptors[i][1] = y * self.y_resolution
            self.dn.xCoords.append(x * self.x_resolution)
            self.dn.yCoords.append(y * self.y_resolution)
        for i in range(self.dn.M):
            found = True
            x = 0
            y = 0
            while found:
                found = False
                x = np.random.randint(xInt)
                y = np.random.randint(xInt)
                for j in range(len(self.dn.xCoords)):
                    if self.dn.xCoords[j] == x and self.dn.yCoords[j] == y:
                        found = True
            self.dn.donors[i][0] = x * self.x_resolution 
            self.dn.donors[i][1] = y * self.y_resolution
            self.dn.xCoords.append(x * self.x_resolution)
            self.dn.yCoords.append(y * self.y_resolution)
        self.dn.initialize(dopant_placement=False, charge_placement=False)

    def yieldNeighbours(self):
        N = self.dn.N + self.dn.M
        shifts = [(self.x_resolution, 0), (-self.x_resolution, 0), (0, self.y_resolution), (0, -self.y_resolution), (self.x_resolution, self.y_resolution), (-self.x_resolution, self.y_resolution), (-self.x_resolution, -self.y_resolution), (self.x_resolution, -self.y_resolution)]
        options = [(i, shifts[j][0], shifts[j][1]) for i in range(N) for j in range(len(shifts))]
        indexes = [x for x in range(len(options))]
        random.shuffle(indexes)
        for index in indexes:
            option = options[index]
            if option[0] < self.dn.N:
                dopant = self.dn.acceptors[option[0]]
                pos = (dopant[0]+option[1], dopant[1]+option[2])
            else:
                donor = self.dn.donors[option[0]-self.dn.N]
                pos = (donor[0]+option[1]*self.donor_dist, donor[1]+option[2]*self.donor_dist)
            if self.doesPositionFit(pos[0], pos[1]):
                newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.xdim, 
                    self.ydim, 0, electrodes = self.dn.electrodes, 
                    acceptors=self.dn.acceptors, donors=self.dn.donors)
                newDn.xCoords = self.dn.xCoords.copy()
                newDn.yCoords = self.dn.yCoords.copy()

                newDn.xCoords[option[0]] = pos[0]
                newDn.yCoords[option[0]] = pos[1]
                if option[0] < self.dn.N:
                    newDn.acceptors[option[0]][0] = pos[0]
                    newDn.acceptors[option[0]][1] = pos[1]
                else:
                    newDn.donors[option[0]-self.dn.N][0] = pos[0]
                    newDn.donors[option[0]-self.dn.N][1] = pos[1]
                    print ("Donor position swap")
                newDn.initialize( dopant_placement=False, charge_placement=False)
                yield newDn, option[0], pos



    def doesPositionFit(self, x, y):
        if x < 0 or x > self.xdim or y < 0 or y > self.ydim:
            return False
        for i in range(len(self.dn.xCoords)):
            if x==self.dn.xCoords[i] and y==self.dn.yCoords[i]:
                return False
        return True

    def greedySearch(self):
        best = self.validate_error(self.dn)
        print ("Best is %.3f"%(best))
        found = True
        while found:
            found = False
            for neighbour, _, _ in self.yieldNeighbours():
                if best < 0.001:
                    break
                error = self.validate_error(neighbour)
                print ("error %.3f"%(error))
                if error < best:
                    found = True
                    print ("New best is %.3f"%(error))
                    best = error
                    self.dn = neighbour
                    break
            if not found and self.x_resolution > self.minimum_resolution:
                if best < self.threshold_per_test*self.N_tests:
                    self.setStrategy(self.current_strategy+1)
                    best = self.validate_error(self.dn)
                    print("New strategy: %d"%(self.current_strategy))
                else:
                    self.x_resolution /= 2
                    self.y_resolution /= 2
                    if self.donor_dist > 2:
                        self.donor_dist-=2
                    print("New resolution: %.5f"%(self.x_resolution))
                found = True
        self.dn.saveSelf("resultDump3.kmc")
        self.dn.go_simulation(hops=100000, record=True)
        plt.clf()
        kmc_utils.visualize_traffic(self.dn, 111, "Result")
        plt.savefig("resultDump3.png")

    def simulatedAnnealingSearch(self, T, cooling_period):
        time = 0.0
        found = True
        best = self.validate_error(self.dn)
        print ("Best is %.3f"%(best))
        found = True
        writer = anim.getWriter(30, "Simulated annealing search")
        acceptor_plot, donor_plot, fig = anim.initScatterAnimation(self.dn)
        with writer.saving(fig, "annealingSearch1.mp4", 100):
            while found:
                found = False
                for neighbour, index, target_pos in self.yieldNeighbours():
                    if best < 0.001:
                        break
                    error = self.validate_error(neighbour)
                    time+=1
                    print ("error %.3f"%(error))
                    if self.P(error, best, (cooling_period-time)/cooling_period*T):
                        found = True
                        print ("Current best is %.3f"%(error))
                        anim.animateTransition(self.dn, donor_plot, acceptor_plot, index, target_pos, writer, 15)
                        best = error
                        self.dn = neighbour
                        break
                if not found and self.x_resolution > self.minimum_resolution:
                    if best < self.threshold_per_test*self.N_tests:
                        self.setStrategy(self.current_strategy+1)
                        best = self.validate_error(self.dn)
                        print ("Current best is %.3f"%(error))
                        print("New strategy: %d"%(self.current_strategy))
                    else:
                        self.x_resolution /= 2
                        self.y_resolution /= 2
                        if self.donor_dist > 2:
                            self.donor_dist-=2
                        print("New resolution: %.5f"%(self.x_resolution))
                    found = True
        #self.dn.saveSelf("resultDump7.kmc")
        #self.dn.go_simulation(hops=100000, record=True)
        #plt.clf()
        #kmc_utils.visualize_traffic(self.dn, 111, "Result")
        #plt.savefig("resultDump7.png")


    def P(self, e, e0, T):
        if T < 0.001:
            return False
        if e < e0:
            return True
        else:
            if random.random() > (e-e0)/T:
                return True
            else:
                return False

# Genetic algorithm
# 1. Evaluating a generation is trivial, we already have everything
# 2. Selecting individuals to reproduce is the harder part.
#    We should take into account both the evalution score and uniqueness.
#    To take into account uniqueness we could consider for each site, 
#    how far is the nearest site and add the distances
# 3. Generating new generation, we have several methods
#  - New random sample
#  - Getting a random neighbour of an existing individual
#  - Combining 2 samples using uniform crossover.
#  - Combining 2 samples using single-point crossover.
# 4. Each method could have some weight, which is gradually changed based on 
#    how much success they provide.