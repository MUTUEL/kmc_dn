import math
import numpy as np
import kmc_dopant_networks as kmc_dn
import matplotlib.pyplot as plt
import kmc_dopant_networks_utils as kmc_utils
import dn_animation as anim
import copy
import random
import time

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
             'threshold_per_test':0.002},
            {'func':"go_simulation",
             'args':{'hops':250000},
             'expected_error':0.002,
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
        channel_indexes = []
        go_slices = []
        diffs = []
        for test in self.tests:
            electrodes = test[0]
            
            for i in range(len(electrodes)):
                dn.electrodes[i][3] = electrodes[i]
            dn.update_V()
            #channel_index, go_slice = dn.start_simulation_parallel()
            #dn.read_simulation_result(channel_index, go_slice)
            #channel_indexes.append(channel_index)
            #go_slices.append(go_slice)
            execpted_currents = test[1]
            getattr(dn, self.simulation_func)(**self.simulation_args)
            for index, current in execpted_currents:
                diff = math.fabs(dn.current[index]-current)
                diffs.append(diff)
        #i = 0
        #for test in self.tests:
        #    channel_index = channel_indexes[i]
        #    go_slice = go_slices[i]
        #    i+=1
        #    dn.read_simulation_result(channel_index, go_slice)
        #    execpted_currents = test[1]

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
                pos = (donor[0]+option[1], donor[1]+option[2])
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
            if self.float_equals(x, self.dn.xCoords[i]) and self.float_equals(y, self.dn.yCoords[i]):
                return False
        return True

    def float_equals(self, a, b):
        if a < (b+0.0001) and a > (b-0.0001):
            return True
        else:
            return False

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

    def simulatedAnnealingSearch(self, T, annealing_schedule):
        real_start_time = time.time()
        annealing_index = 0
        T = annealing_schedule[0][0]
        found = True
        best = self.validate_error(self.dn)
        print ("Best is %.3f"%(best))
        print ("strategy threshold is :%.4f"%(self.threshold_per_test*self.N_tests))
        found = True
        writer = anim.getWriter(60, "")
        acceptor_plot, donor_plot, history_acceptor_plot, history_donor_plot, text_element, fig = anim.initScatterAnimation(self.dn)
        with writer.saving(fig, "annealingSearch7.mp4", 100):
            anim_counter = 0
            anim_erase_history_at = 480
            while found and annealing_index < len(annealing_schedule):
                found = False
                for neighbour, index, target_pos in self.yieldNeighbours():
                    
                    current_real_time = time.time()
                    time_difference = current_real_time - real_start_time

                    if time_difference > annealing_schedule[annealing_index][1]:
                        annealing_index+=1
                        if annealing_index < len(annealing_schedule):
                            T = annealing_schedule[annealing_index][0]
                        else:
                            break
                    if T > 0 and annealing_index < len(annealing_schedule)-1:
                        T_from = annealing_schedule[annealing_index][0]
                        T_to = annealing_schedule[annealing_index+1][0]
                        if annealing_index == 0:
                            time_there = (time_difference)/annealing_schedule[annealing_index][1]
                        else:
                            time_there = (time_difference-annealing_schedule[annealing_index-1][1])/annealing_schedule[annealing_index][1]
                        T = T_from + (T_to-T_from)*time_there
                    print ("time difference: %.1f"%(time_difference))
                    if best < 0.001:
                        break
                    error = self.validate_error(neighbour)
                    print ("error %.3f, strategy: %d"%(error, self.current_strategy))
                    if self.P(error, best, T):
                        found = True
                        print ("Current best is %.3f"%(error))
                        refresh = anim_counter >= anim_erase_history_at
                        if refresh:
                            alpha = 0.5
                            anim_counter = 0
                        else:
                            anim_counter+=1
                            alpha = (anim_erase_history_at-anim_counter)/anim_erase_history_at*0.35+0.15
                        anim.animateTransition(self.dn, donor_plot, acceptor_plot, history_donor_plot, history_acceptor_plot, text_element, index, target_pos, writer, 5, refresh, alpha, "Error: %.3f, Time: %.0f sec, strategy: %d"%(best, time_difference, self.current_strategy))
                        best = error
                        self.dn = neighbour
                        break
                if not found and self.x_resolution > self.minimum_resolution:
                    print ("Best is %.4f and thershold is %.4f"%(best, self.threshold_per_test*self.N_tests))
                    if best < self.threshold_per_test*self.N_tests:
                        self.setStrategy(self.current_strategy+1)
                        best = self.validate_error(self.dn)
                        print ("New strategy best is %.3f"%(best))
                        print("New strategy: %d"%(self.current_strategy))
                    else:
                        self.x_resolution /= 2
                        self.y_resolution /= 2
                        if self.donor_dist > 2:
                            self.donor_dist-=2
                        print("New resolution: %.5f"%(self.x_resolution))
                    found = True
        self.dn.saveSelf("resultDump14.kmc")
        self.dn.go_simulation(hops=100000, record=True)
        plt.clf()
        kmc_utils.visualize_traffic(self.dn, 111, "Result")
        plt.savefig("resultDump14.png")


    def P(self, e, e0, T):

        if e < e0:
            return True
        elif T < 0.001:
            return False
        else:
            print ("Odds are: %.3f"%(math.exp(-(e-e0)/T)))
            if random.random() < math.exp(-(e-e0)/T):
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