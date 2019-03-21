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
        self.use_tests = len(tests)
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
             'args':{'hops':5000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.025,
             'threshold_per_test':0.005},
            {'func':"go_simulation",
             'args':{'hops':50000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.01,
             'threshold_per_test':0.002},
            {'func':"go_simulation",
             'args':{'hops':250000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.002,
             'threshold_per_test':0.0},
        ]
        self.setStrategy(0)
        self.error_func = self.average_cumulative_error
        self.initRandomPlacement()

    def setUseTests(self, N):
        self.use_tests = N
        sum = 0
        for i in range(self.use_tests):
            for _ in self.tests[i]:
                sum+=1
        self.N_tests = sum
        

    def setStrategy(self, index):
        self.current_strategy = index
        self.simulation_func = self.simulation_strategy[index]['func']
        self.simulation_args = self.simulation_strategy[index]['args']
        self.expected_error =  self.simulation_strategy[index]['expected_error']
        self.threshold_per_test =  self.simulation_strategy[index]['threshold_per_test']

    def evaluate_error(self, dn):
        #channel_indexes = []
        #go_slices = []
        diffs = []
        for j in range(self.use_tests):
            test = self.tests[j]
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

    def validate_error(self, dn):
        diffs = []
        for j in range(self.use_tests, len(self.tests)):
            test = self.tests[j]
            electrodes = test[0]
            
            for i in range(len(electrodes)):
                dn.electrodes[i][3] = electrodes[i]
            dn.update_V()
            execpted_currents = test[1]
            getattr(dn, self.simulation_func)(**self.simulation_args)
            for index, current in execpted_currents:
                diff = math.fabs(dn.current[index]-current)
                diffs.append(diff)
        return self.error_func(diffs) 
        
    def average_cumulative_error(self, diffs):
        error = 0
        for diff in diffs:
            if diff > self.expected_error:
                error+=diff-self.expected_error
        return error / len(diffs)

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
                    acceptors=self.dn.acceptors, donors=self.dn.donors, copy_from=self.dn)
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
        if a < (b+0.001) and a > (b-0.001):
            return True
        else:
            return False

    def greedySearch(self):
        best = self.evaluate_error(self.dn)
        print ("Best is %.3f"%(best))
        found = True
        while found:
            found = False
            for neighbour, _, _ in self.yieldNeighbours():
                if best < 0.001:
                    break
                error = self.evaluate_error(neighbour)
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
                    best = self.evaluate_error(self.dn)
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

    def simulatedAnnealingSearch(self, T, annealing_schedule, file_prefix):
        real_start_time = time.time()
        annealing_index = 0
        T = annealing_schedule[0][0]
        found = True
        best = self.evaluate_error(self.dn)
        print ("Best is %.3f"%(best))
        print ("strategy threshold is :%.4f"%(self.threshold_per_test*self.N_tests))
        found = True
        writer = anim.getWriter(60, "")
        acceptor_plot, donor_plot, history_acceptor_plot, history_donor_plot, text_element, fig = anim.initScatterAnimation(self.dn)
        with writer.saving(fig, "annealingSearch%s.mp4"%(file_prefix), 100):
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
                            if annealing_schedule[annealing_index][2] > self.current_strategy:
                                self.setStrategy(annealing_schedule[annealing_index][2])
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
                    error = self.evaluate_error(neighbour)
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
                        best = self.evaluate_error(self.dn)
                        print ("New strategy best is %.3f"%(best))
                        print("New strategy: %d"%(self.current_strategy))
                    else:
                        self.x_resolution /= 2
                        self.y_resolution /= 2
                        if self.donor_dist > 2:
                            self.donor_dist-=2
                        print("New resolution: %.5f"%(self.x_resolution))
                    found = True
        self.dn.saveSelf("resultDump%s.kmc"%(file_prefix))
        self.dn.go_simulation(hops=100000, record=True)
        plt.clf()
        kmc_utils.visualize_traffic(self.dn, 111, "Result")
        plt.savefig("resultDump%s.png"%(file_prefix))
        return best, self.current_strategy

    def genetic_search(self, gen_size, time_available, disparity, uniqueness, file_prefix):
        dns = []
        self.current_strategy = 0
        disparity_step = disparity / (gen_size-1)
        for i in range (gen_size):
            newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, self.dn.ydim, 
                self.dn.zdim, electrodes=self.dn.electrodes, copy_from = self.dn)
            newDn.initialize()
            setattr(newDn, "genes", self.getGeneList(newDn))
            dns.append(newDn)
        best_dn = dns[0]
        start_time = time.time()
        gen = 0
        while True:
            gen += 1
            best_error = 1000
            print ("generation: %d"%(gen))
            results = []
            total_error = 0
            for dn in dns:
                error = self.evaluate_error(dn)
                if best_error > error:
                    best_error = error
                    best_dn = dn
                total_error+=error
                results.append((error, dn))
            time_difference = time.time() - start_time
            average_error = total_error / gen_size
            print ("average error: %.4f\nbest error: %.3f"%(average_error, best_error))
            if time_difference > time_available:
                break
            results = sorted(results, key=lambda x:x[0])
            intermediate_dns = []
            current_disparity = disparity
            space = 0
            tot = 0
            for _,dn in results:
                space += current_disparity
                tot+=current_disparity
                while space >= 1:
                    intermediate_dns.append(dn)
                    space-=1
                if random.random() < space:
                    space-=1
                    intermediate_dns.append(dn)
                current_disparity-=disparity_step
            random.shuffle(intermediate_dns)
            new_generation_genes = self.getNexGenerationGenes(intermediate_dns, uniqueness)
            i = 0
            for gene in new_generation_genes:
                self.getDnFromGenes(gene, dns[i])
                i+=1
            
            
            
            if self.current_strategy < len(self.simulation_strategy)-1 \
                    and best_error < self.simulation_strategy[self.current_strategy]['threshold_per_test'] \
                    and average_error < self.simulation_strategy[self.current_strategy]['threshold_per_test']*3:
                self.setStrategy(self.current_strategy+1)
        best_dn.saveSelf("GeneticResultDump%s.kmc"%(file_prefix))
        best_dn.go_simulation(hops=1000000, record=True)
        plt.clf()
        kmc_utils.visualize_traffic(best_dn, 111, "Result")
        plt.savefig("GeneticResultDump%s.png"%(file_prefix))
        return self.validate_error(best_dn), self.current_strategy
            
    def getGeneList(self, dn):
        genes = []
        for acceptor in dn.acceptors:
            x = np.uint16(acceptor[0]/dn.xdim * 65535)
            y = np.uint16(acceptor[1]/dn.ydim * 65535)
            genes.append(x)
            genes.append(y)
        for donor in dn.donors:
            x = np.uint16(donor[0]/dn.xdim * 65535)
            y = np.uint16(donor[1]/dn.ydim * 65535)
            genes.append(x)
            genes.append(y)
        return genes
    
    def getNexGenerationGenes(self, dns, uniqueness):
        newGeneration = []
        print(len(dns))
        for i in range(len(dns)):
            if i % 2 == 0:
                j = i+1
            else:
                j = i-1 
            parent_1 = dns[i].genes
            parent_2 = dns[j].genes
            newGenes = dn_search.single_point_crossover(parent_1, parent_2)
            ok, problem = self.isAllowed(newGeneration, newGenes, uniqueness, 65)
            tries = 0
            while not ok:
                if problem == -1:
                    problem = math.floor(random.random()*len(newGenes))

                newGenes[problem] = dn_search.mutate(newGenes[problem])
                ok, problem = self.isAllowed(newGeneration, newGenes, uniqueness, 65)
                tries+=1
                if tries == 100:
                    print ("i: %d, j: %d"%(i, j))
                    print ("that does not bode well")
            newGeneration.append(newGenes)
        return newGeneration

    @staticmethod
    def mutate(a):
        rnd = math.floor(random.random()*16)
        b = np.uint16(2**rnd)
        return np.bitwise_xor(a, b)

    def isAllowed(self, prev_genes, gene, uniqueness, resolution):
        for coord in range(0, len(gene), 2):
            for coord2 in range(0, len(gene), 2):
                if coord == coord2:
                    continue
                if dn_search.uInt16Diff( gene[coord], gene[coord2]) < resolution \
                    and dn_search.uInt16Diff(gene[coord+1], gene[coord2+1]) < resolution:
                    return False, coord
        for prev_gene in prev_genes:
            total_diff = 0
            for coord in range(len(gene)):
                total_diff+=dn_search.uInt16Diff(prev_gene[coord], gene[coord])
            if total_diff < uniqueness:
                return False, -1
        return True, -1

    @staticmethod
    def uInt16Diff(a, b):
        if a < b:
            return b-a
        else:
            return a-b

    @staticmethod
    def single_point_crossover(parent_1_genes, parent_2_genes):
        genes = []
        rnd_index = round(random.random()*len(parent_1_genes))
        genes.extend(parent_1_genes[:rnd_index])
        genes.extend(parent_2_genes[rnd_index:])
        return genes

    def getDnFromGenes(self, genes, dn):

        setattr(dn, "genes", genes)

        for i in range(self.dn.N):
            x = genes[i*2]/65535*self.dn.xdim
            y = genes[i*2+1]/65535*self.dn.ydim
            dn.acceptors[i] = (x, y, 0)
        for i in range(self.dn.M):
            x = genes[self.dn.N*2+i*2]/65535*self.dn.xdim
            y = genes[self.dn.N*2+i*2+1]/65535*self.dn.ydim
            dn.donors[i] = (x, y, 0)
        dn.initialize( dopant_placement=False, charge_placement=False)
    def P(self, e, e0, T):

        if e < e0:
            return True
        elif T < 0.001:
            return False
        else:
            if random.random() < math.exp(-(e-e0)/T):
                return True
            else:
                return False

# Genetic algorithm
# 1. Evaluating a generation is trivial, we already have everything
# 2. Selecting individuals to reproduce: We used idea from literature, Tutorial
# 3. Generating new generation, we have several methods. AGain use Tutorial and Unique measurement.
#  - New random sample
#  - Getting a random neighbour of an existing individual
#  - Combining 2 samples using uniform crossover.
#  - Combining 2 samples using single-point crossover.
# 4. Each method could have some weight, which is gradually changed based on 
#    how much success they provide. Idea to test in the future. First write base-line solution, that can be tested against.
