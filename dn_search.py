import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_utils
import dn_animation as anim

import numpy as np
import matplotlib.pyplot as plt

import copy
import random
import time
import math

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
             'threshold_error':0.005},
            {'func':"go_simulation",
             'args':{'hops':5000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.025,
             'threshold_error':0.005},
            {'func':"go_simulation",
             'args':{'hops':50000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.01,
             'threshold_error':0.002},
            {'func':"go_simulation",
             'args':{'hops':250000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.002,
             'threshold_error':0.0},
        ]
        self.setStrategy(0)
        self.error_func = self.average_cumulative_error
        self.initRandomPlacement()
        self.genetic_allowed_overlap = 65
        self.order_distance_function = dn_search.degreeDistance

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
        self.threshold_error =  self.simulation_strategy[index]['threshold_error']

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
        cur_strat = self.current_strategy
        self.setStrategy(len(self.simulation_strategy)-1)
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
        self.setStrategy(cur_strat)
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


    def randomSearch(self, time_budget):
        self.setStrategy(len(self.simulation_strategy)-1)
        errors = []
        vals = []
        diffs = []
        best = self.evaluate_error(self.dn)
        bestDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, self.dn.ydim, 
                self.dn.zdim, electrodes=self.dn.electrodes, copy_from = self.dn)
        real_start_time = time.time()
        time_difference = time.time() - real_start_time
        while time_difference < time_budget:
            print (time_difference)
            self.dn.initialize()
            error = self.evaluate_error(self.dn)
            val = self.validate_error(self.dn)
            vals.append(val)
            diffs.append(math.fabs(error-val))
            if error < best:
                self.copyDnFromBtoA(bestDn, self.dn)
                best = error
            errors.append(error)
            time_difference = time.time() - real_start_time
        return bestDn, errors, vals, diffs



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
                if best < self.threshold_error*self.N_tests:
                    self.setStrategy(self.current_strategy+1)
                    best = self.evaluate_error(self.dn)
                    print("New strategy: %d"%(self.current_strategy))
                else:
                    self.x_resolution /= 2
                    self.y_resolution /= 2
                    print("New resolution: %.5f"%(self.x_resolution))
                found = True
        self.dn.saveSelf("resultDump3.kmc")
        self.dn.go_simulation(hops=100000, record=True)
        plt.clf()
        kmc_utils.visualize_traffic(self.dn, 111, "Result")
        plt.savefig("resultDump3.png")


#Sunykated abbeakubf searcg
    def simulatedAnnealingSearch(self, T, annealing_schedule, file_prefix, validation_timestep=3600, animate=True):
        real_start_time = time.time()
        annealing_index = 0
        next_validation = validation_timestep
        validations = []
        T = annealing_schedule[0][0]
        found = True
        best = self.evaluate_error(self.dn)
        abs_best = best
        print ("Best is %.3f"%(best))
        print ("strategy threshold is :%.4f"%(self.threshold_error*self.N_tests))
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
                    if time_difference > next_validation:
                        validation_error = self.validate_error(self.dn)
                        validations.append((validation_error, best, time_difference))
                        next_validation+=validation_timestep    

                    if time_difference > annealing_schedule[annealing_index][1]:
                        annealing_index+=1
                        if annealing_index < len(annealing_schedule):
                            T = annealing_schedule[annealing_index][0]
                            if annealing_schedule[annealing_index][2] > self.current_strategy:
                                self.setStrategy(annealing_schedule[annealing_index][2])
                                best = self.evaluate_error(self.dn)
                                if best < abs_best:
                                    abs_best = best
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
                        if animate:
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
                        if best < self.threshold_error:
                            self.setStrategy(self.current_strategy+1)
                            best = self.evaluate_error(self.dn)
                            print ("New strategy best is %.3f"%(best))
                            print("New strategy: %d"%(self.current_strategy))
                        break
                if not found and self.x_resolution > self.minimum_resolution:
                    print ("Best is %.4f and thershold is %.4f"%(best, self.threshold_error))

                    self.x_resolution /= 2
                    if hasattr(self, "y_resolution"):
                        self.y_resolution /= 2
                    print("New resolution: %.5f"%(self.x_resolution))
                    found = True
        self.dn.saveSelf("resultDump%s.kmc"%(file_prefix))

        validation_error = self.validate_error(self.dn)
        validations.append((validation_error, best, time_difference))
        self.dn.go_simulation(hops=100000, record=True)
        plt.clf()
        kmc_utils.visualize_traffic(self.dn, 111, "Result")
        plt.savefig("resultDump%s.png"%(file_prefix))
        print ("\nAbs best is %.3f\n"%(abs_best))
        return best, self.current_strategy, validations


#Genetic search
    def genetic_search(self, gen_size, time_available, disparity, uniqueness, 
            file_prefix, cross_over_function, mut_pow=1, order_center=None, 
            u_schedule = None):
        dns = []
        validation_timestep = time_available / 10
        self.current_strategy = 0
        disparity_sum = 0
        preserved_top = 4 - (gen_size % 2)
        cross_over_gen_size = gen_size - preserved_top
        for i in range(cross_over_gen_size):
            disparity_sum+= math.fabs(disparity * ((1-(i+0.5)/cross_over_gen_size)**(disparity-1)))
        print (disparity_sum)
        disparity_offset = (cross_over_gen_size - disparity_sum) / cross_over_gen_size
        print (disparity_offset)
        for i in range (gen_size):
            newDn = self.getRandomDn()
            setattr(newDn, "genes", self.getGenes(newDn))
            dns.append(newDn)
        best_dn = dns[0]
        start_time = time.time()
        next_validation = validation_timestep
        validations = []
        gen = 0
        if u_schedule is not None:
            us_i = 0
            us_from = uniqueness
            us_start_time = 0
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
            if u_schedule is not None:
                if u_schedule[us_i][1] > time_difference:
                    us_from = u_schedule[us_i][0]
                    us_i += 1
                uniqueness = us_from + (u_schedule[us_i]-us_from)\
                    *(time_difference-us_start_time)/(u_schedule[us_i][1])
            average_error = total_error / gen_size
            print ("average error: %.4f\nbest error: %.3f"%(average_error, best_error))
            if time_difference > next_validation:
                cur_str = self.current_strategy
                self.setStrategy(len(self.simulation_strategy)-1)
                validation_error = self.validate_error(best_dn)
                tmp_error = self.evaluate_error(best_dn)
                validations.append((validation_error, tmp_error, time_difference))
                self.setStrategy(cur_str)
                next_validation+=validation_timestep
            if time_difference > time_available:
                break
            results = sorted(results, key=lambda x:x[0])
            intermediate_dns = []
            new_generation_genes = []
            for i in range(preserved_top):
                new_generation_genes.append(results[i][1].genes) 

            i = 0
            space = 0
            tot = 0
            for _,dn in results:
                space_for_index = math.fabs(disparity * ((1-(i+0.5)/cross_over_gen_size)**(disparity-1))) + disparity_offset
                i+=1
                space += space_for_index
                tot+=space_for_index
                while space >= 1:
                    intermediate_dns.append(dn)
                    space-=1
                if random.random() < space:
                    space-=1
                    intermediate_dns.append(dn)
                if i >= cross_over_gen_size:
                    break
            random.shuffle(intermediate_dns)
            new_generation_genes.extend(self.getNextGenerationGenes(intermediate_dns, uniqueness, cross_over_function, mut_pow))
            tmp_sum = sum([new_generation_genes[y][x] for x in range(len(new_generation_genes[0])) for y in range(len(new_generation_genes))])
            i = 0
            for gene in new_generation_genes:
                self.getDnFromGenes(gene, dns[i], order_center)
                i+=1
            tmp_sum2 = sum([new_generation_genes[y][x] for x in range(len(new_generation_genes[0])) for y in range(len(new_generation_genes))])
            assert tmp_sum == tmp_sum2            
            
            if self.current_strategy < len(self.simulation_strategy)-1 \
                    and best_error < self.simulation_strategy[self.current_strategy]['threshold_error'] \
                    and average_error < self.simulation_strategy[self.current_strategy]['threshold_error']*3:
                self.setStrategy(self.current_strategy+1)
        best_dn.saveSelf("GeneticResultDump%s.kmc"%(file_prefix))
        cur_str = self.current_strategy
        self.setStrategy(len(self.simulation_strategy)-1)
        tmp_error = self.evaluate_error(best_dn)
        validation_error = self.validate_error(best_dn)
        validations.append((validation_error, tmp_error, time_difference))
        self.setStrategy(cur_str)
        best_dn.go_simulation(hops=1000000, record=True)
        plt.clf()
        kmc_utils.visualize_traffic(best_dn, 111, "Result")
        plt.savefig("GeneticResultDump%s.png"%(file_prefix))
        return best_error, self.current_strategy, validations



    def getRandomDn(self):
        newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, self.dn.ydim, 
                self.dn.zdim, electrodes=self.dn.electrodes, copy_from = self.dn)
        return newDn


    def getGenes(self, dn):
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
    

    def getNextGenerationGenes(self, dns, uniqueness, cross_over_function, power=1):
        newGeneration = []
        for i in range(len(dns)):
            if i % 2 == 0:
                j = i+1
            else:
                j = i-1 
            if j >= len(dns):
                break
            parent_1 = dns[i].genes
            parent_2 = dns[j].genes
            newGenes = cross_over_function(parent_1, parent_2)
            ok, problem = self.isAllowed(newGeneration, newGenes, uniqueness, self.genetic_allowed_overlap)
            tries = 0
            while not ok:
                if problem == -1:
                    problem = math.floor(random.random()*len(newGenes))

                newGenes[problem] = dn_search.mutate(newGenes[problem], power)
                ok, problem = self.isAllowed(newGeneration, newGenes, uniqueness, 65)
                tries+=1
                if tries == 100:
                    print ("i: %d, j: %d"%(i, j))
                    print ("that does not bode well")
            newGeneration.append(newGenes)
        return newGeneration


    @staticmethod
    def mutate(a, power):
        rnd = math.floor(random.random()**power*16)
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


    def singlePointCrossover(self, parent_1_genes, parent_2_genes):
        genes = []
        rnd_index = round(random.random()*len(parent_1_genes))
        genes.extend(parent_1_genes[:rnd_index])
        genes.extend(parent_2_genes[rnd_index:])
        return genes


    def alteredTwoPointCrossOver(self, parent_1_genes, parent_2_genes):
        genes = []
        rnd_index = round(random.random()*self.dn.N*2)
        rnd_index_2 = round(random.random()*self.dn.M*2)+self.dn.N*2
        assert rnd_index <= rnd_index_2, "first index after second"
        assert rnd_index_2 <= len(parent_1_genes), "index longr than gene length"
        genes.extend(parent_1_genes[:rnd_index])
        genes.extend(parent_2_genes[rnd_index:rnd_index_2])
        genes.extend(parent_1_genes[rnd_index_2:])
        assert len(genes) == len(parent_1_genes) == len(parent_2_genes), "gene lengths unstable"
        return genes


    def getDnFromGenes(self, genes, dn, order_center=None):

        setattr(dn, "genes", genes)

        for i in range(self.dn.N):
            x = genes[i*2]/65535*self.dn.xdim
            y = genes[i*2+1]/65535*self.dn.ydim
            dn.acceptors[i] = (x, y, 0)
        for i in range(self.dn.M):
            x = genes[self.dn.N*2+i*2]/65535*self.dn.xdim
            y = genes[self.dn.N*2+i*2+1]/65535*self.dn.ydim
            dn.donors[i] = (x, y, 0)
        if order_center is not None:
            self.orderPlacement(dn, center = order_center)
        dn.initialize( dopant_placement=False, charge_placement=False)
    

    @staticmethod
    def nDimSquareDistance(a, b):
        sum=0
        for i in range(len(a)):
            sum+=(a[i]-b[i])**2
        return sum


    @staticmethod
    def getDegree(x, y):
        d = math.sqrt(x**2 + y**2)
        asinv = math.asin(y/d)*180/math.pi
        if x < 0:
            asinv = 180 - asinv
        if asinv < 0:
            asinv+=360
        return asinv


    @staticmethod
    def degreeDistance(a, b):
        x = a[0] - b[0]
        y = a[1] - b[1]
        return dn_search.getDegree(x, y)

    def orderPlacement(self, dn, center):
        distances = []
        for i in range(self.dn.N):
            dist = self.order_distance_function(dn.acceptors[i], center)
            distances.append((dist, i))
        distances = sorted(distances, key=lambda x:x[0])
        newAcceptors = []
        for _,index in distances:
            newAcceptors.append(dn.acceptors[index])
        dn.acceptors = np.array(newAcceptors)
        distances = []
        for i in range(self.dn.M):
            dist = dn_search.nDimSquareDistance(dn.donors[i], center)
            distances.append((dist, i))
        distances = sorted(distances, key=lambda x:x[0])
        newDonors = []
        for _,index in distances:
            newDonors.append(dn.donors[i])
        dn.donors = np.array(newDonors)

    
    def copyDnFromBtoA(self, dna, dnb):
        setattr(dna, "genes", getattr(dnb, "genes", []).copy())
        dna.acceptors = dnb.acceptors.copy()
        dna.donors = dnb.donors.copy()
        dna.initialize( dopant_placement=False, charge_placement=False)
    
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
