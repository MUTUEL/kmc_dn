from dn_search import dn_search
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_utils

import numpy as np


import random
import math
import time

class voltage_search(dn_search):
    def __init__(self, initial_dn, voltage_range, voltage_resolution, tests, corr_pow=1):
        '''
        =======================
        dn_search dopant network placement search class
        ======================= 
        This class is made for performing a search to find electrode configuration that
            would perform some boolean operation.

        Input arguments
        ===============
        initial_dn; kmc_dn
            Instance of a kmc_dn class with initial dopant placements, which is used to create
            neighbouring kmc_dn instances to perform search. We expect the input electrodes to
            be the first 2, and the output ones the last one.
        tests; list
            This is a list of tests, each test is a tuple, which contains voltages of 
            input electrodes nad expected true value for output electrode.
        '''

        self.minimum_resolution = 0.1
        self.donor_dist = 10
        self.x_resolution = voltage_resolution
        self.voltage_range = voltage_range

        self.dn = initial_dn
        self.tests = tests
        self.perfect_correlation = []
        for test in tests:
            if test[1]:
                self.perfect_correlation.append(10)
            else:
                self.perfect_correlation.append(0)
        self.perfect_correlation = np.array(self.perfect_correlation)
        self.simulation_strategy = [
            {'func':"go_simulation",
             'args':{'hops':1000000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.000002,
             'threshold_error':-0.00001},
            {'func':"go_simulation",
             'args':{'hops':5000000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.000002,
             'threshold_error':-0.00001},
        ]
        """    {'func':"go_simulation",
             'args':{'hops':5000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.025,
             'threshold_error':1.96},
            {'func':"go_simulation",
             'args':{'hops':50000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.01,
             'threshold_error':1.97},
            {'func':"go_simulation",
             'args':{'hops':250000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.002,
             'threshold_error':0},
        ]"""
        self.setStrategy(0)
        self.output_electrode = len(self.dn.electrodes)-1
        self.N = self.output_electrode - 2
        self.N_tests = 1
        self.dn.xCoords = []
        self.dn.yCoords = []
        self.init_random_voltages(self.dn)
        self.genetic_allowed_overlap = -1
        self.evaluate_error = self.evaluate_error_corr
        self.corr_pow = corr_pow

    def init_random_voltages(self, dn):
        dn.true_voltage = random.random()*self.voltage_range*2 - self.voltage_range
        for i in range(self.N):
            dn.electrodes[i+2][3] = random.random()*self.voltage_range*2 - self.voltage_range

    def evaluate_error_diff(self, dn):
        lowest_true = 1
        highest_false = -1
        for test in self.tests:
            for i in range(len(test[0])):
                if test[0][i]:
                    dn.electrodes[i][3] = dn.true_voltage
                else:
                    dn.electrodes[i][3] = 0
            dn.update_V()
            getattr(dn, self.simulation_func)(**self.simulation_args)
            if test[1]:#expected result is True, so we adjust lowest_true value.
                if lowest_true > dn.current[self.output_electrode]:
                    lowest_true = dn.current[self.output_electrode]
            else:
                if highest_false < dn.current[self.output_electrode]:
                    highest_false = dn.current[self.output_electrode]
        return highest_false - lowest_true

    def evaluate_error_corr(self, dn):
        lowest_true = 1
        highest_false = -1
        values = []
        for test in self.tests:
            for i in range(len(test[0])):
                dn.electrodes[i][3] = test[0][i]
            dn.update_V()
            getattr(dn, self.simulation_func)(**self.simulation_args)
            values.append(dn.current[self.output_electrode])
            if test[1]:#expected result is True, so we adjust lowest_true value.
                if lowest_true > dn.current[self.output_electrode]:
                    lowest_true = dn.current[self.output_electrode]
            else:
                if highest_false < dn.current[self.output_electrode]:
                    highest_false = dn.current[self.output_electrode]
        values = np.array(values)
        corr = np.corrcoef(self.perfect_correlation, values)[0][1]
        separation = highest_false - lowest_true
        if corr < 0:
            corr = 0
        if separation > 0:
            return separation
        else:
            return (corr**self.corr_pow) * separation

    def yieldNeighbours(self):
        shifts = [self.x_resolution, -self.x_resolution]
        options = [(i+2, shifts[j]) for i in range(self.N) for j in range(len(shifts))]
        indexes = [x for x in range(len(options))]
        random.shuffle(indexes)
        for index in indexes:
            option = options[index]
            electrode_voltage = self.dn.electrodes[option[0]][3] + option[1]

            if math.fabs(electrode_voltage) < self.voltage_range:
                newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, 
                    self.dn.ydim, 0, electrodes = self.dn.electrodes, 
                    acceptors=self.dn.acceptors, donors=self.dn.donors, copy_from=self.dn)
                newDn.electrodes[option[0]][3] = electrode_voltage
                yield newDn, option[0], (0, 0)

    def validate_error(self, dn):
        return self.evaluate_error(dn)

    def getRandomDn(self):
        newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, self.dn.ydim, 
                self.dn.zdim, electrodes=self.dn.electrodes, acceptors=self.dn.acceptors, 
                donors=self.dn.donors, copy_from = self.dn)
        newDn.tests = self.tests
        self.init_random_voltages(newDn)
        return newDn
            
    def getGenes(self, dn):
        genes = []
        for i in range(self.N):
            value = dn.electrodes[i+2][3]
            x = np.uint16((value + self.voltage_range)/self.voltage_range/2 * 65535)
            genes.append(x)
        return genes

    def getDnFromGenes(self, genes, dn, order_center=None):
        setattr(dn, "genes", genes)
        assert len(genes) == 5
        for i in range(0, len(genes)):
            value = genes[i]/65535 * 2 * self.voltage_range - self.voltage_range
            dn.electrodes[i+2][3] = value

    def copyDnFromBtoA(self, dna, dnb):
        setattr(dna, "genes", getattr(dnb, "genes", []).copy())
        dna.electrodes = dnb.electrodes.copy()
        dna.update_V()

    def isAllowed(self, prev_genes, gene, uniqueness, resolution):
        for prev_gene in prev_genes:
            total_diff = 0
            for coord in range(len(gene)):
                total_diff+=dn_search.uInt16Diff(prev_gene[coord], gene[coord])
            if total_diff < uniqueness:
                return False, -1
        return True, -1

    def checkRange(self, dn):
        if dn.true_voltage < self.voltage_range/4:
            dn.true_voltage = self.voltage_range/4
        if dn.true_voltage > self.voltage_range:
            dn.true_voltage = self.voltage_range
        for i in range(self.N):
            if math.fabs(dn.electrodes[i+2][3]) > self.voltage_range:
                dn.electrodes[i+2][3] = dn.electrodes[i+2][3] / math.fabs(dn.electrodes[i+2][3]) * self.voltage_range

    def getAdjustedDn(self, dn, adjustment):
        newDn = kmc_dn.kmc_dn(dn.N, dn.M, dn.xdim, dn.ydim, 
                self.dn.zdim, electrodes=dn.electrodes, acceptors=dn.acceptors, 
                donors=dn.donors, copy_from = dn)
        newDn.true_voltage = dn.true_voltage + adjustment[0]
        for i in range(self.N):
            newDn.electrodes[i+2][3] = newDn.electrodes[i+2][3] + adjustment[i+1]
        self.checkRange(newDn)
        return newDn

    def SPSA_search(self, time_budget, a, c, A, alfa, gamma, file_prefix):
        k = 1
        self.dn = self.getRandomDn()
        self.init_search()
        L_largest = 0.0000001
        start_time = time.time()
        time_difference = time.time() - start_time
        validation_step = time_budget / 10
        next_validation = validation_step
        while time_difference < time_budget:
            time_difference = time.time() - start_time
            delta = [2*round(random.random())-1 for _ in range(self.N + 1)]
            ak= a/((A+k)**alfa)
            ck=c/(k**gamma)
            deltaplus = [d*ck for d in delta]
            deltaminus = [-d*ck for d in delta]
            Lplus = self.evaluate_error(self.getAdjustedDn(self.dn, deltaplus))
            Lminus = self.evaluate_error(self.getAdjustedDn(self.dn, deltaminus))
            for L in [Lplus, Lminus]:
                if math.fabs(L) > L_largest:
                    L_largest = math.fabs(L)
            Lplus /=L_largest
            Lminus /= L_largest
            gradDelta = [-(Lplus - Lminus)/2/ck * d * ak for d in delta]
            self.dn = self.getAdjustedDn(self.dn, gradDelta)
            k+=1
            print ("%.3g, %.3g, %.3g"%(Lplus, Lminus, -(Lplus - Lminus)/2/ck * ak))
            volts = [e[3] for e in self.dn.electrodes]
            print (volts)
            if time_difference > next_validation:
                next_validation += validation_step
                tmp_error = self.evaluate_error(self.dn)
                self.appendValidationData(tmp_error, time_difference)
        self.dn.saveSelf("resultDump%s.kmc"%(file_prefix))
        tmp_error = self.evaluate_error(self.dn)
        self.appendValidationData(tmp_error, time_difference)
        return tmp_error, self.current_strategy, self.validations

