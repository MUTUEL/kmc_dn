from dn_search import dn_search
import kmc_dopant_networks as kmc_dn

import numpy as np

import random
import math

class voltage_search(dn_search):
    def __init__(self, initial_dn, voltage_range, voltage_resolution, tests):
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
        
        self.simulation_strategy = [
            {'func':"go_simulation",
             'args':{'hops':1000000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.000002,
             'threshold_error':-1},
            {'func':"go_simulation",
             'args':{'hops':5000000, 'goSpecificFunction':"wrapperSimulateRecord"},
             'expected_error':0.000002,
             'threshold_error':-1},
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
        self.error_func = self.fittness
        self.output_electrode = len(self.dn.electrodes)-1
        self.N = self.output_electrode - 2
        self.N_tests = 1
        self.dn.xCoords = []
        self.dn.yCoords = []
        self.init_random_voltages(self.dn)
        #for _ in range(10):
        #    print (self.evaluate_error(self.dn))
        self.genetic_allowed_overlap = -1

    def init_random_voltages(self, dn):
        dn.true_voltage = random.random()*self.voltage_range*2 - self.voltage_range
        for i in range(self.N):
            dn.electrodes[i+2][3] = random.random()*self.voltage_range*2 - self.voltage_range

    def evaluate_error(self, dn):
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

    def yieldNeighbours(self):
        shifts = [self.x_resolution, -self.x_resolution]
        options = [(i+2, shifts[j]) for i in range(self.N+1) for j in range(len(shifts))]
        indexes = [x for x in range(len(options))]
        random.shuffle(indexes)
        for index in indexes:
            option = options[index]
            if option[0] == 0:
                electrode_voltage = self.dn.true_voltage + option[1]

                if electrode_voltage < self.voltage_range and electrode_voltage > self.voltage_range/10:
                    newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, 
                        self.dn.ydim, 0, electrodes = self.dn.electrodes, 
                        acceptors=self.dn.acceptors, donors=self.dn.donors, copy_from=self.dn)
                    newDn.true_voltage = electrode_voltage
                    yield newDn, option[0], (0, 0)
            else:
                electrode_voltage = self.dn.electrodes[option[0]][3] + option[1]

                if math.fabs(electrode_voltage) < self.voltage_range:
                    newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, 
                        self.dn.ydim, 0, electrodes = self.dn.electrodes, 
                        acceptors=self.dn.acceptors, donors=self.dn.donors, copy_from=self.dn)
                    newDn.electrodes[option[0]][3] = self.dn.electrodes[option[0]][3] + option[1]
                    yield newDn, option[0], (0, 0)

    #There is no special way to validate this solution as tests have 
    # full coverage of what we are searching.
    def validate_error(self, dn):
        return self.evaluate_error(dn)
        
    def fittness(self):
        fittness = 0
        return fittness

    def getRandomDn(self):
        newDn = kmc_dn.kmc_dn(self.dn.N, self.dn.M, self.dn.xdim, self.dn.ydim, 
                self.dn.zdim, electrodes=self.dn.electrodes, acceptors=self.dn.acceptors, 
                donors=self.dn.donors, copy_from = self.dn)
        self.init_random_voltages(newDn)
        return newDn
            
    def getGenes(self, dn):
        genes = []
        x = np.uint16(dn.true_voltage / self.voltage_range * 65535)
        genes.append(x)
        for i in range(self.N):
            value = dn.electrodes[i+2][3]
            x = np.uint16((value + self.voltage_range)/self.voltage_range/2 * 65535)
            genes.append(x)
        return genes

    def getDnFromGenes(self, genes, dn, order_center=None):
        setattr(dn, "genes", genes)
        value = genes[0]/65535 * self.voltage_range
        setattr(dn, "true_voltage", value)
        assert len(genes) == 6
        for i in range(1, len(genes)):
            value = genes[i]/65535 * 2 * self.voltage_range - self.voltage_range
            dn.electrodes[i+1][3] = value

    def copyDnFromBtoA(self, dna, dnb):
        setattr(dna, "genes", getattr(dnb, "genes", []).copy())
        dna.electrodes = dnb.electrodes.copy()
        dna.true_voltage = dnb.true_voltage
        dna.update_V()

    def isAllowed(self, prev_genes, gene, uniqueness, resolution):
        for prev_gene in prev_genes:
            total_diff = 0
            for coord in range(len(gene)):
                total_diff+=dn_search.uInt16Diff(prev_gene[coord], gene[coord])
            if total_diff < uniqueness:
                return False, -1
        return True, -1
