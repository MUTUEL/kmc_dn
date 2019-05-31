from ctypes import *
from numpy import float64
import numpy as np
import time as time_lib
import sys
sys.path.insert(0,'./goSimulation')
from goSimulation.pythonBind import flattenDouble, getGoSlice, GoSlice, getSliceValues

class parrallelSimulation():
    def __init__(self):
        self.N_acceptors = []
        self.N_electrodes = []
        self.nu = []
        self.kT = []
        self.I_0 = []
        self.R = []
        self.time = []
        self.occupation = []
        self.distances = []
        self.E_constant = []
        self.transitions_constant = []
        self.electrode_occupation = []
        self.site_energies = []
        self.hops = []
        self.dns = []

    def flatten(self, arr):
        d = len(arr[0])
        r = []
        for row in arr:
            for ele in row:
                r.append(ele)
            if len(row) != d:
                raise Exception("None uniform array")
        return r

    def addSimulation(self, dn, hops):
        self.N_acceptors.append(dn.N)
        self.N_electrodes.append(len(dn.electrode_occupation))
        for attr_name in ['nu', 'kT', 'I_0', 'R', 'time']:
            getattr(self, attr_name).append(getattr(dn, attr_name))
        self.hops.append(hops)
        for attr_name in ['occupation','electrode_occupation', 'E_constant', 'site_energies']:
            getattr(self, attr_name).extend(getattr(dn, attr_name))
        for attr_name in ['distances', 'transitions_constant']:
            getattr(self, attr_name).extend(self.flatten(getattr(dn, attr_name)))
        dn.parrallel_results = []
        self.dns.append(dn)

    def runSimulation(self):
        for i in range(len(self.occupation)):
            if self.occupation[i]:
                self.occupation[i] = 1
            else:
                self.occupation[i] = 0
        for attr_name in ['N_acceptors', 'N_electrodes', 'nu', 'kT', 'I_0', 'hops','R', 'time', 'occupation',
            'electrode_occupation', 'E_constant', 'distances', 'transitions_constant', 'site_energies']:
            setattr(self, 'go_%s'%(attr_name), getGoSlice(getattr(self, attr_name)))
        lib = cdll.LoadLibrary("./goSimulation/libSimulation.so")
        lib.parallelSimulations.argtypes = [GoSlice]*14
        lib.parallelSimulations.restype = c_longlong

        done = lib.parallelSimulations(self.go_N_acceptors, self.go_N_electrodes, 
            self.go_nu, self.go_kT, self.go_I_0, self.go_R, self.go_occupation, self.go_distances,
            self.go_E_constant, self.go_transitions_constant, self.go_electrode_occupation,
            self.go_hops, self.go_time, self.go_site_energies)
        totalElectrodes = 0
        electrode_occupations = getSliceValues(self.go_electrode_occupation)
        times = getSliceValues(self.go_time)
        i = 0
        for dn in self.dns:
            eo = electrode_occupations[totalElectrodes:(totalElectrodes+len(dn.electrodes))]
            current = []
            for j in range(len(eo)):
                current.append(eo[j]/times[i])
            result = (times[i], eo, current)
            dn.parrallel_results.append(result)
            i+=1
            totalElectrodes+=len(dn.electrodes)
