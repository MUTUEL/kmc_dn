from ctypes import *
from numpy import float64
import numpy as np
import timeit

def flattenDouble(arr2):
    d = len(arr2[0])
    arr = []
    for row in arr2:
        for ele in row:
            arr.append(c_double(float64(ele).item()))
        if len(row) != d:
            raise Exception("array dimensions not uniform!")
            occupation = np.array([[]])
    return (c_double * len(arr))(*arr), d, len(arr)

def getGoSlice(arr):
    rArr = []
    for ele in arr:
        rArr.append(c_double(ele))
    return GoSlice((c_double * len(rArr))(*rArr), len(rArr), len(rArr))


class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_double)),
                ("len", c_longlong), ("cap", c_longlong)]
                

def getSliceValues(slice):
    r = []
    for i in range(slice.len):
        r.append(slice.data[i])
    return r

def callGoSimulation(NSites, NElectrodes, nu, kT, I_0, R, time, occupation, 
		distances , E_constant, site_energies, transitions_constant, transitions, problist, electrode_occupation, hops, record, goSpecificFunction):
    newDistances, d, s = flattenDouble(distances)
    newTransConstants, _, tcs = flattenDouble(transitions_constant)
    #print ("d is %d and s is %d"%(d, s))
    newDistances = GoSlice(newDistances, s, s)
    newTransConstants = GoSlice(newTransConstants, tcs, tcs)
    newOccupation = getGoSlice(occupation)
    newE_constant = getGoSlice(E_constant)
    newSite_energies = getGoSlice(site_energies)
    newElectrode_occupation = getGoSlice(electrode_occupation)
    lib = cdll.LoadLibrary("./libSimulation.so")
    getattr(lib, goSpecificFunction).argtypes = [c_longlong, c_longlong, c_double, c_double, c_double, c_double, c_double,
        GoSlice, GoSlice, GoSlice, GoSlice, GoSlice, GoSlice, c_int]
    getattr(lib, goSpecificFunction).restype = c_double
    #print (electrode_occupation)
    #printSlice(newElectrode_occupation)
    time = getattr(lib, goSpecificFunction)(NSites, NElectrodes, nu, kT, I_0, R, time, newOccupation, 
		newDistances , newE_constant, newTransConstants, newElectrode_occupation, newSite_energies, hops)
    #printSlice (newElectrode_occupation)
    rElectrode_occupation = np.array([int(i) for i in getSliceValues(newElectrode_occupation)])
    occupation = np.array([int(i) for i in getSliceValues(newOccupation)])
    #print (rElectrode_occupation)

    return (time, occupation, rElectrode_occupation)
