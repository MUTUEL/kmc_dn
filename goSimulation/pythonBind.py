from ctypes import *
from numpy import float64
import timeit

def flattenDouble(arr2):
    print (arr2)
    d = len(arr2[0])
    arr = []
    for row in arr2:
        for ele in row:
            arr.append(c_double(float64(ele).item()))
        if len(row) != d:
            raise Exception("array dimensions not uniform!")
    return (c_double * len(arr))(*arr), d, len(arr)

def getGoSlice(arr):
    rArr = []
    for ele in arr:
        rArr.append(c_double(ele))
    return GoSlice((c_double * len(rArr))(*rArr), len(rArr), len(rArr))


class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_double)),
                ("len", c_longlong), ("cap", c_longlong)]

def callSimulation(NSites, NElectrodes, nu, kT, I_0, R, time, occupation, 
		distances , E_constant, transitions_constant, electrode_occupation, site_energies, hops):
    newDistances, d, s = flattenDouble(distances)
    newDistances = GoSlice(newDistances, s, s)
    newOccupation = getGoSlice(occupation)
    newE_constant = getGoSlice(E_constant)
    newSite_energies = getGoSlice(site_energies)
    newElectrode_occupation = getGoSlice(electrode_occupation)
    lib = cdll.LoadLibrary("./libSimulation.so")
    lib.simulateWrapper.argtypes = [c_longlong, c_longlong, c_double, c_double, c_double, c_double, c_double,
        GoSlice, GoSlice, GoSlice, c_double, GoSlice, GoSlice, c_int]
    lib.simulateWrapper.restype = c_double

    time, newElectrodes_occupation = lib.simulateWrapper(NSites, NElectrodes, nu, kT, I_0, R, time, newOccupation, 
		newDistances , newE_constant, 1.0, newElectrode_occupation, newSite_energies, hops)
    print (time)
    print (newElectrode_occupation)
