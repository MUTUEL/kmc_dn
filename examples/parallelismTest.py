import os
import sys
#sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
from goSimulation.parrallelSimulationBind import parrallelSimulation
import numpy as np
import random
import math


def get8Electrodes(xdim, ydim):
    electrodes = np.zeros((2, 4))
    volt_range = 600
    electrodes[0] = [0, 3*ydim/4, 0, volt_range*random.random()]
    electrodes[1] = [xdim/4, 0, 0, volt_range*random.random()]
    # electrodes[2] = [xdim, ydim/4, 0, volt_range*random.random()]
    # electrodes[3] = [xdim, 3*ydim/4, 0, volt_range*random.random()]
    # electrodes[4] = [0, ydim/4, 0, volt_range*random.random()]
    # electrodes[5] = [3*xdim/4, 0, 0, volt_range*random.random()]
    # electrodes[6] = [xdim/4, ydim, 0, volt_range*random.random()]
    # electrodes[7] = [3*xdim/4, ydim, 0, 0]

    return electrodes

def getRandomDn(N_acceptors, N_donors):
    electrodes = get8Electrodes(1, 1)
    dn = kmc_dn.kmc_dn(N_acceptors, N_donors, 1, 1, 0, electrodes = electrodes)
    return dn

def main():
    dns = []
    N_acceptors = 5
    N_donors = 3
    for i in range(4):
        dns.append(getRandomDn(N_acceptors, N_donors))
    for i in range(3):
        for dn in dns:
            dn.go_simulation(1000000, goSpecificFunction="wrapperSimulateRecord")
            print (dn.current[1])
    ps = parrallelSimulation()
    for dn in dns:
        ps.addSimulation(dn, 1000000)
    ps.runSimulation()
    for dn in dns:
        print(dn.current[1])

if __name__== "__main__":
  main()

