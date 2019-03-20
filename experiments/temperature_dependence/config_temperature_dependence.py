import numpy as np
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''
    This file defines a temperature dependence experiment in order
    to see if we get the expected behaviour for 2D mott vrh.
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        self.filepath = ''
        self.name = 'testrun'

        # Model parameters
        self.kT = np.logspace(-1, 3, 25)
        self.I_0 = 100
        self.ab_R = 0.25
        self.layout = 0
        self.prehops = 100000
        self.hops = 1000000

        # Define experiment
        self.voltages = [-7.5, 7.5]

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
