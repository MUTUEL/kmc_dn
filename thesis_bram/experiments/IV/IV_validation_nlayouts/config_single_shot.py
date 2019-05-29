import numpy as np
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''
    Config for fitting the model to an experimental IV curve.
    For speed considerations:
    35-point IV curve with 1E5 hops per point takes ~60s
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        # Model parameters
        self.kT = 1
        self.I_0 = 100*self.kT
        self.ab_R = 0.25
        self.layouts = list(range(100))
        self.prehops = 0
        self.hops = int(1E6)
        self.bias_electrode = 0
        self.use_random_layout = True
        self.layoutsize = 120

        self.voltagelist = np.linspace(-320, 320, 50)

        self.Fitness = self.FitnessCorr

        # Save settings
        self.filepath = ''#Important: end path with double backslash
        self.name = f'IV_{self.layoutsize}'

        ################################################
        ################# OFF-LIMITS ###################
        ################################################
        # Check if genomes parameter has been changed
        if(self.genomes != sum(self.default_partition)):
            if(self.genomes%5 == 0):
                self.partition = [int(self.genomes/5)]*5  # Construct equally partitioned genomes
            else:
                print('WARNING: The specified number of genomes is not divisible by 5.'
                      + ' The remaining genomes are generated randomly each generation. '
                      + ' Specify partition in the config instead of genomes if you do not want this.')
                self.partition = [self.genomes//5]*5  # Construct equally partitioned genomes
                self.partition[-1] += self.genomes%5  # Add remainder to last entry of partition

        self.genomes = int(sum(self.partition))  # Make sure genomes parameter is correct
        self.genes = int(len(self.generange))  # Make sure genes parameter is correct

        self.fitnessparameters = [0, 0, 1, 0]

    #####################################################
    ############# USER-SPECIFIC METHODS #################
    #####################################################
    # Optionally define new methods here that you wish to use in your experiment.
    # These can be e.g. new fitness functions or input/output generators.
    def FitnessCorr(self, x, target):
        # Check if x has single value
        valid = False
        for i in range(len(x)):
            if(not x[i] == x[0]):
                valid = True

        if(valid):
            return np.corrcoef(x, target)[0, 1] 
        else:
            return -1
