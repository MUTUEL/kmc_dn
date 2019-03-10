import numpy as np
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''
    Config for fitting the model to an experimental IV curve
    '''

    def __init__(self):
        super().__init__() #DO NOT REMOVE!
        ################################################
        ######### SPECIFY PARAMETERS ###################
        ################################################
        # Model parameters
        self.kT = 1
        self.I_0 = 50*self.kT
        self.ab_R = 0.5
        self.layout = 0
        self.prehops = int(5E5)
        self.hops = int(5E5)

        # Define experiment
        self.generations = 40
        self.generange = [[0.1, 100], [0.01, 10]]
        self.avg = 1

        self.Fitness = self.FitnessGap

        # Specify either partition or genomes
        self.partition = [5, 5, 5, 5, 5]
        #self.genomes = 20

        # Save settings
        self.filepath = ''#Important: end path with double backslash
        self.name = 'IV_fit'

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
    def FitnessFix(self, x, target):
        '''
        This implements the fitness function
        F = self.fitnessparameters[0] * m / (sqrt(r) + self.fitnessparameters[3] * abs(c)) + self.fitnessparameters[1] / r + self.fitnessparameters[2] * Q
        where m,c,r follow from fitting x = m*target + c to minimize r
        and Q is the fitness quality as defined by Celestine in his thesis
        appendix 9
        W is a weight array consisting of 0s and 1s with equal length to x and
        target. Points where W = 0 are not included in fitting.
        '''

        #extract fit data with weights W
        indices = np.arange(len(x))  #indices where W is nonzero (i.e. 1)

        x_weighed = np.empty(len(indices))
        target_weighed = np.empty(len(indices))
        for i in range(len(indices)):
        	x_weighed[i] = x[indices[i]]
        	target_weighed[i] = target[indices[i]]


    	#fit x = m * target + c to minimize res
        A = np.zeros((len(target_weighed), 1))
        A[:, 0] = target_weighed 
        m = np.linalg.lstsq(A, x_weighed)[0]
        m = m[0]
        res = np.linalg.lstsq(A, x_weighed)[1]
        res = res[0]

        return 1/res

    def FitnessQuality(self, x, target):
        max1 = np.max(x[target==1])
        min1 = np.min(x[target==1])
        max0 = np.max(x[target==0])
        min0 = np.min(x[target==0])
        if(min1 - max0 < 0):
            F = 0
        else:
            F = (min1 - max0)/(max1 - min0)
        return F

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

    def FitnessGap(self, x, target):
        '''
        This fitness tries to force a gap.
        '''
        diff = np.sum(np.abs(x)[6:-8])
        norm = x[-1] - x[0]
        if(norm == 0):
            return -1
        else
            return diff/norm
