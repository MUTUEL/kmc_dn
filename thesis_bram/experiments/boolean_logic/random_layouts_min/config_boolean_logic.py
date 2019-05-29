import numpy as np
from SkyNEt.config.config_class import config_class

class experiment_config(config_class):
    '''This is the experiment configuration file for the boolean logic experiment.
    It also serves as a template for other experiments, so please model your work
    after this when you make new experiments.
    This experiment_config class inherits from config_class default values that are known to work well with boolean logic.
    You can define user-specific parameters in the construction of the object in __init__() or define
    methods that you might need after, e.g. a new fitness function or input and output generators.
    Remember if you define a new fitness function or generator, you have to redefine the self.Fitness,
    self.Target_gen and self.Input_gen in __init__()

    ----------------------------------------------------------------------------
    Description of general parameters
    ----------------------------------------------------------------------------
    comport; the COM port to which the ivvi rack is connected.
    amplification; specify the amount of nA/V. E.g. if you set the IVVI to 100M,
        then amplification = 10
    generations; the amount of generations for the GA
    generange; the range that each gene ([0, 1]) is mapped to. E.g. in the Boolean
        experiment the genes for the control voltages are mapped to the desired
        control voltage range.
    partition; this tells the GA how it will evolve the next generation.
        In order, this will make the GA evolve the specified number with
        - promoting fittest partition[0] genomes unchanged
        - adding Gaussian noise to the fittest partition[1] genomes
        - crossover between fittest partition[2] genomes
        - crossover between fittest partition[3] and randomly selected genes
        - randomly adding parition[4] genomes
    genomes; the amount of genomes in the genepool, speficy this parameter instead
        of partition if you don't care about the specific partition.
    genes; the amount of genes per genome
    mutationrate; the probability of mutation for each gene (between 0 and 1)
    fitnessavg; the amount of times the same genome is tested to obtain the fitness
        value.
    fitnessparameters; the parameters for FitnessEvolution (see its doc for
        specifics)
    filepath; the path used for saving your experiment data
    name; name used for experiment data file (date/time will be appended)

    ----------------------------------------------------------------------------
    Description of method parameters
    ----------------------------------------------------------------------------
    signallength; the length in s of the Boolean P and Q signals
    edgelength; the length in s of the edge between 0 and 1 in P and Q
    fs; sample frequency for niDAQ or ADwin

    ----------------------------------------------------------------------------
    Description of methods
    ----------------------------------------------------------------------------
    TargetGen; specify the target function you wish to evolve, options are:
        - OR
        - AND
        - NOR
        - NAND
        - XOR
        - XNOR
    Fitness; specify the fitness function, standard options are:
        - FitnessEvolution; standard fitness used for boolean logic
        - FitnessNMSE; normalised mean squared error
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
        self.prehops = 10000
        self.hops = 100000

        # Define experiment
        self.generations = 25
        self.generange = [[0, 1]]*5
        self.avg = 4
        self.P = 0
        self.Q = 1
        self.output = 2
        self.controls = [3, 4, 5, 6, 7]
        self.controlrange = np.array([-2000, 2000])*self.kT
        self.inputrange = 1000*self.kT

        # Define targets
        self.AND_discrete = np.array([0, 0, 0, 1])
        self.AND = np.zeros(4*self.avg)
        self.OR_discrete = np.array([0, 1, 1, 1])
        self.OR = np.zeros(4*self.avg)
        self.NOR_discrete = np.array([1, 0, 0, 0])
        self.NOR = np.zeros(4*self.avg)
        self.NAND_discrete = np.array([1, 1, 1, 0])
        self.NAND = np.zeros(4*self.avg)
        self.XOR_discrete = np.array([0, 1, 1, 0])
        self.XOR = np.zeros(4*self.avg)
        self.XNOR_discrete = np.array([1, 0, 0, 1])
        self.XNOR = np.zeros(4*self.avg)
        for i in range(4):
            self.AND[i*self.avg:(i+1)*self.avg] = self.AND_discrete[i]
            self.OR[i*self.avg:(i+1)*self.avg] = self.OR_discrete[i]
            self.NOR[i*self.avg:(i+1)*self.avg] = self.NOR_discrete[i]
            self.NAND[i*self.avg:(i+1)*self.avg] = self.NAND_discrete[i]
            self.XOR[i*self.avg:(i+1)*self.avg] = self.XOR_discrete[i]
            self.XNOR[i*self.avg:(i+1)*self.avg] = self.XNOR_discrete[i]

        # Set target
        self.target = self.XOR
        self.Fitness = self.FitnessCorr

        # Specify either partition or genomes
        self.partition = [5, 5, 5, 5, 5]
        #self.genomes = 20

        # Save settings
        self.filepath = ''#Important: end path with double backslash
        if(np.array_equal(self.target, self.AND)):
            self.name = 'AND'
        if(np.array_equal(self.target, self.OR)):
            self.name = 'OR'
        if(np.array_equal(self.target, self.NOR)):
            self.name = 'NOR'
        if(np.array_equal(self.target, self.NAND)):
            self.name = 'NAND'
        if(np.array_equal(self.target, self.XOR)):
            self.name = 'XOR'
            self.target = self.XOR_discrete
        if(np.array_equal(self.target, self.XNOR)):
            self.name = 'XNOR'

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
        A = np.vstack([target_weighed, np.ones(len(indices))]).T  #write x = m*target + c as x = A*(m, c)
        m, c = np.linalg.lstsq(A, x_weighed)[0]
        res = np.linalg.lstsq(A, x_weighed)[1]
        res = res[0]

        #determine fitness quality
        indices1 = np.argwhere(target_weighed)  #all indices where target is nonzero
        x0 = np.empty(0)  #list of values where x should be 0
        x1 = np.empty(0)  #list of values where x should be 1
        for i in range(len(target_weighed)):
            if(i in indices1):
                x1 = np.append(x1, x_weighed[i])
            else:
                x0 = np.append(x0, x_weighed[i])
        if(min(x1) < max(x0)):
            Q = 0
        else:
            Q = (min(x1) - max(x0)) / (max(x1) - min(x0) + abs(min(x0)))

        F = self.fitnessparameters[0] * m / (res**(.5) + self.fitnessparameters[3] * abs(c)) + self.fitnessparameters[1] / res + self.fitnessparameters[2] * Q

        return F

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
        return np.corrcoef(x, target)[0, 1] 
