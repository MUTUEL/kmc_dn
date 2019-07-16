from voltage_search_tests import getRandomDn
from voltage_search import voltage_search
from utils import parseArgs

def testSample(dn, tests, hours = 10, disparity=2, 
        mut_pow=1, gen_size = 50, index = 0, times = 20, folder=""):
    search = voltage_search(dn, 150, 10, tests, corr_pow=2, parallelism=0)
    cross_over_function = search.singlePointCrossover
    dns = []
    for i in range(gen_size):
        dn = search.getRandomDn()
        dns.append(dn)
    u_schedule = [(1000, 5), (0, 20)]
    results = search.genetic_search(gen_size, 3600*hours, 2, uniqueness=5000, 
        cross_over_function = cross_over_function, mut_pow=mut_pow, 
        max_generations=10, mut_rate=0, initial_dns=dns, u_schedule=u_schedule)
    search.best_dn.genetic_search_results = search.validations
    search.saveResults(True, False, "%sresultDump"%(folder), index+3*times)

    results = search.genetic_search(gen_size, 3600*hours, 2, uniqueness=0, 
        cross_over_function = cross_over_function, mut_pow=mut_pow, 
        max_generations=10, mut_rate=0.05, initial_dns=dns)
    search.best_dn.genetic_search_results = search.validations
    search.saveResults(True, False, "%sresultDump"%(folder), index)

    results = search.genetic_search(gen_size, 3600*hours, 2, uniqueness=0, 
        cross_over_function = cross_over_function, mut_pow=mut_pow, 
        max_generations=10, mut_rate=0.3, initial_dns=dns)
    search.best_dn.genetic_search_results = search.validations
    search.saveResults(True, False, "%sresultDump"%(folder), index+times)
    
    results = search.genetic_search(gen_size, 3600*hours, 2, uniqueness=1000, 
        cross_over_function = cross_over_function, mut_pow=mut_pow, 
        max_generations=10, mut_rate=0, initial_dns=dns)
    search.best_dn.genetic_search_results = search.validations
    search.saveResults(True, False, "%sresultDump"%(folder), index+2*times)
    

#Can pass parameters. d - number of dopants. i - starting index for file numbers. t - number of times to do the whole process.
def main():
    points = [(0, 0), (0, 75), (75, 0), (75, 75)]#, (-50, 0), (50, 0)]
    tests = []
    for i in range(4):
        tests.append((points[i], 6&(2**i)))
    args = parseArgs()
    print (args)
    if "d" in args:
        dop = int(args["d"])
    else:
        dop = 30
    if "i" in args:
        startIndex = int(args["i"])
    else:
        startIndex = 0
    if "t" in args:
        times = int(args["t"])
    else:
        times = 20
    if "s" in args:
        skips = int(args["s"])
    else:
        skips = 0
    if "f" in args:
        folder = args["f"] + "/"
    else:
        folder = ""
    for index in range(startIndex, startIndex+skips):
        print (index)
        dn = getRandomDn(dop, round(dop/10))
        testSample(dn, tests, hours=1, gen_size=100, times=times, index=index+startIndex, folder=folder)


if __name__== "__main__":
  main()