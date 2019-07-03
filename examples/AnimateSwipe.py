import os
import sys
sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import dn_animation

# This is a good example of how the animation scripts are used.
# Here we also have to generate a swipe result, which if you want a slightly different animation
# is probably the place you would want to change.


def openKmc(abs_file_path):
    '''
    This returns a kmc_dn object that is in the given absolute file path.
    :param abs_file_path:
    :return:
    '''
    kmc = kmc_dn.kmc_dn(10, 3, 1, 1, 0)
    kmc.loadSelf(abs_file_path)
    return kmc

def appendToSwipeResults(dn, expected_result):
    '''
    This is a helper function which is called in multiple places in getSwipeResults.
    You can see the structure of the search_results here. as we append a copy of electrodes, current,
    traffic, time and expected result to the search_results array.
    :param dn:
    :param expected_result:
    :return:
    '''
    dn.swipe_results.append((dn.electrodes.copy(), dn.current.copy(), dn.traffic.copy(), dn.time, expected_result))


def getSwipeResults(dn, steps, hops, waits):
    '''
    This is used to generate the data that will be used by each of the frames in the animation.
    :param dn: kmc_dn
        The initial kmc_dn object.
    :param bool_func: array
        boolean function, with input values and expected output that form the points in between
        we do the swipe transitions
    :param steps: int
        The number of frames we use for each swipe.
    :param hops:
        The number of hops we use to get traffic and current data from simulation.
    :param waits:
        The number of frames we wait before going to another swipe transition. We also do a
        simulation for each of the frames, so that we visualize the variance.
    :return:
    '''
    dn.swipe_results = []

    tests = dn.tests
    print (tests)
    bool_func = []
    for test in tests:
        bool_func.append(test[0])
    dn.electrodes[0][3] = bool_func[0][0]
    dn.electrodes[1][3] = bool_func[0][1]
    dn.update_V()
    current_expected = 1 if tests[0][1] else 0
    
    for i in range(1, len(bool_func)):
        next_expected = 1 if tests[i][1] else 0
        print ("i is %d"%(i))
        from_voltage = [dn.electrodes[0][3], dn.electrodes[1][3]]
        to_voltage = [bool_func[i][0], bool_func[i][1]]
        for j in range(waits):
            dn.go_simulation(hops = hops, record=True,  goSpecificFunction="wrapperSimulateRecord")

            appendToSwipeResults(dn, current_expected)

        for j in range(steps):
            dn.electrodes[0][3] = from_voltage[0] + (to_voltage[0]-from_voltage[0])*(j*1.0/steps)
            dn.electrodes[1][3] = from_voltage[1] + (to_voltage[1]-from_voltage[1])*(j*1.0/steps)
            dn.update_V()
            dn.go_simulation(hops = hops, record=True,  goSpecificFunction="wrapperSimulateRecord")
            appendToSwipeResults(dn, current_expected + (next_expected-current_expected)*(j*1.0/steps))
            print (current_expected + (next_expected-current_expected)*(j*1.0/steps))
        dn.electrodes[0][3] = to_voltage[0]
        dn.electrodes[1][3] = to_voltage[1]
        dn.update_V()
        current_expected = next_expected
    for j in range(waits):
        dn.go_simulation(hops = hops, record=True,  goSpecificFunction="wrapperSimulateRecord")
        appendToSwipeResults(dn, current_expected)

def animateExample(index, useCalcs=False, animation_index=None, dmp_name="resultDump"):
    '''
        This function, given the index of the kmc dump file, uses it to generate data
        for each frame and then create an animation.
    :param index:
        index that points to the right file
    :param useCalcs:
        Have we perhaps already made the calculations for the
        frame already but something went wrong during animation?
        No problem, just set this to true
    :param animation_index:
        Want to use multiple animations for the same index, for whatever
        reason? No problem, set this to true.
    :return:
    '''
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    if not useCalcs:
        rel_path = "../%s%d.kmc"%(dmp_name, index)
    else:
        rel_path = "swipeResults/xor%d.kmc"%(index)
    
    abs_file_path = os.path.join(script_dir, rel_path)
    try:
        dn = openKmc(abs_file_path)
    except:
        print (abs_file_path)
        return
    
    #dn.electrodes[2][3] = 0
    if not hasattr(dn, "swipe_results"):
        getSwipeResults(dn, 40, 5000000, 40)
        rel_write_path = "swipeResults/xor%d.kmc"%(index)
        abs_file_path = os.path.join(script_dir, rel_write_path)
        dn.saveSelf(abs_file_path)
    writer = dn_animation.getWriter(10, "Swipe animation")
    if animation_index:
        animation_file = "swipe_animation%d_%d.mp4"%(index, animation_index)
    else:
        animation_file = "swipe_animation%d.mp4"%(index)
    dn_animation.trafficAnimation(dn, dn.swipe_results, writer, animation_file)
  

def main():
    #for index in range(4000, 4070):
    #    animateExample(index, False, dmp_name="resultDump")
    for index in [1]:
        animateExample(index, False, dmp_name="GeneticResultDumpVoltageGenetic")

if __name__== "__main__":
    main()