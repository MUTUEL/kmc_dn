import os
import sys
sys.path.insert(0,'../')
import dn_animation
from utils import getSwipeResults, openKmc

# This is a good example of how the animation scripts are used.
# Here we also have to generate a swipe result, which if you want a slightly different animation
# is probably the place you would want to change.



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
    :param dmp_name:
        Your kmc object is saved with a different name than resultDump[number].kmc, specify,
        what is instead of resultDump
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