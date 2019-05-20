import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as manimation
from kmc_dopant_networks_utils import visualize_traffic, visualize_V_and_traffic

# NB! Requires installation: sudo apt-get install ffmpeg
# This module contains code for 2 animations. GetWriter is a function shared by both.
# initScatterAnimation and animateTransition is used to animate DopantPlacement search in SimulatedAnnealing search.
# trafficAnimation is used to animate the change in potential landscape and electron jump traffic as the input changes.
# It also animates a graph in the side which shows the output current in parallel.


def getWriter(fps, title):
    '''
        This returns a writer object that is used to make the animation. This object is used to write frames that
        will be generated using matplotlib, to form an animation.
        Input arguments
        ---------------
        fps; int
            Frames per second.
        title; string
            Title of the animation.
        Returns
        -------
            writer: FFMPegWriter
                returns writer object used to make the animation.
        '''
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title)
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer

def initScatterAnimation(kmc):
    '''
        Used in the dopant placement annealing search animation.
        Input arguments
        ---------------

        Returns
        -------

        '''
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(right=kmc.xdim)
    ax.set_ylim(top=kmc.ydim)
    acceptors, = ax.plot(kmc.xCoords[:kmc.N], kmc.yCoords[:kmc.N], 'o', color='black')
    history_acceptors, = ax.plot(kmc.xCoords[:kmc.N], kmc.yCoords[:kmc.N], 'o', color='black')
    donors, = ax.plot(kmc.xCoords[kmc.N:], kmc.yCoords[kmc.N:], 'o', color='red')
    history_donors, = ax.plot(kmc.xCoords[kmc.N:], kmc.yCoords[kmc.N:], 'o', color='red')

    text_element = ax.text(0.5, -0.1, "Error: , time: , strategy: ", horizontalalignment='center')

    return acceptors, donors, history_acceptors, history_donors, text_element, fig



def animateTransition(kmc, donors, acceptors, history_donors, history_acceptors, text_element, index, target_pos, writer, splits, refresh_history, alpha, text):
    '''
        Used in the dopant placement annealing search animation.
        Input arguments
        ---------------

        Returns
        -------

        '''
    acceptorDataX = [kmc.xCoords[j] for j in range(kmc.N)]
    acceptorDataY = [kmc.yCoords[j] for j in range(kmc.N)]
    donorDataX = [kmc.xCoords[j] for j in range(kmc.N, kmc.N+kmc.M)]
    donorDataY = [kmc.yCoords[j] for j in range(kmc.N, kmc.N+kmc.M)]

    text_element.set_text(text)

    for i in range(splits):
        if index < kmc.N:
            acceptorDataX[index] = (kmc.xCoords[index]*(splits-i)+target_pos[0]*i) / splits
            acceptorDataY[index] = (kmc.yCoords[index]*(splits-i)+target_pos[1]*i) / splits
        else:
            donorDataX[index-kmc.N] = (kmc.xCoords[index]*(splits-i)+target_pos[0]*i) / splits
            donorDataY[index-kmc.N] = (kmc.yCoords[index]*(splits-i)+target_pos[1]*i) / splits
        donors.set_data(donorDataX, donorDataY)
        acceptors.set_data(acceptorDataX, acceptorDataY)
        if refresh_history:
            history_acceptors.set_data(acceptorDataX, acceptorDataY)
            history_acceptors.set_alpha(0.5)
            history_donors.set_data(donorDataX, donorDataY)
            history_donors.set_alpha(0.5)
        else:
            history_acceptors.set_alpha(alpha)
            history_donors.set_alpha(alpha)
        writer.grab_frame()

def trafficAnimation(kmc_dn, search_results, writer, file_name):
    '''
        Used in the dopant placement annealing search animation.
        Input arguments
        ---------------
            kmc_dn: kmc_dn
                Dopant network object that is used for animation. This provides the dopant placement and physical
                parameters.
            search_results: array
                This is an array of entries, where each entry is in the format (electrodes, currents, traffic, time, ideal_current).
                This is precalculated and used to perform the animation, as each entry represents one frame.
            writer: FFMPegWriter
                Writer object used to write frames to. This is initialized in the getWriter function.
            file_name: string
                Name of the file we write the animation video to.

        Returns
            None
                The output is the file, so nothing is returned.
        -------

        '''
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    
    vmin = 300
    vmax = -300
    highest_current = 0
    x_data = [i for i in range(len(search_results))]
    y_data_max = -1
    y_data_min = 1
    
    y_data = []
    y_data_expected = []
    for entry in search_results:
        kmc_dn.electrodes = entry[0]
        kmc_dn.current = entry[1]
        y_data.append(entry[1][7])
        if entry[1][7] < y_data_min:
            y_data_min = entry[1][7]
        if entry[1][7] > y_data_max:
            y_data_max = entry[1][7]
        kmc_dn.traffic = entry[2]
        kmc_dn.time = entry[3]
        kmc_dn.update_V()
        x = np.arange(0, kmc_dn.xdim, kmc_dn.res)
        y = np.arange(0, kmc_dn.ydim, kmc_dn.res)
        for i in range(len(x)):
            for j in range(len(y)):
                val = kmc_dn.V(x[i], y[j])
                if val > vmax:
                    vmax = val
                if val < vmin:
                    vmin = val
        
        for row in kmc_dn.traffic:
            for ele in row:
                if highest_current < ele / kmc_dn.time:
                    highest_current =  ele / kmc_dn.time
    for entry in search_results:
        expected = entry[4]
        y_data_expected.append(y_data_min + (y_data_max-y_data_min)*expected)
    
    with writer.saving(fig, file_name, 100):
        i = 0
        for entry in search_results:
            kmc_dn.electrodes = entry[0]
            kmc_dn.current = entry[1]
            kmc_dn.traffic = entry[2]
            time = entry[3]
            kmc_dn.update_V()
            plt.clf()
            ax0 = plt.subplot(gs[0])
            text = ["I1: ", "I2: ", "C1: ", "C2: ", "C3: ", "C4: ", "C5: ", "O: "]
            text_positions = [(-0.2, 0.75), (0.25, -0.1), (1.05, 0.25), (1.05, 0.75), (-0.2, 0.25), (0.75, -0.1), (0.25, 1.05), (0.75, 1.05)]
            for h in range(len(text)):
                text[h]+="%.3g V"%(kmc_dn.electrodes[h][3]/150)
            visualize_V_and_traffic(kmc_dn, ax_given=ax0, figure=fig, max_traffic=highest_current*time, v_min=vmin, v_max=vmax, text=text, text_positions=text_positions)
            plotax = plt.subplot(gs[1])
            plotax.set_xlim(0, len(search_results))
            plotax.plot(x_data[:i], y_data[:i])
            plotax.plot(x_data[:i], y_data_expected[:i], color='k')
            writer.grab_frame()
            i+=1
