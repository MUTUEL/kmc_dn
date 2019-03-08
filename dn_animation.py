import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# Requires installation: sudo apt-get install ffmpeg

def getWriter(fps, title):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title)
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer

def initScatterAnimation(kmc):
    plt.clf()
    fig = plt.figure()
    acceptors, = plt.plot(kmc.xCoords[:kmc.N], kmc.yCoords[:kmc.N], 'o', color='black')
    donors, = plt.plot(kmc.xCoords[kmc.N:], kmc.yCoords[kmc.N:], 'o', color='red')

    return acceptors, donors, fig



def animateTransition(kmc, donors, acceptors, index, target_pos, writer, splits):
    acceptorDataX = [kmc.xCoords[j] for j in range(kmc.N)]
    acceptorDataY = [kmc.yCoords[j] for j in range(kmc.N)]
    donorDataX = [kmc.xCoords[j] for j in range(kmc.N, kmc.N+kmc.M)]
    donorDataY = [kmc.yCoords[j] for j in range(kmc.N, kmc.N+kmc.M)]

    for i in range(splits):
        if index < kmc.N:
            acceptorDataX[index] = (kmc.xCoords[index]*(splits-i)+target_pos[0]*i) / splits
            acceptorDataY[index] = (kmc.yCoords[index]*(splits-i)+target_pos[1]*i) / splits
        else:
            donorDataX[index-kmc.N] = (kmc.xCoords[index]*(splits-i)+target_pos[0]*i) / splits
            donorDataY[index-kmc.N] = (kmc.yCoords[index]*(splits-i)+target_pos[1]*i) / splits
        donors.set_data(donorDataX, donorDataY)
        acceptors.set_data(acceptorDataX, acceptorDataY)
        writer.grab_frame()

