import matplotlib.pyplot as plt
import numpy as np


def plotPerformance(data, schema):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    i = 0
    colors = ['r', 'b', 'g', 'k', 'y', 'c', 'm', 'w', 'brown', 'tomato', 'purple', 'pink']
    for key in data:
        for s in schema:
            x_data = [x[s[0]] for x in data[key]]
            y_data = [x[s[1]] for x in data[key]]
            l = plt.plot(x_data, y_data, label=(key + s[2]))
            plt.setp(l, linewidth=2, color=colors[i])
            i+=1
    ax.legend()
