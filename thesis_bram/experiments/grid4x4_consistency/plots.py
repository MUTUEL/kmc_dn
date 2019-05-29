# Make some collection of subplots

import numpy as np
import matplotlib.pyplot as plt

data = np.load('current_map.npz')
current = data['current']
bias = data['bias']
I_0 = data['I_0']
ab = data['ab_R']

for i in range(len(ab)):
    fig, axs = plt.subplots(2, 5, constrained_layout=True)
    axs = axs.flatten()
    fig.suptitle(f'a/R = {ab[i]}')
    for j in range(len(I_0)):
        axs[j].plot(bias, current[j, i])
        axs[j].set_title(f'I_0 = {I_0[j]:.4}')
        axs[j].set_xlabel('bias')
        axs[j].set_ylabel('current')
        
plt.show()
