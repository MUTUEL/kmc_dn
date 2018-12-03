# Make a report ready plot of the SET IV curve

import numpy as np
import matplotlib.pyplot as plt

data = np.load('data.npz')
current = data['current']
bias = data['bias']

def exp_tresh(x):
    '''
    Exp tresholded at 1
    '''
    y = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] <= 0:
            y[i] = np.exp(x[i])
        else:
            y[i] = 1
    return y

def analytical(U_S, U_G, U_D, kT, nu_0 = 1):
    sg_forward = nu_0 * exp_tresh((U_S - U_G)/kT) 
    sg_backward = nu_0 * exp_tresh((U_G - U_S)/kT) 
    gd_forward = nu_0 * exp_tresh((U_G - U_D)/kT) 
    gd_backward = nu_0 * exp_tresh((U_D - U_G)/kT) 
    return sg_forward*gd_forward/(sg_forward + gd_forward)- sg_backward*gd_backward/(sg_backward + gd_backward)

mean_current = np.mean(current, axis = 0)
std_current = np.std(current, axis = 0)

plt.figure()
plt.errorbar(bias, mean_current[1], yerr = std_current[1], fmt='o')
plt.plot(bias, analytical(bias, bias/2, 0, 1))
plt.xlabel(r'$U_S$ (kT)')
plt.ylabel(r'I $(e\nu_0)$')
plt.grid()
plt.show()
