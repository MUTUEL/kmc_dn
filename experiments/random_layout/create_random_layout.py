# Simple script that generates 100 random acceptor/donor layouts

import numpy as np

layouts = 100
N = 20
M = 2
xdim = 1
ydim = 1
zdim = 0

acceptor_layouts = np.random.rand(layouts, N, 3)
acceptor_layouts[:, :, 0] *= xdim
acceptor_layouts[:, :, 1] *= ydim
acceptor_layouts[:, :, 2] *= zdim 

donor_layouts = np.random.rand(layouts, M, 3)
donor_layouts[:, :, 0] *= xdim
donor_layouts[:, :, 1] *= ydim
donor_layouts[:, :, 2] *= zdim 

np.save('acceptor_layouts_20', acceptor_layouts)
np.save('donor_layouts_20', donor_layouts)

