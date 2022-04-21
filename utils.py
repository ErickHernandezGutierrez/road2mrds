"""
Util functions for the scripts
"""

import numpy as np

def load_scheme(scheme_filename, bval=None):
    bvecs, bvals = [], []
    idx = []

    scheme_file = open(scheme_filename, 'rt')

    for i,line in enumerate(scheme_file.readlines()):
        x,y,z,b = line.split(' ')
        x,y,z,b = float(x),float(y),float(z),float(b)
        bvecs.append( np.array([x,y,z]) )
        bvals.append( b )
        if bval!=None and np.abs(bval-b) < 1e-6:
            idx.append(i)

    return np.array(bvecs), np.array(bvals), np.array(idx)

def cart2sph(x, y, z):
    h = np.hypot(x, y)
    r = np.hypot(h, z)
    theta = np.arctan2(h, z)
    if x>0:
        phi = np.arctan2(y, x)
    elif x<0 and y>=0:
        phi = np.arctan2(y, x)+np.pi
    elif x<0 and y<0:
        phi = np.arctan2(y, x)-np.pi
    elif np.abs(x<1e-6) and y>0:
        phi = np.pi/2
    elif np.abs(x<1e-6) and y<0:
        phi = -np.pi/2
    else:
        phi=0#np.inf
    
    return np.array( [theta, phi, r] )

