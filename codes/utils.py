"""
Util functions for the scripts
"""

import numpy as np

def lambdas2fa(lambdas):
    a = np.sqrt(0.5)
    b = np.sqrt( (lambdas[0]-lambdas[1])**2 + (lambdas[1]-lambdas[2])**2 + (lambdas[2]-lambdas[0])**2 )
    c = np.sqrt( lambdas[0]**2 + lambdas[1]**2 + lambdas[2]**2 )

    return a*b/c

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

def load_directions(directions_filename):
    dirs = []

    dirs_file = open(directions_filename, 'rt')

    for line in dirs_file.readlines():
        x,y,z = line.split(' ')
        dirs.append( np.array([float(x), float(y), float(z)]) )

    return np.array(dirs)

def load_dirs(dirs_filename, ndirs=500):
    directions = np.fromfile(dirs_filename, dtype=np.float64)
    return np.reshape(directions, (ndirs,3))

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

def getSkewSymmetricMatrixFromVector(v):
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

def getRotationFromDir(a, dir):
    #a = (0,1,0)
    v = np.cross(a, dir)
    c = np.dot(a, dir)

    V = getSkewSymmetricMatrixFromVector(v)

    return np.identity(3) + V + V@V * (1/(1+c))