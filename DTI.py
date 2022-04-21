import numpy as np
import nibabel as nib
import sys, itertools
from utils import *

DWI_FILENAME    = sys.argv[1]
SCHEME_FILENAME = sys.argv[2]
DTI_FILENAME    = sys.argv[3]

dwi = nib.load(DWI_FILENAME)
g,b,idx = load_scheme(SCHEME_FILENAME)
s = dwi.get_fdata()
mask = nib.load( 'mask.nii' ).get_fdata()

xdim,ydim,zdim,N = dwi.shape[0], dwi.shape[1], dwi.shape[2], dwi.shape[3]
voxels = itertools.product( range(xdim), range(ydim), range(zdim) )

G = np.zeros((N,7), dtype=np.float32)
S = np.zeros(N, dtype=np.float32)
D = np.zeros((xdim,ydim,zdim,6), dtype=np.float32)

for i in range(N):
    G[i, 0] = 1
    G[i,1:] = (-b[i]) * np.array( [g[i,0]**2, g[i,1]**2, g[i,2]**2, g[i,0]*g[i,1], g[i,0]*g[i,2], g[i,1]*g[i,2]] )

#poner iter 0 en mrtrix

for (x,y,z) in voxels:
    if mask[x,y,z]:
        S = np.log( s[x,y,z, :] )

        D[x,y,z, :] = (np.linalg.inv( G.transpose()@G ) @ G.transpose() @ S) [1:]
        #D[x,y,z, :] = np.linalg.lstsq(G, S) [0] [1:]
        print(D[x,y,z, :])
        #print(np.linalg.lstsq(G, S))

nib.save( nib.Nifti1Image(D , dwi.affine, dwi.header), DTI_FILENAME )
