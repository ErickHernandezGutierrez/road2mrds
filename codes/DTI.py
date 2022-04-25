import numpy as np
import nibabel as nib
import sys, itertools
from scipy.optimize import nnls
from utils import *

DWI_FILENAME    = sys.argv[1]
SCHEME_FILENAME = sys.argv[2]
MASK_FILENAME   = sys.argv[3]
DTI_FILENAME    = sys.argv[4]

dwi = nib.load(DWI_FILENAME)
g,b,idx = load_scheme(SCHEME_FILENAME, bval=1000)
s = dwi.get_fdata()
mask = nib.load( MASK_FILENAME ).get_fdata()

#g,b = g[idx], b[idx]
#s = s[:,:,:, idx]

xdim,ydim,zdim,N = s.shape
voxels = itertools.product( range(xdim), range(ydim), range(zdim) )
# [ (0,0,0), (0,0,1), ..., (X-1,Y-1,Z-1) ]

G = np.zeros((N,7), dtype=np.float32) #matriz de dise
S = np.zeros(N, dtype=np.float32) 
D = np.zeros((xdim,ydim,zdim,6), dtype=np.float32)
S0 = np.zeros((xdim,ydim,zdim), dtype=np.float32)

for i in range(N):
    G[i, 0] = 1
    G[i, 1:7] = (-b[i]) * np.array([ g[i,0]**2, g[i,1]**2, g[i,2]**2, g[i,0]*g[i,1], g[i,0]*g[i,2], g[i,1]*g[i,2] ])
    #G[i, 6] = 1

#poner iter 0 en mrtrix y poner -ols

for (x,y,z) in voxels:
    if mask[x,y,z]:
        S = np.log( s[x,y,z, :] )

        #sol = [ (np.linalg.inv( G.transpose()@G ) @ G.transpose() @ S), -1]
        sol = np.linalg.lstsq(G, S)
        #sol = nnls(G, S)

        print('D = ', end='')
        for d in sol[0]:
            print('%.12f ' % d, end=' ')
        print(' || residuals=%.12f' % sol[1])

        D[x,y,z, :] = sol[0] [1:7]
        S0[x,y,z] = np.exp(sol[0][0])

        """
        D_mat = np.array([
            np.array([D[x,y,z, 0], D[x,y,z, 3], D[x,y,z, 4]]),
            np.array([D[x,y,z, 3], D[x,y,z, 1], D[x,y,z, 5]]),
            np.array([D[x,y,z, 4], D[x,y,z, 5], D[x,y,z, 2]])
        ])
        eigenvals, eigenvecs = np.linalg.eigh(D_mat)
        eigenvals.clip( min=1e-6 )

        print('lambdas = ', end='')
        for eigenval in eigenvals:
            print('%.12f' % eigenval, end=' ')
        print('\n')

        D[x,y,z, :] = np.array([ eigenvals[0], eigenvals[1], eigenvals[2], 0, 0, 0 ])"""

#fit_result = np.einsum('...ij,...j', np.linalg.pinv(G), np.log(s))
#print(fit_result.shape)
#idx = [0, 2, 5, 1, 3, 4]
#D[x,y,z, :] = fit_result[idx]

#nib.save( nib.Nifti1Image(fit_result , dwi.affine, dwi.header), 'calis.nii' )
nib.save( nib.Nifti1Image(S0 , dwi.affine, dwi.header), 'S0.nii' )
nib.save( nib.Nifti1Image(D , dwi.affine, dwi.header), DTI_FILENAME )
