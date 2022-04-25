from logging import exception
import numpy as np
import nibabel as nib
import sys, itertools, math, cmath
from scipy.special import lpmn
from scipy.optimize import nnls
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DWI_FILENAME    = sys.argv[1]
SCHEME_FILENAME = sys.argv[2]
MASK_FILENAME   = sys.argv[3]
DIRS_FILENAME   = sys.argv[4]
DBF_FILENAME    = sys.argv[5]

dirs = load_directions(DIRS_FILENAME)
ndirs = dirs.shape[0]

dwi = nib.load(DWI_FILENAME)
g,b,idx = load_scheme(SCHEME_FILENAME)
s = dwi.get_fdata()
mask = nib.load( MASK_FILENAME ).get_fdata()

xdim,ydim,zdim,N = dwi.shape[0], dwi.shape[1], dwi.shape[2], dwi.shape[3]
voxels = itertools.product( range(xdim), range(ydim), range(zdim) )

lambdas = np.array([0.0003,0.0003,0.004])

PHI = np.zeros((N,ndirs), dtype=np.float32)
S = np.zeros(N, dtype=np.float32)
S_hat = np.zeros((xdim,ydim,zdim,N), dtype=np.float32)
alphas = np.zeros((xdim,ydim,zdim,ndirs), dtype=np.float32)

for i in range(N):
    for j in range(ndirs):
        Rj = getRotationFromDir( (0,1,0), dirs[j])
        Tj = Rj.transpose() @ np.diag(lambdas) @ Rj
        PHI[i,j] = np.exp( -b[i]*(g[i].transpose()@Tj@g[i]) )

for (x,y,z) in voxels:
    if mask[x,y,z]:
        S = s[x,y,z, :]
        #alpha = np.linalg.lstsq(PHI, S) [0] # usar non-negative least squaresO
        alpha, res = nnls(PHI, S)
        S_hat[x,y,z, :] = PHI@alpha
        alphas[x,y,z, :] = alpha

#mandar volumen de alphas con las direccciones a mrtrix
nib.save( nib.Nifti1Image(alphas , dwi.affine, dwi.header), 'alphas.nii' )
nib.save( nib.Nifti1Image(S_hat , dwi.affine, dwi.header), DBF_FILENAME )

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

ax.plot(dirs[:,0], dirs[:,1], dirs[:,2], linewidth=0, marker='o', markersize=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()#"""
