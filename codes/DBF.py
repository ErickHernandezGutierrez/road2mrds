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

dwi  = nib.load(DWI_FILENAME)
mask = nib.load(MASK_FILENAME).get_fdata()
g,b,_ = load_scheme(SCHEME_FILENAME)
S = dwi.get_fdata()

X,Y,Z,N = S.shape
voxels = itertools.product( range(X), range(Y), range(Z) )

lambdas = np.array([0.0003,0.0003,0.015])

design_matrix = np.zeros((N,ndirs), dtype=np.float32)
S_hat = np.zeros((X,Y,Z,N), dtype=np.float32)
alphas = np.zeros((X,Y,Z,ndirs), dtype=np.float32)
bases = np.zeros((X,Y,Z,6), dtype=np.float64)

for i in range(N):
    for j in range(ndirs):
        Rj = getRotationFromDir( (0,0,1), dirs[j])
        Tj = Rj.transpose() @ np.diag(lambdas) @ Rj
        design_matrix[i,j] = np.exp( -b[i]*(g[i].transpose()@Tj@g[i]) )

for (x,y,z) in voxels:
    if mask[x,y,z]:
        #alpha = np.linalg.lstsq(design_matrix, S) [0] # usar non-negative least squaresO
        sol, res = nnls(design_matrix, S[x,y,z, :])

        T = np.diag(lambdas)
        bases[x,y,z, :] = np.array([ T[0,0], T[1,1], T[2,2], T[0,1], T[0,2], T[1,2] ])
        alphas[x,y,z, :] = sol
        S_hat[x,y,z, :] = design_matrix@sol

#mandar volumen de alphas con las direccciones a mrtrix
nib.save( nib.Nifti1Image(alphas , dwi.affine, dwi.header), 'dbf_alphas.nii' )
nib.save( nib.Nifti1Image(bases , dwi.affine, dwi.header), 'dbf_base.nii' )
nib.save( nib.Nifti1Image(S_hat , dwi.affine, dwi.header), DBF_FILENAME )


