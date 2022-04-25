import numpy as np
import nibabel as nib
import sys, itertools, math, cmath
from scipy.special import lpmn
from scipy.special import sph_harm
from utils import *

lmax = 6

DWI_FILENAME = sys.argv[1]
SCHEME_FILENAME = sys.argv[2]
SH_FILENAME = sys.argv[3]
bval = int( sys.argv[4] )

dwi = nib.load(DWI_FILENAME)
g,b,idx = load_scheme(SCHEME_FILENAME, bval=bval)
s = dwi.get_fdata()

# get shell
g,b = g[idx], b[idx]
s = s[:,:,:, idx]

xdim,ydim,zdim,N = s.shape
voxels = itertools.product( range(xdim), range(ydim), range(zdim) )

R = int( (lmax+1)*(lmax+2)/2 )
Y = np.zeros((N,R), dtype=np.float32)
S = np.zeros(N, dtype=np.float32)

SH = np.zeros((xdim,ydim,zdim,N), dtype=np.float32)
coefs = np.zeros((xdim,ydim,zdim,R), dtype=np.float32)

for i in range(N):
    theta_i, phi_i, r_i = cart2sph(g[i,0], g[i,1], g[i,2])

    for l in range(0, lmax+1,2):
        for m in range(-l, l+1):
            j = int( l*(l+1)/2 + m )
            Ylm = sph_harm(abs(m), l, phi_i, theta_i)

            if m < 0:
                Y[i,j] = np.sqrt(2) * (-1)**m * Ylm.imag
            else:
                Y[i,j] = np.sqrt(2) * (-1)**m * Ylm.real

for x,y,z in voxels:
    S = s[x,y,z, :]

    coefs[x,y,z, :] = np.linalg.lstsq(Y, S) [0]
    SH[x,y,z, :] = Y@coefs[x,y,z, :]

nib.save( nib.Nifti1Image(coefs, dwi.affine, dwi.header), 'sh_coefs.nii' )
nib.save( nib.Nifti1Image(SH, dwi.affine, dwi.header), SH_FILENAME )
#"""