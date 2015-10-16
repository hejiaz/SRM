#!/usr/bin/env python

# using pPCA for multisubject fMRI data alignment

#movie_data is a three dimensional matrix of size voxel x TR x nsubjs
#movie_data[:,:,m] is the data for subject m, which will be X_m^T in the standard 
#mathematic notation

# do PCA on bX (nsubjs*nvoxel x nTR) concatenate the data vertically


import numpy as np, scipy, random, sys, math, os
from scipy import stats
import sys
from sklearn.decomposition import PCA, FastICA

def align(movie_data, options, args, lrh):
    print 'Group ICA'
    nvoxel = movie_data.shape[0]
    nTR    = movie_data.shape[1]
    nsubjs = movie_data.shape[2]
    
    align_algo = args.align_algo
    nfeature   = args.nfeature

    if not os.path.exists(options['working_path']):
        os.makedirs(options['working_path'])

    # zscore the data
    bY = np.zeros((nTR,nvoxel,nsubjs))
    for m in range(nsubjs):
        bY[:,:,m] = stats.zscore(movie_data[:,:,m].T ,axis=0, ddof=1)
    del movie_data
    
    # First PCA
    Fi = np.zeros((nTR,nfeature,nsubjs))
    Xi = np.zeros((nfeature,nvoxel,nsubjs))
    X_stack = np.zeros((nfeature*nsubjs,nvoxel))
    
    for m in range(nsubjs):
        U, s, VT = np.linalg.svd(bY[:,:,m], full_matrices=False)
        Fi[:,:,m] = U[:,range(nfeature)]
        Xi[:,:,m] = np.diag(s[range(nfeature)]).dot(VT[range(nfeature),:])
        X_stack[m*nfeature:(m+1)*nfeature,:] = Xi[:,:,m]
  
    # Choose N for second PCA
    U, s, VT = np.linalg.svd(X_stack, full_matrices=False)
    r = np.linalg.matrix_rank(X_stack)
    AIC  = np.zeros((r-1))
    MDL = np.zeros((r-1))
    tmp1 = 1.0
    tmp2 = 0.0
    for N in range(r-2,-1,-1):
        tmp1 = tmp1*s[N+1]
        tmp2 = tmp2+s[N+1]
        L_N = np.log(tmp1**(1/(r-1-N))/((tmp2/(r-1-N))))
        AIC[N] = -2*nvoxel*(nfeature*nsubjs-N-1)*L_N + 2*(1+(N+1)*nfeature+N/2)
        MDL[N] = -nvoxel*(nfeature*nsubjs-N-1)*L_N + 0.5*(1+(N+1)*nfeature+N/2)*np.log(nvoxel)
    
    nfeat2 = int(round(np.mean([np.argmin(AIC), np.argmin(MDL)])))+1 # N
    
    # Second PCA
    G = U[:,range(nfeat2)]
    X = np.diag(s[range(nfeat2)]).dot(VT[range(nfeat2),:]) # N-by-V
    
    # ICA
    randseed = 0
    np.random.seed(randseed) # randseed = 0
    tmp = np.mat(np.random.random((nfeat2,nfeat2)))
    
    ica = FastICA(n_components= nfeat2, max_iter=500,w_init=tmp,whiten=False,random_state=randseed)
    St = ica.fit_transform(X.T)
    S = St.T
    A = ica.mixing_

    # Partitioning
    Gi = np.zeros((nfeature,nfeat2,nsubjs))
    Si = np.zeros((nfeat2,nvoxel,nsubjs))

    for m in range(nsubjs):
        Gi[:,:,m] = G[m*nfeature:(m+1)*nfeature,:]
        Si[:,:,m] = np.linalg.pinv(Gi[:,:,m].dot(A)).dot(Xi[:,:,m])
        

    # Forming the factorization matrices such that Yi.T = bWi*bSi
    bW = np.zeros((nvoxel,nfeat2,nsubjs))
    bS = np.zeros((nfeat2,nTR,nsubjs))
    for m in range(nsubjs):
        bW[:,:,m] = Si[:,:,m].T
        bS[:,:,m] = (Fi[:,:,m].dot(Gi[:,:,m]).dot(A)).T
    
    
    niter = 10
    # initialization when first time run the algorithm
    np.savez_compressed(options['working_path']+align_algo+'_'+lrh+'_'+str(niter)+'.npz',\
          bW = bW, bS = bS, niter=niter)
    return niter
