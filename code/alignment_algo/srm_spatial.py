#!/usr/bin/env python

# Non-probabilistic Shared Response Model with sptial contiguity regularization
# using Laplacian Matrix

# movie_data is a three dimensional matrix of size voxel x TR x nsubjs
# movie_data[:,:,m] is the data for subject m, which will be X_m^T in the standard 
# mathematic notation
# Needs args.roi, args.kernel and args.alpha (args.gamma if kernel is gauss)

import numpy as np, scipy, random, sys, math, os
from scipy import stats

def align(movie_data, options, args, lrh):
  
  print 'SRM spatial',
  sys.stdout.flush()

  nvoxel = movie_data.shape[0]
  nTR    = movie_data.shape[1]
  nsubjs = movie_data.shape[2]
  align_algo = args.align_algo
  nfeature = args.nfeature

  current_file = options['working_path']+align_algo+'_'+lrh+'_current.npz' 
  dist_file = '/jukebox/ramadge/pohsuan/data/working/PMC_3mm_width2_laplacian_mtx/L_mtx.npz'

  movie_data_zscore = np.zeros ((nvoxel,nTR,nsubjs))
  for m in range(nsubjs):
      movie_data_zscore[:,:,m] = stats.zscore(movie_data[:,:,m].T ,axis=0, ddof=1).T

  if not os.path.exists(current_file):
      W = np.zeros((nvoxel,nfeature,nsubjs))
      S = np.zeros((nfeature,nTR))

      #initialization
      #TODO think about what would be a reasonable initialization
      if args.randseed != None:
          print 'randinit',
          np.random.seed(args.randseed)
          A = np.mat(np.random.random((nvoxel,nfeature)))
          Q, R_qr = np.linalg.qr(A)
      else:
          Q = np.eye(nvoxel,nfeature)

      S_inv_tmp = np.zeros((nfeature,nfeature))
      S_tmp = np.zeros((nfeature,nTR))
      for m in range(nsubjs):
          W[:,:,m] = Q
          S_inv_tmp = S_inv_tmp + W[:,:,m].T.dot(W[:,:,m])
          S_tmp = S_tmp + W[:,:,m].T.dot(movie_data_zscore[:,:,m])
      S = np.linalg.inv(S_inv_tmp).dot(S_tmp)

      niter = 0
      np.savez_compressed(options['working_path']+align_algo+'_'+lrh+'_'+str(niter)+'.npz',\
                      W = W, S = S, niter=niter)
  else:
      workspace = np.load(current_file)
      niter = workspace['niter']
      workspace = np.load(options['working_path']+align_algo+'_'+lrh+'_'+str(niter)+'.npz')
      W = workspace['W']
      S = workspace['S']
      niter = workspace['niter']

  # Load distance matrix
  dist = np.load(dist_file)
  
  # L is a discrete Laplacian operator on 3D voxel space
  L = dist['L']
  D = L.T.dot(L)

  print str(niter+1)+'th',
  # Update W
  for m in range(nsubjs):
      print '.',
      sys.stdout.flush()
      W[:,:,m] = scipy.linalg.solve_sylvester(args.sigma*D,S.dot(S.T),movie_data_zscore[:,:,m].dot(S.T))
      # Normalize each column of W
      W_norm = np.linalg.norm(W[:,:,m],axis=0)
      W[:,:,m] = W[:,:,m]/W_norm

  # Update S
  S_inv_tmp = np.zeros((nfeature,nfeature))
  S_tmp = np.zeros((nfeature,nTR))
  for m in range(nsubjs):
      S_inv_tmp = S_inv_tmp + W[:,:,m].T.dot(W[:,:,m])
      S_tmp = S_tmp + W[:,:,m].T.dot(movie_data_zscore[:,:,m])

  S = np.linalg.inv(S_inv_tmp).dot(S_tmp)
  print '.'
  
  new_niter = niter + 1
  np.savez_compressed(current_file, niter = new_niter)
  np.savez_compressed(options['working_path']+align_algo+'_'+lrh+'_'+str(new_niter)+'.npz',\
                      W = W, S = S, niter=new_niter)
  
  # print objectives
  align_obj = 0.
  cont_obj = 0.
  for m in range(nsubjs):
    align_obj = align_obj+np.linalg.norm(movie_data_zscore[:,:,m]-W[:,:,m].dot(S),ord='fro')**2
    cont_obj = cont_obj+np.trace(W[:,:,m].T.dot(D).dot(W[:,:,m]))
    
  print 'align_obj = '+str(align_obj)
  print 'cont_obj = '+str(cont_obj)
  print 'obj = '+str(align_obj+args.sigma*cont_obj)
  sys.stdout.flush()
  return new_niter
