#!/usr/bin/env python

import scipy.io
import os,sys
import numpy as np
sys.path.append('/jukebox/ramadge/pohsuan/PyMVPA-hyper/')
import mvpa2
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets.mri import map2nifti
from scipy.signal import butter, lfilter
import nibabel as nib

roi = 'pmc'


sig = 100.0
featNum = 50
iters = 10

spatial = False

template_path = '/jukebox/fastscratch/janice/sherlock_movie/'
output_path = '/jukebox/ramadge/pohsuan/SRM/data/output/synthesized/sherlock_smooth_'+roi+'/'

if spatial:
    input_path = '/jukebox/fastscratch/pohsuan/SRM/data/working/sherlock_smooth_pmc_noLR/813vx/1976TR/'
    in_fname = 'mysseg_1st_winsize9/srm_spatial_sig{sig}/{featNum}feat/rand0/all/srm_spatial__{iters}.npz'.format(sig=sig, featNum = featNum, iters = iters)
    algoname = 'srm_spatial'
else:
    input_path = '/fastscratch/pohsuan/pHA/data/working/sherlock_smooth_pmc_noLR/813vx/1976TR/'
    in_fname = 'mysseg_1st_winsize9/pha_em/{featNum}feat/rand0/loo0/pha_em__{iters}.npz'.format(sig=sig, featNum = featNum, iters = iters)
    algoname = 'srm'

if not os.path.exists(output_path):
    os.makedirs(output_path)

mask_fname = os.path.join(template_path, 'PMC_3mm.nii')

subj_idx_all = range(1,18)
#subj_idx_all = range(1,2)
subj_idx_all.remove(5)
movie_all = np.empty((len(subj_idx_all),1), dtype=object)

# load mask
mask = nib.load(mask_fname)
maskdata = mask.get_data()
(i,j,k) = np.where(maskdata>0)

# load alignment results
ws = np.load(input_path+in_fname)
if spatial:
    W = ws['W']
    S = ws['S']
else:
    W = ws['bW']
    S = ws['ES']

datadim = maskdata.shape  #+(S.shape[1],)

selefeat = 1

print in_fname
for idx,subj_idx in enumerate(subj_idx_all):
    syn_data = np.zeros(datadim, dtype=np.float32)
    if spatial:
        syn_data[i,j,k] = W[:,selefeat,idx]
    else:
        ##TODO this is incorrect! FIX THIS!
        syn_data[i,j,k] = W[idx*813:(idx+1)*813:,selefeat]
    
    #print W[:,selefeat,idx]
    #data = np.ones((32, 32, 15, 100), dtype=np.int16)

    out_fname = 'sherlock_{algoname}_sig{sig}_featNum{featNum}_s{subj_idx}_selefeat{selefeat}th_iters{iters}'\
                .format(algoname=algoname, sig=sig, featNum=featNum, subj_idx = subj_idx, selefeat = selefeat, iters = iters) 
    img_out = nib.Nifti1Image(syn_data,None)#, np.eye(4))

    nib.save(img_out,  output_path+out_fname+'.nii.gz')
    print output_path+out_fname+'.nii.gz'
