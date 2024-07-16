# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:22:34 2023

Author info: 
    Original codes written in MATLAB by: 
        Prof. Ralf Deichmann, 
        Brain Imaging Center Frankfurt

    Modified for Fast B1 mapping with mp_qMRI by:
        Dennis C. Thomas,
        Institute of Neuroradiology, University Hospital Frankfurt
        
    Ported to Python by:
        Mariem Ghazouani,
        Institute of Neuroradiology, University Hospital Frankfurt
        
Date of completion of version 1 of 'Frankfurt_qMRI' python package:
    05/06/2024
"""

import os
import numpy as np
import nibabel as nib
from scipy.signal import convolve
import scipy.signal
import ralf_qMRI_calc_b1map

def B1mapping_Ralf(path, name_b1_noprep, name_b1_prep, basefilename):
    # Set some names for saving data:
    name_b1map = 'standard_B1_MAP.nii'
    name_b1map_smoothed = 'standard_B1_MAP_smoothed.nii'

    # Concatenate paths and filenames
    name_b1_noprep = os.path.join(path, name_b1_noprep)
    name_b1_prep = os.path.join(path, name_b1_prep)
    brain_mask = os.path.join(path, f'{basefilename}_brain_mask.nii.gz')
    wm1 = os.path.join(path, f'{basefilename}_brain_pve_2.nii.gz')
    wm2 = os.path.join(path, f'{basefilename}_brain_pve_3.nii.gz')
    gm = os.path.join(path, f'{basefilename}_brain_pve_1.nii.gz')
    csf = os.path.join(path, f'{basefilename}_brain_pve_0.nii.gz')

    # Parameters for B1 acquisition:
    tr = 11
    fa = 11
    pa = 45
    ndum = 3

    # Relaxation times for relaxation correction in B1 mapping:
    t1_wm = 900
    t1_gm = 1400
    t1_csf = 4500
    t1_other = 1000

    # Load data for B1 mapping and create masks
    print('T1-Mapping: Reading and masking data')
    nii = nib.load(name_b1_noprep).get_fdata()
    affine=nib.load(name_b1_noprep).affine
    volb1 = np.double(nii)

    nii = nib.load(name_b1_prep).get_fdata()
    volb2 = np.double(nii)

    maskeb = np.double(nib.load(brain_mask).get_fdata())
    wm1 = np.double(nib.load(wm1).get_fdata())
    wm2 = np.double(nib.load(wm2).get_fdata())
    gm = np.double(nib.load(gm).get_fdata())
    csf = np.double(nib.load(csf).get_fdata())

    # Perform B1 mapping
    print('Performing B1 mapping')
    wm = wm1 + wm2
    b1map = ralf_qMRI_calc_b1map.ralf_qMRI_calc_b1map(volb1, volb2, fa, tr, pa, ndum, wm, gm, csf, t1_wm, t1_gm, t1_csf, t1_other, maskeb)

    minb1 = 0.7
    maxb1 = 1.3
    b1map = b1map * ((b1map > minb1) & (b1map < maxb1))

    # Outlier removal
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0

    zaehler = convolve(b1map, kernel, mode='same',method='direct')
    nenner = np.round(convolve((b1map > 0).astype(float), kernel, mode='same',method='direct'))
    meanvol = zaehler/(np.finfo(float).eps+nenner)*(nenner>0);

    qq = np.divide(b1map, np.maximum(np.finfo(float).eps, meanvol), out=np.zeros_like(b1map), where=(meanvol > 0))
    mydev = np.abs(qq - 1) * (b1map > 0)
    
    # If mydev exceeds 0.1 (10% deviation), replace the pixel by meanvol
    maxdev = 0.1
    b1map = b1map * (mydev <= maxdev) + meanvol * (mydev > maxdev)

    # Save b1map
    nifti_img_1 = nib.Nifti1Image(b1map, affine=affine)
    nifti_img_1.header.set_data_dtype(np.float64)
    nib.save(nifti_img_1, os.path.join(path, name_b1map))

    # Smooth B1 map
    print('Smoothing B1 map')
    kernel = np.ones((3, 3, 3)) / 27
    zaehler = np.abs(scipy.signal.convolve(b1map, kernel, mode='same', method='direct'))
    nenner = np.abs(scipy.signal.convolve((b1map > 0).astype(float), kernel, mode='same', method='direct'))
    b1map_sm = zaehler * (nenner > 1e-3) / (np.finfo(float).eps + nenner)

    # Save smoothed B1 map
    nifti_img = nib.Nifti1Image(b1map_sm, affine=affine)
    nifti_img.header.set_data_dtype(np.float64)
    nib.save(nifti_img, os.path.join(path, name_b1map_smoothed))

    return b1map_sm

