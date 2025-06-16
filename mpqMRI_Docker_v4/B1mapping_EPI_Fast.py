# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:31:50 2024

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

import numpy as np
import nibabel as nib
from ralf_qMRI_calc_local_mean_and_std import ralf_qMRI_calc_local_mean_and_std
from ralf_qMRI_get_image_mask_improved import ralf_qMRI_get_image_mask_improved
from ralf_qMRI_calc_b1map_from_epi_data import ralf_qMRI_calc_b1map_from_epi_data
from ralf_qMRI_improve_B1_map import ralf_qMRI_improve_B1_map
import os
def B1mapping_EPI_fast(path, name_b0map, name_EPI_45deg, name_EPI_90deg,sign_gradient):
    os.chdir(path)
    print('\n######################################\n')
    print('#  STARTING B1-MAPPING FOR RELEASE DATA\n')
    print('######################################\n\n')
    # Set sequence parameters
    best_btp = 5.8  #BTP that was determined for the Siemens release EPI from the phantom measurement
    b0_slice_resol = 4 #interslice distance (in mm) of the B0 map (slice center to slice center), including the interslice gap
    epi_slice_thickness = 2 #slice thickness (in mm) of the EPI data, without interslice gaps
    #REMARK: epi_slice_thickness differs from b0_slice_resol as the latter includes interslice gaps
    te = 21 # echo time TE (in ms) of the EPI sequence: te=19 for sagittal data and te=21 for axial data
    


    # Extended data
    # B1 map with B0 correction:
    name_b1map = 'B1_MAP_from_fast_EPI_standard.nii.gz'

    # Non-Extended data
    # B1 map with B0 correction:
    name_b1map_nonext = 'B1_MAP_from_fast_EPI_standard_nonext.nii.gz'

    # Load B0 map B1 map with B0 correction:
    nii3 = nib.load(os.path.join(path,name_b0map))
    b0 = nii3.get_fdata()
    affine=nib.load(os.path.join(path,name_b0map)).affine


    # Load 45 deg data
    nii = nib.load((os.path.join(path,name_EPI_45deg)))
    vol1 = nii.get_fdata()


    # Load 90 deg data
    nii = nib.load(os.path.join(path,name_EPI_90deg))
    vol2 = nii.get_fdata()
    # Code for deriving mask:
    # We calculate separate masks for vol1 and vol2 and multiply them later
    mm1 = ralf_qMRI_get_image_mask_improved(vol1, 4, 0)
    
    # Step for further exclusion of pixels with low SNR from mm1:
    #Average values in vol1 (only pixels inside mm1) across large areas (15x15x15)
    
    meanvol, stdvol = ralf_qMRI_calc_local_mean_and_std(vol1, mm1, 15)
    # Get the quotient of the original vol1 and meanvol:
    qq = vol1 / (np.finfo(float).eps + meanvol) * (meanvol > 0)
    
    # This quotient is scaled and should be 1 on average
    # Exclude pixels where the quotient is 0.5 or less:
    mm1 = mm1 * (qq > 0.5)
    #same procedure for vol2:
    mm2 = ralf_qMRI_get_image_mask_improved(vol2, 4, 0)
    
    meanvol, stdvol = ralf_qMRI_calc_local_mean_and_std(vol2, mm2, 15)
    qq = vol2 / (np.finfo(float).eps+ meanvol) * (meanvol > 0)
    mm2 = mm2 * (qq > 0.5)
   
    # For total mask, take product of mm1 and mm2:

    mm = mm1 * mm2 

    # Calculate B1 map
    b1map = ralf_qMRI_calc_b1map_from_epi_data(vol1, vol2, b0, b0_slice_resol, epi_slice_thickness, te, sign_gradient, best_btp, mm)
    
    # Improve B1 map
    # This function yields 2 B1 maps:
    # 1. b1map_improved is valid for the areas covered by the input b1map where B1 is within a given range (here: 0.7 to 1.3)
    # 2. b1map_improved_extended is basically b1map_improved, but several (here: 5) surrounding layers have been added to areas not covered by b1map_improved
    b1map_improved, b1map_improved_extended = ralf_qMRI_improve_B1_map(b1map, 0.7, 1.3, 5)

    # Save B1 maps
   
    info = nib.load(os.path.join(path,name_EPI_90deg)).header
    nii_b1map = nib.Nifti1Image(b1map_improved_extended, affine=affine, header=info)
    nib.save(nii_b1map, name_b1map)

    nii_b1map_nonext = nib.Nifti1Image(b1map_improved, affine=affine, header=info)
    nib.save(nii_b1map_nonext, name_b1map_nonext)

    print('\n######################################')
    print('#  FINISHED: B1-MAPPING FOR RELEASE DATA')
    print('######################################\n\n\n')
    return b1map_improved








