# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:38:38 2023

@author: denni
"""

"QSM mapping using to tsv_qsm"

import numpy as np 
import nibabel as nib
import shutil
import os
from Useful_functions import fsl_brain_masking
from Useful_functions import threshold_masking
from Useful_functions import tgv_qsm2 as tgv_qsm2
from Useful_functions import split_all_echoes


class QSM_mapping_mpqMRI():
    

    def __init__(self, arguments):
        
        self.arguments = arguments
        
    
    def run(self, gre='gre2'):
        
        if gre == 'gre2':
            
            dst = os.path.split(self.arguments.phase_image2)[0]
            dirpath = os.path.join(dst, 'QSM')
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
            os.mkdir(dirpath)
            
            phase_image = shutil.copy(self.arguments.phase_image2, dirpath)
            
            mask_path, filename = self.brain_masking()
            shutil.copy(mask_path, dirpath)
            mask_path = os.path.join(dirpath, f"{filename}_brain_mask.nii.gz")
            
            split_all_echoes(phase_image)
            
            filename = os.path.basename(self.arguments.phase_image2)
            if filename[-3:] == '.gz':
                filename = filename[:-7]
            else:
                filename = filename[:-4]
                
        elif gre == 'gre1':
            
            dst = os.path.split(self.arguments.phase_image1)[0]
            dirpath = os.path.join(dst, 'QSM')
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
            os.mkdir(dirpath)
            
            phase_image = shutil.copy(self.arguments.phase_image1, dirpath)
            
            mask_path, filename = self.brain_masking()
            shutil.copy(mask_path, dirpath)
            mask_path = os.path.join(dirpath, f"{filename}_brain_mask.nii.gz")
            
            split_all_echoes(phase_image)
            
            filename = os.path.basename(self.arguments.phase_image1)
            if filename[-3:] == '.gz':
                filename = filename[:-7]
            else:
                filename = filename[:-4]
                
        for i in range(len(self.arguments.QSM_average_echoes_qsm)):
            
            phase_image_echo =  dirpath + '/'+ filename + '_%i_echo.nii.gz'%self.arguments.QSM_average_echoes_qsm[i]
            
            t = str((self.arguments.echotimes[self.arguments.QSM_average_echoes_qsm[i]-1])/1000)
            tgv_qsm2(phase_image_echo, mask_path, t, f=3.0)
            
        for i in range(len(self.arguments.QSM_average_echoes_qsm)):
            
            nifti = dict()
            nifti['%i'%i] = nib.load(dirpath + '/'+ filename + 
                                     '_%i_echoQSM_map_000.nii.gz'%self.arguments.QSM_average_echoes_qsm[i]).get_fdata()
            if i == 0:
                avg = nifti['0']
            else: 
                avg = avg + nifti['%i'%i]
                
        avg = avg/len(self.arguments.QSM_average_echoes_qsm)
        
        # Set NaN values to 0
        avg[np.isnan(avg)] = 0
        
        affine = nib.load(dirpath + '/'+ filename + 
                                 '_%i_echoQSM_map_000.nii.gz'%self.arguments.QSM_average_echoes_qsm[i]).affine
                
        nii_avg = nib.Nifti1Image(avg, affine = affine)
        nib.save(nii_avg, dirpath + '/'+ 'QSM_avg_map.nii.gz' )
        
        return avg

    
    def brain_masking(self):
        
        if self.arguments.phantom is not True:
            
            fsl_brain_masking(self.arguments.gre2_path);
            filename = os.path.basename(self.arguments.gre2_path)
            if filename[-3:] == '.gz':
                filename = filename[:-7]
            else:
                filename = filename[:-4]
                
                
            dst = os.path.split(self.arguments.gre2_path)[0]

            mask_path = dst + '/' + filename + '_brain_mask.nii.gz'
            
            mask = nib.load(mask_path).get_fdata();
            
            self.mask = mask
            
        else: 
            
            mask = threshold_masking(self.arguments.gre1_path, self.arguments.Phantom_mask_threshold_T1mapping);
            
            self.mask = mask
            
            
        return mask_path, filename
        
            
        
            
        
        
        
        
        