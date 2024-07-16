# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:43:15 2023

@author: denni
"""

import nibabel as nib
import numpy as np
import os
from Dennis.Useful_functions import fsl_brain_masking
from Dennis.Useful_functions import combine_echoes
from Dennis.Useful_functions import threshold_masking

class t2_map():
    
    def __init__(self, arguments):
        
        self.arguments = arguments
        self.nechoes = arguments['nechoes']
        self.masking_method = arguments['masking_method']
        
    def run(self, save=True, masking=True):
        
        path = combine_echoes(self.arguments, self.nechoes)
        
        se = nib.load(path).get_fdata()
        
        if masking:
            
            mask = self.masking()
            mask = mask[:,:,:,None]
        
        else:
            mask = np.ones(se.shape)

        se = se*mask
        log_se = np.log(se)

        
        se_shape = np.shape(se)
        
        se = np.reshape(log_se, (np.size(log_se)//np.shape(log_se)[-1], 
                                     np.shape(log_se)[-1]))
        
        M0, t2 = self.linear_fit(self.arguments['echotimes'], se)
        M0 = np.exp(M0)
        T2  = -1./ t2
        
        T2[T2<0] = 1000
        T2 = T2.clip(0, 1000)
        
        M0 = np.reshape(M0, se_shape[0:-1])
        T2 = np.reshape(T2, se_shape[0:-1])
        
        self.M0 = M0
        self.T2 = T2

        if save:
            
            self.save()
        
        return M0, T2

        
    
    def save(self):
        
        affine = nib.load(self.arguments['se1_path']).affine
        
        M0 = self.M0
        
        T2 = self.T2

        M0_nii = nib.Nifti1Image(M0, affine = affine)
        T2_nii = nib.Nifti1Image(T2, affine = affine)
        
               
        dst = os.path.split(self.arguments['se1_path'])[0]
        T2_path = dst + '/' + 'T2.nii'
        
        M0_path = dst + '/' + 'M0.nii'

        
        nib.save(T2_nii, T2_path)

        nib.save(M0_nii, M0_path)

        
        
    def linear_fit(self, echotimes, timeseries):
        # just format and add one column initialized at 1
        X_mat=np.vstack((np.ones(len(echotimes)), echotimes)).T
        
        # cf formula : linear-regression-using-matrix-multiplication 
        tmp = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)
        M0 = tmp.dot(timeseries.T)[0] # 0 => intercept_
        t2s = tmp.dot(timeseries.T)[1]# 1 => coef_
        
        return M0, t2s  
        
    
    def masking(self):
        
        
        if self.masking_method == 'thresholding':
            
            mask = threshold_masking(self.arguments['se1_path'], threshold = self.arguments['mask_threshold'])
            
        elif self.masking_method == 'brain':
            
        
            fsl_brain_masking(self.arguments['se1_path'])
            
            filename = os.path.basename(self.arguments['se1_path'])
            if filename[-3:] == '.gz':
                filename = filename[:-7]
            else:
                filename = filename[:-4]
                
                
            dst = os.path.split(self.arguments['se1_path'])[0]
    
            mask_path = dst + '/' + filename + '_brain_mask.nii.gz'

            mask = nib.load(mask_path).get_fdata();

        return mask
    
    
    
        
