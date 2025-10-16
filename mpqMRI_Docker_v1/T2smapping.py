# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:17:47 2023

@author: denni
"""
import os
import nibabel as nib
import numpy as np
from Useful_functions import fsl_brain_masking

class t2s_map_mpqMRI():
    
    def __init__(self, arguments):
        
        self.arguments = arguments
        self.nechoes = len(arguments.echotimes)
        
    def run(self, nechoes=6, save=True):
            
        gre1 = nib.load(self.arguments.gre1_path).get_fdata()
        gre2 = nib.load(self.arguments.gre2_path).get_fdata()
        
        mask1 = self.masking()[0]
        mask2 = self.masking()[1]
        
        mask1 = mask1[:,:,:,None]
        mask2 = mask2[:,:,:,None]
        
        gre1 = gre1 * mask2
        gre2 = gre2 * mask2
        
        log_gre1 = np.log(gre1)
        log_gre2 = np.log(gre2)
        
        gre_shape = np.shape(gre1)
        
        gre1 = np.reshape(log_gre1, (np.size(log_gre1)//np.shape(log_gre1)[-1], np.shape(log_gre1)[-1]))
        gre2 = np.reshape(log_gre2, (np.size(log_gre2)//np.shape(log_gre2)[-1], np.shape(log_gre2)[-1]))
        
        M0_gre1, t2s_gre1 = self.linear_fit(self.arguments.echotimes, gre1)
        M0_gre1 = np.exp(M0_gre1)
        T2Star_gre1  = -1./ t2s_gre1
        T2Star_gre1[T2Star_gre1 < 0] = 1000  # Set negative values to 0
        T2Star_gre1[T2Star_gre1 > 1000] = 1000  # Set values > 1000 to 1000
        T2Star_gre1[np.isnan(T2Star_gre1)] = 0  # Set NaN values to 0
        
        M0_gre2, t2s_gre2 = self.linear_fit(self.arguments.echotimes, gre2)
        M0_gre2 = np.exp(M0_gre2)
        T2Star_gre2  = -1./ t2s_gre2
        T2Star_gre2[T2Star_gre2 < 0] = 1000  # Set negative values to 0
        T2Star_gre2[T2Star_gre2 > 1000] = 1000  # Set values > 1000 to 1000
        T2Star_gre2[np.isnan(T2Star_gre2)] = 0  # Set NaN values to 0
        
        M0_avg = M0_gre1 + M0_gre2
        T2Star_avg = (T2Star_gre1 + T2Star_gre2) / 2
        T2Star_avg[T2Star_avg < 0] = 1000  # Set negative values to 0
        T2Star_avg[T2Star_avg > 1000] = 1000  # Set values > 1000 to 1000
        T2Star_avg[np.isnan(T2Star_avg)] = 0  # Set NaN values to 0
        
        gre12 = gre1 + gre2
        
        M0_gre12, t2s_gre12 = self.linear_fit(self.arguments.echotimes, gre12)
        M0_gre12 = np.exp(M0_gre12)
        T2Star_gre12  = -2. / t2s_gre12  # Multiplied by 2
        T2Star_gre12[T2Star_gre12 < 0] = 1000  # Set negative values to 0
        T2Star_gre12[T2Star_gre12 > 1000] = 1000  # Set values > 1000 to 1000
        T2Star_gre12[np.isnan(T2Star_gre12)] = 0  # Set NaN values to 0
        
        M0_gre1 = np.reshape(M0_gre1, gre_shape[:-1])
        M0_gre2 = np.reshape(M0_gre2, gre_shape[:-1])
        M0_avg = np.reshape(M0_avg, gre_shape[:-1])
        avg_M0 = np.reshape(M0_gre12, gre_shape[:-1])
        
        T2Star_gre1 = np.reshape(T2Star_gre1, gre_shape[:-1])
        T2Star_gre2 = np.reshape(T2Star_gre2, gre_shape[:-1])
        T2Star_avg = np.reshape(T2Star_avg, gre_shape[:-1])
        avg_T2Star = np.reshape(T2Star_gre12, gre_shape[:-1])
        
        self.M0_gre1 = M0_gre1
        self.M0_gre2 = M0_gre2
        self.M0_avg = M0_avg
        self.avg_M0 = avg_M0
        
        self.T2Star_gre1 = T2Star_gre1
        self.T2Star_gre2 = T2Star_gre2
        self.T2Star_avg = T2Star_avg
        self.avg_T2Star = avg_T2Star
        
        if save:
            self.save()
        
        return M0_gre1, M0_gre2, M0_avg, avg_M0, T2Star_gre1, T2Star_gre2, T2Star_avg, avg_T2Star

    
    def save(self):
        
        affine1 = nib.load(self.arguments.gre1_path).affine
        affine2 = nib.load(self.arguments.gre2_path).affine
        
        M0_gre1 = self.M0_gre1
        M0_gre2 = self.M0_gre2
        M0_avg = self.M0_avg
        avg_M0 = self.avg_M0
        
        T2Star_gre1 = self.T2Star_gre1
        T2Star_gre2 = self.T2Star_gre2
        T2Star_avg = self.T2Star_avg
        avg_T2Star = self.avg_T2Star
        
        T2Star_gre1_nii = nib.Nifti1Image(T2Star_gre1, affine = affine1)
        T2Star_gre2_nii = nib.Nifti1Image(T2Star_gre2, affine = affine2)
        T2Star_avg_nii = nib.Nifti1Image(T2Star_avg, affine = affine1)
        avg_T2Star_nii = nib.Nifti1Image(avg_T2Star, affine = affine1)
        
        
        M0_gre1_nii = nib.Nifti1Image(M0_gre1, affine = affine1)
        M0_gre2_nii = nib.Nifti1Image(M0_gre2, affine = affine2)
        M0_avg_nii = nib.Nifti1Image(M0_avg, affine = affine1)
        avg_M0_nii = nib.Nifti1Image(avg_M0, affine = affine1)
        
        
        dst = os.path.split(self.arguments.gre2_path)[0]
        T2Star_gre1_path = f"{dst}/T2Star_FA{int(self.arguments.FA1):02d}.nii.gz"
        T2Star_gre2_path = f"{dst}/T2Star_FA{int(self.arguments.FA2):02d}.nii.gz"
        T2Star_avg_path = f"{dst}/T2Star_avg.nii.gz"
        avg_T2Star_path = f"{dst}/avg_T2Star.nii.gz"
        
        M0_gre1_path = f"{dst}/M0_FA{int(self.arguments.FA1):02d}.nii.gz"
        M0_gre2_path = f"{dst}/M0_FA{int(self.arguments.FA2):02d}.nii.gz"
        M0_avg_path = f"{dst}/M0_avg.nii.gz"
        avg_M0_path = f"{dst}/avg_M0.nii.gz"
        
        nib.save(T2Star_gre1_nii, T2Star_gre1_path)
        nib.save(T2Star_gre2_nii, T2Star_gre2_path)
        nib.save(T2Star_avg_nii, T2Star_avg_path)
        nib.save(avg_T2Star_nii, avg_T2Star_path)
        
        nib.save(M0_gre1_nii, M0_gre1_path)
        nib.save(M0_gre2_nii, M0_gre2_path)
        nib.save(M0_avg_nii, M0_avg_path)
        nib.save(avg_M0_nii, avg_M0_path)
        
        
        
    def linear_fit(self, echotimes, timeseries):
        # just format and add one column initialized at 1
        X_mat=np.vstack((np.ones(len(echotimes)), echotimes)).T
        
        # cf formula : linear-regression-using-matrix-multiplication 
        tmp = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)
        M0 = tmp.dot(timeseries.T)[0] # 0 => intercept_
        t2s = tmp.dot(timeseries.T)[1]# 1 => coef_
        
        return M0, t2s  
        
    
    def masking(self):
        
        fsl_brain_masking(self.arguments.gre1_path)
        fsl_brain_masking(self.arguments.gre2_path)
        
        filename1 = os.path.basename(self.arguments.gre1_path)
        if filename1[-3:] == '.gz':
            filename1 = filename1[:-7]
        else:
            filename1 = filename1[:-4]
            
        filename2 = os.path.basename(self.arguments.gre1_path)
        if filename2[-3:] == '.gz':
            filename2 = filename2[:-7]
        else:
            filename2 = filename2[:-4]
            
        dst1 = os.path.split(self.arguments.gre1_path)[0]
        dst2 = os.path.split(self.arguments.gre2_path)[0]

        mask_path1 = dst1 + '/' + filename1 + '_brain_mask.nii.gz'
        mask_path2 = dst2 + '/' + filename2 + '_brain_mask.nii.gz'
        
        mask1 = nib.load(mask_path1).get_fdata();
        mask2 = nib.load(mask_path2).get_fdata();
        
        return mask1, mask2
        


def T2star_fit(self, echotimes, timeseries):
    # just format and add one column initialized at 1
    X_mat=np.vstack((np.ones(len(echotimes)), echotimes)).T
    
    # cf formula : linear-regression-using-matrix-multiplication 
    tmp = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)
    M0 = tmp.dot(timeseries.T)[0] # 0 => intercept_
    t2s = tmp.dot(timeseries.T)[1]# 1 => coef_
    
    return M0, t2s


