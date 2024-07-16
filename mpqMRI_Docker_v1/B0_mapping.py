# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:13:33 2023

@author: denni
"""

from Useful_functions import fsl_brain_masking
from Useful_functions import fsl_flirt_registration
from Useful_functions import fsl_flirt_applyxfm
from Useful_functions import threshold_masking
from skimage.restoration import unwrap_phase
from scipy.ndimage import center_of_mass
import os
import numpy as np
import nibabel as nib


class B0_mapping():
    
    def __init__(self, arguments):
        
        """
        Initialise the class by providing a single "arguments" dict which 
        contains the following parameters:
        
        """
        
        self.arguments = arguments
        self.s1 = arguments.gre_list[0]
        self.s2 = arguments.gre_list[1]
        
    
    def run(self, gre='gre2', coregister_2_EPI=True):
        
        if gre=='gre2':
        
            phase_image = self.arguments.phase_image2;
            
            mask = self.brain_masking()
            pha = nib.load(phase_image).get_fdata()
            
            pha_rescaled = pha * mask[:,:,:,None]
            pha_rescaled = pha * np.pi/4096

            # unwrap phase
            com = center_of_mass(mask)
            com = np.round(com)

            com = np.asarray(com, dtype=int)

            phase_unwr = np.zeros(pha_rescaled.shape)

            for i in range(phase_unwr.shape[-1]):

                unwr_phase = unwrap_phase(pha_rescaled[:,:,:,i])
                #unwr_phase = unwr_phase/unwr_phase.max() * np.pi
                phase_unwr[:,:,:,i] = unwr_phase
                

            temporal = pha_rescaled[com[0],com[1],com[2],:]
            unwr_temporal = unwrap_phase(temporal)
            unwr_final = phase_unwr- phase_unwr[com[0],com[1],com[2],:] + unwr_temporal
            
            dst = os.path.split(self.arguments.gre2_path)[0]
            
            Unwrapped_nii = nib.Nifti1Image(unwr_final, affine = self.arguments.affine)
            
            name = 'Unwrapped_phase.nii'
            Unwrapped_path = dst + '/' + name 
            nib.save(Unwrapped_nii, Unwrapped_path)
            
            unwr_final_mod = np.reshape(unwr_final, (np.size(unwr_final)//np.shape(unwr_final)[-1], 
                                         np.shape(unwr_final)[-1]))
            
            echoTimes = self.arguments.echotimes
            
            intercerpt, slope = self.linear_fit(echoTimes, unwr_final_mod)

            slope_reshaped = np.reshape(slope, np.shape(unwr_final)[:-1])
            slope_reshaped = slope_reshaped * 1000

            slope_reshaped = slope_reshaped * mask
            
            dst = os.path.split(self.arguments.gre2_path)[0]
            
            B0_nii = nib.Nifti1Image(slope_reshaped, affine = self.arguments.affine)
            
            name = 'B0.nii'
            B0_path = dst + '/' + name 
            nib.save(B0_nii, B0_path)

        if coregister_2_EPI:
            B0map_path = self.coregister_B0_map_EPI()
            
        return B0map_path
    
    
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
            
            
        return mask
    
    
    def linear_fit(self, echotimes, timeseries):
        # just format and add one column initialized at 1
        X_mat=np.vstack((np.ones(len(echotimes)), echotimes)).T
        
        # cf formula : linear-regression-using-matrix-multiplication 
        tmp = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T)
        M0 = tmp.dot(timeseries.T)[0] # 0 => intercept_
        t2s = tmp.dot(timeseries.T)[1]# 1 => coef_
        
        return M0, t2s  
    
    
    def coregister_B0_map_EPI(self):
        
        moving_nii = self.arguments.gre2_path
        fixed_nii = self.arguments.path_epi_90
        matrix_filename = fsl_flirt_registration(moving_nii, fixed_nii, dof=6)
        dst = os.path.split(moving_nii)[0]
        
        matrix_path = dst + '/' + matrix_filename
        moving_nii = dst + '/' + 'B0' + '.nii'
        
        B0map_coreg_name = fsl_flirt_applyxfm(moving_nii, fixed_nii, matrix_path)
        
        B0map_path = dst + '/' + B0map_coreg_name + '.nii.gz'
        
        return B0map_path
    
    
    