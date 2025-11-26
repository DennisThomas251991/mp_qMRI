# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:12:35 2023

Author:
    Dennis C. Thomas
    Institute of Neuroradiology University Hospital Frankfurt
    
Date of completion of version 1 of 'Frankfurt_qMRI' python package:
    05/06/2024

"""
from Useful_functions import fsl_fastseg_B1mapping 
from Useful_functions import fsl_flirt_registration
from Useful_functions import fsl_flirt_applyxfm
from Useful_functions import threshold_masking
from Useful_functions import fsl_brain_masking
from Useful_functions import fsl_EPI_distortion_corr
import nibabel as nib
import numpy as np
import os
import sys
import B1mapping_Ralf_Phantom
import B1mapping_Ralf
from B1mapping_EPI_Fast import B1mapping_EPI_fast
class B1_map_Ralf():
    

    def __init__(self, arguments):
        
        
        self.arguments = arguments
        self.path_no_magneticprep = arguments.path_b1_noprep
        self.path_magneticprep = arguments.path_b1_prep
    
        
    def produce_B1_maps(self, coregister_2_T1 = True):
        
        if self.arguments.b1plus_mapping:
            
            print("Running B1 mapping")
            B1map_coreg = self.run(coregister_2_T1);
            self.arguments.b1plus = B1map_coreg
                 
        
        else:
            print("Creating B1 map with ones")
            gre_nii = nib.load(self.arguments.gre2_path).get_fdata();
            shape = np.shape(gre_nii)[:-1]
            B1map_coreg = np.ones(shape)
            self.arguments.b1plus = B1map_coreg
            
        return B1map_coreg
            
    
    def run(self, coregister_2_T1 = True):
        cwd = os.getcwd();
       
        if self.arguments.phantom:
            
            coregister_2_T1 = True
            
            print("Generating mask for phantom")
            mask = threshold_masking(self.arguments.path_b1_noprep, 
                                     self.arguments.Phantom_mask_threshold_B1mapping);
            
            self.mask = mask
            
            dst = os.path.split(self.path_no_magneticprep)[0]
            filename_b11 = os.path.basename(self.path_no_magneticprep)
            if filename_b11[-3:] == '.gz':
                filename_b11 = filename_b11[:-7]
            else:
                filename_b11 = filename_b11[:-4]
            
            print("Running B1 mapping for phantom")    
            parent_directory = os.path.dirname(cwd)
            path1 = os.path.abspath(parent_directory);
            path2 = os.path.abspath(os.path.join(parent_directory, "Ralf"));
            sys.path.append(path1);
            sys.path.append(path2)  ;
            path = dst
            no_magneticprep = os.path.basename(self.path_no_magneticprep)
            magneticprep = os.path.basename(self.path_magneticprep)
            basefilename = filename_b11
            B1map = B1mapping_Ralf_Phantom.B1mapping_Ralf_Phantom(path, no_magneticprep, magneticprep,basefilename, nargout=1);
            B1map = np.asarray(B1map)
    
            if coregister_2_T1:
                B1map = self.coregister_B1_map_T1_image()
            
        else:
            
            
            fsl_brain_masking(self.path_no_magneticprep);
            filename_b11 = os.path.basename(self.path_no_magneticprep)
            if filename_b11[-3:] == '.gz':
                filename_b11 = filename_b11[:-7]
            else:
                filename_b11 = filename_b11[:-4]
    
    
            filename_b12 = os.path.basename(self.path_magneticprep)
            if filename_b12[-3:] == '.gz':
                filename_b12 = filename_b12[:-7]
            else:
                filename_b12 = filename_b12[:-4]
    
    
            dst = os.path.split(self.path_no_magneticprep)[0]
            masked_brain = dst + '/' + filename_b11 + '_brain.nii.gz'
            fsl_fastseg_B1mapping(masked_brain);
    
            parent_directory = os.path.dirname(cwd)
            path1 = os.path.abspath(parent_directory);
            path2 = os.path.abspath(os.path.join(parent_directory, "Ralf"));
            sys.path.append(path1);
            sys.path.append(path2)  ;
            path = dst
            no_magneticprep = os.path.basename(self.path_no_magneticprep)
            magneticprep = os.path.basename(self.path_magneticprep)
            basefilename = filename_b11
            
            B1map = B1mapping_Ralf.B1mapping_Ralf(path, no_magneticprep, magneticprep, basefilename)
            B1map = np.asarray(B1map)
            
            if coregister_2_T1:
                B1map = self.coregister_B1_map_T1_image()
            
        return B1map
    
    
    
    def coregister_B1_map_T1_image(self):
        
        moving_nii = self.arguments.path_b1_noprep
        fixed_nii = self.arguments.gre2_path
        matrix_filename = fsl_flirt_registration(moving_nii, fixed_nii, dof=6)
        dst = os.path.split(moving_nii)[0]
        
        matrix_path = dst + '/' + matrix_filename
        moving_nii = dst + '/' + self.arguments.b1map_filename + '.nii'
        
        B1map_coreg_name = fsl_flirt_applyxfm(moving_nii, fixed_nii, matrix_path)
        
        B1map_path = dst + '/' + B1map_coreg_name + '.nii.gz'
        B1map = nib.load(B1map_path).get_fdata()
        
        return B1map
class B1_map_Dennis():
    
    def __init__(self, arguments):
        
        
        self.arguments = arguments
        self.path_EPI45 = arguments.path_epi_45
        self.path_EPI90 = arguments.path_epi_90
    
        
    def produce_B1_maps(self, B0_map_path, coregister_2_T1 = True):
        
        if self.arguments.b1plus_mapping:
            
            print("Running B1 mapping")
            B1map_coreg = self.run(B0_map_path, coregister_2_T1);
            self.arguments.b1plus = B1map_coreg
                 
        
        else:
            print("Creating B1 map with ones")
            gre_nii = nib.load(self.arguments.gre2_path).get_fdata();
            shape = np.shape(gre_nii)[:-1]
            B1map_coreg = np.ones(shape)
            self.arguments.b1plus = B1map_coreg
            
        return B1map_coreg
            
    
    def run(self, B0_map_path, coregister_2_T1 = True):
        """
        

        Parameters
        ----------
        B0_map_path : 3D image of dtype float
            DESCRIPTION.
        coregister_2_T1 : BOOL, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        B1map : TYPE
            DESCRIPTION.

        """

        cwd = os.getcwd();
        if self.arguments.phantom:
           
            filename_b11 = os.path.basename(self.path_EPI45)
            if filename_b11[-3:] == '.gz':
                filename_b11 = filename_b11[:-7]
            else:
                filename_b11 = filename_b11[:-4]
    
    
            filename_b12 = os.path.basename(self.path_EPI90)
            if filename_b12[-3:] == '.gz':
                filename_b12 = filename_b12[:-7]
            else:
                filename_b12 = filename_b12[:-4]
            if self.arguments.B1map_orientation == 'Tra':
                # user overrides
                user_sign = getattr(self.arguments, 'sign_gradient', None)
                user_phase = getattr(self.arguments, 'phase_dir', None)
                if user_sign is not None:
                    sign_gradient = int(user_sign)
                else:
                    sign_gradient = 1
                if user_phase is not None:
                    phasedir = str(user_phase)
                else:
                    phasedir = 'y'

            elif self.arguments.B1map_orientation == 'Sag':
                user_sign = getattr(self.arguments, 'sign_gradient', None)
                user_phase = getattr(self.arguments, 'phase_dir', None)
                if user_sign is not None:
                    sign_gradient = int(user_sign)
                else:
                    sign_gradient = -1
                if user_phase is not None:
                    phasedir = str(user_phase)
                else:
                    phasedir = 'x-'
            dst = os.path.split(self.path_EPI45)[0]
            
            
            corr1 = fsl_EPI_distortion_corr(self.path_EPI45, B0_map_path)
            corr2 = fsl_EPI_distortion_corr(self.path_EPI90, B0_map_path)
            
            
            parent_directory = os.path.dirname(cwd)
            path1 = os.path.abspath(parent_directory);
            path2 = os.path.abspath(os.path.join(parent_directory, "Ralf"));
            sys.path.append(path1);
            sys.path.append(path2);
            path = dst
            corr_45 = os.path.basename(corr1)
            corr_90 = os.path.basename(corr2)
            b0 = os.path.basename(B0_map_path)
            
            basefilename = filename_b11
            B1map = B1mapping_EPI_fast(path, b0, corr_45, corr_90,sign_gradient,self.arguments.Vendor);
            
            B1map = np.asarray(B1map)
            
            if coregister_2_T1:
                B1map = self.coregister_B1_map_T1_image()
            
            
        else:
            
            
            filename_b11 = os.path.basename(self.path_EPI45)
            if filename_b11[-3:] == '.gz':
                filename_b11 = filename_b11[:-7]
            else:
                filename_b11 = filename_b11[:-4]
    
    
            filename_b12 = os.path.basename(self.path_EPI90)
            if filename_b12[-3:] == '.gz':
                filename_b12 = filename_b12[:-7]
            else:
                filename_b12 = filename_b12[:-4]
            # Determine user overrides
            user_sign = getattr(self.arguments, 'sign_gradient', None)
            user_phase = getattr(self.arguments, 'phase_dir', None)
            if user_sign is not None:
                sign_gradient = int(user_sign)
                if self.arguments.B1map_orientation == 'Tra':
                    if self.arguments.Vendor == 'Siemens':
                        phasedir = 'y-'
                    else:
                        phasedir = 'y'
                elif self.arguments.B1map_orientation == 'Sag':
                    phasedir = 'x-'
                # if user provided a phase_dir, override phasedir
                if user_phase is not None:
                    phasedir = str(user_phase)
            else:
                if self.arguments.B1map_orientation == 'Tra':
                    sign_gradient = 1
                    if self.arguments.Vendor == 'Siemens':
                        phasedir = 'y-'
                    else:
                        phasedir = 'y'
                elif self.arguments.B1map_orientation == 'Sag':
                    phasedir = 'x-'
                    sign_gradient = 1
                # user can still override phase_dir even if sign_gradient not set
                if user_phase is not None:
                    phasedir = str(user_phase)
      
            dst = os.path.split(self.path_EPI45)[0]
            
            
            corr1 = fsl_EPI_distortion_corr(self.path_EPI45, B0_map_path, phasedir=phasedir)
            corr2 = fsl_EPI_distortion_corr(self.path_EPI90, B0_map_path, phasedir=phasedir)
            
            # MATLAB processing
            parent_directory = os.path.dirname(cwd)
            path1 = os.path.abspath(parent_directory);
            path2 = os.path.abspath(os.path.join(parent_directory, "Ralf"));
            sys.path.append(path1);
            sys.path.append(path2);
            path = dst
            corr_45 = os.path.basename(corr1)
            corr_90 = os.path.basename(corr2)
            b0 = os.path.basename(B0_map_path)
            
            basefilename = filename_b11
            
            B1map = B1mapping_EPI_fast(path, b0, corr_45, corr_90,sign_gradient,self.arguments.Vendor);
            
            B1map = np.asarray(B1map)
            
            if coregister_2_T1:
                B1map = self.coregister_B1_map_T1_image()
            
        return B1map
    
    
    def coregister_B1_map_T1_image(self):
        """
        
        Returns
        -------
        B1map : TYPE
            DESCRIPTION.

        """
        
        moving_nii = self.arguments.path_epi_90
        fixed_nii = self.arguments.gre2_path
        #additional_args= '-searchrx -20 20 -searchry -20 20 -searchrz -20 20'
        additional_args= '-applyxfm -usesqform'
        matrix_filename = fsl_flirt_registration(moving_nii, fixed_nii,additional_args, dof=6)
        dst = os.path.split(moving_nii)[0]
        
        matrix_path = dst + '/' + matrix_filename
        moving_nii = dst + '/' + self.arguments.b1map_filename + '.nii.gz'
        
        B1map_coreg_name = fsl_flirt_applyxfm(moving_nii, fixed_nii, matrix_path)
        
        B1map_path = dst + '/' + B1map_coreg_name + '.nii.gz'
        B1map = nib.load(B1map_path).get_fdata()
        
        return B1map