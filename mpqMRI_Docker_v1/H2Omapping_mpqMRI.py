# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 20:01:04 2023

@author: denni
"""
import gzip
import shutil
import nibabel as nib
import numpy as np
import os
from Useful_functions import fsl_fastseg
from Useful_functions import fsl_brain_masking
from Useful_functions import SPM_segment

class H2O_map_mpqMRI():
    
    def __init__(self, arguments, T1_map, B1_map, T2star_map):
        
        self.arguments = arguments
        self.T1_map = T1_map/1000 # convert to seconds
        self.B1_map = B1_map     
        self.T2star_map = T2star_map 
        self.TR = self.arguments.TR1/1000 # convert to seconds
        self.FA = np.deg2rad(self.arguments.FA1)
        self.affine = nib.load(self.arguments.gre1_path).affine
        self.gre1 = nib.load(self.arguments.gre1_path).get_fdata()[:,:,:,0]
        
        filename2 = os.path.basename(self.arguments.gre2_path)
        if filename2[-3:] == '.gz':
            filename2 = filename2[:-7]
        else:
            filename2 = filename2[:-4]
        
        filename1 = os.path.basename(self.arguments.gre1_path)
        if filename1[-3:] == '.gz':
            filename1 = filename1[:-7]
        else:
            filename1 = filename1[:-4]
            
        dst1 = os.path.split(self.arguments.gre1_path)[0]
        dst2 = os.path.split(self.arguments.gre2_path)[0]
        
        
        self.filename2 = filename2
        self.dst2 = dst2
        self.filename1 = filename1
        self.dst1 = dst1
    
    
    def run(self):
        # T1 correction
        M0_T1corr_path = self.T1_correction()
        M0_T1_biascorr = self.Bias_field_correction(M0_T1corr_path)
        M0_T1_biascorr[~np.isfinite(M0_T1_biascorr)] = 0   # Set NaN values to 0
        M0_T1_biascorr_nii = nib.Nifti1Image(M0_T1_biascorr, affine=self.affine)
        M0_T1_biascorr_path = os.path.join(self.dst1, 'M0_T1_biascorr.nii')
        nib.save(M0_T1_biascorr_nii, M0_T1_biascorr_path)
        
        M0_T1_T2s_biascorr = self.T2s_correction(M0_T1_biascorr)
        M0_T1_T2s_biascorr[~np.isfinite(M0_T1_T2s_biascorr)] = 0  # Set NaN values to 0
        M0_T1_T2s_biascorr_nii = nib.Nifti1Image(M0_T1_T2s_biascorr, affine=self.affine)
        M0_T1_T2s_biascorr_path = os.path.join(self.dst1, 'M0_T1_T2s_biascorr.nii')
        nib.save(M0_T1_T2s_biascorr_nii, M0_T1_T2s_biascorr_path)
        
        CSF_calib_factor, csf_mask_binary, M0_T1_biascorr_CSFT1mod = self.Calc_CSF_calibration_factor()
        CSF_calib = csf_mask_binary * CSF_calib_factor
        CSF_calib[~np.isfinite(CSF_calib)] = 0
        CSF_calib_nii = nib.Nifti1Image(CSF_calib, affine=self.affine)
        CSF_calib_path = os.path.join(self.dst1, 'CSF_calib.nii')
        nib.save(CSF_calib_nii, CSF_calib_path)
        
        H2O = self.CSF_calibration(M0_T1_T2s_biascorr)
        H2O[~np.isfinite(H2O)] = 0  # Set NaN values to 0
        H2O_nii = nib.Nifti1Image(H2O, affine=self.affine)
        H2O_path = os.path.join(self.dst1, 'H2O.nii')
        nib.save(H2O_nii, H2O_path)
        
        M0_T1_T2s_biascorr_CSFT1mod = self.T2s_correction_exceptCSF(M0_T1_biascorr_CSFT1mod, csf_mask_binary)
        M0_T1_T2s_biascorr_CSFT1mod[~np.isfinite(M0_T1_T2s_biascorr_CSFT1mod)] = 0  # Set NaN values to 0
        M0_T1_T2s_biascorr_CSFT1mod_nii = nib.Nifti1Image(M0_T1_T2s_biascorr_CSFT1mod, affine=self.affine)
        M0_T1_T2s_biascorr_path = os.path.join(self.dst1, 'M0_T1_T2s_biascorr_CSFT1mod.nii')
        nib.save(M0_T1_T2s_biascorr_CSFT1mod_nii, M0_T1_T2s_biascorr_path)

        H2O_CSFT1mod = self.CSF_calibration(M0_T1_T2s_biascorr_CSFT1mod)
        H2O_CSFT1mod[~np.isfinite(H2O_CSFT1mod)] = 0  # Set NaN values to 0
        H2O_CSFT1mod_nii = nib.Nifti1Image(H2O_CSFT1mod, affine=self.affine)
        H2O_path = os.path.join(self.dst1, 'H2O_CSFT1mod.nii')
        nib.save(H2O_CSFT1mod_nii, H2O_path)
        
        return H2O
    def T1_correction(self):
        
        TR = self.TR
        FA = self.FA
        T1 = self.T1_map
        B1_map = self.B1_map
        gre1 = self.gre1
        
    
        SF_map = np.sin(B1_map*FA)*(1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1_map*FA))
        M0_T1corr = gre1/SF_map
        
        brain_mask = self.masking()[0];
        
        M0_T1corr = M0_T1corr * brain_mask
        M0_T1corr[np.isnan(M0_T1corr)] = 0
        
        self.M0_T1corr = M0_T1corr
        
        M0_T1corr_nii = nib.Nifti1Image(M0_T1corr, affine = self.affine)
        M0_T1corr_path = self.dst1 + '/' + 'M0_T1corr.nii'
        nib.save(M0_T1corr_nii, M0_T1corr_path)
        
        return M0_T1corr_path
        
    def Bias_field_correction(self, M0_T1corr_path):
        
       bias_field = self.bias_field(M0_T1corr_path)
       if self.arguments.Bias_field_corr_method == 'SPM':
         M0_T1_biascorr = self.M0_T1corr*bias_field
       else:
         M0_T1_biascorr = self.M0_T1corr/bias_field
       
       self.M0_T1_biascorr = M0_T1_biascorr
       
       return M0_T1_biascorr

    def T2s_correction(self, M0_T1_biascorr):
       # T2* correction
       
       TE0 = self.arguments.echotimes[0]
       T2s_map = self.T2star_map
       corr = np.exp(-TE0/T2s_map)
       
       M0_T1_T2s_biascorr = M0_T1_biascorr/corr
       
       
       return M0_T1_T2s_biascorr
   
    
    def T2s_correction_exceptCSF(self, M0_T1_biascorr, mask):
       # T2* correction
       
       TE0 = self.arguments.echotimes[0]
       T2s_map = self.T2star_map
       corr = np.exp(-TE0/T2s_map)
       corr[mask==1] = 1
       
       M0_T1_T2s_biascorr = M0_T1_biascorr/corr
       
       return M0_T1_T2s_biascorr
        
    

    def Calc_CSF_calibration_factor(self):
        
        T1 = self.T1_map
        TR = self.TR
        FA = self.FA
        B1_map = self.B1_map
        gre1 = self.gre1
      
        if self.arguments.Bias_field_corr_method == 'SPM':
            masked_brain_path = self.arguments.gre2_path 
            file_extension = os.path.splitext(masked_brain_path)[1]

            if file_extension == '.gz':
                masked_brain_path2 = masked_brain_path
                masked_brain_extracted = os.path.splitext(masked_brain_path)[0]  # Remove .gz to get .nii
                # Extract the .nii file from the .nii.gz
                with gzip.open(masked_brain_path2, 'rb') as f_in:
                    with open(masked_brain_extracted, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                SPM_segment(masked_brain_extracted)
            elif file_extension == '.nii':
                masked_brain_path2 = masked_brain_path
                SPM_segment(masked_brain_path2)
            else:
                raise ValueError("Unsupported file format")

            CSF_mask = nib.load(self.dst2 + '/' + 'c3' + self.filename2 + '.nii').get_fdata()
            GM_mask = nib.load(self.dst2 + '/' + 'c2' + self.filename2 + '.nii').get_fdata()
            WM_mask = nib.load(self.dst2 + '/' + 'c1' + self.filename2 + '.nii').get_fdata()
        else:
            
            masked_brain_path2 = self.dst2 + '/' + self.filename2 + '_brain.nii.gz'
            fsl_fastseg(masked_brain_path2)
            CSF_mask = nib.load(self.dst2 + '/' + self.filename2 + '_brain_pve_0.nii.gz').get_fdata()
            GM_mask = nib.load(self.dst2 + '/' + self.filename2 + '_brain_pve_1.nii.gz').get_fdata()
            WM_mask = nib.load(self.dst2 + '/' + self.filename2 + '_brain_pve_2.nii.gz').get_fdata()


        WM_mask_binary = WM_mask>0.98
        WM_mask_binary[CSF_mask>0.80] = False 
        WM_mask_binary[GM_mask>0.80] = False 

        GM_mask_binary = GM_mask>0.98 
        GM_mask_binary[CSF_mask>0.80] = False 
        GM_mask_binary[WM_mask>0.80] = False 

        csf_mask_binary = np.zeros(CSF_mask.shape)
        csf_mask_binary[CSF_mask>0.95] = 1
        csf_mask_binary[WM_mask>0.95] = 0
        csf_mask_binary[GM_mask>0.95] = 0
        
        # New addition:
            
        T1[csf_mask_binary==1] = 4.3
        
        #Corr_spoiling = (0.200*(B1_map**2)) - (0.211*B1_map) + 0.959 According to publication, Volz. et al 2012
        
        Corr_spoiling = (0.112*(B1_map**2)) - (0.154*B1_map) + 1.005 #Calculated by Ralf Deichmann, BIC
    
        SF_map = np.sin(B1_map*FA)*(1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1_map*FA))
        
        SF_map = SF_map * Corr_spoiling
        
        M0_T1corr = gre1/SF_map
        if self.arguments.Bias_field_corr_method == 'SPM':
          M0_T1_biascorr = M0_T1corr*self.bias_field_map
        else:
          M0_T1_biascorr = M0_T1corr/self.bias_field_map

        # Define the threshold and parameters
        threshold = 0.95
        bins = 200
        interval = [4000, 9000]
        
        # Extract the relevant CSF data using the threshold
        CSF = M0_T1_biascorr[csf_mask_binary > threshold]
        
        # Calculate the bin width
        bin_width = (interval[1] - interval[0]) / bins
        
        # Create bin edges and calculate the bin indices for the CSF data
        bin_indices = np.floor((CSF - interval[0]) / bin_width).astype(int)
        
        # Filter out data that falls outside the interval
        valid_indices = (bin_indices >= 0) & (bin_indices < bins)
        filtered_indices = bin_indices[valid_indices]
        
        # Calculate the mode of the bin indices
        mode_bin_index = mode(filtered_indices).mode[0]
        
        # Calculate the center of the mode bin
        CSF_calib_factor = interval[0] + (mode_bin_index + 0.5) * bin_width
        
        M0_T1_biascorr_CSFT1mod = M0_T1_biascorr
        
        self.CSF_calib_factor = CSF_calib_factor
        
        return CSF_calib_factor, csf_mask_binary, M0_T1_biascorr_CSFT1mod
    
    
    def CSF_calibration(self, M0):
        
        
        H2O = M0 / self.CSF_calib_factor
        H2O = H2O*self.brain_mask
        
        return H2O

    
    def bias_field(self, path):
        
        if self.arguments.Bias_field_corr_method == 'SPM':
           M0_T1corr=nib.load(path).get_fdata()/10
           path_brain_mask=self.dst2 + '/' + self.filename2 + '_brain_mask.nii.gz'
        
           Brain_mask=nib.load(path_brain_mask).get_fdata()
           T1w=nib.load(self.arguments.gre2_path).get_fdata()
           T1w=T1w[:,:,:,0]
           T1w_mod=np.where(Brain_mask<1,T1w,M0_T1corr)
           T1w_mod[~np.isfinite(T1w_mod)]=0
           affine=self.affine
           nii=nib.Nifti1Image(T1w_mod, affine=affine)
           path_mod=self.dst2 + '/mod_M0_T1corr.nii'
           nib.save(nii,path_mod)

           SPM_segment(path_mod);
           filename = os.path.basename(path_mod)
           if filename[-3:] == '.gz':
             filename = filename[:-7]
           else:
             filename = filename[:-4]
            
           dst = os.path.split(path_mod)[0]
           bias_field_path = dst + '/BiasField_mod_M0_T1corr.nii'
        else:
           fsl_fastseg(path);
           filename = os.path.basename(path)
           if filename[-3:] == '.gz':
             filename = filename[:-7]
           else:
             filename = filename[:-4]
            
           dst = os.path.split(path)[0]
           bias_field_path = dst + '/' + filename + '_bias.nii.gz'


        bias_field = nib.load(bias_field_path).get_fdata();
        self.bias_field_map = bias_field
        
        return bias_field
    
    
    def masking(self):
        
        fsl_brain_masking(self.arguments.gre2_path)

        masked_brain_path1 = self.dst1 + '/' + self.filename2 + '_brain.nii.gz'
        brain_mask = nib.load(self.dst1 + '/' + self.filename2 + '_brain_mask.nii.gz').get_fdata()
        
        self.brain_mask = brain_mask
        
        return brain_mask, masked_brain_path1
    


    
        
        
        
        
        
