# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:20:48 2023

@author: denni
"""
from scipy import optimize as opt
import numpy as np
from datetime import datetime
from Useful_functions import fsl_brain_masking
from Useful_functions import fsl_flirt_registration
from Useful_functions import fsl_flirt_applyxfm
from Useful_functions import threshold_masking
from Useful_functions import split_all_echoes
from Useful_functions import combine_echoes
import os
import nibabel as nib

class T1_mapping_mpqMRI():
    
    """
    
    This class creates an object which makes it convenient to carry out
    T1 mapping using VFA method mGRE as carried out in Schall et. al, 
    PLoS One, 2018
    
    Parameters
    ----------
    gre_list : TYPE, list
        DESCRIPTION. A list of the two mGRE files acquired
    b1plus : TYPE, list
        DESCRIPTION. B1+ map coregistered to mGRE space
    FA_list : TYPE, list
        DESCRIPTION. A list of the two flip angles used
    TR_list : TYPE, list
        DESCRIPTION. A list of the two TRs used for the two mGREs
    nTE : TYPE, optional
        DESCRIPTION. Number of echoes to be used for the fitting. 
        The default is 5.
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    slices : TYPE, optional
        DESCRIPTION. The default is all.
    x0: TYPE, float
        DESCRIPTION: Initialisation of T1 value. The default is 1000
        
        
    Functions:
    ----------
    
    run(): Runs the T1 mapping

    """
    
    
    
    def __init__(self, arguments, B1map_coreg):
        
        """
        Initialise the class by providing a single "arguments" dict which 
        contains the following parameters:
        
        """
        
        self.arguments = arguments
        self.gre1 = arguments.gre_list[0]
        self.gre2 = arguments.gre_list[1]
        
        self.FA_list = arguments.FA_list
        
        self.b1plus = B1map_coreg
        self.TR_list = arguments.TR_list
        
        self.nTE = arguments.nTE
        self.slices = arguments.slices 
        
        self.shape = np.shape(arguments.b1plus)
        self.index = np.ones(np.shape(arguments.b1plus))
        
        self.initial_t1 = arguments.x0
        
        self.signal1 = self.gre2[..., 0:self.nTE] * np.sin(self.b1plus*self.FA_list[0])[..., None]
        self.signal2 = self.gre1[..., 0:self.nTE] * np.sin(self.b1plus*self.FA_list[1])[..., None]
        
        self.B1_corr_cos1 = np.cos(self.b1plus*self.FA_list[0])
        self.B1_corr_cos2 = np.cos(self.b1plus*self.FA_list[1])
        
        self.TR1 = arguments.TR_list[0]
        self.TR2 = arguments.TR_list[1]
        
             
        
    def run(self, save = True):
        
        
        if self.arguments.corr_for_imperfect_spoiling:
            
            FA1, FA2 = self.corr_for_imperf_spoiling();
            self.FA1 = FA1
            self.FA2 = FA2
            
            self.signal1 = self.gre2[..., 0:self.nTE] * np.sin(self.FA1)[..., None]
            self.signal2 = self.gre1[..., 0:self.nTE] * np.sin(self.FA2)[..., None]
            
            self.B1_corr_cos1 = np.cos(self.FA1)
            self.B1_corr_cos2 = np.cos(self.FA2)
            
            if save:
                
                dst = os.path.split(self.arguments.gre2_path)[0]
            
                FA1_nii = nib.Nifti1Image(FA1, affine = self.arguments.affine)
                
                name = 'FA1_map_B1corr_Spoilcorr.nii'
                
                FA1_path = dst + '/' + name 
                nib.save(FA1_nii, FA1_path)
                
                dst = os.path.split(self.arguments.gre2_path)[0]
                
                FA2_nii = nib.Nifti1Image(FA2, affine = self.arguments.affine)
                
                name = 'FA2_map_B1corr_Spoilcorr.nii'
                
                FA2_path = dst + '/' + name 
                nib.save(FA2_nii, FA2_path)
        
        if self.arguments.coregister_mGREs:
            
            self.coregistration_mGREs()
            
        self.T1 = np.zeros(self.shape)
        
        if self.slices is all:

            slice_range = range(np.shape(self.gre1)[2])

        else:

            slice_range = range(self.slices[0], self.slices[1])

        print(f"Start Time = {datetime.now().strftime('%H:%M:%S')}")

        if self.arguments.masking:
            
            self.brain_masking();
            
        
        for slice_idx in slice_range:
            
            self._process_slice(slice_idx)
            
            
        self.T1[~np.isfinite(self.T1)] = 0
        self.T1 = self.T1.clip(0, 10000)

        print(f"Stop Time = {datetime.now().strftime('%H:%M:%S')}")
        
        if save:
            dst = os.path.split(self.arguments.gre2_path)[0]
            
            T1_nii = nib.Nifti1Image(self.T1, affine = self.arguments.affine)
            
            name = 'T1_map_B1corr_%s_Spoilcorr_%s_%iechoes.nii'%(self.arguments.b1plus_mapping, 
                                                       self.arguments.corr_for_imperfect_spoiling,
                                                       self.arguments.nTE)
            
            T1_path = dst + '/' + name 
            nib.save(T1_nii, T1_path)
        return self.T1
    
    
    def cost_function(self, T1, TR1, TR2, B1_corr_cos1, B1_corr_cos2, signal1, signal2):
        

        term1 = signal1 * (1-np.exp(-TR1/T1)) / (1-(B1_corr_cos1*np.exp(-TR1/T1)))
        term2 = signal2 * (1-np.exp(-TR2/T1)) / (1-(B1_corr_cos2*np.exp(-TR2/T1)))

        return term1-term2

    
    def _process_slice(self, slice_idx):
        
        index_slice = self.mask[..., slice_idx]
        cos1_slice = self.B1_corr_cos1[..., slice_idx]
        cos2_slice = self.B1_corr_cos2[..., slice_idx]
        signal1_slice = self.signal1[..., slice_idx, :]
        signal2_slice = self.signal2[..., slice_idx, :]

        mask_slice = np.ones(index_slice.shape) if self.mask is None else self.mask[..., slice_idx]
        
        if self.slices is all:

            slice_range = range(np.shape(self.gre1)[2])

        else:

            slice_range = range(self.slices[0], self.slices[1])
            
        print('%s/%s' % (slice_idx, len(slice_range)))
        
        self.T1[..., slice_idx] = self._fit_t1_map(index_slice, cos1_slice, cos2_slice, signal1_slice, signal2_slice, mask_slice)
        

    def _fit_t1_map(self, index_slice, cos1_slice, cos2_slice, signal1_slice, signal2_slice, mask_slice):
        
        shape = np.shape(index_slice)
        flat_index = index_slice.ravel()
        flat_cos1 = cos1_slice.ravel()
        flat_cos2 = cos2_slice.ravel()
        flat_signal1 = signal1_slice.reshape(np.shape(flat_index) + (np.shape(signal1_slice)[-1],))
        flat_signal2 = signal2_slice.reshape(np.shape(flat_index) + (np.shape(signal2_slice)[-1],))
        flat_mask = mask_slice.ravel()
        
        t1_values = np.zeros(np.shape(flat_index))

        for INDEX in range(np.size(flat_index)):
            if flat_index[INDEX] and flat_mask[INDEX]:
                t1_values[INDEX] = opt.leastsq(self.cost_function, self.initial_t1,
                                             args=(self.TR1, self.TR2, flat_cos1[INDEX], flat_cos2[INDEX], flat_signal1[INDEX, :], flat_signal2[INDEX, :]))[0]

        return t1_values.reshape(shape).clip(0, 10000)
    
    
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
        
    
    def corr_for_imperf_spoiling(self):
        
        spoilinc = self.arguments.spoil_increment
        
        if self.arguments.corr_for_imperfect_spoiling:
            if spoilinc==50:
                 polyorder=5;
                 P=np.zeros([polyorder+1,polyorder+1]);
                 P[0,:]=[ 9.639e-1,  4.989e-3, -1.254e-4, -3.180e-6,  1.527e-7, -1.462e-9];
                 P[1,:]=[ 5.880e-3, -1.056e-3,  4.801e-5, -8.549e-7,  5.382e-9,      0  ];
                 P[2,:]=[ 4.143e-4, -4.920e-6, -1.560e-7,  2.282e-9,      0,         0  ];
                 P[3,:]=[-1.059e-5,  2.334e-7, -1.189e-9,      0,         0,         0  ];
                 P[4,:]=[ 9.449e-8, -1.025e-9,     0,          0,         0,         0  ];
                 P[5,:]=[-4.255e-10,      0,        0,          0,         0,         0 ];
                
            elif spoilinc==117:
                 polyorder=5;
                 P=np.zeros([polyorder+1,polyorder+1]);
                 P[0,:]=[ 9.381e-1,  4.266e-3,  2.535e-4, -2.289e-5,  5.402e-7, -4.146e-9 ];
                 P[1,:]=[ 1.653e-2, -2.172e-3,  7.491e-5, -1.051e-6,  5.331e-9,      0  ];
                 P[2,:]=[ 3.145e-4,  3.704e-5, -1.123e-6,  8.369e-9,      0,         0  ]; 
                 P[3,:]=[-3.848e-5,  2.773e-7,  1.662e-9,      0,         0,         0  ];
                 P[4,:]=[ 6.230e-7, -4.019e-9,     0,          0,         0,         0  ];
                 P[5,:]=[-2.988e-9,      0,        0,          0,         0,         0  ];
                
            elif spoilinc==150:
                 polyorder=5;
                 P=np.zeros([polyorder+1,polyorder+1]);
                 P[0,:]=[ 6.678e-1,  9.131e-2, -7.728e-3,  2.863e-4, -4.869e-6,  3.112e-8 ];
                 P[1,:]=[-3.710e-2,  2.845e-3, -7.786e-5,  8.546e-7, -2.837e-9,      0  ];
                 P[2,:]=[ 1.448e-3, -7.537e-5,  1.403e-6, -8.865e-9,      0,         0  ];
                 P[3,:]=[-2.181e-5,  6.141e-7, -5.141e-9,      0,         0,         0  ];
                 P[4,:]=[ 1.990e-7, -1.978e-9,     0,          0,         0,         0  ];
                 P[5,:]=[-8.617e-10,     0,        0,          0,         0,         0  ];
            
            else:
              raise Exception('spoilinc must be 50 or 117 or 150')
             
            
            
            FA1 = np.rad2deg(self.b1plus*self.FA_list[0])
            FA2 = np.rad2deg(self.b1plus*self.FA_list[1])
        
            corr1 = np.zeros(FA1.shape)
            for k in range(polyorder+1):
                for l in range(polyorder+1):
                    corr1 = corr1 + (P[k,l]*(FA1**(k))*(self.arguments.TR1**(l)));
                    

            corr2 = np.zeros(FA2.shape)
            for k in range(polyorder+1):
                for l in range(polyorder+1):
                    corr2 = corr2 + (P[k,l]*(FA2**(k))*(self.arguments.TR2**(l)));
            
            
            FA1 = FA1*corr1*np.pi/180
            FA2 = FA2*corr2*np.pi/180
            
            return FA1, FA2 #in radians
            

    def coregistration_mGREs(self):
        
        if self.arguments.coregister_mGREs:
            
            moving_nii = self.arguments.gre1_path
            fixed_nii = self.arguments.gre2_path
            matrix_file = fsl_flirt_registration(moving_nii, fixed_nii, dof=6)
            
            filename1 = os.path.basename(self.arguments.gre1_path)
            if filename1[-3:] == '.gz':
                filename1 = filename1[:-7]
            else:
                filename1 = filename1[:-4]
                
            filename2 = os.path.basename(self.arguments.gre2_path)
            if filename2[-3:] == '.gz':
                filename2 = filename2[:-7]
            else:
                filename2 = filename2[:-4]
            
            dst = os.path.split(self.arguments.gre2_path)[0]
            
            split_all_echoes(self.arguments.gre1_path)
            
            echoes_dict = dict()
            
            matrix_file = dst + '/' + matrix_file
            for i in range(np.shape(self.s1)[-1]):
                
                moving_nii = dst + '/' + filename1 + '_%i_echo'%(i+1) + '.nii.gz'
                fixed_nii = self.arguments.gre2_path
                out_path = fsl_flirt_applyxfm(moving_nii, fixed_nii, matrix_file)
                
                
                echoes_dict['se%i_path'%(i+1)] = dst + '/' + out_path + '.nii.gz'
            
            self.arguments.gre1_path = combine_echoes(arguments = echoes_dict, nechoes = np.shape(self.gre1)[-1])
                
        
            
    def run_linear_approach(self, save=True):
        
        if self.arguments.corr_for_imperfect_spoiling:
            
            FA1, FA2 = self.corr_for_imperf_spoiling();
            self.FA1 = FA1
            self.FA2 = FA2
            
            self.signal1 = self.gre2[..., 0:self.nTE] * np.sin(self.FA1)[..., None]
            self.signal2 = self.gre1[..., 0:self.nTE] * np.sin(self.FA2)[..., None]
            
            self.B1_corr_cos1 = np.cos(self.FA1)
            self.B1_corr_cos2 = np.cos(self.FA2)
            
            if save:
                
                dst = os.path.split(self.arguments.gre2_path)[0]
            
                FA1_nii = nib.Nifti1Image(FA1, affine = self.arguments.affine)
                
                name = 'FA1_map_B1corr_Spoilcorr.nii'
                
                FA1_path = dst + '/' + name 
                nib.save(FA1_nii, FA1_path)
                
                
                dst = os.path.split(self.arguments.gre2_path)[0]
            
                FA2_nii = nib.Nifti1Image(FA2, affine = self.arguments.affine)
                
                name = 'FA2_map_B1corr_Spoilcorr.nii'
                
                FA2_path = dst + '/' + name 
                nib.save(FA2_nii, FA2_path)
        
        if self.arguments.coregister_mGREs:
            
            self.coregistration_mGREs()
        
        if self.arguments.masking:
            
            self.brain_masking();
            
        
        gre1 = self.gre1[:,:,:,0]
        gre2 = self.gre2[:,:,:,0]
        
        if self.mask is not None:
    
            mask = self.mask
        
        else:
            mask = np.ones(gre1.shape)
        
        x1 = gre1/np.tan(self.FA1)
        x2 = gre2/np.tan(self.FA2)
        y1 = gre1/np.sin(self.FA1)
        y2 = gre2/np.sin(self.FA2)
        
        slope = ((y2-y1)/(x2-x1))*mask
        
        slope[~np.isfinite(slope)] = 0
        
        slope[np.isnan(slope)] = 0
        
        t1map = (-1*self.TR1)*mask/np.log(slope)
        
        t1map[~np.isfinite(t1map)] = 0
        t1map = t1map.clip(0, 10000)
        
        
        if save:
            dst = os.path.split(self.arguments.gre2_path)[0]
            
            slope_nii = nib.Nifti1Image(slope, affine = self.arguments.affine)
            T1_nii = nib.Nifti1Image(t1map, affine = self.arguments.affine)
            
            name = 't1map_linear_approach'
            
            T1_path = dst + '/' + name 
            slope_path = dst + '/' + 'slope'
            nib.save(T1_nii, T1_path)
            nib.save(slope_nii, slope_path)
        
        return t1map 
        
            
            
            
            
            

            
            
            
    
        
