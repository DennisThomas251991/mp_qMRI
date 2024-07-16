# -*- coding: utf-8 -*-
"""
Author:
    Dennis C. Thomas
    Institute of Neuroradiology University Hospital Frankfurt
    
Date of completion of version 1 of 'Frankfurt_qMRI' python package:
    05/06/2024
    
"""
import T2smapping
import T1mapping_mpqMRI
import H2Omapping_mpqMRI
import B1mapping
import B0_mapping
import QSM_mapping
from Useful_functions import complex_interpolation
import nibabel as nib
import numpy as np
import os

#####Saving the Parameters#####################################################
def write_arguments_to_log(arguments):
        cwd=os.getcwd()
        log_file_path = os.path.join(cwd, "info.txt")  # Create path for the log file

        with open(log_file_path, "w") as log_file:
            log_file.write("Arguments set for running the script:\n\n")
            for key, value in arguments.items():
                log_file.write("{}: {}\n".format(key, value))
                
# Define arguments
arguments = dict()
arguments['gre1_path'] = r'C:\Users\mmari\Desktop\work\BMX_4_1\M0w_mag.nii.gz'
arguments['phase_image1'] = r'C:\Users\mmari\Desktop\work\BMX_4_1\M0w_pha.nii.gz'
arguments['gre2_path'] = r'C:\Users\mmari\Desktop\work\BMX_4_1\T1w_mag.nii.gz'
arguments['phase_image2'] =r'C:\Users\mmari\Desktop\work\BMX_4_1\T1w_pha.nii.gz'
arguments['path_b1_noprep'] = r'C:\Users\mmari\Desktop\work\BMX_4_1\B1_1.nii.gz'
arguments['path_b1_prep'] = r'C:\Users\mmari\Desktop\work\BMX_4_1\B1_2.nii.gz'
arguments['path_epi_45'] =r'C:\Users\mmari\Desktop\work\BMX_4_1\B1_1.nii.gz'
arguments['path_epi_90'] =r'C:\Users\mmari\Desktop\work\BMX_4_1\B1_2.nii.gz'
arguments['FA1'] = 5 # Flip angle 1
arguments['FA2'] = 32 # Flip angle 2
arguments['TR1'] = 30 # TR1 [ms] 
arguments['TR2'] = 30 # TR2 [ms]
arguments['nTE'] = 2  
arguments['x0'] = 1000

arguments['Bias_field_corr_method']='FSL' #choices are 'SPM' and 'FSL'
arguments['phantom'] = False
arguments['masking'] = True
arguments['b1plus_mapping'] = True
arguments['corr_for_imperfect_spoiling'] = True
arguments['Phantom_mask_threshold_B1mapping'] = 100
arguments['Phantom_mask_threshold_T1mapping'] = 100
arguments['spoil_increment'] = 50
arguments['slices'] = all
arguments['echotimes'] = np.linspace(3.2, 25.70, num=6)
arguments['QSM_average_echoes_qsm'] = [3,4,5,6]
arguments['coregister_mGREs'] = False
arguments['complex_interpolation'] = False
arguments['B1map_orientation']='Sag'
arguments['B1 mapping method'] = 'Fast EPI' # TWO OPTIONS: 'Fast EPI' and 'Volz 2010'
if arguments['B1 mapping method']=='Fast EPI':
    arguments['b1map_filename'] = 'B1_MAP_from_fast_EPI_standard' #Choose this for the fast EPI method

elif arguments['B1 mapping method']=='Volz 2010':
    arguments['b1map_filename'] = 'standard_B1_MAP_smoothed'



###############################################################################

"Preprocessing"

FA1 = np.deg2rad(arguments['FA1'])
FA2 = np.deg2rad(arguments['FA2'])
arguments['FA_list'] = [FA1, FA2]
gre1 = nib.load(arguments['gre1_path']).get_fdata()
gre2 = nib.load(arguments['gre2_path']).get_fdata()
arguments['affine'] = nib.load(arguments['gre2_path']).affine
arguments['b1plus'] = np.ones(gre1.shape[:-1])
arguments['gre_list'] = [gre1, gre2]
arguments['TR_list'] = [arguments['TR1'], arguments['TR1']] 


if arguments['complex_interpolation']:
   
    mag_path = arguments['gre1_path']
    pha_path = arguments['phase_image1']
    save_name = "resampled_1mm_iso_FA%i"%arguments['FA1']
    mag_interp_path, pha_interp_path = complex_interpolation(mag_path, pha_path, save_name=save_name)
    arguments['gre1_path'] = mag_interp_path
    arguments['phase_image1'] = pha_interp_path
   
    mag_path = arguments['gre2_path']
    pha_path = arguments['phase_image2']
    save_name = "resampled_1mm_iso_FA%i"%arguments['FA2']
    mag_interp_path, pha_interp_path = complex_interpolation(mag_path, pha_path, save_name=save_name)
    arguments['gre2_path'] = mag_interp_path
    arguments['phase_image2'] = pha_interp_path
    gre1 = nib.load(arguments['gre1_path']).get_fdata()
    gre2 = nib.load(arguments['gre2_path']).get_fdata()
    arguments['gre_list'] = [gre1, gre2]
    arguments['affine'] = nib.load(arguments['gre2_path']).affine

###############################################################################
write_arguments_to_log(arguments)
if arguments['B1 mapping method']=='Fast EPI':
    
    # B0 mapping
    B0map_object = B0_mapping.B0_mapping(arguments)
    B0map_coreg2B1_path = B0map_object.run(coregister_2_EPI=True)
    
    # B1 mapping
    B1map_object = B1mapping.B1_map_Dennis(arguments)
    B1map_coreg = B1map_object.produce_B1_maps(B0map_coreg2B1_path, coregister_2_T1=True)
    

elif arguments['B1 mapping method']=='Volz 2010':
    
    # B1 mapping
    B1map_object = B1mapping.B1_map_Ralf(arguments)
    B1map_coreg = B1map_object.produce_B1_maps(coregister_2_T1=True)

    

#T1 mapping
T1_object = T1mapping_mpqMRI.T1_mapping_mpqMRI(arguments, B1map_coreg)
T1_map = T1_object.run(save = True)
T1_map2 = T1_object.run_linear_approach(save = True)
#T1_map=nib.load('C:/Users/mmari/Desktop/work/BMX_7_2/T1_map_B1corr_True_Spoilcorr_True_2echoes.nii').get_fdata()
#T2s mapping
T2s_object = T2smapping.t2s_map_mpqMRI(arguments)
output = T2s_object.run(nechoes = 6, save = True)
T2Star_gre12 = output[7]
# B1map_coreg=nib.load('C:/Users/mmari/Desktop/work/BMX_7_2/B1_MAP_from_fast_EPI_standard_2_mGRE_FA_32__14.nii.gz').get_fdata()
# T2Star_gre12=nib.load('C:/Users/mmari/Desktop/work/BMX_7_2/avg_T2Star.nii').get_fdata()
# H2O mapping

H2O_object = H2Omapping_mpqMRI.H2O_map_mpqMRI(arguments, T1_map, B1map_coreg, T2Star_gre12)
H2O = H2O_object.run()

#QSM mapping
QSM_object = QSM_mapping.QSM_mapping_mpqMRI(arguments)
QSM = QSM_object.run(gre='gre2')

