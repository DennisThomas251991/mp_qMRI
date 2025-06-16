# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:02:04 2023

Author:
    Dennis C. Thomas
    Institute of Neuroradiology University Hospital Frankfurt
    
Date of completion of version 1 of 'Frankfurt_qMRI' python package:
    05/06/2024
    
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from Dennis.Useful_functions import fsl_NON_linear_normalization_AFTER_FLIRT
from Dennis.Useful_functions import fsl_fastseg


datasets = ['Volunteer10_SECOND_Prisma_07_08_2023', 'Volunteer10_SECOND_Skyra_11_08_2023',
            'Volunteer11_Prisma_07_07_2023', 'Volunteer11_Skyra_25_07_2023',
            'Volunteer12_Prisma_11_07_2023', 'Volunteer12_Skyra_27_07_2023',
            'Volunteer13_Prisma_26_07_2023', 'Volunteer13_Skyra_31_07_2023',
            'Volunteer15_Prisma_04_08_2023', 'Volunteer15_Skyra_04_08_2023']

for data in datasets:
    
    
    path = 'Z:/NR-qCET/analysis/Volunteers/%s/Final_mp_qMRI_analysis/New_H2O_processing'%data
                             
    os.chdir(path)
    
    M0_T1corr = nib.load(path + '/M0_T1corr.nii').get_fdata()
    T2s_map = nib.load(path + '/T2Star_avg.nii').get_fdata()
    gre1 = nib.load(path + '/resampled_1mm_iso_FA05_mag_brain.nii.gz').get_fdata()
    CSF_mask = nib.load(path + '/c3resampled_1mm_iso_FA25_mag.nii').get_fdata()
    GM_mask =  nib.load(path + '/c1resampled_1mm_iso_FA25_mag.nii').get_fdata()
    WM_mask =  nib.load(path + '/c2resampled_1mm_iso_FA25_mag.nii').get_fdata()
    
    
    
    
    
    
    
    
    Biasfield_spm = nib.load(path + '/BiasField_M0_T1corr.nii').get_data()
    
    B1_map = nib.load(path + '/B1_MAP_from_fast_EPI_standard_2_resampled_1mm_iso_FA25_mag.nii.gz').get_fdata()
        
    affine = nib.load(path + '/resampled_1mm_iso_FA05_mag_brain.nii.gz').affine
    
    FA = np.deg2rad(5)
    TR = 30/1000    
    
    # Bias field correction
    
    M0_T1_bias_corr = M0_T1corr*Biasfield_spm
    
    # T2* correction
    
    TE0 = 3.20
    corr = np.exp(-TE0/T2s_map)
    
    M0_T1_T2s_bias_corr = M0_T1_bias_corr/corr
           
    
    
    # CSF calib
    
    
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
    
    T1 = np.zeros(CSF_mask.shape)
    
    
    # New addition:
        
    T1[csf_mask_binary==1] = 4.3
    
    Corr_spoiling = (0.200*(B1_map**2)) - (0.211*B1_map) + 0.959
    
    Corr_spoiling = (0.112*(B1_map**2)) - (0.154*B1_map) + 1.005
    # Corr_spoiling = 1
    
    SF_map = np.sin(B1_map*FA)*(1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1_map*FA))
    
    SF_map = SF_map * Corr_spoiling
    
    M0_T1corr = gre1/SF_map
    M0_T1_biascorr = M0_T1corr*Biasfield_spm
    
    thresh = 0.95
    CSF = M0_T1_biascorr[csf_mask_binary>thresh]
    bins = 200
    interval = [4000, 9000]
    hist, binedges = np.histogram(CSF, bins=bins, range=interval)
     
    bincenters = binedges[0:-1] + np.diff(binedges)/2
    CSF_calib_factor = bincenters[hist.argmax()]
    
    # plt.hist(CSF, bins=bins, range = interval,density=True)
    
    
    
    # T2s_map[~np.isfinite(T2s_map)] = 0 
    # T2s_map[T2s_map>1000] = 1000
    # T2s_CSF = np.mean(T2s_map[csf_mask_binary==1])
    
    # TE = 3.20
    # T2F_CSF = np.exp(-TE/T2s_CSF)
    # CSF_calib_factor_T2scorr = CSF_calib_factor/T2F_CSF
    
    # T2s_CSF2 = T2s_map[csf_mask_binary==1]
    # plt.hist(T2s_CSF2, bins=200, range = [0, 500],density=True)
    
    
    # CSF correction
    
    H2O = M0_T1_T2s_bias_corr/CSF_calib_factor
    # plt.figure();
    # plt.imshow(H2O[:,:,90])
    
    NII_M0_T1_T2s_bias_corr = nib.Nifti1Image(M0_T1_T2s_bias_corr, affine)
    nib.save(NII_M0_T1_T2s_bias_corr, 'M0_T1_T2s_bias_corr_NEW.nii')
    
    # NII_CSF_calib_factor_T2scorr = nib.Nifti1Image(CSF_calib_factor, affine)
    # nib.save(CSF_calib_factor, 'CSF_calib_factor_NEW.nii')
    H2O[~np.isfinite(H2O)] = 0 
    NII_H2O = nib.Nifti1Image(H2O, affine)
    nib.save(NII_H2O, 'H2O_NEW.nii')
    
    
    
    
    
    
    WM_GM_mask = WM_mask_binary + GM_mask_binary
    
    
    # plt.figure()
    # plt.hist(H2O[WM_GM_mask], bins=1000, range = [0,1.1],density=True)
    
    
    np.mean(H2O[GM_mask_binary==1])
    np.mean(H2O[WM_mask_binary==1])







# SKyra


data_list = ['Volunteer10_SECOND_Skyra_11_08_2023', 'Volunteer11_Skyra_25_07_2023', 
             'Volunteer12_Skyra_27_07_2023', 'Volunteer13_Skyra_31_07_2023', 
             'Volunteer15_Skyra_04_08_2023']
 

T1_list_Skyra_WM_mean = list()
T1_list_Skyra_WM_std = list()
T1_list_Skyra_GM_mean = list()
T1_list_Skyra_GM_std = list()

h2o_list_Skyra_WM_mean = list()
h2o_list_Skyra_WM_std = list()
h2o_list_Skyra_GM_mean = list()
h2o_list_Skyra_GM_std = list()

t2s_list_Skyra_WM_mean = list()
t2s_list_Skyra_WM_std = list()
t2s_list_Skyra_GM_mean = list()
t2s_list_Skyra_GM_std = list()

qsm_list_Skyra_WM_mean = list()
qsm_list_Skyra_WM_std = list()
qsm_list_Skyra_GM_mean = list()
qsm_list_Skyra_GM_std = list()


for data_folder in data_list:
    
    
    mpqmri_t1_map_8min_Skyra = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                                     "Final_mp_qMRI_analysis/T1_map_B1corr_True_Spoilcorr_True_2echoes.nii"%data_folder).get_fdata()
        
    mpqmri_t2s_map_8min_Skyra = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "Final_mp_qMRI_analysis/T2Star_avg.nii"%data_folder).get_fdata()
    
    mpqmri_t2s_map_8min_Skyra[~np.isfinite(mpqmri_t2s_map_8min_Skyra)] = 0
    
    mpqmri_h2o_map_8min_Skyra = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "Final_mp_qMRI_analysis/New_H2O_processing/H2O_NEW.nii"%data_folder).get_fdata() * 100
        
    mpqmri_h2o_map_8min_Skyra[~np.isfinite(mpqmri_t2s_map_8min_Skyra)] = 0
        
    mpqmri_qsm_map_8min_Skyra = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'Final_mp_qMRI_analysis/QSM/QSM_avg_map.nii.gz'%data_folder).get_fdata()
    
    #fsl_fastseg(mpqmri_t1w_image_8min_path);
    
    
    GM_mask = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                       "Final_mp_qMRI_analysis/c1resampled_1mm_iso_FA25_mag.nii"%data_folder).get_fdata()
    WM_mask = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                       "Final_mp_qMRI_analysis/c2resampled_1mm_iso_FA25_mag.nii"%data_folder).get_fdata()
    CSF_mask = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                        "Final_mp_qMRI_analysis/c3resampled_1mm_iso_FA25_mag.nii"%data_folder).get_fdata()
    
    WM_mask_binary = WM_mask>0.98
    WM_mask_binary[CSF_mask>0.80] = False 
    WM_mask_binary[GM_mask>0.80] = False
    
    GM_mask_binary = GM_mask>0.98 
    GM_mask_binary[CSF_mask>0.80] = False 
    GM_mask_binary[WM_mask>0.80] = False 
    GM_mask_binary[mpqmri_t1_map_8min_Skyra<1200]=False
    GM_mask_binary[mpqmri_t1_map_8min_Skyra>1600]=False
    
    csf_mask = np.zeros(CSF_mask.shape)
    csf_mask[CSF_mask>0.95] = 1
    csf_mask[WM_mask>0.95] = 0
    csf_mask[GM_mask>0.95] = 0
    
    # WM_mask_binary[:,:,70:]= False  #The portion of the brain excluded
    #                                 #as the tumour resides in this region
    # GM_mask_binary[:,:,70:]= False #The portion of the brain excluded
    #                                 #as the tumour resides in the region
    # csf_mask[:,:,70:]= 0            #The portion of the brain excluded
    #                                 #as the tumour resides in this region
                                    
    WM_mask_binary[~np.isfinite(mpqmri_t1_map_8min_Skyra)] = False 
    GM_mask_binary[~np.isfinite(mpqmri_t1_map_8min_Skyra)] = False 
    
    WM_mask_binary[~np.isfinite(mpqmri_h2o_map_8min_Skyra)] = False 
    GM_mask_binary[~np.isfinite(mpqmri_h2o_map_8min_Skyra)] = False 
    
    mpqmri_WM_mask_binary_8min = WM_mask_binary
    mpqmri_GM_mask_binary_8min = GM_mask_binary
    
    mpqmri_t1_WM_8min_Skyra = mpqmri_t1_map_8min_Skyra[WM_mask_binary]
    mpqmri_t1_GM_8min_Skyra = mpqmri_t1_map_8min_Skyra[GM_mask_binary]
    
    mpqmri_t2s_WM_8min_Skyra = mpqmri_t2s_map_8min_Skyra[WM_mask_binary]
    mpqmri_t2s_GM_8min_Skyra = mpqmri_t2s_map_8min_Skyra[GM_mask_binary]
    
    mpqmri_h2o_WM_8min_Skyra = mpqmri_h2o_map_8min_Skyra[WM_mask_binary]
    mpqmri_h2o_GM_8min_Skyra = mpqmri_h2o_map_8min_Skyra[GM_mask_binary]
    
    mpqmri_qsm_WM_8min_Skyra = mpqmri_qsm_map_8min_Skyra[WM_mask_binary]
    mpqmri_qsm_GM_8min_Skyra = mpqmri_qsm_map_8min_Skyra[GM_mask_binary]
    
    mpqmri_WM_GM_mask_8min_Skyra = WM_mask_binary + GM_mask_binary
    
    WM_mask_binary_Skyra = WM_mask_binary
    GM_mask_binary_Skyra = GM_mask_binary
    
    T1_list_Skyra_WM_mean.append(np.mean(mpqmri_t1_WM_8min_Skyra))
    T1_list_Skyra_GM_mean.append(np.mean(mpqmri_t1_GM_8min_Skyra))
    T1_list_Skyra_WM_std.append(np.std(mpqmri_t1_WM_8min_Skyra))
    T1_list_Skyra_GM_std.append(np.std(mpqmri_t1_GM_8min_Skyra))
    
    h2o_list_Skyra_WM_mean.append(np.mean(mpqmri_h2o_WM_8min_Skyra))
    h2o_list_Skyra_GM_mean.append(np.mean(mpqmri_h2o_GM_8min_Skyra))
    h2o_list_Skyra_WM_std.append(np.std(mpqmri_h2o_WM_8min_Skyra))
    h2o_list_Skyra_GM_std.append(np.std(mpqmri_h2o_GM_8min_Skyra))
    
    t2s_list_Skyra_WM_mean.append(np.mean(mpqmri_t2s_WM_8min_Skyra))
    t2s_list_Skyra_GM_mean.append(np.mean(mpqmri_t2s_GM_8min_Skyra))
    t2s_list_Skyra_WM_std.append(np.std(mpqmri_t2s_WM_8min_Skyra))
    t2s_list_Skyra_GM_std.append(np.std(mpqmri_t2s_GM_8min_Skyra))
    
    qsm_list_Skyra_WM_mean.append(np.mean(mpqmri_qsm_WM_8min_Skyra))
    qsm_list_Skyra_GM_mean.append(np.mean(mpqmri_qsm_GM_8min_Skyra))
    qsm_list_Skyra_WM_std.append(np.std(mpqmri_qsm_WM_8min_Skyra))
    qsm_list_Skyra_GM_std.append(np.std(mpqmri_qsm_GM_8min_Skyra))


# Prisma

data_list = ['Volunteer10_SECOND_Prisma_07_08_2023', 'Volunteer11_Prisma_07_07_2023', 
             'Volunteer12_Prisma_11_07_2023', 'Volunteer13_Prisma_26_07_2023', 
             'Volunteer15_Prisma_04_08_2023']

T1_list_Prisma_WM_mean = list()
T1_list_Prisma_WM_std = list()
T1_list_Prisma_GM_mean = list()
T1_list_Prisma_GM_std = list()

h2o_list_Prisma_WM_mean = list()
h2o_list_Prisma_WM_std = list()
h2o_list_Prisma_GM_mean = list()
h2o_list_Prisma_GM_std = list()

t2s_list_Prisma_WM_mean = list()
t2s_list_Prisma_WM_std = list()
t2s_list_Prisma_GM_mean = list()
t2s_list_Prisma_GM_std = list()

qsm_list_Prisma_WM_mean = list()
qsm_list_Prisma_WM_std = list()
qsm_list_Prisma_GM_mean = list()
qsm_list_Prisma_GM_std = list()

for data_folder in data_list:
    
    
    mpqmri_t1_map_8min_Prisma = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                                     "Final_mp_qMRI_analysis/T1_map_B1corr_True_Spoilcorr_True_2echoes.nii"%data_folder).get_fdata()
        
    mpqmri_t2s_map_8min_Prisma = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "Final_mp_qMRI_analysis/T2Star_avg.nii"%data_folder).get_fdata()
    
    mpqmri_t2s_map_8min_Prisma[~np.isfinite(mpqmri_t2s_map_8min_Prisma)] = 0
    
    mpqmri_h2o_map_8min_Prisma = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "Final_mp_qMRI_analysis/New_H2O_processing/H2O_NEW.nii"%data_folder).get_fdata() * 100
        
    mpqmri_h2o_map_8min_Prisma[~np.isfinite(mpqmri_t2s_map_8min_Prisma)] = 0
        
    mpqmri_qsm_map_8min_Prisma = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'Final_mp_qMRI_analysis/QSM/QSM_avg_map.nii.gz'%data_folder).get_fdata()
    
    #fsl_fastseg(mpqmri_t1w_image_8min_path);
    
    
    GM_mask = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                       "Final_mp_qMRI_analysis/c1resampled_1mm_iso_FA25_mag.nii"%data_folder).get_fdata()
    WM_mask = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                       "Final_mp_qMRI_analysis/c2resampled_1mm_iso_FA25_mag.nii"%data_folder).get_fdata()
    CSF_mask = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                        "Final_mp_qMRI_analysis/c3resampled_1mm_iso_FA25_mag.nii"%data_folder).get_fdata()
    
    
    WM_mask_binary = WM_mask>0.98
    WM_mask_binary[CSF_mask>0.80] = False 
    WM_mask_binary[GM_mask>0.80] = False
    
    GM_mask_binary = GM_mask>0.98 
    GM_mask_binary[CSF_mask>0.80] = False 
    GM_mask_binary[WM_mask>0.80] = False 
    GM_mask_binary[mpqmri_t1_map_8min_Prisma<1200]=False
    GM_mask_binary[mpqmri_t1_map_8min_Prisma>1600]=False
    
    csf_mask = np.zeros(CSF_mask.shape)
    csf_mask[CSF_mask>0.95] = 1
    csf_mask[WM_mask>0.95] = 0
    csf_mask[GM_mask>0.95] = 0
    
    # WM_mask_binary[:,:,70:]= False  #The portion of the brain excluded
    #                                 #as the tumour resides in this region
    # GM_mask_binary[:,:,70:]= False #The portion of the brain excluded
    #                                 #as the tumour resides in the region
    # csf_mask[:,:,70:]= 0            #The portion of the brain excluded
    #                                 #as the tumour resides in this region
                                    
    WM_mask_binary[~np.isfinite(mpqmri_t1_map_8min_Prisma)] = False 
    GM_mask_binary[~np.isfinite(mpqmri_t1_map_8min_Prisma)] = False 
    
    WM_mask_binary[~np.isfinite(mpqmri_h2o_map_8min_Prisma)] = False 
    GM_mask_binary[~np.isfinite(mpqmri_h2o_map_8min_Prisma)] = False 
    
    mpqmri_WM_mask_binary_8min = WM_mask_binary
    mpqmri_GM_mask_binary_8min = GM_mask_binary
    
    mpqmri_t1_WM_8min_Prisma = mpqmri_t1_map_8min_Prisma[WM_mask_binary]
    mpqmri_t1_GM_8min_Prisma = mpqmri_t1_map_8min_Prisma[GM_mask_binary]
    
    mpqmri_t2s_WM_8min_Prisma = mpqmri_t2s_map_8min_Prisma[WM_mask_binary]
    mpqmri_t2s_GM_8min_Prisma = mpqmri_t2s_map_8min_Prisma[GM_mask_binary]
    
    mpqmri_h2o_WM_8min_Prisma = mpqmri_h2o_map_8min_Prisma[WM_mask_binary]
    mpqmri_h2o_GM_8min_Prisma = mpqmri_h2o_map_8min_Prisma[GM_mask_binary]
    
    mpqmri_qsm_WM_8min_Prisma = mpqmri_qsm_map_8min_Prisma[WM_mask_binary]
    mpqmri_qsm_GM_8min_Prisma = mpqmri_qsm_map_8min_Prisma[GM_mask_binary]
    
    mpqmri_WM_GM_mask_8min_Prisma = WM_mask_binary + GM_mask_binary
    
    WM_mask_binary_Prisma = WM_mask_binary
    GM_mask_binary_Prisma = GM_mask_binary
    
    T1_list_Prisma_WM_mean.append(np.mean(mpqmri_t1_WM_8min_Prisma))
    T1_list_Prisma_GM_mean.append(np.mean(mpqmri_t1_GM_8min_Prisma))
    T1_list_Prisma_WM_std.append(np.std(mpqmri_t1_WM_8min_Prisma))
    T1_list_Prisma_GM_std.append(np.std(mpqmri_t1_GM_8min_Prisma))
    
    h2o_list_Prisma_WM_mean.append(np.mean(mpqmri_h2o_WM_8min_Prisma))
    h2o_list_Prisma_GM_mean.append(np.mean(mpqmri_h2o_GM_8min_Prisma))
    h2o_list_Prisma_WM_std.append(np.std(mpqmri_h2o_WM_8min_Prisma))
    h2o_list_Prisma_GM_std.append(np.std(mpqmri_h2o_GM_8min_Prisma))
    
    t2s_list_Prisma_WM_mean.append(np.mean(mpqmri_t2s_WM_8min_Prisma))
    t2s_list_Prisma_GM_mean.append(np.mean(mpqmri_t2s_GM_8min_Prisma))
    t2s_list_Prisma_WM_std.append(np.std(mpqmri_t2s_WM_8min_Prisma))
    t2s_list_Prisma_GM_std.append(np.std(mpqmri_t2s_GM_8min_Prisma))
    
    qsm_list_Prisma_WM_mean.append(np.mean(mpqmri_qsm_WM_8min_Prisma))
    qsm_list_Prisma_GM_mean.append(np.mean(mpqmri_qsm_GM_8min_Prisma))
    qsm_list_Prisma_WM_std.append(np.std(mpqmri_qsm_WM_8min_Prisma))
    qsm_list_Prisma_GM_std.append(np.std(mpqmri_qsm_GM_8min_Prisma))
    
    
    
    
# MNI




data_list = ['Volunteer10_SECOND_Skyra_11_08_2023', 'Volunteer11_Skyra_25_07_2023', 
             'Volunteer12_Skyra_27_07_2023', 'Volunteer13_Skyra_31_07_2023', 
             'Volunteer15_Skyra_04_08_2023']


T1_list_Skyra_dict = dict()

h2o_list_Skyra_dict = dict()

t2s_list_Skyra_dict = dict()

qsm_list_Skyra_dict = dict()


for data_folder in data_list:
    
    brain_mask = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/MNI152_T1_1mm_brain_brain_mask.nii.gz'%data_folder).get_fdata() 
    
    T1_list_Skyra_dict[data_folder] = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                                     "MNI_space/T1_map_B1corr_True_Spoilcorr_True_2echoesstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz"%data_folder).get_fdata() * brain_mask
        
    t2s_list_Skyra_dict[data_folder]  = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "MNI_space/T2Star_avgstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz"%data_folder).get_fdata() * brain_mask
    
    (t2s_list_Skyra_dict[data_folder])[~np.isfinite(t2s_list_Skyra_dict[data_folder])] = 0
    
    h2o_list_Skyra_dict[data_folder]  = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "MNI_space/H2O_NEWstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz"%data_folder).get_fdata() * 100 * brain_mask
        
    (h2o_list_Skyra_dict[data_folder])[~np.isfinite(h2o_list_Skyra_dict[data_folder])] = 0
        
    qsm_list_Skyra_dict[data_folder] = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/QSM_avg_mapstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz'%data_folder).get_fdata() * brain_mask
        
    
 





MNI = nib.load('Z:/NR-qCET/analysis/Volunteers/Volunteer10_SECOND_Skyra_11_08_2023/MNI_space/MNI152_T1_1mm_brain.nii.gz').get_fdata()


# Prisma

data_list = ['Volunteer10_SECOND_Prisma_07_08_2023', 'Volunteer11_Prisma_07_07_2023', 
             'Volunteer12_Prisma_11_07_2023', 'Volunteer13_Prisma_26_07_2023', 
             'Volunteer15_Prisma_04_08_2023']

T1_list_Prisma_dict = dict()

h2o_list_Prisma_dict = dict()

t2s_list_Prisma_dict = dict()

qsm_list_Prisma_dict = dict()


for data_folder in data_list:
    
    brain_mask = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/MNI152_T1_1mm_brain_brain_mask.nii.gz'%data_folder).get_fdata()
    
    T1_list_Prisma_dict[data_folder] = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                                     "MNI_space/T1_map_B1corr_True_Spoilcorr_True_2echoesstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz"%data_folder).get_fdata() * brain_mask
        
    t2s_list_Prisma_dict[data_folder]  = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "MNI_space/T2Star_avgstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz"%data_folder).get_fdata() * brain_mask
    
    (t2s_list_Prisma_dict[data_folder])[~np.isfinite(t2s_list_Prisma_dict[data_folder])] = 0
    
    h2o_list_Prisma_dict[data_folder]  = nib.load("Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "MNI_space/H2O_NEWstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz"%data_folder).get_fdata() * 100 * brain_mask
        
    (h2o_list_Prisma_dict[data_folder])[~np.isfinite(h2o_list_Prisma_dict[data_folder])] = 0
        
    qsm_list_Prisma_dict[data_folder] = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/QSM_avg_mapstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz'%data_folder).get_fdata() * brain_mask


data_list_Prisma = ['Volunteer10_SECOND_Prisma_07_08_2023', 'Volunteer11_Prisma_07_07_2023', 
             'Volunteer12_Prisma_11_07_2023', 'Volunteer13_Prisma_26_07_2023', 
             'Volunteer15_Prisma_04_08_2023']


data_list_Skyra = ['Volunteer10_SECOND_Skyra_11_08_2023', 'Volunteer11_Skyra_25_07_2023', 
             'Volunteer12_Skyra_27_07_2023', 'Volunteer13_Skyra_31_07_2023', 
             'Volunteer15_Skyra_04_08_2023']


COV_T1_dict = dict()

for i in range(len(data_list)):  
    
    data_folder = data_list_Prisma[i]
    prisma = T1_list_Prisma_dict[data_folder]
    data_folder = data_list_Skyra[i]
    skyra = T1_list_Skyra_dict[data_folder]
    mean = (prisma + skyra)/2
    var = (prisma - mean)**2 + (skyra - mean)**2
    std = np.sqrt(var)
    COV = std/mean * 100
    COV[~np.isfinite(COV)] = 0
    
    COV_T1_dict[data_folder] = COV

    

COV_h2o_dict = dict()

for i in range(len(data_list)):  
    
    data_folder = data_list_Prisma[i]
    prisma = h2o_list_Prisma_dict[data_folder]
    data_folder = data_list_Skyra[i]
    skyra = h2o_list_Skyra_dict[data_folder]
    mean = (prisma + skyra)/2
    var = (prisma - mean)**2 + (skyra - mean)**2
    std = np.sqrt(var)
    COV = std/mean * 100
    COV[~np.isfinite(COV)] = 0
    
    COV_h2o_dict[data_folder] = COV



COV_t2s_dict = dict()

for i in range(len(data_list)):  
    
    data_folder = data_list_Prisma[i]
    prisma = t2s_list_Prisma_dict[data_folder]
    data_folder = data_list_Skyra[i]
    skyra = t2s_list_Skyra_dict[data_folder]
    mean = (prisma + skyra)/2
    var = (prisma - mean)**2 + (skyra - mean)**2
    std = np.sqrt(var)
    COV = std/mean * 100
    COV[~np.isfinite(COV)] = 0
    
    COV_t2s_dict[data_folder] = COV



COV_qsm_dict = dict()

for i in range(len(data_list)):  
    
    data_folder = data_list_Prisma[i]
    prisma = qsm_list_Prisma_dict[data_folder]
    data_folder = data_list_Skyra[i]
    skyra = qsm_list_Skyra_dict[data_folder]
    mean = (prisma + skyra)/2
    var = (prisma - mean)**2 + (skyra - mean)**2
    std = np.sqrt(var)
    COV = std/mean * 100
    COV[~np.isfinite(COV)] = 0
    
    COV_qsm_dict[data_folder] = COV




 
fig, ax = plt.subplots(1,5)

cmap = 'gray'
ax[0].imshow(COV_T1_dict[data_list_Skyra[0]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[1].imshow(COV_T1_dict[data_list_Skyra[1]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[2].imshow(COV_T1_dict[data_list_Skyra[2]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[3].imshow(COV_T1_dict[data_list_Skyra[3]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[4].imshow(COV_T1_dict[data_list_Skyra[4]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)

for i in range(5):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=-0.05,
                    wspace=-0)

# plt.savefig('PLOT_27_COV_T1.tiff', dpi = 350)




fig, ax = plt.subplots(1,5)

cmap = 'gray'
ax[0].imshow(COV_h2o_dict[data_list_Skyra[0]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[1].imshow(COV_h2o_dict[data_list_Skyra[1]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[2].imshow(COV_h2o_dict[data_list_Skyra[2]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[3].imshow(COV_h2o_dict[data_list_Skyra[3]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[4].imshow(COV_h2o_dict[data_list_Skyra[4]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)

for i in range(5):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=-0.05,
                    wspace=-0)

# plt.savefig('PLOT_27_COV_H2O.tiff', dpi = 350)





fig, ax = plt.subplots(1,5)

cmap = 'gray'
ax[0].imshow(COV_t2s_dict[data_list_Skyra[0]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[1].imshow(COV_t2s_dict[data_list_Skyra[1]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[2].imshow(COV_t2s_dict[data_list_Skyra[2]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[3].imshow(COV_t2s_dict[data_list_Skyra[3]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[4].imshow(COV_t2s_dict[data_list_Skyra[4]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)

for i in range(5):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=-0.05,
                    wspace=-0)

# plt.savefig('PLOT_27_COV_T2s.tiff', dpi = 350)





fig, ax = plt.subplots(1,5)

cmap = 'gray'
ax[0].imshow(COV_qsm_dict[data_list_Skyra[0]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[1].imshow(COV_qsm_dict[data_list_Skyra[1]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[2].imshow(COV_qsm_dict[data_list_Skyra[2]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[3].imshow(COV_qsm_dict[data_list_Skyra[3]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)
ax[4].imshow(COV_qsm_dict[data_list_Skyra[4]][:,:,90], vmin = 0, vmax = 15, cmap = cmap)

for i in range(5):
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=-0.05,
                    wspace=-0)






data_list = ['Volunteer10_SECOND_Skyra_11_08_2023','Volunteer10_SECOND_Prisma_07_08_2023',
            'Volunteer11_Prisma_07_07_2023', 'Volunteer11_Skyra_25_07_2023',
            'Volunteer12_Prisma_11_07_2023', 'Volunteer12_Skyra_27_07_2023',
            'Volunteer13_Prisma_26_07_2023', 'Volunteer13_Skyra_31_07_2023',
            'Volunteer15_Prisma_04_08_2023', 'Volunteer15_Skyra_04_08_2023']



for data_folder in data_list:
    
    mpqmri_t1_map_8min_Skyra = "Z:/NR-qCET/analysis/Volunteers/%s/"\
                                                     "MNI_space/T1_map_B1corr_True_Spoilcorr_True_2echoes.nii"%data_folder
                                                     
    mpqmri_t2s_map_8min_Skyra = "Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "MNI_space/T2Star_avg.nii"%data_folder

    mpqmri_h2o_map_8min_Skyra = "Z:/NR-qCET/analysis/Volunteers/%s/"\
                                  "MNI_space/H2O_NEW.nii"%data_folder

    mpqmri_qsm_map_8min_Skyra = 'Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/QSM_avg_map.nii.gz'%data_folder
                                    
    moving_nii = 'Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/resampled_1mm_iso_FA25_mag_brain.nii.gz'%data_folder
    
    fixed_nii = 'Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/MNI152_T1_1mm_brain.nii.gz'%data_folder
                                          
    additional_images_list = [mpqmri_t1_map_8min_Skyra, mpqmri_t2s_map_8min_Skyra, mpqmri_h2o_map_8min_Skyra, mpqmri_qsm_map_8min_Skyra]
    

    fsl_NON_linear_normalization_AFTER_FLIRT(moving_nii, fixed_nii, additional_images_list = additional_images_list)
    
    fsl_fastseg('Z:/NR-qCET/analysis/Volunteers/%s/'\
                                          'MNI_space/resampled_1mm_iso_FA25_mag_brainstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized.nii.gz'%data_folder)
        
        
        
        

data_list_Skyra = ['Volunteer10_SECOND_Skyra_11_08_2023', 'Volunteer11_Skyra_25_07_2023', 
             'Volunteer12_Skyra_27_07_2023', 'Volunteer13_Skyra_31_07_2023', 
             'Volunteer15_Skyra_04_08_2023']


 
      

        
COV_T1_dict_mean_WM = dict()
COV_h2o_dict_mean_WM = dict()
COV_t2s_dict_mean_WM = dict()
COV_qsm_dict_mean_WM = dict()

COV_T1_dict_mean_GM = dict()
COV_h2o_dict_mean_GM = dict()
COV_t2s_dict_mean_GM = dict()
COV_qsm_dict_mean_GM = dict()

COV_T1_dict_mean_CSF = dict()
COV_h2o_dict_mean_CSF = dict()
COV_t2s_dict_mean_CSF = dict()
COV_qsm_dict_mean_CSF = dict()

COV_T1_dict_mean_WM_list = []
COV_h2o_dict_mean_WM_list = []
COV_t2s_dict_mean_WM_list = []
COV_qsm_dict_mean_WM_list = []

COV_T1_dict_mean_GM_list = []
COV_h2o_dict_mean_GM_list = []
COV_t2s_dict_mean_GM_list = []
COV_qsm_dict_mean_GM_list = []

COV_T1_dict_mean_CSF_list = []
COV_h2o_dict_mean_CSF_list = []
COV_t2s_dict_mean_CSF_list = []
COV_qsm_dict_mean_CSF_list= []


for i in range(len(data_list)):  
    
    data_folder = data_list_Skyra[i]
    
    GM_mask = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                       'MNI_space/resampled_1mm_iso_FA25_mag_brainstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized_pve_1.nii.gz'%data_folder).get_fdata()
    WM_mask = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                       'MNI_space/resampled_1mm_iso_FA25_mag_brainstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized_pve_2.nii.gz'%data_folder).get_fdata()
    CSF_mask = nib.load('Z:/NR-qCET/analysis/Volunteers/%s/'\
                       'MNI_space/resampled_1mm_iso_FA25_mag_brainstd_2_MNI152_T1_1mm_brain_Non_Linear_Normalized_pve_0.nii.gz'%data_folder).get_fdata()

    WM_mask_binary = WM_mask>0.98
    WM_mask_binary[CSF_mask>0.80] = False 
    WM_mask_binary[GM_mask>0.80] = False

    GM_mask_binary = GM_mask>0.98 
    GM_mask_binary[CSF_mask>0.80] = False 
    GM_mask_binary[WM_mask>0.80] = False 

    CSF_mask_binary = np.zeros(CSF_mask.shape)
    CSF_mask_binary[CSF_mask>0.95] = 1
    CSF_mask_binary[WM_mask>0.95] = 0
    CSF_mask_binary[GM_mask>0.95] = 0 
    
    COV_T1 = COV_T1_dict[data_folder]
    COV_h2o = COV_h2o_dict[data_folder]
    COV_t2s = COV_t2s_dict[data_folder]
    COV_qsm = COV_qsm_dict[data_folder]
    
    COV_T1_dict_mean_WM[data_folder] = np.mean(COV_T1[WM_mask_binary==1])
    COV_h2o_dict_mean_WM[data_folder] = np.mean(COV_h2o[WM_mask_binary==1])
    COV_t2s_dict_mean_WM[data_folder] = np.mean(COV_t2s[WM_mask_binary==1])
    COV_qsm_dict_mean_WM[data_folder] = np.mean(COV_qsm[WM_mask_binary==1])  
    
    COV_T1_dict_mean_GM[data_folder] = np.mean(COV_T1[GM_mask_binary==1])
    COV_h2o_dict_mean_GM[data_folder] = np.mean(COV_h2o[GM_mask_binary==1])
    COV_t2s_dict_mean_GM[data_folder] = np.mean(COV_t2s[GM_mask_binary==1])
    COV_qsm_dict_mean_GM[data_folder] = np.mean(COV_qsm[GM_mask_binary==1])  
    
    COV_T1_dict_mean_CSF[data_folder] = np.mean(COV_T1[CSF_mask_binary==1])
    COV_h2o_dict_mean_CSF[data_folder] = np.mean(COV_h2o[CSF_mask_binary==1])
    COV_t2s_dict_mean_CSF[data_folder] = np.mean(COV_t2s[CSF_mask_binary==1])
    COV_qsm_dict_mean_CSF[data_folder] = np.mean(COV_qsm[CSF_mask_binary==1])  
    
    COV_T1_dict_mean_WM_list.append(np.mean(COV_T1[WM_mask_binary==1]))
    COV_h2o_dict_mean_WM_list.append(np.mean(COV_h2o[WM_mask_binary==1]))
    COV_t2s_dict_mean_WM_list.append(np.mean(COV_t2s[WM_mask_binary==1]))
    COV_qsm_dict_mean_WM_list.append(np.mean(COV_qsm[WM_mask_binary==1]))
    
    COV_T1_dict_mean_GM_list.append(np.mean(COV_T1[GM_mask_binary==1]))
    COV_h2o_dict_mean_GM_list.append(np.mean(COV_h2o[GM_mask_binary==1]))
    COV_t2s_dict_mean_GM_list.append(np.mean(COV_t2s[GM_mask_binary==1]))
    COV_qsm_dict_mean_GM_list.append(np.mean(COV_qsm[GM_mask_binary==1]))
    
    
    
    
        

    
