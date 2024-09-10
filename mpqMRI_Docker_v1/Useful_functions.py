# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:33:43 2022

@author: denni

Useful functions
"""
import subprocess
import os
import shutil
from pathlib import Path
import nipype.interfaces.spm as spm
import nibabel as nib
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import resample

cwd = os.getcwd()

def SPM_segment(data):
    # Define the WSL command with the modified path
    wsl_command = f" python3 /app/SPM_script.py {data}"
    
    # Execute the WSL command
    subprocess.run(wsl_command, shell=True)


# Brain masking with FSL
def fsl_brain_masking(file_path):
    """

    Parameters
    ----------
    file_path : path to the file to be processed
        

    Returns
    -------
    binary brain mask file as '_brain_mask.nii.gz' appended

    """
  
    FSL_dirpath = os.path.join(cwd, "FSL_temp")
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)

    os.mkdir(FSL_dirpath)
    src = file_path
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)

    filename = os.path.basename(file_path)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
    
    input_path = os.path.join(FSL_dirpath, filename)
    output_path = os.path.join(FSL_dirpath, filename + '_brain')

    
    subprocess.run(['bet2', input_path, output_path, '-f', '0.5', '-g', '0', '-m'], check=True)
    

    src = os.path.join(cwd, "FSL_temp", filename + '_brain.nii.gz')
    dst = os.path.split(file_path)[0]

    shutil.copy(src, dst)

    src = os.path.join(cwd, "FSL_temp", filename + '_brain_mask.nii.gz')
    dst = os.path.split(file_path)[0]

    shutil.copy(src, dst)

    shutil.rmtree(FSL_dirpath)




def fsl_fastseg(file_path):
    
    FSL_dirpath = os.path.join(cwd, "FSL_temp")
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)

    os.mkdir(FSL_dirpath)

    src = file_path
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)

    filename = os.path.basename(file_path)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
        
    
    input_path = os.path.join(FSL_dirpath,filename+'.nii.gz')
    
    subprocess.run(['fast','-b',input_path],check=True)
    for file in Path(FSL_dirpath).glob('%s*'%filename):
        print('transferred %s' %file)
        src = file
        dst = os.path.split(file_path)[0]
        shutil.copy(src, dst)
        
    shutil.rmtree(FSL_dirpath)
   
    
    
    

def rename(directory):
    """
    rename the b1 mapping files to b1_1 and b1_2

    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    os.chdir(directory)
    num = 1
    for file in [file for file in sorted(os.listdir(), key=os.path.getctime, reverse=False) 
                 if os.path.splitext(file)[1] == ".gz"]:
        if os.path.splitext(file)[1] == ".gz":
            print(file)
            os.rename(file, f"b1_{num}.nii.gz")
            num += 1




def fsl_fastseg_B1mapping(file_path):
    
    
    FSL_dirpath =os.path.join(cwd, "FSL_temp")
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)
    
    os.mkdir(FSL_dirpath);
    
    src = file_path
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    filename = os.path.basename(file_path)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]

    input_path = os.path.join(FSL_dirpath,filename)
        
    
    
    subprocess.run(['fast','-I',str(16),'-l', str(15) ,'-t', str(2) ,'-n', str(5) ,'-b',input_path],check=True)
    
    for file in Path(FSL_dirpath).glob('%s*'%filename):
        print('transferred %s' %file)
        src = file
        dst = os.path.split(file_path)[0]
        shutil.copy(src, dst)
        
    shutil.rmtree(FSL_dirpath)   
    




def fsl_flirt_registration(moving_nii, fixed_nii,additional_args=None, dof=6):
      
    FSL_dirpath =os.path.join(cwd, "FSL_temp")
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)
    
    os.mkdir(FSL_dirpath)
    
    src = moving_nii
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    
    src = fixed_nii
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    
    filename1 = os.path.basename(moving_nii)
    if filename1[-3:] == '.gz':
        filename1 = filename1[:-7]
    else:
        filename1 = filename1[:-4]
    
    input_path = os.path.join(FSL_dirpath, filename1)
    
    filename2 = os.path.basename(fixed_nii)
    if filename2[-3:] == '.gz':
        filename2 = filename2[:-7]
    else:
        filename2 = filename2[:-4]
        
    ref_path = os.path.join(FSL_dirpath, filename2)
    
    out_path = os.path.join(FSL_dirpath,  filename1 + '_2_' + filename2)
    
    omat_path =os.path.join(FSL_dirpath,  filename1 + '_2_' + filename2)
    

    dof = str(dof)
  
     # Construct the base command
    command = ['flirt', '-in', input_path, '-ref', ref_path, '-out', out_path, '-omat', omat_path, '-dof', dof]
    
    # Add additional arguments if provided
    if additional_args:
        command.extend(additional_args.split())
    
    # Run the command
    subprocess.run(command, check=True)
    
    
    for file in Path(FSL_dirpath).glob('%s*'%filename1):
        print('transferred %s' %file)
        src = file
        dst = os.path.split(moving_nii)[0]
        shutil.copy(src, dst)
        
    shutil.rmtree(FSL_dirpath)   
    
    return os.path.basename(out_path)
    
    
    
    
    
   


def fsl_flirt_applyxfm(moving_nii, fixed_nii, matrix):
    FSL_dirpath =os.path.join(cwd, "FSL_temp")
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)
    
    os.mkdir(FSL_dirpath);
    
    src = moving_nii
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    
    src = fixed_nii
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    
    src = matrix
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    
    
    filename1 = os.path.basename(moving_nii)
    if filename1[-3:] == '.gz':
        filename1 = filename1[:-7]
    else:
        filename1 = filename1[:-4]
    
    input_path = os.path.join(FSL_dirpath,filename1)
    
    filename2 = os.path.basename(fixed_nii)
    if filename2[-3:] == '.gz':
        filename2 = filename2[:-7]
    else:
        filename2 = filename2[:-4]
    filename3 = os.path.basename(matrix)
     
         
    ref_path = os.path.join(FSL_dirpath , filename2)
     
    out_path = os.path.join(FSL_dirpath , filename1 + '_2_' + filename2)
     
    mat_path = os.path.join(FSL_dirpath , filename3)
   
  
    subprocess.run(['flirt', '-in', input_path, '-ref', ref_path, '-out', out_path, '-init', mat_path, '-applyxfm'], check=True)
   
    
    for file in Path(FSL_dirpath).glob('%s*'%filename1):
        print('transferred %s' %file)
        src = file
        dst = os.path.split(moving_nii)[0]
        shutil.copy(src, dst)
        
    shutil.rmtree(FSL_dirpath)
    

    return os.path.basename(out_path)
    
    
    
def spm_coreg(target_file, source_file):
    """

    Parameters
    ----------
    target_file : path-like string
        path to the file that has to be coregistered
    source_file : path-like string
        path to the file that is to be registered to

    Returns
    -------
    None.

    """

    spm.SPMCommand.set_mlab_paths(matlab_cmd = 'matlab -nodesktop -nosplash', paths ='C:/Users/denni/OneDrive/Desktop/spm12')
    coreg = spm.Coregister()
    coreg.inputs.target = target_file
    coreg.inputs.source = source_file

    coreg.run()

def spm_biasfield(file, reg=0.001, FWHM=60):
    """

    Parameters
    ----------
    file : path-like string
        path to the input file

    Returns
    -------
    Bias corrected images and Bias field is saved in the input file path

    """

    spm.SPMCommand.set_mlab_paths(matlab_cmd = 'matlab -nodesktop -nosplash', paths ='C:/Users/denni/OneDrive/Desktop/spm12')
    seg = spm.NewSegment()


    seg.inputs.channel_files = file
    seg.inputs.channel_info = (reg, FWHM, (True, True))

    seg.run()


def threshold_masking(mag_path, threshold, save=True):
    """

    Parameters
    ----------
    mag_path : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    mask : TYPE
        DESCRIPTION.

    """
    mag = nib.load(mag_path).get_fdata()
    affine = nib.load(mag_path).affine
    
    if len(mag.shape)>3:
        
        mag = mag[:,:,:,0]
        
    mask = mag> threshold
    mask_bin = np.zeros(mask.shape)
    mask_bin[mask==True]= 1
    
    if save:
        
        mask_nii = nib.Nifti1Image(mask_bin, affine)
        filename = os.path.basename(mag_path)
        if filename[-3:] == '.gz':
            filename = filename[:-7]
        else:
            filename = filename[:-4]
            
        dst = os.path.split(mag_path)[0]
    
        mask_path = dst + '/' + filename + '_mask.nii.gz'
        
        nib.save(mask_nii, mask_path)
    
    return mask
    


def T1_weighting_factor(B1_Plus, T1, FA, TR):

    " FA in radians "

    return np.sin(B1_Plus*FA)*(1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1_Plus*FA))


def convert_3D_to4D(nifti_file, size_of_4th_D = None):
    
    nii = nib.load(nifti_file);
    nii_mag = nii.get_fdata()
    
    nii_mag_appended = nii_mag[:,:,:, None]
    
    if size_of_4th_D is not None:
        nii_mag_appended = np.repeat(nii_mag_appended, repeats=size_of_4th_D, axis=3)

    affine = nii.affine
    
    nii_appended = nib.Nifti1Image(nii_mag_appended, affine)
    
    filename = os.path.basename(nifti_file)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
        
        
    dst = os.path.split(nifti_file)[0]

    mask_path = dst + '/' + filename + '_appended.nii.gz'
    
    nib.save(nii_appended, mask_path)
        
    
    
def save_first_echo(nifti_file):
    
    nii = nib.load(nifti_file);
    nii_mag = nii.get_fdata()
    
    nii_mag_appended = nii_mag[:,:,:,0]
    
    affine = nii.affine
    
    nii_appended = nib.Nifti1Image(nii_mag_appended, affine)
    
    filename = os.path.basename(nifti_file)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
        
        
    dst = os.path.split(nifti_file)[0]

    mask_path = dst + '/' + filename + '_firstecho.nii.gz'
    
    nib.save(nii_appended, mask_path)
    


def save_nth_echo(nifti_file, n):
    
    nii = nib.load(nifti_file);
    nii_mag = nii.get_fdata()
    
    nii_mag_appended = nii_mag[:,:,:,n-1]
    
    affine = nii.affine
    
    nii_appended = nib.Nifti1Image(nii_mag_appended, affine)
    
    filename = os.path.basename(nifti_file)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
        
        
    dst = os.path.split(nifti_file)[0]

    mask_path = dst + '/' + filename + '_%i_echo.nii.gz'%n
    
    nib.save(nii_appended, mask_path)


def split_all_echoes(nifti_file):
    
    nii = nib.load(nifti_file);
    nii_mag = nii.get_fdata()
    n_echoes = np.shape(nii_mag)[-1]
    
    for i in range(n_echoes):
        
        save_nth_echo(nifti_file, i+1)
    
    
    
def tgv_qsm2(phase, mask, t, f=3.0):
        
    qsm_dirpath = os.path.join(cwd, "QSM_temp")
    if os.path.exists(qsm_dirpath):
        shutil.rmtree(qsm_dirpath)
    
    os.mkdir(qsm_dirpath);
    
    src = phase
    dst = qsm_dirpath
    shutil.copy(src, dst)
    
    src = mask
    dst = qsm_dirpath
    shutil.copy(src, dst)
    
    filename = os.path.basename(phase)
    
    if filename[-3:] == '.gz':
        filename = filename[:-7]
        phase_path = os.path.join(qsm_dirpath , filename + '.nii.gz')
    else:
        filename = filename[:-4]
        phase_path = os.path.join(qsm_dirpath , filename + '.nii')
    
    src = qsm_dirpath +'/'+ filename + 'QSM_map_000.nii.gz' 
    
    filename = os.path.basename(mask)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
        mask_path = os.path.join(qsm_dirpath , filename + '.nii.gz')
    else:
        filename = filename[:-4]
        mask_path = os.path.join(qsm_dirpath, filename + '.nii')
        
    
    output = 'QSM_map'
    
    subprocess.run(['tgv_qsm', '-p', phase_path, '-m', mask_path, '-o', output, '-f', str(f), '-t', str(t), '-s'])
    
    dst = os.path.split(phase)[0]
    
    shutil.copy(src, dst)
    
    shutil.rmtree(qsm_dirpath)
    

    

def fit_inversion_recovery(IR_data, TI_times, IE = 1.96, mask=None):
    
    idx = np.ones(np.shape(IR_data)[:-1])
    
    shape = np.shape(idx)
    
    def func(TI, a, T1):
        return np.abs(a*(1-IE*np.exp(-TI/T1)))
    
    idx = idx.ravel()
    IR_data_vector = IR_data.reshape(np.shape(idx) + (np.shape(IR_data)[-1], ))
    

    if mask is None:
        mask = np.ones(idx.shape)
    else:
        mask = mask.ravel()
        
    T1 = np.zeros(np.shape(idx))
    A = np.zeros(np.shape(idx))

    for i in range(np.size(idx)):

        if idx[i]:
            if mask[i]:
                popt, pcov = curve_fit(func, TI_times, IR_data_vector[i, :], bounds=([100, 100], [5000, 5000]))
                
                A[i] = popt[0]
                T1[i] = popt[1]
    
    A = A.reshape(shape)
    T1 = T1.reshape(shape) 
    

    return T1, A
    






def combine_echoes(arguments, nechoes, save_name = '_'):
    """
    

    Parameters
    ----------
    arguments : dict
        arguments should be a dict with the individual echoes named as ['se_echonumber_path]'
    nechoes : int
        DESCRIPTION.

    Returns
    -------
    path : string
        

    """
    
    
    file_list = list()
    
    for i in range(nechoes):
        
        file_list.append(arguments['se%i_path'%(i+1)])
        
    nii_list = dict()
    for i in range(nechoes):
        
        nii_list['%i'%i] = nib.load(file_list[i]).get_fdata()
    for i in range(nechoes):
        
        if i == 0:
            multiple_echoes = nii_list['%i'%i][:,:,:,None]
            
        else: 
            multiple_echoes = np.concatenate([multiple_echoes, 
                                              nii_list['%i'%i][:,:,:,None]], axis=3)
        
    affine = nib.load(file_list[0]).affine
            
    nii = nib.Nifti1Image(multiple_echoes, affine = affine)
    
    dst = os.path.split(file_list[0])[0]
    
    nib.save(nii, dst + '/'+ save_name + 'Combined_%i_echoes.nii.gz'%nechoes)
    
    path = dst + '/' + save_name +'Combined_%i_echoes.nii.gz'%nechoes
    
    return path
    


def complex_interpolation(mag_path, pha_path, interpol2shape = [216, 224, 172], 
                          save_name = "resampled_1mm_iso"):
    """
    

    Parameters
    ----------
    mag_path : path-like
        DESCRIPTION. Path to the mag image that is to be interpolated
    pha_path : path-like
        DESCRIPTION. Path to the phase image that is to be interpolated
    interpol2shape : list, optional
        DESCRIPTION. list of the final image size to be interpolated to. 
        The default is [216, 224, 172].
    save_name : string, optional
        DESCRIPTION. The default is "resampled_1mm_iso".

    Returns
    -------
    None.

    """
    
    path = os.path.split(mag_path)[0]
    os.chdir(path)
    print("Complex interpolation of the mag and phase data")
    gre1_mag = nib.load(mag_path).get_fdata()
    
    gre1_phase = nib.load(pha_path).get_fdata()
    gre1_phase_rescaled = gre1_phase * np.pi/4096
    
    gre1_complex_rescaled = gre1_mag*np.exp(1j*gre1_phase_rescaled)
    real = np.real(gre1_complex_rescaled)
    imag = np.imag(gre1_complex_rescaled)
    
    real_resampled = resample(real, num=216, axis = 0)
    real_resampled = resample(real_resampled, num=224, axis = 1)
    real_resampled = resample(real_resampled, num=172, axis = 2)

    imag_resampled = resample(imag, num=216, axis = 0)
    imag_resampled = resample(imag_resampled, num=224, axis = 1)
    imag_resampled = resample(imag_resampled, num=172, axis = 2)

    G = real_resampled + 1j*imag_resampled
    gre1_mag_resampled = np.sqrt(real_resampled**2 + imag_resampled**2)
    gre1_phase_resampled = np.angle(G)
    gre1_phase_resampled = gre1_phase_resampled*4096/np.pi


    affine = gre1_mag = nib.load(mag_path).affine
        
    if affine[0,2]>0:
        affine[0,2] = 1.0
    else:
        affine[0,2] = -1.0

    if affine[1,0]>0:
        affine[1,0] = 1.0
    else:
        affine[1,0] = -1.0

    if affine[2,1]>0:
        affine[2,1] = 1.0
    else:
        affine[2,1] = -1.0

    nii_1_2mm_resampled_mag = nib.Nifti1Image(gre1_mag_resampled, affine=affine)
    nii_1_2mm_resampled_phase = nib.Nifti1Image(gre1_phase_resampled, affine=affine)
    nib.save(nii_1_2mm_resampled_mag, '%s'%save_name + '_mag.nii.gz')
    nib.save(nii_1_2mm_resampled_phase, '%s'%save_name + '_pha.nii.gz')
    
    return path + '/' + '%s'%save_name + '_mag.nii.gz', path + '/' + '%s'%save_name + '_pha.nii.gz'
def fsl_EPI_distortion_corr(file_path, B0map_path, mydwell = 0.0005, phasedir='y'):
    """
    
    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.
    B0map_path : TYPE
        DESCRIPTION.
    mydwell : TYPE, optional
        DESCRIPTION. The default is 0.0005.
    phasedir : TYPE, optional
        DESCRIPTION. The default is 'y'.

    Returns
    -------
    output_name : TYPE
        DESCRIPTION.

    """
    FSL_dirpath = os.path.join(cwd, "FSL_temp")
    
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)
    os.mkdir(FSL_dirpath)

    
    src = file_path
    dst = os.path.join(cwd, "FSL_temp")
 
    shutil.copy(src, dst)
    
    src = B0map_path
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
   
    filename = os.path.basename(file_path)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
     
    input_path = os.path.join(FSL_dirpath , filename)
        
    
    output_name = input_path + '_Corr' + '.nii.gz'
    
    filename = os.path.basename(B0map_path)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
    
    B0_path = os.path.join(FSL_dirpath , filename)
    
    
    
    subprocess.run(['fugue','-i', input_path,'--icorr','--dwell={}'.format(mydwell),'--unwarpdir={}'.format(phasedir),'--loadfmap={}'.format(B0_path),
    '-u', output_name],check=True)
    
    filename = os.path.basename(file_path)
    if filename[-3:] == '.gz':
        filename = filename[:-7]
    else:
        filename = filename[:-4]
        
    
    for file in Path(os.path.join(cwd, "FSL_temp")).glob('%s*'%filename):
        print('transferred %s' %file)
        src = file
        dst = os.path.split(file_path)[0]
        shutil.copy(src, dst)
        
    shutil.rmtree(FSL_dirpath)
    
    return output_name
    

def fsl_linear_normalization(moving_nii, fixed_nii, additional_images_list=None, dof=6):
    FSL_dirpath = os.path.join(cwd, "FSL_temp")
    
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)
    os.mkdir(FSL_dirpath)

    
    src = moving_nii
    dst = os.path.join(cwd, "FSL_temp")
 
    shutil.copy(src, dst)
    
    src = fixed_nii
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    
    filename1 = os.path.basename(moving_nii)
    if filename1[-3:] == '.gz':
        filename1 = filename1[:-7]
    else:
        filename1 = filename1[:-4]
    
    input_path = os.path.join(FSL_dirpath , filename1)
    
    
    filename2 = os.path.basename(fixed_nii)
    if filename2[-3:] == '.gz':
        filename2 = filename2[:-7]
    else:
        filename2 = filename2[:-4]
    
    # FSL orient2std
    
    out_path = os.path.join(FSL_dirpath , filename1 + 'std')
    
    
    subprocess.run(['fslreorient2std',input_path,out_path],check=True)
    if additional_images_list is not None:
        
        for i in additional_images_list:
            
            src = i
            dst = os.path.join(cwd, "FSL_temp")
            shutil.copy(src, dst)
            
            filename_additional = os.path.basename(i)
            if filename_additional[-3:] == '.gz':
                filename_additional = filename_additional[:-7]
            else:
                filename_additional = filename_additional[:-4]
            
            input_path_additional = os.path.join(FSL_dirpath , filename_additional)
            
            out_path_additional = os.path.join(FSL_dirpath , filename_additional + 'std')
            
            
            subprocess.run(['fslreorient2std',input_path_additional, out_path_additional], check=True)        
            
    input_path = out_path
    
    filename1 = filename1 + 'std'
    
    ref_path = os.path.join(FSL_dirpath , filename2)
    
    out_path = os.path.join(FSL_dirpath , filename1 + '_2_' + filename2)
    
    omat_path = os.path.join(FSL_dirpath , filename1 + '_2_' + filename2 + '.txt')
    
    dof = int(dof)
    
    # Come on lets flirt:
    subprocess.run(['flirt','-in',input_path,'-ref',ref_path,'-out',out_path,'-omat',omat_path,'-dof',dof],check=True)
   
    
    for file in Path(os.path.join(cwd, "FSL_temp")).glob('*'):
        print('transferred %s' %file)
        src = file
        dst = os.path.split(moving_nii)[0]
        shutil.copy(src, dst)
        
    shutil.rmtree(FSL_dirpath)   
    
    # Additional images
    
    if additional_images_list is not None:
        
        for i in additional_images_list:

            filename_additional = os.path.basename(i)
            if filename_additional[-3:] == '.gz':
                filename_additional = filename_additional[:-7]
            else:
                filename_additional = filename_additional[:-4]
            
            dst = os.path.split(moving_nii)[0]
            moving_nii_additional = dst + '/' + filename_additional + 'std' + '.nii.gz'
            
            filename_affine = os.path.basename(omat_path)
                
            dst = os.path.split(moving_nii)[0]
            
            affine_guess = dst + '/' + filename_affine

            fsl_flirt_applyxfm(moving_nii_additional, fixed_nii, matrix = affine_guess)
        
    
    return os.path.basename(omat_path)




def fsl_NON_linear_normalization(moving_nii, fixed_nii, affine_guess, additional_images_list=None):
    
    
    FSL_dirpath = os.path.join(cwd, "FSL_temp")
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)
    
    os.mkdir(FSL_dirpath);
    
    src = moving_nii
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    src = fixed_nii
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    src = affine_guess
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)
    
    filename1 = os.path.basename(moving_nii)
    if filename1[-3:] == '.gz':
        filename1 = filename1[:-7]
    else:
        filename1 = filename1[:-4]
    
    input_path = os.path.join(FSL_dirpath , filename1)
    
    
    filename2 = os.path.basename(fixed_nii)
    if filename2[-3:] == '.gz':
        filename2 = filename2[:-7]
    else:
        filename2 = filename2[:-4]
        
    filename3 = os.path.basename(affine_guess)
                    
    ref_path = os.path.join(FSL_dirpath ,filename2)
    
    out_path = os.path.join(FSL_dirpath , filename1 + '_2_' + filename2 + '_my_nonlinear_transf')
    
    affine_path = os.path.join(FSL_dirpath , filename3)
    
    
    # Come on lets FNIRT:
    subprocess.run(['fnirt','--ref=',ref_path,'--in=',input_path,'--aff',affine_path,'--cout=',out_path],check=True)
    
    
    warp_path = out_path
    
    out_path = os.path.join(FSL_dirpath , filename1 + '_2_' + filename2 + '_Non_Linear_Normalized')
    subprocess.run(['applywarp','--in=',input_path,'--ref=',ref_path,'--out=',out_path,'--warp=',warp_path],check=True)
    
    
    if additional_images_list is not None:
        
        for i in additional_images_list:
            
            src = i
            dst = os.path.join(cwd, "FSL_temp")
            shutil.copy(src, dst)
            
            filename_additional = os.path.basename(i)
            if filename_additional[-3:] == '.gz':
                filename_additional = filename_additional[:-7]
            else:
                filename_additional = filename_additional[:-4]
            
            input_path_additional = os.path.join(FSL_dirpath , filename_additional)
            
            out_path_additional = os.path.join(FSL_dirpath , filename_additional +  '_2_' + filename2 + '_Non_Linear_Normalized')
            subprocess.run(['applywarp','--in=',input_path_additional,'--ref=',ref_path,'--out=',out_path_additional,'--warp=',warp_path],check=True)
            
   
    for file in Path(os.path.join(cwd, "FSL_temp")).glob('*'):
        print('transferred %s' %file)
        src = file
        dst = os.path.split(moving_nii)[0]
        shutil.copy(src, dst)
    
    shutil.rmtree(FSL_dirpath)   
    
        
    return os.path.basename(out_path)






def fsl_NON_linear_normalization_AFTER_FLIRT(moving_nii, fixed_nii, additional_images_list=None, dof=6):
    """
    

    Parameters
    ----------
    moving_nii : TYPE
        DESCRIPTION.
    fixed_nii : TYPE
        DESCRIPTION.
    additional_images_list : TYPE, optional
        DESCRIPTION. The default is None.
    dof : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    None.

    """
    
    
    omat_path = fsl_linear_normalization(moving_nii, fixed_nii, additional_images_list=additional_images_list, dof=6)
    
    filename1 = os.path.basename(moving_nii)
    if filename1[-3:] == '.gz':
        filename1 = filename1[:-7]
    else:
        filename1 = filename1[:-4]
        
    dst = os.path.split(moving_nii)[0]
    
    moving_nii = dst + '/' + filename1 + 'std' + '.nii.gz'
    
    filename_affine = os.path.basename(omat_path)

    dst = os.path.split(moving_nii)[0]
    
    affine_guess = dst + '/' + filename_affine
    
    
    if additional_images_list is not None:
        index = 0
        for i in additional_images_list:

            filename_additional = os.path.basename(i)
            if filename_additional[-3:] == '.gz':
                filename_additional = filename_additional[:-7]
            else:
                filename_additional = filename_additional[:-4]
            
            dst = os.path.split(moving_nii)[0]
            name = dst + '/' + filename_additional + 'std' + '.nii.gz'
            
            additional_images_list[index] = name
            index = index + 1 
            
    fsl_NON_linear_normalization(moving_nii, fixed_nii, affine_guess, additional_images_list=additional_images_list)
