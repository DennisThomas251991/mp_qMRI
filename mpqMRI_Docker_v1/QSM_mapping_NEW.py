import os
import shutil
import numpy as np
import nibabel as nib
from Useful_functions import fsl_brain_masking, threshold_masking, tgv_qsm2, split_all_echoes, tgv_qsm_Philips

class QSM_mapping_mpqMRI():

    def __init__(self, arguments):
        self.arguments = arguments

    def run(self, gre='gre2'):
        if gre == 'gre2':
            
            dst = os.path.split(self.arguments.phase_image2)[0]
            dirpath = os.path.join(dst, 'QSM')
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
            os.makedirs(dirpath,exist_ok=True)
            
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
            os.makedirs(dirpath,exist_ok=True)
            
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
            if self.arguments.Vendor == 'Siemens':
                tgv_qsm2(phase_image_echo, mask_path, t, f=3.0)
            elif self.arguments.Vendor == 'Philips':
                # Normalize phase images
                normalized_phase_image = self.normalize_phase(phase_image_echo)

                tgv_qsm_Philips(normalized_phase_image, mask_path, t, f=3.0)
            
        for i in range(len(self.arguments.QSM_average_echoes_qsm)):
            
            nifti = dict()
            if self.arguments.Vendor == 'Siemens':
                nifti['%i'%i] = nib.load(dirpath + '/'+ filename + 
                                        '_%i_echoQSM_map_000.nii.gz'%self.arguments.QSM_average_echoes_qsm[i]).get_fdata()
            elif self.arguments.Vendor == 'Philips':
                nifti['%i'%i] = nib.load(dirpath + '/'+ filename +
                                 '_%i_echo_normalizedQSM_map_000.nii.gz'%self.arguments.QSM_average_echoes_qsm[i]).get_fdata()
            if i == 0:
                avg = nifti['0']
            else: 
                avg = avg + nifti['%i'%i]
                
        avg = avg/len(self.arguments.QSM_average_echoes_qsm)
        
        # Set NaN values to 0
        avg[np.isnan(avg)] = 0
        if self.arguments.Vendor == 'Siemens':
                affine = nib.load(dirpath + '/'+ filename+
                                 '_%i_echoQSM_map_000.nii.gz'%self.arguments.QSM_average_echoes_qsm[i]).affine
        elif self.arguments.Vendor == 'Philips':
                affine = nib.load(dirpath + '/'+ filename + '_normalized'+
                                 '_%i_echo_normalizedQSM_map_000.nii.gz'%self.arguments.QSM_average_echoes_qsm[i]).affine
       
                
        nii_avg = nib.Nifti1Image(avg, affine = affine)
        nib.save(nii_avg, dirpath + '/'+ 'QSM_avg_map.nii.gz' )
        # Copy everything back to the main folder
        for file_name in os.listdir(dirpath):
            shutil.copy(os.path.join(dirpath, file_name), dst)
        shutil.rmtree(dirpath)
        return avg

    def brain_masking(self):
        if not self.arguments.phantom:
            fsl_brain_masking(self.arguments.gre2_path);
            filename = os.path.basename(self.arguments.gre2_path)
            filename = filename[:-7] if filename.endswith('.nii.gz') else filename[:-4]
            dst = os.path.split(self.arguments.gre2_path)[0]
            mask_path = os.path.join(dst, f"{filename}_brain_mask.nii.gz")
            mask = nib.load(mask_path).get_fdata();
            self.mask = mask
        else:
            mask = threshold_masking(self.arguments.gre1_path, self.arguments.Phantom_mask_threshold_T1mapping);
            self.mask = mask
        return mask_path, filename

    def normalize_phase(self, phase_image_path):
        phase_data = nib.load(phase_image_path)
        phase = phase_data.get_fdata()

        # Normalize phase
        phase_normalized = (phase - phase.min()) / (phase.max() - phase.min()) * (2 * np.pi) - np.pi

        # Save the normalized phase image
        normalized_path = phase_image_path.replace(".nii.gz", "_normalized.nii.gz")
        nib.save(nib.Nifti1Image(phase_normalized, phase_data.affine, phase_data.header), normalized_path)

        return normalized_path

