import os
import shutil
import nibabel as nib
import numpy as np
cwd= os.getcwd()
# Helper function to normalize phase
def normalize_phase(phase_image_path):
    """
    Normalizes the phase image to the range [-pi, pi].

    Parameters
    ----------
    phase_image_path : str
        Path to the phase NIfTI image.

    Returns
    -------
    normalized_path : str
        Path to the normalized phase NIfTI image.
    """
    phase_data = nib.load(phase_image_path)
    phase = phase_data.get_fdata()

    # Normalize phase
    phase_normalized = (phase - phase.min()) / (phase.max() - phase.min()) * (2 * np.pi) - np.pi

    # Save the normalized phase image
    normalized_path = phase_image_path.replace(".nii.gz", "_normalized.nii.gz")
    nib.save(nib.Nifti1Image(phase_normalized, phase_data.affine, phase_data.header), normalized_path)

    return normalized_path


def fsl_brain_masking(file_path):
    """
    Brain masking using FSL's BET tool.

    Parameters
    ----------
    file_path : str
        Path to the input NIfTI file to be brain masked.

    Returns
    -------
    mask_path : str
        Path to the binary brain mask file.
    """
    FSL_dirpath = os.path.join(cwd, "FSL_temp")
    if os.path.exists(FSL_dirpath):
        shutil.rmtree(FSL_dirpath)

    os.makedirs(FSL_dirpath)
    src = file_path
    dst = os.path.join(cwd, "FSL_temp")
    shutil.copy(src, dst)

    filename = os.path.basename(file_path)
    if filename.endswith('.gz'):
        filename = filename[:-7]
    else:
        filename = filename[:-4]
    wsl_cwd = "/mnt/" + cwd[0].lower() + cwd[2:].replace("\\", "/") + "/FSL_temp/"
    
    input_path = wsl_cwd + filename
    output_path = wsl_cwd + filename + '_brain'

    # Run FSL's BET tool
    os.system(f'wsl -e bash -lic "bet2 {input_path} {output_path} -f 0.5 -g 0 -m"')

    # Move mask back to the original directory
    final_mask_path = os.path.join(os.path.dirname(file_path), filename + '_brain_mask.nii.gz')
    shutil.copy(os.path.join(FSL_dirpath, filename + '_brain_mask.nii.gz'), final_mask_path)

    shutil.rmtree(FSL_dirpath)
    
    return final_mask_path

def tgv_qsm(phase_image, mask_image, te, field_strength):
    """
    Runs TGV-QSM on a phase image.

    Parameters
    ----------
    phase_image : str
        Path to the phase NIfTI image.
    mask_image : str
        Path to the brain mask NIfTI image.
    te : float
        Echo time in seconds.
    field_strength : float
        Magnetic field strength in Tesla.

    Returns
    -------
    qsm_path : str
        Path to the generated QSM map.
    """
    input_dir = os.path.dirname(phase_image)
    qsm_dirpath = os.path.join(input_dir, "QSM_temp")
    if os.path.exists(qsm_dirpath):
        shutil.rmtree(qsm_dirpath)
    os.makedirs(qsm_dirpath)
    
    shutil.copy(phase_image, qsm_dirpath)
    shutil.copy(mask_image, qsm_dirpath)
    
    wsl_cwd = "/mnt/" + input_dir[0].lower() + input_dir[2:].replace("\\", "/") + "/QSM_temp/"
    phase_filename = os.path.basename(phase_image)
    mask_filename = os.path.basename(mask_image)
    
    phase_path = os.path.join(wsl_cwd, phase_filename)
    mask_path = os.path.join(wsl_cwd, mask_filename)
    output = "_QSM_map"

    os.system(f'wsl -e bash -lic "tgv_qsm -p {phase_path} -m {mask_path} -o {output} -f {field_strength} -t {te}"')

    qsm_filename = os.path.basename(phase_image).replace(".nii.gz", "_QSM_map_000.nii.gz")
    final_output_path = os.path.join(input_dir, qsm_filename)
    shutil.copy(os.path.join(qsm_dirpath, qsm_filename), final_output_path)

    shutil.rmtree(qsm_dirpath)
    return final_output_path

def calculate_qsm_average(dirpath, filename, mask_path, echo_indices, echo_times):
    """
    Calculate the QSM average map using specified echoes.

    Parameters:
    - dirpath: Path to the directory containing input data.
    - filename: Base name of the input files.
    - mask_path: Path to the brain mask.
    - echo_indices: List of indices of echoes to process.
    - echo_times: List of echo times corresponding to the input indices.

    Returns:
    - avg_path: Path to the saved averaged QSM map.
    """
    avg = None
    for i, echo_index in enumerate(echo_indices):
        phase_image_echo = os.path.join(dirpath, f'{filename}_e{echo_index}_ph_normalized.nii.gz')
        qsm_path = tgv_qsm(phase_image_echo, mask_path, te=echo_times[echo_index - 1], field_strength=3.0)
        qsm_data = nib.load(qsm_path).get_fdata()

        avg = qsm_data if avg is None else avg + qsm_data

    avg /= len(echo_indices)
    affine = nib.load(qsm_path).affine
    avg_nii = nib.Nifti1Image(avg, affine)
    avg_path = os.path.join(dirpath, f'{filename}_QSM_avg.nii.gz')
    nib.save(avg_nii, avg_path)

    return avg_path

# Main script
def process_qsm_pipeline(phase_images, echo_times,T1w_mag, field_strength=3.0):
    """
    Full QSM pipeline including phase normalization and averaging.

    Parameters
    ----------
    phase_images : list of str
        List of paths to phase NIfTI images.
    echo_times : list of float
        List of echo times in seconds.
    field_strength : float
        Magnetic field strength in Tesla.
    """
    normalized_paths = []
    for phase_image in phase_images:
        normalized_path = normalize_phase(phase_image)
        normalized_paths.append(normalized_path)

    mask_path = fsl_brain_masking(T1w_mag)
    avg_qsm_path = calculate_qsm_average(
        dirpath=os.path.dirname(phase_images[0]),
        filename=os.path.basename(phase_images[0]).split('_e')[0],
        mask_path=mask_path,
        echo_indices=list(range(3, 7)),
        echo_times=echo_times
    )
    print(f"Final averaged QSM map saved at: {avg_qsm_path}")

# Use it on philips data
if __name__ == "__main__":
    phase_images = [r"C:\Users\mmari\Desktop\Philips_Data_and_Codes\Processing\DICOM_3D_nsFFE_FA32_synergy_1_2mm_20241120165729_201_e1_ph.nii.gz", r"C:\Users\mmari\Desktop\Philips_Data_and_Codes\Processing\DICOM_3D_nsFFE_FA32_synergy_1_2mm_20241120165729_201_e2_ph.nii.gz", r"c:\Users\mmari\Desktop\Philips_Data_and_Codes\Processing\DICOM_3D_nsFFE_FA32_synergy_1_2mm_20241120165729_201_e3_ph.nii.gz", 
                    r"C:\Users\mmari\Desktop\Philips_Data_and_Codes\Processing\DICOM_3D_nsFFE_FA32_synergy_1_2mm_20241120165729_201_e4_ph.nii.gz", r"c:\Users\mmari\Desktop\Philips_Data_and_Codes\Processing\DICOM_3D_nsFFE_FA32_synergy_1_2mm_20241120165729_201_e5_ph.nii.gz", r"c:\Users\mmari\Desktop\Philips_Data_and_Codes\Processing\DICOM_3D_nsFFE_FA32_synergy_1_2mm_20241120165729_201_e6_ph.nii.gz"]
    echo_times = np.linspace(3.2, 25.70, num=6)/ 1000  # Check if these are still valid for Philips data

    T1w_mag=r"C:\Users\mmari\Desktop\Philips_Data_and_Codes\Processing\T1w_magscaling_CORR.nii.gz"
    process_qsm_pipeline(phase_images, echo_times,T1w_mag)
