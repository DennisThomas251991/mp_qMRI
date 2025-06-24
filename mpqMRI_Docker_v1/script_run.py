# -*- coding: utf-8 -*-

"""
mp-qMRI Mapping Pipeline - Main Script
======================================

This script is the main entry point for the mp-qMRI Docker pipeline. It processes quantitative MRI data 
to generate T1, T2*, H2O, and QSM maps from DICOM or NIfTI images. The script supports both Siemens and Philips vendors 
and allows for extensive customization of processing parameters via command-line arguments or interactive prompts.

Features:
---------
- Converts DICOM to NIfTI (if needed) and organizes input data.
- Performs B0 and B1 mapping, T1 mapping (linear and non-linear), T2* mapping, H2O mapping, and optional QSM mapping.
- Supports both "Fast EPI" and "Volz 2010" B1 mapping methods.
- Allows parameter customization via command-line or interactive mode.
- Organizes outputs into dedicated folders and generates a summary info file.

Typical Usage:
--------------
Build the Docker image:
    docker build -t mpqmri-pipeline .

Run the pipeline (replace /path/to/input with your data folder):
    docker run --rm -v /path/to/input:/data mpqmri-pipeline --input-folder /data

For help and full argument list:
    docker run --rm mpqmri-pipeline --help

Outputs:
--------
- Final quantitative maps are saved in 'Final_qMRI_maps' inside the output folder.
- Intermediate files are moved to 'intermediate_files'.
- A 'map_info.txt' file describes the output maps.
- An 'info.txt' file logs the parameters used for processing.

See help.txt for more details on input requirements and options.
"""

import os
import sys
import argparse
import glob
import shutil
import T2smapping
import T1mapping_mpqMRI
import H2Omapping_mpqMRI
import B1mapping
import B0_mapping
import preprocessing 
import QSM_mapping_NEW
import preprocessing_Siemens
import numpy as np
import nibabel as nib
from Useful_functions import complex_interpolation, reorient_image
from pathlib import Path
# Argument parser for command-line usage
def parse_args():
    parser = argparse.ArgumentParser(description='Process mapping arguments.')
    parser.add_argument('--input-folder', required=True, help='Path to the input folder containing files to be processed')
    parser.add_argument('--Input_data_type', choices=['Nifti', 'DICOM'], default='DICOM', help='Input data type')
    parser.add_argument('--Philips_acquisition_method', choices=['Combined FA acquisition', 'Separate FA acquisition'], default='Combined FA acquisition', help='FA acquisition type')
    parser.add_argument('--Vendor', choices=['Siemens', 'Philips'], default='Siemens', help='Vendor of the MRI machine used to acquire the data')
    parser.add_argument('--Bias_field_corr_method', choices=['SPM', 'FSL'], default='SPM', help='Bias Field correction method')
    parser.add_argument('--FA1', type=int, default=5, help='Flip angle 1')
    parser.add_argument('--FA2', type=int, default=32, help='Flip angle 2')
    parser.add_argument('--TR1', type=int, default=30, help='TR1 [ms]')
    parser.add_argument('--TR2', type=int, default=30, help='TR2 [ms]')
    parser.add_argument('--nTE', type=int, default=2, help='Number of echo times')
    parser.add_argument('--x0', type=int, default=1000, help='x0 value')
    parser.add_argument('--phantom', type=bool, default=False, help='Use phantom')
    parser.add_argument('--masking', type=bool, default=True, help='Apply masking')
    parser.add_argument('--b1plus_mapping', type=bool, default=True, help='Perform B1+ mapping')
    parser.add_argument('--corr_for_imperfect_spoiling', type=bool, default=True, help='Correct for imperfect spoiling')
    parser.add_argument('--Phantom_mask_threshold_B1mapping', type=int, default=100, help='Threshold for B1 mapping phantom mask')
    parser.add_argument('--Phantom_mask_threshold_T1mapping', type=int, default=100, help='Threshold for T1 mapping phantom mask')
    parser.add_argument('--spoil_increment', type=int, default=50, help='Spoil increment')
    parser.add_argument('--slices', default=all, help='Slices')
    parser.add_argument('--echotimes', nargs='+', type=float, default=np.linspace(3.20, 25.70, num=6), help='Echo times')
    parser.add_argument('--QSM_average_echoes_qsm', nargs='+', type=int, default=[3, 4, 5, 6], help='Echoes for QSM averaging')
    parser.add_argument('--coregister_mGREs', type=bool, default=False, help='Coregister mGREs')
    parser.add_argument('--B1map_orientation', choices=['Sag', 'Tra'], default='Tra', help='Orientation of the B1 map')
    parser.add_argument('--complex_interpolation', type=bool, default=False, help='Use complex interpolation')
    parser.add_argument('--B1_mapping_method', choices=['Fast EPI', 'Volz 2010'], default='Fast EPI', help='B1 mapping method')
    parser.add_argument('--QSM_mapping', type=bool, default=True, help='Perform QSM mapping')
    return parser.parse_args()

# Helper for interactive input with default
def user_input_with_default(prompt, default):
    user_input = input(prompt)
    if user_input:
        return user_input
    else:
        return default

# Show help.txt if --help is passed
def show_help():
    help_file_path = os.path.join(os.path.dirname(__file__), 'help.txt')
    with open(help_file_path, 'r') as file:
        print(file.read())
    sys.exit(0)

if '--help' in sys.argv:
    show_help()

# Parse arguments
arguments = parse_args()

# Print default parameters for user reference
print('Default parameters are currently set to:')
print('Input_data_type = DICOM')
print('Acquisition_method = Combined FA acquisition')
print('Vendor = Siemens')
print('Bias_field_corr_method = SPM')
print('FA1 = 5')
print('FA2 = 32')
print('TR1 = 30')
print('TR2 = 30')
print('nTE = 2')
print('x0 = 1000')
print('phantom = False')
print('masking = True')
print('b1plus_mapping = True')
print('corr_for_imperfect_spoiling = True')
print('Phantom_mask_threshold_B1mapping = 100')
print('Phantom_mask_threshold_T1mapping = 100')
print('spoil_increment = 50')
print('slices = all')
print('echotimes [ms] = 3.2, 7.7, 12.2, 16.7, 21.2, 25.7')
print('QSM_average_echoes_qsm = [3, 4, 5, 6]')
print('coregister_mGREs = False')
print('B1map_orientation = Tra')
print('complex_interpolation = False')
print('B1_mapping_method = Fast EPI')
print('QSM_mapping = True ')

# Optionally allow user to change parameters interactively
answer = input('Do you want to change the input parameters line by line? If yes, type "yes" if not you can click ENTER: ')
if answer.lower() == 'yes':
    print('If you do not want to change an argument click ENTER ')
    arguments.Input_data_type = user_input_with_default('Input_data_type (Input data type): Choices are "Nifti" or "DICOM": ', 'DICOM')
    arguments.Vendor = user_input_with_default('Vendor : Choices are "Siemens" or "Philips": ', 'Siemens')
    if arguments.Vendor == 'Philips':
        arguments.Philips_acquisition_method = user_input_with_default('Philips_acquisition_method (FA acquisition type): Choices are "Combined FA acquisition" or "Separate FA acquisition": ', 'Combined FA acquisition')
    arguments.Bias_field_corr_method = user_input_with_default('Bias_field_corr_method (Bias Field correction method): Choices are "SPM" or "FSL": ', 'SPM')
    arguments.FA1 = int(user_input_with_default('FA1 (Flip angle 1) [\u00b0]: ', '5'))
    arguments.FA2 = int(user_input_with_default('FA2 (Flip angle 2)  [\u00b0]: ', '32'))
    arguments.TR1 = int(user_input_with_default('TR1 [ms]: ', '30'))
    arguments.TR2 = int(user_input_with_default('TR2 [ms]: ', '30'))
    arguments.nTE = int(user_input_with_default('nTE (Number of echo times to be used for T1 mapping): ', '2'))
    arguments.x0 = int(user_input_with_default('x0 value: ', '1000'))
    arguments.phantom = user_input_with_default('Phantom: (Use Phantom? Answer with True/False): ', 'False').lower() == 'true'
    arguments.masking = user_input_with_default('masking: (Apply masking? Answer with True/False): ', 'True').lower() == 'true'
    arguments.b1plus_mapping = user_input_with_default('b1plus_mapping: (Perform B1+ mapping? Answer with True/False): ', 'True').lower() == 'true'
    arguments.corr_for_imperfect_spoiling = user_input_with_default('corr_for_imperfect_spoiling: (Correct for imperfect spoiling? Answer with True/False): ', 'True').lower() == 'true'
    arguments.Phantom_mask_threshold_B1mapping = int(user_input_with_default('Phantom_mask_threshold_B1mapping: (Threshold for B1 mapping phantom mask): ', '100'))
    arguments.Phantom_mask_threshold_T1mapping = int(user_input_with_default('Phantom_mask_threshold_T1mapping: (Threshold for T1 mapping phantom mask): ', '100'))
    arguments.spoil_increment = int(user_input_with_default('Spoil increment: ', '50'))
    arguments.slices = user_input_with_default('Slices: ', all)
    echo = input('would you like to change the echotimes? Answer with yes or no: ') == 'yes'
    if echo:
        arguments.T0 = float(user_input_with_default('Initial echo time: ', '3.20'))
        arguments.delta_t = float(user_input_with_default('Time difference between echoes: ', '4.15'))
        arguments.num_echoes = int(user_input_with_default('Number of echoes: ', '6'))
        # Calculate echo times based on user input
        arguments.echotimes = [arguments.T0 + i * arguments.delta_t for i in range(arguments.num_echoes)]
    arguments.QSM_average_echoes_qsm = [int(e) for e in user_input_with_default('QSM_average_echoes_qsm: (Echoes for QSM averaging (comma-separated)): ', '3, 4, 5, 6').split(',')]
    arguments.coregister_mGREs = user_input_with_default('Coregister mGREs? Answer with True/False: ', 'False').lower() == 'true'
    arguments.B1map_orientation = user_input_with_default('B1map_orientation: (Orientation of the B1 map: Choices are "Sag" and "Tra"): ', 'Tra')
    arguments.complex_interpolation = user_input_with_default('complex_interpolation: (Use complex interpolation? Answer with True/False): ', 'False').lower() == 'true'
    arguments.B1_mapping_method = user_input_with_default('B1_mapping_method: (B1 mapping method: choices are "Fast EPI" and "Volz 2010"): ', 'Fast EPI')
    arguments.QSM_mapping = user_input_with_default('QSM_mapping: (Would you like to do QSM mapping? Answer with True/False)', 'True').lower() == 'true'

# Save original input folder path
original_input_folder = arguments.input_folder

# Check if the input folder exists
if not os.path.exists(original_input_folder):
    print("Input folder does not exist.")
    sys.exit(1)

# Function to write arguments to a log file
def write_log_file(arguments, original_input_folder, file_path):
    with open(file_path, 'w') as file:
        file.write(f'input_folder: {original_input_folder}\n')
        for key, value in vars(arguments).items():
            file.write(f'{key}: {value}\n')

# Preprocessing depending on input data type and vendor
if arguments.Input_data_type == 'DICOM':
    if arguments.Vendor == 'Philips': 
        preprocessing.run(original_input_folder,arguments.Philips_acquisition_method)
    elif arguments.Vendor == 'Siemens':
        preprocessing_Siemens.run(original_input_folder)
    # Output folder for NIFTI files
    output_folder_path = Path(original_input_folder) / 'Niftis'
else: 
    output_folder_path = Path(original_input_folder)

# Write arguments to a text file in the input folder
log_file_path = os.path.join(output_folder_path, 'info.txt')
write_log_file(arguments, output_folder_path, log_file_path)

# Define the expected file names for further processing
expected_files = ['M0w_mag', 'T1w_mag', 'M0w_pha', 'T1w_pha', 'B1_1', 'B1_2']
file_paths = {}

# Search for expected files in the directory (both .nii and .nii.gz)
for file_name in expected_files:
    nii_file = glob.glob(os.path.join(output_folder_path, f"{file_name}.nii"))
    nii_gz_file = glob.glob(os.path.join(output_folder_path, f"{file_name}.nii.gz"))
    if nii_file:
        file_paths[file_name] = nii_file[0]
    elif nii_gz_file:
        file_paths[file_name] = nii_gz_file[0]

# Check if all expected files are found
if len(file_paths) != len(expected_files):
    missing_files = set(expected_files) - set(file_paths.keys())
    raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")

# Assign file paths to arguments for further processing
for key, value in file_paths.items():
    if key.startswith('M0w_mag'):
        arguments.gre1_path = value
    elif key.startswith('M0w_pha'):
        arguments.phase_image1 = value
    elif key.startswith('T1w_mag'):
        arguments.gre2_path = value
    elif key.startswith('T1w_pha'):
        arguments.phase_image2 = value
    elif key.startswith('B1_1'):
        arguments.path_b1_noprep = value
    elif key.startswith('B1_2'):
        arguments.path_b1_prep = value

# Set B1 mapping method-specific parameters
if arguments.B1_mapping_method == 'Fast EPI':
    arguments.b1map_filename = 'B1_MAP_from_fast_EPI_standard'
    arguments.path_epi_45 = arguments.path_b1_noprep 
    arguments.path_epi_90 = arguments.path_b1_prep
elif arguments.B1_mapping_method == 'Volz 2010':
    arguments.b1map_filename = 'standard_B1_MAP_smoothed'

###############################################################################
# Preprocessing and mapping steps

# Prepare flip angles and GRE images
FA1 = np.deg2rad(arguments.FA1)
FA2 = np.deg2rad(arguments.FA2)
arguments.FA_list = [FA1, FA2]
gre1 = nib.load(arguments.gre1_path).get_fdata()
gre2 = nib.load(arguments.gre2_path).get_fdata()
arguments.affine = nib.load(arguments.gre2_path).affine
arguments.b1plus = np.ones(gre1.shape[:-1])
arguments.gre_list = [gre1, gre2]
arguments.TR_list = [arguments.TR1, arguments.TR1]

# Optional: complex interpolation of images
if arguments.complex_interpolation:
    mag_path = arguments.gre1_path
    pha_path = arguments.phase_image1
    save_name = "resampled_1mm_iso_FA%i" % arguments.FA1
    mag_interp_path, pha_interp_path = complex_interpolation(mag_path, pha_path, save_name=save_name)
    arguments.gre1_path = mag_interp_path
    arguments.phase_image1 = pha_interp_path
    
    mag_path = arguments.gre2_path
    pha_path = arguments.phase_image2
    save_name = "resampled_1mm_iso_FA%i" % arguments.FA2
    mag_interp_path, pha_interp_path = complex_interpolation(mag_path, pha_path, save_name=save_name)
    arguments.gre2_path = mag_interp_path
    arguments.phase_image2 = pha_interp_path
    gre1 = nib.load(arguments.gre1_path).get_fdata()
    gre2 = nib.load(arguments.gre2_path).get_fdata() 
    arguments.gre_list = [gre1, gre2]
    arguments.affine = nib.load(arguments.gre2_path).affine

###############################################################################

# B0 and B1 mapping
if arguments.B1_mapping_method == 'Fast EPI':
    # B0 mapping
    B0map_object = B0_mapping.B0_mapping(arguments)
    B0map_coreg2B1_path = B0map_object.run(gre='gre1', coregister_2_EPI=True)
    # B1 mapping
    B1map_object = B1mapping.B1_map_Dennis(arguments)
    B1map_coreg = B1map_object.produce_B1_maps(B0map_coreg2B1_path, coregister_2_T1=True)
elif arguments.B1_mapping_method == 'Volz 2010':
    # B1 mapping
    B1map_object = B1mapping.B1_map_Ralf(arguments)
    B1map_coreg = B1map_object.produce_B1_maps(coregister_2_T1=True)

# T1 mapping
T1_object = T1mapping_mpqMRI.T1_mapping_mpqMRI(arguments, B1map_coreg)
T1_map2 = T1_object.run_linear_approach(save=True)
T1_map = T1_object.run(save=True)

# T2* mapping
T2s_object = T2smapping.t2s_map_mpqMRI(arguments)
output = T2s_object.run(nechoes=6, save=True)
T2Star_gre12 = output[7]

# H2O mapping
H2O_object = H2Omapping_mpqMRI.H2O_map_mpqMRI(arguments, T1_map, B1map_coreg, T2Star_gre12)
H2O = H2O_object.run()
if arguments.Input_data_type == 'DICOM':
    for file in output_folder_path.glob("*.nii.gz"):
        reorient_image(file, file, ('x', '-y', 'z'))

# QSM mapping (optional)
if arguments.QSM_mapping == True:
    QSM_object = QSM_mapping_NEW.QSM_mapping_mpqMRI(arguments)
    QSM = QSM_object.run(gre='gre2')

# Organize output files into folders
def move_files(src_folder, dst_folder, except_files):
    os.makedirs(dst_folder, exist_ok=True)
    for item in os.listdir(src_folder):
        item_path = os.path.join(src_folder, item)
        if os.path.isfile(item_path) and item not in except_files:
            shutil.move(item_path, dst_folder)

# Define output and main files
output_files = [
    'T1_map_B1corr_True_Spoilcorr_True_2echoes.nii.gz',        
    'T2Star_avg.nii.gz',  
    'H2O.nii.gz',           
    'avg_T2Star.nii.gz',
    't1map_linear_approach.nii.gz',
    'QSM_avg_map.nii.gz'
]
main_maps = [
    'M0w_mag.nii.gz', 'M0w_pha.nii.gz', 'T1w_mag.nii.gz', 'T1w_pha.nii.gz', 'B1_1.nii.gz', 'B1_2.nii.gz'
]

# Paths for intermediary and output folders
intermediary_folder = os.path.join(output_folder_path, 'intermediate_files')
output_folder = os.path.join(output_folder_path, 'Final_qMRI_maps')

# Move all files except output and main maps to intermediary folder
move_files(output_folder_path, intermediary_folder, output_files + main_maps)

# Move output files to output folder
os.makedirs(output_folder, exist_ok=True)
for file_name in output_files:
    src_path = os.path.join(output_folder_path, file_name)
    if os.path.exists(src_path):
        shutil.move(src_path, output_folder)

# Write mapping info to a text file for user reference
map_info = {
    'T1_map_B1corr_True_Spoilcorr_True_2echoes.nii': 'T1 map non-linear approach',
    't1map_linear_approach.nii': 'T1 map linear approach',
    'avg_T2Star.nii': 'T2* map of the average of gre1 and gre2',
    'T2Star_avg.nii': 'Averaging the T2* maps of gre1 and gre2',
    'H2O.nii': 'H2O Map',
    'QSM_avg_map.nii.gz': 'QSM Map'
}

with open(os.path.join(output_folder_path, 'map_info.txt'), 'w') as f:
    for file_name, mapping in map_info.items():
        f.write(f'{file_name}: {mapping}\n')

