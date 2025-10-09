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

import sys
import argparse
import shutil
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import T2smapping
import T1mapping_mpqMRI
import H2Omapping_mpqMRI
import B1mapping
import B0_mapping
import preprocessing_Siemens
import preprocessing_Philips
import QSM_mapping_NEW
from Useful_functions import complex_interpolation

def setup_logging(log_file):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_args():
    """Parse command-line arguments."""
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
    parser.add_argument('--slices', default='all', help='Slices')
    parser.add_argument('--echotimes', nargs='+', type=float, default=[3.2, 7.7, 12.2, 16.7, 21.2, 25.7], help='Echo times')
    parser.add_argument('--QSM_average_echoes_qsm', nargs='+', type=int, default=[3, 4, 5, 6], help='Echoes for QSM averaging')
    parser.add_argument('--coregister_mGREs', type=bool, default=False, help='Coregister mGREs')
    parser.add_argument('--B1map_orientation', choices=['Sag', 'Tra'], default='Sag', help='Orientation of the B1 map')
    parser.add_argument('--complex_interpolation', type=bool, default=False, help='Use complex interpolation')
    parser.add_argument('--B1_mapping_method', choices=['Fast EPI', 'Volz 2010'], default='Fast EPI', help='B1 mapping method')
    parser.add_argument('--QSM_mapping', type=bool, default=True, help='Perform QSM mapping')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    return parser.parse_args()

def user_input_with_default(prompt, default):
    """Helper for interactive input with default."""
    user_input = input(f"{prompt} [{default}]: ")
    return user_input if user_input else default

def interactive_update_args(args):
    """Interactively update arguments."""
    logging.info("Entering interactive mode. Press ENTER to keep default.")
    args.Input_data_type = user_input_with_default('Input_data_type (Nifti/DICOM)', args.Input_data_type)
    args.Vendor = user_input_with_default('Vendor (Siemens/Philips)', args.Vendor)
    if args.Vendor == 'Philips':
        args.Philips_acquisition_method = user_input_with_default('Philips_acquisition_method (Combined FA acquisition/Separate FA acquisition)', args.Philips_acquisition_method)
    args.Bias_field_corr_method = user_input_with_default('Bias_field_corr_method (SPM/FSL)', args.Bias_field_corr_method)
    args.FA1 = int(user_input_with_default('FA1 (Flip angle 1)', args.FA1))
    args.FA2 = int(user_input_with_default('FA2 (Flip angle 2)', args.FA2))
    args.TR1 = int(user_input_with_default('TR1 [ms]', args.TR1))
    args.TR2 = int(user_input_with_default('TR2 [ms]', args.TR2))
    args.nTE = int(user_input_with_default('nTE (Number of echo times)', args.nTE))
    args.x0 = int(user_input_with_default('x0 value', args.x0))
    args.phantom = user_input_with_default('Phantom (True/False)', args.phantom).lower() == 'true'
    args.masking = user_input_with_default('masking (True/False)', args.masking).lower() == 'true'
    args.b1plus_mapping = user_input_with_default('b1plus_mapping (True/False)', args.b1plus_mapping).lower() == 'true'
    args.corr_for_imperfect_spoiling = user_input_with_default('corr_for_imperfect_spoiling (True/False)', args.corr_for_imperfect_spoiling).lower() == 'true'
    args.Phantom_mask_threshold_B1mapping = int(user_input_with_default('Phantom_mask_threshold_B1mapping', args.Phantom_mask_threshold_B1mapping))
    args.Phantom_mask_threshold_T1mapping = int(user_input_with_default('Phantom_mask_threshold_T1mapping', args.Phantom_mask_threshold_T1mapping))
    args.spoil_increment = int(user_input_with_default('Spoil increment', args.spoil_increment))
    args.slices = user_input_with_default('Slices', args.slices)
    if user_input_with_default('Change echotimes? (yes/no)', 'no').lower() == 'yes':
        T0 = float(user_input_with_default('Initial echo time', 3.20))
        delta_t = float(user_input_with_default('Time difference between echoes', 4.15))
        num_echoes = int(user_input_with_default('Number of echoes', 6))
        args.echotimes = [T0 + i * delta_t for i in range(num_echoes)]
    args.QSM_average_echoes_qsm = [int(e) for e in user_input_with_default('QSM_average_echoes_qsm (comma-separated)', ','.join(map(str, args.QSM_average_echoes_qsm))).split(',')]
    args.coregister_mGREs = user_input_with_default('Coregister mGREs? (True/False)', args.coregister_mGREs).lower() == 'true'
    args.B1map_orientation = user_input_with_default('B1map_orientation (Sag/Tra)', args.B1map_orientation)
    args.complex_interpolation = user_input_with_default('complex_interpolation (True/False)', args.complex_interpolation).lower() == 'true'
    args.B1_mapping_method = user_input_with_default('B1_mapping_method (Fast EPI/Volz 2010)', args.B1_mapping_method)
    args.QSM_mapping = user_input_with_default('QSM_mapping (True/False)', args.QSM_mapping).lower() == 'true'
    return args

def write_log_file(args, output_folder):
    """Write arguments to a log file."""
    log_file = output_folder / 'info.txt'
    with open(log_file, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

def find_expected_files(output_folder, expected_files):
    """Find expected files (.nii or .nii.gz) in output_folder."""
    file_paths = {}
    for file_name in expected_files:
        for ext in ('.nii', '.nii.gz'):
            candidate = output_folder / f"{file_name}{ext}"
            if candidate.exists():
                file_paths[file_name] = str(candidate)
                break
    missing = set(expected_files) - set(file_paths.keys())
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")
    return file_paths

def assign_file_paths(args, file_paths):
    """Assign file paths to argument attributes."""
    args.gre1_path = file_paths['M0w_mag']
    args.phase_image1 = file_paths['M0w_pha']
    args.gre2_path = file_paths['T1w_mag']
    args.phase_image2 = file_paths['T1w_pha']
    args.path_b1_noprep = file_paths['B1_1']
    args.path_b1_prep = file_paths['B1_2']

def move_files(src_folder, dst_folder, except_files):
    """Move all files except those in except_files from src_folder to dst_folder."""
    dst_folder.mkdir(exist_ok=True)
    for item in src_folder.iterdir():
        if item.is_file() and item.name not in except_files:
            shutil.move(str(item), str(dst_folder))

def plot_map_histogram(map_path, mask_paths, output_path, map_type, bins=200):
    """Plot histogram for quantitative map and save as TIFF image."""
    map_img = nib.load(str(map_path))
    map_data = map_img.get_fdata()
    c1 = nib.load(str(mask_paths[0])).get_fdata()
    c2 = nib.load(str(mask_paths[1])).get_fdata()
    c3 = nib.load(str(mask_paths[2])).get_fdata()
    wm_mask = (c2 > 0.98) & (c3 <= 0.80) & (c1 <= 0.80)
    gm_mask = (c1 > 0.98) & (c3 <= 0.80) & (c2 <= 0.80)
    brain_mask = (wm_mask | gm_mask) & np.isfinite(map_data)
    map_values = map_data[brain_mask]
    if map_type == 'T1':
        range_vals = (1, 3000)
        xlabel = 'T1 [ms]'
    elif map_type == 'T2s':
        range_vals = (10, 300)
        xlabel = 'T2* [ms]'
    elif map_type == 'H2O':
        range_vals = (0.4, 1.3)
        xlabel = 'H2O fraction'
    elif map_type == 'QSM':
        range_vals = (-0.3, 0.3)
        xlabel = 'QSM [ppm]'
    else:
        range_vals = (np.nanmin(map_values), np.nanmax(map_values))
        xlabel = f'{map_type} Value'
    plt.figure()
    plt.hist(map_values, bins=bins, range=range_vals, edgecolor='blue', fc='blue', alpha=0.7, density=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks([])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.title(f'{map_type} Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=350, format='tiff')
    plt.close()

def main():
    args = parse_args()
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Input folder does not exist: {input_folder}")
        sys.exit(1)
    output_folder = input_folder / 'Niftis' if args.Input_data_type == 'DICOM' else input_folder
    output_folder.mkdir(exist_ok=True)
    setup_logging(output_folder / "pipeline.log")
    if getattr(args, 'interactive', False):
        args = interactive_update_args(args)
    logging.info("Starting mp-qMRI pipeline.")
    # Preprocessing
    if args.Input_data_type == 'DICOM':
        if args.Vendor == 'Philips':
            preprocessing_Philips.run(str(input_folder), args.Philips_acquisition_method)
        elif args.Vendor == 'Siemens':
            preprocessing_Siemens.run(str(input_folder))
    write_log_file(args, output_folder)
    # Find and assign files
    expected_files = ['M0w_mag', 'T1w_mag', 'M0w_pha', 'T1w_pha', 'B1_1', 'B1_2']
    file_paths = find_expected_files(output_folder, expected_files)
    assign_file_paths(args, file_paths)
    # B1 mapping method-specific parameters
    if args.B1_mapping_method == 'Fast EPI':
        args.b1map_filename = 'B1_MAP_from_fast_EPI_standard'
        args.path_epi_45 = args.path_b1_noprep
        args.path_epi_90 = args.path_b1_prep
    elif args.B1_mapping_method == 'Volz 2010':
        args.b1map_filename = 'standard_B1_MAP_smoothed'
    # Prepare flip angles and GRE images
    FA1 = np.deg2rad(args.FA1)
    FA2 = np.deg2rad(args.FA2)
    args.FA_list = [FA1, FA2]
    gre1 = nib.load(args.gre1_path).get_fdata()
    gre2 = nib.load(args.gre2_path).get_fdata()
    args.affine = nib.load(args.gre2_path).affine
    args.b1plus = np.ones(gre1.shape[:-1])
    args.gre_list = [gre1, gre2]
    args.TR_list = [args.TR1, args.TR1]
    # Optional: complex interpolation
    if args.complex_interpolation:
        for idx, (mag_path, pha_path, FA) in enumerate([
            (args.gre1_path, args.phase_image1, args.FA1),
            (args.gre2_path, args.phase_image2, args.FA2)
        ]):
            save_name = f"resampled_1mm_iso_FA{FA}"
            mag_interp_path, pha_interp_path = complex_interpolation(mag_path, pha_path, save_name=save_name)
            if idx == 0:
                args.gre1_path, args.phase_image1 = mag_interp_path, pha_interp_path
            else:
                args.gre2_path, args.phase_image2 = mag_interp_path, pha_interp_path
        gre1 = nib.load(args.gre1_path).get_fdata()
        gre2 = nib.load(args.gre2_path).get_fdata()
        args.gre_list = [gre1, gre2]
        args.affine = nib.load(args.gre2_path).affine
    # B0 and B1 mapping
    if args.B1_mapping_method == 'Fast EPI':
        B0map_object = B0_mapping.B0_mapping(args)
        B0map_coreg2B1_path = B0map_object.run(gre='gre1', coregister_2_EPI=True)
        B1map_object = B1mapping.B1_map_Dennis(args)
        B1map_coreg = B1map_object.produce_B1_maps(B0map_coreg2B1_path, coregister_2_T1=True)
    elif args.B1_mapping_method == 'Volz 2010':
        B1map_object = B1mapping.B1_map_Ralf(args)
        B1map_coreg = B1map_object.produce_B1_maps(coregister_2_T1=True)
    # T1 mapping
    T1_object = T1mapping_mpqMRI.T1_mapping_mpqMRI(args, B1map_coreg)
    T1_map2 = T1_object.run_linear_approach(save=True)
    T1_map = T1_object.run(save=True)
    # T2* mapping
    T2s_object = T2smapping.t2s_map_mpqMRI(args)
    output = T2s_object.run(nechoes=6, save=True)
    T2Star_gre12 = output[7]
    # H2O mapping
    H2O_object = H2Omapping_mpqMRI.H2O_map_mpqMRI(args, T1_map, B1map_coreg, T2Star_gre12)
    H2O = H2O_object.run()
    # Clean up segmentation masks
    for mask_name in ['c1T1w_mag.nii', 'c2T1w_mag.nii', 'c3T1w_mag.nii']:
        mask_path = output_folder / mask_name
        if mask_path.exists():
            mask_path.unlink()
    # QSM mapping (optional)
    if args.QSM_mapping:
        QSM_object = QSM_mapping_NEW.QSM_mapping_mpqMRI(args)
        QSM = QSM_object.run(gre='gre2')
    # Organize output files
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
    intermediary_folder = output_folder / 'intermediate_files'
    final_maps_folder = output_folder / 'Final_qMRI_maps'
    move_files(output_folder, intermediary_folder, output_files + main_maps)
    final_maps_folder.mkdir(exist_ok=True)
    for file_name in output_files:
        src_path = output_folder / file_name
        if src_path.exists():
            shutil.move(str(src_path), str(final_maps_folder))
    # Write mapping info
    map_info = {
        'T1_map_B1corr_True_Spoilcorr_True_2echoes.nii': 'T1 map non-linear approach',
        't1map_linear_approach.nii': 'T1 map linear approach',
        'avg_T2Star.nii': 'T2* map of the average of gre1 and gre2',
        'T2Star_avg.nii': 'Averaging the T2* maps of gre1 and gre2',
        'H2O.nii': 'H2O Map',
        'QSM_avg_map.nii.gz': 'QSM Map'
    }
    with open(output_folder / 'map_info.txt', 'w') as f:
        for file_name, mapping in map_info.items():
            f.write(f'{file_name}: {mapping}\n')
    # Histogram generation
    try:
        mask_files = [
            intermediary_folder / 'c1T1w_Mag.nii.gz',
            intermediary_folder / 'c2T1w_Mag.nii.gz',
            intermediary_folder / 'c3T1w_Mag.nii.gz'
        ]
        map_files = {
            'T1': final_maps_folder / 'T1_map_B1corr_True_Spoilcorr_True_2echoes.nii.gz',
            'T2s': final_maps_folder / 'avg_T2Star.nii.gz',
            'H2O': final_maps_folder / 'H2O.nii.gz',
            'QSM': final_maps_folder / 'QSM_avg_map.nii.gz' if args.QSM_mapping else None
        }
        if all(mask.exists() for mask in mask_files):
            for map_type, map_path in map_files.items():
                if map_path and map_path.exists():
                    output_path = final_maps_folder / f'{map_type}_histogram.tiff'
                    plot_map_histogram(map_path, mask_files, output_path, map_type)
                    logging.info(f"Generated histogram for {map_type} at {output_path}")
                elif map_type == 'QSM' and not args.QSM_mapping:
                    logging.info("Skipping QSM histogram (QSM mapping was not performed)")
                else:
                    logging.warning(f"Map file not found for {map_type}: {map_path}")
        else:
            logging.warning("Segmentation masks not found. Skipping histogram generation.")
            for mask in mask_files:
                if not mask.exists():
                    logging.warning(f"Missing mask: {mask}")
    except Exception as e:
        logging.error(f"Error generating histograms: {str(e)}")
    logging.info("Pipeline processing complete!")

if __name__ == "__main__":
    main()
