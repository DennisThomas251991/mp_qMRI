import os
import sys
from Useful_functions import convert_dicom_to_nifti, combine_echoes, reorient_image
from pathlib import Path
import re

def check_for_echoes(folder):
    """
    Check if the folder contains any files ending with _e1, _e2, etc.
    """
    return any(re.search(r'_e\d+\.nii\.gz$', file.name) for file in Path(folder).glob("*.nii.gz"))


def extract_numeric_suffix(filename):
    """
    Extract the numeric suffix and echo information from the filename.
    """
    match = re.search(r'_(\d+)_e(\d+)(?:_ph)?\.nii\.gz$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))  # Numeric suffix, echo number
    return None, None

def group_files_by_suffix(input_folder):
    """
    Group files by their numeric suffix and whether they are magnitude or phase files.
    """
    groups = {}
    for file in input_folder.glob("*.nii.gz"):
        suffix, echo = extract_numeric_suffix(file.name)
        if suffix is not None:
            is_phase = "_ph" in file.name
            key = (suffix, is_phase)
            if key not in groups:
                groups[key] = []
            groups[key].append((echo, file))

    # Sort each group by echo number
    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    
    return groups

def process_files(input_folder, combine_echoes_enabled):
    """
    Process the NIfTI files to group echoes, combine them if needed, and filter the final outputs.
    """
    input_folder = Path(input_folder)
    grouped_files = group_files_by_suffix(input_folder)
    combined_files = {}

    if combine_echoes_enabled:
        # Combine echoes if enabled
        for (suffix, is_phase), files in grouped_files.items():
            if len(files) == 6:  # Ensure there are 6 echoes
                arguments = {f'se{i+1}_path': file for i, (echo, file) in enumerate(files)}
                first_file = files[0][1]
                base_name = re.sub(r'_e\d+(_ph)?\.nii\.gz$', '', first_file.name)
                if is_phase:
                    base_name += '_ph_'

                # Combine echoes
                combined_file = combine_echoes(arguments, 6, base_name)
                combined_file_path = Path(combined_file)
                base_name += 'Combined_6_echoes'
                combined_files[(suffix, is_phase)] = combined_file_path

    # Handle B1_1 and B1_2 directly
    b1_files = {}
    for file in input_folder.glob("*.nii.gz"):
        if "FA_45" in file.name and "_ph" not in file.name:
            b1_files["B1_1"] = file
        elif "FA_90" in file.name and "_ph" not in file.name:
            b1_files["B1_2"] = file

    # Filter final output files
    final_files = {}
    if combined_files:
        # Use combined files if they exist
        for (suffix, is_phase), combined_file in combined_files.items():
            filename = combined_file.name
            if "FA_05" in filename and not is_phase:
                if "M0w_mag" not in final_files or suffix < final_files["M0w_mag"][0]:
                    final_files["M0w_mag"] = (suffix, combined_file)
            elif "FA_05" in filename and is_phase:
                if "M0w_pha" not in final_files or suffix < final_files["M0w_pha"][0]:
                    final_files["M0w_pha"] = (suffix, combined_file)
            elif ("FA_32" in filename or "FA_25" in filename) and not is_phase:
                if "T1w_mag" not in final_files or suffix < final_files["T1w_mag"][0]:
                    final_files["T1w_mag"] = (suffix, combined_file)
            elif ("FA_32" in filename or "FA_25" in filename) and is_phase:
                if "T1w_pha" not in final_files or suffix < final_files["T1w_pha"][0]:
                    final_files["T1w_pha"] = (suffix, combined_file)
    else:
        # Use individual files if no combined files
        for file in input_folder.glob("*.nii.gz"):
            if "FA_05" in file.name and "_ph" not in file.name:
                final_files["M0w_mag"] = (0, file)
            elif "FA_05" in file.name and "_ph" in file.name:
                final_files["M0w_pha"] = (0, file)
            elif ("FA_32" in file.name or "FA_25" in file.name) and "_ph" not in file.name:
                final_files["T1w_mag"] = (0, file)
            elif ("FA_32" in file.name or "FA_25" in file.name) and "_ph" in file.name:
                final_files["T1w_pha"] = (0, file)

    # Include rescaled B1 files
    for key, file in b1_files.items():
        if file:
            final_files[key] = (0, file)

    final_files_paths = {key: file for key, (_, file) in final_files.items()}
    return final_files_paths

def finalize_files(output_folder_path, final_files):
    """
    Rename the final files and remove all other files in the folder.
    """
    output_folder_path = Path(output_folder_path)

    # Rename the final files
    for new_name, file_path in final_files.items():
        if file_path.exists():
            new_file_path = output_folder_path / f"{new_name}.nii.gz"
            file_path.rename(new_file_path)
            print(f"Renamed {file_path.name} to {new_file_path.name}")
        else:
            print(f"File not found for {new_name}: {file_path}")

    # Delete all other files in the folder
    for file in output_folder_path.glob("*"):
        if file.name not in [f"{key}.nii.gz" for key in final_files]:
            file.unlink()
    
    print("Finalized folder contents:")
    for file in output_folder_path.glob("*"):
        print(file.name)


def run(input_folder):
    # Capture the original input folder path
    original_input_folder = input_folder

    # Check if the input folder exists
    if not os.path.exists(original_input_folder):
        print("Input folder does not exist.")
        sys.exit(1)

    # === Preprocesing step for Philips data ===
    # Create a Path object for the output folder to save the NIFTI files
    output_folder_path = Path(original_input_folder) / 'Niftis'

    # Ensure the output folder exists
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Convert DICOMs to NIfTI and save to the output folder
    convert_dicom_to_nifti(original_input_folder, output_folder_path)
    print('DICOMs converted')
    # Check for echo files
    combine_echoes_enabled = check_for_echoes(output_folder_path)

    # Process files
    final_files = process_files(output_folder_path, combine_echoes_enabled)
    # Rename and finalize files
    finalize_files(output_folder_path, final_files)
