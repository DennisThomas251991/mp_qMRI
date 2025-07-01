import sys
import re
import os
import nibabel as nib
from pathlib import Path
from Useful_functions import convert_dicom_to_nifti, Philips_scaling_correct_MRI_CroGL, extract_scaling_parameters, combine_echoes, split_all_echoes

def split_combined_fa_echoes_in_folder(input_folder):
    """
    Detects NIfTI files that contain combined flip angle (FA) information and splits their echoes.
    Uses known FA patterns to identify files that need splitting.
    """
    fa_patterns = [
        ("FA32", "5"),
        ("FA32", "FA05"),
        ("FA05", "FA32"),
        ("FA45", "FA90"),
        ("FA90", "FA45"),
    ]
    input_folder = Path(input_folder)
    for file in input_folder.glob("*.nii.gz"):
        fname = file.name
        for fa1, fa2 in fa_patterns:
            if fa1 in fname and fa2 in fname:
                print(f"Splitting all echoes for: {fname}")
                split_all_echoes(str(file))
                break

def detect_and_process_echo_files(input_folder):
    """
    Scans the input folder for NIfTI files with echo information in their names.
    Groups files by (FA, phase/magnitude, group_id) and sorts echoes within each group.
    Returns a mapping for further processing.
    """
    input_folder = Path(input_folder)
    echo_map = {}

    for file in input_folder.glob("*_echo.nii.gz"):
        # Regex matches both multi-echo and single-echo files
        match = re.search(r'(FA\d+).*?(?:_e(\d+))?(_ph)?_(\d+)_echo\.nii', file.name)
        if not match:
            continue

        fa_value = match.group(1)
        echo_num = int(match.group(2)) if match.group(2) else 1  # default echo number for single-echo
        is_phase = match.group(3) is not None
        group_id = int(match.group(4))

        base_key = (fa_value, is_phase, group_id)
        if base_key not in echo_map:
            echo_map[base_key] = []

        echo_map[base_key].append((echo_num, file))

    # Sort echoes within each group
    for key in echo_map:
        echo_map[key].sort(key=lambda x: x[0])

    return echo_map

def process_files(input_folder):
    """
    Combines echoes if needed and maps files to M0w_mag, M0w_pha, T1w_mag, T1w_pha, B1_1, B1_2 based on filename patterns.
    """
    input_folder = Path(input_folder)
    final_files = {}

    # First, combine echoes for all groups (if needed)
    echo_groups = detect_and_process_echo_files(input_folder)
    for group, files in echo_groups.items():
        if len(files) > 1:
            # Combine echoes
            is_phase = group[1]
            arguments = {f'se{i+1}_path': file for i, (echo, file) in enumerate(files)}
            # Use the first file's name as base for output
            first_file = files[0][1]
            base_name = first_file.name.replace('.nii.gz', '')
            if is_phase:
                base_name += '_ph'
            combined_file = combine_echoes(arguments, len(files), base_name)
            combined_file_path = Path(combined_file)
            # Remove original files after combining
            for _, file in files:
                if file.exists():
                    file.unlink()
            # Add the combined file to the folder for mapping below
            # We'll use the combined file's name for mapping
            files = [(0, combined_file_path)]
        else:
            # Only one file, keep as is
            files = [files[0]]

        # For each file (combined or single), do the mapping below
        for _, file in files:
            fname = file.name

            # M0w (FA32 or FA05, _1_echo)
            if ("FA32" in fname or "FA05" in fname) and (("_1_echoCombined" in fname) or ("_1_echo_phCombined" in fname)):
                # if "_ph" in fname:
                #     final_files["M0w_pha"] = file
                # else:
                #     final_files["M0w_mag"] = file
                if "_ph" in fname:
                    final_files["T1w_pha"] = file
                else:
                    final_files["T1w_mag"] = file
            # T1w (FA32 or FA05, _2_echo)
            elif ("FA32" in fname or "FA05" in fname) and (("_2_echoCombined" in fname) or ("_2_echo_phCombined" in fname)):
                # if "_ph" in fname:
                #     final_files["T1w_pha"] = file
                # else:
                #     final_files["T1w_mag"] = file
                if "_ph" in fname:
                    final_files["M0w_pha"] = file
                else:
                    final_files["M0w_mag"] = file

            # B1_1 (FA45 or FA90, _1_echo)
            elif ("FA45" in fname or "FA90" in fname) and ("_1_echo" in fname):
                if "_ph_" not in fname:
                     final_files["B1_1"] = file

            # B1_2 (FA45 or FA90, _2_echo)
            elif ("FA45" in fname or "FA90" in fname) and ("_2_echo" in fname):
                if "_ph_" not in fname:
                    final_files["B1_2"] = file

    # Rename files to standard names
    for key, file in final_files.items():
        target_path = input_folder / f"{key}.nii.gz"
        if file != target_path:
            file.rename(target_path)
            final_files[key] = target_path

    return final_files

def finalize_files(output_folder_path, final_files):
    """
    Removes all files not in final_files and prints the final folder contents.
    """
    output_folder_path = Path(output_folder_path)
    keep_files = {f"{key}.nii.gz" for key in final_files}

    # Remove all other NIfTI files not in the final set
    for file in output_folder_path.glob("*.nii.gz"):
        if file.name not in keep_files:
            file.unlink()

    # Remove all JSON files
    for file in output_folder_path.glob("*.json"):
        file.unlink()

    print("Finalized folder contents:")
    for file in output_folder_path.glob("*.nii.gz"):
        print(file.name)

def run(input_folder, acquisition_method):
    """
    Main entry point for preprocessing.
    Runs different logic depending on the acquisition method.
    """
    if acquisition_method == "Combined FA acquisition":
        # Combined FA acquisition logic
        original_input_folder = input_folder
        if not os.path.exists(original_input_folder):
            print("Input folder does not exist.")
            sys.exit(1)

        # Create output folder for NIfTI files
        output_folder_path = Path(original_input_folder) / 'Niftis'
        output_folder_path.mkdir(parents=True, exist_ok=True)

        # Convert DICOMs to NIfTI
        convert_dicom_to_nifti(original_input_folder, output_folder_path)
        print('DICOMs converted')

        # Split combined FA echoes
        split_combined_fa_echoes_in_folder(output_folder_path)

        # Process and finalize files
        final_files = process_files(output_folder_path)
        print(final_files)
        finalize_files(output_folder_path, final_files)
    else:
        # Separate FA acquisition logic
        run_separate_fa_acquisition(input_folder)

def run_separate_fa_acquisition(input_folder):
    """
    Handles preprocessing for separate FA acquisition.
    Converts DICOMs, applies scaling, combines echoes if needed, and cleans up.
    """

    def check_for_echoes(folder):
        """
        Check if the folder contains any files with echo suffix (_e1, _e2, etc.).
        """
        return any(re.search(r'_e\d+\.nii\.gz$', file.name) for file in Path(folder).glob("*.nii.gz"))

    def extract_numeric_suffix(filename):
        """
        Extract numeric suffix and echo number from filename.
        """
        match = re.search(r'_(\d+)_e(\d+)(?:_ph)?\.nii\.gz$', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def group_files_by_suffix(input_folder):
        """
        Group files by their numeric suffix and phase/magnitude status.
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
        Process NIfTI files: combine echoes if needed, apply scaling, and collect final outputs.
        """
        input_folder = Path(input_folder)
        grouped_files = group_files_by_suffix(input_folder)
        combined_files = {}
        individual_files = []

        if combine_echoes_enabled:
            # Combine echoes for each group if there are 6 echoes
            for (suffix, is_phase), files in grouped_files.items():
                if len(files) == 6:
                    arguments = {f'se{i+1}_path': file for i, (echo, file) in enumerate(files)}
                    first_file = files[0][1]
                    base_name = re.sub(r'_e\d+(_ph)?\.nii\.gz$', '', first_file.name)
                    if is_phase:
                        base_name += '_ph_'
                    # Combine echoes into one file
                    combined_file = combine_echoes(arguments, 6, base_name)
                    combined_file_path = Path(combined_file)
                    base_name += 'Combined_6_echoes'
                    if is_phase:
                        print(f"Phase image: {combined_file}")
                    else:
                        # Apply scaling correction if JSON exists
                        json_file = first_file.with_suffix('').with_suffix('.json')
                        if json_file.exists():
                            try:
                                rescale_slope, rescale_intercept, scale_slope = extract_scaling_parameters(json_file)
                                Philips_scaling_correct_MRI_CroGL(
                                    combined_file_path,
                                    rescale_slope,
                                    rescale_intercept,
                                    scale_slope
                                )
                                corrected_file = combined_file_path.with_name(f"{base_name}scaling_CORR.nii.gz")
                                print(f"Scaling correction applied: {corrected_file}")
                                combined_file_path = corrected_file
                            except Exception as e:
                                print(f"Error processing {combined_file}: {e}")
                        else:
                            print(f"JSON file not found for {base_name}.json")
                    combined_files[(suffix, is_phase)] = combined_file_path
        else:
            # Process individual files (no echo combining)
            for file in input_folder.glob("*.nii.gz"):
                if "_ph" not in file.name:
                    base_name = re.sub(r'\.nii\.gz$', '', file.name)
                    json_file = file.with_suffix('').with_suffix('.json')
                    if json_file.exists():
                        try:
                            rescale_slope, rescale_intercept, scale_slope = extract_scaling_parameters(json_file)
                            Philips_scaling_correct_MRI_CroGL(
                                file,
                                rescale_slope,
                                rescale_intercept,
                                scale_slope
                            )
                            corrected_file = file.with_name(f"{base_name}scaling_CORR.nii.gz")
                            print(f"Scaling correction applied: {corrected_file}")
                            individual_files.append(corrected_file)
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                    else:
                        print(f"JSON file not found for {file.name}")
                        individual_files.append(file)

        # Handle B1 mapping files (FA45 and FA90)
        b1_files = {}
        for file in input_folder.glob("*.nii.gz"):
            if "FA45" in file.name and "_ph" not in file.name:
                b1_files["B1_1"] = file
            elif "FA90" in file.name and "_ph" not in file.name:
                b1_files["B1_2"] = file
        for key, b1_file in b1_files.items():
            if b1_file:
                base_name = re.sub(r'\.nii\.gz$', '', b1_file.name)
                json_file = b1_file.with_suffix('').with_suffix('.json')
                if json_file.exists():
                    try:
                        rescale_slope, rescale_intercept, scale_slope = extract_scaling_parameters(json_file)
                        Philips_scaling_correct_MRI_CroGL(
                            b1_file,
                            rescale_slope,
                            rescale_intercept,
                            scale_slope
                        )
                        corrected_file = b1_file.with_name(f"{base_name}scaling_CORR.nii.gz")
                        print(f"Scaling correction applied to {key}: {corrected_file}")
                        b1_files[key] = corrected_file
                    except Exception as e:
                        print(f"Error processing {b1_file}: {e}")

        # Collect final files for output
        final_files = {}
        if combined_files:
            for (suffix, is_phase), combined_file in combined_files.items():
                filename = combined_file.name
                if "FA05" in filename and not is_phase:
                    if "M0w_mag" not in final_files or suffix < final_files["M0w_mag"][0]:
                        final_files["M0w_mag"] = (suffix, combined_file)
                elif "FA05" in filename and is_phase:
                    if "M0w_pha" not in final_files or suffix < final_files["M0w_pha"][0]:
                        final_files["M0w_pha"] = (suffix, combined_file)
                elif ("FA32" in filename ) and not is_phase:
                    if "T1w_mag" not in final_files or suffix < final_files["T1w_mag"][0]:
                        final_files["T1w_mag"] = (suffix, combined_file)
                elif ("FA32" in filename ) and is_phase:
                    if "T1w_pha" not in final_files or suffix < final_files["T1w_pha"][0]:
                        final_files["T1w_pha"] = (suffix, combined_file)
        else:
            for file in individual_files:
                if "FA05" in file.name and "_ph" not in file.name:
                    final_files["M0w_mag"] = (0, file)
                elif "FA05" in file.name and "_ph" in file.name:
                    final_files["M0w_pha"] = (0, file)
                elif("FA32" in file.name ) and "_ph" not in file.name:
                    final_files["T1w_mag"] = (0, file)
                elif ("FA32" in file.name) and "_ph" in file.name:
                    final_files["T1w_pha"] = (0, file)

        # Add B1 files to final output
        for key, file in b1_files.items():
            if file:
                final_files[key] = (0, file)

        # Return only the file paths
        final_files_paths = {key: file for key, (_, file) in final_files.items()}
        return final_files_paths

    def finalize_files(output_folder_path, final_files):
        """
        Rename final files and remove all other files in the output folder.
        """
        output_folder_path = Path(output_folder_path)
        for new_name, file_path in final_files.items():
            if file_path.exists():
                new_file_path = output_folder_path / f"{new_name}.nii.gz"
                file_path.rename(new_file_path)
                print(f"Renamed {file_path.name} to {new_file_path.name}")
            else:
                print(f"File not found for {new_name}: {file_path}")
        # Remove all files not in the final output
        for file in output_folder_path.glob("*"):
            if file.name not in [f"{key}.nii.gz" for key in final_files]:
                file.unlink()
        print("Finalized folder contents:")
        for file in output_folder_path.glob("*"):
            print(file.name)

    # --- Main logic for separate FA acquisition ---
    original_input_folder = input_folder
    if not os.path.exists(original_input_folder):
        print("Input folder does not exist.")
        sys.exit(1)
    # Create output folder
    output_folder_path = Path(original_input_folder) / 'Niftis'
    output_folder_path.mkdir(parents=True, exist_ok=True)
    # Convert DICOMs to NIfTI
    convert_dicom_to_nifti(original_input_folder, output_folder_path)
    print('DICOMs converted')
    # Check for echo files
    combine_echoes_enabled = check_for_echoes(output_folder_path)
    # Process and finalize files
    final_files = process_files(output_folder_path, combine_echoes_enabled)
    finalize_files(output_folder_path, final_files)
