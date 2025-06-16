# -*- coding: utf-8 -*-import subprocess
import subprocess
import os
import nibabel as nib

def CalcSPMBatch(data):
    # Convert Windows path to WSL-compatible path
    linux_path = data.replace('\\', '/')
    if ':' in linux_path:
        drive_letter, rest = linux_path.split(':', 1)
        linux_path = f"/mnt/{drive_letter.lower()}/{rest}"
    
    # Define the WSL command with the modified path
    wsl_command = f"wsl python3 /mnt/c/spm_linux/script.py {linux_path}"
    
    # Execute the WSL command
    subprocess.run(wsl_command, shell=True)
    
