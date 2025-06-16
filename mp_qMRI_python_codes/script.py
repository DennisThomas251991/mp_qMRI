# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Author info: 
        Mariem Ghazouani,
        Institute of Neuroradiology, University Hospital Frankfurt
        
Date of completion of version 1 of 'Frankfurt_qMRI' python package:
    05/06/2024
    
"""

import os
import sys
from nipype.interfaces import spm
import io

if len(sys.argv) < 2:
    print("Usage: python script.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]

# Set MATLAB paths for SPM using MCR
matlab_cmd = '/mnt/c/spm_linux/spm12_r7771_Linux_R2022a/spm12/run_spm12.sh /mnt/c/spm_linux/v912/ script'

spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
spm.SPMCommand().version
spm.no_spm()

# Run SPM segmentation
seg = spm.NewSegment()
seg.inputs.channel_files = input_file
seg.inputs.channel_info = (0.0001, 60, (True, True))
seg.run()
