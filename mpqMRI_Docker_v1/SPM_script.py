# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
matlab_cmd = '/opt/spm12/run_spm12.sh /opt/mcr/v97/ script'

spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

print('#### Starting SPM Segmentation ####')
# Run SPM segmentation
seg = spm.NewSegment()
seg.inputs.channel_files = input_file
seg.inputs.channel_info = (0.001, 60, (True, True))
seg.run()
print('#### SPM Segmentation finished ####')