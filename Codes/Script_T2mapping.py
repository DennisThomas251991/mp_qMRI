# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:06:38 2023

Author:
    Dennis C. Thomas
    Institute of Neuroradiology University Hospital Frankfurt
    
Date of completion of version 1 of 'Frankfurt_qMRI' python package:
    05/06/2024
"""

from qCET import T2mapping

arguments = dict()

arguments['se1_path'] = 'W:/NR-GLIOGLUT/GLIOGLUT/03_Results/GLIOGLUT10_T2mapping/se_25_optimised2_6min_protocol__16.nii.gz'
arguments['se2_path'] = 'W:/NR-GLIOGLUT/GLIOGLUT/03_Results/GLIOGLUT10_T2mapping/se_50_optimised2_6min_protocol__17.nii.gz'
arguments['se3_path'] = 'W:/NR-GLIOGLUT/GLIOGLUT/03_Results/GLIOGLUT10_T2mapping/se_75_optimised2_6min_protocol__18.nii.gz'
arguments['se4_path'] = 'W:/NR-qCET/analysis/Phantoms/Phantom_T2mapping_DLBCL/se_90_optimised_2.28ms__8.nii.gz'

arguments['echotimes'] = [25, 50, 75]
arguments['nechoes'] = 3
arguments['masking_method'] = 'thresholding'
arguments['mask_threshold'] = 100

T2_object = T2mapping.t2_map(arguments)
output = T2_object.run(save = True, masking=True)




