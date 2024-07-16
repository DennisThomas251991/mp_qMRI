# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:49:53 2024

@author: mmariAuthor info: 
    Original codes written in MATLAB by: 
        Prof. Ralf Deichmann, 
        Brain Imaging Center Frankfurt

    Ported to Python by:
        Mariem Ghazouani,
        Institute of Neuroradiology, University Hospital Frankfurt
        
Date of completion of version 1 of 'Frankfurt_qMRI' python package:
    05/06/2024

"""

import numpy as np
from ralf_qMRI_conv_3D import ralf_qMRI_conv_3D
def ralf_qMRI_get_edges_of_mask(input_mask):
    # Calculates edges of a binary 3D mask
    # edge1 is the inner edge, edge2 the outer edge

    # Convert input_mask into something which definitely contains only ones and zeros:
    input_mask = (input_mask > 0).astype(int)

    # Get kernel 3x3x3:
    ks = 3
    kernel = np.ones((ks, ks, ks))
    myhelp = np.round(ralf_qMRI_conv_3D(input_mask, kernel))

    # Inside the mask, apart from the inner edge (edge1),
    # myhelp has the value ks^3
    # Outside the mask, apart from the outer edge (edge2),
    # myhelp has the value 0

    # Inside edge1: input_mask is 1 but myhelp is less than ks^3
    edge1 = (input_mask == 1) * (myhelp <= (ks**3 - 1))
    
    # Inside edge2: input_mask is 0 but myhelp is >=1
    edge2 = (input_mask == 0) * (myhelp >= 1)

    return edge1, edge2
