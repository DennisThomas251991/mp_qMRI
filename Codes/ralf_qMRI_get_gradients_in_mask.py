# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:52:25 2024

Author info: 
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

def ralf_qMRI_get_gradients_in_mask(vol, mask):
    # Calculates gradients in 3D vol
    # gx, gy, gz are the gradients in the first, second, and third direction, respectively
    # Only pixels that are non-zero in mask are used for calculation
    # The function first calculates the "upper" and "lower" gradient
    # Upper gradient: gradient at position i is difference vol(i+1) - vol(i)
    # Lower gradient: gradient at position i is difference vol(i) - vol(i-1)
    # Both gradients are averaged
    # However, if one gradient cannot be calculated because one of the contributing pixels is not inside the mask:
    # Only the other gradient is used, and if both cannot be calculated, zero is exported

    # make sure mask contains only ones and zeros:
    mask = (mask > 0)

    myhelp1 = np.zeros_like(vol)
    myhelp2 = np.zeros_like(vol)
    weight1 = np.zeros_like(vol)
    weight2 = np.zeros_like(vol)

    myhelp1[1:, :, :] = vol[1:, :, :] - vol[:-1, :, :]
    weight1[1:, :, :] = mask[1:, :, :] * mask[:-1, :, :]
    # myhelp1 is the lower gradient
    # weight1 is 1 if both values contributing to myhelp1 are inside the mask, otherwise, it is zero
    # Same for upper gradient:
    myhelp2[:-1, :, :] = vol[1:, :, :] - vol[:-1, :, :]
    weight2[:-1, :, :] = mask[1:, :, :] * mask[:-1, :, :]
    # Calculate gradient as a weighted average:
    gx = (myhelp1 * weight1 + myhelp2 * weight2) / (np.finfo(float).eps + weight1 + weight2) * ((weight1 + weight2) > 0)

    # Same for the second direction:
    myhelp1 = np.zeros_like(vol)
    myhelp2 = np.zeros_like(vol)
    weight1 = np.zeros_like(vol)
    weight2 = np.zeros_like(vol)
    myhelp1[:, 1:, :] = vol[:, 1:, :] - vol[:, :-1, :]
    weight1[:, 1:, :] = mask[:, 1:, :] * mask[:, :-1, :]
    myhelp2[:, :-1, :] = vol[:, 1:, :] - vol[:, :-1, :]
    weight2[:, :-1, :] = mask[:, 1:, :] * mask[:, :-1, :]
    gy = (myhelp1 * weight1 + myhelp2 * weight2) / (np.finfo(float).eps + weight1 + weight2) * ((weight1 + weight2) > 0)

    # Same for the third direction:
    myhelp1 = np.zeros_like(vol)
    myhelp2 = np.zeros_like(vol)
    weight1 = np.zeros_like(vol)
    weight2 = np.zeros_like(vol)
    myhelp1[:, :, 1:] = vol[:, :, 1:] - vol[:, :, :-1]
    weight1[:, :, 1:] = mask[:, :, 1:] * mask[:, :, :-1]
    myhelp2[:, :, :-1] = vol[:, :, 1:] - vol[:, :, :-1]
    weight2[:, :, :-1] = mask[:, :, 1:] * mask[:, :, :-1]
    gz = (myhelp1 * weight1 + myhelp2 * weight2) / (np.finfo(float).eps + weight1 + weight2) * ((weight1 + weight2) > 0)

    return gx, gy, gz

