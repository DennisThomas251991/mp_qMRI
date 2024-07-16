# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:56:10 2024

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

from ralf_qMRI_conv_3D import ralf_qMRI_conv_3D
def ralf_qMRI_calc_local_mean_and_std(vol, mask, ks):
    """
    Calculation of local mean and standard deviations of values inside 'vol'.

    Parameters:
    - vol: Input 2D or 3D array.
    - mask: Binary mask with the same size as 'vol'. Only pixels where mask>0 are considered.
    - ks: Size of the area (in pixels) across which mean and std are calculated.

    Returns:
    - meanvol: Local mean values.
    - stdvol: Local standard deviations.

    Note: If 'vol' is 2D, the area is a square of ks x ks pixels. If 'vol' is 3D, the area is a cube of ks x ks x ks pixels.
    """
    # Make sure the mask contains only zeros and ones:
    mask = (mask > 0)

    # Get the 2D or 3D kernel:
    if vol.ndim == 2: 
      kernel = np.ones((ks, ks), dtype=float)  
    else:
      kernel = np.ones((ks, ks, ks), dtype=float)

    # Get local sum of values inside 'vol' for which mask is one:
    volmask=(vol*mask)
    sumvol = np.round(ralf_qMRI_conv_3D(volmask,kernel))
    
    # Same for sum of squares of values:
    vol2=vol**2 * mask
    sumvolsq = np.round(ralf_qMRI_conv_3D(vol2, kernel))

    # Calculate sum of pixels inside mask across this area:
    sumpix = np.round(ralf_qMRI_conv_3D(mask, kernel))

    # Calculate averages of values and squared values across this area:
    meanvol = sumvol / (np.finfo(float).eps + sumpix) * (sumpix > 0)
    meanvolsq = sumvolsq / (np.finfo(float).eps + sumpix) * (sumpix > 0)

    # Use known formula for getting standard deviation across area: std(x) = sqrt[<x^2> - <x>^2]
    stdvol =np.sqrt((meanvolsq - meanvol**2) * (meanvolsq >= (meanvol**2)))


    return meanvol, stdvol

