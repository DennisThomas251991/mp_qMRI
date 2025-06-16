# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:56:00 2024

@author: mmari
"""
import numpy as np
import ralf_qMRI_conv_3D
def ralf_qMRI_get_image_mask_improved(vol,level=4,max_holesize=0):

# function for finding a mask excluding noise in 3D image data
# vol is a 3D modulus image data set
# level influences the cut-off level for noise. Low values may include noise in the mask, high levels max exclude areas with signal from mask
#       Recommended: level=4    (this is the default if level is not set)
# max_holesize is an optional argument: if set and non-zero, holes will be filled in the final mask
#  -> Enter here the largest expected diameters of holes in the mask (in pixels)
#  -> If you do not want to fill holes: use max_holesize=0 or omit this argument
#       Recommended: max_holesize=7 (but default is 0 if max_holesize is not set)
# imamask is a binary mask (zero where vol contains noise only)
# This function works for data where the bulk part of the signal is concentrated in an inner volume with noise outside

    # PART 1
    #An improved version of vol is derived (vol0) where areas of noise are suppressed in intensity
    # The actual masking procedure will then be applied to vol0, rather than to vol
    
    # First, we calculate a rough mask mmtemp
    # This is done by obtaining noise from the 8 corners of vol (cuboid subvolumes)
    # A lower limit is then derived from these noise data to get mmtemp
    
    # Get size of vol:
    mysize = np.array(vol.shape)
    # The size of the cuboids is 1/8 of the size of vol:
    sz1= round(mysize(1)/8) 
    sz2=round(mysize(2)/8) 
    sz3 = round(mysize(3)/8) 
    # Empty object for data from corners:
    corners = np.empty((sz1, sz2, sz3, 8))
    # Collect data from vol within the 8 subvolumes close to the corners
    corners[:, :, :, 0] = vol[:sz1, :sz2, :sz3]
    corners[:, :, :, 1] = vol[:sz1, -sz2:, :sz3]
    corners[:, :, :, 2] = vol[-sz1:, :sz2, :sz3]
    corners[:, :, :, 3] = vol[-sz1:, -sz2:, :sz3]
    corners[:, :, :, 4] = vol[:sz1, :sz2, -sz3:]
    corners[:, :, :, 5] = vol[:sz1, -sz2:, -sz3:]
    corners[:, :, :, 6] = vol[-sz1:, :sz2, -sz3:]
    corners[:, :, :, 7] = vol[-sz1:, -sz2:, -sz3:]
    # For each corner an upper noise limit is derived
    # empty object for limits:
    mylim = np.zeros(8)
    # Loop over corners:
    for i in range(8):
        #Get data from corner:
        myhelp = corners[:, :, :, i]
        # Get non-zero data (as there may be zeros in the edges of vol):
        data = myhelp[myhelp > 0]
        #removing outliers:
        mymean, mystd = np.mean(data), np.std(data)
        datared = data[data < (mymean + mystd)]
        # Since datared is without outliers: use maximum value for mylim(i):
        mylim[i] = np.max(datared)
    # One or more entries in mylim may have high values, e.g. if there is object signal in one or more of the corners
    # To remove these outliers, get the median of values in mylim:
    mymedian = np.median(mylim)
    # The limit is the mean of those entries in mylim that are not larger than 1.1*mymedian:
    limit = np.mean(mylim[mylim < 1.1 * mymedian])
    
    # This is a temporary mask: remove all data with values up to limit:
    mmtemp = (vol > limit)
    # We now derive a weighting factor for reducing the intensity of noisy data in vol
    # Convolve mmtemp with a 3x3x3 kernel to get a smooth weighting function:
    myweight = ralf_qMRI_conv_3D(mmtemp, np.ones((3, 3, 3)))
    myweight = myweight * (myweight >= 0)
    # Derive vol0 where noise is reduced:
    vol0 = vol * myweight
    # In vol0, outer noise should be suppressed to a certain degree
    # The following parts work on vol0, rather than vol

    # PART 2
    # It is assumed that vol0 has roughly an ellipsoidal shape
    # We caluclated the mass and the moments of inertia of vol0
    # From this, the main radii of the ellipsoid can be derived
    # Get mass of vol0:
    mass = np.sum(vol0)
    # Get 3 objects containing the coordinates in x, y, and z direction:
    xx, yy, zz = np.meshgrid(np.arange(1, mysize[0] + 1), np.arange(1, mysize[1] + 1), np.arange(1, mysize[2] + 1))
    # Get the center of mass for all 3 directions:
    spx = np.sum(vol0 * xx / mass)
    spy = np.sum(vol0 * yy / mass)
    spz = np.sum(vol0 * zz / mass)
    # Transfrom coordinates so center of mass has coordinates (0,0,0):
    xx, yy, zz = xx - spx, yy - spy, zz - spz
    #  Get the moments of inertia when rotating the ellipsoid around the x, y, or z axis:
    Ix = np.sum(vol0 * (yy**2 + zz**2))
    Iy = np.sum(vol0 * (xx**2 + zz**2))
    Iz = np.sum(vol0 * (xx**2 + yy**2))
    # Get radius in x-direction, y-direction and z-diirection:
    aa = max(np.sqrt(5 / 2 / mass * max(Iz + Iy - Ix, 0)), 1)
    bb = max(np.sqrt(5 / 2 / mass * max(Iz + Ix - Iy, 0)), 1)
    cc = max(np.sqrt(5 / 2 / mass * max(Ix + Iy - Iz, 0)), 1)
    # Now obtain a mask denoting this ellipsoid:
    mm_ellipsoid = ((xx / aa)**2 + (yy / bb)**2 + (zz / cc)**2 <= 1)

    # PART 3
    # Collect non-zero data of vol0 outside mm_ellipsoid:
    data = vol0[(mm_ellipsoid == 0) & (vol0 > 0)]
    #  removing outliers:
    for _ in range(5):
        mymean, mystd = np.mean(data), np.std(data)
        data = data[data < mymean + level * mystd]
    # Upper noise limit:
    limit = np.max(data)
    # This is the start mask: remove all data with values up to limit:
    mmstart = (vol0 > limit)

    # PART 4
    # PART 4a:Remove isolated small areas in mmstart (mainly isolated pixels)
    # Get a 3x3x3 kernel highlighting the nearest neighbours only:
    kernel = np.ones(3, 1)
    kernel[2,2,2]=0
    # Get the number of non-zero nearest neighbours for each pixel:
    numneighbours = np.round(ralf_qMRI_conv_3D(mmstart,kernel))
    # leave only pixels where numneighbours is at least 25% of the kernel volume:
    mminterim = mmstart * (numneighbours >= 0.25 * np.sum(kernel))

    # PART 4b: hole filling (only if selected):
    # If no hole filling is added, mminterim is already the final mask:
    if max_holesize == 0:
        imamask = mminterim
    #  If hole filling is added:
    if max_holesize>0:
        mymm = mminterim
        # get maximum kernel size which is about max_holesize but must be odd number >=3:
        maxsize = 1 + 2 * np.ceil((max_holesize - 1) / 2)
        maxsize = max(int(maxsize), 3)
        # Loop across kernels with decreasing size from maxsize down to 3 (odd number only):
        for i in range(maxsize, 2, -2):
            kernel = np.ones((i, i, i))
            ksum = np.sum(kernel)
            myhelp = np.round(ralf_qMRI_conv_3D(mymm, kernel))
            newmm = mymm + (myhelp >= ksum * 3 / 4) * (mymm == 0)
            mymm = newmm
        
    imamask = mymm
    
    return imamask
