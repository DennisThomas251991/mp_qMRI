# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:58:14 2024

@author: mmari
"""
import numpy as np
from ralf_qMRI_conv_3D import ralf_qMRI_conv_3D
from ralf_qMRI_calc_local_mean_and_std import ralf_qMRI_calc_local_mean_and_std
from ralf_qMRI_fill_holes_in_mask_3D_v02 import ralf_qMRI_fill_holes_in_mask_3D_v02
from ralf_qMRI_get_edges_of_mask import ralf_qMRI_get_edges_of_mask
from ralf_qMRI_conv_3D_fast import ralf_qMRI_conv_3D_fast
def ralf_qMRI_improve_B1_map(b1map,minb1,maxb1,nlayers):
    # Function for improving quality of input B1 maps
    # Steps:
    # 1: B1 is restricted to range between minb1 and maxb1
    #    Recommended: minb1=0.7, maxb1=1.3
    # 2. Outliers are removed via comparison with average of the non-zero next neighbours
    # 3. Solitary areas in the B1 map are removed
    # 4. B1 is smoothed via convolution with a 3x3x3 kernel
    # The above steps yield b1map_improved
    # Additional step:
    # 5. B1 map is extended by adding nlayers layers around the non-zero part of b1map_improved
    #    Recommended: nlayers=5
    # STEP 1: Restrict B1 map to chosen range:
    b1map=b1map*(np.logical_and(b1map>minb1,b1map<maxb1))
    # STEP 2: Outlier removal:
    # First, define a kernel covering the 26 nearest neighbours
    kernel=np.ones((3,3,3))
    # Inner pixel is zero as only nearest neighbours are included in kernel:
    kernel[1,1,1]=0
    # The outlier correction is performed recursively in 2 steps:
    for i in range(1, 3):
        # Integrate B1 across nearest neighbors:
        b1_integral = ralf_qMRI_conv_3D(b1map, kernel)
        
        # Get the number of pixels that had non-zero contribution:
        numpix = np.round(ralf_qMRI_conv_3D((b1map > 0).astype(int), kernel))
        
        # Calculate mean B1, only for pixels that are non-zero in the original B1 map:
        meanb1 = b1_integral / (np.finfo(float).eps + numpix) * (numpix > 0) * (b1map > 0)
        
        # We now have 2 B1 maps: the original b1map and meanb1, the latter containing average values across nearest neighbors
        # To remove outliers, we replace single pixels in b1map by the respective pixels in meanb1,
        # provided b1map deviates too much from meanb1
        # For this, calculate the quotient of b1map and mean1...
        qq = b1map / (np.finfo(float).eps + meanb1) * (b1map > 0) * (meanb1 > 0)
        # ...and now the relative deviation of b1map from mean1:
        mydev = np.abs(qq - 1) * (qq > 0)
        
        # Find non-zero values and sort them in ascending order:
        data = mydev[mydev > 0]
        data2 = np.sort(data)
        
        # We assume that the largest 10% of these deviations denote outliers
        # Get the minimum value of the largest 10%:
        maxdev = data2[int(0.9 * len(data2))]

        # If mydev exceeds maxdev, the pixel is considered an outlier and replaced by meanb1:
        b1map = b1map * (mydev <= maxdev) + meanb1 * (mydev > maxdev)
        
    # STEP 3: Remove solitary areas in b1map
    # These are pixels for which less than 5 nearest neighbors are non-zero
    cond=(b1map>0).astype(int)
    myhelp=np.round(ralf_qMRI_conv_3D(cond, kernel))
    b1map[(b1map>0) & (myhelp<5)]=0
    
    # STEP 4: Smooth B1 map
    kernel = np.ones((3,3,3))
    b1_integral = np.abs(ralf_qMRI_conv_3D_fast(b1map, kernel))
    vol_new=(b1map > 0).astype(int)
    numpix = np.round(np.abs(ralf_qMRI_conv_3D_fast(vol_new, kernel)))

    b1map_sm = (b1_integral / (np.finfo(float).eps + np.float64(numpix)))* (numpix > 0)

    b1map_sm = b1map_sm * ((b1map_sm > minb1) & (b1map_sm < maxb1)) * (b1map > 0)

    # STEP 5: Extension of B1 map
    b1map_curr = b1map_sm

    for _ in range(nlayers):
        edge1, edge2 = ralf_qMRI_get_edges_of_mask((b1map_curr>0).astype(int));
        b1map_fitted, _ = ralf_qMRI_calc_local_mean_and_std(b1map_curr, (b1map_curr > 0).astype(int), 5)
        b1map_ext=b1map_curr + b1map_fitted*edge2;

        b1map_ext = b1map_ext * (b1map_ext > minb1) * (b1map_ext < maxb1)
        b1map_curr = b1map_ext

    # b1map_improved_extended is now b1map_sm wherever this is non-zero and b1map_ext elsewhere:
    b1map_improved_extended = b1map_sm * (b1map_sm > 0).astype(int) + b1map_ext * (b1map_sm == 0)

    # For export of non-extended B1:
    maske_fill = ralf_qMRI_fill_holes_in_mask_3D_v02((b1map_sm > 0).astype(int), 9)
    b1map_improved = b1map_improved_extended * maske_fill

    # This means: b1map_improved is almost everywhere b1map_sm, but takes values from b1map_improved_extended in the holes

    return b1map_improved,b1map_improved_extended
