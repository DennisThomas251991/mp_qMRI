# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:20:05 2024

@author: mmari
"""
import numpy as np
import ralf_qMRI_conv_3D
def ralf_qMRI_conv_3D_fast(vol,kernel):
    # convolution of 3D data set vol with 3D kernel
    # kernel size must be odd number in both directions!
    # kernel size must be smaller than size of vol!

    if vol.ndim != 3:
        raise ValueError('vol must be 3D')

    if kernel.ndim != 3:
        raise ValueError('kernel must be 3D')

    ks1, ks2, ks3 = kernel.shape

    if ks1 % 2 == 1:
        offs1 = (ks1 - 1) // 2
    else:
        raise ValueError('kernel size must be odd number')

    if ks2 % 2 == 1:
        offs2 = (ks2 - 1) // 2
    else:
        raise ValueError('kernel size must be odd number')

    if ks3 % 2 == 1:
        offs3 = (ks3 - 1) // 2
    else:
        raise ValueError('kernel size must be odd number')
    n1, n2, n3 = vol.shape

    if np.any(vol != 0):
        vol_conv=vol
    else:
        # profile along first axis:
        profile = np.sum(np.sum(vol, axis=1), axis=2)
        ll1 = max(np.min(np.where(profile > 0)), 1 + offs1)
        ul1 = min(np.max(np.where(profile > 0)), n1 - offs1)

        # profile along second axis:
        profile = np.sum(np.sum(vol, axis=0), axis=2)
        ll2 = max(np.min(np.where(profile > 0)), 1 + offs2)
        ul2 = min(np.max(np.where(profile > 0)), n2 - offs2)

        # profile along third axis:
        profile = np.sum(np.sum(vol, axis=0), axis=1)
        ll3 = max(np.min(np.where(profile > 0)), 1 + offs3)
        ul3 = min(np.max(np.where(profile > 0)), n3 - offs3)

        # Extract small volume and perform convolution for this:
        vol_small = vol[ll1 - offs1:ul1 + offs1, ll2 - offs2:ul2 + offs2, ll3 - offs3:ul3 + offs3]
        vol_conv_small = ralf_qMRI_conv_3D(vol_small, kernel)

        # Embed small result into large one with full matrix size:
        vol_conv = np.zeros_like(vol)
        vol_conv[ll1 - offs1:ul1 + offs1, ll2 - offs2:ul2 + offs2, ll3 - offs3:ul3 + offs3] = vol_conv_small

        if np.all(np.imag(vol) == 0):
            vol_conv = np.real(vol_conv)

    return vol_conv
    