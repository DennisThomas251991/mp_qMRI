# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:28:25 2024

@author: mmari
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def ralf_qMRI_conv_3D(vol, kernel):
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

    vol2 = np.zeros((vol.shape[0] + 2 * offs1, vol.shape[1] + 2 * offs2, vol.shape[2] + 2 * offs3))
    vol2[offs1:vol.shape[0] + offs1, offs2:vol.shape[1] + offs2, offs3:vol.shape[2] + offs3] = vol
    np1, cp1 = vol2.shape[0], (vol2.shape[0] // 2) + 1
    np2, cp2 = vol2.shape[1], (vol2.shape[1] // 2) + 1
    np3, cp3 = vol2.shape[2], (vol2.shape[2] // 2) + 1

    kernel_large = np.zeros_like(vol2)
    kernel_large[cp1 - offs1:cp1 + offs1, cp2 - offs2:cp2 + offs2, cp3 - offs3:cp3 + offs3] = kernel

    myhelp = fftshift(fftn(ifftshift(vol2)))
    raw1 = fftshift(fftn(ifftshift(myhelp), axes=(2,)))
    myhelp = fftshift(fftn(ifftshift(kernel_large)))
    raw2 = fftshift(fftn(ifftshift(myhelp), axes=(2,)))
    raw3 = raw1 * raw2
    myhelp = fftshift(ifftn(ifftshift(raw3)))
    myhelp2 = fftshift(ifftn(ifftshift(myhelp), axes=(2,)))
    vol_conv = myhelp2[offs1:vol.shape[0] + offs1, offs2:vol.shape[1] + offs2, offs3:vol.shape[2] + offs3]

    if not np.any(np.iscomplex(vol)):
        vol_conv = np.real(vol_conv)

    return vol_conv
