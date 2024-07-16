# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:28:25 2024

@author: mmari
"""

import numpy as np

from numpy.fft import fft2, ifft2, fftshift, ifftshift,fft,ifft
import math
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

    # Create a larger volume
    vol2 = np.zeros((vol.shape[0] + 2 * offs1, vol.shape[1] + 2 * offs2, vol.shape[2] + 2 * offs3))
    vol2[offs1:offs1+vol.shape[0], offs2:offs2+vol.shape[1], offs3:offs3+vol.shape[2]] = vol

    np1, cp1 = vol2.shape[0], math.floor(vol2.shape[0] / 2 +1)
    np2, cp2 = vol2.shape[1], math.floor(vol2.shape[1] / 2 +1)
    np3, cp3 = vol2.shape[2], math.floor(vol2.shape[2] / 2 +1)

    # Create a larger kernel
    kernel_large = np.zeros_like(vol2)
    kernel_large[cp1-offs1-1:cp1+offs1, cp2-offs2-1:cp2+offs2, cp3-offs3-1:cp3+offs3] = kernel


    myhelp = fftshift(fft2(ifftshift(vol2),axes=(0,1)))
    raw1 = fftshift(fft(ifftshift(myhelp), axis=2))
    myhelp = fftshift(fft2(ifftshift(kernel_large),axes=(0,1)))
    raw2 = fftshift(fft(ifftshift(myhelp), axis=2))
    raw3 = raw1 * raw2
    myhelp = fftshift(ifft2(ifftshift(raw3),axes=(0,1)))
    myhelp2 = fftshift(ifft(ifftshift(myhelp), axis=2))
    vol_conv = myhelp2[offs1:- offs1, offs2:- offs2, offs3:- offs3]

    if not np.any(np.iscomplex(vol)):
        vol_conv = np.real(vol_conv)

    return vol_conv
