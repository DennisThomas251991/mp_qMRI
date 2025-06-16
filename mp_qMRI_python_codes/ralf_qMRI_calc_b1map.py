# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:31:50 2023

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
import numpy.matlib as matlib

def ralf_qMRI_calc_b1map(vol1, vol2, fa, tr, pa, ndum, wm, gm, csf, t1_wm, t1_gm, t1_csf, t1_other, maske):
    """
% function b1map=ralf_qMRI_calc_b1map(vol1,vol2,fa,tr,pa,ndum,wm,gm,csf,t1_wm,t1_gm,t1_csf,t1_other,maske)
% Calculation of B1-Map according to Volz et al., NeuroImage 49 (2010) 3015–3026
% vol1/vol2 are the data sets acquired without/with preparation pulse
% vol1 and vol2 are based on FLASH with centric PE
% WARNING: in vol1 and vol2, the PE direction must be in the first dimension!!
% fa is the flip (excitation) angle in deg (usually 11)
% tr is TR in msec (usually 11) 
% pa is the preparation angle in deg (usually 45)
% ndum is the number of initial dummy scans (usually 3)
% wm, gm, csf are probability maps that a pixel belongs to respective tissue
% t1_wm,t1_gm,t1_csf,t1_other are the T1 values (in msec) of WM, GM, CSF, other tissue (at 3T usually 900, 1400, 4500, 1100)
% maske is the mask denoting the area where B1 is to be calculated
% Attention: vol1, vol2, wm, gm, csf, maske must have same matrix size!
% b1map is the B1 map (normalized: b1map=1.0 where the real angle matches the nominal one)

    """

    nslc = vol1.shape[2]
    nphs = vol1.shape[0]

    alpeff = fa * 0.87 * np.pi / 180 #change to double
    bet = pa * np.pi / 180
    corr1 = np.zeros(nphs)
    corr2 = np.zeros(nphs)
    raw1c = np.zeros_like(vol1)
    raw2c = np.zeros_like(vol2)

    for i in range(4):
        if i == 0:
            t1 = t1_wm
            probmap = wm
        elif i == 1:
            t1 = t1_gm
            probmap = gm
        elif i == 2:
            t1 = t1_csf
            probmap = csf
        elif i == 3:
            t1 = t1_other
            probmap = np.ones_like(wm) - wm - gm - csf

        t1s = 1 / (1 / t1 - 1 / tr * np.log(np.cos(alpeff)))

        for i in range(nphs):
            time = (2 * np.abs(i+1 - (nphs / 2 + 1)) + ndum) * tr
            corr1[i] = 1 / (t1s / t1 + (1 - t1s / t1) * np.exp(-time / t1s))
            corr2[i] = np.cos(bet) / (t1s / t1 + (np.cos(bet) - t1s / t1) * np.exp(-time / t1s))

        corrfeld1 =np.tile(corr1[:,np.newaxis,np.newaxis],(1,vol1.shape[1], vol1.shape[2]))
        raw1 = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(vol1 * probmap, axes=0), axis=0), axes=0)
        raw1c = raw1c+ (raw1 * corrfeld1)


        corrfeld2 = np.tile(corr2[:,np.newaxis,np.newaxis],(1,vol1.shape[1], vol1.shape[2]))
        raw2 = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(vol2 * probmap, axes=0), axis=0), axes=0)
        raw2c = raw2c + (raw2 * corrfeld2)


    vol1c = np.abs(np.fft.ifftshift(np.fft.fft(np.fft.fftshift(raw1c), axis=0)))
    vol2c = np.abs(np.fft.ifftshift(np.fft.fft(np.fft.fftshift(raw2c), axis=0)))

    quot = vol2c / (np.finfo(float).eps + vol1c)
    quot = quot * (quot < 1)
    anglemap = 180 / np.pi * np.arccos(quot) * maske
    anglemap = anglemap * (anglemap < 90)
    b1map = anglemap / pa

    return b1map
