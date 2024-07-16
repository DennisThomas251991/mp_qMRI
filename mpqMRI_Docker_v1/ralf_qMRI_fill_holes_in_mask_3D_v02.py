# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:37:38 2024

@author: mmari
"""
import numpy as np
from ralf_qMRI_conv_3D import ralf_qMRI_conv_3D
def ralf_qMRI_fill_holes_in_mask_3D_v02(vol, minnb):
    # Fills holes in a 3D binary mask vol
    # minnb is the minimum number of non-zero neighbors required for a pixel to be included in the growth process
    # Initial field "grow" contains zeros:
    
    grow = np.zeros_like(vol)
    # Step 1
    #The initial coarse step
    # vol is surrounded by a "frame" of zeros. This will be our initial guess for grow

    # find lowest/highest non-zero pixel in vol for each direction
    # Direction 1: get profile of vol along first direction:
 
    g1l, g1u = np.min(np.where(np.sum(np.sum(vol, axis=1,keepdims=True), axis=2) > 0)[0]), np.max(np.where(np.sum(np.sum(vol, axis=1,keepdims=True), axis=2) > 0)[0])
    g2l, g2u = np.min(np.where(np.sum(np.sum(vol, axis=0,keepdims=True), axis=2) > 0)[0]), np.max(np.where(np.sum(np.sum(vol, axis=0,keepdims=True), axis=2) > 0)[0])
    g3l, g3u = np.min(np.where(np.sum(np.sum(vol, axis=0,keepdims=True), axis=1) > 0)[0]), np.max(np.where(np.sum(np.sum(vol, axis=0,keepdims=True), axis=1) > 0)[0])
    # The outer frame comprises pixels from 1 to g1l-1 and from g1u+1 to end (in first direction)
    # To save time, we directly take the pixels from 1 to g1l and g1u to end and exlude from this pixels that belong to vol:
    # So initialise grow like this: 
    grow[0:g1l, :, :] = 1
    grow[g1u:, :, :] = 1
    grow[:, 0:g2l, :] = 1
    grow[:, g2u:, :] = 1
    grow[:, :, 0:g3l] = 1
    grow[:, :, g3u:] = 1
    # But exclude pixels where vol is not zero:
    grow *= (vol == 0)
    #  END OF STEP 1

    # Step 2
    ks_list = [21, 11, 3]
    for ks in ks_list:
        kernel = np.ones((ks, ks, ks))
        myhelp2 = np.round(ralf_qMRI_conv_3D(vol, kernel))
        while True:
            #  Convolve kernel with grow to get myhelp1
            myhelp1 = np.round(ralf_qMRI_conv_3D(grow, kernel))
            #  Pixels to be added to grow are outside grow (grow==0), myhelp2 is 0 and myhelp1 is at least 1:
            add_to_grow = (myhelp1 >= 1) & (myhelp2 == 0) & (grow == 0)
            #  Leave loop if there is nothing more to add:
            if not np.any(add_to_grow):
                break
            grow += add_to_grow

    # Step 3
    kernel = np.ones((3, 3, 3))
    while True:
        myhelp = np.round(ralf_qMRI_conv_3D(grow, kernel))
        add_to_grow = (myhelp >= minnb) & (vol == 0) & (grow == 0)
        if not np.any(add_to_grow):
            break
        grow += add_to_grow
    #  Output is inverted version of grow:
    vol2 = 1 - grow
    return vol2

