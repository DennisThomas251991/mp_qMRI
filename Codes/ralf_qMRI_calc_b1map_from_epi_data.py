# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:57:49 2024

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
import math
from ralf_qMRI_get_gradients_in_mask import ralf_qMRI_get_gradients_in_mask
def ralf_qMRI_calc_b1map_from_epi_data(vol1, vol2, b0, b0_slice_resol, epi_slice_thickness, te, sign_gradient, btp, maske):
    """
    Calculation of B1-Map from EPI data acquired with 45deg and 90deg excitation.

    Parameters:
    - vol1/vol2: EPI data sets acquired with 45deg and 90deg excitation, respectively.
    - b0: B0 map in rad/sec with the same size, coverage, and spatial resolution as the EPI data.
    - b0_slice_resol: Interslice distance (in mm) of the B0 map (slice center to slice center), including the interslice gap.
    - epi_slice_thickness: Slice thickness (in mm) of the EPI data, without interslice gaps.
    - te: Echo time TE (in ms) of the EPI sequence.
    - sign_gradient: Sign of the B0 gradient in slice direction when calculating the xmap for B0 correction.
    - btp: Bandwidth-time product of the EPI excitation pulse.
    - maske: Mask denoting the area where B1 is to be calculated.

    Returns:
    - b1map: B1 map (normalized: b1map=1.0 where the real angle matches the nominal one).

    Reference:
    Noeth et al., Magn Reson Med. 2023;90:103-116, DOI: 10.1002/mrm.29632

    """

    # Set the P matrix required for slice profile and B0 corrections:
    polyorder = 7
    P = np.array([
        [9.6278e-01, 1.3703e-01, 2.1145e-02, -3.5724e-02, 9.5990e-03, -1.1457e-03, 6.5315e-05, -1.4501e-06],
        [-8.2277e-01, 5.1865e-01, -2.2348e-01, 5.7841e-02, -8.9368e-03, 8.0267e-04, -3.8563e-05, 7.6510e-07],
        [6.0244e-01, -6.7345e-01, 3.3312e-01, -8.5546e-02, 1.2581e-02, -1.0679e-03, 4.8669e-05, -9.2195e-07],
        [-1.3125e+00, 1.3137e+00, -5.9778e-01, 1.4789e-01, -2.1324e-02, 1.7895e-03, -8.0996e-05, 1.5279e-06],
        [-1.7412e-01, 9.3008e-02, -1.2286e-02, -2.3715e-03, 9.3669e-04, -1.1803e-04, 6.7938e-06, -1.5101e-07],
        [5.1417e-02, -1.3307e-02, -1.1423e-02, 6.0876e-03, -1.2320e-03, 1.2626e-04, -6.5307e-06, 1.3564e-07],
        [2.4038e-01, -2.7786e-01, 1.3329e-01, -3.3036e-02, 4.6870e-03, -3.8427e-04, 1.6944e-05, -3.1109e-07],
        [-1.7083e-01, 2.1472e-01, -1.0769e-01, 2.7565e-02, -4.0189e-03, 3.3766e-04, -1.5230e-05, 2.8565e-07]
    ])

    # Calculate scaled quotient of data sets:
    qq = np.sqrt(2) * vol1 / (np.finfo(float).eps + vol2) * maske

    # Derive theoretical B1 value:
    # Theory: qq = 1/sqrt(2)/cos(pi/4*b1theo)
    # So: cos(pi/4*b1theo) = 1/sqrt(2)/qq
    # Or: b1theo = 4/pi*acos(1/sqrt(2)/qq);
    # 1/sqrt(2)/qq must not exceed 1, so qq must be at least 1/sqrt(2)

    # Calculate b1theo for these values only:
    b1theo = 4/math.pi * np.arccos(1 / np.sqrt(2) / (np.finfo(float).eps + qq) * (qq >= 1/np.sqrt(2))) * maske
    # b1theo has to be corrected for slice profile and B0 effects

    # Get B0 gradient:
    gx, gy, gz = ralf_qMRI_get_gradients_in_mask(b0, (np.abs(b0) > 0))

    # B0 is in rad/sec, so gz is the B0 change from slice to slice in rad/sec
    # Convert gz into uT/m:
    # First, we calculate the B0 change from slice to slice in uT
    # The change in Hz follows from gz via division by 2*pi
    # The frequency f follows from f=42.5764e6*B0 where B0 is given in Tesla or f=42.5764*B0 where B0 is given in uT
    # So the change in uT follows from the change in Hz via division by 42.5764
    # In summary:
    myhelp = gz / (2 * math.pi) / 42.5764
    # myhelp is now the B0 change from slice to slice in uT
    # For getting the gradient in uT/m, we have to divide myhelp by the interslice distance of the B0 map (in m)
    # This distance is given by b0_slice_resol / 1000
    gz2 = myhelp / (b0_slice_resol / 1000)
    # gz2 is now in uT/m

    # We now get the xmap from x = 2*gamma*Gz*TE*ST
    # Here, gamma = 42.5764e6Hz/T, Gz is in T/m, TE is in sec
    # Since we have gz2 in uT/m, te in ms and epi_slice_thickness in mm, this becomes:
    # xmap=2*gz2*te*epi_slice_thickness*42.5764e-6;
    # Furthermore, we have to use the right sign given by sign_gradient (+1 for axial, -1 for sagittal data):

    xmap = sign_gradient * 2 * gz2 * te * epi_slice_thickness * 42.5764e-6
    
    # Limit xmap values to be between +1 and -1
    xmap=xmap*(abs(xmap)<=1)+1*(xmap>1)-1*(xmap<-1)
    
    # Get correction map for the chosen btp and xmap
    corrmap = np.zeros_like(xmap)
    for k in range(1, polyorder + 2):
        for l in range(1, polyorder + 2):
            corrmap += P[k - 1, l - 1] * np.power(xmap, k - 1) * np.power(btp, l - 1)
    
    # Get corrected B1 map
    b1map = corrmap * b1theo
    return b1map
