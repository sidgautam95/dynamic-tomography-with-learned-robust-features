"""
Module to spin 2D density profiles into 3D volumes using axisymmetric rotation.

Original Author:
    Gabriel Maliakal (Michigan State University)

Contributors / Editors:
    Siddhant Gautam (Michigan State University)

Description:
    This script takes a 2D density projection and creates a 3D axisymmetric
    volume by revolving the profile around the central axis.
"""

import numpy as np
import cv2

def rotate_rho_3d(rho_clean):
    """
    Generates a 3D spun volume from a 2D density image by applying 
    a polar transform and remapping it back to Cartesian space along the z-axis.

    Parameters:
        rho_clean (np.ndarray): 2D density image (Height x Width)

    Returns:
        np.ndarray: 3D spun image volume (Height x Width x Width)
    """
    Height, Width = rho_clean.shape
    vol2 = np.zeros((Height, Width, Width), dtype=rho_clean.dtype)
    vol2[Height // 2, :, :] = rho_clean  # place slice in central yz-plane

    spun_img = np.zeros_like(vol2)

    center = (Height / 2, Width / 2)
    max_radius = np.sqrt((Height / 2) ** 2 + (Width / 2) ** 2)

    for cc in range(Width):  # loop over z-axis
        cart_img = vol2[:, :, cc].copy()

        # Transform to polar coordinates
        polar_image = cv2.linearPolar(cart_img, center, max_radius, cv2.WARP_FILL_OUTLIERS)

        # Collapse to a radial maximum projection (circular sweep)
        polar_image = np.repeat(polar_image.max(axis=0, keepdims=True), len(polar_image), axis=0)

        # Transform back to Cartesian coordinates
        spun_img[:, :, cc] = cv2.linearPolar(polar_image, center, max_radius, cv2.WARP_INVERSE_MAP)

    return spun_img
