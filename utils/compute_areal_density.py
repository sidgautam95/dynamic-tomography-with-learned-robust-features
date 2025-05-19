"""
Cone-beam Computation of Areal Density Using ASTRA Toolbox
===========================================================

- Computes areal density for tantalum and air components from 3D density volumes.
- Generates direct radiographs using exponential attenuation with mass attenuation coefficients.
"""

import numpy as np
import astra


def get_areal_density_astra(rho_3d, num_views=1, dl=0.25, dso=1330, dsd=5250):
    """
    Computes areal density from 3D density volume using cone-beam CT geometry in ASTRA.

    Parameters:
        rho_3d (np.ndarray): 3D density volume (H x W x W) in g/cm³
        num_views (int): Number of projection angles
        dl (float): Pixel spacing in cm
        dso (float): Distance from source to origin in mm
        dsd (float): Distance from source to detector in mm

    Returns:
        np.ndarray: Areal density image (num_views x H x W) in g/cm²

    References:
        ASTRA Toolbox 3D Cone-Beam Geometry:
        https://astra-toolbox.com/docs/geom3d.html
    """
    img = np.copy(rho_3d) * 1e-3  # Convert from g/cm³ to g/mm³
    img[img < 0] = 0              # Remove negative densities

    Height, Width, _ = img.shape
    dod = dsd - dso  # Distance from origin to detector
    detector_pixel_size = dl * dsd / dso  # Detector pixel size in mm

    detector_spacing_x = detector_spacing_y = 1  # unit spacing
    det_row_count = Height
    det_col_count = Width
    angles = np.linspace(0, np.pi, num_views, endpoint=False)

    # Normalize source-detector distances to 1mm detector pixel spacing
    source_origin = (dso + dod) / detector_pixel_size
    origin_det = 0  # Detector is at the origin

    vol_geom = astra.create_vol_geom(Height, Width, Width)
    proj_geom = astra.create_proj_geom('cone', detector_spacing_x, detector_spacing_y,
                                       det_row_count, det_col_count, angles,
                                       source_origin, origin_det)
    
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    W = astra.OpTomo(proj_id)

    # Rotate image so ASTRA projects along the correct axis
    spun = img.transpose((2, 1, 0))

    # Compute areal density
    areal_density = (W @ spun.reshape(-1)).reshape((det_row_count, num_views, det_col_count))
    areal_density *= dl * 100  # Convert from mm to cm scale (g/mm² to g/cm²)

    return areal_density.transpose((1, 0, 2))  # shape: [views, height, width]


def simulate_radiograph(areal_density_ta, areal_density_air, collimator, mac_ta=4e-2, mac_air=3e-2):
    """
    Generates a direct radiograph from areal density volumes.

    Parameters:
        areal_density_ta (np.ndarray): Areal density of tantalum (views x H x W)
        areal_density_air (np.ndarray): Areal density of air (views x H x W)
        collimator (np.ndarray): Areal mass of collimator (H x W)
        mac_ta (float): Mass attenuation coefficient for tantalum (cm²/g)
        mac_air (float): Mass attenuation coefficient for air (cm²/g)

    Returns:
        np.ndarray: Direct radiograph (views x H x W)
    """
    num_views, Height, Width = areal_density_ta.shape
    direct_rad = np.zeros_like(areal_density_ta)

    for view in range(num_views):
        direct_rad[view] = np.exp(
            -mac_ta * (areal_density_ta[view] + collimator)
            -mac_air * areal_density_air[view]
        )

    # Scale using MCNP calibration constant
    return direct_rad * 3.201e-4
