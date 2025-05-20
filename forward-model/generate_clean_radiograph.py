import numpy as np
from utils.utils import *
from utils.rotate_rho_3d import *
from compute_areal_density import *

# ------------------------------#
# Parameters and Configuration  #
# ------------------------------#
frames = np.array([19, 23, 27, 31])                               # Frame indices to process
Height = Width = 650                                              # Output image size
air_threshold = 10                                                # Threshold to separate air and metal

dl = 0.25      # Detector pixel size (mm)
dso = 1330     # Distance source to object (mm)
dsd = 5250     # Distance source to detector (mm)
mac_ta = 4e-2  # Mass attenuation coefficient for tantalum
mac_air = 3e-2 # Mass attenuation coefficient for air
num_views = 1  # Number of views (angles)

# Replace by your collimator file of same dimensions as the density image
collimator = np.load("collimator.npy") 

# Load rho sequence from sample file (shape: [T, H, W])
# Replace by your density image
rho_seq = np.load(f"rho_clean_.npy")  # shape: (T, H, W)


# Allocate memory for clean radiographs
direct_rad = np.zeros((len(frames), num_views, Height, Width))

# Loop over frames
for frame_idx in range(len(frames)):
    rho_2d = rho_seq[frame_idx]                                   # Get 2D density for frame
    
    rho_3d = rotate_rho_3d(rho_2d)                                # Rotate to 3D (assumes that the object is axisymmetric

    # Extract metal (tantalum) and air components
    rho_ta_3d = np.copy(rho_3d)
    rho_ta_3d[rho_ta_3d < air_threshold] = 0                      # Zero out non-metal regions

    rho_air_3d = np.copy(rho_3d)
    rho_air_3d[rho_air_3d >= air_threshold] = 0                   # Zero out metal regions

    # Compute areal density projections for each
    areal_density_ta = get_areal_density_astra(rho_ta_3d, num_views, dl, dso, dsd)
    areal_density_air = get_areal_density_astra(rho_air_3d, num_views, dl, dso, dsd)

    # Generate clean radiograph
    direct_rad[frame_idx] = simulate_radiograph(areal_density_ta, areal_density_air, collimator, mac_ta, mac_air)

# Save clean radiograph with metadata
np.savez(f'clean_radiograph_{sim_name}.npz',
         direct_rad=direct_rad, air_threshold=air_threshold, frames=frames,
         num_views=num_views, dl=dl, dso=dso, dsd=dsd, mac_ta=mac_ta, mac_air=mac_air)

print('✅ Clean radiograph saved.')
