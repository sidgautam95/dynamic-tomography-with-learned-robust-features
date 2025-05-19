import numpy as np
from utils import *
from rotate_rho_3d import *
from generate_direct_rad import *

# ------------------------------#
# Parameters and Configuration  #
# ------------------------------#

hydro_path = '/egr/research-slim/shared/hydro_simulations/data/'  # Path to hydro simulations
collimator_file = '../kernels/RMI_Collimator_ArealMass.dat'       # Collimator areal mass file
frames = np.array([19, 23, 27, 31])                               # Frame indices to process
Height = Width = 650                                              # Output image size
air_threshold = 10                                                # Threshold to separate air and metal

dl = 0.25      # Detector pixel size (mm)
dso = 1330     # Distance source to object (mm)
dsd = 5250     # Distance source to detector (mm)
mac_ta = 4e-2  # Mass attenuation coefficient for tantalum
mac_air = 3e-2 # Mass attenuation coefficient for air
num_views = 1  # Number of views (angles)

# ------------------------------#
# Load and Crop Collimator      #
# ------------------------------#

collimator = np.genfromtxt(collimator_file).reshape((880, 880))  # Load and reshape collimator
# Crop the collimator to match image size
collimator = collimator[(880 - Height)//2:(880 + Height)//2, (880 - Width)//2:(880 + Width)//2]

# ------------------------------#
# Radiograph Generation         #
# ------------------------------#

sim_name = 'data_ta_2d_profile0.vel0.mgrg00.s10.cs0.cv0.ptwg00'   # Simulation name
filename = hydro_path + sim_name + '.nc'                          # Full simulation file path
rho_seq = get_rho(filename, frames, Height, Width)                # Load rho sequence

# Allocate memory for clean radiographs
direct_rad = np.zeros((len(frames), num_views, Height, Width))

# Loop over frames
for frame_idx in range(len(frames)):
    rho_2d = rho_seq[frame_idx].cpu().detach().numpy()            # Get 2D density for frame
    rho_3d = rotate_rho_3d(rho_2d)                                # Rotate to 3D

    # Extract metal (tantalum) and air components
    rho_ta_3d = np.copy(rho_3d)
    rho_ta_3d[rho_ta_3d < air_threshold] = 0                      # Zero out non-metal regions

    rho_air_3d = np.copy(rho_3d)
    rho_air_3d[rho_air_3d >= air_threshold] = 0                   # Zero out metal regions

    # Compute areal density projections for each
    areal_density_ta = get_areal_density_astra(rho_ta_3d, num_views, dl, dso, dsd)
    areal_density_air = get_areal_density_astra(rho_air_3d, num_views, dl, dso, dsd)

    # Generate clean radiograph
    direct_rad[frame_idx] = generate_direct_rad(areal_density_ta, areal_density_air, collimator, mac_ta, mac_air)

# Save clean radiograph with metadata
np.savez(f'clean_radiograph_{sim_name}.npz',
         direct_rad=direct_rad, air_threshold=air_threshold, frames=frames,
         num_views=num_views, dl=dl, dso=dso, dsd=dsd, mac_ta=mac_ta, mac_air=mac_air)

print('✅ Clean radiograph saved.')
