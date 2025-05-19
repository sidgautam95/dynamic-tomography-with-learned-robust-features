import numpy as np
from add_noise_to_radiograph import *

# ------------------------------#
# Load Clean Radiograph         #
# ------------------------------#

sim_name = 'data_ta_2d_profile0.vel0.mgrg00.s10.cs0.cv0.ptwg00'

# Load the clean radiograph file generated earlier
data = np.load('clean_radiograph_'+sim_name + '.npz')
direct_rad = data['direct_rad']
frames = data['frames']
num_views = data['num_views']

# Allocate memory for noisy radiographs
noisy_rad = np.zeros_like(direct_rad)

# ------------------------------#
# Noise Parameters Configuration#
# ------------------------------#

# --- Scatter-related noise ---
sigma_scatter = 10              # Std dev of correlated scatter Gaussian kernel
scatter_scaling = 0.2            # Scaling factor for scatter component
scatter_polynomial_order = 1     # Degree of background polynomial for tilt field

# --- Gamma noise parameters ---
gammascaling = 1                 # Scaling of the correlated gamma noise component
gammalevel = None                # Expected gamma photons per pixel (sampled if None, ~39000-50000)

# --- Photon noise parameters ---
photonscaling = 1                # Scaling of the correlated photon noise component
photonlevel = None               # Expected photon counts per pixel (sampled if None, ~350-450)

# --- Source blur (anisotropic Gaussian) ---
x_stddevlevel = None             # Std dev along x-axis of the source blur kernel (sampled if None, ~1.0-3.1)
y_stddevlevel = None             # Std dev along y-axis of the source blur kernel (sampled if None, ~1.0-3.1)

# --- Background tilt noise ---
siglevel = None                  # Strength of the tilt background field (sampled if None, ~0.5-1.6)
x_tiltlevel = None               # Coefficient for tilt in x-direction (sampled if None, ~-3.9e-5 to 3.9e-6)
y_tiltlevel = None               # Coefficient for tilt in y-direction (sampled if None, ~-3.9e-5 to 3.9e-6)

# --- Kernel paths ---
gamma_kernel_path = '../kernels/gamma_kernel.dat'            # Path to precomputed gamma kernel (301x301)
photon_kernel_path = '../kernels/photon_kernel.dat'          # Path to precomputed photon kernel (81x81)
detector_kernel_path = '../kernels/detector_blur_az.dat'     # Path to detector blur kernel (201x201)

# ------------------------------#
# Generate Noisy Radiographs    #
# ------------------------------#

for frame_idx in range(direct_rad.shape[0]):
    for view in range(num_views):
        # Call noise function with all explicitly passed parameters
        noisy_rad[frame_idx, view] = add_noise_to_radiograph(
            direct=direct_rad[frame_idx, view],
            x_stddevlevel=x_stddevlevel,
            y_stddevlevel=y_stddevlevel,
            sigma_scatter=sigma_scatter,
            scatter_scaling=scatter_scaling,
            siglevel=siglevel,
            x_tiltlevel=x_tiltlevel,
            y_tiltlevel=y_tiltlevel,
            gammalevel=gammalevel,
            photonlevel=photonlevel,
            scatter_polynomial_order=scatter_polynomial_order,
            gammascaling=gammascaling,
            photonscaling=photonscaling,
            gamma_kernel_path=gamma_kernel_path,
            photon_kernel_path=photon_kernel_path,
            detector_kernel_path=detector_kernel_path
        )

# ------------------------------#
# Save Noisy Radiograph with Metadata #
# ------------------------------#

np.savez(f'noisy_radiograph_{sim_name}.npz',
         noisy_rad=noisy_rad, **{k: data[k] for k in data.files if k != 'direct_rad'})

print('✅ Noisy radiograph saved.')
