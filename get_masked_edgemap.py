import numpy as np                      # For numerical operations and array handling
import matplotlib.pyplot as plt         # For plotting and visualization
from skimage import feature             # For edge detection (Canny filter)
import os                               # For file system operations
from utils.utils import *

# ----------------------------#
# Configuration Parameters    #
# ----------------------------#

Height = Width = 650                    # Image dimensions (square image)
frames = np.array([19, 23, 27, 31])     # Indices of frames (time steps) to process
sample_id = "sample001"                 # ID string used to identify the sample dataset
save_dir = "sample_output"              # Output directory for saving results


# ----------------------------#
# Load Sample Rho Sequence    #
# ----------------------------#

# Load clean density data (e.g., ground truth simulation) for a given sample.
# Expected shape: (T, H, W) where T = number of frames, H = height, W = width
rho_seq = np.load(f"sample_data/rho_clean_{sample_id}.npy")


# ----------------------------#
# Generate Canny Edgemaps     #
# ----------------------------#

# Create two circular masks:
# - Inner mask: preserves edges outside a small central region
# - Outer mask: removes edges far from center (outermost part)
inner_mask = masking(np.ones((Height, Width)), radius=50, mask_outer=False)
outer_mask = masking(np.ones((Height, Width)), radius=200, mask_outer=True)  # 200 = 400 // 2

# Allocate space to store edge maps for all frames
edgemap_rho_clean = np.zeros_like(rho_seq)

# Loop through each time frame
for frame_idx in range(len(frames)):
    rho = rho_seq[frame_idx]                     # Extract 2D density for current frame
    edge = feature.canny(rho, sigma=20)          # Compute Canny edge map (sigma controls smoothing)
    edge *= inner_mask                           # Apply inner circular mask
    edge *= outer_mask                           # Apply outer circular mask
    edgemap_rho_clean[frame_idx] = edge          # Store the masked edge map


# ----------------------------#
# Save Edgemap as npy         #
# ----------------------------#

# Save the final 3D array of edge maps to disk as a NumPy file
# Shape: (T, H, W)
np.save(os.path.join(save_dir, f"edgemap_canny_2d_{sample_id}.npy"), edgemap_rho_clean)
