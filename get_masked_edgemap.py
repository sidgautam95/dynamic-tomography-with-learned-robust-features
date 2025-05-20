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
# Utility: Circular Masking   #
# ----------------------------#
def masking(img, radius=60, mask_outer=False):
    """
    Applies a circular mask to the input image.

    Parameters:
        img (np.ndarray): Input image to mask.
        radius (int): Radius of the circular region to mask.
        mask_outer (bool): 
            - If True: zeros out everything *outside* the circle.
            - If False: zeros out everything *inside* the circle.

    Returns:
        np.ndarray: Masked image.
    """
    img_masked = np.copy(img)                       # Make a copy of the input to avoid altering original
    center = img.shape[0] // 2                      # Assume image is square; center is at (H/2, W/2)
    
    for x in range(img.shape[0]):                   # Iterate through each pixel
        for y in range(img.shape[1]):
            # Mask logic: preserve only points inside or outside a circle
            if mask_outer:
                if (x - center) ** 2 + (y - center) ** 2 > radius ** 2:
                    img_masked[x, y] = 0            # Outside the circle → zero
            else:
                if (x - center) ** 2 + (y - center) ** 2 < radius ** 2:
                    img_masked[x, y] = 0            # Inside the circle → zero
    return img_masked


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
