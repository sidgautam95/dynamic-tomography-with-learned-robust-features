import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt

def get_rho(filename, frames, Height = 880, Width = 880, clampval=50, air_threshold=0):
    
    #########################
    # Parameters 
    # filename: Name of the .nc simulation file containing the 2D density sequence
    # clampval: Value at which the image is to be clamped
    # Height, Width: Dimensions of the image
    
    rho=np.zeros((len(frames),Height,Width)) # Initializing the density array sequence
    sim = xr.open_dataarray(filename)  # Reading .nc xarray file
    
    for idx, frame in enumerate(frames):
        a=sim.isel(t=frame);
        a=a[:Height//2, :Width//2] # Cropping the image 
        ar=np.concatenate((np.flipud(a),a), axis=0) # Flipping array to get the right part
        al=np.concatenate((np.flipud(np.fliplr(a)),np.fliplr(a)), axis=0) # Flipping array to get the left part
        # Combining to form a full circle from a quarter image
        rho[idx]=np.concatenate((al,ar), axis=1)
        
    rho=torch.tensor(rho,dtype=torch.float)
    rho=torch.clamp(rho, min=None, max=clampval) # Clamping value at rho

    # Removing air density values by thresholding everything below 10 to 0
    rho[rho<air_threshold]=0

    return rho


def torch_nrmse(img_gt,img_recon):
    return torch.linalg.norm(img_gt-img_recon)/torch.linalg.norm(img_gt)

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

