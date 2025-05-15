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
