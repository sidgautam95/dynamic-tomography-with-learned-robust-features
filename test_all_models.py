import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from unet import Unet
from utils import *
from add_noise_to_radiograph import add_noise_to_radiograph

# Set GPU device
gpu_no = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained model
def load_network(path):
    net = Unet(in_chans=1, out_chans=1)
    net.load_state_dict(torch.load(path))
    net.to(device)
    net.eval()
    print(f"Loaded network from {path}")
    return net

# Paths to saved models
Enet_sslr = load_network("sslr_model_enet.pt")
Dnet_sslr = load_network("sslr_model_dnet.pt")

Enet_ulr = load_network("ulr_model_enet.pt")
Dnet_ulr = load_network("ulr_model_dnet.pt")

Enet_pislr = load_network("pislr_model_enet.pt")
Dnet_pislr = load_network("pislr_model_dnet.pt")

# Load validation filenames
validation_filenames = open('validation_filenames.txt', 'r').readlines()

# Extract numpy array from torch tensor
def get_arr(tensor):
    return torch.squeeze(tensor).cpu().detach().numpy()

# Data paths
hydro_path = '/egr/research-slim/shared/hydro_simulations/data/'
rad_path = '/egr/research-slim/shared/hydro_simulations/radiographs-conebeam-all-files/'

# Image/frame configuration
frames = np.array([19, 23, 27, 31])
Height = Width = 650
nFrames = len(frames)
select_view = 0

# Output directories
rmse_dir = 'rmse-unet2d-1000files/'
recon_dir = 'reconstructed_outputs/'
os.makedirs(recon_dir, exist_ok=True)

# Number of test samples
ntest = len(validation_filenames)

# Initialize RMSE arrays
rmse_sslr = np.zeros((len(frames), ntest))
rmse_ulr = np.zeros((len(frames), ntest))
rmse_pislr = np.zeros((len(frames), ntest))

# Define noise configurations
sigma_scatter = 10 
scatter_scaling = 0.2
scatter_polynomial_order = 1

# Loop over test samples
for test_idx in range(ntest):
    filename = validation_filenames[test_idx].strip()
    print(f"\nProcessing sample {test_idx+1}/{ntest} - {filename}")

    # Load clean rho ground truth
    rho_seq = get_rho(filename, frames, Height, Width, clampval=50, air_threshold=0)
    rho_clean = rho_seq.unsqueeze(1).float().to(device)
    img_rho_clean = get_arr(rho_clean)

    # Load and noise clean radiograph
    sim_name = filename[49:-3]
    rad_clean_allview = np.load(rad_path + f'direct/direct_{sim_name}.npz')['direct_rad']
    rad_noisy_allview = np.zeros_like(rad_clean_allview)

    for i, frame in enumerate(frames):
        rad_clean = rad_clean_allview[i, :, select_view, :]
        rad_noisy_allview[i, :, select_view, :] = add_noise_to_radiograph(
            rad_clean,
            sigma_scatter=sigma_scatter,
            scatter_scaling=scatter_scaling,
            scatter_polynomial_order=scatter_polynomial_order
        )

    # Convert noisy radiograph to tensor
    rad_noisy = torch.tensor(rad_noisy_allview[:, :, select_view, :])
    rad_noisy[rad_noisy <= 0] = 1e-15
    rad_noisy = rad_noisy.unsqueeze(1).float().to(device)

    # Inference through networks
    recon_sslr = get_arr(Dnet_sslr(Enet_sslr(-torch.log(rad_noisy))))
    recon_ulr = get_arr(Dnet_ulr(Enet_ulr(-torch.log(rad_noisy))))
    recon_pislr = get_arr(Dnet_pislr(Enet_pislr(-torch.log(rad_noisy))))

    # Save reconstructed outputs
    np.save(os.path.join(recon_dir, f'sample{test_idx}_sslr.npy'), recon_sslr)
    np.save(os.path.join(recon_dir, f'sample{test_idx}_ulr.npy'), recon_ulr)
    np.save(os.path.join(recon_dir, f'sample{test_idx}_pislr.npy'), recon_pislr)

    # Optional: compute and log RMSE here (commented for brevity)
    # for f in range(nFrames):
    #     rmse_sslr[f, test_idx] = compute_rmse(img_rho_clean[f], recon_sslr[f])
    #     rmse_ulr[f, test_idx] = compute_rmse(img_rho_clean[f], recon_ulr[f])
    #     rmse_pislr[f, test_idx] = compute_rmse(img_rho_clean[f], recon_pislr[f])
