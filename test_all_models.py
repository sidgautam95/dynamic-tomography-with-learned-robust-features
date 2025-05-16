import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from unet import Unet
from utils import *
from add_noise_to_radiograph import add_noise_to_radiograph

# Set GPU device
gpu_no = 6
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
Enet_sslr = load_network("../saved-models-repo/sslr_model_enet.pt")
Dnet_sslr = load_network("../saved-models-repo/sslr_model_dnet.pt")

Enet_ulr = load_network("../saved-models-repo/ulr_model_enet.pt")
Dnet_ulr = load_network("../saved-models-repo/ulr_model_dnet.pt")

Enet_pislr = load_network("../saved-models-repo/pislr_model_enet.pt")
Dnet_pislr = load_network("../saved-models-repo/pislr_model_dnet.pt")


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


# -------------------------------#
# Define all noise configurations
# -------------------------------#
# In population noise parameters
sigma_scatter = 10                 # Std dev of correlated scatter Gaussian kernel
scatter_scaling = 0.2              # Scaling factor for scatter component
scatter_polynomial_order = 1       # Degree of background polynomial for tilt field

gammascaling = 1                   # Scaling of the correlated gamma noise component
photonscaling = 1                  # Scaling of the correlated photon noise component

# Leave as None to sample defaults, or set specific values for reproducibility
x_stddevlevel = None               # Std dev of source blur along x-axis (~1-3.1 if None)
y_stddevlevel = None               # Std dev of source blur along y-axis (~1-3.1 if None)

siglevel = None                    # Strength of the tilt background field (~0.5-1.6 if None)
x_tiltlevel = None                 # Tilt coefficient in x-direction (~-3.9e-5 to 3.9e-6 if None)
y_tiltlevel = None                 # Tilt coefficient in y-direction (~-3.9e-5 to 3.9e-6 if None)

gammalevel = None                  # Expected gamma photons per pixel (~39000-50000 if None)
photonlevel = None                 # Expected photon counts per pixel (~350-450 if None)

gamma_kernel_path = '../kernels/gamma_kernel.dat'
photon_kernel_path = '../kernels/photon_kernel.dat'
detector_kernel_path = '../kernels/detector_blur_az.dat'


filename = '/egr/research-slim/shared/hydro_simulations/data/data_ta_2d_profile2.vel5.mgrg01.s12.cs2.cv2.ptwg02.nc'

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

    # Apply full noise model with all parameters
    rad_noisy_allview[i, :, select_view, :] = add_noise_to_radiograph(
        rad_clean,
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

# Convert noisy radiograph to tensor
rad_noisy = torch.tensor(rad_noisy_allview[:, :, select_view, :])
rad_noisy[rad_noisy <= 0] = 1e-15
rad_noisy = rad_noisy.unsqueeze(1).float().to(device)

# Inference through networks
recon_sslr = get_arr(Dnet_sslr(Enet_sslr(-torch.log(rad_noisy))))
recon_ulr = get_arr(Dnet_ulr(Enet_ulr(-torch.log(rad_noisy))))
recon_pislr = get_arr(Dnet_pislr(Enet_pislr(-torch.log(rad_noisy))))

# Save reconstructed outputs
np.save('im_sslr.npy', recon_sslr)
np.save('im_ulr.npy', recon_ulr)
np.save('im_pislr.npy', recon_pislr)
