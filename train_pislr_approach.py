# Code for training the physics inspired latent representation (PISLR) based density reconstruction model
# Author: Siddhant Gautam (2025)

import os                                  # For environment variable management
import sys                                 # For file path and system operations
import numpy as np                         # For numerical operations
import matplotlib.pyplot as plt            # For plotting loss/NRMSE curves
import torch                               # For deep learning operations
from skimage import filters, feature       # For image processing (not directly used here)
from unet import Unet                      # U-Net model architecture
from utils import *                  # Function to load and process hydro data

# ------------------------ Environment Setup ------------------------ #
torch.cuda.empty_cache()                   # Clear GPU cache to avoid OOM issues

# ------------------------ File Paths ------------------------ #
hydro_path = '/egr/research-slim/shared/hydro_simulations/data/'  # Path to hydro data
rad_path = '/egr/research-slim/shared/hydro_simulations/radiographs-conebeam-all-files/'  # Path to radiographs

# Load training and validation filenames
training_filenames = open('training_filenames.txt', 'r').readlines()
validation_filenames = open('validation_filenames.txt', 'r').readlines()

# ------------------------ Configuration ------------------------ #
lamda = 1e4                                # Regularization parameter for edge map term
gpu_no = 2                                 # GPU index to use

Height = Width = 650                       # Image dimensions
clampval = 50                              # Clamp max value for rho
batch_size = 1                             # Batch size
learning_rate = 1e-3                       # Learning rate
nEpochs = 100                              # Number of epochs
nChannels = 1                              # Number of input/output channels
frames = np.array([19, 23, 27, 31])        # Selected time steps
select_view = 0                            # Radiograph view index
num_alternate = 3                          # Alternate update steps for Enet and Dnet

# Number of samples
nTrain = len(training_filenames)
nValidation = len(validation_filenames)
nFiles = nTrain + nValidation
Nframes = len(frames)

# File name prefix for saving outputs
PATH = "pislr_model"

# ------------------------ Device Setup ------------------------ #
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)                     # Set GPU visibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Set computation device

# ------------------------ Shuffle Data ------------------------ #
perm = np.random.permutation(nFiles)        # Random permutation of indices
train_idx = perm[:nTrain]                   # Training indices
val_idx = perm[nTrain:nTrain + nValidation] # Validation indices

# ------------------------ Logging Info ------------------------ #
print(f"Regularization λ: {lamda}")
print(f"GPU number: {gpu_no}")
print(f"Image size: {Height}x{Width}, Frames: {Nframes}")
print(f"Train: {nTrain}, Val: {nValidation}, Epochs: {nEpochs}, LR: {learning_rate}")
print("Predict frames:", frames)

# ------------------------ Model Setup ------------------------ #
Enet = Unet(in_chans=nChannels, out_chans=nChannels, chans=32 * nChannels).to(device).train()
Dnet = Unet(in_chans=nChannels, out_chans=nChannels, chans=32 * nChannels).to(device).train()

# Optimizers for encoder and decoder
optimizer_enet = torch.optim.Adam(Enet.parameters(), lr=learning_rate)
optimizer_dnet = torch.optim.Adam(Dnet.parameters(), lr=learning_rate)

# ------------------------ Utility: Optional Masking ------------------------ #
def masking(img, radius=60, mask_outer=False):
    """
    Create a circular mask for removing inside/outside pixels.
    """
    img_masked = np.copy(img)
    centre = img.shape[0] // 2
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if mask_outer:
                if (x - centre)**2 + (y - centre)**2 > radius**2:
                    img_masked[x, y] = 0
            else:
                if (x - centre)**2 + (y - centre)**2 < radius**2:
                    img_masked[x, y] = 0
    return img_masked

# ------------------------ Loss Function ------------------------ #
def compute_loss(filename):
    """
    Compute total loss: reconstruction error + edge regularization term.
    """
    rho_seq = get_rho(filename, frames, Height, Width, clampval=50, air_threshold=0)
    rho_clean = rho_seq.unsqueeze(1).float().to(device)  # Add channel dimension

    sim_name = filename[49:-3]  # Extract sim ID

    # Load radiograph data
    noisy_npz = np.load(f"{rad_path}noisy/noisy_{sim_name}.npz")['noisy_rad']
    clean_npz = np.load(f"{rad_path}direct/direct_{sim_name}.npz")['direct_rad']

    rad_noisy = torch.tensor(noisy_npz[:, :, select_view, :]).unsqueeze(1).float().to(device)
    rad_clean = torch.tensor(clean_npz[:, :, select_view, :]).unsqueeze(1).float().to(device)

    # Prevent log(0)
    rad_noisy[rad_noisy <= 0] = 1e-15
    rad_clean[rad_clean <= 0] = 1e-15

    # Forward passes
    Enet_rad_noisy = Enet(-torch.log(rad_noisy))
    Dnet_output = Dnet(Enet_rad_noisy)

    # Load Canny edge map
    edgemap_path = f"/egr/research-slim/gautamsi/shared/hydro_simulations/edgemap-canny/edgemap_canny_2d_{sim_name}.npy"
    edgemap = torch.tensor(np.load(edgemap_path)).unsqueeze(1).float().to(device)

    # Reconstruction loss (NRMSE)
    term1 = torch.linalg.norm(Dnet_output - rho_clean) / torch.linalg.norm(rho_clean)

    # Edge loss
    term2 = torch.sum(torch.abs(Enet_rad_noisy - edgemap)) / torch.sum(torch.abs(edgemap))

    return term1 + lamda * term2, term1

# ------------------------ Training Loop ------------------------ #
training_loss = np.zeros(nEpochs)
validation_loss = np.zeros(nEpochs)
nrmse_training = np.zeros(nEpochs)
nrmse_validation = np.zeros(nEpochs)

for epoch in range(nEpochs):
    tloss, vloss = 0, 0

    # -------- Training -------- #
    for b_idx in range(0, nTrain - 2 * num_alternate, 2 * num_alternate):
        torch.cuda.empty_cache()

        for update_iter in range(2 * num_alternate):
            filename = training_filenames[b_idx + update_iter].strip()
            loss, term1 = compute_loss(filename)
            loss.backward()

            with torch.no_grad():
                tloss += loss.item()
                nrmse_training[epoch] += term1.item() / nTrain

            # Alternate between Enet and Dnet
            if update_iter < num_alternate:
                Enet.requires_grad_(True)
                Dnet.requires_grad_(False)
                optimizer_enet.step()
                optimizer_enet.zero_grad()
            else:
                Enet.requires_grad_(False)
                Dnet.requires_grad_(True)
                optimizer_dnet.step()
                optimizer_dnet.zero_grad()

            torch.cuda.empty_cache()

    # -------- Validation -------- #
    with torch.no_grad():
        for b_idx in range(nValidation):
            filename = validation_filenames[b_idx].strip()
            loss, term1 = compute_loss(filename)
            vloss += loss.item()
            nrmse_validation[epoch] += term1.item() / nValidation
            torch.cuda.empty_cache()

    # Store epoch results
    training_loss[epoch] = tloss / nTrain
    validation_loss[epoch] = vloss / nValidation

    # Print epoch summary
    print(f"Epoch {epoch+1:03d} | Training Loss: {training_loss[epoch]:.4f} | Validation Loss: {validation_loss[epoch]:.4f}")

    # -------- Save Models -------- #
    torch.save(Enet.state_dict(), f"{PATH}_enet.pt")
    torch.save(Dnet.state_dict(), f"{PATH}_dnet.pt")

    # -------- Save Logs -------- #
    np.savez(f"{PATH}.npz", train_idx=train_idx, val_idx=val_idx,
             batch_size=batch_size, nChannels=nChannels, nEpochs=nEpochs,
             learning_rate=learning_rate, training_filenames=training_filenames,
             validation_filenames=validation_filenames, training_loss=training_loss,
             validation_loss=validation_loss)

    # -------- Plot Loss Curve -------- #
    plt.figure()
    plt.plot(training_loss[:epoch+1], label='Training')
    plt.plot(validation_loss[:epoch+1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PATH}_loss.png")
    plt.close()

    # -------- Plot NRMSE Curve -------- #
    plt.figure()
    plt.plot(nrmse_training[:epoch+1], label='Training')
    plt.plot(nrmse_validation[:epoch+1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('NRMSE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PATH}_nrmse.png")
    plt.close()
