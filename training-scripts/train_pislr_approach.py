# Code for training the physics inspired latent representation (PISLR) based density reconstruction model
# Author: Siddhant Gautam (2025)

import os                                  # For environment variable management
import sys                                 # For file path and system operations
import numpy as np                         # For numerical operations
import matplotlib.pyplot as plt            # For plotting loss/NRMSE curves
import torch                               # For deep learning operations
from skimage import filters, feature       # For image processing (not directly used here)
from models.unet import Unet                      # U-Net model architecture
from utils.utils import *                  # Function to load and process hydro data

# ------------------------ Environment Setup ------------------------ #
torch.cuda.empty_cache()                   # Clear GPU cache to avoid OOM issues


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

# ------------------------ Loss Function ------------------------ #
def compute_loss(filename=None):
    """
    Compute total loss
    
    Returns:
        total_loss (torch.Tensor): Combined reconstruction and edge loss.
        term1 (torch.Tensor): Normalized reconstruction error (NRMSE).
    """

    # ----------- Load pre-generated numpy arrays from 'sample_data/' folder ----------- #

    # Load ground truth density (rho) sequence
    # Expected shape: (T, H, W) where T = number of frames (e.g., 4), H=W=650
    rho_clean = torch.tensor(np.load("sample_data/rho_clean" + filename + ".npy"))  # Load " + filename + ".npy as torch tensor
    rho_clean = rho_clean.unsqueeze(1).float().to(device)           # Add channel dimension: (T, 1, H, W)

    # Load noisy radiograph input
    # Expected shape: (T, H, W) matching rho_clean
    rad_noisy = torch.tensor(np.load("sample_data/rad_noisy" + filename + ".npy"))
    rad_noisy = rad_noisy.unsqueeze(1).float().to(device)           # (T, 1, H, W)

    # Load the precomputed Canny edgemaps (from the clean training densities)
    edgemap = torch.tensor(np.load("sample_data/edgemap" + filename + ".npy"))
    edgemap = edgemap.unsqueeze(1).float().to(device)               # (T, 1, H, W)

    # ----------- Safety clamp to avoid log(0) or log(negative) ----------- #
    rad_noisy[rad_noisy <= 0] = 1e-15
    rad_clean[rad_clean <= 0] = 1e-15

    # ----------- Forward pass through the encoder (Enet) ----------- #
    # Get Noisy Features
    Enet_rad_noisy = Enet(-torch.log(rad_noisy))                    # Output: (T, 1, H, W)

    # ----------- Forward pass through the decoder (Dnet) ----------- #
    # Apply decoder output on the features to get reconstructed density
    Dnet_output = Dnet(Enet_rad_noisy)                              # Output: (T, 1, H, W)

    # ----------- Compute Normalized RMSE reconstruction loss ----------- #
    # term1 = ||Dnet_output - rho_clean|| / ||rho_clean||
    term1 = torch.linalg.norm(Dnet_output - rho_clean) / torch.linalg.norm(rho_clean)

    # ----------- Compute Edge Consistency Loss ----------- #
    # This encourages encoder outputs to be structurally aligned with the Canny edge map
    # term2 = mean absolute error between encoder output and edge map
    term2 = torch.sum(torch.abs(Enet_rad_noisy - edgemap)) / torch.sum(torch.abs(edgemap))

    # ----------- Total Loss: Weighted sum of reconstruction + edge regularization ----------- #
    total_loss = term1 + lamda * term2

    return total_loss, term1



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
    np.savez(f"{PATH}.npz",
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
