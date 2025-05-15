# Code for training the self-supervised latent representation (PISLR) based density reconstruction model
# Author: Siddhant Gautam (2025)

import os                                 # Used for setting environment variables (e.g., selecting GPUs)
import sys                                # Used for interacting with the system and reading files
import numpy as np                        # For array operations
import torch                              # PyTorch for deep learning
import matplotlib.pyplot as plt           # For plotting training/validation loss and NRMSE curves
from unet import Unet                     
from utils import *               

# Clear the CUDA memory cache to avoid out-of-memory errors
torch.cuda.empty_cache()

# ------------------------ Configuration ------------------------ #

Height = Width = 650                      # Image dimensions (square)
clampval = 50                             # Maximum clamp value for rho
batch_size = 1                            # Batch size (set to 1 for single-sample updates)
learning_rate = 1e-3                      # Learning rate for the optimizers
nEpochs = 1000                            # Number of training epochs
nChannels = 1                             # Number of channels (grayscale image)
frames = np.array([19, 23, 27, 31])       # Time frames to be used for training
select_view = 0                           # Radiograph view index to use
num_alternate = 3                         # Number of alternating updates for Enet and Dnet
lamda = 1e3                               # Regularization coefficient
gpu_no = 0                                # GPU index to use

# Paths to data directories
hydro_path = '/egr/research-slim/shared/hydro_simulations/data/'
rad_path = '/egr/research-slim/shared/hydro_simulations/radiographs-conebeam-all-files/'
PATH = "sslr_model"                       # Prefix for saved model files and logs

# ------------------------ Data Preparation ------------------------ #

# Read training and validation filenames
training_filenames = open('training_filenames.txt', 'r').readlines()
validation_filenames = open('validation_filenames.txt', 'r').readlines()

# Dataset sizes
nTrain = len(training_filenames)
nValidation = len(validation_filenames)
Nframes = len(frames)                    # Number of time frames per sample

# Shuffle indices for randomness in data splitting
perm = np.random.permutation(nTrain + nValidation)
train_idx = perm[:nTrain]
val_idx = perm[nTrain:nTrain + nValidation]

# Print configuration details
print(f"Lambda: {lamda} | GPU: {gpu_no}")
print(f"Image Size: {Height}x{Width} | Frames: {Nframes}")
print(f"Train: {nTrain} | Val: {nValidation} | Epochs: {nEpochs}")
print("Predict frames:", frames)

# ------------------------ Device Setup ------------------------ #

# Set the GPU device to use
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------ Model Initialization ------------------------ #

# Initialize encoder and decoder networks using U-Net
Enet = Unet(in_chans=nChannels, out_chans=nChannels, chans=32 * nChannels).to(device).train()
Dnet = Unet(in_chans=nChannels, out_chans=nChannels, chans=32 * nChannels).to(device).train()

# Create separate optimizers for Enet and Dnet
optimizer_enet = torch.optim.Adam(Enet.parameters(), lr=learning_rate)
optimizer_dnet = torch.optim.Adam(Dnet.parameters(), lr=learning_rate)

# ------------------------ Loss Function ------------------------ #
def compute_loss(filename):
    """
    Compute the SSLR loss: 
    - term1: NRMSE between Dnet(Enet(rad_noisy)) and rho
    - term2: consistency between Enet(rad_noisy) and Enet(rad_clean)
    """
    # Load ground truth rho values
    rho_seq = get_rho(filename, frames, Height, Width, clampval=clampval, air_threshold=0)
    rho_clean = rho_seq.unsqueeze(1).float().to(device)  # [T, 1, H, W]

    # Extract simulation name from filename
    sim_name = filename[49:-3]

    # Load noisy and clean radiograph data
    rad_noisy_npz = np.load(f"{rad_path}noisy/noisy_{sim_name}.npz")['noisy_rad']
    rad_clean_npz = np.load(f"{rad_path}direct/direct_{sim_name}.npz")['direct_rad']

    # Select the desired view and convert to tensors
    rad_noisy = torch.tensor(rad_noisy_npz[:, :, select_view, :]).unsqueeze(1).float().to(device)
    rad_clean = torch.tensor(rad_clean_npz[:, :, select_view, :]).unsqueeze(1).float().to(device)

    # Clamp values to avoid log(0)
    rad_noisy[rad_noisy <= 0] = 1e-15
    rad_clean[rad_clean <= 0] = 1e-15

    # Forward pass through encoder and decoder
    Enet_rad_noisy = Enet(-torch.log(rad_noisy))
    Enet_rad_clean = Enet(-torch.log(rad_clean))
    Dnet_output = Dnet(Enet_rad_noisy)

    # Compute relative reconstruction loss (NRMSE)
    term1 = torch.linalg.norm(Dnet_output - rho_clean) / torch.linalg.norm(rho_clean)

    # Compute self-supervised consistency loss
    term2 = torch.sum(torch.abs(Enet_rad_noisy - Enet_rad_clean)) / torch.sum(torch.abs(Enet_rad_clean))

    # Return total loss and term1 for logging
    return term1 + lamda * term2, term1

# ------------------------ Training Loop ------------------------ #

# Initialize arrays to track loss and NRMSE
training_loss = np.zeros(nEpochs)
validation_loss = np.zeros(nEpochs)
nrmse_training = np.zeros(nEpochs)
nrmse_validation = np.zeros(nEpochs)

# Begin training over epochs
for epoch in range(nEpochs):
    tloss, vloss = 0, 0  # Accumulators for this epoch

    # --- Training Phase --- #
    for b_idx in range(0, nTrain - 2 * num_alternate, 2 * num_alternate):
        torch.cuda.empty_cache()

        for update_iter in range(2 * num_alternate):
            filename = training_filenames[b_idx + update_iter].strip()
            loss, term1 = compute_loss(filename)
            loss.backward()

            # Update loss and NRMSE
            with torch.no_grad():
                tloss += loss.item()
                nrmse_training[epoch] += term1.item() / nTrain

            # Alternate updates between Enet and Dnet
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

    # --- Validation Phase --- #
    with torch.no_grad():
        for b_idx in range(nValidation):
            filename = validation_filenames[b_idx].strip()
            loss, term1 = compute_loss(filename)
            vloss += loss.item()
            nrmse_validation[epoch] += term1.item() / nValidation
            torch.cuda.empty_cache()

    # Record average loss for this epoch
    training_loss[epoch] = tloss / nTrain
    validation_loss[epoch] = vloss / nValidation

    # Print progress
    print(f"Epoch {epoch+1:03d} | Train Loss: {training_loss[epoch]:.4f} | Val Loss: {validation_loss[epoch]:.4f}")

    # --- Save Model Checkpoints --- #
    torch.save(Enet.state_dict(), f"{PATH}_enet.pt")
    torch.save(Dnet.state_dict(), f"{PATH}_dnet.pt")

    # Save training metadata
    np.savez(f"{PATH}.npz", train_idx=train_idx, val_idx=val_idx,
             batch_size=batch_size, nChannels=nChannels, nEpochs=nEpochs,
             learning_rate=learning_rate, training_filenames=training_filenames,
             validation_filenames=validation_filenames,
             training_loss=training_loss, validation_loss=validation_loss)

    # --- Plot and Save Loss Curve --- #
    plt.figure()
    plt.plot(training_loss[:epoch+1], label='Training')
    plt.plot(validation_loss[:epoch+1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PATH}_loss.png")
    plt.close()

    # --- Plot and Save NRMSE Curve --- #
    plt.figure()
    plt.plot(nrmse_training[:epoch+1], label='Training')
    plt.plot(nrmse_validation[:epoch+1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('NRMSE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PATH}_nrmse.png")
    plt.close()
