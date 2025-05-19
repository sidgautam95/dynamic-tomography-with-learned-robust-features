# Code for training the unsupervised latent representation (ULR) based density reconstruction model
# Author: Siddhant Gautam (2025)

import os                            # For setting environment variables
import sys                           # For interacting with the system and file paths
import numpy as np                   # For numerical operations and arrays
import matplotlib.pyplot as plt      # For plotting training/validation curves
import torch                         # Main PyTorch library
from unet import Unet                # Custom U-Net model architecture
from utils import get_rho            # Function to load ground truth data (rho)

# Clear any residual memory from previous CUDA sessions
torch.cuda.empty_cache()

# ------------------------ Paths ------------------------ #
PATH = "ulr_model"  # Path prefix to save model and log files

# ------------------------ Load Filenames ------------------------ #
training_filenames = open('training_filenames.txt', 'r').readlines()  # Load training file list
validation_filenames = open('validation_filenames.txt', 'r').readlines()  # Load validation file list

# ------------------------ Hyperparameters ------------------------ #
gpu_no = 5                            # Index of GPU to use
Height = 650                          # Image height
Width = Height                        # Image width (assumed square)
clampval = 50                         # Max clamp value for rho
crop_pixel = Height // 2              # Not used here, but could be for cropping
batch_size = 1                        # Batch size
learning_rate = 1e-3                  # Learning rate for optimizers
nEpochs = 100                         # Number of training epochs
nChannels = 1                         # Number of input/output channels
nClasses = 1                          # Number of output classes (unused)
frames = np.array([19, 23, 27, 31])   # Time frames to extract from simulation
select_view = 0                       # Which radiograph view to use
split_ratio = 0.1                     # Not used here, could be for auto split
num_alternate = 3                     # Alternate updates between encoder and decoder

nTrain = 10                           # Limit training size (for debugging/speed)
nValidation = 1                       # Limit validation size
nFiles = nTrain + nValidation         # Total number of files
Nframes = len(frames)                 # Number of frames per sample

# ------------------------ Device Setup ------------------------ #
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)                      # Restrict GPU visibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device

# Print training summary
print(f"GPU number: {gpu_no}")
print(f"Simulation Parameters\nSize: {Height}x{Width}, Frames: {Nframes}, Training: {nTrain}, "
      f"Validation: {nValidation}, Batch Size: {batch_size}, Channels: {nChannels}, "
      f"Epochs: {nEpochs}, LR: {learning_rate}")
print("Predict frames:", frames)

# ------------------------ Model Setup ------------------------ #
# Initialize encoder and decoder models with U-Net architecture
Enet = Unet(in_chans=nChannels, out_chans=nChannels, chans=32 * nChannels).to(device).train()
Dnet = Unet(in_chans=nChannels, out_chans=nChannels, chans=32 * nChannels).to(device).train()

# Setup optimizers for Enet and Dnet
optimizer_enet = torch.optim.Adam(Enet.parameters(), lr=learning_rate)
optimizer_dnet = torch.optim.Adam(Dnet.parameters(), lr=learning_rate)

# ------------------------ Loss Function ------------------------ #
def compute_loss(filename):
    """
    Compute the reconstruction loss using sample .npy files instead of simulation data.

    Args:
        filename (str): Identifier string to match sample file names (e.g., 'sample001').

    Returns:
        tuple: (total_loss, NRMSE) where total_loss = NRMSE (no regularization used).
    """

    # ---------------- Load rho ground truth sequence ---------------- #
    # Expected shape: (T, H, W)
    rho_path = f"sample_data/rho_clean_{filename}.npy"
    rho_seq = torch.tensor(np.load(rho_path))                      # Load as torch tensor
    rho_clean = rho_seq.unsqueeze(1).float().to(device)            # Add channel dim → (T, 1, H, W)

    # ---------------- Load noisy radiograph ---------------- #
    rad_noisy_path = f"sample_data/rad_noisy_{filename}.npy"
    rad_noisy = torch.tensor(np.load(rad_noisy_path))              # Shape: (T, H, W)
    rad_noisy = rad_noisy.unsqueeze(1).float().to(device)          # Shape: (T, 1, H, W)

    # ---------------- Load clean radiograph ---------------- #
    rad_clean_path = f"sample_data/rad_clean_{filename}.npy"
    rad_clean = torch.tensor(np.load(rad_clean_path))              # Shape: (T, H, W)
    rad_clean = rad_clean.unsqueeze(1).float().to(device)          # Shape: (T, 1, H, W)

    # ---------------- Avoid log(0) ---------------- #
    rad_noisy[rad_noisy <= 0] = 1e-15
    rad_clean[rad_clean <= 0] = 1e-15

    # ---------------- Forward passes ---------------- #
    Enet_rad_noisy = Enet(-torch.log(rad_noisy))                   # Getting encdoer features
    Dnet_output = Dnet(Enet_rad_noisy)                             # Getting decoder output

    # ---------------- Compute NRMSE loss ---------------- #
    # Normalized root mean square error: ||prediction - ground_truth|| / ||ground_truth||
    term1 = torch.linalg.norm(Dnet_output - rho_clean) / torch.linalg.norm(rho_clean)

    # Return both total loss and term1 (since they are identical here)
    return term1, term1


# ------------------------ Training Loop ------------------------ #
# Initialize loss trackers
training_loss = np.zeros(nEpochs)
validation_loss = np.zeros(nEpochs)
nrmse_training = np.zeros(nEpochs)
nrmse_validation = np.zeros(nEpochs)

# Begin training
for epoch in range(nEpochs):
    tloss, vloss = 0, 0  # Reset epoch losses

    # ---- Training Phase ---- #
    for b_idx in range(0, nTrain - num_alternate, num_alternate * 2):
        torch.cuda.empty_cache()  # Free GPU memory

        for update_iter in range(num_alternate * 2):
            filename = training_filenames[b_idx + update_iter].strip()  # Get current file
            loss, term1 = compute_loss(filename)
            loss.backward()  # Backpropagate

            with torch.no_grad():
                tloss += loss.item()
                nrmse_training[epoch] += term1.item() / nTrain

            # Alternate training encoder and decoder
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

    # ---- Validation Phase ---- #
    with torch.no_grad():
        for b_idx in range(nValidation):
            filename = validation_filenames[b_idx].strip()
            loss, term1 = compute_loss(filename)
            vloss += loss.item()
            nrmse_validation[epoch] += term1.item() / nValidation

    # Save loss history
    training_loss[epoch] = tloss / nTrain
    validation_loss[epoch] = vloss / nValidation

    print(f"Epoch {epoch+1:03d} | Training Loss: {training_loss[epoch]:.4f} | Validation Loss: {validation_loss[epoch]:.4f}")

    # ---- Save Model ---- #
    torch.save(Enet.state_dict(), f"{PATH}_enet.pt")
    torch.save(Dnet.state_dict(), f"{PATH}_dnet.pt")

    # ---- Save Logs ---- #
    np.savez(f"{PATH}.npz", batch_size=batch_size,
             nChannels=nChannels, nEpochs=nEpochs, learning_rate=learning_rate,
             training_filenames=training_filenames, validation_filenames=validation_filenames,
             training_loss=training_loss, validation_loss=validation_loss)

    # ---- Plot Loss Curve ---- #
    plt.figure()
    plt.plot(training_loss[:epoch+1], label="Training")
    plt.plot(validation_loss[:epoch+1], label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PATH}_loss.png")
    plt.close()

    # ---- Plot NRMSE Curve ---- #
    plt.figure()
    plt.plot(nrmse_training[:epoch+1], label="Training")
    plt.plot(nrmse_validation[:epoch+1], label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('NRMSE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{PATH}_nrmse.png")
    plt.close()
