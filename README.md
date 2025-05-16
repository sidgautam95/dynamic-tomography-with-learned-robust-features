# Hydro Robust Features
## Learning Robust Features for Scatter Removal and Reconstruction in Dynamic ICF X-Ray Tomography

Python implementation for the paper:

S. Gautam, M. L. Klasky, B. T. Nadiga, T. Wilcox, G. Salazar, and S. Ravishankar (2025). Learning Robust Features for Scatter Removal and Reconstruction in Dynamic ICF X-Ray Tomography. Optics Express - Issue Number.

arXiv preprint [arXiv:2408.12766](https://arxiv.org/abs/2408.12766) (2024)
 


## Overview
In this work, we propose a robust feature extraction technique for scatter removal and density reconstruction in dynamic inertial confinement fusion (ICF) X-ray radiography. An encoder is used to extract robust features from noisy/corrupted X-ray projections and the decoder reconstructs the underlying density image from the features extracted by the encoder. We explore three options for the latent-space representation of features: physics-inspired supervision, self-supervision, and no supervision. We find that variants based on self-supervised and physics-inspired supervised features perform better over a range of unknown scatter and noise. The loss functions for these latent representation models are as follows:


1. Physics-inspired supervised latent representation (PISLR) approach:

   
\begin{equation}
    \underset{\bm{\theta}_1, \bm{\theta}_2}{\min} \,\,  \mathbb{E}_{(\bm{\rho},\,\mathbf{T})} \frac{\|D_{\bm{\theta}_2}(E_{\bm{\theta}_1}(\mathbf{T}))- \bm{\rho}\|_2}{\| \bm{\rho}\|_2} + \lambda_{\text{PISLR}} \frac{\|E_{\bm{\theta}_1}(\mathbf{T}) - \mathbf{M} \odot E_f(\bm{\rho})\|_1}{\|\mathbf{M} \odot E_f(\bm{\rho})\|_1}.   
    \label{eq:masked}
\end{equation}

2. Self-Supervised Latent Representation (SSLR) Approach:

3. Unsupervised Latent Representation (ULR) Approach:




Here, $E_{\bm{\theta}_1}$ and $D_{\bm{\theta}_2}$ are the encoder and decoder networks with parameters $\bm{\theta}_1$ and $\bm{\theta}_2$, respectively. $\mathbf{D}$ and $\mathbf{T}$ are the clean and noisy radiographs, respectively, $\bm{\rho}$ is the underlying clean density that leads to the clean radiograph. $\lambda_{\text{PISLR}}$ is the hyperparameter controlling weighting of two terms. The expectation in the loss above is with respect to the distribution of the densities and radiographs.


Here, $E_{\theta}$ and 𝐷𝜽2 are the encoder and decoder networks with parameters 𝜽1 and 𝜽2, respectively.
D and T are the clean and noisy radiographs, respectively, 𝝆 is the underlying clean density
that leads to the clean radiograph. 𝜆PISLR is the hyperparameter controlling weighting of two
terms. The expectation in the loss above is with respect to the distribution of the densities and
radiographs. 𝐸 𝑓 ( 𝝆) is the binary edgemap of the clean density above obtained after applying
a Canny edge detection filter [28] on the clean density. M is an operator that masks out the
gas-metal interface in the images retaining primarily the shock feature.

Both approaches utilize scatter-aware features extracted from the input radiographs to guide robust reconstruction under different noise conditions.

We further investigate the robustness of the proposed methods against varying levels of gamma and photon noise, as well as generalization to unseen scatter conditions.

Key Contributions:
Physics-guided feature extraction and integration into inverse models.

Robustness analysis under varying gamma and photon noise levels.

Generalization study to different scatter models.

Comparison against standard ULR (Unsupervised Learned Reconstruction) baselines.

Code Structure
The code includes the following components:

Data Simulation
Scripts to generate synthetic radiographs with varying scatter, gamma, and photon noise levels.

Feature Extraction Models
Feature extractor (UNet) trained to predict scatter maps or scatter features from noisy radiographs.

Reconstruction Models

PISLR model
SSLR model
ULR model

Evaluation Scripts
Evaluation of RMSE of reconstructed densities under different noise settings and scatter models.

Running the Experiments
Data Simulation:
Generate simulated radiographs with desired scatter and noise settings:


python simulate_radiographs.py --scatter_model [gaussian/phys] --gamma_scaling X --photon_scaling Y
Training Feature Extractor:
Train the scatter feature predictor network:


python train_feature_extractor.py --config configs/feature_unet.yaml
Training Reconstruction Models:
Train PISLR, SSLR, and ULR models:


python train_reconstruction.py --config configs/pislr.yaml
python train_reconstruction.py --config configs/sslr.yaml
python train_reconstruction.py --config configs/ulr.yaml
Evaluation:
Evaluate the reconstruction performance under different noise conditions:


python evaluate_noise_robustness.py --config configs/eval_noise.yaml
Evaluate generalization to unseen scatter conditions:


python
RMSE vs Gamma and Photon Noise:

Generalization to unseen scatter models:

Reconstructed densities under different noise conditions:

Datasets
Synthetic datasets used in this work are generated using the provided scripts.
You may optionally use your own experimental radiographs formatted as 2D projections and adjust the scatter models accordingly.

Contact
The code is provided for reproducible research purposes.
If you have any questions, suggestions, or encounter any issues running the code, please feel free to contact:

Siddhant Gautam (gautamsi@msu.edu)
Saiprasad Ravishankar (ravisha3@msu.edu)
