# Hydro Robust Features
## Learning Robust Features for Scatter Removal and Reconstruction in Dynamic ICF X-Ray Tomography

Python implementation for the paper:

S. Gautam, M. L. Klasky, B. T. Nadiga, T. Wilcox, G. Salazar, and S. Ravishankar (2025). Learning Robust Features for Scatter Removal and Reconstruction in Dynamic ICF X-Ray Tomography. Optics Express - Issue Number.
arXiv: https://arxiv.org/abs/2408.12766


Overview
In this work, we propose feature-guided inverse modeling approaches for scatter removal and density reconstruction in dynamic inertial confinement fusion (ICF) X-ray radiography.
We introduce three learning-based methods:

PISLR (Physics-Informed Scatter Learning and Reconstruction)

SSLR (Scatter and Structure Learning and Reconstruction)

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
