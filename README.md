# Hydro Robust Features

## Learning Robust Features for Scatter Removal and Reconstruction in Dynamic ICF X-Ray Tomography

Python implementation for the paper:

**Siddhant Gautam, Marc L. Klasky, Balasubramanya T. Nadiga, Trevor Wilcox, Gary Salazar, and Saiprasad Ravishankar.**  
*"Learning Robust Features for Scatter Removal and Reconstruction in Dynamic ICF X-Ray Tomography."*  
*Opt. Express* **33**, 12345-12367 (2025).  
DOI: [https://doi.org/xxxx](https://doi.org/xxxc)  

arXiv preprint [arXiv:2408.12766](https://arxiv.org/abs/2408.12766) (2024)

---

## Overview

In this work, we propose a robust feature extraction technique for scatter removal and density reconstruction in dynamic inertial confinement fusion (ICF) X-ray radiography.  
An encoder is used to extract robust features from noisy or corrupted X-ray projections, and a decoder reconstructs the underlying density image from the extracted features.
The block diagram of encoder-decoder based architecture with four different frames of a temporal sequence used for training can be given by:
<p align="center">
  <img src="https://github.com/sidgautam95/hydro-robust-features/blob/main/figures/encoder_decoder_block_stack.png" alt="Block diagram of encoder-decoder based architecture" width="700"/>
</p>


We explore three strategies for the latent-space representation of features:
1. **Physics-Inspired Supervised Latent Representation (PISLR)**
2. **Self-Supervised Latent Representation (SSLR)**
3. **Unsupervised Latent Representation (ULR)**

We find that self-supervised and physics-inspired supervised feature models consistently perform better across a range of unknown scatter and noise conditions.

The loss functions for these models are as follows:
- **PISLR Approach:**
<p align="center">
  <img src="https://github.com/sidgautam95/hydro-robust-features/blob/main/figures/pislr_loss_equation.png" alt="PISLR Loss Equation" width="500"/>
</p>

- **SSLR Approach:**
<p align="center">
  <img src="https://github.com/sidgautam95/hydro-robust-features/blob/main/figures/sslr_loss_equation.png" alt="SSLR Loss Equation" width="500"/>
</p>

- **ULR Approach:**  
<p align="center">
  <img src="https://github.com/sidgautam95/hydro-robust-features/blob/main/figures/ulr_loss_equation.png" alt="ULR Loss Equation" width="300"/>
</p>

Where:
- \( E_{\theta_1} \) and \( D_{\theta_2} \) are the encoder and decoder networks with parameters \( \theta_1 \) and \( \theta_2 \).
- \( D \) and \( T \) are the clean and noisy radiographs, respectively.
- \( \rho \) is the underlying clean density.
- \( \lambda_{\text{PISLR}} \) and \( \lambda_{\text{SSLR}} \) are hyperparameters controlling the weighting of the respective loss terms.
- The expectation is taken over the distribution of densities and radiographs.

---

## Key Contributions

- Physics-guided feature extraction and integration into inverse models.
- Robustness analysis under varying scatter scaling, and gamma and photon noise levels.

---

## Code Structure

- **Data Simulation:**  
  Scripts to generate clean and noisy radiographs with varying scatter, gamma, and photon noise levels.
  
- **Model Training:**  
  Scripts to train:
  - PISLR model  
  - SSLR model  
  - ULR model  

- **Evaluation:**  
  Scripts to evaluate the learned models on radiographs corrupted by different noise and scatter levels (both in-population and out-of-population).

---

## Running the Experiments

### Data Simulation
Generate simulated radiographs with desired scatter and noise settings:
```
python get_clean_and_noisy_radiographs.py
```



### Training Reconstruction Models:
Train PISLR, SSLR, and ULR models:
```
python train_pislr_model.py
python train_sslr_model.py
python train_ulr_model.py
```

### Evaluation:
Evaluate the reconstruction performance under different noise conditions:
python evaluate_noise_robustness.py --config configs/eval_noise.yaml
Evaluate generalization to unseen scatter conditions:
```
python test_all_models.py
```

Default parametes for testing using in-population noise parameters included in the script are :
1. `sigma_scatter = 10`
2. `scatter_scaling = 0.2`
3. `scatter_polynomial_order = 1`

### Datasets:
Synthetic datasets used in this work are not available and are protected under copyright agreement with Los Alamos National Laboratory.
You may optionally use your own experimental radiographs formatted as 2D projections and adjust the scatter models accordingly.

### Contact:
The code is provided for reproducible research purposes.
If you have any questions, suggestions, or encounter any issues running the code, please feel free to contact:

Siddhant Gautam (gautamsi@msu.edu)
Saiprasad Ravishankar (ravisha3@msu.edu)
