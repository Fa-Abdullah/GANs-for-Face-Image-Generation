# GANs for Face Generation

A deep learning project that implements a **Generative Adversarial Network (GAN)** using **PyTorch** to generate realistic human face images. The model is trained on the **CelebA** dataset and learns to synthesize high-quality 64×64 RGB face images from random latent vectors.

---

## Overview

This project demonstrates the implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** for realistic face generation.

The model consists of two competing neural networks:

* **Generator:** Generates realistic face images from random noise.
* **Discriminator:** Distinguishes between real and generated images.

Through adversarial training, the Generator progressively improves until it produces realistic faces capable of fooling the Discriminator.

---

## Dataset

* **Dataset:** 50K Celebrity Faces Image Dataset
* **Source:** Kaggle
* **Total Images:** 50,000
* **Image Resolution:** 64 × 64 RGB
* **Preprocessing:**

  * Resize to 64×64
  * Center Crop
  * Normalize images to the range [-1, 1]

---

## Model Architecture

### Generator

* Input: 100-dimensional latent vector
* ConvTranspose2D layers
* Batch Normalization
* ReLU activations
* Tanh output activation

### Discriminator

* Input: 64×64 RGB image
* Convolutional layers
* Batch Normalization
* LeakyReLU activations
* Sigmoid output for binary classification

---

## Training Configuration

| Parameter          |                          Value |
| ------------------ | -----------------------------: |
| Framework          |                        PyTorch |
| Epochs             |                            150 |
| Batch Size         |                            128 |
| Learning Rate      |                         0.0002 |
| Optimizer          |                           Adam |
| Loss Function      | Binary Cross Entropy (BCELoss) |
| Latent Vector Size |                            100 |

---

## Features

* Deep Convolutional GAN (DCGAN) implementation
* Automatic CelebA dataset download
* GPU acceleration with CUDA support
* Generator and Discriminator checkpoint saving
* Generated image visualization after each epoch
* Modular and easy-to-understand PyTorch implementation

---

## Results

The Generator gradually learns to transform random noise into realistic human face images.

The notebook includes:

* Generator and Discriminator loss curves
* Generated face samples during training
* Saved model checkpoints
* Epoch-wise visualization of generated images

