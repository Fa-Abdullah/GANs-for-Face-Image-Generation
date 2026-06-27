# GANs for Image Generation

A PyTorch implementation of a **Generative Adversarial Network (GAN)** for generating realistic celebrity face images. The model is trained on the **50k Celebrity Faces Image Dataset** from Kaggle using the Deep Convolutional GAN (DCGAN) architecture.

---

## Features

* Deep Convolutional GAN (DCGAN) implementation
* Trained on the 50k Celebrity Faces dataset
* Automatic image preprocessing and normalization
* GPU acceleration (CUDA support)
* Model checkpoint saving
* Best model tracking
* Generated image visualization after each training epoch

---

## Dataset

**Dataset:** 50k Celebrity Faces Image Dataset (Kaggle)

### Preprocessing

* Resize images to **64 × 64**
* Center crop
* Convert to tensors
* Normalize to **[-1, 1]**

---

## Model Architecture

### Generator

Transforms a 100-dimensional latent vector into a 64×64 RGB image using:

* ConvTranspose2D layers
* Batch Normalization
* ReLU activation
* Tanh output activation

### Discriminator

Classifies images as **Real** or **Fake** using:

* Convolutional layers
* Batch Normalization
* LeakyReLU activation
* Sigmoid output layer

---

## Training Configuration

| Parameter        | Value                          |
| ---------------- | ------------------------------ |
| Epochs           | 150                            |
| Batch Size       | 128                            |
| Latent Dimension | 100                            |
| Learning Rate    | 0.0002                         |
| Optimizer        | Adam                           |
| β1               | 0.5                            |
| β2               | 0.999                          |
| Loss Function    | Binary Cross Entropy (BCELoss) |

---

## Training Pipeline

1. Load and preprocess the dataset.
2. Train the **Discriminator** using real and generated images.
3. Train the **Generator** to fool the Discriminator.
4. Save checkpoints and generated samples after each epoch.

---

## Output Files

* `generated_epoch_X.png` – Generated samples after each epoch
* `gan_checkpoint.pth` – Training checkpoint
* `best_generator.pth` – Best Generator model

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* KaggleHub
* Matplotlib
* tqdm

---

## Results

After training, the Generator progressively learns to synthesize realistic celebrity face images from random noise. Image quality improves over successive epochs as the adversarial training converges.

---
