# PyTorch Implementation of Noise Conditional Score Networks (NCSN)

## Description
This repository contains a PyTorch implementation of the concepts introduced in the paper **"Generative Modeling by Estimating Gradients of the Data Distribution"** (Song & Ermon, NeurIPS 2019).

The goal of this project is to explore a new principle for generative modeling based on estimating and sampling from the Stein score ($ \nabla_x \log p_{data}(x) $) of the data distribution.

This implementation focuses on three key aspects:
1.  **Score Estimation:** Learning the vector field of gradients using a simple MLP.
2.  **Annealed Langevin Dynamics:** Solving the mixing problem of multimodal distributions using multiple noise levels.
3.  **Image Generation:** Generating MNIST digits using a **simplified U-Net** architecture (lighter than the RefineNet used in the original paper) to allow for faster training on standard hardware.

## Experiments

### Visualization of Score Estimation

The manifold hypothesis states that real-world data tends to reside on low-dimensional manifolds. The paper argues that score estimation is only consistent where data resides and becomes inaccurate in low-density regions.

This experiment trains a simple MLP on a 2D "circle" distribution to visualize the estimated score field. It demonstrates that the estimated gradients are accurate near the data points (the circle) but unreliable far away from the manifold.
```
python ./training_toy.py --generator circle
```

### Annealed Langevin Dynamics (Mixture of Gaussians)

Standard Langevin dynamics often fail to mix properly between separated modes because regions of low data density act as barriers.

To solve this, the paper proposes Annealed Langevin Dynamics, which uses a sequence of noise levels {σi​} to gradually guide the sampling process from a high-noise distribution to the data distribution.

This experiment reproduces the Mixture of Gaussians toy example (similar to Fig. 3 in the paper ). It compares sampling with standard Langevin dynamics versus the annealed version to show how the latter correctly recovers the relative weights of the modes.
```
python ./training_toy.py --generator gaussian
```

### MNIST Generation

This script trains a Noise Conditional Score Network (NCSN) on the MNIST dataset.

While the original paper utilizes a complex RefineNet architecture with dilated convolutions and specific instance normalization, this repository implements a simplified U-Net. This modification significantly reduces training time while still demonstrating the capability of NCSNs to generate recognizable handwritten digits.

To train the model and generate samples:
```
python ./training_MNIST.py
```
