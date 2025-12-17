import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from  tqdm import tqdm
import numpy as np
import copy
import os

from losses import dsm_loss
from samplings import annealed_sampling
from models import MNIST_NCSN
from visualizations import save_visualization
from torchmetrics.image.fid import FrechetInceptionDistance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We also applied the techniques 1 & 2 from the paper "Improved Techniques for Training Score-based Generative Models"
# to stabilize training and improve sample quality
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 50
SIGMA_MAX = 50.0
SIGMA_MIN = 0.01
NUM_LEVEL = 200
EPS = 1e-5
T = 10
N_SAMPLE = 16
SAVE_DIR = "results/MNIST_EVA/"

sigmas_np = np.exp(np.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), NUM_LEVEL))
SIGMAS = torch.tensor(sigmas_np, dtype=torch.float32, device=DEVICE)

# Exponential Moving Average (EMA) class for model parameters : Technique 5
# Enabled to smooth the training process and improve sample quality
# Parameters are updated smoothly over time
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.to(DEVICE)
        self.shadow.eval()

    def update(self, model):
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                if param.requires_grad:
                    # Update the EMA parameters
                    # We applay the exponential moving average formula
                    s_param.data = self.decay * s_param.data + (1.0 - self.decay) * param.data

    def get_model(self):
        return self.shadow

# This function is necessary to adapt the MNIST images for FID computation
# by converting grayscale images to 3-channel format and scaling pixel values appropriately
def preprocess_fid(images):
    img = (images + 1) / 2
    img = img.clamp(0, 1)
    img = img.repeat(1, 3, 1, 1)
    img = (img * 255).to(torch.uint8)
    return img

def train(model, loader, optimizer):
    # Initialize the EMA model
    ema = EMA(model, decay=0.999)

    # Initialize FID metric
    # Used to evaluate the quality of generated samples against real images
    fid = FrechetInceptionDistance(feature=64).to(DEVICE)
    fixed_real_batch, _= next(iter(loader))
    fixed_real_batch=fixed_real_batch.to(DEVICE)

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        for x, _ in pbar:
            # Training step with DSM loss
            # The model learns to estimate the score function of the data distribution
            x = x.to(DEVICE)
            loss = dsm_loss(model, x, SIGMAS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            pbar.set_description(f'Epoch {epoch+1}/{EPOCHS} - Loss {loss.item():.4f}')

        if (epoch+1)%5 == 0:
            
            # FID computation
            # Evaluates the quality of generated samples every 5 epochs
            fid.update(preprocess_fid(fixed_real_batch), real=True)
            samples_fid , _ = annealed_sampling(ema.get_model(), N_SAMPLE, SIGMAS, T, EPS, DEVICE)
            final_fake_images = samples_fid[-1]
            fid.update(preprocess_fid(final_fake_images), real=False)
            fid_score = fid.compute().item()
            fid.reset()
            print(f"-> FID Score: {fid_score:.4f}")

            samples, _ = annealed_sampling(model, N_SAMPLE, SIGMAS, T, EPS, DEVICE)
            save_visualization(samples, epoch, SAVE_DIR)

            save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Save for epoch : {epoch+1}")
            print()
            model.train()


def main():
    # Data preparation and model initialization
    # Standard preprocessing for MNIST dataset to scale images to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x*2-1
    ])
    
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    model = MNIST_NCSN().to(DEVICE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training starts")
    train(model, loader, optimizer)
    print("Training finished")

if __name__ == "__main__":
    main()