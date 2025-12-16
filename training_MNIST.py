import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from  tqdm import tqdm
import numpy as np
import os

from losses import dsm_loss
from samplings import annealed_sampling
from models import MNIST_NCSN
from visualizations import save_visualization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 50
SIGMA_MAX = 2.0
SIGMA_MIN = 0.01
NUM_LEVEL = 10
EPS = 1e-5
T = 100
N_SAMPLE = 16
SAVE_DIR = "results"

sigmas_np = np.exp(np.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), NUM_LEVEL))
SIGMAS = torch.tensor(sigmas_np, dtype=torch.float32, device=DEVICE)

def train(model, loader, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        for x, _ in pbar:
            x = x.to(DEVICE)
            loss = dsm_loss(model, x, SIGMAS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Epoch {epoch+1}/{EPOCHS} - Loss {loss.item():.4f}')

        if (epoch+1)%5 == 0:
            print(f"Save for epoch : {epoch+1}")

            samples = annealed_sampling(model, N_SAMPLE, SIGMAS, T, EPS, DEVICE)
            save_visualization(samples, epoch, SAVE_DIR)

            save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

            model.train()


def main():
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