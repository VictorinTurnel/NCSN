import torch
import numpy as np
from tqdm import tqdm
import argparse


from models import Toy_NCSN
from losses import dsm_loss_toy
from samplings import annealed_langevin_toy, standard_langevin_toy
from visualizations import visualization_grid_circle, visualization_grid_gaussian
from utils import sample_generator_gaussian, sample_generator_circle, get_ground_circle


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters for training on 2D synthetic data
# These parameters were chosen thannks to empirical testing to ensure effective learning
MEANS = [torch.tensor([-5.0,-5.0]), torch.tensor([5.0, 5.0])]
WEIGHTS = [1/5, 4/5]
NUM_LEVEL = 10
N_SAMPLES = 128
ITERATION = 15000
DATA_SIGMA = 0.1
RADIUS = 5.0
LR = 1e-3

# The training function for the toy 2D data
# It supports training with either a Gaussian mixture model or a circle distribution
# depending on the chosen data generator.
def train(model, generator, optimizer, sigmas_list):
    model.train()
    
    for i in range(ITERATION):

        # Generate a batch of samples
        # Depending on the generator, we create samples from either a Gaussian mixture or a circle distribution.
        if generator==sample_generator_circle:
            x = generator(N_SAMPLES, DATA_SIGMA, RADIUS, device=DEVICE)
        else:
            x = generator(N_SAMPLES, WEIGHTS, MEANS, device=DEVICE)

        sigma_idx = torch.randint(0, NUM_LEVEL, (N_SAMPLES,), device= DEVICE)
        sigmas = sigmas_list[sigma_idx]

        # Compute the DSM loss for the current batch
        loss = dsm_loss_toy(model,x,sigmas)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%1000 == 0:
            print(f"Iteration {i+1}/{ITERATION} - Loss {loss.item():.4f}")

    # After training, we visualize the learned score function
    # and compare it to the ground truth for evaluation.
    if generator==sample_generator_circle:
        samples = generator(N_SAMPLES, DATA_SIGMA, RADIUS, device=DEVICE)
        visualization_grid_circle(model, samples, sigmas_list, DATA_SIGMA, RADIUS, get_ground_circle, DEVICE)
    else:
        samples_gt = generator(N_SAMPLES*10, WEIGHTS, MEANS, DEVICE)
        samples_standard = standard_langevin_toy(model, N_SAMPLES*10, sigmas_list, T=1000, epsilon=0.01, device=DEVICE)
        samples_annealed = annealed_langevin_toy(model, N_SAMPLES*10, sigmas_list, T=1000, epsilon=0.01, device=DEVICE)
        visualization_grid_gaussian(samples_gt, samples_standard, samples_annealed)

def main():
    parser = argparse.ArgumentParser(description="Train NCSN model with different data generators")
    parser.add_argument('--generator', type=str, choices=['gaussian', 'circle'], default='gaussian',
                        help='Choose the data generator: gaussian or circle')
    args = parser.parse_args()

    model = Toy_NCSN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Select the appropriate data generator based on user input
    # and set corresponding sigma parameters.
    if args.generator=='gaussian':
        generator =sample_generator_gaussian
        SIGMA_MAX = 10.0
        SIGMA_MIN = 1 
    elif args.generator=='circle':
        generator =sample_generator_circle
        SIGMA_MAX = 5.0
        SIGMA_MIN = 0.01

    sigmas_np = np.exp(np.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), NUM_LEVEL))
    SIGMAS = torch.tensor(sigmas_np, dtype=torch.float32, device=DEVICE)

    train(model, generator, optimizer, SIGMAS)
    


if __name__ == "__main__":
    main()


