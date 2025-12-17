import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from samplings import annealed_sampling
from models import MNIST_NCSN

# Selected values for T and epsilon to perform grid search
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_VALUES = [10, 30, 100]
EPS_VALUES = [1e-5, 2e-5, 1e-4, 1e-3]

def grid_search_sampling(model_path, n_samples, sigmas, device=DEVICE):
    model = MNIST_NCSN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    count = 0
    results = []
    total_combinations = len(T_VALUES) * len(EPS_VALUES)
    os.makedirs("results/sampling_grid/", exist_ok=True)
    print("Start grid search for sampling hyperparameters")

    # Here we iterate over all combinations of T and epsilon
    # For each combination, we generate samples and store the results for visualization
    for T in T_VALUES:
        for epsilon in EPS_VALUES:
            count += 1
            print(f"Combination {count}/{total_combinations} : T={T}, epsilon={epsilon}")

            samples = annealed_sampling(model, n_samples, sigmas, T=T,epsilon=epsilon,device=device)

            results.append({
                "T": T,
                "epsilon": epsilon,
                "samples": samples
            })

    # Visualization of all sampling results in a grid format
    # Each row corresponds to a different combination of T and epsilon
    # This allows for easy comparison of sample quality across different hyperparameter settings
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12,2*n_rows))

    for ax, res in zip(axes, results):
        t_val = res["T"]
        eps_val = res["epsilon"]
        _, imgs = res["samples"]


        grid_img = make_grid(imgs.view(-1,1,28,28), nrow=8, normalize=True, value_range=(-1,1))

        ax.imshow(grid_img.permute(1,2,0).cpu().numpy())
        ax.set_title(f"T={t_val} - eps={eps_val}", fontsize=12, fontweight='bold')
        ax.axis("off")

    save_path = os.path.join(SAVE_DIR, "grid_search_sampling.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    SIGMA_MAX = 2.0
    SIGMA_MIN = 0.01
    NUM_LEVEL = 10
    N_SAMPLES = 6

    sigmas_np = np.exp(np.linspace(np.log(SIGMA_MAX), np.log(SIGMA_MIN), NUM_LEVEL))
    SIGMAS = torch.tensor(sigmas_np, dtype=torch.float32, device=DEVICE)
    MODEL_PATH = "results/MNIST/model_epoch_50.pth"
    SAVE_DIR = "results/sampling_grid/"


    grid_search_sampling(MODEL_PATH, N_SAMPLES, SIGMAS, DEVICE)





