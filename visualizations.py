import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def save_visualization(samples, epoch, save_dir):
    fig, axes = plt.subplots(1, len(samples), figsize=(20,4))
    fig.suptitle(f"Epoch {epoch}", fontsize=16)

    os.makedirs(save_dir, exist_ok=True)
    for i,(ax,x) in enumerate(zip(axes, samples)):
        grid = make_grid(x.clamp(-1,1), nrow=4, normalize=True, value_range=(-1,1))

        ax.imshow(grid.permute(1,2,0).cpu())
        ax.axis("off")
        ax.set_title(f"Step {i}")

    save_path = os.path.join(save_dir, f"sample_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()

def visualization_grid_circle(model, samples, sigmas, data_sigma, radius, get_ground_truth, device):

    range_lim = 10
    x=np.linspace(-range_lim, range_lim, 40)
    y=np.linspace(-range_lim, range_lim, 40)
    X, Y = np.meshgrid(x,y)
    grid = torch.tensor(np.stack([X,Y],axis=-1), dtype=torch.float32).view(-1,2).to(device)


    last_sigma = sigmas[-1].item()
    sigma_tensor = torch.ones(grid.shape[0], device=device)*last_sigma

    model.eval()
    with torch.no_grad():
        score = model(grid, sigma_tensor.unsqueeze(1)).cpu()

    score_gt = get_ground_truth(grid.cpu(), data_sigma, radius, last_sigma)
    error = score - score_gt
    error_norm = torch.norm(error, dim=1).reshape(X.shape)
    
    norm_pred = torch.norm(score, dim=1, keepdim=True)
    score_pred_norm = (score/(norm_pred+1e-5)).numpy()

    norm_gt = torch.norm(score_gt, dim=1, keepdim=True)
    score_gt_norm = (score_gt/(norm_gt+1e-5)).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20,6))

    # Predicted Score
    axes[0].set_title(f"1. Predicted Score (Sigma={last_sigma:.2f})")
    axes[0].quiver(grid[:,0].cpu(), grid[:,1].cpu(), score_pred_norm[:,0], score_pred_norm[:,1], color='r', scale=30)
    axes[0].add_patch(plt.Circle((0,0), radius, color='b', fill=False, linestyle='--'))
    axes[0].set_xlim([-range_lim, range_lim])
    axes[0].set_ylim([-range_lim, range_lim])

    # Ground Truth
    axes[1].set_title(f"1. Predicted Score (Sigma={last_sigma:.2f})")
    axes[1].quiver(grid[:,0].cpu(), grid[:,1].cpu(), score_gt_norm[:,0], score_gt_norm[:,1], color='r', scale=30)
    axes[1].add_patch(plt.Circle((0,0), radius, color='b', fill=False, linestyle='--'))
    axes[1].set_xlim([-range_lim, range_lim])
    axes[1].set_ylim([-range_lim, range_lim])

    # Error map
    axes[2].set_title("2. Error Norm Map (log scale)")
    im = axes[2].imshow(np.log1p(error_norm.numpy()), extent=(-range_lim, range_lim, -range_lim, range_lim), origin='lower', cmap='magma', alpha=0.9)
    plt.colorbar(im, ax=axes[2], label="Log(Error)")

    axes[2].scatter(samples[:,0].cpu(), samples[:,1].cpu(), color='cyan', s=1, label='Training Samples')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def visualization_grid_gaussian(samples_gt, samples_standard, sample_annealed):

    range_lim = 10
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    samples_gt = samples_gt.cpu()
    samples_standard = samples_standard.cpu()
    sample_annealed = sample_annealed.cpu()

    # Ground thruth
    axes[0].scatter(samples_gt[:,0], samples_gt[:,1],s=2,alpha=0.5,color="green")
    axes[0].set_title("Ground truth")
    axes[0].set_xlim(-range_lim,range_lim)
    axes[0].set_ylim(-range_lim,range_lim)

    # Standard Langevin
    axes[1].scatter(samples_standard[:,0], samples_standard[:,1],s=2,alpha=0.5,color="blue")
    axes[1].set_title("Standard Langevin Dynamics")
    axes[1].set_xlim(-range_lim,range_lim)
    axes[1].set_ylim(-range_lim,range_lim)

    # Annealed Langevin
    axes[2].scatter(sample_annealed[:,0], sample_annealed[:,1],s=2,alpha=0.5,color="purple")
    axes[2].set_title("Annealed Langevin Dynamics")
    axes[2].set_xlim(-range_lim,range_lim)
    axes[2].set_ylim(-range_lim,range_lim)

    plt.show()




    

