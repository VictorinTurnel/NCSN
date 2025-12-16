import torch
import numpy as np

def sample_generator_gaussian(n_samples, weights, means, device="cuda"):

    indices = np.random.choice(len(weights),size=n_samples,p=weights)
    samples = []
    for idx in indices:
        m = means[idx]

        z = torch.randn(2)
        x = m+z
        samples.append(x)

    return torch.stack(samples).to(device)

def sample_generator_circle(n_samples, data_sigma, radius, device="cuda"):
    theta = torch.randn(n_samples)*2*np.pi
    r = torch.randn(n_samples) * data_sigma + radius
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x,y],dim=1).to(device)


def get_ground_circle(x, data_sigma, radius, sigma_noise):

    norm = torch.norm(x, dim=1, keepdim=True)
    safe_norm = norm + 1e-10
    direction = x / safe_norm
    var_totale = data_sigma**2 + sigma_noise**2
    magnitude = - (norm - radius) / var_totale
    
    return direction * magnitude