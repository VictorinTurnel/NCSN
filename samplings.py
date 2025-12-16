import torch

@torch.no_grad()
def annealed_sampling(model, n_samples, sigmas, T = 100, epsilon = 1e-5, device="cuda"):
    model.eval()
    x= torch.randn(n_samples, 1, 28, 28, device=device)

    samples = []
    for i, sigma in enumerate(sigmas):
        alpha = epsilon*(sigma/sigmas[-1])**2

        for t in range(T):
            z = torch.randn_like(x)
            score_model = model(x, sigma.repeat(n_samples)) / sigma
            x = x + 0.5*alpha*score_model

            last_step = (i==len(sigmas)-1) and (t==T-1)
            if not last_step:
                x=x+torch.sqrt(alpha)* z

        # We need this clamp to prevent from value explosion
        x=x.clamp(-1.2, 1.2)
        samples.append(x.clone())

    return samples

@torch.no_grad()
def annealed_langevin_toy(model, n_samples, sigmas, T = 100, epsilon = 2e-5, device="cuda"):
    model.eval()
    x = torch.rand(n_samples,2, device=device) * 16 - 8

    for i, sigma in enumerate(sigmas):
        alpha = epsilon*(sigma/sigmas[-1])**2

        for t in range(T):
            z = torch.randn_like(x)
            score_model = model(x, sigma.repeat(n_samples))
            x = x + 0.5*alpha*score_model

            last_step = (i==len(sigmas)-1) and (t==T-1)
            if not last_step:
                x=x+torch.sqrt(alpha)* z

    return x

@torch.no_grad()
def standard_langevin_toy(model, n_samples, sigmas, T = 100, epsilon = 1e-5, device="cuda"):
    model.eval()
    x = torch.rand(n_samples,2, device = device) * 16 - 8

    sigma = sigmas[-1]
    for t in range(T):
        z = torch.randn_like(x)
        score_model = model(x, sigma.repeat(n_samples))
        x = x + 0.5*epsilon*score_model

        last_step = (t==T-1)
        if not last_step:
            x=x+ epsilon**0.5 * z

    return x