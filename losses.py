import torch

def dsm_loss(model, x, sigmas):

    B= x.size(0)

    sigma = sigmas[torch.randint(0,len(sigmas), (B,), device=x.device)]
    sigma = sigma.view(B,1,1,1)

    noise = torch.randn_like(x)
    x_noisy = x+sigma*noise

    score = model(x_noisy,sigma.squeeze())/sigma
    target= -noise/sigma

    return (((score-target)**2)*(sigma**2)).mean()
