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

def dsm_loss_toy(model, x, sigmas):

    noise = torch.randn_like(x)
    x_noisy = x + sigmas.view(-1,1)*noise

    score = model(x_noisy, sigmas)
    target = -noise/sigmas.view(-1,1)

    loss = 0.5 * ((score-target)**2).sum(dim=1) * (sigmas**2).squeeze()
    return loss.mean()



