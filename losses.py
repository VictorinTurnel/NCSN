import torch

# Denoising Score Matching (DSM) loss function
# Used to train score-based generative models by estimating the score function of the data distribution
def dsm_loss(model, x, sigmas):

    B= x.size(0)

    # Randomly select a noise level sigma for each sample in the batch
    sigma = sigmas[torch.randint(0,len(sigmas), (B,), device=x.device)]
    sigma = sigma.view(B,1,1,1)

    # Add Gaussian noise to the input data
    noise = torch.randn_like(x)
    x_noisy = x+sigma*noise

    # Compute the model's score and the target score
    # The target score is derived from the added noise
    score = model(x_noisy,sigma.squeeze())/sigma
    target= -noise/sigma

    # Compute the DSM loss
    return (((score-target)**2)*(sigma**2)).mean()

# DSM loss function for 2D synthetic data
# Similar to the main DSM loss but adapted for 2D inputs
def dsm_loss_toy(model, x, sigmas):

    noise = torch.randn_like(x)
    x_noisy = x + sigmas.view(-1,1)*noise

    score = model(x_noisy, sigmas)
    target = -noise/sigmas.view(-1,1)

    loss = 0.5 * ((score-target)**2).sum(dim=1) * (sigmas**2).squeeze()
    return loss.mean()



