import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmaCond(nn.Module):
    
    def __init__(self, dim = 32):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(1,dim),
            nn.SiLU(),
            nn.Linear(dim,dim)
        )

    def forward(self,sigma):
        return self.net(sigma[:,None])
    
class ConvBlock(nn.Module):

    def __init__(self, input_dim, output_dim, emd_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 3, padding = 1)
        self.norm = nn.GroupNorm(8, output_dim)
        self.emb = nn.Linear(emd_dim,output_dim)

    def forward(self, x, emb):
        h = self.conv(x)
        h = self.norm(h)
        h = h + self.emb(emb)[:,:,None,None]
        return F.silu(h)
    

class MNIST_NCSN(nn.Module):
    def __init__(self, embd_dim=32):
        super().__init__()
        self.emb = SigmaCond(embd_dim)
        self.enc1 = ConvBlock(1,32,embd_dim)
        self.enc2 = ConvBlock(32,64,embd_dim)
        self.enc3 = ConvBlock(64,128,embd_dim)
        self.pool = nn.AvgPool2d(2)
        self.dec3 = ConvBlock(128,64,embd_dim)
        self.dec2 = ConvBlock(64,32,embd_dim)
        self.out = nn.Conv2d(32,1,3,padding = 1)

    def forward(self, x, sigma):
        emb = self.emb(torch.log(sigma))
        e1 = self.enc1(x,emb)
        e2 = self.enc2(self.pool(e1),emb)
        e3 = self.enc3(self.pool(e2),emb)
        d2 = F.interpolate(e3, scale_factor=2)
        d2 = self.dec3(d2,emb) + e2
        d1 = F.interpolate(d2, scale_factor=2)
        d1 = self.dec2(d1, emb) + e1
        
        return self.out(d1)
    

class Toy_NCSN(nn.Module):

    def __init__(self, emb_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            # Concatenation of x (2D) and sigma (1D)
            nn.Linear(3, emb_dim),
            nn.Softplus(),
            nn.Linear(emb_dim, emb_dim),
            nn.Softplus(),
            nn.Linear(emb_dim,2)
        )

    def forward(self, x, sigma):
        sigma = sigma.view(-1,1).expand(x.shape[0],1)
        input = torch.cat([x, torch.log(sigma)],dim=1)
        return self.net(input)