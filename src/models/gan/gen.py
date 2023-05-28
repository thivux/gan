import torch
from torch import nn
from pytorch_lightning import LightningModule


def get_gen_block(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True)) 
    
class Generator(LightningModule):
    def __init__(self, z_dim=64, img_dim=28*28, hidden_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            get_gen_block(z_dim, hidden_dim),  # weight.shape = (64, 128)
            get_gen_block(hidden_dim, hidden_dim*2), # weight.shape = (128, 256) 
            get_gen_block(hidden_dim*2, hidden_dim*4),
            get_gen_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, img_dim),
        )

    def forward(self, noise): # noise.shape = (batch_size, z_dim)
        return self.gen(noise)

if __name__ == '__main__':
    z_dim = 64
    gen = Generator(z_dim)
    noise = torch.rand(2, z_dim)
    res = gen(noise)
    print(res.shape)
    print(res.min(), res.max())