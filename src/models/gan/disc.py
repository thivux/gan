import torch
from torch import nn
from pytorch_lightning import LightningModule

def get_disc_block(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU(negative_slope=0.2))

class Discriminator(LightningModule):
    def __init__(self, in_dim=28*28, hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            get_disc_block(in_dim, hidden_dim*4),
            get_disc_block(hidden_dim*4, hidden_dim*2),
            get_disc_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image):
        return self.disc(image)

if __name__ == '__main__':
    disc = Discriminator()
    input = torch.rand(2, 28*28)
    res = disc(input)
    print(res)