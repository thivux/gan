import torch
from torch import nn 

class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        def block(input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
            layers = []
            if not final_layer:
                layers.append(nn.Conv2d(input_channels, output_channels, kernel_size, stride))
                layers.append(nn.BatchNorm2d(output_channels))
                layers.append(nn.LeakyReLU(negative_slope=0.2))
            else:
                layers.append(nn.Conv2d(input_channels, output_channels, kernel_size, stride))
                # layers.append(nn.Sigmoid()) # sigmoid is not needed for BCEWithLogitsLoss
            return layers

        self.discriminator = nn.Sequential(
            *block(im_chan, hidden_dim),
            *block(hidden_dim, hidden_dim * 2),
            *block(hidden_dim * 2, 1, final_layer=True)
        )

    def forward(self, image):
        return self.discriminator(image).view(len(image), -1)

if __name__ == "__main__":
    disc = Discriminator(hidden_dim=64)
    image = torch.rand(128, 1, 28 , 28)
    res = disc(image)
    print(res.shape)
    print(res)
