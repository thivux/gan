from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, z_dim=64, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        def block(input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
            layers = []
            if not final_layer:
                layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride))
                layers.append(nn.BatchNorm2d(output_channels)) # only add batch norm to hidden layers
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride))
                layers.append(nn.Tanh())
            return layers

        self.generator = nn.Sequential(
            *block(z_dim, hidden_dim * 4),
            *block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            *block(hidden_dim * 2, hidden_dim),
            *block(hidden_dim, im_chan, kernel_size=4, final_layer=True)
        )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.generator(x)

if __name__ == "__main__":
    z_dim = 64
    gen = Generator(z_dim=z_dim)
    noise = torch.randn(128, z_dim)
    print(gen(noise).shape) # [128, 1, 28, 28]
