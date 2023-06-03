from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, input_channel=74, im_chan= 1, hidden_channel=64):
        super(Generator, self).__init__()
        self.input_channel = input_channel

        def block(input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
            layers = []
            layers.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride))
            if not final_layer:
                layers.append(nn.BatchNorm2d(output_channels, momentum=0.8)) # only add batch norm to hidden layers
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.Tanh())
            return layers

        self.generator = nn.Sequential(
            *block(input_channel, hidden_channel * 4),
            *block(hidden_channel * 4, hidden_channel * 2, kernel_size=4, stride=1),
            *block(hidden_channel * 2, hidden_channel),
            *block(hidden_channel, im_chan, kernel_size=4, final_layer=True)
        )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.input_channel, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.generator(x)

if __name__ == "__main__":
    z_dim = 64
    gen = Generator(input_channel=z_dim + 10)
    noise = torch.randn(128, z_dim + 10)
    print(gen(noise).shape) # [128, 1, 28, 28]
