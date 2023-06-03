import torch
from torch import nn 

class Discriminator(nn.Module):
    def __init__(self, input_channel=1, hidden_channel=64):
        super(Discriminator, self).__init__()
        def block(input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
            layers = []
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size, stride))
            if not final_layer:
                layers.append(nn.BatchNorm2d(output_channels, momentum=0.8))
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            return layers

        self.discriminator = nn.Sequential(
            *block(input_channel, hidden_channel),
            *block(hidden_channel, hidden_channel * 2),
            *block(hidden_channel * 2, 1, final_layer=True)
        )

    def forward(self, image):
        return self.discriminator(image).view(len(image), -1)

if __name__ == "__main__":
    disc = Discriminator(hidden_channel=64)
    image = torch.rand(128, 1, 28 , 28)
    res = disc(image)
    print(res.shape)
    print(res)
