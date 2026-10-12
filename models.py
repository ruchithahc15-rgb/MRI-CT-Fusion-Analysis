"""Generator and PatchGAN discriminator models."""
import torch
from torch import nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="mish"):
        super().__init__()
        act1 = nn.PReLU() if activation == "prelu" else Mish()
        act2 = nn.PReLU() if activation == "prelu" else Mish()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            act1,
            nn.Conv2d(out_channels, out_channels, 3, 2, 1),
            nn.InstanceNorm2d(out_channels),
            act2,
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            Mish(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            Swish(),
        )

    def forward(self, x):
        return x + self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(out_channels),
            Mish(),
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mri_encoder1 = EncoderBlock(1, 64, activation="prelu")
        self.mri_encoder2 = EncoderBlock(64, 128)
        self.mri_encoder3 = EncoderBlock(128, 256)
        self.ct_encoder1 = EncoderBlock(1, 64, activation="prelu")
        self.ct_encoder2 = EncoderBlock(64, 128)
        self.ct_encoder3 = EncoderBlock(128, 256)
        self.fusion = nn.Conv2d(512, 512, 3, 1, 1)
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.res3 = ResidualBlock(512)
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.final = nn.Sequential(nn.Conv2d(64, 1, 3, 1, 1), nn.Tanh())

    def forward(self, mri, ct):
        m = self.mri_encoder3(self.mri_encoder2(self.mri_encoder1(mri)))
        c = self.ct_encoder3(self.ct_encoder2(self.ct_encoder1(ct)))
        x = self.fusion(torch.cat([m, c], dim=1))
        x = self.res3(self.res2(self.res1(x)))
        x = self.dec3(self.dec2(self.dec1(x)))
        return self.final(x)

class Discriminator(nn.Module):
    """PatchGAN discriminator returning a grid of logits."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)
