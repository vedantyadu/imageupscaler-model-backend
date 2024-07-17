from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU()
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        residual1 = self.residual1(x)
        block3 = self.block3(block2 + residual1)
        pixelshuffle = nn.PixelShuffle(2)
        residual2 = self.residual2(x)
        block4 = self.block4(pixelshuffle(block3) + pixelshuffle(residual2))
        return block4
