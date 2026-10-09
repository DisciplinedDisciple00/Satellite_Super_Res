import torch
import torch.nn as nn


class DenseResidualBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            growth_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            channels + growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            channels + 2 * growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            channels + 3 * growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv5 = nn.Conv2d(
            channels + 4 * growth_channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(
            self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        )

        x5 = self.conv5(
            torch.cat([x, x1, x2, x3, x4], dim=1)
        )

        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()

        self.block1 = DenseResidualBlock(
            channels,
            growth_channels,
        )
        self.block2 = DenseResidualBlock(
            channels,
            growth_channels,
        )
        self.block3 = DenseResidualBlock(
            channels,
            growth_channels,
        )

    def forward(self, x):
        residual = self.block1(x)
        residual = self.block2(residual)
        residual = self.block3(residual)

        return x + 0.2 * residual


class ESRGANGenerator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=64,
        num_rrdb=8,
        scale=4,
    ):
        super().__init__()

        self.scale = scale

        # Initial feature extraction
        self.conv_first = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        # RRDB trunk
        self.rrdb_blocks = nn.Sequential(
            *[
                RRDB(channels)
                for _ in range(num_rrdb)
            ]
        )

        self.trunk_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        # 4x upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.PixelShuffle(2),

            nn.Conv2d(
                channels // 4,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.PixelShuffle(2),
        )

        # Final reconstruction
        self.hr_conv = nn.Conv2d(
            channels // 4,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.conv_last = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        # Feature extraction
        feat = self.conv_first(x)

        # RRDB feature processing
        trunk = self.rrdb_blocks(feat)
        trunk = self.trunk_conv(trunk)

        # Global residual connection
        feat = feat + trunk

        # Upsampling
        feat = self.upsample(feat)

        # Reconstruction
        feat = self.hr_conv(feat)
        out = self.conv_last(feat)

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def block(in_ch, out_ch, stride=1, use_bn=True):
            layers = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                )
            ]

            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))

            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )

            return layers

        self.model = nn.Sequential(
            # 256 → 256
            *block(
                in_channels,
                base_channels,
                stride=1,
                use_bn=False,
            ),

            # 256 → 128
            *block(
                base_channels,
                base_channels,
                stride=2,
            ),

            # 128 → 128
            *block(
                base_channels,
                base_channels * 2,
                stride=1,
            ),

            # 128 → 64
            *block(
                base_channels * 2,
                base_channels * 2,
                stride=2,
            ),

            # 64 → 64
            *block(
                base_channels * 2,
                base_channels * 4,
                stride=1,
            ),

            # 64 → 32
            *block(
                base_channels * 4,
                base_channels * 4,
                stride=2,
            ),

            # 32 → 16
            *block(
                base_channels * 4,
                base_channels * 8,
                stride=2,
            ),

            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------
    # Generator
    # -------------------------
    generator = ESRGANGenerator(
        in_channels=3,
        out_channels=3,
        channels=64,
        num_rrdb=8,
        scale=4,
    ).to(device)

    # -------------------------
    # Discriminator
    # -------------------------
    discriminator = Discriminator(
        in_channels=3,
        base_channels=64,
    ).to(device)

    # Fake LR input
    lr = torch.randn(
        4,
        3,
        64,
        64,
        device=device,
    )

    # Fake HR image
    hr = torch.randn(
        4,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():
        fake_hr = generator(lr)

        real_output = discriminator(hr)
        fake_output = discriminator(fake_hr)

    print("Device:", device)

    print("\nGenerator:")
    print("Input:", lr.shape)
    print("Output:", fake_hr.shape)

    print("\nDiscriminator:")
    print("Real input:", hr.shape)
    print("Real output:", real_output.shape)
    print("Fake input:", fake_hr.shape)
    print("Fake output:", fake_output.shape)

    generator_params = sum(
        p.numel()
        for p in generator.parameters()
    )

    discriminator_params = sum(
        p.numel()
        for p in discriminator.parameters()
    )

    print(
        f"\nGenerator parameters: "
        f"{generator_params:,}"
    )

    print(
        f"Discriminator parameters: "
        f"{discriminator_params:,}"
    )