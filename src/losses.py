import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg19(
            weights=VGG19_Weights.DEFAULT
        )

        # Use early/mid-level VGG features
        self.features = nn.Sequential(
            *list(vgg.features.children())[:36]
        )

        # Freeze VGG
        for param in self.features.parameters():
            param.requires_grad = False

        self.features.eval()

        self.criterion = nn.L1Loss()

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def forward(self, fake, real):
        fake = (fake - self.mean) / self.std
        real = (real - self.mean) / self.std

        fake_features = self.features(fake)
        real_features = self.features(real)

        return self.criterion(
            fake_features,
            real_features.detach(),
        )


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        perceptual_weight=0.01,
        adversarial_weight=0.005,
    ):
        super().__init__()

        self.content_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()

        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight

    def forward(
        self,
        fake_hr,
        real_hr,
        discriminator_fake,
    ):
        # Pixel-level reconstruction
        content = self.content_loss(
            fake_hr,
            real_hr,
        )

        # Feature-level reconstruction
        perceptual = self.perceptual_loss(
            fake_hr,
            real_hr,
        )

        # Generator wants discriminator
        # to classify generated images as real
        adversarial = -discriminator_fake.mean()

        total = (
            content
            + self.perceptual_weight * perceptual
            + self.adversarial_weight * adversarial
        )

        return {
            "total": total,
            "content": content,
            "perceptual": perceptual,
            "adversarial": adversarial,
        }


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        real_output,
        fake_output,
    ):
        real_target = torch.ones_like(
            real_output
        )

        fake_target = torch.zeros_like(
            fake_output
        )

        real_loss = self.criterion(
            real_output,
            real_target,
        )

        fake_loss = self.criterion(
            fake_output,
            fake_target,
        )

        return 0.5 * (
            real_loss + fake_loss
        )


if __name__ == "__main__":
    from model import (
        ESRGANGenerator,
        Discriminator,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    generator = ESRGANGenerator().to(device)
    discriminator = Discriminator().to(device)

    loss_g = GeneratorLoss().to(device)
    loss_d = DiscriminatorLoss().to(device)

    lr = torch.randn(
        2, 3, 64, 64,
        device=device,
    )

    real_hr = torch.rand(
        2, 3, 256, 256,
        device=device,
    )

    fake_hr = generator(lr)

    real_output = discriminator(
        real_hr
    )

    fake_output = discriminator(
        fake_hr
    )

    g_losses = loss_g(
        fake_hr,
        real_hr,
        fake_output,
    )

    d_loss = loss_d(
        real_output,
        fake_output,
    )

    print("Device:", device)

    print("\nGenerator losses:")

    for name, value in g_losses.items():
        print(
            f"{name}: {value.item():.6f}"
        )

    print(
        f"\nDiscriminator loss: "
        f"{d_loss.item():.6f}"
    )