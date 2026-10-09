from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import WorldStratDataset
from model import ESRGANGenerator


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "ablation_perceptual_best.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage2_validation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Dataset
# ============================================================

full_dataset = WorldStratDataset()

total = len(full_dataset)

train_size = int(0.8 * total)
val_size = int(0.1 * total)

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    total,
    generator=generator,
).tolist()

val_indices = indices[
    train_size:
    train_size + val_size
]

val_locations = [
    full_dataset.locations[i]
    for i in val_indices
]

val_dataset = WorldStratDataset(
    locations=val_locations,
    training=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ============================================================
# Model
# ============================================================

model = ESRGANGenerator(
    in_channels=3,
    out_channels=3,
    channels=64,
    num_rrdb=8,
    scale=4,
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Metrics
# ============================================================

psnr_metric = PeakSignalNoiseRatio(
    data_range=1.0
).to(DEVICE)

ssim_metric = StructuralSimilarityIndexMeasure(
    data_range=1.0
).to(DEVICE)


# ============================================================
# Evaluation
# ============================================================

print("Device:", DEVICE)
print("Validation samples:", len(val_dataset))

with torch.no_grad():

    for batch_idx, batch in enumerate(val_loader):

        lr = batch["lr"].to(
            DEVICE,
            non_blocking=True,
        )

        hr = batch["hr"].to(
            DEVICE,
            non_blocking=True,
        )

        sr = model(lr)

        if batch_idx == 0:
            print(
                f"LR range: {lr.min().item():.4f} - {lr.max().item():.4f} | "
                f"SR range: {sr.min().item():.4f} - {sr.max().item():.4f} | "
                f"HR range: {hr.min().item():.4f} - {hr.max().item():.4f}"
            )

            print(
                f"LR mean: {lr.mean().item():.4f}\n"
                f"SR mean: {sr.mean().item():.4f}\n"
                f"HR mean: {hr.mean().item():.4f}"
            )

        # Keep output in valid image range
        sr = sr.clamp(0, 1)

        psnr_metric.update(
            sr,
            hr,
        )

        ssim_metric.update(
            sr,
            hr,
        )

        # Save first few examples
        if batch_idx < 3:

            for i in range(
                min(2, lr.shape[0])
            ):

                sample_number = (
                    batch_idx * BATCH_SIZE + i
                )

                # LR needs to be enlarged only for
                # visualization.
                lr_display = torch.nn.functional.interpolate(
                    lr[i:i + 1],
                    size=(256, 256),
                    mode="bilinear",
                    align_corners=False,
                )[0]

                sr_display = sr[i]

                hr_display = hr[i]

                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(15, 5),
                )

                axes[0].imshow(
                    lr_display.cpu().permute(1, 2, 0)
                )
                axes[0].set_title(
                    "Sentinel-2 LR"
                )
                axes[0].axis("off")

                axes[1].imshow(
                    sr_display.cpu().permute(1, 2, 0)
                )
                axes[1].set_title(
                    "ESRGAN SR"
                )
                axes[1].axis("off")

                axes[2].imshow(
                    hr_display.cpu().permute(1, 2, 0)
                )
                axes[2].set_title(
                    "SPOT HR Reference"
                )
                axes[2].axis("off")

                plt.tight_layout()

                save_path = (
                    RESULTS_DIR
                    / f"sample_{sample_number:03d}.png"
                )

                plt.savefig(
                    save_path,
                    dpi=150,
                    bbox_inches="tight",
                )

                plt.close(fig)


# ============================================================
# Final metrics
# ============================================================

psnr = psnr_metric.compute().item()
ssim = ssim_metric.compute().item()

print("\n" + "=" * 50)

print(
    f"Validation PSNR: {psnr:.4f} dB"
)

print(
    f"Validation SSIM: {ssim:.4f}"
)

print("=" * 50)

print(
    f"\nExample images saved to:"
    f"\n{RESULTS_DIR}"
)