from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WorldStratDataset
from model import ESRGANGenerator
from losses import PerceptualLoss


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRETRAIN_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "pretrain_best.pth"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ablation_perceptual_best.pth"
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4
EPOCHS = 5

LEARNING_RATE = 1e-5

PERCEPTUAL_WEIGHT = 0.01

NUM_WORKERS = 0
SEED = 42


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Dataset split
# ============================================================

full_dataset = WorldStratDataset()

num_locations = len(full_dataset)

generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    num_locations,
    generator=generator
).tolist()

train_size = int(0.8 * num_locations)
val_size = int(0.1 * num_locations)

train_indices = indices[:train_size]

val_indices = indices[
    train_size:
    train_size + val_size
]

train_locations = [
    full_dataset.locations[i]
    for i in train_indices
]

val_locations = [
    full_dataset.locations[i]
    for i in val_indices
]


train_dataset = WorldStratDataset(
    locations=train_locations,
    training=True
)

val_dataset = WorldStratDataset(
    locations=val_locations,
    training=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


print(f"Device: {DEVICE}")
print(f"Train locations: {len(train_dataset)}")
print(f"Validation locations: {len(val_dataset)}")


# ============================================================
# Model
# ============================================================

model = ESRGANGenerator(
    in_channels=3,
    out_channels=3,
    channels=64,
    num_rrdb=8,
    scale=4
).to(DEVICE)


# ============================================================
# Load Stage-1 pretrained generator
# ============================================================

checkpoint = torch.load(
    PRETRAIN_CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Loaded pretrained generator from:\n"
    f"{PRETRAIN_CHECKPOINT}"
)


# ============================================================
# Losses
# ============================================================

content_loss_fn = nn.L1Loss()

perceptual_loss_fn = PerceptualLoss().to(DEVICE)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.99)
)


# ============================================================
# AMP
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_total = 0.0
    running_content = 0.0
    running_perceptual = 0.0


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for batch_idx, batch in enumerate(train_loader):

        lr = batch["lr"].to(
            DEVICE,
            non_blocking=True
        )

        hr = batch["hr"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            sr = model(lr)

            content_loss = content_loss_fn(
                sr,
                hr
            )

            perceptual_loss = perceptual_loss_fn(
                sr,
                hr
            )

            total_loss = (
                content_loss
                + PERCEPTUAL_WEIGHT * perceptual_loss
            )


        scaler.scale(total_loss).backward()

        scaler.step(optimizer)

        scaler.update()


        running_total += total_loss.item()

        running_content += content_loss.item()

        running_perceptual += (
            perceptual_loss.item()
        )


        if (batch_idx + 1) % 25 == 0:

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {total_loss.item():.5f}"
            )


    # --------------------------------------------------------
    # Average training losses
    # --------------------------------------------------------

    num_batches = len(train_loader)

    avg_total = running_total / num_batches

    avg_content = running_content / num_batches

    avg_perceptual = (
        running_perceptual / num_batches
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    running_val_l1 = 0.0

    with torch.no_grad():

        for batch in val_loader:

            lr = batch["lr"].to(
                DEVICE,
                non_blocking=True
            )

            hr = batch["hr"].to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):

                sr = model(lr)

                val_loss = content_loss_fn(
                    sr,
                    hr
                )

            running_val_l1 += val_loss.item()


    avg_val_l1 = (
        running_val_l1 / len(val_loader)
    )


    # --------------------------------------------------------
    # Epoch summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(f"Epoch {epoch} complete")
    print(f"Total Loss:       {avg_total:.6f}")
    print(f"Content Loss:     {avg_content:.6f}")
    print(f"Perceptual Loss:  {avg_perceptual:.6f}")
    print(f"Validation L1:    {avg_val_l1:.6f}")
    print("=" * 70)


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if avg_val_l1 < best_val_loss:

        best_val_loss = avg_val_l1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "total_loss":
                    avg_total,

                "content_loss":
                    avg_content,

                "perceptual_loss":
                    avg_perceptual,

                "val_l1":
                    avg_val_l1,
            },
            BEST_CHECKPOINT
        )

        print(
            f"Best ablation checkpoint saved to:\n"
            f"{BEST_CHECKPOINT}"
        )

    print()


print("Perceptual ablation finished.")
print(f"Best validation L1: {best_val_loss:.6f}")