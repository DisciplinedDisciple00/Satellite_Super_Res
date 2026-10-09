from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import WorldStratDataset
from model import ESRGANGenerator


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "pretrain_best.pth"


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4

NUM_WORKERS = 0

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Create the same train/validation split
# ============================================================

full_dataset = WorldStratDataset()

num_locations = len(full_dataset)

generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(num_locations, generator=generator).tolist()

train_size = int(0.8 * num_locations)
val_size = int(0.1 * num_locations)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]

train_locations = [full_dataset.locations[i] for i in train_indices]
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

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

# ============================================================
# Loss + optimizer
# ============================================================

criterion = nn.L1Loss()

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

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):

        lr = batch["lr"].to(
            DEVICE,
            non_blocking=True
        )

        hr = batch["hr"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            sr = model(lr)

            loss = criterion(sr, hr)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_train_loss += loss.item()

        if (batch_idx + 1) % 25 == 0:

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_train_loss = (
        running_train_loss / len(train_loader)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

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

                val_loss = criterion(sr, hr)

            running_val_loss += val_loss.item()

    avg_val_loss = (
        running_val_loss / len(val_loader)
    )


    # --------------------------------------------------------
    # Print epoch results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"Epoch {epoch} complete")
    print(f"Train L1 Loss: {avg_train_loss:.6f}")
    print(f"Val L1 Loss:   {avg_val_loss:.6f}")
    print("=" * 60)


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            CHECKPOINT_PATH
        )

        print(
            f"Best checkpoint saved to:\n"
            f"{CHECKPOINT_PATH}"
        )

    print()


print("Pretraining finished.")
print(f"Best validation loss: {best_val_loss:.6f}")