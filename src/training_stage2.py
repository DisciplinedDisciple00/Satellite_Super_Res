from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import WorldStratDataset
from model import ESRGANGenerator, Discriminator
from losses import GeneratorLoss, DiscriminatorLoss


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
    / "stage2_best.pth"
)

RESUME_CHECKPOINT = BEST_CHECKPOINT

# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4
EPOCHS = 10

GENERATOR_LR = 1e-5
DISCRIMINATOR_LR = 1e-5

PERCEPTUAL_WEIGHT = 0.01
ADVERSARIAL_WEIGHT = 0.001

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
# Models
# ============================================================

generator_model = ESRGANGenerator(
    in_channels=3,
    out_channels=3,
    channels=64,
    num_rrdb=8,
    scale=4
).to(DEVICE)


discriminator_model = Discriminator(
    in_channels=3,
    base_channels=64
).to(DEVICE)


# ============================================================
# Load pretrained generator
# ============================================================


checkpoint = torch.load(
    RESUME_CHECKPOINT,
    map_location=DEVICE
)

generator_model.load_state_dict(
    checkpoint["generator_state_dict"]
)

discriminator_model.load_state_dict(
    checkpoint["discriminator_state_dict"]
)

print(
    f"Loaded Stage 2 checkpoint from:\n"
    f"{RESUME_CHECKPOINT}"
)

# ============================================================
# Losses
# ============================================================

generator_loss_fn = GeneratorLoss(
    perceptual_weight=PERCEPTUAL_WEIGHT,
    adversarial_weight=ADVERSARIAL_WEIGHT
).to(DEVICE)

discriminator_loss_fn = DiscriminatorLoss().to(DEVICE)


# ============================================================
# Optimizers
# ============================================================

generator_optimizer = torch.optim.Adam(
    generator_model.parameters(),
    lr=GENERATOR_LR,
    betas=(0.9, 0.99)
)

discriminator_optimizer = torch.optim.Adam(
    discriminator_model.parameters(),
    lr=DISCRIMINATOR_LR,
    betas=(0.9, 0.99)
)

generator_optimizer.load_state_dict(
    checkpoint["generator_optimizer_state_dict"]
)

discriminator_optimizer.load_state_dict(
    checkpoint["discriminator_optimizer_state_dict"]
)

# ============================================================
# AMP
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler_g = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)

scaler_d = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# Training
# ============================================================

best_val_loss = checkpoint["val_l1"]

start_epoch = checkpoint["epoch"] + 1
end_epoch = start_epoch + EPOCHS

for epoch in range(start_epoch, end_epoch):

    generator_model.train()
    discriminator_model.train()

    running_g_loss = 0.0
    running_d_loss = 0.0
    running_content = 0.0
    running_perceptual = 0.0
    running_adversarial = 0.0


    # --------------------------------------------------------
    # Training loop
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


        # ====================================================
        # 1. Train discriminator
        # ====================================================

        discriminator_optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            fake_hr = generator_model(lr)

            real_output = discriminator_model(hr)

            fake_output = discriminator_model(
                fake_hr.detach()
            )

            d_loss = discriminator_loss_fn(
                real_output,
                fake_output
            )

        scaler_d.scale(d_loss).backward()

        scaler_d.step(
            discriminator_optimizer
        )

        scaler_d.update()


        # ====================================================
        # 2. Train generator
        # ====================================================

        generator_optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            fake_hr = generator_model(lr)

            fake_output = discriminator_model(
                fake_hr
            )

            losses = generator_loss_fn(
                fake_hr,
                hr,
                fake_output
            )

            g_loss = losses["total"]


        scaler_g.scale(g_loss).backward()

        scaler_g.step(
            generator_optimizer
        )

        scaler_g.update()


        # ====================================================
        # Accumulate statistics
        # ====================================================

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

        running_content += (
            losses["content"].item()
        )

        running_perceptual += (
            losses["perceptual"].item()
        )

        running_adversarial += (
            losses["adversarial"].item()
        )


        if (batch_idx + 1) % 25 == 0:

            print(
                f"Epoch [{epoch}/{end_epoch - 1}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"G: {g_loss.item():.5f} "
                f"D: {d_loss.item():.5f}"
            )


    # ========================================================
    # Average training losses
    # ========================================================

    num_batches = len(train_loader)

    avg_g_loss = running_g_loss / num_batches
    avg_d_loss = running_d_loss / num_batches

    avg_content = running_content / num_batches
    avg_perceptual = running_perceptual / num_batches
    avg_adversarial = running_adversarial / num_batches


    # ========================================================
    # Validation
    # ========================================================

    generator_model.eval()

    val_l1 = 0.0

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

                sr = generator_model(lr)

                loss = torch.nn.functional.l1_loss(
                    sr,
                    hr
                )

            val_l1 += loss.item()


    avg_val_l1 = val_l1 / len(val_loader)


    # ========================================================
    # Epoch results
    # ========================================================

    print()
    print("=" * 70)
    print(f"Epoch {epoch} complete")
    print(f"Generator Loss:     {avg_g_loss:.6f}")
    print(f"Discriminator Loss: {avg_d_loss:.6f}")
    print(f"Content Loss:       {avg_content:.6f}")
    print(f"Perceptual Loss:    {avg_perceptual:.6f}")
    print(f"Adversarial Loss:   {avg_adversarial:.6f}")
    print(f"Validation L1:      {avg_val_l1:.6f}")
    print("=" * 70)


    # ========================================================
    # Save best generator
    # ========================================================

    if avg_val_l1 < best_val_loss:

        best_val_loss = avg_val_l1

        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict":
                    generator_model.state_dict(),

                "discriminator_state_dict":
                    discriminator_model.state_dict(),

                "generator_optimizer_state_dict":
                    generator_optimizer.state_dict(),

                "discriminator_optimizer_state_dict":
                    discriminator_optimizer.state_dict(),

                "generator_loss":
                    avg_g_loss,

                "discriminator_loss":
                    avg_d_loss,

                "content_loss":
                    avg_content,

                "perceptual_loss":
                    avg_perceptual,

                "adversarial_loss":
                    avg_adversarial,

                "val_l1":
                    avg_val_l1,
            },
            BEST_CHECKPOINT
        )

        print(
            f"Best Stage 2 checkpoint saved to:\n"
            f"{BEST_CHECKPOINT}"
        )

    print()


print("Stage 2 training finished.")
print(f"Best validation L1: {best_val_loss:.6f}")