from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader


# WorldStrat dataset location
DATASET_ROOT = Path("/run/media/akileshk/(C:)/dltsih")

HR_ROOT = DATASET_ROOT / "hr_dataset" / "12bit"
LR_ROOT = DATASET_ROOT / "lr_dataset"

def crop_corresponding_pair(lr, hr, x, y, crop_size=64, scale=4):
    """
    Crop an LR patch and its corresponding region from the native HR image.

    LR patch:
        crop_size x crop_size

    HR target:
        crop_size * scale x crop_size * scale
    """

    _, lr_height, lr_width = lr.shape
    _, hr_height, hr_width = hr.shape

    # Corresponding HR coordinates based on proportional position
    hr_x = round(x / lr_width * hr_width)
    hr_y = round(y / lr_height * hr_height)

    # Native HR region corresponding to the LR crop
    hr_crop_width = round(crop_size / lr_width * hr_width)
    hr_crop_height = round(crop_size / lr_height * hr_height)

    # Crop LR
    lr_crop = lr[
        :,
        y:y + crop_size,
        x:x + crop_size,
    ]

    # Crop HR
    hr_crop = hr[
        :,
        hr_y:hr_y + hr_crop_height,
        hr_x:hr_x + hr_crop_width,
    ]

    # Resize native HR region to our 4x target
    hr_crop = torch.nn.functional.interpolate(
        hr_crop.unsqueeze(0),
        size=(crop_size * scale, crop_size * scale),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return lr_crop, hr_crop

class WorldStratDataset(Dataset):
    def __init__(
            self,
            root=DATASET_ROOT,
            crop_size=64,
            scale=4,
            locations=None,
            training=True,
    ):
        self.root = Path(root)

        self.hr_root = self.root / "hr_dataset" / "12bit"
        self.lr_root = self.root / "lr_dataset"

        self.crop_size = crop_size
        self.scale = scale
        self.training = training

        # Find locations that have both HR and LR data
        hr_locations = {p.name for p in self.hr_root.iterdir() if p.is_dir()}
        lr_locations = {p.name for p in self.lr_root.iterdir() if p.is_dir()}

        all_locations = sorted(hr_locations & lr_locations)

        if locations is None:
            self.locations = all_locations
        else:
            self.locations = list(locations)

        print(f"Dataset contains {len(self.locations)} locations")

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        location = self.locations[idx]

        # -------------------------
        # Find files
        # -------------------------
        hr_path = self.hr_root / location / f"{location}_ps.tiff"

        lr_dir = self.lr_root / location / "L2A"
        lr_files = sorted(lr_dir.glob("*-L2A_data.tiff"))

        if not lr_files:
            raise FileNotFoundError(f"No Sentinel-2 data found for {location}")

        # Random revisit during training
        # Fixed first revisit during validation/test
        if self.training:
            revisit_idx = torch.randint(len(lr_files), (1,)).item()
        else:
            revisit_idx = 0

        lr_path = lr_files[revisit_idx]

        # -------------------------
        # Load Sentinel-2 RGB
        # -------------------------
        with rasterio.open(lr_path) as src:
            # B04, B03, B02 = RGB
            lr = src.read([4, 3, 2])

        # -------------------------
        # Load SPOT RGB
        # -------------------------
        with rasterio.open(hr_path) as src:
            # Bands 1, 2, 3 = RGB
            hr = src.read([1, 2, 3])

        # Convert to tensors
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()

        # -------------------------
        # Random LR crop
        # -------------------------
        _, lr_height, lr_width = lr.shape

        if lr_height < self.crop_size or lr_width < self.crop_size:
            raise ValueError(
                f"LR image too small for crop: "
                f"{lr.shape}"
            )

        if self.training:
            x = torch.randint(
                0,
                lr_width - self.crop_size + 1,
                (1,),
            ).item()

            y = torch.randint(
                0,
                lr_height - self.crop_size + 1,
                (1,),
            ).item()
        else:
            # Center crop for validation/test
            x = (lr_width - self.crop_size) // 2
            y = (lr_height - self.crop_size) // 2

        # -------------------------
        # Create corresponding pair
        # -------------------------
        lr_crop, hr_crop = crop_corresponding_pair(
            lr,
            hr,
            x=x,
            y=y,
            crop_size=self.crop_size,
            scale=self.scale,
        )

        # Normalize to [0, 1]
        lr_crop = lr_crop.clamp(0, 1)

        hr_crop = (hr_crop / 10000.0).clamp(0, 1)

        return {
            "lr": lr_crop,
            "hr": hr_crop,
            "location": location,
        }

if __name__ == "__main__":
    full_dataset = WorldStratDataset()

    total = len(full_dataset)

    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    # Reproducible location split
    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(total, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_locations = [
        full_dataset.locations[i]
        for i in train_indices
    ]

    val_locations = [
        full_dataset.locations[i]
        for i in val_indices
    ]

    test_locations = [
        full_dataset.locations[i]
        for i in test_indices
    ]

    # Create separate datasets
    train_dataset = WorldStratDataset(
        locations=train_locations,
        training=True,
    )

    val_dataset = WorldStratDataset(
        locations=val_locations,
        training=False,
    )

    test_dataset = WorldStratDataset(
        locations=test_locations,
        training=False,
    )

    print("\nSplit sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))