from pathlib import Path

from PIL import Image
import tifffile


# CHANGE THIS TO YOUR ACTUAL PATH
DATASET_ROOT = Path("/run/media/akileshk/(C:)/dltsih")

HR_ROOT = DATASET_ROOT / "hr_dataset" / "12bit"
LR_ROOT = DATASET_ROOT / "lr_dataset"


def inspect_file(path):
    print(f"\nFILE: {path.name}")

    suffix = path.suffix.lower()

    if suffix == ".png":
        with Image.open(path) as img:
            print(f"  Format: PNG")
            print(f"  Size:   {img.size}")
            print(f"  Mode:   {img.mode}")
            print(f"  Bands:  {img.getbands()}")

    elif suffix in [".tif", ".tiff"]:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]

            print(f"  Format: TIFF")
            print(f"  Shape:  {page.shape}")
            print(f"  Dtype:  {page.dtype}")
            print(f"  Pages:  {len(tif.pages)}")

            image = page.asarray()
            print(f"  Array shape: {image.shape}")
            print(f"  Array dtype: {image.dtype}")


def main():

    # First location that exists in both datasets
    hr_locations = {
        p.name for p in HR_ROOT.iterdir()
        if p.is_dir()
    }

    lr_locations = {
        p.name for p in LR_ROOT.iterdir()
        if p.is_dir()
    }

    common = sorted(hr_locations & lr_locations)

    location = common[0]

    print(f"Inspecting location: {location}")

    # -------------------------
    # HR
    # -------------------------

    hr_folder = HR_ROOT / location

    print("\n================ HR ================")

    for path in sorted(hr_folder.iterdir()):
        if path.suffix.lower() in [".png", ".tif", ".tiff"]:
            inspect_file(path)

    # -------------------------
    # LR
    # -------------------------

    lr_folder = LR_ROOT / location

    print("\n================ LR ================")

    # Recursively search because L2A is nested
    lr_files = sorted(
        p for p in lr_folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in [".tif", ".tiff"]
    )

    for path in lr_files:
        inspect_file(path)


if __name__ == "__main__":
    main()