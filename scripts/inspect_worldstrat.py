from pathlib import Path
import csv

# CHANGE THIS to the Linux-mounted path of your C: drive
DATASET_ROOT = Path("/run/media/akileshk/(C:)/dltsih")

HR_ROOT = DATASET_ROOT / "hr_dataset" / "12bit"
LR_ROOT = DATASET_ROOT / "lr_dataset"
METADATA = DATASET_ROOT / "metadata.csv"


def inspect_directories():
    print("\n=== DATASET PATHS ===")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"HR:      {HR_ROOT}")
    print(f"LR:      {LR_ROOT}")
    print(f"Metadata:{METADATA}")

    print("\n=== EXISTENCE ===")
    for path in [DATASET_ROOT, HR_ROOT, LR_ROOT, METADATA]:
        print(f"{path}: {'OK' if path.exists() else 'NOT FOUND'}")


def inspect_locations():
    hr_locations = sorted(p.name for p in HR_ROOT.iterdir() if p.is_dir())
    lr_locations = sorted(p.name for p in LR_ROOT.iterdir() if p.is_dir())

    print("\n=== LOCATIONS ===")
    print(f"HR locations: {len(hr_locations)}")
    print(f"LR locations: {len(lr_locations)}")

    print("\nFirst 10 HR locations:")
    for name in hr_locations[:10]:
        print(f"  {name}")

    print("\nFirst 10 LR locations:")
    for name in lr_locations[:10]:
        print(f"  {name}")

    print("\n=== MATCHING ===")
    hr_set = set(hr_locations)
    lr_set = set(lr_locations)

    print(f"Locations in both: {len(hr_set & lr_set)}")
    print(f"HR only:           {len(hr_set - lr_set)}")
    print(f"LR only:           {len(lr_set - hr_set)}")


def inspect_example():
    locations = sorted(p.name for p in HR_ROOT.iterdir() if p.is_dir())

    if not locations:
        print("\nNo HR locations found.")
        return

    location = locations[0]

    hr_folder = HR_ROOT / location
    lr_folder = LR_ROOT / location

    print(f"\n=== EXAMPLE LOCATION ===")
    print(location)

    print("\nHR files:")
    for file in sorted(hr_folder.iterdir()):
        if file.is_file():
            print(f"  {file.name}")

    print("\nLR files:")
    for file in sorted(lr_folder.iterdir()):
        if file.is_file():
            print(f"  {file.name}")


def inspect_metadata():
    print("\n=== METADATA CSV ===")

    with METADATA.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        header = next(reader)
        print("Columns:")
        for column in header:
            print(f"  {column}")

        print("\nFirst 3 rows:")
        for i, row in enumerate(reader):
            print(row)
            if i == 2:
                break


if __name__ == "__main__":
    inspect_directories()
    inspect_locations()
    inspect_example()
    inspect_metadata()