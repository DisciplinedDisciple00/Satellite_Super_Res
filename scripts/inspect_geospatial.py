from pathlib import Path
import rasterio


DATASET_ROOT = Path("/run/media/akileshk/(C:)/dltsih")

HR_ROOT = DATASET_ROOT / "hr_dataset" / "12bit"
LR_ROOT = DATASET_ROOT / "lr_dataset"


def inspect_raster(path):
    print("\n" + "=" * 60)
    print(path)

    with rasterio.open(path) as src:
        print(f"CRS:        {src.crs}")
        print(f"Width:      {src.width}")
        print(f"Height:     {src.height}")
        print(f"Channels:   {src.count}")
        print(f"Dtype:      {src.dtypes}")
        print(f"Resolution: {src.res}")
        print(f"Bounds:     {src.bounds}")
        print(f"Transform:  {src.transform}")

        print("\nBand descriptions:")
        for i in range(1, src.count + 1):
            print(f"  Band {i}: {src.descriptions[i - 1]}")

        print("\nStatistics:")
        for i in range(1, src.count + 1):
            band = src.read(i)

            print(
                f"  Band {i}: "
                f"min={band.min():.6f}, "
                f"max={band.max():.6f}, "
                f"mean={band.mean():.6f}"
            )


def main():

    location = "ASMSpotter-1-1-1"

    hr_folder = HR_ROOT / location
    lr_folder = LR_ROOT / location

    hr_rgb = hr_folder / f"{location}_rgb.png"

    # Find first L2A data file
    l2a_files = sorted(lr_folder.rglob("*L2A_data.tiff"))

    if not l2a_files:
        raise FileNotFoundError("No L2A_data.tiff found")

    lr_l2a = l2a_files[0]

    hr_ps = hr_folder / f"{location}_ps.tiff"

    print("HR PANSHARPENED TIFF")
    print("====================")

    inspect_raster(hr_ps)
    print(
        "Note: PNG itself does not carry the same geospatial metadata "
        "as the source TIFF."
    )

    print(f"\nHR RGB file: {hr_rgb}")

    print("\nLR L2A")
    print("======")

    inspect_raster(lr_l2a)


if __name__ == "__main__":
    main()