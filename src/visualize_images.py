
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path("data/mvtec_anomaly_detection/bottle")


def find_first_image(folder):
    """Find the first supported image in a folder."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}

    if not folder.exists():
        return None

    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    return images[0] if images else None


def main():
    # 1. Find a normal image.
    good_path = find_first_image(DATA_DIR / "train" / "good")

    if good_path is None:
        print("No normal training images found.")
        print("Check the dataset path.")
        return

    # 2. Find a defective test image.
    test_dir = DATA_DIR / "test"
    defect_path = None
    defect_type = None

    for folder in sorted(test_dir.iterdir()):
        if folder.is_dir() and folder.name != "good":
            defect_path = find_first_image(folder)

            if defect_path is not None:
                defect_type = folder.name
                break

    if defect_path is None:
        print("No defective test images found.")
        return

    # 3. Locate the corresponding ground-truth mask.
    mask_path = (
        DATA_DIR
        / "ground_truth"
        / defect_type
        / f"{defect_path.stem}_mask.png"
    )

    # 4. Load images.
    good_image = Image.open(good_path).convert("RGB")
    defect_image = Image.open(defect_path).convert("RGB")

    # 5. Display the results.
    columns = 3 if mask_path.exists() else 2
    fig, axes = plt.subplots(1, columns, figsize=(15, 5))

    axes[0].imshow(good_image)
    axes[0].set_title("Normal product")
    axes[0].axis("off")

    axes[1].imshow(defect_image)
    axes[1].set_title(f"Defective: {defect_type}")
    axes[1].axis("off")

    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Ground-truth defect mask")
        axes[2].axis("off")
    else:
        print(f"Mask not found: {mask_path}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()