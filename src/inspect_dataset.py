
from pathlib import Path

DATA_DIR = Path("data/mvtec_anomaly_detection/bottle")

def main():
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"
    mask_dir = DATA_DIR / "ground_truth"

    for folder in [train_dir, test_dir, mask_dir]:
        if not folder.exists():
            print(f"Missing folder: {folder}")
            return

    print("Dataset folder structure is valid.\n")

    print("Training images:")
    for folder in sorted(train_dir.iterdir()):
        if folder.is_dir():
            count = sum(
                1 for path in folder.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
            print(f"  {folder.name}: {count} images")

    print("\nTest images:")
    for folder in sorted(test_dir.iterdir()):
        if folder.is_dir():
            count = sum(
                1 for path in folder.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
            print(f"  {folder.name}: {count} images")

    print("\nGround-truth mask folders:")
    for folder in sorted(mask_dir.iterdir()):
        if folder.is_dir():
            print(f"  {folder.name}")

if __name__ == "__main__":
    main()