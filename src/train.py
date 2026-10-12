
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import IsolationForest
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path("data/mvtec_anomaly_detection/bottle")
MODEL_DIR = Path("models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
BATCH_SIZE = 16


class ImageFolderDataset(Dataset):
    """Load images from a folder without requiring class labels."""

    def __init__(self, folder):
        self.paths = sorted(
            p for p in Path(folder).rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


def create_feature_extractor():
    """Load pretrained ResNet-18 and remove its classification layer."""

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Keep the learned visual features, not the 1,000-class output.
    model.fc = torch.nn.Identity()
    model = model.to(DEVICE)
    model.eval()

    return model


def extract_features(model, dataset):
    if len(dataset) == 0:
        raise ValueError("No images found in the specified folder.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    batches = []

    with torch.inference_mode():
        for images in loader:
            images = images.to(DEVICE)
            features = model(images)
            batches.append(features.cpu().numpy())

    return np.concatenate(batches, axis=0)


def main():
    train_dir = DATA_DIR / "train" / "good"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found: {train_dir.resolve()}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print("Loading normal training images...")

    train_dataset = ImageFolderDataset(train_dir)
    print(f"Number of training images: {len(train_dataset)}")

    print("Loading pretrained ResNet-18...")
    feature_extractor = create_feature_extractor()

    print("Extracting image features...")
    train_features = extract_features(feature_extractor, train_dataset)

    print("Training Isolation Forest...")
    detector = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    detector.fit(train_features)

    # Persist the detector and preprocessing configuration.
    joblib.dump(detector, MODEL_DIR / "isolation_forest.joblib")

    metadata = {
        "image_size": IMAGE_SIZE,
        "feature_dimension": int(train_features.shape[1]),
        "architecture": "resnet18",
        "weights": "ResNet18_Weights.DEFAULT",
    }
    joblib.dump(metadata, MODEL_DIR / "metadata.joblib")

    print("\nTraining complete!")
    print(f"Feature matrix shape: {train_features.shape}")
    print(f"Detector saved to: {MODEL_DIR / 'isolation_forest.joblib'}")
    print(f"Metadata saved to: {MODEL_DIR / 'metadata.joblib'}")


if __name__ == "__main__":
    main()