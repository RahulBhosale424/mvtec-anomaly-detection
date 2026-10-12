
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DATA_DIR = Path("data/mvtec_anomaly_detection/bottle")
MODEL_DIR = Path("models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16


class LabeledImageDataset(Dataset):
    """Read normal and defective test images with image-level labels."""

    def __init__(self, test_dir):
        self.samples = []

        for folder in sorted(Path(test_dir).iterdir()):
            if not folder.is_dir():
                continue

            label = 0 if folder.name == "good" else 1

            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".bmp"
                }:
                    self.samples.append((path, label))

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return image, label


def extract_features(model, loader):
    features = []
    labels = []

    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(DEVICE)
            batch_features = model(images)

            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())

    return np.concatenate(features), np.asarray(labels)


def main():
    detector_path = MODEL_DIR / "isolation_forest.joblib"

    if not detector_path.exists():
        raise FileNotFoundError(
            "Train the model first by running: python src/train.py"
        )

    detector = joblib.load(detector_path)

    weights = models.ResNet18_Weights.DEFAULT
    feature_extractor = models.resnet18(weights=weights)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()

    dataset = LabeledImageDataset(DATA_DIR / "test")

    if len(dataset) == 0:
        raise ValueError("No test images found.")

    labels = np.asarray([label for _, label in dataset.samples])

    if len(np.unique(labels)) != 2:
        raise ValueError("Test data must contain both normal and defective images.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(dataset)}")
    print("Extracting test image features...")

    features, labels = extract_features(feature_extractor, loader)

    # Larger scores mean more anomalous.
    scores = -detector.decision_function(features)

    # Use a deterministic, stratified split of test data:
    # one part is used to select a threshold, the other to evaluate it.
    indices = np.arange(len(labels))

    threshold_idx, evaluation_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=42,
        stratify=labels,
    )

    threshold_labels = labels[threshold_idx]
    threshold_scores = scores[threshold_idx]

    # Select the threshold that maximizes F1 on the threshold subset.
    candidates = np.unique(threshold_scores)

    best_threshold = None
    best_f1 = -1.0

    for threshold in candidates:
        predictions = (threshold_scores >= threshold).astype(int)
        score = f1_score(
            threshold_labels,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    eval_labels = labels[evaluation_idx]
    eval_scores = scores[evaluation_idx]
    eval_predictions = (eval_scores >= best_threshold).astype(int)

    auc = roc_auc_score(eval_labels, eval_scores)
    precision = precision_score(
        eval_labels, eval_predictions, zero_division=0
    )
    recall = recall_score(
        eval_labels, eval_predictions, zero_division=0
    )
    f1 = f1_score(
        eval_labels, eval_predictions, zero_division=0
    )

    print("\nEvaluation results")
    print("------------------")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Threshold: {best_threshold:.6f}")

    print("\nConfusion matrix:")
    print("Rows = actual [normal, defective]")
    print("Cols = predicted [normal, defective]")
    print(confusion_matrix(eval_labels, eval_predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        eval_labels,
        eval_predictions,
        labels=[0, 1],
        target_names=["normal", "defective"],
        zero_division=0,
    ))

    # Save the chosen threshold for the API.
    joblib.dump(
        {"anomaly_threshold": best_threshold},
        MODEL_DIR / "threshold.joblib",
    )

    print(f"\nThreshold saved to {MODEL_DIR / 'threshold.joblib'}")


if __name__ == "__main__":
    main()