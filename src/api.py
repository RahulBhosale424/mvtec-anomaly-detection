
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Industrial Visual Defect Detection API",
    description="Detect anomalous bottle images using MVTec AD.",
    version="1.0.0",
)

detector = None
threshold = None
feature_extractor = None

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.on_event("startup")
def load_model():
    """Load model artifacts once when the service starts."""
    global detector, threshold, feature_extractor

    detector_path = MODEL_DIR / "isolation_forest.joblib"
    threshold_path = MODEL_DIR / "threshold.joblib"

    if not detector_path.exists() or not threshold_path.exists():
        raise RuntimeError(
            "Model artifacts are missing. Train and evaluate the model first."
        )

    detector = joblib.load(detector_path)
    threshold_data = joblib.load(threshold_path)
    threshold = float(threshold_data["anomaly_threshold"])

    # Use the same pretrained ResNet-18 feature extractor as training.
    weights = models.ResNet18_Weights.DEFAULT
    feature_extractor = models.resnet18(weights=weights)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor = feature_extractor.to(DEVICE)
    feature_extractor.eval()


@app.get("/")
def root():
    return {
        "message": "Industrial Visual Defect Detection API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": detector is not None
        and feature_extractor is not None,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if detector is None or feature_extractor is None or threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB limit.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            image = image.convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        image_tensor = image_tensor.to(DEVICE)

        with torch.inference_mode():
            features = feature_extractor(image_tensor)
            features = features.cpu().numpy()

        # Larger score means more anomalous.
        score = float(-detector.decision_function(features)[0])

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    prediction = (
        "anomalous" if score >= threshold else "normal"
    )

    return {
        "filename": file.filename,
        "prediction": prediction,
        "anomaly_score": round(score, 6),
        "threshold": round(threshold, 6),
        "model": "ResNet-18 + Isolation Forest",
        "note": "Anomaly score is not a calibrated probability.",
    }