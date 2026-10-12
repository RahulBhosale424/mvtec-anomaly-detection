
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download pretrained ResNet-18 weights during image building.
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.DEFAULT)"

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]