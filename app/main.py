"""FastAPI service that classifies uploaded images with the exported ONNX model.

Run from the project root, after exporting the model with ``python export_onnx.py``:

    uvicorn app.main:app --host 127.0.0.1 --port 8000

Example request (interactive docs are at http://127.0.0.1:8000/docs):

    curl -X POST http://127.0.0.1:8000/predict -F "file=@cat.png"
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src.config import load_config
from src.data import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES

TOP_K = 3
IMAGE_SIZE = (32, 32)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
INPUT_NAME = "image"

_MEAN = np.array(CIFAR10_MEAN, dtype=np.float32)
_STD = np.array(CIFAR10_STD, dtype=np.float32)


class Prediction(BaseModel):
    """One class and the probability the model assigns to it."""

    label: str
    probability: float


class PredictionResponse(BaseModel):
    """The top-ranked classes for an uploaded image, most likely first."""

    predictions: list[Prediction]


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 32x32 RGB and normalise exactly like the evaluation pipeline.

    Returns:
        A float32 array of shape ``(1, 3, 32, 32)``.
    """
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = (np.asarray(resized, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return pixels.transpose(2, 0, 1)[None]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def create_app(onnx_path: str | Path | None = None) -> FastAPI:
    """Build the API. The ONNX model is loaded once, when the application starts.

    Args:
        onnx_path: Model file; defaults to ``onnx_path`` from the config.
    """
    model_path = Path(onnx_path) if onnx_path is not None else Path(load_config().onnx_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found; run `python export_onnx.py` first")
        app.state.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        yield

    app = FastAPI(title="CIFAR-10 classifier", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report that the service is up and which model file it serves."""
        return {"status": "ok", "model": model_path.name}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(file: UploadFile = File(...)) -> PredictionResponse:
        """Classify an uploaded image and return the top-3 classes with probabilities."""
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image is larger than 5 MB")
        try:
            image = Image.open(BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from error

        logits = app.state.session.run(None, {INPUT_NAME: preprocess(image)})[0]
        probabilities = softmax(logits)[0]
        top = np.argsort(probabilities)[::-1][:TOP_K]
        return PredictionResponse(
            predictions=[Prediction(label=CLASS_NAMES[i], probability=float(probabilities[i])) for i in top]
        )

    return app


app = create_app()
