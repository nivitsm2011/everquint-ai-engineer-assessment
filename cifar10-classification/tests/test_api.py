"""Tests for the FastAPI service, using an untrained model exported to a temporary ONNX file."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import datasets

from app.main import TOP_K, create_app, preprocess, softmax
from export_onnx import export_model
from src.config import load_config
from src.data import CIFAR10_MEAN, CIFAR10_STD, CLASS_NAMES
from src.models import build_model


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    onnx_path = export_model(build_model("baseline_cnn"), tmp_path_factory.mktemp("onnx") / "model.onnx")
    with TestClient(create_app(onnx_path)) as test_client:
        yield test_client


def _png_bytes(size: tuple[int, int] = (32, 32), mode: str = "RGB") -> bytes:
    pixels = np.random.default_rng(0).integers(0, 256, size=(*size, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).convert(mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _predict(client: TestClient, content: bytes, name: str = "image.png"):
    return client.post("/predict", files={"file": (name, content, "image/png")})


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "model.onnx"}


def test_predict_returns_top3_sorted_probabilities(client: TestClient) -> None:
    response = _predict(client, _png_bytes())
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == TOP_K
    probabilities = [p["probability"] for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert 0 < sum(probabilities) <= 1 + 1e-6
    assert all(p["label"] in CLASS_NAMES for p in predictions)
    assert len({p["label"] for p in predictions}) == TOP_K


@pytest.mark.parametrize("size,mode", [((64, 48), "RGB"), ((32, 32), "L"), ((100, 100), "RGBA")])
def test_predict_accepts_other_sizes_and_modes(client: TestClient, size: tuple[int, int], mode: str) -> None:
    assert _predict(client, _png_bytes(size, mode)).status_code == 200


@pytest.mark.skipif(
    not (Path(load_config().data_dir) / "cifar-10-batches-py").exists(), reason="CIFAR-10 is not downloaded"
)
def test_predict_accepts_a_real_cifar10_test_image(client: TestClient) -> None:
    image, _ = datasets.CIFAR10(load_config().data_dir, train=False, download=False)[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    response = _predict(client, buffer.getvalue(), name="test_image_0.png")
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == TOP_K


def test_predict_is_deterministic(client: TestClient) -> None:
    assert _predict(client, _png_bytes()).json() == _predict(client, _png_bytes()).json()


def test_predict_rejects_non_image_upload(client: TestClient) -> None:
    response = _predict(client, b"definitely not an image", name="notes.txt")
    assert response.status_code == 400


def test_predict_requires_a_file(client: TestClient) -> None:
    assert client.post("/predict").status_code == 422


def test_preprocess_matches_training_normalisation() -> None:
    pixel = (120, 60, 200)
    array = preprocess(Image.new("RGB", (50, 70), pixel))
    assert array.shape == (1, 3, 32, 32) and array.dtype == np.float32
    for channel in range(3):
        expected = (pixel[channel] / 255 - CIFAR10_MEAN[channel]) / CIFAR10_STD[channel]
        assert np.allclose(array[0, channel], expected, atol=1e-5)


def test_softmax_sums_to_one_and_is_stable() -> None:
    probabilities = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_app_fails_at_startup_when_model_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="export_onnx"):
        with TestClient(create_app(tmp_path / "missing.onnx")):
            pass
