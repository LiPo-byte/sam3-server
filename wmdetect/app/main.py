import base64
import binascii
import io
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .model_handler import WatermarkDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("wmdetect")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_path = os.environ.get("WM_MODEL_PATH", "/models/convnext-tiny_watermarks_detector.pth")
    state["model"] = WatermarkDetector(model_path, device=os.environ.get("WM_DEVICE") or None)
    yield
    state.clear()


app = FastAPI(title="Watermark Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("WM_CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image, data URL prefix is allowed")


class DetectResponse(BaseModel):
    width: int
    height: int
    elapsed_ms: int
    watermarked: bool
    label: int
    score: float


def _decode_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")


@app.get("/health")
def health():
    model = state.get("model")
    return {"status": "ok" if model else "loading", "device": model.device if model else None}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    image = _decode_image(req.image)
    start = time.time()
    label, score = state["model"].predict(image)
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info("detect %dx%d watermarked=%s score=%.4f %dms",
                image.width, image.height, bool(label), score, elapsed_ms)
    return DetectResponse(
        width=image.width,
        height=image.height,
        elapsed_ms=elapsed_ms,
        watermarked=bool(label),
        label=label,
        score=score,
    )
