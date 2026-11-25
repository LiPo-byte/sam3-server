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


class CornerResult(BaseModel):
    corner: str
    box: list[int]
    watermarked: bool
    label: int
    score: float


class DetectResponse(BaseModel):
    width: int
    height: int
    elapsed_ms: int
    watermarked: bool
    label: int
    score: float
    corners: list[CornerResult]


def _corner_boxes(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """Square crops of the four corners. Side is 20% of the short side.

    A 2048 image yields about 410px, a 2800 image about 560px. The detector
    then resizes each crop to 256, so a small corner mark stays large enough
    to see. Full-image resize would shrink that mark to a few pixels.
    """
    ratio = float(os.environ.get("WM_CORNER_RATIO", "0.2"))
    side = max(1, int(round(min(width, height) * ratio)))
    side = min(side, width, height)
    return {
        "tl": (0, 0, side, side),
        "tr": (width - side, 0, width, side),
        "bl": (0, height - side, side, height),
        "br": (width - side, height - side, width, height),
    }


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
    corners = []
    for name, box in _corner_boxes(image.width, image.height).items():
        label, score = state["model"].predict(image.crop(box))
        corners.append(CornerResult(
            corner=name,
            box=list(box),
            # 有水印且命中分数需大于0.6才算命中，否则不算命中
            watermarked=bool(label) and score > 0.6,
            label=label,
            score=score,
        ))
    hits = [c for c in corners if c.watermarked]
    chosen = max(hits or corners, key=lambda c: c.score)
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info("detect %dx%d watermarked=%s via %s score=%.4f corners=%s %dms",
                image.width, image.height, bool(hits), chosen.corner, chosen.score,
                ",".join(f"{c.corner}:{int(c.watermarked)}" for c in corners), elapsed_ms)
    return DetectResponse(
        width=image.width,
        height=image.height,
        elapsed_ms=elapsed_ms,
        watermarked=bool(hits),
        label=1 if hits else 0,
        score=chosen.score,
        corners=corners,
    )
