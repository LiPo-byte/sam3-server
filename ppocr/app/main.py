import base64
import binascii
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .model_handler import PpOcr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ppocr")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_dir = os.environ.get("PPOCR_MODEL_DIR", "/models/ppocrv6")
    state["model"] = PpOcr(model_dir, device=os.environ.get("PPOCR_DEVICE") or None)
    yield
    state.clear()


app = FastAPI(title="PP-OCRv6 API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("PPOCR_CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class OcrRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image, data URL prefix is allowed")
    score_threshold: float = Field(0.0, ge=0, le=1, description="drop lines scored below this")
    use_doc_orientation_classify: bool = Field(
        False, description="rotate the page by 0/90/180/270 before detection; requires the extra model")
    use_doc_unwarping: bool = Field(
        False, description="flatten a curved page before detection; changes coordinates; requires the extra model")
    use_textline_orientation: bool = Field(
        False, description="fix upside-down text lines; requires the extra model")


class OcrLine(BaseModel):
    text: str
    score: float
    box: list[float]
    poly: list[list[float]]


class OcrResponse(BaseModel):
    width: int
    height: int
    elapsed_ms: int
    results: list[OcrLine]


def _decode_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")


def _require_module(enabled: bool, loaded: bool, name: str):
    if enabled and not loaded:
        raise HTTPException(
            status_code=400,
            detail=f"{name} model is not loaded; download it with DOWNLOAD_PREPROCESS=1",
        )


@app.get("/health")
def health():
    model = state.get("model")
    if not model:
        return {"status": "loading", "device": None}
    return {
        "status": "ok",
        "device": model.device,
        "doc_orientation": model.doc_orientation,
        "doc_unwarping": model.doc_unwarping,
        "textline_orientation": model.textline_orientation,
    }


@app.post("/ocr", response_model=OcrResponse)
def ocr(req: OcrRequest):
    model: PpOcr = state["model"]
    _require_module(req.use_doc_orientation_classify, model.doc_orientation, "doc orientation")
    _require_module(req.use_doc_unwarping, model.doc_unwarping, "doc unwarping")
    _require_module(req.use_textline_orientation, model.textline_orientation, "textline orientation")

    image = _decode_image(req.image)
    bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    start = time.time()
    lines = model.recognize(
        bgr,
        req.score_threshold,
        req.use_doc_orientation_classify,
        req.use_doc_unwarping,
        req.use_textline_orientation,
    )
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info("ocr %dx%d lines=%d %dms", image.width, image.height, len(lines), elapsed_ms)
    return OcrResponse(
        width=image.width,
        height=image.height,
        elapsed_ms=elapsed_ms,
        results=lines,
    )
