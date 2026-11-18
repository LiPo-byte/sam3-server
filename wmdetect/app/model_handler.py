import logging
import os
import threading
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .convnext import ConvNeXt

logger = logging.getLogger("wmdetect")

WEIGHT_NAME = "convnext-tiny_watermarks_detector.pth"


class WatermarkDetector:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"{WEIGHT_NAME} not found at {model_path}")

        model = ConvNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        model.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        start = time.time()
        weights = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights)
        self.model = model.to(self.device).eval()
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._lock = threading.Lock()
        logger.info("watermark detector loaded from %s on %s in %.1fs",
                    model_path, self.device, time.time() - start)

    def predict(self, image: Image.Image) -> tuple[int, float]:
        """Return (label, score). label 1 is watermarked, 0 is clean."""
        batch = self.transforms(image.convert("RGB")).float().unsqueeze(0).to(self.device)
        with self._lock, torch.inference_mode():
            probs = torch.softmax(self.model(batch), dim=1)[0]
            label = int(torch.argmax(probs).item())
        return label, round(float(probs[label]), 4)
