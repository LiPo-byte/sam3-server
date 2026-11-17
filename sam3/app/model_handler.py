import logging
import os
import threading
import time

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

logger = logging.getLogger("sam3")


class Sam3Segmenter:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._lock = threading.Lock()

        if not os.path.isfile(os.path.join(model_path, "model.safetensors")):
            raise FileNotFoundError(f"model.safetensors not found in {model_path}")

        start = time.time()
        self.model = Sam3Model.from_pretrained(model_path, dtype=self.dtype).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_path)
        logger.info("SAM3 loaded from %s on %s (%s) in %.1fs",
                    model_path, self.device, self.dtype, time.time() - start)

    def segment(self, image: Image.Image, prompts: list[str], threshold: float, mask_threshold: float):
        """Yield (prompt, score, box[x1, y1, x2, y2], mask[H, W] uint8) for every instance found."""
        with self._lock, torch.inference_mode():
            for prompt in prompts:
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, self.dtype)
                outputs = self.model(**inputs)
                result = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                scores = result["scores"].float().cpu().tolist()
                boxes = result["boxes"].float().cpu().tolist()
                masks = result["masks"].cpu().numpy().astype(np.uint8)
                for score, box, mask in zip(scores, boxes, masks):
                    yield prompt, score, box, mask
