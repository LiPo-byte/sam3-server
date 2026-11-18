import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger("ppocr")

DET_NAME = "PP-OCRv6_medium_det"
REC_NAME = "PP-OCRv6_medium_rec"
DOC_ORI_NAME = "PP-LCNet_x1_0_doc_ori"
UNWARP_NAME = "UVDoc"
TEXTLINE_NAME = "PP-LCNet_x0_25_textline_ori"


def _model_ready(path: str) -> bool:
    return any(os.path.isfile(os.path.join(path, name)) for name in (
        "inference.yml", "inference.json", "inference.pdmodel"))


class PpOcr:
    def __init__(self, model_dir: str, device: str | None = None):
        from paddleocr import PaddleOCR

        self.device = device or "gpu:0"
        det_dir = os.path.join(model_dir, DET_NAME)
        rec_dir = os.path.join(model_dir, REC_NAME)
        if not _model_ready(det_dir):
            raise FileNotFoundError(f"{DET_NAME} not found in {model_dir}")
        if not _model_ready(rec_dir):
            raise FileNotFoundError(f"{REC_NAME} not found in {model_dir}")

        doc_ori_dir = os.path.join(model_dir, DOC_ORI_NAME)
        unwarp_dir = os.path.join(model_dir, UNWARP_NAME)
        textline_dir = os.path.join(model_dir, TEXTLINE_NAME)
        self.doc_orientation = _model_ready(doc_ori_dir)
        self.doc_unwarping = _model_ready(unwarp_dir)
        self.textline_orientation = _model_ready(textline_dir)

        kwargs = dict(
            text_detection_model_name=DET_NAME,
            text_detection_model_dir=det_dir,
            text_recognition_model_name=REC_NAME,
            text_recognition_model_dir=rec_dir,
            use_doc_orientation_classify=self.doc_orientation,
            use_doc_unwarping=self.doc_unwarping,
            use_textline_orientation=self.textline_orientation,
            device=self.device,
        )
        if self.doc_orientation:
            kwargs["doc_orientation_classify_model_name"] = DOC_ORI_NAME
            kwargs["doc_orientation_classify_model_dir"] = doc_ori_dir
        if self.doc_unwarping:
            kwargs["doc_unwarping_model_name"] = UNWARP_NAME
            kwargs["doc_unwarping_model_dir"] = unwarp_dir
        if self.textline_orientation:
            kwargs["textline_orientation_model_name"] = TEXTLINE_NAME
            kwargs["textline_orientation_model_dir"] = textline_dir

        start = time.time()
        self.ocr = PaddleOCR(**kwargs)
        self._lock = threading.Lock()
        logger.info(
            "PP-OCRv6 loaded from %s on %s in %.1fs (doc_ori=%s unwarp=%s textline=%s)",
            model_dir, self.device, time.time() - start,
            self.doc_orientation, self.doc_unwarping, self.textline_orientation)

    def recognize(self, image_bgr: np.ndarray, score_threshold: float,
                  use_doc_orientation_classify: bool, use_doc_unwarping: bool,
                  use_textline_orientation: bool):
        """Return a list of {text, score, box, poly} for one BGR image."""
        with self._lock:
            output = self.ocr.predict(
                image_bgr,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                text_rec_score_thresh=score_threshold,
            )
        if not output:
            return []
        result = output[0]
        texts = list(result.get("rec_texts") or [])
        scores = _as_list(result.get("rec_scores"))
        boxes = _as_list(result.get("rec_boxes"))
        polys = _as_list(result.get("rec_polys"))
        lines = []
        for i, text in enumerate(texts):
            score = float(scores[i]) if i < len(scores) else 0.0
            box = boxes[i] if i < len(boxes) else []
            poly = polys[i] if i < len(polys) else []
            lines.append({
                "text": text,
                "score": round(score, 4),
                "box": [_coord(v) for v in box],
                "poly": [[_coord(x), _coord(y)] for x, y in poly],
            })
        return lines


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _coord(value) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else round(number, 1)
