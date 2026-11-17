import cv2
import numpy as np


def _contour_to_path(contour: np.ndarray) -> str:
    points = contour.reshape(-1, 2)
    head = f"M{points[0][0]},{points[0][1]}"
    tail = " ".join(f"{x},{y}" for x, y in points[1:])
    return f"{head}L{tail}Z"


def mask_to_svg_path(mask: np.ndarray, simplify: float = 1.0, min_area: float = 16) -> str:
    """Convert a binary mask (H, W) to one SVG path; holes are included, render with fill-rule="evenodd"."""
    mask = (mask > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return ""

    hierarchy = hierarchy[0]
    kept_outer = set()
    parts = []

    # RETR_CCOMP: top-level contours (parent == -1) are outer borders, their children are holes
    order = sorted(range(len(contours)), key=lambda i: hierarchy[i][3] != -1)
    for i in order:
        parent = hierarchy[i][3]
        if parent != -1 and parent not in kept_outer:
            continue
        if cv2.contourArea(contours[i]) < min_area:
            continue

        contour = cv2.approxPolyDP(contours[i], simplify, True) if simplify > 0 else contours[i]
        if len(contour) < 3:
            continue

        if parent == -1:
            kept_outer.add(i)
        parts.append(_contour_to_path(contour))

    return " ".join(parts)
