import math
import re

import cv2
import numpy as np

_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
_ARGC = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}
_SHIFT = 4


def _steps(*pts) -> int:
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return max(2, min(64, int(length / 2)))


def _cubic(p0, p1, p2, p3):
    n = _steps(p0, p1, p2, p3)
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def _quad(p0, p1, p2):
    n = _steps(p0, p1, p2)
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1 - t
        out.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    return out


def _arc(p0, rx, ry, phi_deg, large, sweep, p1):
    """Endpoint to center parameterization, SVG spec F.6.5."""
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0 or p0 == p1:
        return [p1]
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1p = cos_p * dx + sin_p * dy
    y1p = -sin_p * dx + cos_p * dy
    lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) * (-1 if bool(large) == bool(sweep) else 1)
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (p0[0] + p1[0]) / 2
    cy = sin_p * cxp + cos_p * cyp + (p0[1] + p1[1]) / 2

    def angle(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    t1 = angle(1, 0, ux, uy)
    dt = angle(ux, uy, vx, vy)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(4, min(64, int(abs(dt) * max(rx, ry) / 2)))
    out = []
    for i in range(1, n):
        t = t1 + dt * i / n
        out.append((cos_p * rx * math.cos(t) - sin_p * ry * math.sin(t) + cx,
                    sin_p * rx * math.cos(t) + cos_p * ry * math.sin(t) + cy))
    out.append(p1)
    return out


def parse_path(d: str) -> list[np.ndarray]:
    """Flatten an SVG path into closed polygons (float32, (N, 2)); curves are sampled about every 2px."""
    tokens = _TOKEN.findall(d or "")
    polys: list[list] = []
    cur: list = []
    x = y = sx = sy = 0.0
    cmd = None
    prev = ""
    ctrl = None
    i = 0

    def add(points):
        if not cur:
            cur.append((x, y))
        cur.extend(points)

    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
            if cmd in "Zz":
                if cur:
                    polys.append(cur)
                    cur = []
                x, y = sx, sy
                prev, ctrl = "Z", None
                continue
        elif cmd is None or cmd in "Zz":
            i += 1
            continue

        up = cmd.upper()
        argc = _ARGC[up]
        args = tokens[i:i + argc]
        if len(args) < argc or any(t.isalpha() for t in args):
            break
        a = [float(t) for t in args]
        i += argc
        ox, oy = (x, y) if cmd.islower() else (0.0, 0.0)

        if up == "M":
            if cur:
                polys.append(cur)
            x, y = ox + a[0], oy + a[1]
            sx, sy = x, y
            cur = [(x, y)]
            cmd = "l" if cmd == "m" else "L"
            prev, ctrl = "M", None
            continue
        if up == "L":
            nx, ny = ox + a[0], oy + a[1]
            add([(nx, ny)])
            ctrl = None
        elif up == "H":
            nx, ny = ox + a[0], y
            add([(nx, ny)])
            ctrl = None
        elif up == "V":
            nx, ny = x, oy + a[0]
            add([(nx, ny)])
            ctrl = None
        elif up in "CS":
            if up == "C":
                c1 = (ox + a[0], oy + a[1])
                c2, (nx, ny) = (ox + a[2], oy + a[3]), (ox + a[4], oy + a[5])
            else:
                c1 = (2 * x - ctrl[0], 2 * y - ctrl[1]) if prev in ("C", "S") and ctrl else (x, y)
                c2, (nx, ny) = (ox + a[0], oy + a[1]), (ox + a[2], oy + a[3])
            add(_cubic((x, y), c1, c2, (nx, ny)))
            ctrl = c2
        elif up in "QT":
            if up == "Q":
                c1, (nx, ny) = (ox + a[0], oy + a[1]), (ox + a[2], oy + a[3])
            else:
                c1 = (2 * x - ctrl[0], 2 * y - ctrl[1]) if prev in ("Q", "T") and ctrl else (x, y)
                nx, ny = ox + a[0], oy + a[1]
            add(_quad((x, y), c1, (nx, ny)))
            ctrl = c1
        else:
            nx, ny = ox + a[5], oy + a[6]
            add(_arc((x, y), a[0], a[1], a[2], a[3], a[4], (nx, ny)))
            ctrl = None
        x, y = nx, ny
        prev = up

    if cur:
        polys.append(cur)
    return [np.asarray(p, dtype=np.float32) for p in polys if len(p) >= 3]


def rasterize_path(d: str, width: int, height: int):
    """Fill a path (even-odd) clipped to the image; returns (x0, y0, mask uint8) of its bounding box, or None."""
    polys = parse_path(d)
    if not polys:
        return None
    pts = np.concatenate(polys)
    x0 = max(0, int(math.floor(pts[:, 0].min())))
    y0 = max(0, int(math.floor(pts[:, 1].min())))
    x1 = min(width, int(math.ceil(pts[:, 0].max())) + 1)
    y1 = min(height, int(math.ceil(pts[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    scaled = [np.round((p - (x0, y0)) * (1 << _SHIFT)).astype(np.int32) for p in polys]
    cv2.fillPoly(mask, scaled, 1, lineType=cv2.LINE_8, shift=_SHIFT)
    return x0, y0, mask
