"""取样代表色、Lab 色差、路径内部是否两种颜色反差很大。"""
import cv2
import numpy as np


def _to_hex(rgb) -> str:
    return "#" + "".join(f"{int(round(v)):02x}" for v in rgb)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, np.float32).reshape(-1, 1, 3) / 255
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _delta_e2000(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIEDE2000 between broadcastable Lab arrays (..., 3)."""
    l1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
    l2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    dhh = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))

    l_bar = (l1 + l2) / 2
    cp_bar = (c1p + c2p) / 2
    hsum = h1p + h2p
    h_bar = np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, hsum + 360, hsum - 360), hsum) / 2
    h_bar = np.where(c1p * c2p == 0, hsum, h_bar)
    t = (1 - 0.17 * np.cos(np.radians(h_bar - 30)) + 0.24 * np.cos(np.radians(2 * h_bar))
         + 0.32 * np.cos(np.radians(3 * h_bar + 6)) - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    sl = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    sc = 1 + 0.045 * cp_bar
    sh = 1 + 0.015 * cp_bar * t
    rt = (-2 * np.sqrt(cp_bar ** 7 / (cp_bar ** 7 + 25.0 ** 7))
          * np.sin(np.radians(60 * np.exp(-(((h_bar - 275) / 25) ** 2)))))
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhh / sh) ** 2 + rt * (dc / sc) * (dhh / sh))


def _dominant_color_stats(pixels: np.ndarray, bits: int = 4,
                         bin_min_frac: float = 0.03) -> tuple[np.ndarray, float, float]:
    """主色 + 这块是否「内部反差大」。

    返回 (dominant_rgb, dominant_ratio, peak_delta_e)：
    - dominant_rgb: 4bit 众数桶及其相邻桶的像素均值
    - dominant_ratio: 这些邻域像素占区域内像素的比例；越低说明没有稳定主色
    - peak_delta_e: 占比 >= bin_min_frac 的颜色桶两两 CIEDE2000 最大值；
      柱状图那种浅粉底+深紫柱会到 50+。桶太少（只有一种色）则为 0。
    """
    q = (pixels >> (8 - bits)).astype(np.int32)
    keys = (q[:, 0] << (2 * bits)) | (q[:, 1] << bits) | q[:, 2]
    counts = np.bincount(keys)
    best = int(counts.argmax())
    bq = np.array([best >> (2 * bits), (best >> bits) & ((1 << bits) - 1), best & ((1 << bits) - 1)])
    near = np.all(np.abs(q - bq) <= 1, axis=1)
    rgb = pixels[near].mean(axis=0)
    ratio = float(near.mean())
    # 忽略占比过小的桶（描边、噪点），避免一个高光像素把 peak_delta_e 撑爆
    min_count = max(1, int(np.ceil(bin_min_frac * len(pixels))))
    idx = np.flatnonzero(counts >= min_count)
    peak_de = 0.0
    if len(idx) >= 2:
        means = np.stack([pixels[keys == i].mean(axis=0) for i in idx])
        labs = _rgb_to_lab(means)
        dist = _delta_e2000(labs[:, None, :], labs[None, :, :])
        np.fill_diagonal(dist, 0)
        peak_de = float(dist.max())
    return rgb, ratio, peak_de


def _region_color_stats(crop: np.ndarray, mask: np.ndarray, inset: int,
                       bin_min_frac: float = 0.03) -> tuple[np.ndarray, float, float]:
    """轮廓内缩 inset 像素后再取主色，避免描边和邻色渗入。"""
    inner = mask
    if inset > 0:
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=inset).astype(bool)
        if eroded.any():
            inner = eroded
    return _dominant_color_stats(crop[inner], bin_min_frac=bin_min_frac)
