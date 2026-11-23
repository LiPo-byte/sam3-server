import base64
import binascii
import io
import logging
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .model_handler import Sam3Segmenter
from .color import _region_color_stats, _to_hex, _rgb_to_lab, _delta_e2000
from .svg_path import mask_to_svg_path
from .svg_raster import rasterize_path
from .semantic_order import build_router as build_semantic_order_router
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("sam3")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_path = os.environ.get("SAM3_MODEL_PATH", "/models/sam3.1")
    state["model"] = Sam3Segmenter(model_path, device=os.environ.get("SAM3_DEVICE") or None)
    yield
    state.clear()


app = FastAPI(title="SAM3 Segment API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("SAM3_CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(build_semantic_order_router(lambda: state["model"]))


class SegmentRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image, data URL prefix is allowed")
    prompts: list[str] = Field(..., min_length=1, description="English text prompts, e.g. ['sofa', 'lamp']")
    threshold: float = Field(0.5, ge=0, le=1)
    mask_threshold: float = Field(0.5, ge=0, le=1)
    simplify: float = Field(1.0, ge=0, description="approxPolyDP epsilon in pixels, 0 disables simplification")
    min_area: float = Field(16, ge=0, description="drop contours smaller than this area in pixels")


class SegmentResult(BaseModel):
    prompt: str
    score: float
    box: list[float]
    path: str


class SegmentResponse(BaseModel):
    width: int
    height: int
    elapsed_ms: int
    results: list[SegmentResult]

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


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    prompts = [p.strip() for p in req.prompts if p.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="prompts is empty")

    image = _decode_image(req.image)
    start = time.time()

    results = []
    for prompt, score, box, mask in state["model"].segment(image, prompts, req.threshold, req.mask_threshold):
        path = mask_to_svg_path(mask, simplify=req.simplify, min_area=req.min_area)
        if not path:
            continue
        results.append(SegmentResult(
            prompt=prompt,
            score=round(score, 4),
            box=[round(v, 1) for v in box],
            path=path,
        ))

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info("segment %dx%d prompts=%s results=%d %dms",
                image.width, image.height, prompts, len(results), elapsed_ms)
    return SegmentResponse(width=image.width, height=image.height, elapsed_ms=elapsed_ms, results=results)

class SegmentOrderSection(BaseModel):
    """Any extra field (id, sort_index...) is passed through unchanged."""
    model_config = ConfigDict(extra="allow")
    path: str = ""
    label_pos: str = ""


class SegmentOrderGroup(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int | str | None = None
    delegate_color: str | None = None
    section_list: list[SegmentOrderSection] = Field(default_factory=list)


class SegmentOrderRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image, section paths use its pixel coordinates")
    group_list: list[SegmentOrderGroup] = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=0, le=1)
    mask_threshold: float = Field(0.5, ge=0, le=1)
    simplify: float = Field(1.0, ge=0, description="approxPolyDP epsilon in pixels, 0 disables simplification")
    min_area: float = Field(16, ge=0, description="drop contours smaller than this area in pixels")
    color_similarity: float = Field(
        1, ge=0, le=1,
        description="0-1, higher is stricter; merge a color group into the closest earlier group in the same stage when similarity >= this")
    small_group_count: int = Field(
        2, ge=1,
        description="hair/person color groups with at most this many paths may be absorbed into other/background")
    small_area_ratio: float = Field(
        0.01, ge=0, le=1,
        description="and whose total area / image area <= this; 0 disables absorbing into other/background")
    large_area_ratio: float = Field(
        0.05, ge=0, le=1,
        description="same-stage similar colors are not merged when both groups are >= this share of the image; 0 disables; identical hex still merge")
    impure_delta_e: float = Field(
        50, ge=0,
        description="内部最大 ΔE 达到此值（且主色占比够低）则视为不纯，不参与同阶段相近色合并；0 关闭")
    impure_dominant_ratio: float = Field(
        0.34, ge=0, le=1,
        description="主色邻域像素占比上限，与 impure_delta_e 同时满足才标不纯；约 1/3")
    pair_color_similarity: float = Field(
        0.8, ge=0, le=1,
        description="eyes/eyebrows 各自内部合并的相似度，低于 color_similarity 则左右眼、左右眉更容易并成一组")
    backdrop_near_ratio: float = Field(
        0.18, ge=0, le=1,
        description="other/background 组中心距 <= 此比例×图对角线则算同一片，只改组顺序不换路径；0 退回面积山谷")


class SegmentOrderGroupOut(BaseModel):
    prompt: str | None = None
    id: int | str | None = None
    delegate_color: str = ""
    section_list: list[dict]


class SegmentOrderResponse(BaseModel):
    width: int
    height: int
    elapsed_ms: int
    results: list[SegmentOrderGroupOut]


def point_in_mask(mask, x, y) -> bool:
    h, w = mask.shape
    ix, iy = int(x), int(y)
    return 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0


def section_xy(section: SegmentOrderSection):
    try:
        x, y = (float(v) for v in str(section.label_pos or "").split()[:2])
        return x, y
    except ValueError:
        return None


SEGMENT_ORDER_PROMPTS = ["eyes", "nose", "mouth", "ear", "eyebrows", "face", "hair", "person", "background"]
STAGE_OF_PROMPT = {
    "eyes": "face", "nose": "face", "mouth": "face", "ear": "face", "eyebrows": "face", "face": "face",
    "hair": "hair",
    "person": "person",
    "background": "background",
    "other": "other",
}
STAGE_ORDER = ["face", "hair", "person", "other", "background"]
VALLEY_STAGES = {"person"}
# other/background 先不走空间分片，只按 STAGE_ORDER 出（other 在 background 前）。
# 要恢复空间分片时把这两个阶段加回来：{"other", "background"}
SPATIAL_STAGES: set[str] = set()
_PATH_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
# 成对五官：左右眼 / 左右眉先按更松的阈值自己合并，不和对方部位绑在一起
PAIR_COLOR_PROMPTS = frozenset({"eyes", "eyebrows"})


def dump_section(image: np.ndarray, section: SegmentOrderSection,
                 impure_delta_e: float = 50, impure_dominant_ratio: float = 0.34) -> dict:
    """取样代表色，并标出路径内部是否两种颜色反差很大。

    多写三个字段给合并和下发对照：
    - dominant_ratio: 主色邻域占比，越低越不纯
    - peak_delta_e: 区域内主要颜色桶的最大 ΔE
    - color_impure: ΔE >= impure_delta_e 且占比 <= impure_dominant_ratio
      （例如柱状图浅粉底+深紫柱）。hex 全等仍会在 groups_by_color 里归一桶；
      不纯只阻止后面的相近色模糊合并。path 失败走 label_pos 时当作纯色。
    """
    item = section.model_dump()
    height, width = image.shape[:2]
    crop = rasterize_path(section.path or "", width, height)
    area = 0
    color = None
    dominant_ratio = 1.0
    peak_delta_e = 0.0
    if crop:
        x0, y0, mask = crop
        m = mask.astype(bool)
        area = int(m.sum())
        if area:
            rgb, dominant_ratio, peak_delta_e = _region_color_stats(
                image[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m, 2)
            color = _to_hex(rgb)
    if not color:
        xy = section_xy(section)
        if xy:
            ix, iy = int(xy[0]), int(xy[1])
            if 0 <= ix < width and 0 <= iy < height:
                color = _to_hex(image[iy, ix])
    if color:
        item["delegate_color"] = color
    item["area"] = area
    item["dominant_ratio"] = round(dominant_ratio, 3)
    item["peak_delta_e"] = round(peak_delta_e, 1)
    item["color_impure"] = bool(
        impure_delta_e > 0
        and peak_delta_e >= impure_delta_e
        and dominant_ratio <= impure_dominant_ratio)
    return item


def groups_by_color(sections: list[dict], prompt: str) -> list[SegmentOrderGroupOut]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        buckets[s.get("delegate_color") or ""].append(s)

    def sort_key(color: str):
        return (color == "", -sum(x.get("area") or 0 for x in buckets[color]))

    out = []
    for color in sorted(buckets, key=sort_key):
        members = buckets[color]
        members.sort(key=lambda s: s.get("area") or 0, reverse=True)
        out.append(SegmentOrderGroupOut(prompt=prompt, delegate_color=color, section_list=members))
    return out


def _hex_to_rgb(color: str) -> np.ndarray | None:
    c = (color or "").strip().lstrip("#")
    if len(c) != 6:
        return None
    try:
        return np.array([int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)], np.uint8)
    except ValueError:
        return None


def _color_distance(a: str, b: str) -> float:
    ra, rb = _hex_to_rgb(a), _hex_to_rgb(b)
    if ra is None or rb is None:
        return float("inf")
    lab = _rgb_to_lab(np.stack([ra, rb]))
    return float(_delta_e2000(lab[0], lab[1]))


def _color_similarity(a: str, b: str) -> float:
    d = _color_distance(a, b)
    if d == float("inf"):
        return 0.0
    return max(0.0, 1.0 - d / 100.0)


def _group_area(g: SegmentOrderGroupOut) -> int:
    return sum(s.get("area") or 0 for s in g.section_list)


def _group_impure(g: SegmentOrderGroupOut) -> bool:
    """组内任一区块不纯，整组就不参与相近色合并，避免脏色并进旁边的纯色号。"""
    return any(s.get("color_impure") for s in g.section_list)


def merge_similar_groups(groups: list[SegmentOrderGroupOut],
                         min_similarity: float,
                         image_area: int = 0,
                         large_area_ratio: float = 0,
                         block_impure: bool = True) -> list[SegmentOrderGroupOut]:
    """相近色合并，保留面积更大那组的颜色。

    - hex 全等已在 groups_by_color 合成一组，这里只处理「像但不一样」
    - 两组都 >= large_area_ratio 整图面积：相近也不并；ΔE=0 仍并
    - block_impure 时，任一组 color_impure 则相近也不并
    - 左右眼/左右眉内部合并会关掉 block_impure，否则眼白+眼珠会被不纯挡住
    """
    if not groups:
        return groups
    groups = list(groups)
    if min_similarity >= 1 or len(groups) < 2:
        groups.sort(key=_group_area, reverse=True)
        return groups

    n = len(groups)
    rgbs = np.zeros((n, 3), np.uint8)
    valid = np.zeros(n, bool)
    for i, g in enumerate(groups):
        rgb = _hex_to_rgb(g.delegate_color)
        if rgb is not None:
            rgbs[i] = rgb
            valid[i] = True
    labs = _rgb_to_lab(rgbs)
    dist = _delta_e2000(labs[:, None, :], labs[None, :, :])
    sim = np.maximum(0.0, 1.0 - dist / 100.0)
    sim[~valid, :] = -np.inf
    sim[:, ~valid] = -np.inf
    np.fill_diagonal(sim, -np.inf)

    area_limit = image_area * large_area_ratio if image_area > 0 and large_area_ratio > 0 else 0
    alive = np.ones(n, bool)
    while True:
        masked = np.where(alive[:, None] & alive[None, :], sim, -np.inf)
        i, j = divmod(int(masked.argmax()), n)
        if not np.isfinite(masked[i, j]) or masked[i, j] < min_similarity:
            break
        # 两个都是大色块：相近也不并，避免整块脸/衣服被合成一个号
        if (area_limit > 0 and masked[i, j] < 1
                and _group_area(groups[i]) >= area_limit
                and _group_area(groups[j]) >= area_limit):
            sim[i, j] = sim[j, i] = -np.inf
            continue
        # 内部两种色差很大的组：代表色只是众数，不能拿去跟旁边纯色合并
        if (block_impure and masked[i, j] < 1
                and (_group_impure(groups[i]) or _group_impure(groups[j]))):
            sim[i, j] = sim[j, i] = -np.inf
            continue
        if _group_area(groups[j]) > _group_area(groups[i]):
            i, j = j, i
        groups[i].section_list.extend(groups[j].section_list)
        alive[j] = False

    out = [g for k, g in enumerate(groups) if alive[k]]
    for g in out:
        g.section_list.sort(key=lambda s: s.get("area") or 0, reverse=True)
    out.sort(key=_group_area, reverse=True)
    return out


def arrange_area_valley(groups: list[SegmentOrderGroupOut]) -> list[SegmentOrderGroupOut]:
    """Largest at both ends, smallest in the middle."""
    if len(groups) <= 2:
        return groups
    ordered = sorted(groups, key=_group_area, reverse=True)
    left, right = [], []
    for i, g in enumerate(ordered):
        if i % 2 == 0:
            left.append(g)
        else:
            right.append(g)
    return left + right[::-1]


def _dict_xy(section: dict) -> tuple[float, float] | None:
    try:
        x, y = (float(v) for v in str(section.get("label_pos") or "").split()[:2])
        return x, y
    except ValueError:
        return None


def _path_bbox(path: str) -> tuple[float, float, float, float] | None:
    nums = [float(x) for x in _PATH_NUM.findall(path or "")]
    if len(nums) < 2:
        return None
    xs, ys = nums[0::2], nums[1::2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _group_xy(g: SegmentOrderGroupOut) -> tuple[float, float] | None:
    """组的空间位置：label_pos 面积加权；没有数字则用 path 包围盒中心。不改 section。"""
    pts, weights = [], []
    for s in g.section_list:
        xy = _dict_xy(s)
        if xy is None:
            bb = _path_bbox(s.get("path") or "")
            if bb:
                xy = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
        if xy is None:
            continue
        pts.append(xy)
        weights.append(max(s.get("area") or 0, 1))
    if not pts:
        return None
    w = np.asarray(weights, np.float64)
    p = np.asarray(pts, np.float64)
    c = (p * w[:, None]).sum(axis=0) / w.sum()
    return float(c[0]), float(c[1])


def _group_dist(a: SegmentOrderGroupOut, b: SegmentOrderGroupOut) -> float:
    pa, pb = _group_xy(a), _group_xy(b)
    if pa is None or pb is None:
        return float("inf")
    return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))


def cluster_groups_spatial(groups: list[SegmentOrderGroupOut],
                           near_px: float) -> list[list[SegmentOrderGroupOut]]:
    """从最大组开始，把中心距 <= near_px 的组收进同一片；只聚类，不合并 section。"""
    leftover = sorted(groups, key=_group_area, reverse=True)
    slices: list[list[SegmentOrderGroupOut]] = []
    while leftover:
        slice_g = [leftover.pop(0)]
        progressed = True
        while progressed:
            progressed = False
            still = []
            for g in leftover:
                if min(_group_dist(g, s) for s in slice_g) <= near_px:
                    slice_g.append(g)
                    progressed = True
                else:
                    still.append(g)
            leftover = still
        slices.append(slice_g)
    return slices


def order_slice_big_then_near(groups: list[SegmentOrderGroupOut]) -> list[SegmentOrderGroupOut]:
    """一片之内：先最大组，再反复接离已排组最近的组（贴着涂，色号可不同）。"""
    if len(groups) <= 1:
        return list(groups)
    leftover = sorted(groups, key=_group_area, reverse=True)
    ordered = [leftover.pop(0)]
    while leftover:
        leftover.sort(key=lambda g: (
            min(_group_dist(g, p) for p in ordered),
            -_group_area(g),
        ))
        ordered.append(leftover.pop(0))
    return ordered


def arrange_spatial_slices(groups: list[SegmentOrderGroupOut],
                           width: int, height: int,
                           near_ratio: float) -> list[SegmentOrderGroupOut]:
    """other/background：空间分片后，大片放两头、小片放中间；路径不换组。"""
    if len(groups) <= 2 or near_ratio <= 0 or width <= 0 or height <= 0:
        return arrange_area_valley(groups)
    near_px = float(np.hypot(width, height) * near_ratio)
    slices = cluster_groups_spatial(groups, near_px)
    ranked = sorted(slices, key=lambda sl: sum(_group_area(g) for g in sl), reverse=True)
    if len(ranked) > 2:
        left, right = [], []
        for i, sl in enumerate(ranked):
            (left if i % 2 == 0 else right).append(sl)
        ranked = left + right[::-1]
    out: list[SegmentOrderGroupOut] = []
    for sl in ranked:
        out.extend(order_slice_big_then_near(sl))
    return out


def reorder_valley_stages(results: list[SegmentOrderGroupOut],
                          width: int = 0, height: int = 0,
                          near_ratio: float = 0.18) -> list[SegmentOrderGroupOut]:
    if not results:
        return results
    head, rest = results[:1], results[1:]
    by_stage: dict[str, list[SegmentOrderGroupOut]] = defaultdict(list)
    for g in rest:
        by_stage[g.prompt or ""].append(g)
    out = list(head)
    for stage in STAGE_ORDER:
        groups = by_stage[stage]
        if stage in SPATIAL_STAGES:
            groups = arrange_spatial_slices(groups, width, height, near_ratio)
        elif stage in VALLEY_STAGES:
            groups = arrange_area_valley(groups)
        out.extend(groups)
    return out


def absorb_small_subject_groups(results: list[SegmentOrderGroupOut], image_area: int,
                                count_limit: int, area_ratio: float,
                                min_similarity: float) -> list[SegmentOrderGroupOut]:
    if area_ratio <= 0 or image_area <= 0:
        return results
    targets = [g for g in results if g.prompt in ("other", "background") and g.section_list]
    if not targets:
        return results
    kept: list[SegmentOrderGroupOut] = []
    for g in results:
        if g.prompt not in ("hair", "person"):
            kept.append(g)
            continue
        n = len(g.section_list)
        area = _group_area(g)
        if n == 0 or n > count_limit or area / image_area > area_ratio:
            kept.append(g)
            continue
        best = max(targets, key=lambda t: _color_similarity(t.delegate_color, g.delegate_color))
        if _color_similarity(best.delegate_color, g.delegate_color) >= min_similarity:
            best.section_list.extend(g.section_list)
            best.section_list.sort(key=lambda s: s.get("area") or 0, reverse=True)
            continue
        kept.append(g)
    return kept


@app.post("/segment_order", response_model=SegmentOrderResponse)
def segment_order(req: SegmentOrderRequest):
    image = _decode_image(req.image)
    start = time.time()

    image_np = np.asarray(image)
    sections = []
    for group in req.group_list:
        sections.extend(group.section_list)

    remaining = list(sections)
    grouped = defaultdict(list)

    for prompt, score, box, mask in state["model"].segment(
            image, SEGMENT_ORDER_PROMPTS, req.threshold, req.mask_threshold):
        still = []
        for section in remaining:
            xy = section_xy(section)
            if xy and point_in_mask(mask, *xy):
                grouped[prompt].append(dump_section(
                    image_np, section, req.impure_delta_e, req.impure_dominant_ratio))
            else:
                still.append(section)
        remaining = still

    other_sections = [
        dump_section(image_np, s, req.impure_delta_e, req.impure_dominant_ratio)
        for s in remaining
    ]

    results = [SegmentOrderGroupOut(
        id=req.group_list[0].id,
        delegate_color="",
        section_list=[],
    )]
    image_area = image.width * image.height
    for stage in STAGE_ORDER:
        # 眼/眉先按更松阈值自己合并（左右成对）；其余部位仍用全局 color_similarity
        pair_groups: list[SegmentOrderGroupOut] = []
        rest: list[dict] = []
        for p in SEGMENT_ORDER_PROMPTS:
            if STAGE_OF_PROMPT[p] != stage:
                continue
            if p in PAIR_COLOR_PROMPTS:
                pair_groups.extend(merge_similar_groups(
                    groups_by_color(grouped[p], stage), req.pair_color_similarity,
                    image_area, req.large_area_ratio, block_impure=False))
            else:
                rest.extend(grouped[p])
        if stage == "other":
            rest.extend(other_sections)
        rest_groups = merge_similar_groups(
            groups_by_color(rest, stage), req.color_similarity,
            image_area, req.large_area_ratio)
        # 再和脸的其他部分做严格合并；眼/眉不纯仍不会并进皮肤
        results.extend(merge_similar_groups(
            pair_groups + rest_groups, req.color_similarity,
            image_area, req.large_area_ratio))
    results = absorb_small_subject_groups(
        results, image_area,
        req.small_group_count, req.small_area_ratio, req.color_similarity)
    results = reorder_valley_stages(
        results, image.width, image.height, req.backdrop_near_ratio)

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info("segment_order %dx%d sections=%d groups=%d %dms",
                image.width, image.height, len(sections), len(results), elapsed_ms)
    return SegmentOrderResponse(
        width=image.width, height=image.height, elapsed_ms=elapsed_ms, results=results)