"""语义排序：头部（脸 → 头发 → 头部装饰）→ 身体（皮肤/其余装饰/穿戴从上到下）→ 其它 → 背景。

- 语义不同的区块即使同色也不进一组；身体（颈/肩/手臂…）先归一个桶，再按颜色弱合并
- 身体/穿戴/其它/背景合并时要求空间相邻，避免一组散在图的两头
- 阶段内就近接力排序，下一阶段从离上一阶段末组最近的组开始
- 其它/背景绕人物中心按扇区顺（逆）时针推进，同一扇区涂完再走下一个，不在角落留空缺

与 /segment_order 完全独立，只共用模型和取色、栅格化工具。
"""
import base64
import binascii
import io
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .color import _delta_e2000, _region_color_stats, _rgb_to_lab, _to_hex
from .svg_raster import rasterize_path

logger = logging.getLogger("sam3")

# 归类优先级：五官/耳朵 > 装饰 > 脸部皮肤 > 头发 > 穿戴 > 身体 > 背景（见 _assign_tiers）；
# 同一档内取列表里第一个覆盖率达标的 prompt，所以具体词放前面
DEFAULT_STAGE_PROMPTS = {
    "decor": ["hat", "headband", "glasses", "earring", "necklace", "jewelry"],
    "face": ["eyes", "eyebrows", "nose", "mouth", "ear", "face"],
    "hair": ["hair"],
    "apparel": ["shoe", "bag", "clothing"],
    "body": ["person"],
    "background": ["background"],
}
ASSIGN_ORDER = ["decor", "face", "hair", "apparel", "body", "background"]
# 排序：头部（脸 → 头发 → 头部装饰）→ 身体（皮肤/其余装饰/穿戴混排，水平条带从上往下）→ other → background
# 没识别到脸时，按细类判断哪些装饰属于头部
HEAD_DECOR_LABELS = frozenset({"hat", "headband", "glasses", "earring"})
PERSON_STAGES = frozenset({"face", "hair", "decor", "body", "apparel"})
# 这些阶段绕人物中心按扇区顺/逆时针推进，扇区内再就近接力
SWEEP_STAGES = ("other", "background")
# 脸部皮肤：优先级低于装饰，否则压在脸上的眼镜、耳环会被归进脸
FACE_SKIN_PROMPT = "face"
# 脸部分桶：眼、眉各自成桶（左右成对用更松的阈值），鼻/嘴/耳并入脸部肤色桶
FACE_BUCKET = {"eyes": "eyes", "eyebrows": "eyebrows", "nose": "face", "mouth": "face", "ear": "face", "face": "face"}
PAIR_BUCKETS = frozenset({"eyes", "eyebrows"})
# 这些阶段按细类分桶：帽子和耳环同色也不合并
PER_LABEL_STAGES = frozenset({"decor", "apparel"})
# 这些阶段合并要求空间相邻；脸/头发/装饰本身紧凑或天然成对（左右耳环），不加约束
SPATIAL_MERGE_STAGES = frozenset({"body", "apparel", "other", "background"})


class SemanticOrderSection(BaseModel):
    """Any extra field (id, sort_index...) is passed through unchanged."""
    model_config = ConfigDict(extra="allow")
    path: str = ""
    label_pos: str = ""


class SemanticOrderGroup(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int | str | None = None
    delegate_color: str | None = None
    section_list: list[SemanticOrderSection] = Field(default_factory=list)


class SemanticOrderRequest(BaseModel):
    image: str = Field(..., description="base64 encoded image, section paths use its pixel coordinates")
    group_list: list[SemanticOrderGroup] = Field(..., min_length=1)
    threshold: float = Field(0.5, ge=0, le=1)
    mask_threshold: float = Field(0.5, ge=0, le=1)
    coverage_ratio: float = Field(
        0.5, ge=0, le=1,
        description="区块落在某个 prompt mask 内的面积占比 >= 此值才算命中；路径栅格化失败时退回 label_pos 圆心判断")
    color_similarity: float = Field(
        0.9, ge=0, le=1,
        description="同一桶内相近色合并阈值，1 - ΔE2000/100；0.9 约等于 ΔE<=10；1 表示只合并 hex 全等")
    pair_color_similarity: float = Field(
        0.8, ge=0, le=1, description="眼/眉桶的合并阈值，更松，左右眼、左右眉更容易并成一组")
    merge_gap_ratio: float = Field(
        0.02, ge=0, le=1,
        description="身体/穿戴/其它/背景：两组包围盒最近距离 <= 此比例×图对角线才允许合并；0 关闭空间约束")
    order_tolerance_ratio: float = Field(
        0.01, ge=0, le=1,
        description="就近接力时，距离在最近者 + 此比例×图对角线以内的候选里挑面积最大的")
    large_area_ratio: float = Field(
        0.05, ge=0, le=1,
        description="两组都 >= 此比例整图面积时相近色不合并（hex 全等仍合并）；0 关闭")
    impure_delta_e: float = Field(
        50, ge=0, description="区块内部最大 ΔE 达到此值（且主色占比够低）视为不纯，不参与相近色合并；0 关闭")
    impure_dominant_ratio: float = Field(
        0.34, ge=0, le=1, description="主色邻域像素占比上限，与 impure_delta_e 同时满足才标不纯")
    decor_prompts: list[str] | None = Field(None, description="覆盖默认装饰 prompt，具体词放前面")
    apparel_prompts: list[str] | None = Field(None, description="覆盖默认穿戴 prompt，具体词放前面")
    background_prompts: list[str] | None = Field(None, description="覆盖默认背景 prompt，如 ['sky','wall','floor','background']")
    body_band_ratio: float = Field(
        0.12, ge=0, le=1,
        description="身体阶段水平条带高度 = 身体阶段纵向范围 × 此值；组按顶边落入条带，条带从上往下、条带内就近接力；0 关闭条带")
    sweep_sectors: int = Field(
        8, ge=0, le=72,
        description="other/background 绕人物中心切成几个扇区，按扇区顺序推进，同扇区涂完再走下一个；0 关闭，退回整阶段就近接力")
    sweep_clockwise: bool = Field(True, description="扇区推进方向，True 顺时针（屏幕坐标）")
    sweep_joint: bool = Field(
        False,
        description="False：other 先转完一圈，background 再转一圈；True：每个扇区内先 other 再 background，一圈转完两者")


class SemanticOrderGroupOut(BaseModel):
    prompt: str | None = None
    label: str | None = None
    id: int | str | None = None
    delegate_color: str = ""
    section_list: list[dict]


class SemanticOrderResponse(BaseModel):
    width: int
    height: int
    elapsed_ms: int
    results: list[SemanticOrderGroupOut]


@dataclass
class _Sec:
    item: dict
    color: str
    area: int
    impure: bool
    box: np.ndarray
    xy: tuple[float, float] | None
    crop: tuple[int, int, np.ndarray] | None


@dataclass
class _Group:
    stage: str
    label: str
    color: str
    members: list[_Sec] = field(default_factory=list)
    sector: int = -1

    @property
    def area(self) -> int:
        return sum(s.area for s in self.members)


def _decode_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")


def _parse_label_pos(label_pos: str) -> tuple[tuple[float, float] | None, float]:
    parts = str(label_pos or "").split()
    try:
        x, y = float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return None, 0.0
    try:
        r = float(parts[2]) if len(parts) > 2 else 0.0
    except ValueError:
        r = 0.0
    return (x, y), max(r, 0.0)


def _prepare_section(image: np.ndarray, section: SemanticOrderSection,
                     impure_delta_e: float, impure_dominant_ratio: float) -> _Sec:
    item = section.model_dump()
    height, width = image.shape[:2]
    xy, r = _parse_label_pos(section.label_pos)
    raster = rasterize_path(section.path or "", width, height)
    crop = None
    area = 0
    color = ""
    dominant_ratio, peak_delta_e = 1.0, 0.0
    box = None
    if raster:
        x0, y0, mask = raster
        m = mask.astype(bool)
        area = int(m.sum())
        if area:
            crop = (x0, y0, m)
            ys, xs = np.nonzero(m)
            box = np.array([x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1], np.float32)
            rgb, dominant_ratio, peak_delta_e = _region_color_stats(
                image[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m, 2)
            color = _to_hex(rgb)
    if not color and xy:
        ix, iy = int(xy[0]), int(xy[1])
        if 0 <= ix < width and 0 <= iy < height:
            color = _to_hex(image[iy, ix])
    if box is None:
        if xy:
            box = np.array([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], np.float32)
        else:
            box = np.array([0, 0, width, height], np.float32)
    impure = bool(impure_delta_e > 0
                  and peak_delta_e >= impure_delta_e
                  and dominant_ratio <= impure_dominant_ratio)
    if color:
        item["delegate_color"] = color
    item["area"] = area
    item["dominant_ratio"] = round(dominant_ratio, 3)
    item["peak_delta_e"] = round(peak_delta_e, 1)
    item["color_impure"] = impure
    return _Sec(item=item, color=color, area=area, impure=impure, box=box, xy=xy, crop=crop)


def _coverage(sec: _Sec, mask: np.ndarray) -> float:
    if sec.crop:
        x0, y0, m = sec.crop
        sub = mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        return float(sub[m].mean())
    if sec.xy:
        h, w = mask.shape
        ix, iy = int(sec.xy[0]), int(sec.xy[1])
        return 1.0 if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] else 0.0
    return 0.0


def _assign_tiers(stage_prompts: dict[str, list[str]]) -> list[tuple[str, str]]:
    """按归类优先级展开成 (stage, prompt) 列表。"""
    face = stage_prompts["face"]
    tiers = [("face", p) for p in face if p != FACE_SKIN_PROMPT]
    tiers += [("decor", p) for p in stage_prompts["decor"]]
    tiers += [("face", p) for p in face if p == FACE_SKIN_PROMPT]
    for stage in ASSIGN_ORDER:
        if stage not in ("face", "decor"):
            tiers += [(stage, p) for p in stage_prompts[stage]]
    return tiers


def _assign(sec: _Sec, unions: dict[str, np.ndarray],
            tiers: list[tuple[str, str]], coverage_ratio: float) -> tuple[str, str, float]:
    """返回 (stage, 细类 prompt, 覆盖率)；都不命中归 other。"""
    for stage, prompt in tiers:
        mask = unions.get(prompt)
        if mask is None:
            continue
        cov = _coverage(sec, mask)
        if cov > 0 and cov >= coverage_ratio:
            return stage, prompt, cov
    return "other", "other", 0.0


def _bucket_of(stage: str, prompt: str) -> str:
    if stage == "face":
        return FACE_BUCKET.get(prompt, "face")
    if stage in PER_LABEL_STAGES:
        return prompt
    return stage


def _box_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组包围盒 (N,4)/(M,4) 两两最近距离，相交或相贴为 0。"""
    dx = np.maximum(0, np.maximum(a[:, None, 0] - b[None, :, 2], b[None, :, 0] - a[:, None, 2]))
    dy = np.maximum(0, np.maximum(a[:, None, 1] - b[None, :, 3], b[None, :, 1] - a[:, None, 3]))
    return np.hypot(dx, dy).astype(np.float32)


def _group_gap_matrix(groups: list[_Group]) -> np.ndarray:
    """组与组之间：两组区块包围盒的最近距离。"""
    boxes = np.stack([s.box for g in groups for s in g.members])
    starts = np.cumsum([0] + [len(g.members) for g in groups[:-1]])
    sec_gap = _box_gap(boxes, boxes)
    return np.minimum.reduceat(np.minimum.reduceat(sec_gap, starts, axis=0), starts, axis=1)


def _connected(members: list[_Sec], gap_px: float) -> list[list[_Sec]]:
    boxes = np.stack([s.box for s in members])
    near = _box_gap(boxes, boxes) <= gap_px
    parent = list(range(len(members)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(near, 1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    comps: dict[int, list[_Sec]] = defaultdict(list)
    for i, s in enumerate(members):
        comps[find(i)].append(s)
    return list(comps.values())


def _hex_to_lab(color: str) -> np.ndarray | None:
    c = (color or "").strip().lstrip("#")
    if len(c) != 6:
        return None
    try:
        rgb = np.array([int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)], np.uint8)
    except ValueError:
        return None
    return _rgb_to_lab(rgb[None, :])[0]


def _merge_bucket(secs: list[_Sec], stage: str, label: str, *,
                  min_similarity: float, gap_px: float,
                  large_area_px: float, block_impure: bool, sector: int = -1) -> list[_Group]:
    """hex 全等先成组（有空间约束时按相邻拆成连通块），再贪心合并相近色，保留面积大那组的颜色。"""
    by_hex: dict[str, list[_Sec]] = defaultdict(list)
    for s in secs:
        by_hex[s.color].append(s)
    groups: list[_Group] = []
    for color, members in by_hex.items():
        comps = _connected(members, gap_px) if gap_px > 0 and len(members) > 1 else [members]
        groups.extend(_Group(stage, label, color, comp, sector) for comp in comps)

    n = len(groups)
    if min_similarity >= 1 or n < 2:
        return groups

    labs = np.zeros((n, 3), np.float32)
    valid = np.zeros(n, bool)
    for i, g in enumerate(groups):
        lab = _hex_to_lab(g.color)
        if lab is not None:
            labs[i] = lab
            valid[i] = True
    sim = np.maximum(0.0, 1.0 - _delta_e2000(labs[:, None, :], labs[None, :, :]) / 100.0)
    sim[~valid, :] = -np.inf
    sim[:, ~valid] = -np.inf
    np.fill_diagonal(sim, -np.inf)
    gap = _group_gap_matrix(groups) if gap_px > 0 else None
    area = np.array([g.area for g in groups], np.float64)
    impure = np.array([any(s.impure for s in g.members) for g in groups])
    alive = np.ones(n, bool)

    while True:
        ok = alive[:, None] & alive[None, :]
        if gap is not None:
            ok &= gap <= gap_px
        masked = np.where(ok, sim, -np.inf)
        i, j = divmod(int(masked.argmax()), n)
        s_ij = masked[i, j]
        if not np.isfinite(s_ij) or s_ij < min_similarity:
            break
        # 面积只增不减、不纯只会传染，这两类阻断可以永久生效
        if s_ij < 1 and large_area_px > 0 and area[i] >= large_area_px and area[j] >= large_area_px:
            sim[i, j] = sim[j, i] = -np.inf
            continue
        if s_ij < 1 and block_impure and (impure[i] or impure[j]):
            sim[i, j] = sim[j, i] = -np.inf
            continue
        if area[j] > area[i]:
            i, j = j, i
        groups[i].members.extend(groups[j].members)
        area[i] += area[j]
        impure[i] |= impure[j]
        alive[j] = False
        if gap is not None:
            gap[i, :] = gap[:, i] = np.minimum(gap[i, :], gap[j, :])
            gap[i, i] = 0
    return [g for k, g in enumerate(groups) if alive[k]]


def _pick(dist: np.ndarray, remaining: list[int], tie: np.ndarray, tol_px: float) -> int:
    best = min(dist[k] for k in remaining)
    cands = [k for k in remaining if dist[k] <= best + tol_px]
    return max(cands, key=lambda k: tie[k])


def _gap_to_tail(tail: _Group, groups: list[_Group]) -> np.ndarray:
    tail_boxes = np.stack([s.box for s in tail.members])
    boxes = np.stack([s.box for g in groups for s in g.members])
    starts = np.cumsum([0] + [len(g.members) for g in groups[:-1]])
    return np.minimum.reduceat(_box_gap(tail_boxes, boxes).min(axis=0), starts)


def _order_stage(groups: list[_Group], tail: _Group | None, tol_px: float,
                 start: int | None = None, prefer_top: bool = False) -> list[_Group]:
    """就近接力：起点是 start，或离上一阶段末组最近的组（第一阶段用最大组），之后每步接离当前组最近的。

    距离在容差内的候选默认取面积最大的；prefer_top 时取顶边最靠上的。
    """
    if len(groups) <= 1:
        return list(groups)
    areas = np.array([g.area for g in groups], np.float64)
    tie = -np.array([_group_top(g) for g in groups]) if prefer_top else areas
    gap = _group_gap_matrix(groups)
    remaining = list(range(len(groups)))
    if start is not None:
        cur = start
    elif tail is None:
        cur = int(areas.argmax())
    else:
        cur = _pick(_gap_to_tail(tail, groups), remaining, tie, tol_px)
    ordered = [cur]
    remaining.remove(cur)
    while remaining:
        cur = _pick(gap[cur], remaining, tie, tol_px)
        ordered.append(cur)
        remaining.remove(cur)
    return [groups[k] for k in ordered]


def _group_top(g: _Group) -> float:
    return float(min(s.box[1] for s in g.members))


def _split_head_decor(decor: list[_Group], face: list[_Group]) -> tuple[list[_Group], list[_Group]]:
    """顶边高于脸部最下沿的装饰算头部（帽子、眼镜、耳环），其余（项链等）跟身体一起排。"""
    if face:
        face_bottom = max(float(s.box[3]) for g in face for s in g.members)
        is_head = [_group_top(g) < face_bottom for g in decor]
    else:
        is_head = [g.label in HEAD_DECOR_LABELS for g in decor]
    return ([g for g, h in zip(decor, is_head) if h],
            [g for g, h in zip(decor, is_head) if not h])


def _order_bands(groups: list[_Group], tail: _Group | None, tol_px: float, band_ratio: float) -> list[_Group]:
    """按组顶边切成水平条带，条带从上往下；条带内从最靠上的组起步（并列时取离 tail 最近的），再就近接力。"""
    if not groups:
        return []
    tops = np.array([_group_top(g) for g in groups])
    bottom = max(float(s.box[3]) for g in groups for s in g.members)
    band_px = (bottom - tops.min()) * band_ratio
    if band_px <= 0:
        return _order_stage(groups, tail, tol_px)
    band_of = ((tops - tops.min()) // band_px).astype(int)
    out: list[_Group] = []
    for band in sorted(set(band_of.tolist())):
        members = [g for g, b in zip(groups, band_of) if b == band]
        member_tops = np.array([_group_top(g) for g in members])
        topmost = [k for k in range(len(members)) if member_tops[k] <= member_tops.min() + tol_px]
        if tail is not None and len(topmost) > 1:
            to_tail = _gap_to_tail(tail, members)
            start = min(topmost, key=lambda k: to_tail[k])
        else:
            start = topmost[0]
        ordered = _order_stage(members, tail, tol_px, start=start, prefer_top=True)
        out.extend(ordered)
        if ordered:
            tail = ordered[-1]
    return out


def _sec_center(sec: _Sec) -> tuple[float, float]:
    if sec.xy:
        return sec.xy
    b = sec.box
    return float(b[0] + b[2]) / 2, float(b[1] + b[3]) / 2


def _subject_center(secs: list[_Sec], width: int, height: int) -> tuple[float, float]:
    """人物各阶段区块按面积加权的中心；没有人物退回图片中心。"""
    if not secs:
        return width / 2, height / 2
    pts = np.array([_sec_center(s) for s in secs], np.float64)
    w = np.array([max(s.area, 1) for s in secs], np.float64)
    c = (pts * w[:, None]).sum(axis=0) / w.sum()
    return float(c[0]), float(c[1])


def _sector_of(sec: _Sec, center: tuple[float, float], n: int) -> int:
    # 屏幕坐标 y 向下，atan2 角度增大即视觉上的顺时针
    x, y = _sec_center(sec)
    angle = math.atan2(y - center[1], x - center[0]) % (2 * math.pi)
    return int(angle / (2 * math.pi / n)) % n


def _order_sweep(stage_groups: list[list[_Group]], tail: _Group | None, tol_px: float,
                 n_sectors: int, clockwise: bool) -> list[_Group]:
    """从离 tail 最近那组所在扇区开始，按方向逐个扇区推进；扇区内依 stage_groups 的阶段顺序就近接力。"""
    everything = [g for groups in stage_groups for g in groups]
    if not everything:
        return []
    if tail is None:
        start = max(everything, key=lambda g: g.area).sector
    else:
        start = everything[int(_gap_to_tail(tail, everything).argmin())].sector
    step = 1 if clockwise else -1
    out: list[_Group] = []
    for k in range(n_sectors):
        sector = (start + step * k) % n_sectors
        for groups in stage_groups:
            ordered = _order_stage([g for g in groups if g.sector == sector], tail, tol_px)
            out.extend(ordered)
            if ordered:
                tail = ordered[-1]
    return out


def build_router(get_model: Callable[[], object]) -> APIRouter:
    router = APIRouter()

    @router.post("/segment_semantic_order", response_model=SemanticOrderResponse)
    def segment_semantic_order(req: SemanticOrderRequest):
        image = _decode_image(req.image)
        start = time.time()
        image_np = np.asarray(image)
        width, height = image.width, image.height
        diag = math.hypot(width, height)

        stage_prompts = dict(DEFAULT_STAGE_PROMPTS)
        if req.decor_prompts:
            stage_prompts["decor"] = [p.strip() for p in req.decor_prompts if p.strip()]
        if req.apparel_prompts:
            stage_prompts["apparel"] = [p.strip() for p in req.apparel_prompts if p.strip()]
        if req.background_prompts:
            stage_prompts["background"] = [p.strip() for p in req.background_prompts if p.strip()]
        tiers = _assign_tiers(stage_prompts)
        prompts = list(dict.fromkeys(p for _, p in tiers))

        unions: dict[str, np.ndarray] = {}
        for prompt, _score, _box, mask in get_model().segment(
                image, prompts, req.threshold, req.mask_threshold):
            m = mask > 0
            unions[prompt] = unions[prompt] | m if prompt in unions else m

        assigned: list[tuple[_Sec, str, str]] = []
        for group in req.group_list:
            for section in group.section_list:
                sec = _prepare_section(image_np, section, req.impure_delta_e, req.impure_dominant_ratio)
                stage, prompt, cov = _assign(sec, unions, tiers, req.coverage_ratio)
                sec.item["stage"] = stage
                sec.item["semantic"] = prompt
                sec.item["coverage"] = round(cov, 3)
                assigned.append((sec, stage, prompt))
        n_sections = len(assigned)

        sweep = req.sweep_sectors > 0
        center = _subject_center([s for s, st, _ in assigned if st in PERSON_STAGES], width, height)
        # 扇区进桶：other/background 不跨扇区合并，保证按扇区推进时同一组不会横跨两边
        buckets: dict[tuple[str, str, int], list[_Sec]] = defaultdict(list)
        for sec, stage, prompt in assigned:
            sector = _sector_of(sec, center, req.sweep_sectors) if sweep and stage in SWEEP_STAGES else -1
            if sector >= 0:
                sec.item["sector"] = sector
            buckets[(stage, _bucket_of(stage, prompt), sector)].append(sec)

        large_area_px = width * height * req.large_area_ratio
        by_stage: dict[str, list[_Group]] = defaultdict(list)
        for (stage, label, sector), secs in buckets.items():
            pair = label in PAIR_BUCKETS
            by_stage[stage].extend(_merge_bucket(
                secs, stage, label,
                min_similarity=req.pair_color_similarity if pair else req.color_similarity,
                gap_px=req.merge_gap_ratio * diag if stage in SPATIAL_MERGE_STAGES else 0,
                large_area_px=large_area_px,
                block_impure=not pair,
                sector=sector))

        tail = None
        tol_px = req.order_tolerance_ratio * diag
        ordered_all: list[_Group] = []
        head_decor, body_decor = _split_head_decor(by_stage["decor"], by_stage["face"])
        for groups in (by_stage["face"], by_stage["hair"], head_decor):
            ordered = _order_stage(groups, tail, tol_px)
            ordered_all.extend(ordered)
            if ordered:
                tail = ordered[-1]
        ordered = _order_bands(by_stage["body"] + body_decor + by_stage["apparel"],
                               tail, tol_px, req.body_band_ratio)
        ordered_all.extend(ordered)
        if ordered:
            tail = ordered[-1]
        if not sweep:
            for stage in SWEEP_STAGES:
                ordered = _order_stage(by_stage[stage], tail, tol_px)
                ordered_all.extend(ordered)
                if ordered:
                    tail = ordered[-1]
        else:
            passes = ([[by_stage[s] for s in SWEEP_STAGES]] if req.sweep_joint
                      else [[by_stage[s]] for s in SWEEP_STAGES])
            for stage_groups in passes:
                ordered = _order_sweep(stage_groups, tail, tol_px, req.sweep_sectors, req.sweep_clockwise)
                ordered_all.extend(ordered)
                if ordered:
                    tail = ordered[-1]

        results = [SemanticOrderGroupOut(id=req.group_list[0].id, delegate_color="", section_list=[])]
        for g in ordered_all:
            g.members.sort(key=lambda s: s.area, reverse=True)
            results.append(SemanticOrderGroupOut(
                prompt=g.stage, label=g.label, delegate_color=g.color,
                section_list=[s.item for s in g.members]))

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info("segment_semantic_order %dx%d sections=%d groups=%d %dms",
                    width, height, n_sections, len(results), elapsed_ms)
        return SemanticOrderResponse(width=width, height=height, elapsed_ms=elapsed_ms, results=results)

    return router
