"""NYUv2 相对深度点对（E1 的 T 文本任务；DEV-0012 任务书 §3）。由 GT 构造，固定种子，逐字节可复现。

约束（src/data/constants.py，默认值以 E1 预注册为准）：
- 每张图 NYU_PAIRS_PER_IMAGE 对；两点都在 Eigen crop（src/constants.NYU_EIGEN_CROP）内且深度有效
  （rawDepths 落在 (NYU_DEPTH_MIN, NYU_DEPTH_MAX) 开区间）；
- 像素距离 ≥ 短边 × 10%；深度相对差 ≥ 15%（max ≥ 1.15 × min）；同一张图里的点不重复使用；
- 深度边缘保护（遮挡边界，"相邻像素跳变"判据）：以点为中心 (2w+1)² 窗口（w = 标记环外半径）内深度全有效，
  且窗口内任意 4 邻接的两个像素深度 max/min ≤ 1.05（审查第 2 轮裁定；平滑斜面不剔）；
- 两个标签都能放下（不出图、不压另一个点的环与标签）——这是渲染的硬条件，不改变深度约束；
- "A 更近"与"B 更近"各半：每个划分内按哈希排序把点对分成两半，前 floor(n/2) 对 A 更近，其余 B 更近。

随机性：第 t 次尝试的两个点 = 候选像素表里第 sha256(salt|nyu_pairs|idx|t|a) mod n 与 …|b 个，与 numpy 版本无关。
坐标：点是整数像素 (x, y) = (列, 行)，标记以该像素为中心；深度取该像素的值。

渲染：两个环形标记与两个字母合成一个 mask，交给 src/executors/overlay.render_overlay（执行器白名单里的叠加渲染）。
对调标记（DEV-0036，D72-2；E1 事后读数 letter_swap 用）：render_pair(…, swap_labels=True) 只把两个标签位置上的字形互换，位置与环
不变；默认 False 与原来逐位相同（点对配置记录的 render_sha256 照旧）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.constants import NYU_DEPTH_MAX, NYU_DEPTH_MIN, NYU_EIGEN_CROP
from src.data.constants import (
    NYU_LABEL_GAP,
    NYU_LABEL_GLYPHS,
    NYU_LABEL_SCALE,
    NYU_MARKER_COLOR_INDEX,
    NYU_MARKER_INNER_RADIUS,
    NYU_MARKER_OUTER_RADIUS,
    NYU_PAIR_DEPTH_KEY,
    NYU_PAIR_LOCAL_WINDOW,
    NYU_PAIR_MAX_ATTEMPTS,
    NYU_PAIR_MIN_DIST_DEN,
    NYU_PAIR_MIN_DIST_NUM,
    NYU_PAIR_MIN_REL_DEPTH_DIFF,
    NYU_PAIR_NEIGHBOR_MAX_REL,
    NYU_PAIRS_PER_IMAGE,
    SUBSET_SALT,
)
from src.data.subsets import hash_rank
from src.executors.overlay import render_overlay

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRS_PATH = REPO_ROOT / "configs" / "e1_nyu_pairs.json"
PAIRS_VERSION = 1
LETTERS = ("A", "B")


# ---------------------------------------------------------------------------
# 点对生成
# ---------------------------------------------------------------------------


def _u64(*parts: Any) -> int:
    s = "|".join(str(p) for p in (SUBSET_SALT, "nyu_pairs", *parts))
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def valid_mask(depth: np.ndarray) -> np.ndarray:
    """Eigen crop 内、深度在 (NYU_DEPTH_MIN, NYU_DEPTH_MAX) 开区间的像素。"""
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    d = np.asarray(depth, dtype=np.float64)
    m = np.zeros(d.shape, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m & (d > NYU_DEPTH_MIN) & (d < NYU_DEPTH_MAX)


def _box_any(mask: np.ndarray, r_lo: int, r_hi: int, c_lo: int, c_hi: int) -> np.ndarray:
    """对每个中心 (y, x)：mask[y+r_lo : y+r_hi+1, x+c_lo : x+c_hi+1] 里有没有 True（图外不计）。用积分图，结果精确。"""
    H, W = mask.shape
    S = np.zeros((H + 1, W + 1), dtype=np.int64)
    S[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    ys, xs = np.arange(H), np.arange(W)
    r0, r1 = np.clip(ys + r_lo, 0, H), np.clip(ys + r_hi + 1, 0, H)
    c0, c1 = np.clip(xs + c_lo, 0, W), np.clip(xs + c_hi + 1, 0, W)
    total = S[np.ix_(r1, c1)] - S[np.ix_(r0, c1)] - S[np.ix_(r1, c0)] + S[np.ix_(r0, c0)]
    return total > 0


def neighbor_jumps(depth: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """相邻像素跳变：jh[r, c] 表示 (r, c)–(r, c+1)，jv[r, c] 表示 (r, c)–(r+1, c)，两点都有效且 max/min > 1 + NYU_PAIR_NEIGHBOR_MAX_REL。
    形状都补成 (H, W)（最后一列 / 最后一行为 False）。"""
    d = np.asarray(depth, dtype=np.float64)
    lim = 1.0 + NYU_PAIR_NEIGHBOR_MAX_REL
    jh = np.zeros(d.shape, dtype=bool)
    jv = np.zeros(d.shape, dtype=bool)
    a, b = d[:, :-1], d[:, 1:]
    jh[:, :-1] = valid[:, :-1] & valid[:, 1:] & (np.maximum(a, b) > lim * np.minimum(a, b))
    a, b = d[:-1, :], d[1:, :]
    jv[:-1, :] = valid[:-1, :] & valid[1:, :] & (np.maximum(a, b) > lim * np.minimum(a, b))
    return jh, jv


def edge_guard_masks(depth: np.ndarray, valid: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (窗口内全有效, 窗口内没有相邻跳变)，窗口为以像素为中心的 (2w+1)²。窗口不出 Eigen crop（valid 已含 crop）。"""
    from scipy import ndimage

    all_valid = ndimage.minimum_filter(valid.astype(np.uint8), size=2 * w + 1, mode="constant", cval=0).astype(bool)
    jh, jv = neighbor_jumps(depth, valid)
    # 窗口 [y-w, y+w] × [x-w, x+w] 内的水平邻接对：起点列 x-w .. x+w-1；竖直邻接对：起点行 y-w .. y+w-1
    no_jump = ~(_box_any(jh, -w, w, -w, w - 1) | _box_any(jv, -w, w - 1, -w, w))
    return all_valid, no_jump


def _local_ok(depth: np.ndarray, valid: np.ndarray, flat: np.ndarray, w: int) -> np.ndarray:
    """深度边缘保护（相邻像素跳变判据）：(2w+1)² 窗口内全有效，且窗口内任意 4 邻接的两点 max/min ≤ 1 + NYU_PAIR_NEIGHBOR_MAX_REL。"""
    all_valid, no_jump = edge_guard_masks(depth, valid, w)
    return flat[(all_valid & no_jump).ravel()[flat]]


def far_enough(p: tuple[int, int], q: tuple[int, int], short_side: int) -> bool:
    """像素距离 ≥ 短边 × NUM/DEN，整数比较：DEN²·d² ≥ NUM²·短边²。"""
    d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
    return NYU_PAIR_MIN_DIST_DEN**2 * d2 >= NYU_PAIR_MIN_DIST_NUM**2 * short_side**2


def depth_distinct(da: float, db: float) -> bool:
    return max(da, db) >= (1.0 + NYU_PAIR_MIN_REL_DEPTH_DIFF) * min(da, db)


def pairs_for_image(idx: int, depth: np.ndarray, n: int = NYU_PAIRS_PER_IMAGE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """一张图（labeled 集 1-based 下标 idx，深度 (H, W) 米）的点对。返回 (点对, 记账)；点对里 p、q 尚未分 A/B。"""
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape
    short = min(H, W)
    valid = valid_mask(depth)
    cand = np.flatnonzero(valid.ravel())
    n_valid = int(len(cand))
    if NYU_PAIR_LOCAL_WINDOW > 0 and len(cand):
        cand = _local_ok(depth, valid, cand, NYU_PAIR_LOCAL_WINDOW)
    pairs: list[dict[str, Any]] = []
    used: set[int] = set()
    t = 0
    rej = {"same_or_used": 0, "too_close": 0, "depth_ambiguous": 0, "label_layout": 0}
    while len(pairs) < n and t < NYU_PAIR_MAX_ATTEMPTS and len(cand) >= 2:
        a = int(cand[_u64(idx, t, "a") % len(cand)])
        b = int(cand[_u64(idx, t, "b") % len(cand)])
        t += 1
        if a == b or a in used or b in used:
            rej["same_or_used"] += 1
            continue
        p = (a % W, a // W)
        q = (b % W, b // W)
        if not far_enough(p, q, short):
            rej["too_close"] += 1
            continue
        dp, dq = float(depth[p[1], p[0]]), float(depth[q[1], q[0]])
        if not depth_distinct(dp, dq):
            rej["depth_ambiguous"] += 1
            continue
        if label_layout(p, q, H, W) is None or label_layout(q, p, H, W) is None:
            rej["label_layout"] += 1
            continue
        pairs.append({"idx": idx, "k": len(pairs), "p": list(p), "q": list(q), "depth_p": dp, "depth_q": dq})
        used.update((a, b))
    info = {"idx": idx, "attempts": t, "n_pairs": len(pairs), "shortfall": n - len(pairs), "n_valid": n_valid, "n_candidates": int(len(cand)), "rejected": rej}
    return pairs, info


def assign_ab(pairs: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """在一个划分内按哈希排序分两半：前 floor(n/2) 对"A 更近"，其余"B 更近"。返回带 A、B、closer 的点对（按 id 排序）。"""
    by_id = {pair_id(p["idx"], p["k"]): p for p in pairs}
    ranked = hash_rank(by_id, purpose=f"nyu_pairs_ab:{split}")
    half = len(ranked) // 2
    out = []
    for r, pid in enumerate(ranked):
        p = by_id[pid]
        near_first = p["depth_p"] < p["depth_q"]
        near, far = (p["p"], p["q"]) if near_first else (p["q"], p["p"])
        dnear, dfar = (p["depth_p"], p["depth_q"]) if near_first else (p["depth_q"], p["depth_p"])
        closer = "A" if r < half else "B"
        A, B, dA, dB = (near, far, dnear, dfar) if closer == "A" else (far, near, dfar, dnear)
        out.append({"id": pid, "split": split, "idx": p["idx"], "A": A, "B": B, "depth_A": dA, "depth_B": dB, "closer": closer})
    return sorted(out, key=lambda r: r["id"])


def pair_id(idx: int, k: int) -> str:
    return f"nyu{idx:04d}_{k}"


def build_pairs(images: Iterable[tuple[int, str, np.ndarray]], render: Mapping[int, np.ndarray] | None = None) -> dict[str, Any]:
    """images：[(1-based 下标, 划分名, 深度)]。render：{下标: RGB}，给了就算每对渲染结果的像素 sha256。返回配置 dict。"""
    by_split: dict[str, list[dict[str, Any]]] = {}
    infos: dict[str, list[dict[str, Any]]] = {}
    for idx, split, depth in sorted(images, key=lambda t: t[0]):
        ps, info = pairs_for_image(idx, depth)
        by_split.setdefault(split, []).extend(ps)
        infos.setdefault(split, []).append(info)
    pairs: list[dict[str, Any]] = []
    stats = {}
    for split in sorted(by_split):
        assigned = assign_ab(by_split[split], split)
        for p in assigned:
            if render is not None:
                img = render_pair(render[p["idx"]], p["A"], p["B"])
                p["render_sha256"] = pixels_sha256(img)
        pairs.extend(assigned)
        inf = infos[split]
        stats[split] = {
            "n_images": len(inf),
            "n_pairs": len(assigned),
            "n_A_closer": sum(1 for p in assigned if p["closer"] == "A"),
            "n_B_closer": sum(1 for p in assigned if p["closer"] == "B"),
            "n_images_short": sum(1 for i in inf if i["shortfall"] > 0),
            "shortfall_total": sum(i["shortfall"] for i in inf),
            "attempts_total": sum(i["attempts"] for i in inf),
            "rejected_total": {k: sum(i["rejected"][k] for i in inf) for k in inf[0]["rejected"]} if inf else {},
            # 深度边缘保护剔掉的有效像素占比（候选像素 = 有效且窗口内深度平稳）
            "n_valid_pixels_total": sum(i["n_valid"] for i in inf),
            "n_candidate_pixels_total": sum(i["n_candidates"] for i in inf),
        }
    return {
        "version": PAIRS_VERSION,
        "salt": SUBSET_SALT,
        "constraints": constraints_record(),
        "marker": marker_record(),
        "stats": stats,
        "pairs": sorted(pairs, key=lambda r: (r["split"], r["id"])),
    }


def constraints_record() -> dict[str, Any]:
    r0, r1, c0, c1 = NYU_EIGEN_CROP
    return {
        "pairs_per_image": NYU_PAIRS_PER_IMAGE,
        "depth_key": NYU_PAIR_DEPTH_KEY,
        "eigen_crop_rows_cols_half_open": [r0, r1, c0, c1],
        "valid_depth_open_interval_m": [NYU_DEPTH_MIN, NYU_DEPTH_MAX],
        "min_pixel_distance": f"短边 × {NYU_PAIR_MIN_DIST_NUM}/{NYU_PAIR_MIN_DIST_DEN}",
        "min_rel_depth_diff": NYU_PAIR_MIN_REL_DEPTH_DIFF,
        "max_attempts_per_image": NYU_PAIR_MAX_ATTEMPTS,
        "local_window": NYU_PAIR_LOCAL_WINDOW,
        "edge_guard": "窗口内全有效，且窗口内任意 4 邻接两点的深度 max/min ≤ 1 + neighbor_jump_max_rel（相邻像素跳变判据）" if NYU_PAIR_LOCAL_WINDOW else None,
        "neighbor_jump_max_rel": NYU_PAIR_NEIGHBOR_MAX_REL if NYU_PAIR_LOCAL_WINDOW else None,
        "ab_balance": "每个划分内按哈希排序，前 floor(n/2) 对 A 更近",
        "point_coords": "整数像素 (x=列, y=行)，标记以该像素为中心",
    }


def marker_record() -> dict[str, Any]:
    return {
        "renderer": "src.executors.overlay.render_overlay（一个 mask：两个环 + 两个字母）",
        "color_index": NYU_MARKER_COLOR_INDEX,
        "outer_radius": NYU_MARKER_OUTER_RADIUS,
        "inner_radius": NYU_MARKER_INNER_RADIUS,
        "label_scale": NYU_LABEL_SCALE,
        "label_gap": NYU_LABEL_GAP,
        "render_sha256": "sha256(渲染后 RGB 数组的 C 序字节)，与 PNG 编码器无关",
    }


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def glyph(letter: str) -> np.ndarray:
    rows = NYU_LABEL_GLYPHS[letter]
    g = np.array([[c == "#" for c in row] for row in rows], dtype=bool)
    return np.kron(g, np.ones((NYU_LABEL_SCALE, NYU_LABEL_SCALE), dtype=bool))


def ring_mask(shape: tuple[int, int], x: int, y: int) -> np.ndarray:
    """以像素 (x, y) 为中心的环：内半径 < 中心距 ≤ 外半径（按像素中心的整数偏移算）。"""
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    return (d2 <= NYU_MARKER_OUTER_RADIUS**2) & (d2 > NYU_MARKER_INNER_RADIUS**2)


def _ring_box(x: int, y: int) -> tuple[int, int, int, int]:
    R = NYU_MARKER_OUTER_RADIUS
    return (x - R, y - R, x + R, y + R)


def _label_candidates(x: int, y: int, h: int, w: int) -> list[tuple[int, int]]:
    """标签左上角 (x0, y0) 的候选：右上、左上、右下、左下（紧贴环的外接正方形对角，相隔 NYU_LABEL_GAP）。"""
    R, g = NYU_MARKER_OUTER_RADIUS, NYU_LABEL_GAP
    right, left = x + R + g, x - R - g - w + 1
    up, down = y - R - g - h + 1, y + R + g
    return [(right, up), (left, up), (right, down), (left, down)]


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def label_layout(first: Sequence[int], second: Sequence[int], H: int, W: int) -> dict[str, tuple[int, int]] | None:
    """先放 first 的标签 "A"，再放 second 的 "B"。放不下返回 None。"""
    pts = {"A": (int(first[0]), int(first[1])), "B": (int(second[0]), int(second[1]))}
    rings = {k: _ring_box(*v) for k, v in pts.items()}
    placed: dict[str, tuple[int, int, int, int]] = {}
    out: dict[str, tuple[int, int]] = {}
    for k in LETTERS:
        other = "B" if k == "A" else "A"
        g = glyph(k)
        h, w = g.shape
        for x0, y0 in _label_candidates(*pts[k], h, w):
            box = (x0, y0, x0 + w - 1, y0 + h - 1)
            if x0 < 0 or y0 < 0 or box[2] >= W or box[3] >= H:
                continue
            if _overlap(box, rings[other]) or any(_overlap(box, b) for b in placed.values()):
                continue
            placed[k] = box
            out[k] = (x0, y0)
            break
        else:
            return None
    return out


def marker_mask(shape: tuple[int, int], A: Sequence[int], B: Sequence[int], *, swap_labels: bool = False) -> np.ndarray:
    """两个环 + 两个字母的 mask。swap_labels（DEV-0036，D72-2 的"对调标记"去偏）：标签的位置照原样（label_layout(A, B) 给出的两个
    位置），只把两个位置上的字形互换——点 A 旁边写 "B"，点 B 旁边写 "A"；环不变。默认 False 与原来逐位相同。"""
    H, W = shape
    lay = label_layout(A, B, H, W)
    if lay is None:
        raise ValueError(f"A={A}、B={B} 的标签放不下")
    m = ring_mask(shape, int(A[0]), int(A[1])) | ring_mask(shape, int(B[0]), int(B[1]))
    for k in LETTERS:
        g = glyph(_drawn_letter(k, swap_labels))
        x0, y0 = lay[k]
        m[y0 : y0 + g.shape[0], x0 : x0 + g.shape[1]] |= g
    return m


def _drawn_letter(k: str, swap_labels: bool) -> str:
    """位置 k（"A" 的标签位置 = 点 A 旁边）上画的字母：不对调就是 k，对调就是另一个。对调要求两个字形同样大小（标签框不变）。"""
    if not swap_labels:
        return k
    other = LETTERS[1 - LETTERS.index(k)]
    if glyph(k).shape != glyph(other).shape:
        raise ValueError(f"字形 {k} 与 {other} 的大小不同，对调后标签框会变")
    return other


def label_boxes(shape: tuple[int, int], A: Sequence[int], B: Sequence[int]) -> dict[str, tuple[int, int, int, int]]:
    """两个标签的框（半开区间 (x0, y0, x1, y1)，像素坐标），位置同 marker_mask（对调与否都一样）。"""
    H, W = shape
    lay = label_layout(A, B, H, W)
    if lay is None:
        raise ValueError(f"A={A}、B={B} 的标签放不下")
    out = {}
    for k in LETTERS:
        g = glyph(k)
        x0, y0 = lay[k]
        out[k] = (x0, y0, x0 + g.shape[1], y0 + g.shape[0])
    return out


def render_pair(rgb: np.ndarray, A: Sequence[int], B: Sequence[int], *, swap_labels: bool = False) -> np.ndarray:
    """在 RGB 图上画 A、B 两个点标记，返回新图（不改输入）。swap_labels（DEV-0036）：两个标签的字形互换、位置不变（见
    marker_mask）；默认 False 与原来逐位相同。"""
    img = np.asarray(rgb)
    m = marker_mask(img.shape[:2], A, B, swap_labels=swap_labels)
    empty = np.zeros(img.shape[:2], dtype=bool)
    return render_overlay(img, [empty] * NYU_MARKER_COLOR_INDEX + [m])


def pixels_sha256(img: np.ndarray) -> str:
    a = np.ascontiguousarray(img, dtype=np.uint8)
    h = hashlib.sha256(f"{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def dump_pairs(cfg: Mapping[str, Any]) -> str:
    """点对配置的规范文本：每个点对一行，其余缩进 1（逐字节复现靠它）。"""
    head = {k: v for k, v in cfg.items() if k != "pairs"}
    lines = json.dumps(head, ensure_ascii=False, indent=1)[:-2]  # 去掉末尾 "\n}"
    body = ",\n".join("  " + json.dumps(p, ensure_ascii=False, separators=(",", ":")) for p in cfg["pairs"])
    return f'{lines},\n "pairs": [\n{body}\n ]\n}}\n'


def load_pairs(path: str | Path = PAIRS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
