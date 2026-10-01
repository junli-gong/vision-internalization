"""去重与剔除（D11-7、D20；简报 §4 第 7 条）：按图像 id 与感知哈希，加载时就核对，被剔除的样本逐条记进统计。

- phash：与 imagehash.phash 默认定义相同的 64 位感知哈希（只用 PIL、numpy、scipy，无学习组件）。
- Exclusion：要剔除的规范图像键与感知哈希，各带原因（"D20 探针用图"、"开发子集"、"训练集图像"…）。
- apply_exclusions：逐条样本核对，返回保留的样本与统计（每个原因的计数、被剔除样本的 id 与命中的键/距离）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.fft import dct

from src.data.constants import (
    PHASH_HIGHFREQ,
    PHASH_MAX_HAMMING,
    PHASH_SIZE,
    PROBE_IMAGE_KEYS,
    PROBE_IMAGE_PHASH,
)
from src.data.records import Sample


def phash_image(img: Image.Image) -> str:
    """64 位 pHash 的十六进制串（16 个字符，高位在前，按行优先展开）。"""
    n = PHASH_SIZE * PHASH_HIGHFREQ
    g = img.convert("L").resize((n, n), Image.Resampling.LANCZOS)
    px = np.asarray(g, dtype=np.float64)
    d = dct(dct(px, axis=0, type=2), axis=1, type=2)
    low = d[:PHASH_SIZE, :PHASH_SIZE]
    bits = (low > np.median(low)).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:0{PHASH_SIZE * PHASH_SIZE // 4}x}"


def phash_file(path: str | Path) -> str:
    with Image.open(path) as im:
        return phash_image(im)


def phash_bytes(data: bytes) -> str:
    import io

    with Image.open(io.BytesIO(data)) as im:
        return phash_image(im)


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


@dataclass
class Exclusion:
    """要剔除的图像：keys 为 {规范键: 原因}，phashes 为 {感知哈希: (原因, 来源键)}，ids 为 {样本 id: 原因}。"""

    keys: dict[str, str] = field(default_factory=dict)
    phashes: dict[str, tuple[str, str]] = field(default_factory=dict)
    ids: dict[str, str] = field(default_factory=dict)
    max_hamming: int = PHASH_MAX_HAMMING

    def add_keys(self, keys: Iterable[str], reason: str) -> "Exclusion":
        for k in keys:
            self.keys.setdefault(k, reason)
        return self

    def add_phashes(self, items: Mapping[str, str], reason: str) -> "Exclusion":
        """items：{来源键: 感知哈希}。"""
        for src_key, h in items.items():
            if h:
                self.phashes.setdefault(h, (reason, src_key))
        return self

    def add_ids(self, ids: Iterable[str], reason: str) -> "Exclusion":
        for i in ids:
            self.ids.setdefault(i, reason)
        return self


REASON_PROBE = "D20 探针用图"
REASON_DEV = "开发子集"


def probe_exclusion() -> Exclusion:
    """D20：探针用图按规范键与感知哈希两路剔除。"""
    ex = Exclusion().add_keys(PROBE_IMAGE_KEYS, REASON_PROBE)
    ex.add_phashes(PROBE_IMAGE_PHASH, REASON_PROBE)
    return ex


def check_sample(s: Sample, ex: Exclusion) -> dict[str, Any] | None:
    """命中任何一条剔除规则就返回命中记录，否则 None。顺序：样本 id → 图像键 → 感知哈希。"""
    if s.id in ex.ids:
        return {"id": s.id, "reason": ex.ids[s.id], "by": "sample_id"}
    for k in s.image_keys:
        if k in ex.keys:
            return {"id": s.id, "reason": ex.keys[k], "by": "image_key", "key": k}
    if ex.phashes:
        for k, h in zip(s.image_keys, s.phash or (None,) * len(s.image_keys)):
            if not h:
                continue
            for h2, (reason, src_key) in ex.phashes.items():
                d = hamming(h, h2)
                if d <= ex.max_hamming:
                    return {"id": s.id, "reason": reason, "by": "phash", "key": k, "matched": src_key, "hamming": d}
    return None


def apply_exclusions(samples: Sequence[Sample], ex: Exclusion) -> tuple[list[Sample], dict[str, Any]]:
    """逐条核对。返回 (保留的样本, 统计)。统计：n_in、n_kept、按原因计数、被剔除的逐条记录、没有感知哈希的样本数。"""
    kept: list[Sample] = []
    dropped: list[dict[str, Any]] = []
    by_reason: dict[str, int] = {}
    n_no_phash = 0
    for s in samples:
        if not s.phash or any(h is None for h in s.phash):
            n_no_phash += 1
        hit = check_sample(s, ex)
        if hit is None:
            kept.append(s)
        else:
            dropped.append(hit)
            by_reason[hit["reason"]] = by_reason.get(hit["reason"], 0) + 1
    return kept, {
        "n_in": len(samples),
        "n_kept": len(kept),
        "excluded_by_reason": dict(sorted(by_reason.items())),
        "excluded": dropped,
        "n_without_phash": n_no_phash,
    }


def cross_overlap(a: Mapping[str, str], b: Mapping[str, str], max_hamming: int = PHASH_MAX_HAMMING) -> dict[str, Any]:
    """两组图像（{规范键: 感知哈希}）之间的重合：同键的、以及感知哈希近重复的（O(n·m)，评测集规模够用）。"""
    same_key = sorted(set(a) & set(b))
    near = []
    items_b = [(k, int(h, 16)) for k, h in b.items() if h]
    for ka, ha in a.items():
        if not ha:
            continue
        va = int(ha, 16)
        for kb, vb in items_b:
            d = bin(va ^ vb).count("1")
            if d <= max_hamming:
                near.append({"a": ka, "b": kb, "hamming": d})
    return {"same_key": same_key, "n_same_key": len(same_key), "phash_near": near, "n_phash_near": len(near)}
