"""C3 对照的稀疏文本目标（DEV-0025 §2；DEV-0030 §3；简报 §8.4、§8.5）：同样的图像与样本日程，目标换成框、多边形、相对深度陈述。

- 框（box_target；DEV-0030 §3，D56-3、D60-4）：本家族**原生**的框格式，由适配器的 format_boxes 生成（parse_boxes 的逆：坐标按
  原图宽高换算到 0–1000 并取整；Qwen、InternVL x 在前，Gemma 的 box_2d y 在前），label 写指代句（三家族零样本实测的原生输出都
  回写了被指代的物体：model_notes §5.5、§6.5、§7.5）。提示是 E1 定位任务（refexp_box）本家族的提示原文（load_box_template，读
  configs/e1_prompts.json，不另写；要求 native = true），按 E1 的规则渲染（render_e1_template）。往返：本家族的 parse_boxes 解析
  回来，与 GT 框每个坐标差 ≤ W/2000、H/2000（格点精度，tests 钉住）。DEV-0025 暂行的统一格式 "[x0, y0, x1, y1]" 已删除。
- 多边形（polygon_text）：坐标是 0–1000 的整数（x 在前）：x_norm = round(x / W · 1000)、y_norm = round(y / H · 1000)（.5 进位），
  裁剪到 [0, 1000]；与 E1 的解析规则（src/parsing/spatial，geometry.norm1000_to_abs 的逆）相同。GT 的 COCO 多边形（扁平的
  [x0, y0, x1, y1, …]，可多个部件）→ 单个部件 "[[x, y], [x, y], …]"、多个部件 "[[[x, y], …], [[x, y], …]]"（E1 的统一格式，
  parse_polygons 都认）。顶点总数超过 max_points 时等步长抽稀：步长 ceil(总数 / max_points)，每个部件从第 0 个顶点起每隔步长
  取一个；取整后相邻重复的点（含首尾）去掉；少于 3 个点的部件丢掉；一个部件都不剩时返回 None（调用方退回框）。提示是 E1 的统一
  多边形提示（load_polygon_template，要求 provenance = unified）。
- 相对深度（depth_pairs）：点对按 src/data/nyu_pairs.pairs_for_image 的规则从 GT 深度抽样（深度有效、Eigen crop 内、相距 ≥ 短边
  10%、深度相对差 ≥ 15%、边缘保护；随机性由传入的 key 决定），A / B 的先后按 (种子, key, 对的序号) 的哈希决定；点坐标是像素
  中心 (x + 0.5, y + 0.5) 的 0–1000 坐标。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.constants import NORM1000_SCALE, NYU_EVAL_HW, PIXEL_CENTER_OFFSET
from src.data.nyu_pairs import pairs_for_image


def norm1000(v: float, size: float) -> int:
    """绝对像素坐标 → 0–1000 的整数（四舍五入、.5 进位，裁剪到 [0, 1000]）。"""
    if not size > 0:
        raise ValueError(f"size 必须为正，收到 {size}")
    x = math.floor(float(v) / float(size) * NORM1000_SCALE + 0.5)
    return int(min(max(x, 0), int(NORM1000_SCALE)))


def box_target(format_boxes: Callable[..., str], box_xyxy: Sequence[float], width: float, height: float,
               label: str) -> str:
    """GT 框（绝对像素 xyxy）→ 本家族原生框格式的文本；format_boxes 是适配器的 format_boxes（规则见模块说明）。"""
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    if x1 < x0 or y1 < y0:
        raise ValueError(f"框颠倒：{box_xyxy}")
    return str(format_boxes([[x0, y0, x1, y1]], width=float(width), height=float(height), labels=[str(label)]))


def _part_points(flat: Sequence[float], width: float, height: float, stride: int) -> list[list[int]]:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1, 2)[::stride]
    pts: list[list[int]] = []
    for x, y in arr:
        p = [norm1000(x, width), norm1000(y, height)]
        if not pts or pts[-1] != p:
            pts.append(p)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def polygon_text(polygons: Sequence[Sequence[float]], width: float, height: float, *, max_points: int) -> str | None:
    """COCO 多边形 → E1 的 JSON 多边形文本（规则见模块说明）；没有可用的部件时 None。"""
    if max_points < 3:
        raise ValueError(f"max_points 必须 ≥ 3，收到 {max_points}")
    parts = [p for p in polygons if len(p) >= 6 and len(p) % 2 == 0]
    total = sum(len(p) // 2 for p in parts)
    if total == 0:
        return None
    stride = max(1, math.ceil(total / max_points))
    kept = [pts for pts in (_part_points(p, width, height, stride) for p in parts) if len(pts) >= 3]
    if not kept:
        return None
    value: Any = kept[0] if len(kept) == 1 else kept
    return json.dumps(value, separators=(", ", ": "))


def _u64(*parts: Any) -> int:
    s = "|".join(str(p) for p in ("visint-e3-c3-ab", *parts))
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def depth_pairs(depth_m: np.ndarray, *, key: str, n_pairs: int, seed: int) -> list[dict[str, Any]]:
    """GT 深度（米，NYU_EVAL_HW）→ 至多 n_pairs 个点对 [{A: [x, y], B: [x, y], depth_A, depth_B, closer}]（规则见模块说明）。"""
    d = np.asarray(depth_m, dtype=np.float32)
    if d.shape != tuple(NYU_EVAL_HW):
        raise ValueError(f"相对深度点对按 NYU 的 {tuple(NYU_EVAL_HW)} 帧定义（Eigen crop），收到 {d.shape}")
    raw, _ = pairs_for_image(key, d, n_pairs)
    out = []
    for p in raw:
        swap = bool(_u64(seed, key, p["k"]) & 1)
        (A, dA), (B, dB) = ((p["q"], p["depth_q"]), (p["p"], p["depth_p"])) if swap else ((p["p"], p["depth_p"]), (p["q"], p["depth_q"]))
        out.append({"A": list(A), "B": list(B), "depth_A": float(dA), "depth_B": float(dB),
                    "closer": "A" if dA < dB else "B"})
    return out


def point1000(xy: Sequence[int], width: int, height: int) -> tuple[int, int]:
    """整数像素 (x, y) 的中心 → 0–1000 坐标。"""
    return (norm1000(xy[0] + PIXEL_CENTER_OFFSET, width), norm1000(xy[1] + PIXEL_CENTER_OFFSET, height))


def depth_pairs_prompt(template: str, line: str, pairs: Sequence[Mapping[str, Any]], width: int, height: int) -> str:
    lines = []
    for i, p in enumerate(pairs, start=1):
        ax, ay = point1000(p["A"], width, height)
        bx, by = point1000(p["B"], width, height)
        lines.append(line.format(i=i, ax=ax, ay=ay, bx=bx, by=by))
    return template.format(pairs="\n".join(lines))


def depth_pairs_answer(line: str, pairs: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(line.format(i=i, closer=p["closer"]) for i, p in enumerate(pairs, start=1))


def _e1_entry(doc: Mapping[str, Any], key: str, family: str, repo_root: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """c3.<key> 指向的 E1 提示条目（configs/e1_prompts.json 的 tasks.<task>.families.<family>）→ (条目, 出处记录)。"""
    from pathlib import Path

    ref = doc["c3"][key]
    raw = (Path(repo_root) / ref["config"]).read_bytes()
    entry = json.loads(raw)["tasks"][ref["task"]]["families"][family]
    return entry, {"config": ref["config"], "task": ref["task"], "sha256": hashlib.sha256(raw).hexdigest(),
                   "provenance": entry.get("provenance")}


def load_polygon_template(doc: Mapping[str, Any], family: str, repo_root: Any) -> tuple[str, dict[str, Any]]:
    """c3.polygon_template_from 指向的 E1 统一提示（configs/e1_prompts.json 的 tasks.<task>.families.<family>.template）与出处记录
    （配置文件的 sha256、任务、provenance）。要求 provenance 是 unified（三个家族逐字相同的统一提示）。"""
    entry, src = _e1_entry(doc, "polygon_template_from", family, repo_root)
    if entry.get("provenance") != "unified":
        raise ValueError(f"{src['task']} 的 {family} 提示不是统一提示（provenance = {entry.get('provenance')!r}）")
    return str(entry["template"]), src


def load_box_template(doc: Mapping[str, Any], family: str, repo_root: Any) -> tuple[str, dict[str, Any]]:
    """c3.box_template_from 指向的 E1 定位提示（本家族的原文，DEV-0030 §3）与出处记录（另记 native、source_url）。要求 native =
    true：本家族自己的原生提示，与 format_boxes 生成的原生框格式配套。"""
    entry, src = _e1_entry(doc, "box_template_from", family, repo_root)
    if entry.get("native") is not True:
        raise ValueError(f"{src['task']} 的 {family} 提示不是本家族的原生提示（native = {entry.get('native')!r}）")
    return str(entry["template"]), {**src, "native": True, "source_url": entry.get("source_url")}


def render_e1_template(template: str, expression: str) -> str:
    """E1 对 RefCOCO 系任务的渲染规则（src/e1/prompts.PromptConfig.render）：template.format(expression=…).strip()。"""
    return template.format(expression=expression).strip()


__all__ = ["box_target", "depth_pairs", "depth_pairs_answer", "depth_pairs_prompt", "load_box_template",
           "load_polygon_template", "norm1000", "point1000", "polygon_text", "render_e1_template"]
