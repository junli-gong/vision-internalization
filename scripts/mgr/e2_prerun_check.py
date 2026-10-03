"""E2 预跑的收敛判据（E2 预注册 v1 §5 第 3 步；管理员脚本，只读 curves.jsonl，不改任何实验逻辑）。

判据（写于看任何 E2 结果之前）：qwen3.5-0.8b 的深度，线性探针在第 0 层、中间层（floor(N/2)）、最后一层（L{N}，最终归一化之后）
三处，留出子集的 AbsRel 都满足：
  (1) 最后一个曲线点比第一个曲线点好（AbsRel 更小）；
  (2) 倒数第二个与最后一个曲线点的相对差 < REL_TOL（|a − b| / b）。
配置里深度 3,000 步、每 250 步一个点，所以就是"第 3,000 步比第 250 步好"与"第 2,750 步与第 3,000 步相对差 < 2%"。

用法：python scripts/mgr/e2_prerun_check.py <叶目录：.../qwen3.5-0.8b/depth> [--out prerun_check.json]
退出码：0 = 三处都满足；3 = 有不满足的；2 = 读不到所需的曲线点（文件缺失、层不全、点数 < 2）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REL_TOL = 0.02
METRIC = "abs_rel"
KIND = "linear"
LAYER = re.compile(r"L(\d+)")


def read_curves(path: Path) -> list[dict]:
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1  # 末尾不完整的一行不读
    return [json.loads(l) for l in raw[:keep].decode("utf-8").splitlines() if l.strip()]


def check(leaf: Path) -> tuple[int, dict]:
    curves = read_curves(leaf / "curves.jsonl")
    lin = [c for c in curves if c.get("kind") == KIND]
    layers = sorted({int(m.group(1)) for c in lin if (m := LAYER.fullmatch(str(c.get("feature"))))})
    if not layers or layers[0] != 0:
        return 2, {"error": f"线性探针的层不全：{layers}"}
    n_last = layers[-1]
    targets = {"first": 0, "middle": n_last // 2, "last": n_last}
    if targets["middle"] not in layers:  # 9B 隔层时中间层可能不在；取不超过它的最近一层
        targets["middle"] = max(l for l in layers if l <= n_last // 2)
    out = {"criterion": {"metric": METRIC, "kind": KIND, "rel_tol": REL_TOL,
                         "rule": "最后一点 < 第一点，且 |倒数第二点 − 最后一点| / 最后一点 < rel_tol"},
           "leaf": str(leaf), "layers": {}, "pass": True}
    for role, layer in targets.items():
        pts = sorted((c["step"], c["val"][METRIC]) for c in lin if c.get("feature") == f"L{layer}" and c["val"].get(METRIC) is not None)
        if len(pts) < 2:
            return 2, {"error": f"L{layer} 的曲线点不足 2 个：{pts}"}
        (s0, v0), (sp, vp), (sl, vl) = pts[0], pts[-2], pts[-1]
        improved = vl < v0
        rel = abs(vp - vl) / vl if vl else float("inf")
        ok = bool(improved and rel < REL_TOL)
        out["layers"][role] = {"feature": f"L{layer}", "first": [s0, v0], "second_last": [sp, vp], "last": [sl, vl],
                               "improved": improved, "rel_change_last_two": rel, "pass": ok}
        out["pass"] = out["pass"] and ok
    return (0 if out["pass"] else 3), out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("leaf", help="E2 的叶目录（含 curves.jsonl），如 <out>/qwen3.5-0.8b/depth")
    p.add_argument("--out", help="判定结果写成 JSON")
    a = p.parse_args(argv)
    leaf = Path(a.leaf)
    if not (leaf / "curves.jsonl").exists():
        print(f"找不到 {leaf / 'curves.jsonl'}", file=sys.stderr)
        return 2
    rc, doc = check(leaf)
    doc["exit_code"] = rc
    text = json.dumps(doc, ensure_ascii=False, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
