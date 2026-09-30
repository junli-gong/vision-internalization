#!/usr/bin/env python
"""RUN-0006 §4：E5 生成器的跨机器核对与生成速度。只调用 src.synth 的公开接口（generate、Sample 的公开字段、TASK_BINS）。

usage (在 $REPO 下):
  python experiments/P0/0006_env_synth/probe_scripts/p10_synth_probe.py digests <out.json>   # §4.1 + 分量子摘要 + §4.2 的服务器侧数据
  python experiments/P0/0006_env_synth/probe_scripts/p10_synth_probe.py speed <out.json>     # §4.3
  python experiments/P0/0006_env_synth/probe_scripts/p10_synth_probe.py digests_only          # 打印 CASES 的摘要（诊断子进程用）

钉住的摘要从 tests/test_synth_repro.py 读取（PINNED_CE、PINNED_D），不在这里重抄。
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

from src.synth import TASK_BINS, generate  # noqa: E402
import test_synth_repro as T  # noqa: E402  (只读取 PINNED_CE / PINNED_D)

PINNED = {**T.PINNED_CE, **T.PINNED_D}
CASES = sorted(T.PINNED_CE) + sorted(T.PINNED_D)


def arr_digest(a):
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype.str}{a.shape}".encode() + a.tobytes()).hexdigest()[:16]


def component_digests(s):
    """各分量各自的 sha256 前 16 位（与 sample_digest 的输入分量一一对应），供两台机器逐分量对照。"""
    meta = {k: v for k, v in s.meta.items() if k != "provenance"}
    out = {"instruction": hashlib.sha256(s.instruction.encode()).hexdigest()[:16],
           "image": arr_digest(s.image), "target_mask": arr_digest(s.target_mask), "target_grid": arr_digest(s.target_grid),
           "object_masks": arr_digest(s.object_masks)}
    for name in sorted(s.baselines):
        out[f"baseline:{name}:mask"] = arr_digest(s.baselines[name].mask)
        out[f"baseline:{name}:grid"] = arr_digest(s.baselines[name].grid)
    out["meta"] = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()[:16]
    for k in ("scene", "final_poses", "difficulty", "baseline_iou"):
        if k in s.meta:
            out[f"meta.{k}"] = hashlib.sha256(json.dumps(s.meta[k], sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return out


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def cmd_digests(out_path):
    import pymunk
    res = {"platform": {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
                        "numpy": np.__version__, "pymunk": pymunk.version, "chipmunk": pymunk.chipmunk_version},
           "cases": []}
    for case in CASES:
        s = generate(*case)
        d = s.digest()
        res["cases"].append({"case": list(case), "server_digest": d, "pinned_digest": PINNED[case], "equal": d == PINNED[case],
                             "scene_seed": s.meta.get("scene_seed"), "difficulty": s.meta.get("difficulty"),
                             "component_digests": component_digests(s), "provenance": s.meta.get("provenance")})
    # §4.2：不一致的 d case，同一 (split, index) 在各档 k 下的服务器侧位姿（开发机上跑同一段即可逐项对照）
    loc = []
    for c in res["cases"]:
        if c["case"][0] != "physics" or c["equal"]:
            continue
        _, split, index = c["case"]
        per_k = []
        ref = generate("physics", split, index)
        for b, k in enumerate(TASK_BINS["physics"]):
            s = generate("physics", split, index, bin=b)
            per_k.append({"bin": b, "k": s.meta["k"], "digest": s.digest(), "component_digests": component_digests(s),
                          "scene_identical_to_default_bin": s.meta["scene"] == ref.meta["scene"],
                          "final_poses": s.meta["final_poses"], "target_displacement_px": s.meta["target_displacement_px"],
                          "max_penetration_px": s.meta.get("max_penetration_px"), "baseline_iou": s.meta.get("baseline_iou"),
                          "target_mask_iou_vs_default_bin": iou(s.target_mask, ref.target_mask)})
        loc.append({"case": c["case"], "per_k": per_k,
                    "note": "开发机侧的位姿与 mask 只以整样本摘要钉住，服务器上无法直接算出与开发机的位姿差和 IoU；"
                            "在开发机上对同一 (split, index) 跑本脚本的 digests 模式即可逐分量、逐档对照。"})
    res["physics_localization_server_side"] = loc
    Path(out_path).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for c in res["cases"]:
        print(c["case"], "equal" if c["equal"] else "DIFF", c["server_digest"][:16], c["pinned_digest"][:16], "scene_seed", c["scene_seed"])


def cmd_speed(out_path):
    res = {}
    for task in ("occlusion", "physics", "imagery"):
        n = 200
        t0w, t0c = time.perf_counter(), time.process_time()
        for i in range(n):
            generate(task, "train", i)  # bin=None → i % 档数（默认轮转）
        w, c = time.perf_counter() - t0w, time.process_time() - t0c
        res[task] = {"n": n, "split": "train", "indices": [0, n - 1], "bins_rotation": f"bin = index % {len(TASK_BINS[task])}",
                     "wall_s": round(w, 3), "cpu_s": round(c, 3), "samples_per_s_wall": round(n / w, 2), "samples_per_s_cpu": round(n / c, 2)}
        print(task, res[task], flush=True)
    res["platform"] = {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(), "numpy": np.__version__}
    Path(out_path).write_text(json.dumps(res, indent=1))


def cmd_digests_only():
    for case in CASES:
        print("|".join(map(str, case)), generate(*case).digest())


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "digests":
        cmd_digests(sys.argv[2])
    elif mode == "speed":
        cmd_speed(sys.argv[2])
    elif mode == "digests_only":
        cmd_digests_only()
    else:
        raise SystemExit(f"unknown mode {mode}")
