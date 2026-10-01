#!/usr/bin/env python
"""批处理核对（DEV-0011，D28-5）：同一批输入逐条生成一遍、批量生成一遍，报 raw_text 的逐字一致率与吞吐。输出一个 JSON。

服务器（真模型，权重按注册表的固定 revision 加载）：
    python scripts/validate_batching.py --model qwen3.5-0.8b --image-policy pixel_budget:300 \\
        --jsonl <rows.jsonl> --batch-sizes 8 16 32 --out <json>
开发机彩排（本地构造的小型随机模型，CPU；自动合成 8 条输入；--model 选家族，不给即 Qwen3.5）：
    python scripts/validate_batching.py --tiny --model gemma-4-e2b-it --out /tmp/vb_tiny.json

输入 JSONL，每行一个对象：
    {"id": "可选", "image": "图路径" | ["图1", "图2"] | null, "instruction": "…", "enable_thinking": false, "max_new_tokens": 64,
     "order": "instruction_first"（可选，默认 instruction_first；或 image_first）}
相对路径相对 JSONL 所在目录。

做法：
- 图像口径 --image-policy（写法见 src/adapters/image_policy.parse_policy：pixel_budget:300、fixed:480x640、soft_tokens:280、
  tiles:single、tiles:native、tiles:true:6）；不给时取 E1 冻结的口径（D28-3）。
- 先预热（D21）：逐条一次、按最大 batch 大小批量一次，耗时记下但不进吞吐。
- 逐条：每行 generate（greedy，do_sample=False 显式传入）。
- 批量：每个 batch 大小各跑一遍；行按 (enable_thinking, max_new_tokens) 分组（批内这两个参数必须相同），组内按输入顺序切批。
- 每行比较逐条与批量的 raw_text：逐字相同记一致；不同时记第一个不同的字符位置与 token 位置，并存下批量的 raw_text。
- 吞吐按"只含 model.generate 的墙钟"计（GenerationResult 的 batch_wall_time_s 求和；逐条是 wall_time_s 求和）：
  样本/秒 = 行数 / 墙钟和，token/秒 = 生成 token 总数 / 墙钟和。build_inputs 的耗时单列，两种做法相同。
- 解析后答案的一致率与指标的 bootstrap CI 由 E1 评测框架在开发子集上算，不在本脚本里做（任务书）。

退出码：0 = 跑完（一致率高低不影响）；2 = 出错（已有结果与报错先写进 --out 再退出）。argparse 用法错误也是 2。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.adapters import registry  # noqa: E402
from src.adapters.base import ORDERS  # noqa: E402
from src.adapters.factory import default_image_policy, make_adapter  # noqa: E402
from src.adapters.image_policy import PixelBudget, SoftTokenBudget, TilePolicy, parse_policy  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}  # 与 scripts/smoke_adapter.py 的 --dtype 取值相同
ROW_KEYS = {"id", "image", "instruction", "enable_thinking", "max_new_tokens", "order"}
TINY_BATCH_SIZES = (4, 8)
TINY_INIT_STD = 0.05  # 彩排的小模型重取参数（factory 的 init_std），否则 greedy 输出几乎与输入无关
# 彩排用的口径：小一些，前向快；Qwen 用像素预算以覆盖每张图网格不同的情形，InternVL 用多 tile 以覆盖每行 tile 数不同
TINY_POLICIES = {
    registry.FAMILY_QWEN35: PixelBudget(64),
    registry.FAMILY_GEMMA4: SoftTokenBudget(70),
    registry.FAMILY_INTERNVL35: TilePolicy(crop_to_patches=True, max_patches=4),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="注册键（见 src/adapters/registry.py）；--tiny 时选家族与小模型 config 的来源")
    p.add_argument("--image-policy", help="图像口径，如 pixel_budget:300；默认取 E1 冻结的口径（--tiny 时取小口径）")
    p.add_argument("--jsonl", help="输入 JSONL；--tiny 时可省略（自动合成 8 条）")
    p.add_argument("--batch-sizes", nargs="+", type=int, help=f"批大小；--tiny 默认 {list(TINY_BATCH_SIZES)}")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--limit", type=int, help="只取 JSONL 的前 N 行（调试用）")
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--tiny", action="store_true", help="用本地构造的小型随机模型在 CPU 上彩排")
    args = p.parse_args(argv)
    if args.tiny:
        args.device = args.device or "cpu"
        args.dtype = args.dtype or "fp32"
        args.batch_sizes = args.batch_sizes or list(TINY_BATCH_SIZES)
    else:
        if not args.model or not args.jsonl or not args.batch_sizes:
            p.error("真模型核对需要 --model、--jsonl 与 --batch-sizes")
        args.device = args.device or "cuda"
        args.dtype = args.dtype or "bf16"
    if any(b < 1 for b in args.batch_sizes):
        p.error("--batch-sizes 必须都是正整数")
    if args.limit is not None and args.limit < 1:
        p.error("--limit 必须是正整数")
    if args.image_policy is not None:
        try:
            parse_policy(args.image_policy)
        except ValueError as e:
            p.error(str(e))
    return args


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------


def load_rows(path: Path, *, limit: int | None = None) -> list[dict]:
    """读 JSONL 并逐行校验；image 统一成路径列表（null → []），相对路径相对 JSONL 所在目录。"""
    rows = []
    for k, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        extra = set(r) - ROW_KEYS
        if extra:
            raise ValueError(f"第 {k + 1} 行有不认识的键 {sorted(extra)}")
        if not isinstance(r.get("instruction"), str):
            raise ValueError(f"第 {k + 1} 行的 instruction 必须是字符串")
        if not isinstance(r.get("enable_thinking"), bool):
            raise ValueError(f"第 {k + 1} 行的 enable_thinking 必须是 true/false")
        n = r.get("max_new_tokens")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"第 {k + 1} 行的 max_new_tokens 必须是正整数")
        order = r.get("order", "instruction_first")
        if order not in ORDERS:
            raise ValueError(f"第 {k + 1} 行的 order 必须是 {ORDERS} 之一")
        img = r.get("image")
        paths = [] if img is None else [img] if isinstance(img, str) else list(img)
        if not all(isinstance(x, str) for x in paths):
            raise ValueError(f"第 {k + 1} 行的 image 必须是路径、路径列表或 null")
        resolved = [(path.parent / x).resolve() for x in paths]
        missing = [str(x) for x in resolved if not x.is_file()]
        if missing:
            raise ValueError(f"第 {k + 1} 行的图不存在：{missing}")
        rows.append({"id": str(r.get("id", len(rows))), "images": [str(x) for x in resolved],
                     "instruction": r["instruction"], "enable_thinking": r["enable_thinking"], "max_new_tokens": n,
                     "order": order})
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"{path} 里没有输入行")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("JSONL 里的 id 有重复")
    return rows


def write_tiny_rows(directory: Path) -> Path:
    """彩排用的合成输入：8 行，指令长短不一，图一横一竖、方图、宽图、两张图、没有图；两行开思考（另一组 max_new_tokens）。"""
    rng = np.random.default_rng(20261001)
    sizes = {"wide": (90, 200), "tall": (180, 110), "square": (120, 120), "land": (100, 150), "port": (150, 100)}
    for name, (h, w) in sizes.items():
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        arr[h // 4: h // 2, w // 4: w // 2, 1] = 230
        Image.fromarray(arr).save(directory / f"{name}.png")
    rows = [
        {"id": "r0", "image": "land.png", "instruction": "Describe the cat.", "enable_thinking": False, "max_new_tokens": 16},
        {"id": "r1", "image": "tall.png", "instruction": "What is in this picture? Answer in detail, mentioning colours.",
         "enable_thinking": False, "max_new_tokens": 16},
        {"id": "r2", "image": ["square.png", "wide.png"], "instruction": "Compare the two images.",
         "enable_thinking": False, "max_new_tokens": 16},
        {"id": "r3", "image": None, "instruction": "Say hi.", "enable_thinking": False, "max_new_tokens": 16},
        {"id": "r4", "image": "port.png", "instruction": "Where is the dog?", "enable_thinking": True, "max_new_tokens": 12},
        {"id": "r5", "image": "wide.png", "instruction": "Count the objects and output a number.",
         "enable_thinking": False, "max_new_tokens": 16, "order": "image_first"},
        {"id": "r6", "image": "land.png", "instruction": "x", "enable_thinking": True, "max_new_tokens": 12},
        {"id": "r7", "image": "square.png", "instruction": "Which part of the image is brightest?",
         "enable_thinking": False, "max_new_tokens": 16},
    ]
    path = directory / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def group_batches(rows: list[dict], batch_size: int) -> list[list[int]]:
    """行号的分批：先按 (enable_thinking, max_new_tokens) 分组（组按首次出现的顺序），组内按输入顺序每 batch_size 行切一批。"""
    groups: dict[tuple[bool, int], list[int]] = {}
    for k, r in enumerate(rows):
        groups.setdefault((r["enable_thinking"], r["max_new_tokens"]), []).append(k)
    return [idx[s: s + batch_size] for idx in groups.values() for s in range(0, len(idx), batch_size)]


# ---------------------------------------------------------------------------
# 比较
# ---------------------------------------------------------------------------


def first_diff(a, b) -> int | None:
    """两个序列（字符串或 token 列表）第一个不同的位置；完全相同为 None（一个是另一个的前缀时取较短者的长度）。"""
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def throughput(n_rows: int, n_tokens: int, wall: float) -> dict:
    return {"n_rows": n_rows, "generated_tokens": n_tokens, "gen_wall_s": round(wall, 6),
            "samples_per_s": n_rows / wall if wall > 0 else None, "tokens_per_s": n_tokens / wall if wall > 0 else None}


def run(args, report: dict) -> None:
    key = args.model or registry.keys_for_family(registry.FAMILY_QWEN35)[0]
    entry = registry.get_model(key)
    if args.image_policy is not None:
        policy = parse_policy(args.image_policy)
    else:
        policy = TINY_POLICIES[entry.family] if args.tiny else default_image_policy(entry.family)
    family_kw = {"init_std": TINY_INIT_STD} if args.tiny else {}
    ad = make_adapter(key, device=args.device, dtype=DTYPES[args.dtype], image_policy=policy, tiny=args.tiny,
                      **family_kw)
    report["model"] = {"key": entry.key, "family": entry.family, "repo": entry.repo, "revision": entry.revision,
                       "adapter_model_key": ad.model_key, "device": str(ad.device), "dtype": str(ad.dtype),
                       "image_policy": {"type": type(policy).__name__, **policy.as_dict()},
                       "tiny_init_std": family_kw.get("init_std")}

    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(args.jsonl) if args.jsonl else write_tiny_rows(Path(tmp))
        raw = jsonl.read_bytes()
        report["jsonl"] = {"path": str(args.jsonl) if args.jsonl else "(合成，--tiny)", "sha256": hashlib.sha256(raw).hexdigest()}
        rows = load_rows(jsonl, limit=args.limit)
        report["jsonl"]["n_rows"] = len(rows)
        t0 = time.perf_counter()
        inputs = []
        for r in rows:
            imgs = [Image.open(p).convert("RGB") for p in r["images"]]
            inputs.append(ad.build_inputs(imgs, r["instruction"], r["order"]))
        build_wall = time.perf_counter() - t0
        first_imgs = [Image.open(p).convert("RGB") for p in rows[0]["images"]]

    # 预热（不进吞吐）：逐条一次、批量（最大的 batch 大小）一次
    report["warmup"] = {
        "single_wall_s": round(ad.warmup(first_imgs), 4),
        "batch_wall_s": round(ad.warmup(first_imgs, batch_size=max(args.batch_sizes)), 4),
        "batch_size": max(args.batch_sizes), "counted": False,
    }
    report["build_inputs_wall_s"] = round(build_wall, 4)

    single = []
    for r, inp in zip(rows, inputs):
        single.append(ad.generate(inp, enable_thinking=r["enable_thinking"], max_new_tokens=r["max_new_tokens"],
                                  do_sample=False))
    report["sequential"] = throughput(len(rows), sum(s.n_generated_tokens for s in single),
                                      sum(s.wall_time_s for s in single))

    per_row = [{"id": r["id"], "enable_thinking": r["enable_thinking"], "max_new_tokens": r["max_new_tokens"],
                "n_images": len(r["images"]), "n_prompt_tokens": s.n_prompt_tokens, "n_visual_tokens": s.n_visual_tokens,
                "n_image_pixels": s.n_image_pixels, "grids_hw": [list(i.grid_hw) for i in inp.images],
                "single": {"raw_text": s.raw_text, "n_generated_tokens": s.n_generated_tokens, "stopped": s.stopped,
                           "wall_time_s": round(s.wall_time_s, 6)},
                "batched": {}}
               for r, inp, s in zip(rows, inputs, single)]
    report["batched"] = {}
    for bs in args.batch_sizes:
        batches = group_batches(rows, bs)
        results: dict[int, object] = {}
        wall = 0.0
        for idx in batches:
            r0 = rows[idx[0]]
            out = ad.generate_batch([inputs[k] for k in idx], enable_thinking=r0["enable_thinking"],
                                   max_new_tokens=r0["max_new_tokens"], do_sample=False)
            wall += out[0].batch_wall_time_s
            results.update(zip(idx, out))
        mismatches = []
        n_equal = n_tok_equal = 0
        for k, s in enumerate(single):
            b = results[k]
            equal = b.raw_text == s.raw_text
            tok_equal = b.token_ids == s.token_ids
            n_equal += equal
            n_tok_equal += tok_equal
            rec = {"raw_text_equal": equal, "token_ids_equal": tok_equal, "n_generated_tokens": b.n_generated_tokens,
                   "stopped": b.stopped}
            if not equal:
                rec.update(first_diff_char=first_diff(s.raw_text, b.raw_text),
                           first_diff_token=first_diff(s.token_ids, b.token_ids), raw_text=b.raw_text)
                mismatches.append({"id": rows[k]["id"], "first_diff_char": rec["first_diff_char"],
                                   "first_diff_token": rec["first_diff_token"]})
            per_row[k]["batched"][str(bs)] = rec
        report["batched"][str(bs)] = {
            **throughput(len(rows), sum(results[k].n_generated_tokens for k in range(len(rows))), wall),
            "n_batches": len(batches), "batch_row_counts": [len(i) for i in batches],
            "raw_text_equal": n_equal, "raw_text_equal_rate": n_equal / len(rows),
            "token_ids_equal": n_tok_equal, "token_ids_equal_rate": n_tok_equal / len(rows),
            "mismatches": mismatches,
        }
    report["rows"] = per_row


def main(argv=None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"script": "scripts/validate_batching.py", "argv": sys.argv[1:] if argv is None else list(argv),
                    "mode": "tiny" if args.tiny else "real",
                    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "ok": False,
                    "timing_note": "吞吐只含 model.generate 的墙钟（GenerationResult.wall_time_s / batch_wall_time_s），"
                                   "预热不计；build_inputs 单列"}
    t_all = time.perf_counter()
    try:
        run(args, report)
        report["ok"] = True
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        report.update(error=repr(e), traceback=traceback.format_exc()[-4000:])
    report["wall_time_s"] = round(time.perf_counter() - t_all, 2)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if not report["ok"]:
        print(f"失败：{report['error']}（已写出 {out_path}）", file=sys.stderr)
        return 2
    seq = report["sequential"]
    print(f"{report['model']['key']} {report['model']['image_policy']}  rows={seq['n_rows']}")
    print(f"sequential      {seq['samples_per_s']:.3f} samples/s  {seq['tokens_per_s']:.1f} tokens/s")
    for bs, b in report["batched"].items():
        print(f"batch {bs:>4s}      {b['samples_per_s']:.3f} samples/s  {b['tokens_per_s']:.1f} tokens/s  "
              f"raw_text equal {b['raw_text_equal']}/{b['n_rows']}  token_ids equal {b['token_ids_equal']}/{b['n_rows']}")
    print(f"wall={report['wall_time_s']} s  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
