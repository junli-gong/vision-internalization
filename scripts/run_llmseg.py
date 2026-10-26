#!/usr/bin/env python
"""外部基线 LLMSeg（DEV-0066；D125-4）：官方代码（third_party/llmseg @ f2191e66）+ 本地 Qwen3.5 提供方，在 W2 test 子集上评测。

运行（服务器；--out 是本次运行的目录 experiments/W2/<NNNN_slug>，结果写进 <out>/<模型>/refseg/llmseg/<划分>/）：
    python scripts/run_llmseg.py --model qwen3.5-9b --split all --data-root $PROJ/data --out experiments/W2/0074_llmseg_hpcgpu06 [--resume]
    python scripts/run_llmseg.py --model qwen3.5-9b --split test_refcoco_unc_testA --limit 32 ...     # 试跑：量调用数与用时
- 划分：W2 的 5 个 test 划分（--split all 依次跑），每个取 configs/w2_llmseg_subsets.json 的 300 句（hash_rank(salt = w2_llmseg_v1)
  前 300 句，src/llmseg/runner.py）。--limit N 只取前 N 句（试跑；进运行身份）。
- 主配置（官方论文的主结果）：--iterations 8，--topology / --refine-box / --no-postprocess 都关（这里不提供这三个开关）；--iterations 别的值
  写进设置名（llmseg_it<T>）。
- 本地提供方：经我们的适配器（E1 的图像口径），关思考、贪心（E1 的解码参数），图在前；各阶段的输出上限见 src/llmseg/provider.ANSWER_CAP_RULE。
  --thinking --thinking-budget N：开思考的对照（设置名 llmseg_think；不是主配置）。不调用任何闭源 API（官方的四家提供方调用被禁用）。
- 合批：--slots（同时在跑的管线数，默认 150）、--batch-size（子批行数上限，默认 32）、--max-batch-tokens（行数 × 最长提示，默认 48000，
  同 configs/w2.json 的 W2 挑选，A100-40GB 上实测过的预填规模）、--chunk-size（续跑粒度，默认 150）。都进运行身份（批的组成会影响
  GPU 上的数值）。先用 --limit 试跑一小段，看 logs/gpu_summary.json 的外推与显存，再跑全量。
- --save N：前 N 条留下官方管线的全部 PNG（叠加图、mask，每条约 20 张），其余只留 JSON 与日志（默认 3）。工作目录
  <叶目录>/work/ 每条约几十 KB（含回复原文），不必推进 git。
清单：--write-subsets 按规则从 configs/w2_splits.json 写 configs/w2_llmseg_subsets.json；--check-subsets 重算核对（不要数据）。
开发机彩排：--tiny（小型随机模型、CPU、tests/fixtures 的小数据，划分与子集按同一规则现算；输出上限再取 --cap-limit，默认 48）；
--adapter-factory MODULE:ATTR 换成测试的假适配器（只配 --tiny）。
退出码：0 = 跑完；2 = 出错（报错先写进 <out>/<模型>/errors/，argparse 用法错误也是 2）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.adapters import registry  # noqa: E402
from src.adapters.factory import default_image_policy, make_adapter  # noqa: E402
from src.data.subsets import dump_config  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.tasks import DataPaths  # noqa: E402
from src.llmseg import provider as PV  # noqa: E402
from src.llmseg import runner as R  # noqa: E402
from src.w2.data import W2_SPLITS_PATH, load_splits, tiny_paths, tiny_splits, w2_items  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 run_e1、run_w2 相同
TINY_CAP_LIMIT = 48
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
ALL = "all"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="注册键（src/adapters/registry.py；只支持 Qwen3.5 家族）")
    p.add_argument("--split", help=f"W2 的 test 划分名，或 {ALL}（5 个依次跑）")
    p.add_argument("--out", help="本次运行的目录（experiments/W2/<NNNN_slug>）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过已完成的组（身份必须相同）")
    p.add_argument("--limit", type=int, help="只取子集的前 N 句（试跑）")
    p.add_argument("--iterations", type=int, default=R.MAIN_ITERATIONS, help=f"修正轮数 T（默认 {R.MAIN_ITERATIONS}，官方主配置）")
    p.add_argument("--slots", type=int, default=150, help="同时在跑的管线数（默认 150 = 一组的条数）")
    p.add_argument("--batch-size", type=int, default=32, help="子批行数上限（默认 32）")
    p.add_argument("--max-batch-tokens", type=int, default=48000, help="行数 × 最长提示的上限（默认 48000，同 W2）")
    p.add_argument("--chunk-size", type=int, default=150, help="续跑粒度（默认 150 = 每个划分 2 组）")
    p.add_argument("--save", type=int, default=3, help="前 N 条留全部 PNG（默认 3）")
    p.add_argument("--thinking", action="store_true", help="开思考的对照（设置名 llmseg_think；要 --thinking-budget）")
    p.add_argument("--thinking-budget", type=int)
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--cap-limit", type=int, help=f"各阶段输出上限再取 min（只配 --tiny；默认 {TINY_CAP_LIMIT}）")
    p.add_argument("--adapter-factory", help="MODULE:ATTR，返回适配器（只配 --tiny）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--write-subsets", action="store_true", help="按规则写 configs/w2_llmseg_subsets.json（不要数据）")
    p.add_argument("--check-subsets", action="store_true", help="重算核对 configs/w2_llmseg_subsets.json（不要数据）")
    a = p.parse_args(argv)
    if a.write_subsets or a.check_subsets:
        return a
    for name in ("model", "split", "out"):
        if getattr(a, name) is None:
            p.error(f"--{name} 必给")
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    if registry.get_model(a.model).family != registry.FAMILY_QWEN35:
        p.error("本地提供方只支持 Qwen3.5 家族（官方 llm.py 的 Qwen-VL 坐标约定）")
    splits = R.test_splits()
    if a.split != ALL and a.split not in splits:
        p.error(f"--split 只能是 {ALL} 或 {splits}")
    for name in ("limit", "slots", "batch_size", "max_batch_tokens", "chunk_size", "thinking_budget", "cap_limit"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.iterations < 0 or a.save < 0:
        p.error("--iterations、--save 不能为负")
    if a.thinking != (a.thinking_budget is not None):
        p.error("--thinking 与 --thinking-budget 一起给")
    if not a.tiny and (a.adapter_factory or a.cap_limit is not None):
        p.error("--adapter-factory、--cap-limit 只配 --tiny")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0074_llmseg，收到 {a.out!r}")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    if a.tiny and a.cap_limit is None:
        a.cap_limit = TINY_CAP_LIMIT
    return a


def _load_attr(spec: str):
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        raise ValueError(f"应写成 MODULE:ATTR，收到 {spec!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def setting_of(a) -> str:
    s = R.SETTING_THINK if a.thinking else R.SETTING
    return s if a.iterations == R.MAIN_ITERATIONS else f"{s}_it{a.iterations}"


def subsets_cmd(a) -> int:
    if a.write_subsets:
        doc = R.build_subsets(load_splits(W2_SPLITS_PATH), splits=R.test_splits(), source=R.subsets_source())
        text = dump_config(doc)
        R.SUBSETS_PATH.write_text(text, encoding="utf-8")
        print(f"写出 {R.SUBSETS_PATH}：" + "；".join(f"{k} {v['n']} 句 {v['sha256'][:12]}…" for k, v in doc["splits"].items()))
        return 0
    problems = R.check_subsets()
    print("一致（按规则重算）" if not problems else "；".join(problems))
    return 0 if not problems else 1


def cmd(a, argv) -> int:
    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    if a.tiny:
        splits_doc = tiny_splits()
        subsets_doc = R.build_subsets(splits_doc, splits=R.test_splits(), source={"path": "tiny（src/w2/data.tiny_splits）"})
        paths = tiny_paths()
        data_files: list[Path] = []
    else:
        splits_doc = load_splits(W2_SPLITS_PATH)
        problems = R.check_subsets()
        if problems:
            raise ValueError("；".join(problems))
        subsets_doc = R.load_subsets()
        paths = DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))
        data_files = [W2_SPLITS_PATH, R.SUBSETS_PATH, *paths.config_files()]
    if a.adapter_factory:
        ad = _load_attr(a.adapter_factory)(entry)
    else:
        kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
        ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=policy, tiny=a.tiny, **kw)
    prompts = load_prompt_config(PROMPTS_PATH)
    gen = PV.GenerationConfig(decoding=prompts.decoding(), order=prompts.order, enable_thinking=a.thinking,
                              thinking_budget=a.thinking_budget, batch_size=a.batch_size, max_batch_tokens=a.max_batch_tokens,
                              cap_limit=a.cap_limit)
    pcfg = R.PipelineConfig(iterations=a.iterations)
    pol = getattr(ad, "image_policy", policy)
    splits = R.test_splits() if a.split == ALL else [a.split]
    for split in splits:
        samples = R.subset_samples(w2_items("refseg", split, paths, splits=splits_doc), subsets_doc["splits"][split]["ids"])
        if a.limit is not None:
            samples = samples[: a.limit]
        spec = R.RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                         adapter_model_key=str(getattr(ad, "model_key", entry.key)), split=split, setting=setting_of(a), tiny=a.tiny,
                         limit=a.limit, image_policy={"tag": "default", "type": type(pol).__name__, **pol.as_dict()})
        m = R.run_split(ad, spec, samples, a.out, gen=gen, pcfg=pcfg, slots=a.slots, chunk_size=a.chunk_size, splits_doc=splits_doc,
                        subsets_doc=subsets_doc, data_files=data_files, resume=a.resume, save=a.save, argv=argv)
        head = "; ".join(f"{ro}.{h['name']}={h.get('value')}" for ro, r in m["readouts"].items() for h in r["headline"])
        print(f"LLMSeg {a.model} {spec.setting}/{split} n={m['n']} complete={m['complete']} {head} "
              f"calls/image={(m['calls_per_image'] or {}).get('mean')}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    if a.write_subsets or a.check_subsets:
        return subsets_cmd(a)
    try:
        return cmd(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / a.model / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"llmseg_error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
