#!/usr/bin/env python
"""健康检查（简报 §8.11；DEV-0008）：固定的 200 题（MMLU-Pro 100、GSM8K 50、IFEval 50）上跑一个模型，输出一个 JSON。

协议见 src/health/__init__.py。题目文件 configs/health_check_items.json 已含全部评分字段，运行时不下载数据集；
加载时核对它的 sha256（protocol.ITEMS_SHA256）。

服务器（真模型，权重按注册表的固定 revision 加载）：
    export NLTK_DATA=$PROJ/nltk_data          # 事先：python -m nltk.downloader -d $PROJ/nltk_data punkt_tab
    python scripts/health_check.py --model qwen3.5-0.8b --device cuda --dtype bf16 --out <json>
先生成、后评分（punkt_tab 的许可答复之前，服务器上不装 punkt_tab；审查第 1 轮）：
    python scripts/health_check.py --model qwen3.5-0.8b --skip-ifeval-scoring --out <json>
        # 生成照常，IFEval 的原始输出照常保存，IFEval 的分数为 null 并写明原因；加载模型前不检查 punkt_tab 与 IFEval 依赖
    python scripts/health_check.py --rescore <json> --out <新 json> [--only-ifeval]
        # 不加载模型，从保存的输出重算三项（或只重算 IFEval），规则与一次跑完相同；记下评分时间、代码 commit、punkt_tab 哈希
开发机彩排（本地构造的小型随机模型，CPU；数值无意义，只验证全流程；--model 选家族，不给即 Qwen3.5）：
    python scripts/health_check.py --tiny --model gemma-4-e2b-it --limit 3 --out /tmp/health_tiny.json
比较两次运行（门报告用；标题读数合并后比基座低超过 5 个百分点的项标出，并列给出配对 bootstrap CI 与符号检验 p 值；
另列截断率提示）：
    python scripts/health_check.py --compare <基座 json> <合并后 json> [--out <json>]

协议 v2（D28-7，DEV-0013）：MMLU-Pro 生成到 256 个 token，标题读数 mmlu_pro 取全部，诊断读数 mmlu_pro_fmt32 取同一次
生成的前 32 个 token（不标红）。v1 的输出（RUN-0007/0009）照旧可读、可 --rescore、可 --compare；与 v2 比较时 MMLU-Pro
只比 v1 的 32 对 v2 的 fmt32，输出里写明"口径不同、标题不可比"。算力机上可加 --audit-fmt32 N：MMLU-Pro 前 N 题再真的用
max_new_tokens=32 各生成一次，与截出的读数逐字段比较（GPU bf16 上"前缀相同"的运行期核对；不计入记账，不改分数）。

退出码：0 = 完成（分数高低不影响）；2 = 出错（运行模式下已有结果与报错先写进 --out 再退出；重评模式写出报错）；
argparse 用法错误也是 2。--limit N：每项只跑前 N 题（调试用；这样的结果不能拿去比较）。
真模型运行（以及对真模型输出的重评）要求 punkt_tab/english 与 protocol.PUNKT_TAB_ENGLISH_SHA256 逐个相同；--tiny 只记录。

适配器经 src/adapters/factory.make_adapter 构造（DEV-0011；按家族分派只在那一处）。纯文本输入不用图像，
构造适配器必须给的视觉参数取冒烟脚本的同一组值（vision_policy），只记录，不影响结果。
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

import torch  # noqa: E402

from src import runlog  # noqa: E402
from src.adapters import factory, registry  # noqa: E402
from src.adapters.image_policy import FixedSize, SoftTokenBudget, TilePolicy  # noqa: E402
from src.health import protocol as P  # noqa: E402
from src.health.items import load_items  # noqa: E402
from src.health.run import audit_prefix_readings, rescore_records, run_items  # noqa: E402
from src.health.scoring import IFEVAL_SKIP_REASON, compare, require_punkt, summarize  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}  # 与 scripts/smoke_adapter.py 的 --dtype 取值相同
QWEN_IMAGE_SIZE = (480, 640)  # 冒烟脚本 --image-size 的默认值
IFEVAL_PACKAGES = ("nltk", "langdetect", "immutabledict")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="注册键，如 qwen3.5-0.8b（见 src/adapters/registry.py）；--tiny 时选家族与小模型 config 的来源")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--out", help="输出 JSON 路径（运行与重评时必填；--compare 时可选）")
    p.add_argument("--tiny", action="store_true", help="用本地构造的小型随机模型在 CPU 上彩排")
    p.add_argument("--limit", type=int, help="每项只跑前 N 题（调试用）")
    p.add_argument("--skip-ifeval-scoring", action="store_true",
                   help="IFEval 只生成、不评分（分数为 null 并写明原因）；加载模型前不检查 punkt_tab 与 IFEval 依赖")
    p.add_argument("--rescore", metavar="IN_JSON", help="不加载模型，从已保存的输出重算分数，写到 --out")
    p.add_argument("--only-ifeval", action="store_true", help="与 --rescore 一起用：只重算 IFEval，其余两项原样保留")
    p.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "MERGED_JSON"), help="比较两次运行，不跑模型")
    p.add_argument("--audit-fmt32", type=int, metavar="N",
                   help="跑完后 MMLU-Pro 前 N 题再真的用 max_new_tokens=32 各生成一次，与截出的 mmlu_pro_fmt32 逐字段比较"
                        "（不计入记账，不改分数）")
    args = p.parse_args(argv)
    run_only = {"--model": args.model, "--device": args.device, "--dtype": args.dtype, "--tiny": args.tiny,
                "--limit": args.limit, "--skip-ifeval-scoring": args.skip_ifeval_scoring,
                "--audit-fmt32": args.audit_fmt32}
    if args.compare and args.rescore:
        p.error("--compare 与 --rescore 不能同时用")
    if args.only_ifeval and not args.rescore:
        p.error("--only-ifeval 只能与 --rescore 一起用")
    if args.compare or args.rescore:
        used = [k for k, v in run_only.items() if v not in (None, False)]
        if used:
            p.error(f"{'--compare' if args.compare else '--rescore'} 不跑模型，不能与 {used} 一起用")
        if args.rescore:
            if not args.out:
                p.error("--rescore 需要 --out")
            if Path(args.out).resolve() == Path(args.rescore).resolve():
                p.error("--out 不能覆盖 --rescore 的输入文件")
        return args
    if not args.out:
        p.error("运行需要 --out")
    if args.limit is not None and args.limit < 1:
        p.error("--limit 必须是正整数")
    if args.audit_fmt32 is not None and args.audit_fmt32 < 1:
        p.error("--audit-fmt32 必须是正整数")
    if args.tiny:
        args.device = args.device or "cpu"
        args.dtype = args.dtype or "fp32"
    else:
        if not args.model:
            p.error("真模型运行需要 --model")
        args.device = args.device or "cuda"
        args.dtype = args.dtype or "bf16"
    return args


def vision_policy(family: str, *, tiny: bool):
    """构造适配器必须给的图像口径（纯文本输入用不到，只记录）：取冒烟脚本的同一组值。"""
    if family == registry.FAMILY_QWEN35:
        return FixedSize(*QWEN_IMAGE_SIZE)
    if family == registry.FAMILY_GEMMA4:
        from src.adapters import gemma4

        return SoftTokenBudget(gemma4.TINY_MAX_SOFT_TOKENS if tiny else gemma4.SMOKE_MAX_SOFT_TOKENS)
    if family == registry.FAMILY_INTERNVL35:
        from src.adapters import internvl35

        return TilePolicy(**internvl35.SMOKE_POLICY)
    raise ValueError(f"健康检查还不支持 {family} 家族")


def make_adapter(args):
    """构造适配器（经 src/adapters/factory.make_adapter，DEV-0011）。返回 (adapter, 注册项, 构造时给的视觉参数)。"""
    key = args.model or registry.keys_for_family(registry.FAMILY_QWEN35)[0]
    entry = registry.get_model(key)
    policy = vision_policy(entry.family, tiny=args.tiny)
    ad = factory.make_adapter(key, device=args.device, dtype=DTYPES[args.dtype], image_policy=policy, tiny=args.tiny)
    return ad, entry, policy.as_dict()


def run_record(config: dict, items_path: Path) -> dict:
    """src/runlog.py 的运行记录（config、种子、git commit、题目文件 sha256、主机、包版本……），嵌进输出 JSON。"""
    with tempfile.TemporaryDirectory() as d:
        meta = runlog.write_run_metadata(d, config=config, seed=P.SEED, data_files=[items_path], repo_root=REPO_ROOT)
    return {"config": config, "meta": meta, "packages_ifeval": runlog.package_versions(IFEVAL_PACKAGES)}


def check_ifeval_ready(*, pinned: bool) -> dict:
    """IFEval 评分之前：依赖能 import（langdetect、nltk、immutabledict），punkt_tab 在且（pinned 时）哈希相符。"""
    from src.health.ifeval import evaluation_lib  # noqa: F401

    return require_punkt(pinned=pinned)


def scoring_block(*, mode: str, benchmarks, punkt: dict | None, source: dict | None = None) -> dict:
    """这次评分的来历：何时、哪个代码 commit、评了哪几项；IFEval 是否评分、punkt_tab 路径与哈希、langdetect 种子。"""
    ifeval_scored = "ifeval" in benchmarks
    return {
        "mode": mode,
        "scored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code": runlog.git_info(REPO_ROOT),
        "benchmarks_scored": list(benchmarks),
        "ifeval": {
            "scored": ifeval_scored,
            "skip_reason": None if ifeval_scored or mode != "inline" else IFEVAL_SKIP_REASON,
            "punkt_tab": punkt,
            "langdetect_seed": P.LANGDETECT_SEED if ifeval_scored else None,
            "packages": runlog.package_versions(IFEVAL_PACKAGES),
        },
        "source": source,
    }


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _print_summary(report: dict, out_path: Path) -> None:
    layout = P.record_readings(report["protocol_version"])
    for key, reading in layout.items():
        note = "  (未评分)" if key in report["unscored"] else ""
        metric = P.HEADLINE[reading.benchmark]
        kind = "诊断" if key not in report["headline"] else "标题"
        diag = report["diagnostics"][key]
        print(f"{key:15s} {metric:20s} {report['scores'][key][metric]['pct']}  [{kind}, max_new_tokens="
              f"{reading.max_new_tokens}]  (n={diag['n']}, 写满 {diag['n_reached_max_new_tokens']}){note}")
    for name, t in report["parse_tally"].items():
        print(f"parse {name:21s} n={t['n']} failed={t['n_failed']} {t['failures']}")
    total = report["compute"]["total"]
    print(f"compute: generated={total['generated_tokens']} prompt={total['prompt_tokens']} "
          f"call_time={total['call_time_s']:.1f}s  -> {out_path}")


def main_compare(args) -> int:
    res = compare(*args.compare)
    print(f"protocol: base v{res['protocol_versions']['base']}, merged v{res['protocol_versions']['merged']}")
    for r in res["rows"]:
        head = f"{r['reading']:15s} {r['metric']:25s}"
        if not r["comparable"]:
            print(f"{head} 口径不同、标题不可比（见 notes）")
            continue
        if not r["scored"]:
            print(f"{head} 未评分（{r['reading']} 在 unscored 里）")
            continue
        mark = "  <-- 标红" if r["flag"] else ("  [诊断，不标红]" if r["diagnostic"] else "")
        delta = "n/a" if r["delta_pp"] is None else f"{r['delta_pp']:+.1f}"
        ci = "n/a" if r["ci95_pp"] is None else f"[{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]"
        st = r["sign_test"]
        print(f"{head} base={r['base_pct']} merged={r['merged_pct']} "
              f"delta={delta} pp  CI95={ci}  sign test {st['n_better']}+/{st['n_worse']}- p={st['p_value']:.3g}  "
              f"(lost {r['lost']}, gained {r['gained']}, n={r['n']}){mark}")
    for t in res["truncation"]["rows"]:
        mark = "  <-- 截断率提示" if t["notice"] else ""
        delta = "n/a" if t["delta_pp"] is None else f"{t['delta_pp']:+.1f}"
        print(f"truncation {t['reading']:15s} max_new_tokens={t['max_new_tokens']:<5d} base={t['base_pct']}% "
              f"merged={t['merged_pct']}% delta={delta} pp (n={t['n']}){mark}")
    for note in res["notes"]:
        print(f"note: {note}")
    print(f"any_flag={res['any_flag']}  flagged={res['flagged']}  truncation_notices={res['truncation']['notices']}  "
          f"incomparable={res['incomparable']}  unscored={res['unscored']}  partial_run={res['partial_run']}")
    if args.out:
        _write(Path(args.out), res)
    return 0


def main_rescore(args) -> int:
    src_path, out_path = Path(args.rescore), Path(args.out)
    benchmarks = ("ifeval",) if args.only_ifeval else P.BENCHMARKS
    try:
        raw = src_path.read_bytes()
        report = json.loads(raw.decode("utf-8"))
        if report.get("ok") is not True:
            raise ValueError(f"{src_path} 不是一次跑完的运行（ok={report.get('ok')!r}）")
        # v1 的输出（RUN-0007/0009）按 v1 的布局重评，输出仍记 v1（生成是按 v1 做的）
        version = P.check_protocol_version(report.get("protocol_version"))
        items_obj, items_sha = load_items()
        if report["items_file"]["sha256"] != items_sha:
            raise ValueError("输入用的题目文件与当前题目文件的 sha256 不同")
        punkt = check_ifeval_ready(pinned=report["mode"] == "real") if "ifeval" in benchmarks else None
        records = rescore_records(report["records"], items_obj, benchmarks=benchmarks, protocol_version=version)
    except Exception as e:  # noqa: BLE001 — 输入文件原样不动，报错写进 --out
        _write(out_path, {"script": "scripts/health_check.py", "rescore_of": str(src_path), "ok": False,
                          "error": repr(e), "traceback": traceback.format_exc()[-4000:]})
        print(f"重评失败：{e!r}（已写出 {out_path}）", file=sys.stderr)
        return 2
    out = dict(report)
    out["records"] = records
    out.update(summarize(records, protocol_version=version))
    out["scoring"] = scoring_block(mode="rescore", benchmarks=benchmarks, punkt=punkt,
                                   source={"path": str(src_path), "sha256": hashlib.sha256(raw).hexdigest()})
    out["scoring_history"] = list(report.get("scoring_history", [])) + [report.get("scoring")]
    _write(out_path, out)
    _print_summary(out, out_path)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.compare:
        return main_compare(args)
    if args.rescore:
        return main_rescore(args)
    out_path = Path(args.out)
    report: dict = {
        "script": "scripts/health_check.py",
        "argv": sys.argv[1:] if argv is None else list(argv),
        "mode": "tiny" if args.tiny else "real",
        "protocol_version": P.PROTOCOL_VERSION,
        "limit": args.limit,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "ok": False,
    }
    state: dict = {}
    t_all = time.perf_counter()
    score_ifeval = not args.skip_ifeval_scoring
    try:
        items_obj, items_sha = load_items()
        report["items_file"] = {"path": P.ITEMS_PATH, "sha256": items_sha}
        # 评 IFEval 时，依赖与 punkt_tab 在加载模型之前核对（缺了立即失败）；--skip-ifeval-scoring 时两者都不碰
        punkt = check_ifeval_ready(pinned=not args.tiny) if score_ifeval else None
        config = {"script": "scripts/health_check.py", "model": args.model, "tiny": args.tiny, "device": args.device,
                  "dtype": args.dtype, "limit": args.limit, "skip_ifeval_scoring": args.skip_ifeval_scoring,
                  "audit_fmt32": args.audit_fmt32, "protocol_version": P.PROTOCOL_VERSION, "seed": P.SEED}
        report["run"] = run_record(config, REPO_ROOT / P.ITEMS_PATH)
        ad, entry, vision = make_adapter(args)
        report["model"] = {
            "key": entry.key, "family": entry.family, "repo": entry.repo, "revision": entry.revision,
            "adapter_model_key": ad.model_key, "device": str(ad.device), "dtype": str(ad.dtype),
            "num_layers": ad.num_layers(), "hidden_size": ad.hidden_size(),
            "vision_settings_unused": vision,
            "model_generation_config": ad.model.generation_config.to_diff_dict(),
        }
        report["generation"] = {"order": P.ORDER, "n_images": 0, "enable_thinking": P.ENABLE_THINKING,
                                "greedy_kwargs": dict(P.GREEDY_KWARGS), "max_new_tokens": dict(P.MAX_NEW_TOKENS),
                                "readings": {r.name: {"benchmark": r.benchmark, "max_new_tokens": r.max_new_tokens,
                                                      "role": r.role, "prefix_of": r.prefix_of} for r in P.READINGS}}
        _log(f"模型 {entry.key}（{report['mode']}，{ad.device}，{ad.dtype}）已就绪，开始跑题"
             + ("；IFEval 只生成、不评分" if not score_ifeval else ""))
        run_items(ad, items_obj, limit=args.limit, score_ifeval=score_ifeval, state=state, progress=_log)
        if args.audit_fmt32:
            _log(f"前缀读数审计：MMLU-Pro 前 {args.audit_fmt32} 题再用 max_new_tokens={P.FMT32_MAX_NEW_TOKENS} 各生成一次")
            state["prefix_audit"] = audit_prefix_readings(ad, items_obj, state["records"], n=args.audit_fmt32)
        report.update(state)
        report.update(summarize(state["records"]))
        scored = [b for b in P.BENCHMARKS if score_ifeval or b != "ifeval"]
        report["scoring"] = scoring_block(mode="inline", benchmarks=scored, punkt=punkt)
        report["ok"] = True
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        report.update(state)
        report.update(error=repr(e), traceback=traceback.format_exc()[-4000:])
        report["wall_time_s"] = round(time.perf_counter() - t_all, 2)
        _write(out_path, report)
        print(f"失败：{e!r}（已写出 {out_path}）", file=sys.stderr)
        return 2
    report["wall_time_s"] = round(time.perf_counter() - t_all, 2)
    _write(out_path, report)
    _print_summary(report, out_path)
    for name, a in report.get("prefix_audit", {}).items():
        print(f"prefix audit {name}: {a['n_identical']}/{a['n']} identical to a real max_new_tokens={a['max_new_tokens']} "
              f"generation; mismatched ids {[x['id'] for x in a['mismatches']]}")
    print(f"wall={report['wall_time_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
