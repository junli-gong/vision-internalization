#!/usr/bin/env python
"""Rex-Omni 外部参照（DEV-0083；D165 第四节、D177-c）：只做检测，每个"图像 × 类别"对调用一次，用 X-M 的 F1 计分。规则见
src/rexomni/（protocol.py：官方的提示、预处理、生成、解析与出处；model.py：权重、分词器、处理器、生成；runner.py：叶目录与续跑）。

硬约束第 1 条：Rex-Omni 是学习型检测器，只作单独运行的对照系统；输出绝不进入我们的训练数据、标签、奖励或任何挑选
（tests/test_rexomni_isolation.py）。

子命令：
    weights --download     按 configs/rexomni_weights.json 钉住的 revision 下载（HF_HUB_CACHE；慢时加 HF_ENDPOINT=https://hf-mirror.com）
                           并逐文件核对字节数与 sha256，打印快照目录
    weights --verify       只核对（离线）
    run                    跑一个划分，每个设置输出 <out>/rex-omni/xm/<设置>/<划分>/（同 X-M 的叶目录格式，每对一行）

设置（run --setting，缺省两个都跑，同一个模型只加载一次）：
- rexomni_det_joint：Rex-Omni 自己的协议（每图一次，提示里列出该图本划分里的全部正类别，category_id 升序），论文里外部参照的标题读数；
- rexomni_det：每个（图, 类别）对一次、只问这一个类别，与 X-M 同协议的诊断。

划分（run --split）：
- xm_tune：冒烟，只与论文里 COCO 的量级对一下（不调任何设置）；不要 gate；
- xm_final、lvis_sub：要 --xm-gate <gate_xm.json>，走 src/xm/gate.check_xm_gate 的基座那条路（不比检查点的 manifest，同 run_xm 的 base_*）；
- xm_decision（密封，G-XM 已判完）、reasonseg_val、train：拒绝。

服务器（HPC；$VI/data 见 DEV-0079；--out 的末级目录名以四位运行编号开头且含 _rexomni）：
    HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_ENDPOINT=https://hf-mirror.com $VI/env/bin/python scripts/run_rexomni.py weights --download
    HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1 $VI/env/bin/python scripts/run_rexomni.py run --split xm_tune --data-root $VI/data \\
        --out experiments/XM/<NNNN>_rexomni_<node>/<NNNN>_rexomni_tune [--resume]          # 两个设置；退出码 3 = 没过冒烟线
    ... run --split xm_final --xm-gate <gate_xm.json> --data-root $VI/data --out experiments/XM/<NNNN>_rexomni_<node>/<NNNN>_rexomni_final
开发机彩排（--tiny：小型随机 Qwen2.5-VL、CPU；数据是 tests/_xm_run_fakes.write_fixture 的 pairs.json，或 DEV-0079 的假数据根）：
    python scripts/run_rexomni.py run --tiny --split xm_tune --pairs-json <dir>/pairs.json --out <dir>/0001_rexomni_tiny
冒烟线：xm_tune 上 joint 的 f1_rexomni_exact 低于 0.60 就先停，不跑 xm_final（管理员，DEV-0083 审阅意见第 2 条）。跑完会打印
"冒烟线 … 通过 / 不通过"，并以退出码 3 结束；也可以直接查叶目录的 metrics.json：
    python -c "import json,sys; s=json.load(open(sys.argv[1]))['smoke_line']; print(s); sys.exit(0 if s['pass'] else 3)" \
        <out>/rex-omni/xm/rexomni_det_joint/xm_tune/metrics.json
退出码：0 = 跑完；2 = 用法、gate 或数据不合格（什么都不写），或运行出错（报错先写进 <out>/rex-omni/errors/）；3 = 跑完了，但 xm_tune 上
joint 没过冒烟线（--tiny 不拦）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.rexomni import protocol as P  # noqa: E402
from src.rexomni import runner as RR  # noqa: E402
from src.xm import data as XD  # noqa: E402
from src.xm import gate as XG  # noqa: E402

SPLIT_CHOICES = (*RR.SPLITS, *RR.REFUSED_SPLITS)
DEFAULT_SETTINGS = (RR.SETTING_JOINT, RR.SETTING_PER_CATEGORY)  # joint 是标题读数，先跑
SMOKE_FAIL_RC = 3  # xm_tune 上 joint 没过冒烟线（src/rexomni/runner.smoke_line）
DTYPES = ("bfloat16", "float32")


def check_gates(a) -> dict | None:
    """划分与 gate（parse_args 与 cmd_run 共用）：不合格抛 RexOmniRunError / XMGateError。返回 check_xm_gate 的结果或 None。"""
    RR.check_split(a.split)
    if a.split in XG.GATED:
        return XG.check_xm_gate(a.xm_gate, checkpoint=None)  # 基座那条路：verdict、rehearsal、checks，不比 manifest
    if a.xm_gate is not None:
        raise XG.XMGateError(f"--xm-gate 只配 {list(XG.GATED)}（--split {a.split} 不要）")
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("weights", help="下载 / 核对权重（configs/rexomni_weights.json）")
    g = w.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", action="store_true")
    g.add_argument("--verify", action="store_true")
    w.add_argument("--sizes-only", action="store_true", help="只核对字节数，不算 sha256")
    r = sub.add_parser("run", help="跑一个划分")
    r.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    r.add_argument("--out", required=True, help="本次运行的目录（末级目录名：四位运行编号开头且含 _rexomni）")
    r.add_argument("--setting", nargs="+", choices=list(RR.SETTINGS), default=list(DEFAULT_SETTINGS),
                   help="缺省两个都跑（先 rexomni_det_joint：Rex-Omni 自己的协议，标题读数；再 rexomni_det：与 X-M 同协议的诊断）")
    r.add_argument("--xm-gate", metavar="GATE_XM_JSON", help="xm_final、lvis_sub：G-XM 的 gate_xm.json（RUN-0108）")
    r.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--pairs-json 时取它所在的目录）")
    r.add_argument("--splits-file", help="划分清单（默认 configs/xm_splits.json）")
    r.add_argument("--pairs-json", help="只配 --tiny：对的 JSON（tests/_xm_run_fakes.write_fixture 写）")
    r.add_argument("--model-dir", help="权重目录（默认按清单从 HF 缓存取快照，离线）")
    r.add_argument("--sizes-only", action="store_true", help="权重只核对字节数，不算 sha256（默认都核对）")
    r.add_argument("--limit", type=int, help="只取划分的前 N 个对（调试）")
    r.add_argument("--batch-size", type=int, default=1, help="一批的对数（默认 1：同官方评测每次一个请求；进运行身份）")
    r.add_argument("--resume", action="store_true", help="续跑（身份必须相同）")
    r.add_argument("--tiny", action="store_true", help="小型随机 Qwen2.5-VL + CPU（开发机彩排）")
    r.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    r.add_argument("--dtype", choices=DTYPES, help="默认：真模型 bfloat16（官方），--tiny 时 float32")
    a = p.parse_args(argv)
    if a.cmd == "weights":
        return a
    if not XD.RUN_DIR_RE.fullmatch(Path(a.out).name) or RR.RUN_DIR_TAG not in Path(a.out).name:
        p.error(f"--out 的末级目录名必须以四位运行编号开头且含 {RR.RUN_DIR_TAG}（形如 0110_rexomni_tune），收到 {a.out!r}")
    if len(set(a.setting)) != len(a.setting):
        p.error(f"--setting 重复：{a.setting}")
    for name in ("limit", "batch_size"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.pairs_json is not None and not a.tiny:
        p.error("--pairs-json 只配 --tiny（真运行经 src/xm/data.load_pairs）")
    if a.pairs_json is not None and (a.data_root is not None or a.splits_file is not None):
        p.error("--pairs-json 不与 --data-root、--splits-file 同给")
    if a.tiny and a.pairs_json is None and a.data_root is None:
        p.error("--tiny 要给 --pairs-json，或 --data-root（加 --splits-file：tests/_xm_fakes 的假数据根）")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if a.tiny and a.model_dir is not None:
        p.error("--model-dir 不配 --tiny")
    try:
        check_gates(a)
    except (RR.RexOmniRunError, XG.XMGateError) as e:
        p.error(str(e))
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("float32" if a.tiny else P.TORCH_DTYPE)
    return a


def data_root(a) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if a.pairs_json is not None:
        return Path(a.pairs_json).resolve().parent
    return Path(os.environ["VISINT_DATA_ROOT"])


def load_split_pairs(a, gate: dict | None) -> list:
    if a.pairs_json is not None:
        pairs = [XD.pair_from_dict(d) for d in json.loads(Path(a.pairs_json).read_text(encoding="utf-8"))]
    else:
        pairs = XD.load_pairs(a.split, data_root(a), xm_gate=None if gate is None else gate["pass"], splits_path=a.splits_file)
    wrong = sorted({q.split for q in pairs} - {a.split})
    if wrong:
        raise RR.RexOmniRunError(f"对的 split {wrong} 不是 --split {a.split}")
    return pairs[: a.limit] if a.limit is not None else pairs


def weights_cmd(a) -> int:
    from src.rexomni import model as RM

    doc = RM.load_weights_manifest()
    snap = None
    try:
        snap = RM.snapshot_dir(doc, download=a.download)
        res = {**RM.verify_weights(snap, doc, full_hash=not a.sizes_only), "ok": True}
    except Exception as err:  # noqa: BLE001 — 下载失败（网络、离线缓存里没有）与核对不过都按"不合格"报
        res = {"dir": None if snap is None else str(snap), "ok": False, "error": f"{type(err).__name__}: {err}"}
    print(json.dumps({**res, "total_gb": round(RM.weights_total_bytes(doc) / 1e9, 3)}, indent=1, ensure_ascii=False))
    return 0 if res["ok"] else 1


def load_engine(a):
    from src.rexomni import model as RM

    if a.tiny:
        return RM.build_tiny(device=a.device, dtype=a.dtype)
    doc = RM.load_weights_manifest()
    d = Path(a.model_dir) if a.model_dir else RM.snapshot_dir(doc, download=False)
    weights = RM.verify_weights(d, doc, full_hash=not a.sizes_only)
    return RM.load_real(d, device=a.device, dtype=a.dtype, weights=weights)


def run_cmd(a, argv) -> int:
    gate = check_gates(a)
    pairs = load_split_pairs(a, gate)
    engine = load_engine(a)
    gid = None if gate is None else gate["identity"]
    if gate is not None:
        XG.append_log(a.out, RR.SYSTEM, XG.UNLOCK_LOG, {**gid, "gate": gate["summary"], "resolved": str(Path(a.xm_gate).resolve()),
                                                       "unlocked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                                       "split": a.split, "argv": list(argv)})
    files = [f for f in (a.pairs_json, a.splits_file or (None if a.pairs_json else XD.XM_SPLITS_PATH)) if f is not None and Path(f).is_file()]
    rc = 0
    for setting in a.setting:
        m = RR.run_split(engine, pairs, a.out, split=a.split, setting=setting, data_root=data_root(a), batch_size=a.batch_size,
                         resume=a.resume, limit=a.limit, xm_gate=gid, data_files=files, tiny=a.tiny, argv=argv)
        d, s = m["det"], m["summary"]
        print(f"Rex-Omni {setting} {a.split} n={m['n']} calls={s['n_calls']} det f1_rexomni {d['f1_rexomni']['value']} "
              f"f1_rexomni_exact {d['f1_rexomni_exact']['value']} f1_micro {d['f1_micro']['value']} parse_failures {s['n_parse_failures']} "
              f"truncated {s['n_truncated']} unasked {s['n_unasked_category']}")
        sl = m["smoke_line"]
        if sl is not None:
            verdict = "通过" if sl["pass"] else ("不通过：先停，不跑 xm_final" if not a.tiny else "不通过（--tiny 只记不拦）")
            print(f"冒烟线 {sl['statistic']} = {sl['value']}（门槛 {sl['threshold']}）：{verdict}")
            if not sl["pass"] and not a.tiny:
                rc = SMOKE_FAIL_RC
    return rc


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    if a.cmd == "weights":
        return weights_cmd(a)
    try:
        return run_cmd(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / RR.SYSTEM / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]}, indent=1, ensure_ascii=False)
                        + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
