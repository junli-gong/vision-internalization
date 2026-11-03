#!/usr/bin/env python
"""X-M 的去留判定（G-XM；DEV-0081）：照录 D165 第五节（预注册 XM-5）的四条，规则与阈值写死、不留参数；写于任何 X-M 训练与评测之前。
F1 是 Rex-Omni 的宏平均 F1（预注册 XM-3a：src/xm/metrics 的主读数 PRIMARY = f1_rexomni，按类别合计 TP / FP / FN 得 P_c、R_c，在类别上
宏平均，F1 = 2PR / (P + R)）。

    python scripts/gate_xm.py --gate G-XM --data-root $VI/data --xm-checkpoint <X-M 的训练叶目录>/ckpt/final/model \\
        --xm-main      <X-M 的 run_xm 输出>/qwen3.5-9b/xm/xm_spline_trimap/xm_decision \\
        --xm-box-grabcut <同一次输出>/qwen3.5-9b/xm/xm_box_grabcut/xm_decision \\
        --base-det     <基座的 run_xm 输出>/qwen3.5-9b/xm/base_det/xm_decision \\
        --xm-refseg  <X-M 的 run_x2 输出>/qwen3.5-9b/refseg/x2_spline_trimap/<2,300 句的 5 个划分> \\
        --x2-refseg  <X-2 seed 0 的 run_x2 输出>/qwen3.5-9b/refseg/x2_spline_trimap/<同样 5 个划分> \\
        --lb-xm <X-M 的 scripts/lang_battery.py 运行目录> --lb-base <同节点基座的运行目录（07 上 RUN-0084 的 lb_base）> \\
        --lb-x2 <X-2 seed 0 的运行目录（只给 D165 原文的诊断）> [--lb-compare <--compare 的输出 JSON，只作诊断>] \\
        --out-dir <目录> [--rehearsal] [--criterion4 d170]     # 写 <目录>/gate_xm.json 与 <目录>/gate_xm.md

四条（第 1–3 条是 D165 第五节原文 RULES；第 4 条按预注册 XM-5a（D170）的改写 RULE_4_D170；全部满足才"通过"）：
1. 实例分割 F1（mask IoU 0.5 的 Rex-Omni 宏平均 F1）：X-M 主设置 xm_spline_trimap − xm_box_grabcut，判定集上按图像重抽的配对 bootstrap
   （src/xm/metrics.paired_bootstrap），95% CI 下界 > 0；
2. 检测 F1（框 IoU 0.5 的宏平均 F1；X-M 的框取 --xm-main 叶目录里调用 1 的检测计数，各 xm_* 设置相同）：X-M − base_det，同上，下界 > 0；
3. cIoU（Σinter / Σunion）：2,300 句开发数据（explore_val 三层 400 / 400 / 200、devtune800、devcheck2）上 X-M − X-2 seed 0，逐句配对 bootstrap
   （口径同 scripts/mgr/x_test_table.paired：句子按 id 排序，src.stats 的重抽下标），95% CI 下界 > −0.02；
4. **d170（XM-5a，进结论，--criterion4 的缺省与唯一取值）**：MMLU-Pro（mmlu_pro_letter 的 accuracy，2,000 题）与 IFEval
   （prompt_level_strict，541 条），X-M − **同节点基座**（--lb-base）：两次 lang_battery 运行的逐题结果（scripts/lang_battery.py 的
   load_run 与逐题计分 item_counts，与它的 --compare 同一套读法与核对）逐题配对，差为百分点、不取整，配对 bootstrap 95%、10,000 次、
   种子 src/constants.BOOTSTRAP_SEED；CI 下界 MMLU-Pro > −5.7、IFEval > −8.1（D170_THRESHOLDS：X-2 三个种子相对各自同节点基座的最低
   下界 −3.7、−6.1 再减 2，写死）。
   **d165（D165 原文，诊断，不入结论）**：X-M − X-2 seed 0（--lb-x2），同样的逐题配对，下界都 > −2；总是照算，写进 diagnostics 与
   门报告的单独一行（"若计入"的结果）。lb_compare（--compare 的输出，1,000 次、取整到 0.1）给了也只作诊断。
(1)–(4) 都是 95%、10,000 次、种子 src/constants.BOOTSTRAP_SEED。

先核对（不满足就拒绝判定，退出码 2，什么都不写）：--gate 恰是 G-XM（判定集的读取经 src/xm/data.load_pairs 的守卫）；所有输入都在、都写完；
三个判定集叶目录的设置与划分对、对的集合恰是 xm_decision 的全部对、没有重复；--xm-checkpoint 的 manifest sha256 = X-M 的两个判定集
叶目录、X-M 的指代分割叶目录与 --lb-xm 运行的检查点，base_det 与 --lb-base 不带检查点；X-2 一边（指代分割叶目录与 --lb-x2 运行）是
同一个、不同于 X-M 的检查点；指代分割两边的句子集合相同、划分都在 2,300 句的 5 个里；三次 lang_battery 运行的题目与数据相同（同 --compare）。
判定点核对（checks；不全为真时照算，但结论标为"彩排，不作数"，src/xm/data.check_xm_gate 不放行）：
- 同节点（节点 = 各叶目录 / 运行目录的运行身份 meta.json 的 hostname，src/runlog.write_run_metadata 写的；lang_battery 另可取
  lang_battery.json 的 run.meta.hostname）：(a) decision_same_node：三个判定集叶目录同一节点；(b) refseg_same_node：X-M 与 X-2 的指代分割
  叶目录同一节点；(c) lb_same_node：第 4 条配对的 X-M 与基座两次 lang_battery 运行同一节点（D165 原文的诊断另记 same_node，不进 checks）。
  计划：判定集在 06，X-M 的指代分割（2,300 句）与大题量在 07，与 RUN-0080、RUN-0084（lb_base 与 X-2 seed 0）配对。
- prompt_version_matches：X-M 检查点训练时记的提示版本（训练叶目录 config.yaml 的 xm.prompts.version）= 三个判定集叶目录记的版本
  （叶目录 config.yaml 的 prompts.version，src/xm/runner 写的 src/xm/prompts.identity()）；decision_prompts_frozen：三个叶目录记的
  prompts.frozen 都是 true（D168）。
- 其余：判定集 500 张图、用的是入库的 configs/xm_splits.json（没给 --splits）；判定集叶目录都由 --gate G-XM 放行；都不是 --tiny、没有
  --limit；同一个模型键；指代分割恰是 5 个划分 2,300 句；三次 lang_battery 都不是 --limit、MMLU-Pro 2,000 题、IFEval 541 条；X-M 的检查点
  来自 configs/xm_train.json、X-2 的来自 configs/x2_train.json，都是种子 0。
--rehearsal 时同样标"彩排，不作数"。
gate_xm.json 的键（src/xm/data.check_xm_gate 与 scripts/run_xm.py 读，XM_GATE_KEYS）：verdict（通过时恰是 src/xm/data.XM_GATE_PASS
"通过"；不通过是"不通过"；彩排时以"彩排，不作数"开头）、rehearsal（bool）、checks（判定点核对，全部为真才算判定点）、
candidate.manifest_sha256（X-M 检查点的 manifest sha256）；另有 decision、pass、criteria（第 4 条是 d170）、diagnostics（criterion4_d165：
D165 原文，counts_toward_verdict = false；lb_compare）、nodes、prompt_versions、inputs（输入文件的 sha256）。

**用到 src/xm/metrics（DEV-0080）的只有这几个名字**（经 `from src.xm import metrics as M`）：M.table(rows, kind) → Table、
M.point(table, statistic) → float | None、M.paired_bootstrap(a, b, statistic, *, n_resamples, level, seed) → src.stats.CI | None、
M.PRIMARY（= "f1_rexomni"）。rows 是 predictions.jsonl 的行原样（pair_rows 按 id 排序后给，两个系统同一顺序）。不用 src/xm/prompts。
**叶目录**（src/xm/runner.derive_settings 写的 <out>/<模型>/xm/<设置>/<划分>/）：predictions.jsonl（每对一行：id、image_key、
category_id、det 与 seg = src/xm/metrics.eval_pair 的逐阈值计数，base_det 的 seg 是 null）、metrics.json（complete）、config.yaml
（model.key、model.tiny；checkpoint 为 null 或 {manifest_sha256, training: {config_sha256, seed}, xm: {prompts_version}}；xm 一节有
setting、split、limit、unseal.gate；prompts 一节 = src/xm/prompts.identity()：version、frozen……）、meta.json（hostname）。
退出码：0 = 判完（通过与否不影响）；2 = 不能判定。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import constants as C  # noqa: E402
from src import stats as S  # noqa: E402
from src.e3.train import checkpoint_leaf  # noqa: E402
from src.stats import ratio_of_sums  # noqa: E402
from src.w2.runner import read_identity as read_w2_identity  # noqa: E402
from src.xm import data as XD  # noqa: E402
from src.xm import metrics as M  # noqa: E402  （只用 M.table、M.point、M.paired_bootstrap、M.PRIMARY）


def _load_lang_battery():
    """scripts/lang_battery.py（load_run、_identity、item_counts 与读数名：与它的 --compare 同一套读法）。scripts 不是包，按路径加载。"""
    spec = importlib.util.spec_from_file_location("_lang_battery_for_gate_xm", REPO_ROOT / "scripts" / "lang_battery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LB = _load_lang_battery()

# D165 第五节（负责人原文；预注册 XM-5；不得修改）
RULES: Mapping[str, str] = {
    "1": "描轮廓仍然关键：实例分割 F1，X-M 显著高于\"X-M 的框直接交给 GrabCut\"，即差值的 CI 下界 > 0。",
    "2": "成为检测器：检测 F1，X-M 显著高于基座零样本，即差值的 CI 下界 > 0。",
    "3": "指代分割不退步：在 2,300 句开发数据上，X-M 的 cIoU 不劣于 X-2，即差值的 CI 下界 > −0.02。",
    "4": "语言不明显变差：大题量的 MMLU-Pro 和 IFEval，X-M 相对 X-2（同尺寸、种子 0）的差值，CI 下界都要 > −2 个百分点。",
}
THRESHOLDS: Mapping[str, float] = {"1": 0.0, "2": 0.0, "3": -0.02, "4": -2.0}  # 都是严格的 >；"4" 是 D165 原文（现在只作诊断）
# 第 4 条的改写（预注册 XM-5a，D170；N47 的答复；写死，不随任何读数改动）
RULE_4_D170 = ("大题量 MMLU-Pro（字母准确率，2,000 题）与 IFEval（prompt-level strict，541 条），X-M 相对同节点基座的差值，95% CI 下界都不低于"
               "\"X-2 配方三个种子相对各自同节点基座的 CI 下界中最低者，再减 2 个百分点\"：MMLU-Pro：CI 下界 > −5.7；IFEval：CI 下界 > −8.1。")
D170_THRESHOLDS: Mapping[str, float] = {
    "mmlu_pro": -5.7,  # XM-5a：X-2 三个种子的下界 −1.6、−3.7、−3.7，最低 −3.7，再减 2
    "ifeval": -8.1,  # XM-5a：X-2 三个种子的下界 −6.1、−2.8、−4.4，最低 −6.1，再减 2
}
LEVEL = 0.95
N_RESAMPLES = 10_000
SEED = C.BOOTSTRAP_SEED
CRITERION4_VARIANTS: tuple[str, ...] = ("d170",)  # 进结论的只有 XM-5a（D170）；D165 原文（X-M − X-2 seed 0 > −2）总是照算，只作诊断
DECISION_N_IMAGES = XD.COCO_EVAL_N[XD.XM_DECISION]  # 500（src/xm/data 一处定义）
REFSEG_SPLITS: Mapping[str, int] = {"explore_val_refcoco_unc_val": 400, "explore_val_refcoco+_unc_val": 400,
                                    "explore_val_refcocog_umd_val": 200, "devtune800": 800, "devcheck2": 500}
REFSEG_N = 2300
assert sum(REFSEG_SPLITS.values()) == REFSEG_N
XM_MAIN, XM_BOX, BASE_DET, REFSEG_SETTING = "xm_spline_trimap", "xm_box_grabcut", "base_det", "x2_spline_trimap"
DET, SEG = "det", "seg"  # predictions.jsonl 每对的检测、实例分割计数（接口的假设）
PAIR_COUNT_KEYS = ("tp", "fp", "fn")  # predictions.jsonl 每对 det / seg 一节里 src/xm/metrics.table 读的计数（每个 IoU 阈值一个）
IFEVAL_N = 541  # D165 第四节："IFEval（541 条）"，即 IFEval 的全部 prompt（scripts/lang_battery.py 不抽样）
# 读数 → (lang_battery 的读数名, 评分的项, 口径, 题数)
LB_ROWS: Mapping[str, tuple[str, str, str, int]] = {
    "mmlu_pro": (LB.LETTER, LB.READINGS[LB.LETTER], "accuracy", LB.DEFAULT_MMLU_PRO_N),
    "ifeval": (LB.IFEVAL, LB.READINGS[LB.IFEVAL], "prompt_level_strict", IFEVAL_N),
}
XM_TRAIN_CONFIG = REPO_ROOT / "configs" / "xm_train.json"
X2_TRAIN_CONFIG = REPO_ROOT / "configs" / "x2_train.json"
PASS, FAIL = XD.XM_GATE_PASS, "不通过"  # 结论串只在 src/xm/data.XM_GATE_PASS 一处（D167-6）
MET, NOT_MET = "满足", "不满足"
REHEARSAL = "彩排，不作数"
PREDICTIONS, METRICS, CONFIG, META = "predictions.jsonl", "metrics.json", "config.yaml", "meta.json"


class GateInputError(ValueError):
    """不能判定（缺输入、没写完、对或句子的集合不对、检查点对不上……）。"""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _yaml(p: Path) -> dict:
    return yaml.safe_load(Path(p).read_text(encoding="utf-8")) or {}


def run_node(run_dir: Path, *docs: Mapping[str, Any] | None) -> str | None:
    """一个运行的节点：运行身份 meta.json 的 hostname（src/runlog.write_run_metadata）；没有时依次看 docs 里的 run.meta.hostname、
    meta.hostname、hostname、host、node（config.yaml 或 lang_battery.json）。都没有返回 None（同节点核对不过）。"""
    m = Path(run_dir) / META
    if m.is_file():
        h = json.loads(m.read_text(encoding="utf-8")).get("hostname")
        if h:
            return str(h)
    for d in docs:
        if not isinstance(d, Mapping):
            continue
        for v in (((d.get("run") or {}).get("meta") or {}).get("hostname"), (d.get("meta") or {}).get("hostname"), d.get("hostname"),
                  d.get("host"), d.get("node")):
            if v:
                return str(v)
    return None


def _files(leaf: Path, names: Sequence[str] = (PREDICTIONS, METRICS, CONFIG, META)) -> dict[str, str]:
    return {f: _sha(leaf / f) for f in names if (leaf / f).is_file()}


def _leaf_path(path: str | Path, what: str) -> Path:
    p = Path(path)
    leaf = p.parent if p.name == PREDICTIONS else p
    if not leaf.is_dir():
        raise GateInputError(f"{what}：{leaf} 不在")
    return leaf


# ---------------------------------------------------------------------------
# 读输入（DEV-0080 的叶目录：接口的假设集中在 read_xm_leaf 与 pair_counts）
# ---------------------------------------------------------------------------


def read_xm_leaf(path: str | Path, what: str) -> dict[str, Any]:
    """DEV-0080 的一个叶目录 → {leaf, identity, records, node, files}。没写完（metrics.json 的 complete 不是 true）就拒绝。"""
    leaf = _leaf_path(path, what)
    for f in (PREDICTIONS, METRICS, CONFIG):
        if not (leaf / f).is_file():
            raise GateInputError(f"{what}：{leaf} 里没有 {f}")
    m = json.loads((leaf / METRICS).read_text(encoding="utf-8"))
    if m.get("complete") is not True:
        raise GateInputError(f"{what}：{leaf} 没有写完（metrics.json 的 complete = {m.get('complete')!r}）")
    cfg = _yaml(leaf / CONFIG)
    x = cfg.get("xm") or {}
    model = cfg.get("model") or {}
    pr = cfg.get("prompts") or {}
    ident = {"model": model.get("key"), "tiny": bool(model.get("tiny")), "setting": x.get("setting"), "split": x.get("split"),
             "limit": x.get("limit"), "unseal_gate": (x.get("unseal") or {}).get("gate"), "checkpoint": cfg.get("checkpoint"),
             "prompt_version": pr.get("version"), "prompts_frozen": pr.get("frozen")}
    return {"leaf": str(leaf), "identity": ident, "records": _jsonl(leaf / PREDICTIONS), "node": run_node(leaf, cfg), "files": _files(leaf)}


def _nonneg(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def pair_rows(records: Sequence[Mapping[str, Any]], kind: str, what: str) -> list[Mapping[str, Any]]:
    """src/xm/runner 写的 predictions.jsonl 的行（每对一行：id、image_key、category_id、det / seg = src/xm/metrics.eval_pair 的计数，
    tp / fp / fn 各是每个 IoU 阈值一个的非负整数列表）→ 按 id 排好的行，直接交给 src/xm/metrics.table（两个系统按同一顺序，
    metrics.align 要求对、图像、类别的顺序相同）。kind = det 或 seg；这一项是 null（如 base_det 的 seg）就拒绝。"""
    seen: set[str] = set()
    for r in records:
        pid = r.get("id") if isinstance(r, Mapping) else None
        c = r.get(kind) if isinstance(r, Mapping) else None
        ok = isinstance(pid, str) and isinstance(r.get("image_key"), str) and "category_id" in r and isinstance(c, Mapping) and all(
            isinstance(c.get(k), list) and c[k] and all(_nonneg(v) for v in c[k]) for k in PAIR_COUNT_KEYS) and isinstance(c.get("matched_iou"), list)
        if not ok:
            raise GateInputError(f"{what}：记录 {str(r)[:160]} 不是 src/xm/runner 的格式（id、image_key、category_id 与 {kind} 的逐阈值 tp / fp / fn）")
        if pid in seen:
            raise GateInputError(f"{what}：对 {pid} 重复出现")
        seen.add(pid)
    return sorted(records, key=lambda r: r["id"])


def read_refseg_leaf(path: str | Path, what: str) -> dict[str, Any]:
    """run_x2 的一个叶目录（W2 格式，同 scripts/mgr/seed_variance.py 读的那些）→ {leaf, identity, scores: {id: (inter, union)}, node, files}。"""
    leaf = _leaf_path(path, what)
    for f in (PREDICTIONS, METRICS, CONFIG):
        if not (leaf / f).is_file():
            raise GateInputError(f"{what}：{leaf} 里没有 {f}")
    try:
        ident = read_w2_identity(leaf)
    except (KeyError, ValueError) as e:
        raise GateInputError(f"{what}：{leaf} 的 config.yaml 不是 W2 / X-2 的输出（{e}）") from e
    recs = _jsonl(leaf / PREDICTIONS)
    if ident["complete"] is not True or len(recs) != ident["n_samples"]:
        raise GateInputError(f"{what}：{leaf} 没有写完（{len(recs)}/{ident['n_samples']} 句，complete = {ident['complete']}）")
    scores: dict[str, tuple[int, int]] = {}
    for r in recs:
        sc = r.get("score") or {}
        if not isinstance(r.get("id"), str) or not _nonneg(sc.get("inter")) or not _nonneg(sc.get("union")) or sc["inter"] > sc["union"]:
            raise GateInputError(f"{what}：记录 {str(r)[:160]} 不是 W2 格式（id 与 score 里非负整数的 inter ≤ union）")
        if r["id"] in scores:
            raise GateInputError(f"{what}：句子 {r['id']} 重复出现")
        scores[r["id"]] = (int(sc["inter"]), int(sc["union"]))
    return {"leaf": str(leaf), "identity": ident, "scores": scores, "node": run_node(leaf, _yaml(leaf / CONFIG)), "files": _files(leaf)}


def read_lb_run(path: str | Path, what: str) -> dict[str, Any]:
    """一次 lang_battery 运行（--out 目录或其中的 lang_battery.json）：LB.load_run（一次跑完、版本、predictions.jsonl 的 sha256，
    同 --compare）→ {dir, doc, records, node, checkpoint, model, limit, files}。"""
    p = Path(path)
    if not p.exists():
        raise GateInputError(f"{what}：{p} 不在")
    try:
        doc, records, jp = LB.load_run(p)
    except (LB.CompareError, KeyError, OSError, ValueError) as e:
        raise GateInputError(f"{what}：{p} 不是一次跑完的 scripts/lang_battery.py 运行（{e}）") from e
    d = jp.parent
    return {"dir": str(d), "doc": doc, "records": records, "node": run_node(d, doc), "model": (doc.get("model") or {}).get("key"),
            "checkpoint": (doc.get("checkpoint") or {}).get("manifest_sha256") if doc.get("checkpoint") else None, "limit": doc.get("limit"),
            "files": _files(d, (LB.OUT_JSON, LB.PREDICTIONS, CONFIG, META))}


def _manifest(ck: Any) -> str | None:
    return (ck or {}).get("manifest_sha256") if isinstance(ck, Mapping) or ck is None else None


def _training(ck: Any) -> dict[str, Any]:
    return dict((ck or {}).get("training") or {}) if isinstance(ck, Mapping) else {}


def read_xm_checkpoint(path: str | Path) -> dict[str, Any]:
    """X-M 的检查点（<训练叶目录>/ckpt/final/model）→ {dir, manifest_sha256, config_sha256, seed, prompt_version, version}：manifest 的
    sha256 与训练叶目录 config.yaml（运行身份）的 config、seed、xm.prompts.version。不在这个布局下、或没有 config.yaml 就拒绝。"""
    d = Path(path)
    man = d / "manifest.json"
    leaf = checkpoint_leaf(d)
    if not man.is_file() or leaf is None or not (leaf / CONFIG).is_file():
        raise GateInputError(f"--xm-checkpoint {d} 应是 <训练叶目录>/ckpt/final/model（有 manifest.json，叶目录有 config.yaml）")
    ident = _yaml(leaf / CONFIG)
    return {"dir": str(d), "manifest_sha256": _sha(man), "config_sha256": (ident.get("config") or {}).get("sha256"),
            "version": (ident.get("config") or {}).get("version"), "seed": ident.get("seed"),
            "prompt_version": ((ident.get("xm") or {}).get("prompts") or {}).get("version")}


# ---------------------------------------------------------------------------
# 读数
# ---------------------------------------------------------------------------


def paired_ciou_ci(a: np.ndarray, b: np.ndarray) -> S.CI:
    """(n, 2) 的 (inter, union)，按句对齐 → cIoU 之差 a − b 的配对 bootstrap CI；与 scripts/mgr/w2_test_table.fast_paired_ratio_ci、
    src.stats.paired_bootstrap_ci(statistic = ratio_of_sums) 逐位相同（10,000 次、95%、BOOTSTRAP_SEED）。"""
    idx = S._resample_indices(a.shape[0], N_RESAMPLES, SEED)
    boot = a[idx, 0].sum(1) / a[idx, 1].sum(1) - b[idx, 0].sum(1) / b[idx, 1].sum(1)
    return S._percentile_ci(float(ratio_of_sums(a) - ratio_of_sums(b)), boot, LEVEL, N_RESAMPLES)


def f1_criterion(key: str, metric: str, kind: str, cand: Sequence[Mapping[str, Any]], ctrl: Sequence[Mapping[str, Any]], cand_name: str,
                 ctrl_name: str) -> dict[str, Any]:
    """(1)(2)：两个系统按 id 排好的行 → src/xm/metrics.table → 主读数 M.PRIMARY（f1_rexomni，XM-3a）的点估计与按图像重抽的配对 bootstrap
    （M.paired_bootstrap，95%、10,000 次、BOOTSTRAP_SEED）。"""
    ta, tb = M.table(cand, kind), M.table(ctrl, kind)
    try:
        ci = M.paired_bootstrap(ta, tb, M.PRIMARY, n_resamples=N_RESAMPLES, level=LEVEL, seed=SEED)
    except ValueError as e:
        raise GateInputError(f"{metric}：两个系统不能配对或读数无定义（{e}）") from e
    fc, fb = M.point(ta, M.PRIMARY), M.point(tb, M.PRIMARY)
    if ci is None or fc is None or fb is None:
        raise GateInputError(f"{metric}：读数无定义（没有图像或没有真值）")
    ok = bool(ci.low > THRESHOLDS[key])
    return {"rule": RULES[key], "metric": metric, "statistic": M.PRIMARY, "iou": "box" if kind == DET else "mask",
            "candidate_name": cand_name, "control_name": ctrl_name, "candidate": fc, "control": fb,
            "diff": ci.estimate, "ci_low": ci.low, "ci_high": ci.high, "threshold": THRESHOLDS[key], "pass": ok,
            "result": MET if ok else NOT_MET, "n_pairs": len(ta.pair_ids), "n_images": len(ta.images),
            "bootstrap": {"unit": "image", "n_resamples": N_RESAMPLES, "level": LEVEL, "seed": SEED,
                          "function": "src/xm/metrics.paired_bootstrap(table, table, PRIMARY)"}}


def ciou_criterion(xm: Mapping[str, tuple[int, int]], x2: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    ids = sorted(xm)
    a = np.array([xm[i] for i in ids], dtype=np.float64)
    b = np.array([x2[i] for i in ids], dtype=np.float64)
    ci = paired_ciou_ci(a, b)
    ok = bool(ci.low > THRESHOLDS["3"])
    return {"rule": RULES["3"], "metric": "ciou", "candidate_name": "X-M", "control_name": "X-2 seed 0", "candidate": float(ratio_of_sums(a)),
            "control": float(ratio_of_sums(b)), "diff": ci.estimate, "ci_low": ci.low, "ci_high": ci.high, "threshold": THRESHOLDS["3"],
            "pass": ok, "result": MET if ok else NOT_MET, "n": len(ids),
            "bootstrap": {"unit": "sentence", "n_resamples": N_RESAMPLES, "level": LEVEL, "seed": SEED,
                          "function": "口径同 scripts/mgr/x_test_table.paired（w2_test_table.fast_paired_ratio_ci）"}}


def lb_item_values(run: Mapping[str, Any], name: str) -> tuple[list[str], np.ndarray]:
    """一次运行在一个读数上的逐题 0/1（LB.item_counts：MMLU-Pro 的 correct、IFEval prompt-level 的 strict.follow_all；同 --compare）。"""
    reading, bench, metric, _ = LB_ROWS[name]
    recs = run["records"][reading]
    counts = LB.item_counts(bench, metric, recs)
    if any(n != 1 for _, n in counts):
        raise GateInputError(f"{reading}.{metric} 不是每题一个评分单位")
    return [str(r["id"]) for r in recs], np.array([c for c, _ in counts], dtype=np.float64)


LB_BOOTSTRAP = {"unit": "item", "n_resamples": N_RESAMPLES, "level": LEVEL, "seed": SEED,
                "function": "src.stats.paired_bootstrap_ci（逐题 0/1 的均值之差 × 100，不取整）"}


def lb_paired(xm: Mapping[str, Any], other: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    """两次 lang_battery 运行逐题配对：每个读数 X-M − other（百分点，不取整）的配对 bootstrap 95%、10,000 次、BOOTSTRAP_SEED 的 CI，
    下界与该读数的阈值比（严格的 >）。"""
    rows: dict[str, Any] = {}
    for name, (reading, _, metric, n_want) in LB_ROWS.items():
        ids_m, vm = lb_item_values(xm, name)
        ids_o, vo = lb_item_values(other, name)
        if ids_m != ids_o or not ids_m:
            raise GateInputError(f"{reading}：两次 lang_battery 运行的题目不同（--limit 不同？）")
        ci = S.paired_bootstrap_ci(vm, vo, n_resamples=N_RESAMPLES, level=LEVEL, seed=SEED)
        lo, hi, est = 100.0 * ci.low, 100.0 * ci.high, 100.0 * ci.estimate
        rows[name] = {"reading": reading, "metric": metric, "n": len(ids_m), "n_expected": n_want, "xm_pct": 100.0 * float(vm.mean()),
                      "control_pct": 100.0 * float(vo.mean()), "diff_pp": est, "ci_low_pp": lo, "ci_high_pp": hi,
                      "threshold": float(thresholds[name]), "pass": bool(lo > thresholds[name])}
    return rows


def lb_criterion(xm: Mapping[str, Any], base: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """第 4 条（XM-5a，D170；进结论）：X-M − 同节点基座，MMLU-Pro 下界 > −5.7、IFEval 下界 > −8.1。"""
    if variant not in CRITERION4_VARIANTS:
        raise GateInputError(f"--criterion4 只有 {list(CRITERION4_VARIANTS)}，收到 {variant!r}")
    rows = lb_paired(xm, base, D170_THRESHOLDS)
    ok = all(v["pass"] for v in rows.values())
    return {"rule": RULE_4_D170, "variant": variant, "source": "预注册 XM-5a（D170；N47 的答复）",
            "metric": "逐题配对的差（X-M − 同节点基座，百分点）", "control_name": "同节点基座", "rows": rows,
            "thresholds": dict(D170_THRESHOLDS), "pass": ok, "result": MET if ok else NOT_MET, "bootstrap": dict(LB_BOOTSTRAP)}


def lb_d165_diagnostic(xm: Mapping[str, Any], x2: Mapping[str, Any]) -> dict[str, Any]:
    """D165 第五节第 4 条原文（X-M − X-2 seed 0，下界都 > −2）：总是照算、写进门报告，不进结论（XM-5a："原文的写法照算……不进结论"）。"""
    rows = lb_paired(xm, x2, {k: THRESHOLDS["4"] for k in LB_ROWS})
    ok = all(v["pass"] for v in rows.values())
    return {"rule": RULES["4"], "variant": "d165", "counts_toward_verdict": False, "metric": "逐题配对的差（X-M − X-2 seed 0，百分点）",
            "control_name": "X-2 seed 0", "rows": rows, "threshold": THRESHOLDS["4"], "would_pass": ok,
            "result_if_counted": MET if ok else NOT_MET, "same_node": _same([xm["node"], x2["node"]]), "bootstrap": dict(LB_BOOTSTRAP)}


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def decision_pairs(data_root: str | Path, gate: str | None, splits_path: str | Path | None = None) -> list[str]:
    """判定集 xm_decision 的全部对（经 src/xm/data.load_pairs 的守卫：门名必须是 G-XM；清单缺省是入库的 configs/xm_splits.json）。"""
    pairs = XD.load_pairs(XD.XM_DECISION, data_root, gate=gate, splits_path=splits_path)
    ids = sorted(p.id for p in pairs)
    if len(set(ids)) != len(ids) or not ids:
        raise GateInputError("xm_decision 的对为空或有重复")
    return ids


def _same(vals: Sequence[Any]) -> bool:
    return len(vals) > 0 and None not in vals and len(set(vals)) == 1


def evaluate(*, gate: str | None, data_root: str | Path, xm_checkpoint: str | Path, xm_main: str, xm_box: str, base_det: str,
             xm_refseg: Sequence[str], x2_refseg: Sequence[str], lb_xm: str, lb_base: str, lb_x2: str, lb_compare: str | None = None,
             rehearsal: bool = False, criterion4: str = "d170", splits_path: str | Path | None = None) -> dict[str, Any]:
    if gate != XD.GATE:
        raise XD.SealedSplitError(f"判定集 {XD.XM_DECISION} 是密封的：--gate 必须是 {XD.GATE}，收到 {gate!r}")
    if criterion4 not in CRITERION4_VARIANTS:
        raise GateInputError(f"--criterion4 只有 {list(CRITERION4_VARIANTS)}（N47 答复之前不加别的写法），收到 {criterion4!r}")
    ck = read_xm_checkpoint(xm_checkpoint)
    want = decision_pairs(data_root, gate, splits_path)
    dec = {"xm_main": read_xm_leaf(xm_main, "X-M 主设置"), "xm_box_grabcut": read_xm_leaf(xm_box, "X-M 的框交给 GrabCut"),
           "base_det": read_xm_leaf(base_det, "基座零样本")}
    for k, setting in (("xm_main", XM_MAIN), ("xm_box_grabcut", XM_BOX), ("base_det", BASE_DET)):
        i = dec[k]["identity"]
        if i["setting"] != setting or i["split"] != XD.XM_DECISION:
            raise GateInputError(f"{k}：{dec[k]['leaf']} 应是 {setting}/{XD.XM_DECISION}，收到 {i['setting']}/{i['split']}")
    cand_ck = ck["manifest_sha256"]
    for k in ("xm_main", "xm_box_grabcut"):
        if _manifest(dec[k]["identity"]["checkpoint"]) != cand_ck:
            raise GateInputError(f"{k} 的检查点（manifest sha256）不是 --xm-checkpoint 的 X-M 检查点")
    if dec["base_det"]["identity"]["checkpoint"] is not None:
        raise GateInputError("base_det 不能带训练的检查点（它是基座零样本）")
    det_xm = pair_rows(dec["xm_main"]["records"], DET, "X-M 主设置（检测）")
    seg_xm = pair_rows(dec["xm_main"]["records"], SEG, "X-M 主设置（实例分割）")
    seg_box = pair_rows(dec["xm_box_grabcut"]["records"], SEG, "xm_box_grabcut（实例分割）")
    det_base = pair_rows(dec["base_det"]["records"], DET, "base_det（检测）")
    for what, ps in (("X-M 主设置", seg_xm), ("xm_box_grabcut", seg_box), ("base_det", det_base)):
        got = sorted(p["id"] for p in ps)
        if got != want:
            miss, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
            raise GateInputError(f"{what}：对的集合不是 xm_decision 的 {len(want)} 对——缺 {len(miss)} 对（如 {miss[:3]}），"
                                 f"多 {len(extra)} 对（如 {extra[:3]}）；拒绝判定")
    # 指代分割（2,300 句）
    sides: dict[str, list[dict]] = {"xm": [read_refseg_leaf(p, "X-M 指代分割") for p in xm_refseg],
                                    "x2": [read_refseg_leaf(p, "X-2 seed 0 指代分割") for p in x2_refseg]}
    pooled: dict[str, dict[str, tuple[int, int]]] = {}
    for side, leaves in sides.items():
        splits = [lf["identity"]["split"] for lf in leaves]
        if len(set(splits)) != len(splits):
            raise GateInputError(f"{side} 一边的指代分割叶目录有重复的划分 {splits}")
        pool: dict[str, tuple[int, int]] = {}
        for lf in leaves:
            i = lf["identity"]
            if i["setting"] != REFSEG_SETTING or i["split"] not in REFSEG_SPLITS:
                raise GateInputError(f"{side}：{lf['leaf']} 应是 {REFSEG_SETTING} 在 2,300 句的 5 个划分 {list(REFSEG_SPLITS)} 之一，"
                                     f"收到 {i['setting']}/{i['split']}")
            dup = set(pool) & set(lf["scores"])
            if dup:
                raise GateInputError(f"{side}：句子 {sorted(dup)[:3]} 出现在两个叶目录里")
            pool.update(lf["scores"])
        pooled[side] = pool
    if set(pooled["xm"]) != set(pooled["x2"]):
        raise GateInputError(f"指代分割两边的句子集合不同（只在 X-M {len(set(pooled['xm']) - set(pooled['x2']))} 句，"
                             f"只在 X-2 {len(set(pooled['x2']) - set(pooled['xm']))} 句）")
    xm_cks = {_manifest(lf["identity"]["checkpoint"]) for lf in sides["xm"]}
    x2_cks = {_manifest(lf["identity"]["checkpoint"]) for lf in sides["x2"]}
    if xm_cks != {cand_ck}:
        raise GateInputError("X-M 的指代分割叶目录不是 --xm-checkpoint 的 X-M 检查点")
    if len(x2_cks) != 1 or None in x2_cks or cand_ck in x2_cks:
        raise GateInputError("X-2 一边的指代分割叶目录要是同一个、带检查点、且不是 X-M 的检查点")
    ref_ck = next(iter(x2_cks))
    # 语言（三次 lang_battery 运行，逐题配对）：第 4 条 = X-M − 同节点基座（XM-5a）；D165 原文 X-M − X-2 seed 0 只作诊断
    lbm = read_lb_run(lb_xm, "--lb-xm（X-M 的 lang_battery）")
    lbb = read_lb_run(lb_base, "--lb-base（同节点基座的 lang_battery）")
    lb2 = read_lb_run(lb_x2, "--lb-x2（X-2 seed 0 的 lang_battery）")
    if not (LB._identity(lbm["doc"]) == LB._identity(lbb["doc"]) == LB._identity(lb2["doc"])):
        raise GateInputError("三次 lang_battery 运行的题目或数据不同（同 --compare 的核对）")
    if lbm["checkpoint"] != cand_ck:
        raise GateInputError("--lb-xm 的检查点不是 --xm-checkpoint 的 X-M 检查点")
    if lbb["checkpoint"] is not None:
        raise GateInputError("--lb-base 带了训练的检查点：第 4 条的对照是基座本身")
    if lb2["checkpoint"] != ref_ck:
        raise GateInputError("--lb-x2 的检查点不是指代分割里 X-2 seed 0 的检查点")
    lbc = None
    if lb_compare is not None:
        p = Path(lb_compare)
        if not p.is_file():
            raise GateInputError(f"--lb-compare {p} 不在")
        doc = json.loads(p.read_text(encoding="utf-8"))
        lbc = {"path": str(p), "note": "lang_battery --compare（1,000 次、取整到 0.1），只作诊断，不入结论",
               "base": (doc.get("base") or {}).get("label"), "other": (doc.get("other") or {}).get("label"),
               "rows": [{k: r.get(k) for k in ("reading", "metric", "delta_pp", "ci95_pp")} for r in doc.get("rows", []) if r.get("headline")]}
    crit = {"1": f1_criterion("1", "mask_f1_rexomni", SEG, seg_xm, seg_box, "X-M xm_spline_trimap", "X-M xm_box_grabcut"),
            "2": f1_criterion("2", "box_f1_rexomni", DET, det_xm, det_base, "X-M（调用 1 的框）", "基座零样本 base_det"),
            "3": ciou_criterion(pooled["xm"], pooled["x2"]),
            "4": lb_criterion(lbm, lbb, criterion4)}
    d165 = lb_d165_diagnostic(lbm, lb2)
    passed = all(c["pass"] for c in crit.values())
    decision = PASS if passed else FAIL
    # 判定点核对
    dl = list(dec.values())
    rl = sides["xm"] + sides["x2"]
    nodes = {"decision": {k: v["node"] for k, v in dec.items()}, "refseg_xm": sorted({str(lf["node"]) for lf in sides["xm"]}),
             "refseg_x2": sorted({str(lf["node"]) for lf in sides["x2"]}), "lb_xm": lbm["node"], "lb_base": lbb["node"], "lb_x2": lb2["node"]}
    models = [lf["identity"]["model"] for lf in dl + rl] + [lbm["model"], lbb["model"], lb2["model"]]
    x2_tr = [_training(lf["identity"]["checkpoint"]) for lf in sides["x2"]]
    leaf_pv = [lf["identity"]["prompt_version"] for lf in dl]
    checks = {
        "decision_n_images_is_500": crit["1"]["n_images"] == DECISION_N_IMAGES,
        "decision_splits_is_repo_config": splits_path is None,
        "decision_unsealed_with_G-XM": all(lf["identity"]["unseal_gate"] == XD.GATE for lf in dl),
        "decision_same_node": _same([lf["node"] for lf in dl]),  # (a)
        "refseg_same_node": _same([lf["node"] for lf in rl]),  # (b)
        "lb_same_node": _same([lbm["node"], lbb["node"]]),  # (c) 第 4 条（XM-5a）配对的两次：X-M 与同节点基座
        "prompt_version_matches": ck["prompt_version"] is not None and all(v == ck["prompt_version"] for v in leaf_pv),
        "decision_prompts_frozen": all(lf["identity"]["prompts_frozen"] is True for lf in dl),  # D168：判定集只用冻结的提示
        "not_tiny": all(not lf["identity"]["tiny"] for lf in dl) and all(not lf["identity"]["tiny"] for lf in rl)
                    and not any(r["doc"].get("mode") == "tiny" for r in (lbm, lbb, lb2)),
        "no_limit": all(lf["identity"]["limit"] is None for lf in dl + rl),
        "same_model": _same(models),
        "refseg_is_dev2300": len(pooled["xm"]) == REFSEG_N and sorted(lf["identity"]["split"] for lf in sides["xm"]) == sorted(REFSEG_SPLITS)
                             and all(len(lf["scores"]) == REFSEG_SPLITS[lf["identity"]["split"]] for lf in rl),
        "lb_full": all(r["limit"] is None for r in (lbm, lbb, lb2)) and all(crit["4"]["rows"][k]["n"] == LB_ROWS[k][3] for k in LB_ROWS),
        "candidate_is_xm_train_seed0": ck["config_sha256"] == _sha(XM_TRAIN_CONFIG) and ck["seed"] == 0,
        "reference_is_x2_train_seed0": all(t.get("config_sha256") == _sha(X2_TRAIN_CONFIG) and t.get("seed") == 0 for t in x2_tr),
    }
    reasons = (["--rehearsal"] if rehearsal else []) + [k for k, v in checks.items() if not v]
    is_rehearsal = bool(reasons)
    sp = Path(splits_path or XD.XM_SPLITS_PATH)
    out = {
        "gate": XD.GATE,
        "rules_source": "D165 第五节（负责人原文，docs/lead_answers/2026-11-03-xm-directive.md）；预注册 XM-5；F1 口径 XM-3a（Rex-Omni 宏平均）",
        "verdict": decision if not is_rehearsal else f"{REHEARSAL}（规则给出：{decision}）",
        "decision": decision,
        "pass": bool(passed),
        "rehearsal": is_rehearsal,
        "rehearsal_reasons": reasons,
        "checks": checks,
        "is_gate_point": all(checks.values()),
        "criteria": crit,
        "criterion4": criterion4,
        "diagnostics": {"criterion4_d165": d165, "lb_compare": lbc},
        "candidate": {"name": "X-M", "manifest_sha256": cand_ck, "checkpoint": ck["dir"], "config_sha256": ck["config_sha256"],
                      "config_version": ck["version"], "seed": ck["seed"], "prompt_version": ck["prompt_version"],
                      "model": dec["xm_main"]["identity"]["model"]},
        "reference": {"name": "X-2 seed 0", "manifest_sha256": ref_ck, "training": x2_tr[0]},
        "prompt_versions": {"checkpoint": ck["prompt_version"], **{k: v["identity"]["prompt_version"] for k, v in dec.items()}},
        "decision_split": {"split": XD.XM_DECISION, "n_pairs": len(want), "n_images": crit["1"]["n_images"],
                           "pair_ids_sha256": hashlib.sha256("\n".join(want).encode("utf-8")).hexdigest(), "splits_file": str(sp),
                           "splits_sha256": _sha(sp) if sp.is_file() else None},
        "nodes": nodes,
        "inputs": {"xm_checkpoint": {"dir": ck["dir"], "manifest_sha256": cand_ck},
                   "xm_main": {"leaf": dec["xm_main"]["leaf"], "sha256": dec["xm_main"]["files"]},
                   "xm_box_grabcut": {"leaf": dec["xm_box_grabcut"]["leaf"], "sha256": dec["xm_box_grabcut"]["files"]},
                   "base_det": {"leaf": dec["base_det"]["leaf"], "sha256": dec["base_det"]["files"]},
                   "xm_refseg": [{"leaf": lf["leaf"], "split": lf["identity"]["split"], "sha256": lf["files"]} for lf in sides["xm"]],
                   "x2_refseg": [{"leaf": lf["leaf"], "split": lf["identity"]["split"], "sha256": lf["files"]} for lf in sides["x2"]],
                   "lb_xm": {"dir": lbm["dir"], "sha256": lbm["files"]}, "lb_base": {"dir": lbb["dir"], "sha256": lbb["files"]},
                   "lb_x2": {"dir": lb2["dir"], "sha256": lb2["files"], "role": "D165 原文的诊断，不入结论"},
                   "lb_compare": None if lb_compare is None else {"path": str(lb_compare), "sha256": _sha(Path(lb_compare)),
                                                                  "role": "诊断，不入结论"},
                   "configs": {"xm_train.json": _sha(XM_TRAIN_CONFIG), "x2_train.json": _sha(X2_TRAIN_CONFIG)}},
    }
    return out


def _f(v: Any, nd: int = 4, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def markdown(doc: Mapping[str, Any]) -> str:
    c = doc["criteria"]
    lines = [f"# X-M 的去留判定 {doc['gate']}（D165 第五节，规则写死）", "",
             f"**结论：{doc['verdict']}**" + ("" if not doc["rehearsal"] else f"（{'、'.join(doc['rehearsal_reasons'])}）"), "",
             f"- X-M 检查点 manifest {str(doc['candidate']['manifest_sha256'])[:16]}…（提示 {doc['candidate']['prompt_version']}）；"
             f"X-2 seed 0 检查点 {str(doc['reference']['manifest_sha256'])[:16]}…；模型 {doc['candidate']['model']}",
             f"- 判定集 {doc['decision_split']['split']}：{doc['decision_split']['n_images']} 张图、{doc['decision_split']['n_pairs']} 对"
             f"（{doc['decision_split']['pair_ids_sha256'][:12]}…）",
             f"- 节点：{doc['nodes']}", "",
             "| 条 | 读数 | X-M | 对照 | 差 | 95% CI | 阈值（下界 >） | 结果 |", "|---|---|---|---|---|---|---|---|"]
    for k in ("1", "2", "3"):
        x = c[k]
        lines.append(f"| {k} | {x['metric']}（{x['candidate_name']} − {x['control_name']}） | {_f(x['candidate'])} | {_f(x['control'])} | "
                     f"{_f(x['diff'], signed=True)} | [{_f(x['ci_low'], signed=True)}, {_f(x['ci_high'], signed=True)}] | {x['threshold']:g} | "
                     f"{x['result']} |")
    for r in c["4"]["rows"].values():
        lines.append(f"| 4（d170） | {r['reading']}.{r['metric']}（X-M − 同节点基座，百分点，n = {r['n']}） | {r['xm_pct']:.2f} | "
                     f"{r['control_pct']:.2f} | {r['diff_pp']:+.3f} | [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}] | {r['threshold']:g} | "
                     f"{MET if r['pass'] else NOT_MET} |")
    d165 = doc["diagnostics"]["criterion4_d165"]
    lines += ["", "四条（第 1–3 条为 D165 第五节原文；第 4 条为 XM-5a 的改写，D170）："]
    lines += [f"{k}. {c[k]['rule']} → **{c[k]['result']}**" for k in ("1", "2", "3", "4")]
    lines += ["", f"**第 4 条，d170（进结论）**：X-M − 同节点基座 → **{c['4']['result']}**（MMLU-Pro 下界 "
                  f"{c['4']['rows']['mmlu_pro']['ci_low_pp']:+.3f} > {D170_THRESHOLDS['mmlu_pro']:g}？IFEval 下界 "
                  f"{c['4']['rows']['ifeval']['ci_low_pp']:+.3f} > {D170_THRESHOLDS['ifeval']:g}？）",
              f"**第 4 条原文，d165（诊断，不入结论）**：X-M − X-2 seed 0："
              + "；".join(f"{r['reading']} {r['diff_pp']:+.3f} [{r['ci_low_pp']:+.3f}, {r['ci_high_pp']:+.3f}]" for r in d165["rows"].values())
              + f"；若计入（下界都 > {d165['threshold']:g}）：{d165['result_if_counted']}；两次运行同节点：{d165['same_node']}"]
    if doc["diagnostics"].get("lb_compare"):
        lc = doc["diagnostics"]["lb_compare"]
        lines.append(f"lb_compare（诊断，不入结论；{lc['other']} − {lc['base']}）：" + "；".join(
            f"{r['reading']}.{r['metric']} {r['delta_pp']} {r['ci95_pp']}" for r in lc["rows"]))
    lines += ["", f"(1)(2) 按图像重抽、(3) 逐句配对、(4) 逐题配对：95%、{N_RESAMPLES:,} 次、种子 {SEED}，不取整；第 4 条的写法 {doc['criterion4']}。",
              "", "判定点核对：" + "；".join(f"{k} = {v}" for k, v in doc["checks"].items()), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", help=f"读密封的判定集 xm_decision：门名必须是 {XD.GATE}")
    p.add_argument("--data-root", required=True, help="src/xm/data.load_pairs 的数据根（读 xm_decision 的全部对）")
    p.add_argument("--splits", help="划分清单（缺省 = 入库的 configs/xm_splits.json；给了别的只能彩排）")
    p.add_argument("--xm-checkpoint", required=True, help="X-M 的检查点（<训练叶目录>/ckpt/final/model；核对 manifest 与提示版本）")
    p.add_argument("--xm-main", required=True, help="X-M 的 xm_spline_trimap 在 xm_decision 上的叶目录")
    p.add_argument("--xm-box-grabcut", required=True, help="X-M 的 xm_box_grabcut 在 xm_decision 上的叶目录")
    p.add_argument("--base-det", required=True, help="基座零样本 base_det 在 xm_decision 上的叶目录")
    p.add_argument("--xm-refseg", nargs="+", required=True, help="X-M 的 x2_spline_trimap 在 2,300 句的 5 个划分上的叶目录")
    p.add_argument("--x2-refseg", nargs="+", required=True, help="X-2 seed 0 的 x2_spline_trimap 在同样 5 个划分上的叶目录")
    p.add_argument("--lb-xm", required=True, help="X-M 的 scripts/lang_battery.py 运行目录（或其中的 lang_battery.json）")
    p.add_argument("--lb-base", required=True, help="同节点基座的 scripts/lang_battery.py 运行目录（07 上 RUN-0084 的 lb_base；第 4 条的对照）")
    p.add_argument("--lb-x2", required=True, help="X-2 seed 0 的 scripts/lang_battery.py 运行目录（D165 原文的诊断，不入结论）")
    p.add_argument("--lb-compare", help="lang_battery --compare 的输出 JSON（只作诊断，不入结论）")
    p.add_argument("--criterion4", default="d170", choices=CRITERION4_VARIANTS,
                   help="第 4 条的写法：d170 = 预注册 XM-5a（缺省，也是唯一进结论的写法）；D165 原文总是照算，只作诊断")
    p.add_argument("--rehearsal", action="store_true", help="彩排：照算，结论标为“彩排，不作数”")
    p.add_argument("--out-dir", required=True, help="写 gate_xm.json 与 gate_xm.md 的目录")
    a = p.parse_args(argv)
    try:
        doc = evaluate(gate=a.gate, data_root=a.data_root, xm_checkpoint=a.xm_checkpoint, xm_main=a.xm_main, xm_box=a.xm_box_grabcut,
                       base_det=a.base_det, xm_refseg=a.xm_refseg, x2_refseg=a.x2_refseg, lb_xm=a.lb_xm, lb_base=a.lb_base, lb_x2=a.lb_x2,
                       lb_compare=a.lb_compare, rehearsal=a.rehearsal, criterion4=a.criterion4, splits_path=a.splits)
    except (GateInputError, XD.SealedSplitError, ValueError, OSError, KeyError) as e:  # XD.XMDataError 是 ValueError
        print(f"不能判定：{e}", file=sys.stderr)
        return 2
    md = markdown(doc)
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "gate_xm.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (d / "gate_xm.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
