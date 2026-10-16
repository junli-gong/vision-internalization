"""DEV-0046（D96）：E5 训练后评测前的合并核对改用判据 e5-v2，另加指标层面的合并前后比较（只报告）与只做核对的模式。

- 真实文件的回归（tests/fixtures/e5/merge_checks_0051_hpcgpu07.json：RUN-0051 到 10-15 23:00 UTC 的 15 份核对文件的 checks 与记录的判定，
  不依赖 experiments 目录）：v1 的 judge 逐个复现文件里记录的判定（只有 e3b_c3 4K 种子 0、e3b 4K 种子 0 通过），e5-v2 全部通过；
- e5-v2 的边界（逐条一个用例）：across_merge 0.979 不过；受限教师强制 0.9989 不过；任一比较 gap 最大 2.01 不过、恰好 2.0 通过；中位数
  0.51 不过；有分歧算不出 gap 不过；error 不空不过；exact_rows 0.5、全词表 merge 0.90 照样通过（只报告）；精确模式沿用 v1 的精确规则；
- 指标层面的汇总（纯函数）：差、标记与阈值；旧文件改名留存（.1、.2 后缀；同一版本覆盖）；
- tiny 端到端（CPU fp32，InternVL 的小模型；E5 检查点 = base 起点 + 随机的 E5 LoRA 与网格 token 行，LoRA 确实改变输出）：
  merge_check.json 是 e5-v2，有 v1_report_only 与 metric_level，指标层面的差恰好为 0、预测逐行相同；A 侧解码挂着 LoRA（逐模块审计；
  绕开 LoRA 时审计报错）；已有的 v1 文件被改名留存；--check-only 不建任务目录、退出码正确（通过 0，不过 1，与 --task 无关）；
  正常评测（默认的结构强制）的确定输出、打印的结果行与改动前（4bd7f13 的脚本与模块，原样存在 tests/fixtures/e5/）逐字节相同，合并核对里
  改动前就有的字段与 v1 的判定照旧。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from src.adapters import registry
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3.merge_check import MergeCheckFailed, load_cti
from src.e3.tokens import attach_vocab
from src.e5 import stack as STK
from src.e5 import trained_eval as TE
from src.e5.protocol import E5_TASKS
from src.e5.runner import make_sample
from src.e5.token_grid import STRUCTURED, E5Tokens, decode_grids, sample_images, task_format
from src.e5.train import E5Trainer
from src.e5.train_config import load_train_config, resolve_run
from src.e5.train_data import TrainSet

REPO = Path(__file__).resolve().parents[1]
CFG = load_train_config()
FAST = "internvl3.5-2b-hf"  # 小模型里 1057 步的结构强制解码最快
FIX = REPO / "tests" / "fixtures" / "e5" / "merge_checks_0051_hpcgpu07.json"
SNAP = REPO / "tests" / "fixtures" / "e5"
SNAP_SHA256 = {  # 4bd7f13（DEV-0046 之前）的文件原样
    "trained_eval_4bd7f13.py": "7cc7c6d70b0ebef35a4555fe6fbea313f7a2161d3919df4f64be24801ff3d00f",
    "eval_e5_trained_4bd7f13.py": "38faffc6e88cc356b496041bd175e10fe7cfb17cf363d70e0837d1178e7b1328",
}
DET = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")
V1_PASSED = ("experiments/E5/0051_e5_main_hpcgpu07/trained/e3b/qwen3.5-2b/n4000/seed0/merge_check.json",
             "experiments/E5/0051_e5_main_hpcgpu07/trained/e3b_c3/qwen3.5-2b/n4000/seed0/merge_check.json")
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_e5v2", REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


def _load_path(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _fails(f: dict) -> list[str]:
    return [m for bad in f.values() for m in bad]


# ---------------------------------------------------------------------------
# 阈值（D96 原文的数）
# ---------------------------------------------------------------------------


def test_thresholds_are_d96():
    assert TE.CRITERIA_VERSION == "e5-v2" and "D96-1" in TE.CRITERIA_BASIS
    assert dict(TE.E5V2_THRESHOLDS) == {"across_merge_min_tf_argmax_agreement": 0.98, "restricted_min_tf_agreement": 0.999,
                                        "max_gap": 2.0, "max_median_gap": 0.5}
    assert dict(TE.METRIC_LEVEL_THRESHOLDS) == {"iou_grid_mean_diff_abs": 0.01, "relief_error_mean_diff_abs": 0.02}
    assert "D96-2" in TE.METRIC_LEVEL_BASIS
    with pytest.raises(TypeError):
        TE.E5V2_THRESHOLDS["max_gap"] = 3.0  # 只读
    # 所有比较（D96-1）：merge 的生成与教师强制、restricted 的 decode_vs_generate_batch 与教师强制、across_merge 的教师强制
    assert [(".".join(p), n) for _, p, n, _ in TE.E5V2_GAP_COMPARISONS] == [
        ("merge", "generate"), ("merge", "teacher_forcing"), ("restricted", "decode_vs_generate_batch"),
        ("restricted", "teacher_forcing"), ("restricted.across_merge", "teacher_forcing")]
    crit = TE.e5v2_criteria(False)
    assert crit["version"] == "e5-v2" and crit["basis"] == "D96-1（N34 的 (b′)）" and crit["thresholds"] == dict(TE.E5V2_THRESHOLDS)
    assert any("argmax_agreement（全词表）" in x for x in crit["report_only"]) and any("exact_rows" in x for x in crit["report_only"])
    # v1 的阈值不变（scripts/check_training_interfaces.py 不改）
    cti = load_cti()
    assert (cti.CRITERIA_VERSION, cti.RESTRICTED_MIN_TF_AGREEMENT, cti.RESTRICTED_MIN_EXACT_ROWS_OFF, cti.RESTRICTED_MAX_GAP,
            cti.MERGE_MIN_TF_AGREEMENT, cti.MERGE_MAX_MEDIAN_GAP) == ("v1", 0.999, 0.95, 2.0, 0.98, 0.5)


# ---------------------------------------------------------------------------
# 真实文件的回归（RUN-0051）
# ---------------------------------------------------------------------------


def _fixture() -> dict:
    return json.loads(FIX.read_text(encoding="utf-8"))


def test_fixture_covers_run0051():
    doc = _fixture()
    files = doc["files"]
    assert doc["n_files"] == len(files) == 15 and doc["source_commit"] == "4bd7f13b"
    assert len({e["path"] for e in files}) == 15
    for e in files:
        assert e["format"] == TE.MERGE_CHECK_FORMAT and e["criteria_version"] == "v1" and e["exact"] is False and e["error"] is None
        assert "2026-10-14" <= e["written_utc"] < "2026-10-16"  # 最后一份（e3b_c3 1K 种子 2）写于 10-15 23:00:57
        assert sorted(e["checks"]) == ["merge", "restricted"]


@pytest.mark.parametrize("k", range(15))
def test_real_file_v1_reproduced_e5v2_passes(k):
    e = _fixture()["files"][k]
    v1 = TE.v1_report(e["checks"], e["exact"])
    assert v1["criteria_version"] == "v1" and v1["report_only"] is True
    assert v1["failures"] == e["failures"] and v1["pass"] is e["pass"], e["path"]  # v1 复现文件里记录的判定
    v2 = TE.judge_e5v2(e["checks"], exact=False)
    assert _fails(v2) == [], (e["path"], v2)
    doc = {"criteria": {"exact": e["exact"]}, "checks": e["checks"], "error": e["error"]}
    assert TE.judge_document_e5v2(doc)["pass"] is True


def test_real_files_v1_only_two_passed():
    files = _fixture()["files"]
    assert sorted(e["path"] for e in files if e["pass"]) == sorted(V1_PASSED)  # 管理员按 (b′) 手算：v1 下只有这两个
    assert sum(TE.judge_e5v2(e["checks"], exact=False) == {"merge": [], "restricted": []} for e in files) == 15
    # 被拒的原因只有两种（N34）：restricted 的 exact_rows 与 merge 的全词表教师强制
    for e in files:
        for m in _fails(e["failures"]):
            assert ("exact_rows" in m) or ("merge/thinking_off：教师强制的 argmax_agreement" in m), m


# ---------------------------------------------------------------------------
# e5-v2 的边界（合成的 checks）
# ---------------------------------------------------------------------------


def _gap(n: int = 3, median: float = 0.1, mx: float = 0.3) -> dict:
    return {"n": n, "median": median if n else None, "max": mx if n else None, "frac_lt_threshold": 1.0, "threshold": 2.0}


def _gen(n_div: int = 2, mx: float = 0.3, median: float = 0.1, exact_rows: float = 0.95) -> dict:
    return {"n_rows": 40, "exact_rows": exact_rows, "token_agreement": 0.99, "n_agree_tokens": 41000, "n_tokens": 42280,
            "n_divergent_rows": n_div, "n_divergence_without_gap": 0, "gap": _gap(n_div, median, mx), "top2_gap": _gap(n_div, median, mx)}


def _tf(key: str, agreement: float, n_dis: int = 3, mx: float = 0.3, median: float = 0.1, abs_: bool = False) -> dict:
    out = {"n_tokens": 42280, key: agreement, "n_disagreements": n_dis, "gap": _gap(n_dis, median, mx),
           "top2_gap": _gap(n_dis, median, mx), "disagreements": [], "n_listed": 0}
    if abs_:
        out.update(max_abs_logit=1.0, max_abs_hidden=3.0)
    return out


def _checks() -> dict:
    """一份通过 e5-v2 的 checks（结构同 stacked_merge_check 写的）。"""
    m = "thinking_off"
    return {"merge": {"mode": "e5_stacked", "by_thinking": {m: {"n_rows": 40, "generate": _gen(),
                                                              "teacher_forcing": _tf("argmax_agreement", 0.995, abs_=True)}}},
            "restricted": {"by_thinking": {m: {"n_rows": 40, "decode_vs_generate_batch": _gen(),
                                               "teacher_forcing": _tf("agreement", 0.9995)}},
                           "across_merge": {"by_thinking": {m: {"n_rows": 40,
                                                                "teacher_forcing": _tf("argmax_agreement", 0.9992, abs_=True)}}}}}


def _node(checks: dict, path: tuple[str, ...], name: str) -> dict:
    cur = checks
    for p in path:
        cur = cur[p]
    return cur["by_thinking"]["thinking_off"][name]


def _judge(checks: dict, **kw) -> list[str]:
    return _fails(TE.judge_e5v2(checks, exact=False, **kw))


def test_e5v2_baseline_passes():
    assert _judge(_checks()) == []
    assert TE.judge_e5v2(_checks(), exact=False) == {"merge": [], "restricted": []}


@pytest.mark.parametrize("value,ok", [(0.979, False), (0.98, True)])
def test_e5v2_across_merge(value, ok):
    c = _checks()
    _node(c, ("restricted", "across_merge"), "teacher_forcing")["argmax_agreement"] = value
    bad = TE.judge_e5v2(c, exact=False)
    assert (bad["restricted"] == []) is ok and bad["merge"] == []


@pytest.mark.parametrize("value,ok", [(0.9989, False), (0.999, True)])
def test_e5v2_restricted_tf(value, ok):
    c = _checks()
    _node(c, ("restricted",), "teacher_forcing")["agreement"] = value
    assert (TE.judge_e5v2(c, exact=False)["restricted"] == []) is ok


@pytest.mark.parametrize("comp", range(5))
@pytest.mark.parametrize("mx,ok", [(2.01, False), (2.0, True)])
def test_e5v2_gap_max_every_comparison(comp, mx, ok):
    section, path, name, _ = TE.E5V2_GAP_COMPARISONS[comp]
    c = _checks()
    _node(c, path, name)["gap"]["max"] = mx
    bad = TE.judge_e5v2(c, exact=False)
    assert (bad[section] == []) is ok, bad
    assert all(v == [] for k, v in bad.items() if k != section)


@pytest.mark.parametrize("comp", range(5))
@pytest.mark.parametrize("med,ok", [(0.51, False), (0.5, True)])
def test_e5v2_gap_median_every_comparison(comp, med, ok):
    section, path, name, _ = TE.E5V2_GAP_COMPARISONS[comp]
    c = _checks()
    _node(c, path, name)["gap"]["median"] = med
    assert (TE.judge_e5v2(c, exact=False)[section] == []) is ok


def test_e5v2_no_divergence_passes_naturally():
    c = _checks()
    for _, path, name, is_gen in TE.E5V2_GAP_COMPARISONS:
        node = _node(c, path, name)
        node["gap"] = _gap(0)
        node["n_divergent_rows" if is_gen else "n_disagreements"] = 0
    assert _judge(c) == []


@pytest.mark.parametrize("comp", range(5))
def test_e5v2_divergence_without_gap_fails(comp):
    _, path, name, is_gen = TE.E5V2_GAP_COMPARISONS[comp]
    c = _checks()
    node = _node(c, path, name)
    if is_gen:  # GenAgg：分歧行里算不出 gap 的
        node["n_divergence_without_gap"] = 1
        node["n_divergent_rows"] += 1
    else:  # TfAgg 没有这个计数：分歧数多于 gap.n
        node["n_disagreements"] += 1
    assert any("算不出" in m for m in _judge(c))


def test_e5v2_error_fails():
    assert TE.judge_e5v2(_checks(), exact=False, error={"repr": "RuntimeError('x')"})["error"] == ["核对出错：RuntimeError('x')"]
    assert TE.judge_document_e5v2({"criteria": {"exact": False}, "checks": _checks(), "error": {"repr": "x"}})["pass"] is False
    assert TE.judge_document_e5v2({"criteria": {"exact": False}, "checks": {}, "error": {"repr": "x"}})["pass"] is False


@pytest.mark.parametrize("drop", [("merge",), ("restricted",), ("restricted", "across_merge")])
def test_e5v2_missing_section_fails(drop):
    c = _checks()
    parent = c
    for p in drop[:-1]:
        parent = parent[p]
    del parent[drop[-1]]
    assert _judge(c)


def test_e5v2_report_only_items_do_not_judge():
    """exact_rows 0.5、全词表 merge 0.90（以及 merge 生成的逐位一致率很低）照样通过；同一份读数 v1 判不过。"""
    c = _checks()
    _node(c, ("restricted",), "decode_vs_generate_batch")["exact_rows"] = 0.5
    _node(c, ("restricted",), "decode_vs_generate_batch")["token_agreement"] = 0.6
    _node(c, ("merge",), "teacher_forcing")["argmax_agreement"] = 0.90
    _node(c, ("merge",), "generate")["exact_rows"] = 0.1
    _node(c, ("merge",), "generate")["token_agreement"] = 0.3
    _node(c, ("merge",), "teacher_forcing")["top2_gap"]["max"] = 9.0  # top2_gap 只报告
    assert _judge(c) == []
    v1 = TE.v1_report(c, False)
    assert v1["pass"] is False and v1["failures"]["merge"] and v1["failures"]["restricted"]


def test_e5v2_exact_keeps_v1_exact_rules():
    """精确模式（CPU fp32）：沿用 v1 的精确规则，不放宽——GPU 判据下能过的一处分歧，在精确模式下判不过。"""
    c = _checks()
    assert _judge(c) == [] and _fails(TE.judge_e5v2(c, exact=True))  # 合成读数里有分歧：GPU 下过，精确模式不过
    same = copy.deepcopy(c)
    for _, path, name, is_gen in TE.E5V2_GAP_COMPARISONS:
        node = _node(same, path, name)
        node["gap"] = _gap(0)
        if is_gen:
            node.update(exact_rows=1.0, token_agreement=1.0, n_divergent_rows=0)
        else:
            node.update(n_disagreements=0, **{("agreement" if "agreement" in node else "argmax_agreement"): 1.0})
    assert TE.judge_e5v2(same, exact=True) == {"merge": [], "restricted": []}
    cti = load_cti()
    for path, name, field in ((("restricted",), "decode_vs_generate_batch", "exact_rows"), (("merge",), "generate", "token_agreement")):
        x = copy.deepcopy(same)
        _node(x, path, name)[field] = 0.999
        crit = cti.criteria_block(True)
        crit["restricted_grid"] = cti.grid_criteria_block(True)
        assert TE.judge_e5v2(x, exact=True) == cti.judge(x, crit) != {"merge": [], "restricted": []}


# ---------------------------------------------------------------------------
# 指标层面的汇总（纯函数）
# ---------------------------------------------------------------------------


def _row(task: str, ia: float, ib: float, ra: float | None = None, rb: float | None = None, headline: bool = True,
         same: bool = True) -> dict:
    fmt = task_format(task)
    side = lambda iou, rel: {"valid": True, "failure": None, "iou_grid": iou, "iou_full": iou, "relief_error": rel,  # noqa: E731
                             "relief_reason": None}
    return {"id": f"{task}/val/0", "task": task, "format": fmt, "bin": 0, "headline": headline, "a": side(ia, ra), "b": side(ib, rb),
            "identical": same}


def test_metric_level_summary_flags():
    rows = [_row("occlusion", 0.5, 0.5), _row("physics", 0.4, 0.4), _row("rotation3d", 0.3, 0.3, 0.8, 0.8, headline=False),
            _row("viewpoint", 0.2, 0.2, 0.6, 0.6)]
    s = TE.metric_level_summary(rows, E5_TASKS)
    assert s["overall"]["all"]["iou_grid"]["diff"] == 0.0 and s["flags"]["iou"] is False and s["flags"]["relief"] is False
    assert s["overall"]["all"]["n_identical"] == 4 and s["overall"]["headline"]["n"] == 3
    assert s["by_task"]["imagery"]["all"]["n"] == 0 and s["by_task"]["imagery"]["all"]["iou_grid"]["diff"] is None
    assert s["by_task"]["occlusion"]["all"]["relief_error"] is None and s["by_format"]["relief"]["all"]["relief_error"]["n_both_valid"] == 2
    # iou：总体平均差 0.02 / 4 行 = 0.005 → 不标记；0.08 / 4 = 0.02 → 标记
    rows2 = copy.deepcopy(rows)
    rows2[0]["b"]["iou_grid"] = 0.52
    assert TE.metric_level_summary(rows2, E5_TASKS)["flags"]["iou"] is False
    rows2[0]["b"]["iou_grid"] = 0.58
    s2 = TE.metric_level_summary(rows2, E5_TASKS)
    assert s2["flags"]["iou"] is True and s2["overall"]["all"]["iou_grid"]["max_abs_row_diff"] == pytest.approx(0.08)
    # 起伏：只取两侧都有效的行；平均差 0.03 → 标记，0.01 → 不标记
    rows3 = copy.deepcopy(rows)
    rows3[2]["b"]["relief_error"] = 0.86
    assert TE.metric_level_summary(rows3, E5_TASKS)["flags"]["relief"] is True  # (0.06 + 0) / 2 = 0.03
    rows3[2]["b"]["relief_error"] = 0.82
    assert TE.metric_level_summary(rows3, E5_TASKS)["flags"]["relief"] is False  # 0.01
    rows3[3]["b"]["relief_error"] = None  # B 无效的行不算进起伏误差
    s3 = TE.metric_level_summary(rows3, E5_TASKS)
    assert s3["overall"]["all"]["relief_error"]["n_both_valid"] == 1 and s3["overall"]["all"]["relief_error"]["diff"] == \
        pytest.approx(0.02)
    rows4 = [r for r in rows if r["format"] == "binary"]
    assert TE.metric_level_summary(rows4, E5_TASKS)["flags"]["relief"] is None  # 没有起伏的行：判不了


def test_metric_level_line():
    assert "警告" in TE.metric_level_line({"error": "boom"})
    line = TE.metric_level_line({"status": "done", "n": 40, "n_identical": 38, "iou_grid_diff": -0.001, "relief_error_diff": 0.0,
                                 "flags": {"iou": False, "relief": False}, "error": None})
    assert "-0.0010" in line and "38/40" in line and "flags.iou=False" in line


# ---------------------------------------------------------------------------
# 旧文件改名留存
# ---------------------------------------------------------------------------


def test_preserve_previous(tmp_path):
    out = tmp_path / TE.MERGE_CHECK
    assert TE.preserve_previous(out, "e5-v2") is None  # 没有旧文件
    out.write_text(json.dumps({"criteria_version": "e5-v2", "pass": False}))
    assert TE.preserve_previous(out, "e5-v2") is None and out.exists()  # 同一版本：不动（重试照旧覆盖）
    names = []
    for k in range(3):
        out.write_text(json.dumps({"criteria_version": "v1", "k": k}))
        raw = out.read_bytes()
        info = TE.preserve_previous(out, "e5-v2")
        names.append(info["renamed_to"])
        assert info["criteria_version"] == "v1" and not out.exists() and (tmp_path / info["renamed_to"]).read_bytes() == raw
    assert names == ["merge_check.v1.json", "merge_check.v1.1.json", "merge_check.v1.2.json"]
    out.write_text("{not json")
    assert TE.preserve_previous(out, "e5-v2")["renamed_to"] == "merge_check.unknown.json"


def test_reusable_only_e5v2_passed():
    key = {"a": 1}
    good = {"format": TE.MERGE_CHECK_FORMAT, "pass": True, "error": None, "key": key, "criteria_version": "e5-v2"}
    assert TE._reusable(good, key)
    assert not TE._reusable({**good, "criteria_version": "v1"}, key)  # 旧的 v1 文件过了也不复用
    assert not TE._reusable({**good, "pass": False}, key) and not TE._reusable({**good, "key": {"a": 2}}, key)
    assert not TE._reusable({}, key) and not TE._reusable({**good, "error": {"repr": "x"}}, key)


# ---------------------------------------------------------------------------
# tiny 端到端
# ---------------------------------------------------------------------------


def _tiny(key: str):
    t = CFG.e3b.doc["tiny"]
    fam = registry.get_model(key).family
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _write_e5_checkpoint(final: Path, key: str, mutate) -> Path:
    """base 起点的 E5 检查点（同 tests/test_e5_train.test_bad_e5_lora_refused 的写法）；mutate(ad, trainer) 改 LoRA 与网格 token 行。"""
    ad = _tiny(key)
    start = STK.resolve_start(CFG, "base", None, model_key=key, tiny=True)
    tr = E5Trainer(ad, CFG, resolve_run(CFG, n_train=10, n_examples=10, tiny=True), seed=0, start=start)
    tr.setup_fresh()
    mutate(ad, tr)
    ad.save_trained(final / "model")
    ts = TrainSet.build(CFG, 10, None, 0)
    STK.write_stack_file(final, {"model_dir": "model", "start": start.record(), "seed": 0, "tag": ts.tag, "data": ts.identity(),
                                 "fingerprint": {"value": tr.stack_info["fingerprint"], "device_key": STK.device_key(ad)},
                                 "identity_sha256": "test", "config": CFG.identity(), "tokens_e5": tr.e5.layout()})
    return final


def _random_lora(ad, tr) -> None:
    """随机的 E5 LoRA（lora_B ~ N(0, 0.05²)）与随机的网格 token 行：输出各异，挂不挂 LoRA 输出不同；fp32 上合并前后一致。"""
    g = torch.Generator().manual_seed(5)
    mods = dict(ad.model.named_modules())
    with torch.no_grad():
        for name in ad.lora_target_modules():
            w = mods[name].lora_B["default"].weight
            w.copy_(torch.randn(tuple(w.shape), generator=g) * 0.05)
        idx = torch.tensor(sorted(tr.e5.tokens.grid_ids))
        std = float(CFG.e3b.doc["tiny"]["init_std"])
        for p in ad.trainable_token_rows(list(CFG.e3b.vocab.names)):
            p[idx] = torch.randn(len(idx), p.shape[1], generator=g) * std


def _pathological(ad, tr) -> None:
    """同 tests/test_e5_train._pathological_lora：A 的两行几乎相同、B 的两列相反且放大 1e7；合并与不合并分家，fp32 上也不过。"""
    mods = dict(ad.model.named_modules())
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
            u = torch.randn(a_w.shape[1], generator=g)
            a_w.zero_()
            b_w.zero_()
            a_w[0] = u
            a_w[1] = u + 1e-6 * torch.randn(a_w.shape[1], generator=g)
            v = torch.randn(b_w.shape[0], generator=g)
            b_w[:, 0] = v * 1e7
            b_w[:, 1] = -v * 1e7


@pytest.fixture(scope="module")
def e5ck(tmp_path_factory):
    root = tmp_path_factory.mktemp("e5v2_ck")
    return _write_e5_checkpoint(root / "final", FAST, _random_lora)


def _mdir(out: Path, init: str = "base", key: str = FAST) -> Path:
    return out / "trained" / init / key / "n10" / "seed0"


def _load(final: Path):
    st = STK.load_stacked(final, device="cpu", dtype=torch.float32, start_override=None, log=lambda m: None)
    ad = st.adapter
    return ad, E5Tokens(attach_vocab(ad, CFG.e3b.vocab), CFG.e3b.allowed_scope)


def test_metric_level_exact_and_lora_in_path(e5ck, tmp_path):
    """直接调 stacked_merge_check（CPU fp32）：A 侧真的挂着 LoRA（逐模块审计；挂与不挂输出不同），两侧读数逐个相同、差恰好为 0。"""
    ad, e5tok = _load(e5ck)
    pf, n = CFG.prompt_format(), CFG.grid_n
    samples = [make_sample(it) for it in TE.merge_check_items(1, E5_TASKS)]
    grp = [s for s in samples if task_format(s.task) == "relief"]
    inp = [ad.build_inputs(sample_images(s), pf.render(s, n), pf.order) for s in grp]
    with TE.LoraAudit(ad) as au:
        on = decode_grids(ad, e5tok, inp, "relief", n, decode=STRUCTURED)
    audit = au.result()
    assert audit["n_modules"] == audit["n_called"] == len(ad.lora_target_modules()) and audit["min_calls"] > 0
    assert audit["n_not_called"] == 0
    with ad.peft_model.disable_adapter():  # 绕开 LoRA：审计报错，输出也不同
        with TE.LoraAudit(ad) as au2:
            off = decode_grids(ad, e5tok, inp, "relief", n, decode=STRUCTURED)
    with pytest.raises(RuntimeError, match="没走 LoRA 分支"):
        au2.result()
    with TE.LoraAudit(ad) as au3:  # 一个 LoRA 层都没调用（这里什么都不解码）
        pass
    with pytest.raises(RuntimeError, match="一个都没被调用"):
        au3.result()
    assert [list(r.token_ids) for r in on] != [list(r.token_ids) for r in off]  # LoRA 确实改变输出（这个测试不平凡）
    key = {"test": "metric_level"}
    out = tmp_path / "mc" / TE.MERGE_CHECK
    summ = TE.stacked_merge_check(ad, e5tok, pf, n=n, n_per_task=1, tasks=E5_TASKS, batch_size=4, tiny=True, exact=True, key=key,
                                  out_json=out)
    assert ad.lora_state == "merged" and summ["pass"] is True and summ["criteria_version"] == "e5-v2" and summ["v1_pass"] is True
    with pytest.raises(RuntimeError, match="找不到任何 LoRA 层"):  # 合并之后没有 LoRA 层
        with TE.LoraAudit(ad) as au4:
            pass
        au4.result()
    doc = json.loads(out.read_text())
    assert doc["criteria_version"] == "e5-v2" and doc["criteria"]["version"] == "e5-v2" and doc["criteria"]["exact"] is True
    assert doc["failures"] == {"merge": [], "restricted": []} and doc["checks"]["merge"]["pass"] is True
    v1 = doc["v1_report_only"]
    assert v1["criteria_version"] == "v1" and v1["pass"] is True and v1["failures"] == {"merge": [], "restricted": []}
    ml = doc["metric_level"]
    assert ml["status"] == "done" and ml["error"] is None and ml["report_only"] is True and ml["affects_pass"] is False
    assert ml["sides"]["a"]["lora_state"] == "injected" and ml["sides"]["b"]["lora_state"] == "merged"
    assert ml["sides"]["a"]["lora_audit"]["min_calls"] > 0 and set(ml["timing_s"]) == {"a_s", "b_s"}
    assert [r["id"] for r in ml["rows"]] == doc["data"]["sample_ids"] and len(ml["rows"]) == 5
    for r in ml["rows"]:
        assert r["identical"] and r["same_grid"] and r["n_grid_cells_differ"] == 0 and r["a"] == r["b"]
        assert r["diff"]["iou_grid"] == 0.0 and r["diff"]["iou_full"] == 0.0
        if r["format"] == "relief":
            assert r["same_relief"] and r["n_relief_cells_differ"] == 0 and r["diff"]["relief_error"] == 0.0
    for scope in ("all", "headline"):
        b = ml["overall"][scope]
        assert b["iou_grid"]["diff"] == 0.0 and b["iou_grid"]["max_abs_row_diff"] == 0.0 and b["n_identical"] == b["n"]
    rel = ml["overall"]["all"]["relief_error"]
    assert rel["n_both_valid"] == 2 and rel["diff"] == 0.0 and rel["max_abs_row_diff"] == 0.0
    for t in E5_TASKS:
        assert ml["by_task"][t]["all"]["n"] == 1 and ml["by_task"][t]["all"]["iou_grid"]["diff"] == 0.0
    assert ml["flags"]["iou"] is False and ml["flags"]["relief"] is False
    assert len({tuple(r["a"].values()) for r in ml["rows"]}) > 1  # 各行读数不全相同
    # A 侧的解码就是刚才挂着 LoRA 的那次（同一个 decode_grids 调用）：起伏两行的读数与 on 的评分相同
    recs = {s.task: TE.record(s, pf.render(s, n), r, e5tok, n)["score"] for s, r in zip(grp, on)}
    for r in ml["rows"]:
        if r["format"] == "relief":
            assert r["a"]["iou_grid"] == recs[r["task"]]["iou_grid"] and r["a"]["relief_error"] == recs[r["task"]]["relief"]["error"]
    # 已经通过的 e5-v2：复用（只做 merge_lora）
    ad2, e5tok2 = _load(e5ck)
    again = TE.stacked_merge_check(ad2, e5tok2, pf, n=n, n_per_task=1, tasks=E5_TASKS, batch_size=4, tiny=True, exact=True, key=key,
                                   out_json=out)
    assert again["reused"] is True and ad2.lora_state == "merged" and again["metric_level"]["iou_grid_diff"] == 0.0


def test_metric_level_error_does_not_affect_pass(e5ck, tmp_path, monkeypatch):
    """指标层面出错：错误写进 metric_level.error，核对照常判（通过）。"""
    ad, e5tok = _load(e5ck)

    def boom(*a, **kw):
        raise RuntimeError("指标层面的人为错误")

    monkeypatch.setattr(TE, "_ml_decode", boom)
    out = tmp_path / TE.MERGE_CHECK
    summ = TE.stacked_merge_check(ad, e5tok, CFG.prompt_format(), n=CFG.grid_n, n_per_task=1, tasks=E5_TASKS, batch_size=4, tiny=True,
                                  exact=True, key={"k": 1}, out_json=out)
    doc = json.loads(out.read_text())
    assert summ["pass"] is True and doc["pass"] is True and doc["error"] is None
    assert doc["metric_level"]["status"] == "error" and "人为错误" in doc["metric_level"]["error"]["repr"]
    assert "警告" in TE.metric_level_line(summ["metric_level"])


def test_check_only_cli(e5ck, tmp_path, capsys):
    """--check-only：已有的 v1 文件改名留存；不建任务目录；通过退出 0（给了 --task 也一样）；再跑一次复用。"""
    ev = _script("eval_e5_trained")
    out = tmp_path / "o"
    mdir = _mdir(out)
    mdir.mkdir(parents=True)
    v1 = {"format": TE.MERGE_CHECK_FORMAT, "criteria_version": "v1", "pass": True, "error": None, "key": {}, "checks": {}}
    (mdir / TE.MERGE_CHECK).write_text(json.dumps(v1))
    raw = (mdir / TE.MERGE_CHECK).read_bytes()
    capsys.readouterr()
    assert ev.main(["--tiny", "--check-only", "--checkpoint", str(e5ck), "--model", FAST, "--out", str(out), "--task", "c"]) == 0
    cap = capsys.readouterr()
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json", "merge_check.v1.json"]  # 没有任何任务目录
    assert (mdir / "merge_check.v1.json").read_bytes() == raw  # 证据原样留存
    assert sorted(p.name for p in (out / "trained").iterdir()) == ["base"]
    doc = json.loads((mdir / TE.MERGE_CHECK).read_text())
    assert doc["criteria_version"] == "e5-v2" and doc["pass"] is True and doc["previous"] == {"criteria_version": "v1",
                                                                                              "renamed_to": "merge_check.v1.json"}
    assert doc["metric_level"]["status"] == "done" and doc["v1_report_only"]["pass"] is True
    lines = cap.out.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("check-only trained base") and "criteria=e5-v2 pass=True" in lines[0]
    assert "previous=merge_check.v1.json" in lines[0] and "[合并核对·指标层面" in cap.err
    sha = _sha(mdir / TE.MERGE_CHECK)
    assert ev.main(["--tiny", "--check-only", "--checkpoint", str(e5ck), "--model", FAST, "--out", str(out)]) == 0
    assert "reused=True" in capsys.readouterr().out and _sha(mdir / TE.MERGE_CHECK) == sha  # 复用，不改写
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json", "merge_check.v1.json"]
    with pytest.raises(SystemExit):  # 只用于 --checkpoint
        ev.parse_args(["--tiny", "--check-only", "--start-only", "--init", "e3b", "--start-checkpoint", str(e5ck), "--model", FAST,
                       "--seed", "0", "--out", str(out)])


def test_check_only_cli_fail_exit1(tmp_path, capsys):
    """坏的 E5 LoRA（同 test_bad_e5_lora_refused）：--check-only 退出 1（给不给 --task 都一样），只有 merge_check.json。"""
    key = FAST
    final = _write_e5_checkpoint(tmp_path / "final", key, _pathological)
    ev = _script("eval_e5_trained")
    out = tmp_path / "o"
    assert ev.main(["--tiny", "--check-only", "--checkpoint", str(final), "--model", key, "--out", str(out), "--task", "a", "c"]) == 1
    mdir = _mdir(out, key=key)
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json"]
    doc = json.loads((mdir / TE.MERGE_CHECK).read_text())
    assert doc["pass"] is False and doc["criteria_version"] == "e5-v2" and (doc["failures"]["merge"] or doc["failures"]["restricted"])
    assert doc["v1_report_only"]["pass"] is False and doc["metric_level"]["report_only"] is True
    cap = capsys.readouterr()
    assert "pass=False" in cap.out and "合并核对没有通过（--check-only）" in cap.err
    # 同一版本的重试：照旧覆盖，不改名
    assert ev.main(["--tiny", "--check-only", "--checkpoint", str(final), "--model", key, "--out", str(out)]) == 1
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json"]


def test_structured_eval_identical_to_4bd7f13(e5ck, tmp_path, capsys):
    """正常评测（默认的结构强制）：改动前（4bd7f13）的脚本加模块对改动后的，同一进程、同一组参数，确定的输出（predictions、metrics、
    README、config.yaml）与打印的结果行逐字节相同、运行记录的字段相同；合并核对里改动前就有的字段逐个相同，v1 的判定照旧（v1_report_only），
    另有 e5-v2 与 metric_level；stderr 有指标层面的一行。"""
    for name, sha in SNAP_SHA256.items():
        assert _sha(SNAP / name) == sha, name
    old_te = _load_path(SNAP / "trained_eval_4bd7f13.py", "_e5_trained_eval_4bd7f13")
    old_te.REPO_ROOT = TE.REPO_ROOT  # 快照按自己的位置推出的仓库根不对（只影响 meta.json 里配置文件的路径）
    old_script = _load_path(SNAP / "eval_e5_trained_4bd7f13.py", "_eval_e5_trained_4bd7f13")
    old_script.TE = old_te
    capsys.readouterr()
    outs, printed, errs = [], [], []
    for k, mod in enumerate((old_script, _script("eval_e5_trained"))):
        o = tmp_path / f"o{k}"
        assert mod.main(["--tiny", "--checkpoint", str(e5ck), "--model", FAST, "--out", str(o), "--limit", "1"]) == 0
        cap = capsys.readouterr()
        printed.append(cap.out)
        errs.append(cap.err)
        outs.append(_mdir(o))
    assert printed[0] == printed[1] and printed[0].count("\n") == len(E5_TASKS) and "decode=structured" in printed[0]
    assert "[合并核对·指标层面" not in errs[0] and "[合并核对·指标层面，只报告]" in errs[1]
    for task in E5_TASKS:
        leaf = f"{task}__structured"
        for f in DET:
            assert (outs[0] / leaf / f).read_bytes() == (outs[1] / leaf / f).read_bytes(), (task, f)
        r0, r1 = (json.loads((d / leaf / "logs" / "run_001.json").read_text()) for d in outs)
        assert list(r0) == list(r1)
        assert r1["merge_check"]["criteria_version"] == "e5-v2" and r1["merge_check"]["metric_level"]["n"] == 5
    m0, m1 = (json.loads((d / TE.MERGE_CHECK).read_text()) for d in outs)
    for m in (m0, m1):
        m.pop("timing_s")
        m.pop("written_utc")
    assert m0["criteria_version"] == "v1" and m1["criteria_version"] == "e5-v2" and m0["pass"] is m1["pass"] is True
    v1 = m1["v1_report_only"]
    assert (v1["criteria_version"], v1["failures"], v1["pass"], v1["criteria"]) == (m0["criteria_version"], m0["failures"],
                                                                                   m0["pass"], m0["criteria"])
    same = set(m0) - {"criteria_version", "criteria", "failures", "pass", "rule"}
    assert same == {"format", "key", "checks", "data", "generation", "error", "allowed"}
    assert {k: m0[k] for k in same} == {k: m1[k] for k in same}
    assert set(m1) - set(m0) == {"v1_report_only", "metric_level", "previous"}
    assert m1["metric_level"]["status"] == "done" and m1["metric_level"]["overall"]["all"]["iou_grid"]["diff"] == 0.0
