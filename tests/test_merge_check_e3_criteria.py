"""DEV-0063（D102-2 = N37(b)）：E3 的结构强制网格评测可以选用 e5-v2 合并核对判据（scripts/eval_e3b.py --merge-criteria；
src/e3/merge_check.run_merge_check(criteria=…) 与 rejudge_e5v2；src/e3/evaluate.resolve_eval(merge_criteria=…)；scripts/mgr/interference.py
的判据核对）。CPU fp32 小模型。

- 默认 v1 逐字节不变：新脚本与改动前（8207cec 的 scripts/eval_e3b.py、src/e3/evaluate.py、src/e3/merge_check.py，原样存在
  tests/fixtures/e3/）在同一进程、同一组参数下（默认的受限解码、全部 K；结构强制 --k-values 1，即正式口径），确定的输出、
  merge_check.json（去掉写入时刻与耗时，--out 换成占位）、运行记录里合并核对摘要的键与打印行逐字节相同；
- e5-v2 在 tiny 检查点上跑通：merge_check.json 是 e5-v2（v1 的判定原样在 v1_report_only），读数与 v1 的同一份核对相同；运行身份、
  metrics.json、README、打印行记判据；评测本身（predictions、scores）与 v1 逐字节相同；续跑复用 e5-v2 的核对；
- 判据确实不同：
  * 真实文件（tests/fixtures/e3/merge_checks_e3_grid.json，31 份 E3b 型检查点的网格核对）：v1 复现记录的判定，只有 3 份被拒（2B E3b
    种子 1 在 B200 与 A100 上各一次、4B E3b 种子 2），都只是关思考的 exact_rows（平局翻转）；e5-v2 全部通过；
  * 构造的平局翻转（tiny 检查点上模拟 GPU：非精确模式，关思考的 exact_rows = 2/3、分歧处 gap 很小）：v1 拒绝评测（退出码 1，不建任务
    目录），e5-v2 通过并评测；反方向（合并前后生成的分歧处 gap 2.5：v1 只报告、e5-v2 判不过）e5-v2 拒绝；
  * 判法就是 src/e5/trained_eval.judge_document_e5v2（换掉它，结果跟着变）；
- 互不复用、不覆盖：v1 的文件不当 e5-v2 用，反过来也一样；同一个 --out 里已有另一版判据的 merge_check.json → 用法错误、什么都不写；
  已有叶目录换判据续跑拒绝（运行身份不同）；指令检查的正常一侧判据不同不配对；
- 规则：e5-v2 只配结构强制；没有 LoRA 的检查点、检查项不全都拒绝；
- interference.py：补评目录放在最前、先到先得；两臂判据不同的一对不算 P；全是 v1 时输出不变。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.e3 import evaluate as EV
from src.e3 import merge_check as MC
from src.e3.config import load_config
from src.e3.merge_check import load_cti
from src.e5 import trained_eval as TE

REPO = Path(__file__).resolve().parents[1]
FAM = "qwen3.5-0.8b"
CFG = load_config()
FIX = REPO / "tests" / "fixtures" / "e3" / "merge_checks_e3_grid.json"
SNAP_DIR = REPO / "tests" / "fixtures" / "e3"
SNAP_SHA256 = {  # 8207cec（DEV-0063 之前）的文件原样
    "eval_e3b_8207cec.py": "ac1e6d0438181edec272b4aaca010f5e1911d41254a1779f6d95360c1071d05c",
    "evaluate_8207cec.py": "e03c02e107d56c3cf30c92c7fa7f85deeb4fe9a16e5e2abd1a8b928ebceca228",
    "merge_check_8207cec.py": "5cfaaf20041d33421776fa47889fdeab43b03d2025ff7a36c19a8ef8185411a8",
}
DET = ("README.md", "config.yaml", "metrics.json", "scores.jsonl")
STRUCT = ["--k-values", "1", "--decode", "structured"]  # 正式口径（RUN-0052 的 eval.sh 的 grid()）
E5V2 = ["--merge-criteria", "e5-v2"]
REJECTED_V1 = ("experiments/E3/0046_e3_eval_2b_h1/eval/e3b/qwen3.5-2b/seed1/merge_check.json",  # B200，RUN-0046
               "experiments/E3/0052_hpc_eval_4b_hpcgpu06/eval/e3b/qwen3.5-4b/seed2/merge_check.json",  # hpcgpu06，RUN-0052
               "experiments/E5/0048_hpc_measure_hpcgpu07/train/e3b/qwen3.5-2b/start_checks/merge_check_7c3d293818e37788.json")  # 07
_SCRIPTS: dict[str, object] = {}


def _load_script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_dev0063", REPO / "scripts" / f"{name}.py")
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


ITF = _load_path(REPO / "scripts" / "mgr" / "interference.py", "_interference_under_test_dev0063")


def _args(ck: Path, out: Path, *extra: str) -> list[str]:
    return ["--tiny", "--checkpoint", str(ck), "--model", FAM, "--seed", "0", "--out", str(out), *extra]


def _mdir(out: Path) -> Path:
    return out / "eval" / "e3b" / FAM / "seed0"


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """训练框架（scripts/train_e3b.py --tiny）产出的小检查点（Qwen，LoRA + 网格 token 行）。"""
    d = tmp_path_factory.mktemp("tiny_train_dev0063")
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "train_e3b.py"), "--tiny", "--model", FAM, "--supervision", "grid",
                        "--seed", "0", "--out", str(d)], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr[-3000:]
    return d / "e3b" / FAM / "seed0" / "ckpt" / "final" / "model"


# ---------------------------------------------------------------------------
# 约定与规则
# ---------------------------------------------------------------------------


def test_criteria_names_are_the_two_scripts_versions():
    assert MC.CRITERIA == ("v1", "e5-v2") and MC.DEFAULT_CRITERIA == "v1"
    assert MC.V1 == load_cti().CRITERIA_VERSION and MC.E5V2 == TE.CRITERIA_VERSION  # 版本名就是两边脚本里的
    assert ITF.DEFAULT_CRITERIA == MC.DEFAULT_CRITERIA
    for s in ("D102-2", "N37(b)", "N34", "决定于看到被拒种子的读数之前"):
        assert s in MC.E5V2_E3_BASIS
    assert "E3 的结构强制网格评测" in MC.E5V2_APPLIES_TO and "judge_e5v2" in MC.E5V2_APPLIES_TO


def test_resolve_eval_merge_criteria_rules():
    base = EV.resolve_eval(CFG, tiny=True)
    assert "merge_criteria" not in base and EV.resolve_eval(CFG, tiny=True, merge_criteria="v1") == base
    st = EV.resolve_eval(CFG, tiny=True, decode="structured")
    assert EV.resolve_eval(CFG, tiny=True, decode="structured", merge_criteria="v1") == st and EV.merge_criteria_version(st) == "v1"
    s2 = EV.resolve_eval(CFG, tiny=True, decode="structured", merge_criteria="e5-v2")
    assert {k: v for k, v in s2.items() if k != "merge_criteria"} == st
    assert s2["merge_criteria"] == {"version": "e5-v2", "rule": EV.MERGE_CRITERIA_RULE, "basis": MC.E5V2_E3_BASIS}
    assert EV.merge_criteria_version(s2) == "e5-v2" and "merge_criteria" in EV.identity_settings(s2)  # 进运行身份
    with pytest.raises(ValueError, match="结构强制"):
        EV.resolve_eval(CFG, tiny=True, merge_criteria="e5-v2")
    with pytest.raises(ValueError, match="结构强制"):
        EV.resolve_eval(CFG, tiny=True, decode="restricted", merge_criteria="e5-v2")
    with pytest.raises(ValueError, match="merge_criteria"):
        EV.resolve_eval(CFG, tiny=True, decode="structured", merge_criteria="v2")
    assert "merge_criteria" in EV.NORMAL_SAME_EVAL


def test_cli_merge_criteria_args(tmp_path):
    ev = _load_script("eval_e3b")
    base = ["--tiny", "--checkpoint", str(REPO), "--model", FAM, "--seed", "0", "--out", str(tmp_path / "x")]  # parse_args 不读检查点
    assert ev.parse_args(base).merge_criteria == "v1"
    assert ev.parse_args(base + STRUCT + E5V2).merge_criteria == "e5-v2"
    assert ev.parse_args(base + ["--decode", "structured", "--merge-criteria", "v1"]).merge_criteria == "v1"
    for extra in (E5V2, E5V2 + ["--decode", "restricted"], ["--merge-criteria", "v2"]):
        with pytest.raises(SystemExit) as e:
            ev.parse_args(base + extra)
        assert e.value.code == 2, extra
    assert not (tmp_path / "x").exists()


def test_run_merge_check_refusals(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "manifest.json").write_text(json.dumps({"lora": None}), encoding="utf-8")
    out = tmp_path / "mc.json"
    kw = dict(model_key=FAM, tiny=True, n=3, out_json=out, device="cpu", dtype="fp32")
    with pytest.raises(ValueError, match="LoRA"):  # 没有 LoRA（C1′）：judge_e5v2 没有 merge 一节可判
        MC.run_merge_check(ck, checks=["merge", "restricted"], criteria="e5-v2", **kw)
    (ck / "manifest.json").write_text(json.dumps({"lora": {"r": 4}}), encoding="utf-8")
    with pytest.raises(ValueError, match="同时核对"):
        MC.run_merge_check(ck, checks=["merge"], criteria="e5-v2", **kw)
    with pytest.raises(ValueError, match="判据"):
        MC.run_merge_check(ck, checks=["merge", "restricted"], criteria="v3", **kw)
    assert not out.exists()
    # 输出位置已有另一版判据的文件：不覆盖（两个方向）
    for old, new in (("v1", "e5-v2"), ("e5-v2", "v1")):
        out.write_text(json.dumps({"criteria_version": old, "pass": True}), encoding="utf-8")
        before = out.read_bytes()
        with pytest.raises(MC.MergeCriteriaConflict, match="新的 --out"):
            MC.run_merge_check(ck, checks=["merge", "restricted"], criteria=new, **kw)
        assert out.read_bytes() == before
        assert MC.existing_criteria(out) == old
    out.write_text("{not json", encoding="utf-8")  # 读不出来的旧文件不算冲突（与改动前相同：重跑、覆盖）
    assert MC.existing_criteria(out) is None
    MC.check_criteria_conflict(out, "e5-v2")
    MC.check_criteria_conflict(tmp_path / "none.json", "v1")


# ---------------------------------------------------------------------------
# 真实文件的回归（E3 网格评测前的全部核对）
# ---------------------------------------------------------------------------


def _fixture() -> dict:
    return _json(FIX)


def _doc(e: dict) -> dict:
    """fixture 的一条 → 那个脚本写出的 JSON 的形状（判定字段原样）。"""
    cti = load_cti()
    crit = cti.criteria_block(e["exact"])
    crit["restricted_grid"] = cti.grid_criteria_block(e["exact"])
    return {"format": cti.FORMAT, "criteria_version": e["criteria_version"], "criteria": crit, "checks": copy.deepcopy(e["checks"]),
            "failures": e["failures"], "pass": e["pass"], "error": e["error"]}


def test_fixture_covers_real_grid_checks():
    doc = _fixture()
    files = doc["files"]
    assert doc["n_files"] == len(files) == 31 and doc["source_commit"] == "8207cec"
    assert len({e["path"] for e in files}) == 31
    for e in files:
        assert e["format"] == load_cti().FORMAT and e["criteria_version"] == "v1" and e["exact"] is False and e["error"] is None
        assert sorted(e["checks"]) == ["merge", "restricted"] and "across_merge" in e["checks"]["restricted"]


@pytest.mark.parametrize("k", range(31))
def test_real_file_v1_reproduced_e5v2_passes(k):
    e = _fixture()["files"][k]
    d = _doc(e)
    assert load_cti().judge(copy.deepcopy(e["checks"]), d["criteria"]) == e["failures"], e["path"]  # v1 复现记录的判定
    new = MC.rejudge_e5v2(d)
    assert new["criteria_version"] == "e5-v2" and new["pass"] is True, (e["path"], new["failures"])
    assert new["v1_report_only"] == {"criteria_version": "v1", "report_only": True, "failures": e["failures"], "pass": e["pass"],
                                     "criteria": d["criteria"]}
    assert new["v1_report_only"] == TE.v1_report(e["checks"], e["exact"])  # 与 E5 的"v1 只报告"同一套


def test_real_files_only_tie_flips_rejected():
    files = _fixture()["files"]
    assert sorted(e["path"] for e in files if not e["pass"]) == sorted(REJECTED_V1)
    for e in files:
        if e["pass"]:
            continue
        assert e["failures"]["merge"] == [] and len(e["failures"]["restricted"]) == 1, e["path"]
        assert e["failures"]["restricted"][0].startswith("restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows")
        bt = e["checks"]["restricted"]["by_thinking"]["thinking_off"]
        assert bt["decode_vs_generate_batch"]["exact_rows"] < 0.95 and bt["teacher_forcing"]["agreement"] >= 0.999
        assert TE.judge_e5v2(e["checks"], exact=False) == {"merge": [], "restricted": []}


def test_rejudge_uses_the_e5_judge(monkeypatch):
    """判法不另写：rejudge_e5v2 的判定就是 src/e5/trained_eval.judge_document_e5v2 的（换掉它，结果跟着变）。"""
    e = _fixture()["files"][0]
    d = _doc(e)
    calls = []

    def fake(doc):
        calls.append(doc)
        return {"criteria_version": "e5-v2", "failures": {"merge": ["假的"], "restricted": []}, "pass": False}

    monkeypatch.setattr(TE, "judge_document_e5v2", fake)
    new = MC.rejudge_e5v2(d)
    assert len(calls) == 1 and calls[0]["checks"] == d["checks"]
    assert new["pass"] is False and new["failures"] == {"merge": ["假的"], "restricted": []}
    assert new["checks"]["merge"]["pass"] is False and new["checks"]["restricted"]["pass"] is True
    monkeypatch.undo()
    new = MC.rejudge_e5v2(d)
    ref = TE.e5v2_criteria(False)  # 除了适用范围（E5 的原文写的是"E3 仍是 v1"），判据说明逐项就是 E5 的
    assert {k: v for k, v in new["criteria"].items() if k != "e3_basis"} == {**ref, "applies_to": MC.E5V2_APPLIES_TO}
    assert new["criteria"]["applies_to"] == MC.E5V2_APPLIES_TO and new["criteria"]["e3_basis"] == MC.E5V2_E3_BASIS
    assert d["criteria_version"] == "v1" and "v1_report_only" not in d  # 不改输入
    with pytest.raises(ValueError, match="只重判"):
        MC.rejudge_e5v2(new)


# ---------------------------------------------------------------------------
# 默认 v1：与改动前（8207cec）逐字节相同
# ---------------------------------------------------------------------------


def _old_script():
    for name, sha in SNAP_SHA256.items():
        assert hashlib.sha256((SNAP_DIR / name).read_bytes()).hexdigest() == sha, name
    old_mc = _load_path(SNAP_DIR / "merge_check_8207cec.py", "_e3_merge_check_8207cec")
    old_mc.CTI_PATH = MC.CTI_PATH  # 快照放在 tests/fixtures/e3/ 下，按自己的位置推出的仓库根不对：换成真的
    old_ev = _load_path(SNAP_DIR / "evaluate_8207cec.py", "_e3_evaluate_8207cec")
    for k in ("REPO_ROOT", "FIXTURE_REFSEG_ROOT", "FIXTURE_REFSEG_DEV", "FIXTURE_DEPTH_ROOT"):
        setattr(old_ev, k, getattr(EV, k))
    old = _load_path(SNAP_DIR / "eval_e3b_8207cec.py", "_eval_e3b_8207cec")
    old.EV, old.run_merge_check, old.MergeCheckFailed = old_ev, old_mc.run_merge_check, old_mc.MergeCheckFailed
    return old, old_ev


def _mc_normalized(mdir: Path, out: Path) -> str:
    doc = _json(mdir / "merge_check.json")
    for k in ("written_utc", "timing_s"):  # 写入时刻与耗时每次不同
        doc.pop(k)
    doc["argv"] = [x.replace(str(out), "<out>") for x in doc["argv"]]
    return json.dumps(doc, ensure_ascii=False)


@pytest.mark.parametrize("extra", [[], STRUCT], ids=["restricted_all_K", "structured_k1"])
def test_v1_default_identical_to_8207cec(tiny_ckpt, tmp_path, capsys, extra):
    old, old_ev = _old_script()
    new = _load_script("eval_e3b")
    kv = [1] if extra else None
    dec = "structured" if extra else None
    assert old_ev.resolve_eval(CFG, tiny=True, k_values=kv, decode=dec) == EV.resolve_eval(CFG, tiny=True, k_values=kv, decode=dec)
    outs, prints = [], []
    for k, mod in enumerate((old, new)):
        out = tmp_path / f"o{k}"
        assert mod.main(_args(tiny_ckpt, out, *extra)) == 0
        prints.append(capsys.readouterr().out)
        outs.append(out)
    assert prints[0] == prints[1] and prints[0].count("single/guided") == 2 and "merge_criteria" not in prints[1]
    settings = EV.resolve_eval(CFG, tiny=True, k_values=kv, decode=dec)
    stages = tuple(EV.stage_file(s) for s in EV.stage_names(settings))
    for task in ("refseg", "depth"):
        leaf = EV.leaf_name(task, settings)
        for f in DET + stages:
            assert (_mdir(outs[0]) / leaf / f).read_bytes() == (_mdir(outs[1]) / leaf / f).read_bytes(), (leaf, f)
        r0, r1 = (_json(_mdir(o) / leaf / "logs" / "run_001.json")["merge_check"] for o in outs)
        assert list(r0) == list(r1)  # 运行记录里合并核对摘要的键不变（v1 没有 v1_report_only_pass）
        assert {k: v for k, v in r0.items() if k not in ("file", "file_sha256")} == \
               {k: v for k, v in r1.items() if k not in ("file", "file_sha256")}
    assert _mc_normalized(_mdir(outs[0]), outs[0]) == _mc_normalized(_mdir(outs[1]), outs[1])
    assert sorted(p.name for p in _mdir(outs[0]).iterdir()) == sorted(p.name for p in _mdir(outs[1]).iterdir())


# ---------------------------------------------------------------------------
# e5-v2 在 tiny 检查点上（精确模式：沿用 v1 的精确规则，判定与 v1 相同）
# ---------------------------------------------------------------------------


def test_e5v2_tiny_end_to_end_and_no_cross_reuse(tiny_ckpt, tmp_path, capsys):
    ev = _load_script("eval_e3b")
    v1, e5 = tmp_path / "v1", tmp_path / "e5"
    assert ev.main(_args(tiny_ckpt, v1, *STRUCT)) == 0
    print_v1 = capsys.readouterr().out
    assert ev.main(_args(tiny_ckpt, e5, *STRUCT, *E5V2)) == 0
    print_e5 = capsys.readouterr().out
    assert print_e5.splitlines() == [x + " merge_criteria=e5-v2" for x in print_v1.splitlines()]
    mv, me = _mdir(v1), _mdir(e5)
    dv, de = _json(mv / "merge_check.json"), _json(me / "merge_check.json")
    # merge_check.json：e5-v2 的判定，v1 的判定原样只报告；读数是同一份核对（CPU fp32 确定）
    assert de["criteria_version"] == "e5-v2" and de["pass"] is True and de["failures"] == {"merge": [], "restricted": []}
    assert de["criteria"]["version"] == "e5-v2" and de["criteria"]["exact"] is True
    assert de["criteria"]["thresholds"] == dict(TE.E5V2_THRESHOLDS) and de["criteria"]["applies_to"] == MC.E5V2_APPLIES_TO
    assert de["v1_report_only"] == {"criteria_version": "v1", "report_only": True, "failures": dv["failures"], "pass": dv["pass"],
                                    "criteria": dv["criteria"]}
    assert dv["criteria_version"] == "v1" and "v1_report_only" not in dv
    for sec in ("merge", "restricted"):
        assert de["checks"][sec] == dv["checks"][sec]  # 精确模式下两版判定相同（含 checks.<节>.pass）
    assert [k for k in de if k != "v1_report_only"] == list(dv)
    for task in ("refseg", "depth"):
        lv, le = mv / f"{task}__structured", me / f"{task}__structured"
        for f in ("predictions.jsonl", "scores.jsonl"):  # 评测本身与判据无关
            assert (lv / f).read_bytes() == (le / f).read_bytes(), (task, f)
        iv, ie = (yaml.safe_load((x / "config.yaml").read_text(encoding="utf-8")) for x in (lv, le))
        assert ie["eval"]["merge_criteria"]["version"] == "e5-v2" and "merge_criteria" not in iv["eval"]  # 运行身份记判据
        assert {k: v for k, v in ie["eval"].items() if k != "merge_criteria"} == iv["eval"]
        assert {k: v for k, v in ie.items() if k != "eval"} == {k: v for k, v in iv.items() if k != "eval"}
        m_v, m_e = _json(lv / "metrics.json"), _json(le / "metrics.json")
        assert m_e.pop("merge_criteria") == ie["eval"]["merge_criteria"] and m_e == m_v and "merge_criteria" not in m_v
        assert "评测前合并核对的判据：e5-v2" in (le / "README.md").read_text(encoding="utf-8")
        assert "评测前合并核对的判据" not in (lv / "README.md").read_text(encoding="utf-8")
        run = _json(le / "logs" / "run_001.json")["merge_check"]
        assert run["criteria_version"] == "e5-v2" and run["v1_report_only_pass"] is True and run["pass"] is True
        assert "v1_report_only_pass" not in _json(lv / "logs" / "run_001.json")["merge_check"]
    # 续跑：复用 e5-v2 的核对，结果不变
    keep = DET + ("predictions.jsonl", "merge_check.json")
    before_e5 = {k: v for k, v in _files(e5).items() if Path(k).name in keep}
    assert ev.main(_args(tiny_ckpt, e5, *STRUCT, *E5V2, "--resume")) == 0
    cap = capsys.readouterr()
    assert "复用已通过" in cap.err
    assert {k: v for k, v in _files(e5).items() if Path(k).name in keep} == before_e5
    # 复用规则：判据不同不复用（两个方向）
    kw = dict(manifest_sha=de["checkpoint"]["manifest_sha256"], n=de["data"]["n_requested"], checks=list(de["checks"]), tiny=True,
              dtype="fp32")
    assert MC._reusable(de, **kw, criteria_version="e5-v2") and not MC._reusable(de, **kw, criteria_version="v1")
    assert MC._reusable(dv, **kw, criteria_version="v1") and not MC._reusable(dv, **kw, criteria_version="e5-v2")
    # 互不覆盖：v1 进 e5-v2 的目录、e5-v2 进 v1 的目录（续跑与否都一样）→ 用法错误，什么都不写
    before_e5, before_v1 = _files(e5), _files(v1)
    for args in (_args(tiny_ckpt, e5, *STRUCT, "--resume"), _args(tiny_ckpt, e5, *STRUCT),
                 _args(tiny_ckpt, v1, *STRUCT, *E5V2, "--resume"), _args(tiny_ckpt, v1, *STRUCT, *E5V2)):
        with pytest.raises(SystemExit) as e:
            ev.main(args)
        assert e.value.code == 2
        assert "判据不同的结果互不复用、也不覆盖" in capsys.readouterr().err
    assert _files(e5) == before_e5 and _files(v1) == before_v1
    # 已有的 v1 叶目录（没有核对文件的拷贝）换判据续跑：核对照做，叶目录因运行身份不同拒绝混写，已有的输出不动
    mix = tmp_path / "mix"
    shutil.copytree(v1, mix)
    (_mdir(mix) / "merge_check.json").unlink()
    leaf_before = _files(_mdir(mix) / "refseg__structured")
    assert ev.main(_args(tiny_ckpt, mix, *STRUCT, *E5V2, "--resume", "--task", "refseg")) == 2
    capsys.readouterr()
    errs = sorted((_mdir(mix) / "errors").glob("*.json"))
    assert len(errs) == 1 and "运行身份" in _json(errs[0])["error"] and "'eval'" in _json(errs[0])["error"]
    assert {k: v for k, v in _files(_mdir(mix) / "refseg__structured").items() if not k.startswith("logs/")} == \
           {k: v for k, v in leaf_before.items() if not k.startswith("logs/")}
    # 指令检查的正常一侧：判据不同不配对
    leaf = mv / "refseg__structured"
    ident = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
    ids = [json.loads(x)["id"] for x in (leaf / "predictions.jsonl").read_text(encoding="utf-8").splitlines()][:2]
    EV.load_normal_side(leaf, ident, ids)  # 同一判据照常
    ident2 = copy.deepcopy(ident)
    ident2["eval"]["merge_criteria"] = EV.resolve_eval(CFG, tiny=True, decode="structured", merge_criteria="e5-v2")["merge_criteria"]
    with pytest.raises(ValueError, match="eval.merge_criteria"):
        EV.load_normal_side(leaf, ident2, ids)


# ---------------------------------------------------------------------------
# 判据确实不同：构造的平局翻转（tiny 检查点上模拟 GPU 的非精确模式）
# ---------------------------------------------------------------------------


def _simulate_gpu(monkeypatch, mutate):
    """包住那个脚本的 main：照常跑完核对后，把判据换成非精确模式（GPU / bf16 的 v1），按 mutate 改读数，再按 v1 重判、写回——与那个
    脚本自己的收尾（judge → checks.<节>.pass → pass）相同。读数之外一律不动。"""
    cti = load_cti()
    orig = cti.main

    def wrapped(argv):
        orig(argv)
        out = Path(argv[argv.index("--out") + 1])
        doc = _json(out)
        crit = cti.criteria_block(False)
        crit["restricted_grid"] = cti.grid_criteria_block(False)
        doc["criteria"] = crit
        mutate(doc["checks"])
        doc["failures"] = cti.judge(doc["checks"], crit)
        for name, bad in doc["failures"].items():
            doc["checks"][name]["pass"] = not bad
        doc["pass"] = all(not bad for bad in doc["failures"].values())
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0 if doc["pass"] else 1

    monkeypatch.setattr(cti, "main", wrapped)


def tie_flip(checks):
    """平局翻转（同 N34、N37 的真实情形）：关思考时受限解码与 generate_batch 有一行在近似平局处分岔，分歧处 gap 很小；
    教师强制与合并前后都一致。"""
    g = checks["restricted"]["by_thinking"]["thinking_off"]["decode_vs_generate_batch"]
    n = int(g["n_rows"])
    g.update(exact_rows=(n - 1) / n, n_divergent_rows=1, n_divergence_without_gap=0,
             gap={**g["gap"], "n": 1, "median": 0.04, "max": 0.04})


def large_merge_gap(checks):
    """反方向：开思考时合并前后的自由生成在一处分岔，分歧处 gap 2.5（v1 只报告 merge 的生成；e5-v2 判所有比较的 gap）。"""
    g = checks["merge"]["by_thinking"]["thinking_on"]["generate"]
    n = int(g["n_rows"])
    g.update(exact_rows=(n - 1) / n, n_divergent_rows=1, n_divergence_without_gap=0,
             gap={**g["gap"], "n": 1, "median": 2.5, "max": 2.5})


def test_tie_flip_v1_rejects_e5v2_passes(tiny_ckpt, tmp_path, monkeypatch, capsys):
    ev = _load_script("eval_e3b")
    _simulate_gpu(monkeypatch, tie_flip)
    v1, e5 = tmp_path / "v1", tmp_path / "e5"
    assert ev.main(_args(tiny_ckpt, v1, *STRUCT, "--task", "depth")) == 1  # v1：拒绝评测
    assert "拒绝评测" in capsys.readouterr().err
    dv = _json(_mdir(v1) / "merge_check.json")
    assert dv["criteria_version"] == "v1" and dv["pass"] is False and dv["criteria"]["exact"] is False
    assert dv["failures"]["merge"] == [] and len(dv["failures"]["restricted"]) == 1 and "exact_rows 0.6667 < 0.95" in dv["failures"]["restricted"][0]
    assert sorted(p.name for p in _mdir(v1).iterdir()) == ["merge_check.json"]  # 没有任何任务目录
    assert ev.main(_args(tiny_ckpt, e5, *STRUCT, *E5V2, "--task", "depth")) == 0  # e5-v2：通过并评测
    capsys.readouterr()
    de = _json(_mdir(e5) / "merge_check.json")
    assert de["criteria_version"] == "e5-v2" and de["pass"] is True and de["criteria"]["exact"] is False
    assert de["failures"] == {"merge": [], "restricted": []}
    assert de["v1_report_only"]["pass"] is False and de["v1_report_only"]["failures"] == dv["failures"]
    assert de["checks"]["restricted"] == {**dv["checks"]["restricted"], "pass": True}  # 同样的读数，判定不同
    assert _json(_mdir(e5) / "depth__structured" / "metrics.json")["merge_criteria"]["version"] == "e5-v2"


def test_large_merge_gap_v1_passes_e5v2_rejects(tiny_ckpt, tmp_path, monkeypatch, capsys):
    ev = _load_script("eval_e3b")
    _simulate_gpu(monkeypatch, large_merge_gap)
    v1, e5 = tmp_path / "v1", tmp_path / "e5"
    assert ev.main(_args(tiny_ckpt, v1, *STRUCT, "--task", "depth")) == 0
    capsys.readouterr()
    assert _json(_mdir(v1) / "merge_check.json")["pass"] is True
    assert ev.main(_args(tiny_ckpt, e5, *STRUCT, *E5V2, "--task", "depth")) == 1
    assert "e5-v2 没有通过" in capsys.readouterr().err
    de = _json(_mdir(e5) / "merge_check.json")
    assert de["pass"] is False and de["v1_report_only"]["pass"] is True and de["failures"]["restricted"] == []
    assert de["failures"]["merge"] == ["merge/thinking_on/generate：分歧处 gap 最大 2.5 > 2.0",
                                       "merge/thinking_on/generate：分歧处 gap 中位数 2.5 > 0.5"]
    assert sorted(p.name for p in _mdir(e5).iterdir()) == ["merge_check.json"]


# ---------------------------------------------------------------------------
# scripts/mgr/interference.py：补评目录放在最前；两臂判据不同的一对不算
# ---------------------------------------------------------------------------

MODEL = "qwen3.5-2b"


def _grid(root: Path, arm: str, seed: int, task: str, v: float, criteria: str | None = None) -> None:
    key = "cIoU" if task == "refseg" else "abs_rel"
    d = {"headline": {"procedure": "single", "upsample": "guided", key: {"value": v}}}
    if criteria is not None:
        d["merge_criteria"] = {"version": criteria}
    p = root / "eval" / arm / MODEL / f"seed{seed}" / f"{task}__structured" / "metrics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")


def test_interference_prepend_and_criteria_guard(tmp_path, capsys):
    old, new, half = tmp_path / "0046_old", tmp_path / "0071_new", tmp_path / "0072_half"
    for s in (0, 2):  # 原目录：E3b 种子 1 被拒，没有读数
        _grid(old, "e3b", s, "refseg", 0.5 + 0.01 * s)
    for s in (0, 1, 2):
        _grid(old, "e3b_c3", s, "refseg", 0.3 + 0.01 * s)
    _grid(new, "e3b", 1, "refseg", 0.61, "e5-v2")  # 补评：种子 1 的两臂
    _grid(new, "e3b_c3", 1, "refseg", 0.41, "e5-v2")
    _grid(half, "e3b", 1, "refseg", 0.61, "e5-v2")  # 只补评了一臂
    b0 = ITF.size_block("2B", "B200", MODEL, [str(old)])
    assert sorted(b0["P_seg"]) == [0, 2] and "pair_criteria" not in b0 and "mismatched_criteria" not in b0
    b1 = ITF.size_block("2B", "B200", MODEL, [str(new), str(old)])
    assert sorted(b1["P_seg"]) == [0, 1, 2]
    assert b1["P_seg"][1] == pytest.approx(0.61 - 0.41)  # 两臂都从补评目录读（覆盖原目录里 E3b+C3 种子 1 的读数）
    assert b1["P_seg"][0] == pytest.approx(0.5 - 0.3) and b1["P_seg"][2] == pytest.approx(0.52 - 0.32)
    assert b1["pair_criteria"] == {"refseg": {0: "v1", 1: "e5-v2", 2: "v1"}} and "mismatched_criteria" not in b1
    assert {k: v for k, v in b1.items() if k != "pair_criteria"}.keys() == b0.keys()
    assert ITF.grid_criteria([str(new), str(old)], "e3b_c3", 1, MODEL, "refseg") == "e5-v2"
    assert ITF.grid_criteria([str(old)], "e3b_c3", 1, MODEL, "refseg") == "v1"
    assert ITF.grid_criteria([str(old)], "e3b", 1, MODEL, "refseg") is None
    b2 = ITF.size_block("2B", "B200", MODEL, [str(half), str(old)])
    assert sorted(b2["P_seg"]) == [0, 2] and 1 not in b2["R_seg"]  # 一臂补评、一臂回退到原目录：不算
    assert b2["mismatched_criteria"] == [{"seed": 1, "task": "refseg", "e3b": "e5-v2", "e3b_c3": "v1"}]
    ITF.main(["--size", f"2B:B200:{MODEL}:{half},{old}"])
    out = capsys.readouterr().out
    assert "**警告** 2B 种子 1 refseg" in out
    ITF.main(["--size", f"2B:B200:{MODEL}:{new},{old}"])
    out = capsys.readouterr().out
    assert "网格配对按判据 e5-v2 的 refseg 种子 1" in out and "警告" not in out
    ITF.main(["--size", f"2B:B200:{MODEL}:{old}"])
    assert "判据" not in capsys.readouterr().out  # 全是 v1：输出与改动前相同
