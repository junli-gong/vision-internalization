"""DEV-0048（D99-2 的敏感性分析）：E5 训练后评测的"不合并"开关（scripts/eval_e5_trained.py --unmerged）。

tiny、CPU、fp32，InternVL 的小模型；E5 检查点 = base 起点 + 随机的 E5 LoRA 与网格 token 行（同 tests/test_e5_merge_check_e5v2 的 e5ck，
LoRA 确实改变输出）：
- 同一个检查点、同一组参数，合并评测与 --unmerged 评测：predictions.jsonl 逐字节相同（逐条预测与得分相同；fp32 下合并前后一致）；
  metrics.json 除 merge 字段外完全相同；运行身份只多 lora_merge 一组字段；README 写明；打印行只多 merge=unmerged；叶目录
  <任务>__structured__unmerged；正式评测的叶目录与 merge_check.json 一字节不动；运行记录有 LoRA 审计（逐批）与跳过的理由；
- LoRA 审计生效：禁用 adapter 时报错（run_task 与 CLI 两层），这一批不写、不写 metrics，审计结果写进运行记录；修好后可以续跑；
- run_task 的 LoRA 状态检查：只在 spec.unmerged 时收 injected（且只收 injected），其余照旧；
- 参数互斥：--unmerged 与 --start-only、--check-only 一起给（以及 --decode restricted、--merge-check-n-per-task）是用法错误；
- 续跑：中途停下（第 2 批出错 + 末行不完整）再 --resume，确定的输出与一次跑完逐字节相同；运行身份不同（缺 lora_merge）时拒绝混写；
- 不带 --unmerged：正式评测的 predictions、metrics、config.yaml、README、merge_check.json（去掉墙钟）、打印行与改动前（2db1055 的脚本与
  模块，原样存在 tests/fixtures/e5/）逐字节相同。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from src.e5 import trained_eval as TE
from src.e5.protocol import E5_TASKS
from src.e5.runner import make_sample
from src.e5.token_grid import RESTRICTED, STRUCTURED, decode_grids, sample_images
from tests import test_e5_merge_check_e5v2 as M

REPO = Path(__file__).resolve().parents[1]
CFG = M.CFG
FAST = M.FAST
DET = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")
SNAP = REPO / "tests" / "fixtures" / "e5"
SNAP_SHA256 = {  # 2db1055（DEV-0048 之前，DEV-0046 合入）的文件原样
    "trained_eval_2db1055.py": "25a0596bb59f2e298339fdb76a249317d4a00dba15e688cc80eb81bdf5fdb2fe",
    "eval_e5_trained_2db1055.py": "90dcb73889cbec320d5c79770cc2b892e46a1f26f0d411fc4e4f74fb5a7b25a7",
}
TASKS = ["--task", "a", "c", "--limit", "3", "--batch-size", "2"]  # 起伏与二值各一类；每类 2 批（2 + 1 条）
LEAVES = ("rotation3d", "occlusion")
UNMERGED_KEYS = ("lora_merge", "lora_merge_basis", "lora_merge_rule", "skipped_checks")
SKIP_REASON = "D99-2 敏感性分析：评测的就是未合并的模型；合并核对见同一检查点的 merge_check.json"  # 任务书原文
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_unmerged", REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tree(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def _base(ck: Path, out: Path) -> list[str]:
    return ["--tiny", "--checkpoint", str(ck), "--model", FAST, "--out", str(out)]


@pytest.fixture(scope="module")
def e5ck(tmp_path_factory):
    root = tmp_path_factory.mktemp("unmerged_ck")
    return M._write_e5_checkpoint(root / "final", FAST, M._random_lora)


def _spec(**kw) -> TE.EvalSpec:
    base = dict(start="base", model_key=FAST, tag="n10", seed=0, task="rotation3d", subset="eval", n_per_bin=1, n_resamples=10,
                batch_size=2, tiny=True, dtype="fp32", limit=2, decode=STRUCTURED)
    base.update(kw)
    return TE.EvalSpec(**base)


# ---------------------------------------------------------------------------
# 纯函数：叶目录、校验、常量
# ---------------------------------------------------------------------------


def test_spec_leaf_and_validation(tmp_path):
    mdir = TE.model_dir(tmp_path, "base", FAST, "n10", 0)
    assert _spec().leaf(tmp_path) == mdir / "rotation3d__structured"  # 默认不变
    assert _spec(unmerged=True).leaf(tmp_path) == mdir / "rotation3d__structured__unmerged"
    assert _spec(unmerged=True, subset="dev").leaf(tmp_path) == mdir / "rotation3d__dev__structured__unmerged"
    assert _spec(decode=RESTRICTED).leaf(tmp_path).name == "rotation3d"
    with pytest.raises(ValueError, match="结构强制"):
        _spec(unmerged=True, decode=RESTRICTED)
    with pytest.raises(ValueError, match="只用于 E5 检查点"):
        _spec(unmerged=True, start_only=True, start="e3b", tag=TE.START_ONLY_TAG)
    assert TE.UNMERGED_SUFFIX == "__unmerged" and TE.LORA_MERGE_UNMERGED == "unmerged"
    assert TE.UNMERGED_SKIPPED == {"e5_lora_merge": SKIP_REASON}
    assert TE.UNMERGED_BASIS.startswith("D99-2") and "DEV-0048" in TE.UNMERGED_RULE and "DEV-0046" in TE.UNMERGED_RULE


def test_args_exclusive(e5ck, tmp_path, capsys):
    ev = _script("eval_e5_trained")
    out = str(tmp_path / "o")
    ok = ev.parse_args(_base(e5ck, Path(out)) + ["--unmerged"])
    assert ok.unmerged is True and ev.parse_args(_base(e5ck, Path(out))).unmerged is False
    bad = [
        (["--tiny", "--unmerged", "--start-only", "--init", "e3b", "--start-checkpoint", str(e5ck), "--seed", "0", "--model", FAST,
          "--out", out], "--unmerged 与 --start-only 互斥"),
        (_base(e5ck, Path(out)) + ["--unmerged", "--check-only"], "--unmerged 与 --check-only 互斥"),
        (_base(e5ck, Path(out)) + ["--unmerged", "--decode", "restricted"], "只用结构强制"),
        (_base(e5ck, Path(out)) + ["--unmerged", "--merge-check-n-per-task", "2"], "不用于 --unmerged"),
    ]
    for argv, msg in bad:
        capsys.readouterr()
        with pytest.raises(SystemExit) as ei:
            ev.main(argv)
        assert ei.value.code == 2 and msg in capsys.readouterr().err, argv
    assert not (tmp_path / "o").exists()  # 用法错误：什么都不建


# ---------------------------------------------------------------------------
# 合并评测 vs 不合并评测（同一个检查点）
# ---------------------------------------------------------------------------


def test_unmerged_matches_merged(e5ck, tmp_path, capsys):
    ev = _script("eval_e5_trained")
    out = tmp_path / "o"
    capsys.readouterr()
    assert ev.main(_base(e5ck, out) + TASKS) == 0
    merged_out = capsys.readouterr().out.strip().splitlines()
    mdir = M._mdir(out)
    before = _tree(mdir)
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json", "occlusion__structured", "rotation3d__structured"]
    assert ev.main(_base(e5ck, out) + TASKS + ["--unmerged"]) == 0
    cap = capsys.readouterr()
    # 正式评测的叶目录与 merge_check.json 一字节不动；新文件只在 __unmerged 叶目录里
    after = _tree(mdir)
    assert {k: after[k] for k in before} == before
    assert all(k.split("/")[0].endswith("__unmerged") for k in set(after) - set(before))
    assert sorted(p.name for p in mdir.iterdir()) == ["merge_check.json", "occlusion__structured", "occlusion__structured__unmerged",
                                                      "rotation3d__structured", "rotation3d__structured__unmerged"]
    # 打印行：只多 merge=unmerged；stderr 写明跳过、没有合并核对
    assert cap.out.strip().splitlines() == [x.replace("decode=structured", "decode=structured merge=unmerged") for x in merged_out]
    assert len(merged_out) == 2 and all("merge=unmerged" in x for x in cap.out.strip().splitlines())
    assert f"[不合并] 跳过评测前的合并核对：{SKIP_REASON}" in cap.err and "exists=True" in cap.err and "pass=True" in cap.err
    assert "[合并核对" not in cap.err and "unmerged]" in cap.err  # 进度行带 unmerged
    mc_sha = _sha(mdir / TE.MERGE_CHECK)
    for task in LEAVES:
        lm, lu = mdir / f"{task}__structured", mdir / f"{task}__structured__unmerged"
        # predictions：逐字节相同；逐条预测与得分相同
        assert (lm / "predictions.jsonl").read_bytes() == (lu / "predictions.jsonl").read_bytes()
        pm = [json.loads(x) for x in (lm / "predictions.jsonl").read_text().splitlines()]
        pu = [json.loads(x) for x in (lu / "predictions.jsonl").read_text().splitlines()]
        assert len(pm) == len(pu) == 3
        assert [(r["id"], r["pred"], r["score"], r["valid"]) for r in pm] == [(r["id"], r["pred"], r["score"], r["valid"]) for r in pu]
        # metrics：除 merge 外完全相同
        mm, mu = (json.loads((d / "metrics.json").read_text()) for d in (lm, lu))
        assert "merge" not in mm and mu.pop("merge") == "unmerged"
        assert mu == mm and mm["n"] == 3 and mm["decode"]["mode"] == STRUCTURED
        # 运行身份：只多 lora_merge 一组字段
        im, iu = (yaml.safe_load((d / "config.yaml").read_text()) for d in (lm, lu))
        extra = {k: iu.pop(k) for k in UNMERGED_KEYS}
        assert iu == im and not set(UNMERGED_KEYS) & set(im)
        assert extra["lora_merge"] == "unmerged" and extra["lora_merge_basis"] == TE.UNMERGED_BASIS and "D99-2" in extra["lora_merge_basis"]
        assert extra["lora_merge_rule"] == TE.UNMERGED_RULE and extra["skipped_checks"] == {"e5_lora_merge": SKIP_REASON}
        # README：写明不合并、跳过的核对、审计与对照的叶目录；其余行与正式评测相同（只换标题）
        rm, ru = ((d / "README.md").read_text().splitlines() for d in (lm, lu))
        assert rm[0].startswith("# E5 · 训练后（base → E5）") and ru[0] == rm[0].replace("（base → E5）", "（base → E5，E5 LoRA 不合并）")
        added = [x for x in ru[1:] if x not in rm]
        assert len(added) == 3 and [x for x in ru[1:] if x in rm] == rm[1:]
        assert "**不合并**" in added[0] and "D99-2" in added[0] and f"{task}__structured/" in added[0]
        assert SKIP_REASON in added[1] and "LoraAudit" in added[2] and "logs/run_<k>.json" in added[2]
        # 运行记录：逐批的 LoRA 审计、跳过的理由、同一检查点 merge_check.json 的引用
        run = json.loads((lu / "logs" / "run_001.json").read_text())
        au = run["lora_audit"]
        assert au["status"] == "pass" and au["error"] is None and au["n_batches"] == 2 and au["lora_state"] == "injected"
        assert au["n_modules"] == au["n_called"] > 0 and au["n_not_called"] == 0 and au["min_calls"] > 0
        assert run["merge_check"] is None and run["lora_merge"] == "unmerged" and run["mode"] == "unmerged"
        assert run["skipped_checks"] == {"e5_lora_merge": SKIP_REASON} and run["lora_merge_basis"] == TE.UNMERGED_BASIS
        ref = run["merge_check_ref"]
        assert ref["exists"] is True and ref["sha256"] == mc_sha and ref["pass"] is True and ref["criteria_version"] == "e5-v2"
        assert run["finished_utc"] is not None and run["n_batches"] == 2
        rm_run = json.loads((lm / "logs" / "run_001.json").read_text())
        assert "lora_audit" not in rm_run and rm_run["merge_check"]["pass"] is True  # 正式评测的运行记录照旧


# ---------------------------------------------------------------------------
# LoRA 审计与 LoRA 状态
# ---------------------------------------------------------------------------


def test_run_task_audit_and_lora_state(e5ck, tmp_path):
    ad, e5tok = M._load(e5ck)
    pf, n = CFG.prompt_format(), CFG.grid_n
    kw = dict(checkpoint={"test": "unmerged"}, n=n)
    spec = _spec(unmerged=True)
    leaf = spec.leaf(tmp_path)
    assert ad.lora_state == "injected"
    with pytest.raises(RuntimeError, match="评测要在合并之后"):  # 正式评测：injected 照旧报错
        TE.run_task(ad, e5tok, pf, CFG, _spec(), tmp_path, **kw)
    # 禁用 adapter：LoRA 层被调用、lora_B 没有 → 审计报错；这一批不写，不写 metrics，审计结果进运行记录
    with ad.peft_model.disable_adapter():
        with pytest.raises(RuntimeError, match="没走 LoRA 分支") as ei:
            TE.run_task(ad, e5tok, pf, CFG, spec, tmp_path, **kw)
    assert "不合并的评测（--unmerged，D99-2）" in str(ei.value)
    assert not (leaf / "metrics.json").exists() and not (leaf / "predictions.jsonl").exists()
    run1 = json.loads((leaf / "logs" / "run_001.json").read_text())
    assert run1["lora_audit"]["status"] == "fail" and "没走 LoRA 分支" in run1["lora_audit"]["error"]
    assert run1["lora_audit"]["n_batches"] == 0 and "没走 LoRA 分支" in run1["error"] and run1["finished_utc"] is None
    # 恢复后续跑：审计通过，写 metrics
    m = TE.run_task(ad, e5tok, pf, CFG, spec, tmp_path, resume=True, **kw)
    assert m["merge"] == "unmerged" and m["n"] == 2 and (leaf / "metrics.json").exists()
    run2 = json.loads((leaf / "logs" / "run_002.json").read_text())
    assert run2["lora_audit"]["status"] == "pass" and run2["lora_audit"]["n_batches"] == 1 and "error" not in run2
    # 这个审计不平凡：同样的输入，绕开 LoRA 时预测不同
    preds = [json.loads(x)["pred"] for x in (leaf / "predictions.jsonl").read_text().splitlines()]
    samples = [make_sample(it) for it in spec.items()]
    inp = [ad.build_inputs(sample_images(s), pf.render(s, n), pf.order) for s in samples]
    with ad.peft_model.disable_adapter():
        off = decode_grids(ad, e5tok, inp, "relief", n, decode=STRUCTURED)
    assert [TE.record(s, pf.render(s, n), r, e5tok, n)["pred"] for s, r in zip(samples, off)] != preds
    # 合并之后：不合并的评测拒绝（只收 injected），正式评测照常
    ad.merge_lora()
    with pytest.raises(RuntimeError, match="要挂着未合并的 E5 LoRA"):
        TE.run_task(ad, e5tok, pf, CFG, _spec(unmerged=True, task="occlusion"), tmp_path, **kw)
    assert not _spec(unmerged=True, task="occlusion").leaf(tmp_path).exists()
    au = TE.UnmergedAudit()
    with pytest.raises(RuntimeError, match="解码之前 LoRA 状态是 merged"):
        au.decode(ad, e5tok, inp, "relief", n, decode=STRUCTURED, where="[t]")
    assert au.summary()["status"] == "fail"


def test_cli_audit_failure_no_metrics(e5ck, tmp_path, monkeypatch):
    """CLI 层：解码时绕开 LoRA（在 decode_grids 外面禁用 adapter）→ 退出码 2，报错写进 errors/，不写 metrics，预测一条都不写。"""
    ev = _script("eval_e5_trained")
    real = TE.decode_grids

    def bypass(ad, *a, **kw):
        with ad.peft_model.disable_adapter():
            return real(ad, *a, **kw)

    monkeypatch.setattr(TE, "decode_grids", bypass)
    out = tmp_path / "o"
    assert ev.main(_base(e5ck, out) + ["--task", "c", "--limit", "2", "--unmerged"]) == 2
    (err,) = (out / "trained" / "errors").glob("error_*.json")
    assert "没走 LoRA 分支" in json.loads(err.read_text())["error"]
    leaf = M._mdir(out) / "occlusion__structured__unmerged"
    assert not (leaf / "metrics.json").exists() and not (leaf / "predictions.jsonl").exists()
    assert json.loads((leaf / "logs" / "run_001.json").read_text())["lora_audit"]["status"] == "fail"
    assert not (M._mdir(out) / TE.MERGE_CHECK).exists()  # 不做合并核对


# ---------------------------------------------------------------------------
# 续跑
# ---------------------------------------------------------------------------


def test_unmerged_resume_bytewise(e5ck, tmp_path, monkeypatch, capsys):
    ev = _script("eval_e5_trained")
    args = TASKS + ["--unmerged"]
    a_out, b_out = tmp_path / "a", tmp_path / "b"
    assert ev.main(_base(e5ck, a_out) + args) == 0
    # 中途停下：第一类的第 2 批出错（第 1 批已写），再在 predictions 末尾留半行
    real, calls = TE.decode_grids, []

    def crash(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("模拟中断")
        return real(*a, **kw)

    monkeypatch.setattr(TE, "decode_grids", crash)
    assert ev.main(_base(e5ck, b_out) + args) == 2
    monkeypatch.setattr(TE, "decode_grids", real)
    mb, ma = M._mdir(b_out), M._mdir(a_out)
    leaf = mb / "rotation3d__structured__unmerged"
    lines = (leaf / "predictions.jsonl").read_bytes().splitlines(keepends=True)
    assert len(lines) == 2 and not (leaf / "metrics.json").exists() and not (mb / "occlusion__structured__unmerged").exists()
    run1 = json.loads((leaf / "logs" / "run_001.json").read_text())
    assert run1["lora_audit"]["status"] == "pass" and run1["lora_audit"]["n_batches"] == 1 and "模拟中断" in run1["error"]
    with open(leaf / "predictions.jsonl", "ab") as f:
        f.write(lines[0][:9])
    assert ev.main(_base(e5ck, b_out) + args) == 2  # 非空目录不加 --resume：拒绝
    capsys.readouterr()
    assert ev.main(_base(e5ck, b_out) + args + ["--resume"]) == 0
    for task in LEAVES:
        la, lb = ma / f"{task}__structured__unmerged", mb / f"{task}__structured__unmerged"
        for f in DET:
            assert (la / f).read_bytes() == (lb / f).read_bytes(), (task, f)
    run2 = json.loads((leaf / "logs" / "run_002.json").read_text())
    assert run2["n_done_before"] == 2 and run2["lora_audit"]["n_batches"] == 1 and run2["lora_audit"]["status"] == "pass"
    assert not (mb / TE.MERGE_CHECK).exists()
    # 运行身份不同（去掉 lora_merge 一组字段，冒充正式评测的配置）：拒绝混写
    lc = mb / "occlusion__structured__unmerged"
    ident = yaml.safe_load((lc / "config.yaml").read_text())
    for k in UNMERGED_KEYS:
        ident.pop(k)
    (lc / "config.yaml").write_text(yaml.safe_dump(ident, sort_keys=False, allow_unicode=True))
    (lc / "metrics.json").unlink()
    assert ev.main(_base(e5ck, b_out) + ["--task", "c", "--limit", "3", "--batch-size", "2", "--unmerged", "--resume"]) == 2
    errs = sorted((b_out / "trained" / "errors").glob("error_*.json"))
    msg = json.loads(errs[-1].read_text())["error"]
    assert "拒绝混写" in msg and "lora_merge" in msg


# ---------------------------------------------------------------------------
# 不带 --unmerged：与改动前（2db1055）逐字节相同
# ---------------------------------------------------------------------------

_WALL = ("timing_s", "written_utc")


def _strip_mc(doc: dict) -> dict:
    doc = json.loads(json.dumps(doc))
    for k in _WALL:
        doc.pop(k)
    ml = doc["metric_level"]
    ml.pop("timing_s")
    for side in ("a", "b"):
        ml["sides"][side].pop("seconds")
    return doc


def test_merged_eval_identical_to_2db1055(e5ck, tmp_path, capsys):
    """正式评测（默认结构强制，不带 --unmerged）：改动前（2db1055）的脚本加模块对改动后的，同一进程、同一组参数，predictions、metrics、
    README、config.yaml 与打印行逐字节相同；merge_check.json 去掉墙钟之后完全相同；运行记录的字段与非墙钟的值相同。"""
    for name, sha in SNAP_SHA256.items():
        assert _sha(SNAP / name) == sha, name
    old_te = M._load_path(SNAP / "trained_eval_2db1055.py", "_e5_trained_eval_2db1055")
    old_te.REPO_ROOT = TE.REPO_ROOT  # 快照按自己的位置推出的仓库根不对（只影响 meta.json 里配置文件的路径）
    old_script = M._load_path(SNAP / "eval_e5_trained_2db1055.py", "_eval_e5_trained_2db1055")
    old_script.TE = old_te
    capsys.readouterr()
    outs, printed, errs = [], [], []
    for k, mod in enumerate((old_script, _script("eval_e5_trained"))):
        o = tmp_path / f"o{k}"
        assert mod.main(_base(e5ck, o) + ["--limit", "1"]) == 0
        cap = capsys.readouterr()
        printed.append(cap.out)
        errs.append([x for x in cap.err.splitlines() if x.startswith("[合并核对·指标层面")])
        outs.append(M._mdir(o))
    assert printed[0] == printed[1] and printed[0].count("\n") == len(E5_TASKS) and "merge=" not in printed[1]
    assert errs[0] == errs[1] and len(errs[1]) == 1
    for task in E5_TASKS:
        leaf = f"{task}__structured"
        for f in DET:
            assert (outs[0] / leaf / f).read_bytes() == (outs[1] / leaf / f).read_bytes(), (task, f)
        r0, r1 = (json.loads((d / leaf / "logs" / "run_001.json").read_text()) for d in outs)
        assert list(r0) == list(r1) and "lora_audit" not in r1
        for r in (r0, r1):
            for key in ("started_utc", "finished_utc", "wall_s", "argv"):
                r.pop(key)
            for key in ("file", "file_sha256"):
                r["merge_check"].pop(key)
        assert r0 == r1
    assert sorted(p.name for p in outs[1].iterdir()) == sorted(p.name for p in outs[0].iterdir())
    assert not any(p.name.endswith("__unmerged") for p in outs[1].iterdir())
    m0, m1 = (json.loads((d / TE.MERGE_CHECK).read_text()) for d in outs)
    assert list(m0) == list(m1) and _strip_mc(m0) == _strip_mc(m1)
