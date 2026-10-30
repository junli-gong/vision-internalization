"""DEV-0075：LISA-7B 外部对照的"导出清单 → 运行 → 计分"（--tiny 的 W2 小数据，替身运行器，子进程，与服务器同一流程）。

- 子集：导出的清单与 src/llmseg/runner.py 的取样逐 id、逐顺序相同（tiny 现算；真配置用同一份 configs/w2_llmseg_subsets.json）。
- 放行：export 与 run 缺批准编号、编号不对都是退出码 2、什么都不写；给 D154 放行，test_unlock.jsonl 有两步的记录，运行身份里有编号。
- --out 的最后一级不带运行编号（四位数字加 _lisa）时三步都拒绝，什么都不写。
- 计分：W2 的叶目录 <out>/lisa-7b-v1/refseg/lisa_7b/<划分>/，scripts/mgr/w2_test_table.py 的 metrics、preds 能直接读；score 与按 RLE
  重算的 mask_inter_union（W2 的 GT）相同；没有 [SEG]、推理异常按空 mask 计，failures 计数；替身的输出要 --allow-stand-in。
- 续跑：--limit 截断后 --resume 与一次跑完的掩码与计分相同；身份不同、没加 --resume 都拒绝；连续异常就停、那一串不写出。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.e1.scoring import gt_mask
from src.executors.raster import rle_to_mask
from src.lisa import common as C
from src.lisa import data as D
from src.llmseg import runner as LR
from src.metrics.seg import mask_inter_union, sample_iou
from src.w2.data import tiny_paths, tiny_splits, w2_items
from tests import _lisa_fakes as F

REPO = Path(__file__).resolve().parents[1]
W2 = REPO / "scripts" / "lisa_w2.py"
RUN = REPO / "scripts" / "run_lisa.py"
SPLIT = "test_refcocog_umd_test"  # tiny 上 4 句
N_TINY = {"test_refcoco_unc_testA": 2, "test_refcoco_unc_testB": 2, "test_refcoco+_unc_testA": 2, "test_refcoco+_unc_testB": 2,
          SPLIT: 4}


def sh(script: Path, *args: str, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=900)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def export(out: Path, *extra: str, ok: bool = True) -> subprocess.CompletedProcess:
    return sh(W2, "export", "--tiny", "--out", str(out), "--test-approval", "D154", *extra, ok=ok)


def run(out: Path, factory: str = "make_fixed", *extra: str, ok: bool = True) -> subprocess.CompletedProcess:
    return sh(RUN, "--out", str(out), "--test-approval", "D154", "--runner-factory", f"tests._lisa_fakes:{factory}", *extra, ok=ok)


def score(out: Path, *extra: str, ok: bool = True) -> subprocess.CompletedProcess:
    return sh(W2, "score", "--out", str(out), "--allow-stand-in", *extra, ok=ok)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def table_module():
    spec = importlib.util.spec_from_file_location("w2_test_table_for_lisa", REPO / "scripts" / "mgr" / "w2_test_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_samples(split: str):
    sd = tiny_splits()
    sub = LR.build_subsets(sd, splits=LR.test_splits(), source={})
    return LR.subset_samples(w2_items("refseg", split, tiny_paths(), splits=sd), sub["splits"][split]["ids"])


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lisa")
    a = tmp / "0001_lisa_fixed"
    export(a)
    run(a)
    score(a)
    b = tmp / "0002_lisa_failures"
    export(b)
    run(b, "make_with_failures")
    score(b)
    return a, b


# ---------------------------------------------------------------------------
# 子集
# ---------------------------------------------------------------------------


def test_manifest_ids_equal_llmseg_subset(pipeline):
    a, _ = pipeline
    meta, rows, _ = C.read_manifest(a)
    assert list(rows) == LR.test_splits() and meta["tiny"] is True and meta["limit"] is None
    for split, rr in rows.items():
        want = tiny_samples(split)
        assert [r["id"] for r in rr] == [s.id for s in want] and len(rr) == N_TINY[split]
        for r, s in zip(rr, want):
            g = gt_mask(s)
            assert r["sent"] == s.fields["expression"] and (r["height"], r["width"]) == g.shape
            assert r["image"] == str(Path(s.images[0]).resolve()) and r["image_sha256"] == C.sha256_file(s.images[0])
            assert set(r) == set(C.MANIFEST_KEYS)  # 不含 GT


def test_real_selection_is_the_llmseg_subset_file():
    splits_doc, subsets_doc, files = D.load_selection(tiny=False)
    assert subsets_doc == LR.load_subsets() and LR.check_subsets() == []
    assert [p.name for p in files] == ["w2_splits.json", "w2_llmseg_subsets.json"]
    assert list(subsets_doc["splits"]) == LR.test_splits() and all(e["n"] == 300 for e in subsets_doc["splits"].values())
    assert sum(e["n"] for e in subsets_doc["splits"].values()) == 1500


# ---------------------------------------------------------------------------
# 计分与叶目录
# ---------------------------------------------------------------------------


def test_leaf_readable_by_w2_test_table_and_scores_match(pipeline):
    a, _ = pipeline
    t = table_module()
    for split in LR.test_splits():
        lf = C.leaf_dir(a, split)
        assert lf == a / "lisa-7b-v1" / "refseg" / "lisa_7b" / split
        m = t.metrics(lf)
        assert m["complete"] is True and m["n"] == N_TINY[split] and m["cIoU"]["value"] is not None and m["gIoU"]["ci"]
        p = t.preds(lf)
        samples = tiny_samples(split)
        assert sorted(p) == sorted(s.id for s in samples)
        recs = jsonl(lf / "predictions.jsonl")
        for r, s in zip(recs, samples):
            gt = gt_mask(s)
            mask = rle_to_mask(r["answer"]["mask"]["rle"])
            np.testing.assert_array_equal(mask, F.fixed_mask(*gt.shape))
            st = mask_inter_union(mask, gt)
            assert r["score"] == {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}
            assert p[s.id]["inter"] == st.inter and p[s.id]["union"] == st.union
            assert r["failure"] is None and r["n_model_calls"] == 1 and r["lisa"]["forwards"]["llm"] == 1
            assert r["tokens"] == {"prompt": 63, "generated": 0, "inserted": 0, "visual": 256, "image_pixels": 224 * 224}
        cfg = yaml.safe_load((lf / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["w2"]["test_unlock"]["approval"] == "D154" and cfg["w2"]["setting"] == "lisa_7b" and cfg["w2"]["split"] == split
        assert cfg["model"]["stand_in"] is True and cfg["lisa"]["code"]["commit"] == C.UPSTREAM_COMMIT
        assert cfg["lisa"]["protocol"]["name"] == "refer_seg_validate" and "硬约束第 1 条" in cfg["isolation"]
        comp = json.loads((lf / "compute.json").read_text(encoding="utf-8"))
        assert comp["n_samples"] == N_TINY[split] and comp["n_calls"] == N_TINY[split] and comp["generated_tokens"] == 0
        readme = (lf / "README.md").read_text(encoding="utf-8")
        assert "硬约束第 1 条" in readme and "替身" in readme
        assert (lf / "meta.json").is_file() and (lf / "logs" / "timing.jsonl").is_file()


def test_failures_scored_as_empty_masks(pipeline):
    _, b = pipeline
    lf = C.leaf_dir(b, SPLIT)
    m = json.loads((lf / "metrics.json").read_text(encoding="utf-8"))
    assert m["failures"] == {"RuntimeError": 1, "no_seg": 1, "ok": 2} and m["n_failed"] == 2 and m["complete"] is True
    recs = {r["id"]: r for r in jsonl(lf / "predictions.jsonl")}
    for s in tiny_samples(SPLIT):
        r = recs[s.id]
        gt = gt_mask(s)
        sid = s.fields["sent_id"]
        if str(sid) in F.NO_SEG_SENT_IDS | F.RAISE_SENT_IDS:
            assert r["failure"] in ("no_seg", "RuntimeError") and r["answer"]["mask"]["source"] == "empty"
            assert r["score"] == {"inter": 0, "union": int(gt.sum()), "iou": 0.0}
        else:
            assert r["failure"] is None
    raise_rec = next(r for r in recs.values() if r["failure"] == "RuntimeError")
    assert raise_rec["n_model_calls"] == 0 and raise_rec["tokens"]["visual"] == 0 and "替身" in raise_rec["lisa"]["error"]
    raw = jsonl(C.raw_rows_path(b, SPLIT))
    assert [r["id"] for r in raw] == [s.id for s in tiny_samples(SPLIT)]
    ex = jsonl(C.raw_dir(b) / C.EXCEPTIONS_LOG)
    assert len(ex) == 1 and ex[0]["written"] is True
    testa = json.loads((C.leaf_dir(b, "test_refcoco_unc_testA") / "metrics.json").read_text(encoding="utf-8"))
    assert testa["failures"] == {"no_seg": 1, "ok": 1}


def test_score_requires_allow_stand_in_and_rescore(pipeline):
    a, _ = pipeline
    p = sh(W2, "score", "--out", str(a), ok=False)
    assert p.returncode == 2 and "allow-stand-in" in p.stderr
    p = score(a, ok=False)
    assert p.returncode == 2 and "--rescore" in p.stderr
    before = (C.leaf_dir(a, SPLIT) / "predictions.jsonl").read_bytes()
    score(a, "--rescore")
    assert (C.leaf_dir(a, SPLIT) / "predictions.jsonl").read_bytes() == before


# ---------------------------------------------------------------------------
# 放行与运行编号
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("approval", [None, "D147", "d154"])
def test_export_refuses_without_exact_approval(tmp_path, approval):
    out = tmp_path / "0003_lisa_gate"
    args = ["export", "--tiny", "--out", str(out)] + ([] if approval is None else ["--test-approval", approval])
    p = sh(W2, *args, ok=False)
    assert p.returncode == 2 and "D154" in p.stderr
    assert not out.exists()


@pytest.mark.parametrize("approval", [None, "D147"])
def test_run_refuses_without_exact_approval(tmp_path, approval):
    out = tmp_path / "0004_lisa_gate"
    export(out)
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    args = ["--out", str(out), "--runner-factory", "tests._lisa_fakes:make_fixed"] + ([] if approval is None else ["--test-approval", approval])
    p = sh(RUN, *args, ok=False)
    assert p.returncode == 2 and "D154" in p.stderr
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before  # 什么都没写
    assert not C.raw_dir(out).exists()


def test_approval_recorded(pipeline):
    a, _ = pipeline
    recs = C.read_unlock(a)
    assert [r["step"] for r in recs] == ["export", "run"] and all(r["approval"] == "D154" for r in recs)
    assert recs[1]["runner"] == "stand_in" and recs[0]["manifest_rows_sha256"] == recs[1]["manifest_rows_sha256"]
    ident = json.loads((C.raw_dir(a) / C.RUN_CONFIG).read_text(encoding="utf-8"))
    assert ident["approval"] == "D154" and ident["runner"]["kind"] == "stand_in" and ident["weights"] is None
    assert ident["code"] == C.normalized(C.check_vendor()) and ident["protocol"]["answer"] == "[SEG]."


@pytest.mark.parametrize("name", ["lisa_hpcgpu06", "99_lisa", "0005_llmseg", "0005lisa"])
def test_out_without_run_number_refused(tmp_path, name):
    out = tmp_path / name
    p = sh(W2, "export", "--tiny", "--out", str(out), "--test-approval", "D154", ok=False)
    assert p.returncode == 2 and "运行编号" in p.stderr
    p = sh(RUN, "--out", str(out), "--test-approval", "D154", "--runner-factory", "tests._lisa_fakes:make_fixed", ok=False)
    assert p.returncode == 2 and "运行编号" in p.stderr
    p = sh(W2, "score", "--out", str(out), "--allow-stand-in", ok=False)
    assert p.returncode == 2 and "运行编号" in p.stderr
    assert not out.exists()


# ---------------------------------------------------------------------------
# 续跑、身份与异常
# ---------------------------------------------------------------------------


def _masks(out: Path) -> dict[str, list]:
    return {s: [(r["id"], r.get("mask_rle"), r.get("failure")) for r in jsonl(C.raw_rows_path(out, s))] for s in LR.test_splits()}


def test_resume_matches_one_shot(pipeline, tmp_path):
    a, _ = pipeline
    c = tmp_path / "0006_lisa_resume"
    export(c)
    run(c, "make_fixed", "--limit", "1")
    assert {s: len(v) for s, v in _masks(c).items()} == {s: 1 for s in LR.test_splits()}
    p = run(c, ok=False)
    assert p.returncode == 2 and "--resume" in p.stderr
    p = run(c, "make_with_failures", "--resume", ok=False)  # 换了运行器的身份（同为替身，工厂不同）
    assert p.returncode == 2 and "身份" in p.stderr
    with open(C.raw_rows_path(c, SPLIT), "a", encoding="utf-8") as f:
        f.write('{"id": "refcocog_umd/test/405", "trunc')  # 写到一半被打断
    run(c, "make_fixed", "--resume")
    assert _masks(c) == _masks(a)
    assert [json.loads(p.read_text())["state"] for p in sorted((C.raw_dir(c) / "logs").glob("run_*.json"))] == ["done", "done"]


def test_consecutive_exceptions_stop_without_writing_the_streak(tmp_path):
    d = tmp_path / "0007_lisa_crash"
    export(d)
    p = run(d, "make_always_raise", ok=False)
    assert p.returncode == 2 and "连续 3 句" in p.stderr
    # 这一串跨了两个划分（testA 2 句 + testB 1 句），都没有写出（续跑时重试）
    assert not any(C.raw_rows_path(d, s).exists() for s in LR.test_splits())
    ex = jsonl(C.raw_dir(d) / C.EXCEPTIONS_LOG)
    assert len(ex) == C.MAX_CONSECUTIVE_EXCEPTIONS == 3 and all(e["written"] is False for e in ex)
    log = json.loads(next((C.raw_dir(d) / "logs").glob("run_*.json")).read_text())
    assert log["state"] == "error"
    run(d, "make_always_raise", "--resume", ok=False)  # 续跑从头重试同一串
    assert not any(C.raw_rows_path(d, s).exists() for s in LR.test_splits())


def test_export_is_idempotent_and_refuses_a_different_manifest(tmp_path):
    e = tmp_path / "0008_lisa_export"
    export(e)
    before = (C.inputs_dir(e) / C.MANIFEST_ROWS).read_bytes()
    export(e)
    assert (C.inputs_dir(e) / C.MANIFEST_ROWS).read_bytes() == before
    p = export(e, "--limit", "1", ok=False)
    assert p.returncode == 2 and "拒绝覆盖" in p.stderr
    assert [r["step"] for r in C.read_unlock(e)] == ["export", "export"]


def test_run_stops_when_image_changed(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import run_lisa

    s = tiny_samples(SPLIT)[0]
    row = D.manifest_row(s, SPLIT)
    rec, tb = run_lisa.infer_row(F.FixedMaskRunner(), row)
    assert rec["status"] == "ok" and tb is None
    bad = {**row, "image_sha256": "0" * 64}
    with pytest.raises(C.LisaError, match="sha256"):
        run_lisa.infer_row(F.FixedMaskRunner(), bad)
    class WrongSize:  # mask 不是原图尺寸 → 这一句记为失败（异常类名），不写坏的 mask
        def infer(self, r):
            return {**F.FixedMaskRunner().infer(r), "mask": np.zeros((2, 2), bool)}

    rec, tb = run_lisa.infer_row(WrongSize(), row)
    assert rec["failure"] == "MaskFormatError" and rec["mask_rle"] is None and tb
