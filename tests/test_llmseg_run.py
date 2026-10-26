"""DEV-0066：scripts/run_llmseg.py 与 scripts/compare_llmseg.py（--tiny，子进程，与服务器同一流程）。

- 子集清单 configs/w2_llmseg_subsets.json：与按规则重算的逐字节相同；5 个划分各 300 句，都在 W2 的划分里，来源清单的 sha256 对得上。
- 端到端（脚本化的 0–1000 网格适配器，tests/_llmseg_fakes）：官方管线走完检测、分割、修正；两份 mask（修正前 / 后）的计分与按 RLE
  重算的 mask_inter_union 相同，GT 是 W2 的那一份；各阶段调用数、上限、记账、运行身份（官方 commit、PROVENANCE 的 sha256、模型、T）。
  合批（slots 3、batch 2）与逐条（slots 1、batch 1）的 predictions.jsonl 逐字节相同；续跑与一次跑完逐字节相同。
- 小型随机 Qwen 的端到端：跑通、截断照实记录；合批与逐条的 predictions.jsonl 逐字节相同（CPU fp32）。
- 生成报错：退出码 2，那一组不写 predictions（不按空 mask 计分）。
- 比较脚本：与 W2-c 的配对差（cIoU、gIoU，两组读数）与手算相同；W2-c 目录的身份核对。
"""

from __future__ import annotations

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
from src.llmseg import provider as PV
from src.llmseg import runner as R
from src.llmseg import vendor as V
from src.metrics.seg import mask_inter_union, sample_iou
from src.w2.data import W2_SPLITS_PATH, load_splits, tiny_paths, tiny_splits, w2_items

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_llmseg.py"
COMPARE = REPO / "scripts" / "compare_llmseg.py"
QWEN = "qwen3.5-0.8b"
SPLIT = "test_refcocog_umd_test"  # tiny 上 4 句
FAKE = ["--adapter-factory", "tests._llmseg_fakes:make_grid_adapter", "--cap-limit", "16384"]
KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


def run(args, tmp: Path, *, ok: bool = True, script: Path = SCRIPT) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env.update(TMPDIR=str(tmp), HF_HUB_OFFLINE="1")
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def tiny_args(out: Path, *extra: str, split: str = SPLIT) -> list[str]:
    return ["--tiny", "--model", QWEN, "--split", split, "--out", str(out), *extra]


def leaf(out: Path, split: str = SPLIT, setting: str = R.SETTING) -> Path:
    return out / QWEN / "refseg" / setting / split


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_subsets_manifest_matches_rule():
    assert R.check_subsets() == []
    doc = R.load_subsets()
    w2 = load_splits(W2_SPLITS_PATH)
    assert doc["salt"] == "w2_llmseg_v1" and doc["n_per_split"] == 300
    assert doc["source"]["sha256"] == V.sha256_file(W2_SPLITS_PATH)
    assert list(doc["splits"]) == R.test_splits() and len(R.test_splits()) == 5
    for name, e in doc["splits"].items():
        src = w2["splits"][name]
        assert e["n"] == 300 == len(e["ids"]) and len(set(e["ids"])) == 300
        assert set(e["ids"]) <= set(src["ids"]) and e["source_sha256"] == src["sha256"] and src["n"] == 1000
        assert e["ids"] == R.select_subset(src["ids"], name)
    # 规则本身：hash_rank 的前 300，purpose = 划分名（换 purpose 或 salt 就是另一批）
    ids = w2["splits"][R.test_splits()[0]]["ids"]
    assert R.select_subset(ids, R.test_splits()[0]) != R.select_subset(ids, R.test_splits()[0], salt="other")


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("llmseg_fake")
    a, b = tmp / "a", tmp / "b"
    run(tiny_args(a, *FAKE, "--slots", "3", "--batch-size", "2", "--chunk-size", "2", "--save", "1"), tmp)
    run(tiny_args(b, *FAKE, "--slots", "1", "--batch-size", "1", "--chunk-size", "2", "--save", "1"), tmp)
    return tmp, a, b


def test_fake_end_to_end_records_and_scores(fake_runs):
    _, a, _ = fake_runs
    lf = leaf(a)
    recs = jsonl(lf / "predictions.jsonl")
    samples = R.subset_samples(w2_items("refseg", SPLIT, tiny_paths(), splits=tiny_splits()),
                               R.build_subsets(tiny_splits(), splits=R.test_splits(), source={})["splits"][SPLIT]["ids"])
    assert [r["id"] for r in recs] == [s.id for s in samples] and len(recs) == 4
    for r, s in zip(recs, samples):
        gt = gt_mask(s)
        assert r["status"] == "ok" and r["pipeline_rc"] == 0 and r["failure"] is None
        assert r["calls_by_stage"]["detect"] == 1 and r["calls_by_stage"]["segment"] == 1
        assert r["calls_by_stage"]["focus_boxes"] == r["refine"]["iterations"] >= 3 and r["refine"]["converged"]
        assert r["n_model_calls"] == sum(r["calls_by_stage"].values()) <= 18
        assert r["stage2"]["n_model_calls"] == 2 and r["stage2"]["mask"]["source"] == "seg_mask"
        assert r["answer"]["mask"]["source"] == "rec_mask"
        for block, src in ((r["stage2"], r["stage2"]["mask"]), (r, r["answer"]["mask"])):
            m = rle_to_mask(src["rle"])
            st = mask_inter_union(m, gt)
            assert block["score"] == {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}
        assert r["answer"]["mask"]["area"] > r["stage2"]["mask"]["area"]  # 补的焦点框让 mask 变大
        assert r["coords"]["n_gt_1000"] == 0 and r["coords"]["n_values"] > 0
        assert sum(r["truncated_by_stage"].values()) == 0 and sum(r["format_fail_by_stage"].values()) == 0
        assert r["pipeline_events"] == []
    # 保留 PNG 的只有第一条（--save 1）；修正前的 mask 文件与记录一致
    works = sorted((lf / "work").iterdir())
    first = lf / "work" / R.item_key(recs[0]["id"])
    assert (first / "seg_mask.png").exists() and (first / "rec_mask.png").exists()
    assert all(not (w / "rec_mask.png").exists() for w in works if w != first)
    np.testing.assert_array_equal(R.load_mask(first / "seg_mask.png", gt_mask(samples[0]).shape), rle_to_mask(recs[0]["stage2"]["mask"]["rle"]))
    m = json.loads((lf / "metrics.json").read_text())
    assert m["complete"] and m["n"] == 4 and m["llmseg_commit"] == V.UPSTREAM_COMMIT
    assert [h["name"] for h in m["readouts"]["rec"]["headline"]] == ["cIoU", "gIoU"] and m["headline"] == m["readouts"]["rec"]["headline"]
    assert m["readouts"]["seg"]["compute_per_image"]["n_model_calls"] == 2.0
    c = json.loads((lf / "compute.json").read_text())
    c2 = json.loads((lf / "compute_stage2.json").read_text())
    assert c["n_calls"] == sum(r["n_model_calls"] for r in recs) and c2["n_calls"] == 8 and c2["n_rounds"] == 8
    assert c["generated_tokens"] == sum(r["tokens"]["generated"] for r in recs)
    cfg = yaml.safe_load((lf / "config.yaml").read_text())
    assert cfg["llmseg"]["code"]["commit"] == V.UPSTREAM_COMMIT
    assert cfg["llmseg"]["code"]["provenance_sha256"] == V.sha256_file(V.PROVENANCE_PATH)
    assert cfg["llmseg"]["pipeline"]["iterations"] == 8 and cfg["llmseg"]["pipeline"]["main_config"] is True
    assert cfg["llmseg"]["coordinates"]["model_name"] == QWEN and cfg["provider"]["enable_thinking"] is False
    assert cfg["model"]["key"] == QWEN and cfg["batching"]["slots"] == 3
    calls = jsonl(lf / "logs" / "calls.jsonl")
    assert len(calls) == c["n_calls"] and all("text" not in x for x in calls)
    caps = {x["stage"]: x["cap"] for x in calls}
    assert caps == {"detect": 2048, "segment": 2048, "focus_boxes": 896, "focus_splines": 640}
    assert all("text" in x for x in jsonl(first / "calls.jsonl"))
    g = json.loads((lf / "logs" / "gpu_summary.json").read_text())
    assert g["n_items"] == 4 and g["n_sub_batches"] == len(jsonl(lf / "logs" / "batches.jsonl"))


def test_fake_batched_equals_sequential_and_resume(fake_runs):
    tmp, a, b = fake_runs
    for f in ("predictions.jsonl", "metrics.json"):
        assert (leaf(a) / f).read_bytes() == (leaf(b) / f).read_bytes(), f
    rows_a = [x["rows"] for x in jsonl(leaf(a) / "logs" / "batches.jsonl")]
    rows_b = [x["rows"] for x in jsonl(leaf(b) / "logs" / "batches.jsonl")]
    assert max(rows_a) == 2 and set(rows_b) == {1}
    # 续跑：去掉最后一行（第二组没写完）→ --resume → 逐字节相同；配置不同时拒绝续跑
    c = tmp / "c"
    run(tiny_args(c, *FAKE, "--slots", "3", "--batch-size", "2", "--chunk-size", "2", "--save", "1"), tmp)
    p = leaf(c) / "predictions.jsonl"
    full = p.read_bytes()
    p.write_bytes(b"".join(full.splitlines(keepends=True)[:3]))
    run(tiny_args(c, *FAKE, "--slots", "3", "--batch-size", "2", "--chunk-size", "2", "--save", "1", "--resume"), tmp)
    assert p.read_bytes() == full
    bad = run(tiny_args(c, *FAKE, "--slots", "2", "--batch-size", "2", "--chunk-size", "2", "--resume"), tmp, ok=False)
    assert bad.returncode == 2 and "拒绝混写" in bad.stderr


def test_real_tiny_model_end_to_end_batched_equals_sequential(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run(tiny_args(a, "--slots", "4", "--batch-size", "4", "--split", "all"), tmp_path)
    run(tiny_args(b, "--slots", "1", "--batch-size", "1", "--split", "all"), tmp_path)
    for split in R.test_splits():
        pa, pb = leaf(a, split) / "predictions.jsonl", leaf(b, split) / "predictions.jsonl"
        assert pa.read_bytes() == pb.read_bytes(), split
        m = json.loads((leaf(a, split) / "metrics.json").read_text())
        assert m["complete"] and m["truncation"]["all"]["of"] >= m["n"]
        calls = jsonl(leaf(a, split) / "logs" / "calls.jsonl")
        assert all(x["cap"] == 48 for x in calls)  # --tiny：上限再取 --cap-limit 48
        assert m["truncation"]["all"]["n"] == sum(1 for x in calls if x["truncated"])
        recs = jsonl(pa)
        # 随机模型给不出 JSON：官方在检测这一步失败（返回 1、没有 mask），照官方口径按空 mask 计分
        assert all(r["status"] == "failed" and r["pipeline_rc"] == 1 and r["pipeline_events"] == ["detect error"] for r in recs)
        assert all(r["score"]["inter"] == 0 and r["answer"]["mask"]["source"] == "empty" for r in recs)
        assert m["pipeline_events"] == {"detect error": m["n"]} and m["format_failure"]["detect"]["rate"] == 1.0
    rows = [x["rows"] for x in jsonl(leaf(a, SPLIT) / "logs" / "batches.jsonl")]
    assert max(rows) > 1


def test_generation_error_exits_2_without_scoring(tmp_path):
    p = run(tiny_args(tmp_path / "o", "--adapter-factory", "tests._llmseg_fakes:make_failing_adapter"), tmp_path, ok=False)
    assert p.returncode == 2 and "CUDA out of memory" in p.stderr
    lf = leaf(tmp_path / "o")
    assert not (lf / "predictions.jsonl").exists() or (lf / "predictions.jsonl").read_text() == ""
    assert list((tmp_path / "o" / QWEN / "errors").glob("llmseg_error_*.json"))


def test_cli_guards(tmp_path):
    base = ["--model", QWEN, "--out", str(tmp_path / "0001_x"), "--data-root", str(tmp_path)]
    for extra, msg in ((["--split", "devcheck2"], "--split 只能是"), (["--split", SPLIT, "--adapter-factory", "x:y"], "只配 --tiny"),
                       (["--split", SPLIT, "--thinking"], "--thinking-budget")):
        p = run(base + extra, tmp_path, ok=False)
        assert p.returncode == 2 and msg in p.stderr, (extra, p.stderr[-500:])
    p = run(["--model", "gemma-4-e2b-it", "--split", SPLIT, *base[2:]], tmp_path, ok=False)
    assert p.returncode == 2 and "Qwen3.5" in p.stderr
    p = run(["--model", QWEN, "--split", SPLIT, "--out", str(tmp_path / "x"), "--data-root", str(tmp_path)], tmp_path, ok=False)
    assert p.returncode == 2 and "运行编号" in p.stderr
    assert run(["--check-subsets"], tmp_path).stdout.startswith("一致")


def _w2c_leaf(root: Path, split: str, recs: list[dict], shift: int) -> Path:
    """一个假的 W2-c 叶目录：同一批 id 加 2 句别的；inter 在 LLMSeg 的基础上 + shift。"""
    lf = root / split
    lf.mkdir(parents=True)
    (lf / "config.yaml").write_text(yaml.safe_dump({"w2": {"split": split, "setting": "w2c"}, "model": {"key": "qwen3.5-9b"},
                                                    "checkpoint": {"path": "ckpt/final/model"}}, allow_unicode=True))
    rows = []
    for r in recs + [{"id": "extra/1", "score": {"inter": 1, "union": 2, "iou": 0.5}}, {"id": "extra/2", "score": {"inter": 0, "union": 3, "iou": 0.0}}]:
        s = r["score"]
        inter = min(s["union"], s["inter"] + shift) if r["id"].startswith("refcoco") else s["inter"]
        rows.append({"id": r["id"], "score": {"inter": inter, "union": s["union"], "iou": sample_iou(inter, s["union"])}})
    (lf / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return lf


def test_compare_with_w2c(fake_runs, tmp_path):
    _, a, _ = fake_runs
    recs = jsonl(leaf(a) / "predictions.jsonl")
    _w2c_leaf(tmp_path / "w2c", SPLIT, recs, shift=50)
    run(["--llmseg", str(a / QWEN / "refseg" / "llmseg"), "--w2c", str(tmp_path / "w2c"), "--out", str(tmp_path / "cmp"),
         "--splits", SPLIT], tmp_path, script=COMPARE)
    res = json.loads((tmp_path / "cmp" / "compare.json").read_text())
    w = {r["id"]: r for r in jsonl(tmp_path / "w2c" / SPLIT / "predictions.jsonl")}
    for ro, sc in (("rec", lambda r: r["score"]), ("seg", lambda r: r["stage2"]["score"])):
        d = res["splits"][SPLIT][ro]
        la = sum(sc(r)["inter"] for r in recs) / sum(sc(r)["union"] for r in recs)
        wb = sum(w[r["id"]]["score"]["inter"] for r in recs) / sum(w[r["id"]]["score"]["union"] for r in recs)
        assert d["cIoU"]["n"] == 4 and d["cIoU"]["a"] == pytest.approx(la) and d["cIoU"]["b"] == pytest.approx(wb)
        assert d["cIoU"]["diff"] == pytest.approx(la - wb) and d["cIoU"]["diff"] < 0
        gi = np.mean([sc(r)["iou"] for r in recs]) - np.mean([w[r["id"]]["score"]["iou"] for r in recs])
        assert d["gIoU"]["diff"] == pytest.approx(gi) and d["gIoU"]["ci"] is not None
        assert res["pooled"][ro]["cIoU"]["diff"] == pytest.approx(d["cIoU"]["diff"])  # 只有一个划分时合并 = 该划分
    assert res["flag_llmseg_not_lower"]["point_estimate_ge_0"] is False
    assert "| 合并 |" in (tmp_path / "cmp" / "compare.md").read_text()
    # 不是 W2-c 的目录：拒绝
    bad = tmp_path / "notw2c"
    _w2c_leaf(bad, SPLIT, recs, shift=0)
    cfg = yaml.safe_load((bad / SPLIT / "config.yaml").read_text())
    cfg["w2"]["setting"] = "w2a"
    (bad / SPLIT / "config.yaml").write_text(yaml.safe_dump(cfg))
    p = run(["--llmseg", str(a / QWEN / "refseg" / "llmseg"), "--w2c", str(bad), "--out", str(tmp_path / "cmp2"), "--splits", SPLIT],
            tmp_path, ok=False, script=COMPARE)
    assert p.returncode == 2 and "不是 W2-c" in p.stderr


def test_pipeline_event_tags_follow_official_log_lines():
    lines = ["[detect] error: Could not parse JSON from model output:", "[detect] 1 instance(s)",
             "[refine] pass 3: unparseable focus-box reply (x); quiet pass", "[segment] truncated reply; salvaged 1 complete entries",
             "[refine] malformed reply; repaired JSON structure", "[refine] aborted, keeping the passes completed so far: x"]
    assert R.pipeline_events(lines) == ["detect error", "refine pass unparseable", "segment truncated reply", "refine malformed reply",
                                        "refine aborted"]
    src = (V.VENDOR_DIR / "pipeline.py").read_text()
    for s in ('log(f"[detect] error:', 'log(f"[segment] error:', 'log(f"[refine] aborted', "unparseable focus-box reply",
              "unparseable spline reply", 'log(f"[{what}] truncated reply', 'log(f"[{what}] malformed reply'):
        assert s in src, s


def test_stage_constants_consistent():
    assert set(PV.STAGE2_STAGES) <= set(PV.NEW_ROUND_STAGES) and PV.STAGE_FOCUS_SPLINES not in PV.NEW_ROUND_STAGES
    assert R.TARGET_N_ITEMS == 1500
