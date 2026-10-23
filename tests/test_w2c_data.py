"""DEV-0060：W2-c 的训练数据（src/w2/w2c_data.py、scripts/build_w2c_data.py；格式 docs/w2_plan.md §5）。

- 规则（纯函数）：按图像排除评测图与抽样（select_sentences）、manifest 的自洽（check_manifest）、标签（score_candidates：真值 IoU 最高、
  并列取编号小的；1 号候选失败当空 mask；其余失败的用框）、统计（data_metrics）。
- fixture（tests/fixtures/w2c/root）上的总体、去重与抽样：E3 的剔除清单去掉 coco:2005，W2 的评测图（--tiny 划分的 901–906）上的句子去掉，
  选中的与任何评测图都不重叠；抽样可复现（两次逐字节相同，且等于独立按 hash_rank 重算的结果）。
- 命令行 --tiny 端到端（子进程，与服务器同一流程）：出框（小型随机 Qwen，E1 的运行器）→ 换成合法的框（随机模型给不出框；第一条保留
  解析失败，第二条是整图的框，GrabCut 必然失败）→ 候选与标签（真的候选生成器）；§5 的格式、分片、运行身份、统计。
- 续跑逐字节相同（删掉一片、留下半片 .tmp；--stop-after-shards 分段跑）；--candidate-workers 2 与单进程逐字节相同；
  配置未冻结时非 tiny 拒跑（候选一步与 all），出框一步不受限；身份或清单不同就拒绝混写；出框写完不再跑模型。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.subsets import hash_rank
from src.e1.runner import load_images
from src.e1.scoring import gt_mask
from src.executors.raster import box_to_mask, mask_to_rle, rle_to_mask
from src.w1.baselines import workbench_grabcut
from src.w2 import w2c_data as D
from src.w2.config import load_w2_config
from src.w2.data import load_splits

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_w2c_data.py"
W2_JSON = REPO / "configs" / "w2.json"
DETERMINISTIC = ("w2c_data/manifest.json", "metrics.json", "README.md", "config.yaml")


def run(args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def tiny_samples(n: int = D.TINY_N):
    return D.prepare(load_w2_config(), D.W2CPaths.tiny(), n=n, tiny=True)


def inject_boxes(out: Path) -> None:
    """把出框的记录换成合法的框：第一条保留随机模型的解析失败；第二条是整图的框（框外没有像素，GrabCut 必然失败）；其余是 GT 框内缩
    1 像素。只改 failure 与 parsed（同 tests/test_w2_cli.inject_boxes）。"""
    _, samples = tiny_samples()
    gt = {s.id: (s.gt["box"], s.fields["image_size"]) for s in samples}
    p = out / D.PROPOSE_DIR / "predictions.jsonl"
    recs = jsonl(p)
    assert recs[0]["failure"] is not None  # 随机模型给不出框
    for i, r in enumerate(recs[1:], start=1):
        (x0, y0, x1, y1), (w, h) = gt[r["id"]]
        box = [0.0, 0.0, float(w), float(h)] if i == 1 else [x0 + 1, y0 + 1, x1 - 1, y1]
        r.update(failure=None, parsed={"box": box})
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def strip_run(src: Path, dst: Path) -> None:
    """复制一个运行目录，只留出框与清单（候选一步从头再来）。"""
    shutil.copytree(src, dst)
    shutil.rmtree(dst / "logs", ignore_errors=True)
    for p in D.shard_paths(dst):
        p.unlink()
    for name in ("config.yaml", "meta.json", "metrics.json", "README.md"):
        (dst / name).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w2c")
    out = tmp / "out"
    logs = {"propose": run(["--tiny", "--stage", "propose", "--out", str(out)], tmp)}
    inject_boxes(out)
    logs["candidates"] = run(["--tiny", "--stage", "candidates", "--out", str(out)], tmp)
    return {"tmp": tmp, "out": out, "logs": logs, "ref": {f: (out / f).read_bytes() for f in DETERMINISTIC},
            "shards": {p.name: p.read_bytes() for p in D.shard_paths(out)}}


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------


def test_select_sentences_excludes_eval_images_and_ranks():
    pop = [(f"refcoco_unc/train/{i}", f"coco:{i % 7}") for i in range(60)]
    sel = D.select_sentences(pop, ["coco:0", "coco:3", "coco:99"], n=20)
    kept = [i for i, k in pop if k not in ("coco:0", "coco:3")]
    assert sel["ids"] == hash_rank(kept, D.PURPOSE, D.SALT)[:20]  # 哈希顺序就是处理顺序
    assert all(sel["image_of"][i] not in ("coco:0", "coco:3") for i in sel["ids"])
    st = sel["steps"]
    assert (st["w2_eval_exclusion"]["n_before"], st["w2_eval_exclusion"]["n_removed"], st["w2_eval_exclusion"]["n_after"]) == (60, 18, 42)
    assert st["w2_eval_exclusion"]["n_images_hit"] == 2 and st["selected"]["n"] == 20
    assert D.select_sentences(pop, ["coco:0", "coco:3"], n=20) == sel  # 可复现
    assert D.select_sentences(pop, [], n=20)["ids"] != sel["ids"]
    with pytest.raises(D.W2CDataError, match="不够"):
        D.select_sentences(pop, ["coco:0", "coco:3"], n=44)
    assert len(D.select_sentences(pop, ["coco:0", "coco:3"], n=44, allow_fewer=True)["ids"]) == 42
    with pytest.raises(D.W2CDataError, match="重复"):
        D.select_sentences(pop + pop[:1], [], n=5)
    assert D.select_sentences(pop, [], n=5, salt="other")["ids"] != D.select_sentences(pop, [], n=5)["ids"]


def _cand_row(masks, oks, *, h=20, w=30, sid="x"):
    return {"id": sid, "image_size": [w, h], "proposal": {"mode": "box", "box": [0.0, 0.0, 5.0, 5.0], "failure": None,
                                                          "fg_points": [], "bg_points": []},
            "k": len(masks), "n_tool_calls": 1,
            "candidates": [{"index": i, "name": f"g{i}", "ok": ok, "info": {}, "mask": {"rle": mask_to_rle(m), "area": int(m.sum())}}
                           for i, (m, ok) in enumerate(zip(masks, oks), start=1)]}


def test_score_candidates_label_ties_and_first_failure():
    h, w = 20, 30
    gt = box_to_mask([5, 5, 15, 15], h, w)
    near = box_to_mask([5, 5, 14, 15], h, w)
    worse = box_to_mask([2, 2, 15, 15], h, w)
    # 1 号失败（ok = False）时生成器给的是框；这里它恰好等于 GT，若照原样算会是最优——W2 的规则把它当空 mask
    row = _cand_row([gt, near, near, worse, box_to_mask([0, 0, 30, 20], h, w)], [False, True, True, True, False])
    cands, label = D.score_candidates(row, gt)
    assert cands[0]["ok"] is False and cands[0]["iou"] == 0.0 and not rle_to_mask(cands[0]["rle"]).any()
    assert cands[1]["iou"] == cands[2]["iou"] == pytest.approx(90 / 100) and label == 2  # 并列取编号小的
    assert cands[4]["ok"] is False and rle_to_mask(cands[4]["rle"]).all()  # 其余失败的仍用框（原样）
    assert [c["rle"] for c in cands[1:]] == [c["mask"]["rle"] for c in row["candidates"][1:]]
    assert [tuple(c) for c in cands] == [D.CANDIDATE_KEYS] * 5
    # 1 号正常时就照它算；与更大编号并列时取 1
    cands, label = D.score_candidates(_cand_row([near, near, worse], [True, True, True]), gt)
    assert label == 1 and cands[0]["iou"] == cands[1]["iou"]
    assert D.score_candidates({**row, "candidates": None}, gt) == (None, None)
    with pytest.raises(D.W2CDataError, match="形状"):
        D.score_candidates(row, np.zeros((h + 1, w), dtype=bool))


def _row(label, ious, *, failure=None, ok1=True, sid="r"):
    c = None if failure else [{"index": i, "name": f"g{i}", "ok": (ok1 if i == 1 else True),
                               "rle": mask_to_rle(np.zeros((4, 4), dtype=bool) if (i == 1 and not ok1) else np.eye(4, dtype=bool)),
                               "iou": v} for i, v in enumerate(ious, start=1)]
    return {"id": sid, "source": "refcoco", "split": "train", "image": "refcoco_train/images/a.jpg", "image_size": [4, 4],
            "expression": "e", "gt_box": [0.0, 0.0, 1.0, 1.0], "gt_rle": mask_to_rle(np.eye(4, dtype=bool)),
            "proposal": {"box": None if failure else [0.0, 0.0, 1.0, 1.0], "failure": failure, "raw_text": "t"},
            "candidates": c, "label": None if failure else label, "spec_sha256": "s", "w2_config_sha256": "c"}


def test_data_metrics_counts_and_self_check():
    rows = [_row(2, [0.5, 0.7, 0.7]), _row(1, [0.9, 0.1, 0.2]), _row(3, [0.0, 0.2, 0.6], ok1=False), _row(None, [], failure="no_box")]
    assert all(D.check_row(r, k=3) == [] for r in rows)
    m = D.data_metrics(rows)
    assert m["n"] == 4 and m["proposal"] == {"n_failed": 1, "failure_rate": 0.25, "failures": {"no_box": 1}}
    assert m["n_with_candidates"] == 3 and m["k"] == 3
    assert m["label"]["counts"] == {"1": 1, "2": 1, "3": 1, "null": 1}
    assert m["best_with_ties"]["counts"] == {"1": 1, "2": 1, "3": 2} and m["best_with_ties"]["n_rows_with_ties"] == 1
    assert m["best_iou"]["mean_on_rows_with_candidates"] == pytest.approx((0.7 + 0.9 + 0.6) / 3)
    assert m["first_iou"]["mean_on_rows_with_candidates"] == pytest.approx((0.5 + 0.9 + 0.0) / 3)
    assert m["first_iou"]["mean_on_all_rows_failure_as_0"] == pytest.approx((0.5 + 0.9 + 0.0) / 4)
    assert m["per_index"]["1"]["n_not_ok"] == 1 and m["per_index"]["3"]["n_label"] == 1
    assert m["duplicates"]["n"] == 5 and m["duplicates"]["rate"] == pytest.approx(5 / 9)  # 与更小编号 RLE 相同的：2 + 2 + 1
    with pytest.raises(D.W2CDataError, match="label"):
        D.data_metrics([_row(3, [0.5, 0.7, 0.7])])  # 并列时 label 必须是编号小的
    assert D.check_row(_row(3, [0.5, 0.7, 0.7]), k=3) and D.check_row({**_row(1, [0.9]), "label": None}, k=1)
    # GT 为空（并集为 0）的唯一例外：空候选的 IoU 记 1.0、oracle 按 0 比，label = 1
    assert D.data_metrics([_row(1, [0.0, 1.0])])["label"]["n_label_union0_exception"] == 1


# ---------------------------------------------------------------------------
# fixture 上的总体、去重与抽样
# ---------------------------------------------------------------------------


def test_prepare_dedupes_and_samples_reproducibly():
    m1, s1 = tiny_samples()
    m2, s2 = tiny_samples()
    assert D.manifest_text(m1) == D.manifest_text(m2) and [s.id for s in s1] == [s.id for s in s2]  # 可复现
    st = m1["steps"]
    pop = st["population"]
    assert pop["datasets"]["refcoco"] == {"n_total": 24, "n_missing": 0, "n_dedupe_excluded": 2, "n_kept": 22}
    assert pop["datasets"]["refcoco+"]["n_dedupe_excluded"] == 1 and pop["n_empty_expression"] == 1 and pop["n"] == 42
    assert st["e3_exclusions"]["file"] == "dedupe/train_exclusions.json" and st["e3_exclusions"]["n_sentences_removed"] == 3
    assert st["e3_exclusions"]["n_refcoco_train_entries"] == 1
    ev = st["w2_eval_images"]
    assert set(ev["splits"]) == {"devtune800", "devcheck2", "test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA",
                                 "test_refcoco+_unc_testB", "test_refcocog_umd_test"}
    assert ev["images"] == [f"coco:{i}" for i in range(901, 907)]
    ex = st["w2_eval_exclusion"]
    assert (ex["n_before"], ex["n_removed"], ex["n_images_hit"], ex["n_after"]) == (42, 3, 2, 39)  # 901 上 2 句、905 上 1 句
    # 独立重算：fixture 的全部 train 句子去掉 2005（剔除清单）、901、905（评测图）与空句，按 hash_rank 取前 10
    pool_all, _ = D.load_train_pool(D.W2CPaths.tiny(), real_model=False)
    ok = [s.id for s in pool_all if s.image_keys[0] not in ev["images"]]
    assert len(ok) == 39 and [s.id for s in s1] == hash_rank(ok, "w2c_train", "w2c_train_v1")[:10]
    keys = {s.image_keys[0] for s in s1}
    assert not keys & set(ev["images"]) and "coco:2005" not in {s.image_keys[0] for s in pool_all}  # 与任何评测图都不重叠
    assert m1["checks"]["selected_on_w2_eval_images"] == 0 and D.check_manifest(m1) == []
    assert m1["selected"]["ids"] == [s.id for s in s1]
    assert "/Users" not in D.manifest_text(m1) and str(REPO) not in D.manifest_text(m1)  # 不含机器相关的路径
    # 改坏：选中的一句挪到评测图上 → 核对不过
    bad = json.loads(D.manifest_text(m1))
    bad["selected"]["image_of"][s1[0].id] = "coco:905"
    assert any("评测图" in x for x in D.check_manifest(bad))
    m20, s20 = tiny_samples(n=20)
    assert [s.id for s in s20][:10] == [s.id for s in s1]  # 前缀性质：前 k 片恰是更小 n 的抽样


def test_e3_exclusions_are_required(tmp_path):
    """沿用 E3 训练池的剔除清单：真模型口径（real_model = True）照样过；清单缺了 E3 要求的评测集、或没有清单，都报错（硬约束 7）。"""
    pool, steps = D.load_train_pool(D.W2CPaths.tiny(), real_model=True)
    assert len(pool) == 42 and steps["e3_exclusions"]["required_eval_sets"] == D.required_eval_sets()
    root = tmp_path / "root"
    shutil.copytree(D.W2C_FIXTURE_ROOT, root)
    p = root / "dedupe" / "train_exclusions.json"
    doc = json.loads(p.read_text())
    doc["eval_sets"] = [s for s in doc["eval_sets"] if s != "refcoco/eval"]
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    with pytest.raises(ValueError, match="refcoco/eval"):
        D.load_train_pool(D.W2CPaths(train_root=root, eval=D.W2CPaths.tiny().eval), real_model=True)
    p.unlink()
    with pytest.raises(FileNotFoundError, match="硬约束 7"):
        D.load_train_pool(D.W2CPaths(train_root=root, eval=D.W2CPaths.tiny().eval), real_model=False)


def test_real_splits_have_all_eval_splits_built():
    """入库的 configs/w2_splits.json：refseg 的划分恰是 devtune800、devcheck2 与 5 个 test 子集，devcheck2 已生成（不要数据）。"""
    names = D.eval_split_names(load_w2_config(), load_splits())
    assert names[:2] == ["devtune800", "devcheck2"] and len(names) == 7 and all(n.startswith("test_") for n in names[2:])


# ---------------------------------------------------------------------------
# 命令行 --tiny 端到端
# ---------------------------------------------------------------------------


def test_end_to_end_format_and_identity(tiny_run):
    out = tiny_run["out"]
    cfg = load_w2_config()
    man = D.load_manifest(out)
    shards = D.shard_paths(out)
    assert [p.name for p in shards] == ["00000.jsonl", "00001.jsonl", "00002.jsonl"]
    assert [len(jsonl(p)) for p in shards] == [4, 4, 2]
    rows = list(D.read_rows(out))
    assert [r["id"] for r in rows] == man["selected"]["ids"]
    _, samples = tiny_samples()
    by = {s.id: s for s in samples}
    for r in rows:
        assert D.check_row(r, k=cfg.k) == [], r["id"]
        assert r["spec_sha256"] == cfg.candidates_digest() and r["w2_config_sha256"] == hashlib.sha256(W2_JSON.read_bytes()).hexdigest()
        s = by[r["id"]]
        assert (D.W2C_FIXTURE_ROOT / r["image"]).is_file() and r["image"].startswith("refcoco_train/images/")
        assert r["source"] == s.source and r["split"] == "train" and r["expression"] == s.fields["expression"]
        assert r["image_size"] == s.fields["image_size"] and r["gt_box"] == list(s.gt["box"])
        assert np.array_equal(rle_to_mask(r["gt_rle"]), gt_mask(s))
        if r["candidates"] is not None:
            assert [c["name"] for c in r["candidates"]][0] == "gc_default"
            g = gt_mask(s)
            for c in r["candidates"]:
                m = rle_to_mask(c["rle"])
                assert c["iou"] == pytest.approx((m & g).sum() / (m | g).sum())
    # 仪器：1 号候选逐位等于 W1-F（E1 的框交给工作台的同一个 GrabCut）
    gc = workbench_grabcut()
    n_first = 0
    for r in rows:
        if r["candidates"] is not None and r["candidates"][0]["ok"]:
            img = np.asarray(load_images(by[r["id"]])[0])
            assert r["candidates"][0]["rle"] == mask_to_rle(gc.fn(img, box_abs=r["proposal"]["box"], iterations=gc.iterations))
            n_first += 1
    assert n_first == 8
    assert rows[0]["proposal"]["failure"] is not None and rows[0]["candidates"] is None and rows[0]["label"] is None
    assert rows[0]["proposal"]["raw_text"] == jsonl(out / D.PROPOSE_DIR / "predictions.jsonl")[0]["raw_text"]
    full = rows[1]  # 整图的框：GrabCut 失败，1 号当空
    assert full["candidates"][0]["ok"] is False and full["candidates"][0]["iou"] == 0.0 and full["label"] != 1
    assert not rle_to_mask(full["candidates"][0]["rle"]).any()
    m = json.loads((out / "metrics.json").read_text())
    assert m["complete"] is True and m["n"] == 10 and m["n_shards"] == {"done": 3, "expected": 3}
    assert m["proposal"]["n_failed"] == 1 and m["n_with_candidates"] == 9 and m["label"]["counts"]["null"] == 1
    assert sum(v for k, v in m["label"]["counts"].items() if k != "null") == 9 and m["per_index"]["1"]["n_not_ok"] >= 1
    assert m["best_iou"]["mean_on_rows_with_candidates"] >= m["first_iou"]["mean_on_rows_with_candidates"]
    c = yaml.safe_load((out / "config.yaml").read_text())
    assert c["model"]["key"] == "qwen3.5-0.8b" and c["model"]["tiny"] is True and len(c["model"]["revision"]) == 40
    assert c["w2_config"]["config_sha256"] == cfg.sha256 and c["w2_config"]["frozen"] is False
    assert c["candidates"]["spec_sha256"] == cfg.candidates_digest() and c["candidates"]["generators"] == list(cfg.generators)
    assert c["sampling"]["salt"] == "w2c_train_v1" and c["sampling"]["purpose"] == "w2c_train" and c["sampling"]["n"] == 10
    msha = hashlib.sha256(D.manifest_path(out).read_bytes()).hexdigest()
    assert D.manifest_path(out) == out / "w2c_data" / "manifest.json"  # docs/w2_plan.md §5：数据目录的 manifest.json
    assert c["dedupe"]["manifest_sha256"] == msha and c["dedupe"]["e3_exclusions_sha256"] == man["steps"]["e3_exclusions"]["sha256"]
    assert c["propose"]["run"]["leaf"] == "<out>/propose" and c["shard_size"] == 4
    pc = yaml.safe_load((out / D.PROPOSE_DIR / "config.yaml").read_text())  # 出框是 E1 格式的运行
    assert pc["experiment"] == "E1" and pc["task"] == "refexp_box" and pc["subset"] == "eval"
    assert pc["generation"]["enable_thinking"] is False and pc["data"]["selection_stats"]["w2c"]["manifest_sha256"] == msha
    assert (out / "meta.json").exists() and (out / D.PROPOSE_DIR / "metrics.json").exists()
    t = jsonl(out / "logs" / "timing.jsonl")
    assert [x["shard"] for x in t] == [0] * 4 + [1] * 4 + [2] * 2 and all(x["wall_time_s"] >= 0 for x in t)
    assert not list((out / D.DATA_DIR).glob("*.tmp"))


def test_resume_is_byte_identical(tiny_run, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(tiny_run["out"], out)
    d = out / D.DATA_DIR
    part = (d / "00001.jsonl").read_bytes()
    (d / "00001.jsonl.tmp").write_bytes(part[: len(part) // 2])  # 中断在第二片中间：只留下半片 .tmp
    (d / "00001.jsonl").unlink()
    (d / "00002.jsonl").unlink()
    (out / "metrics.json").unlink()
    p = run(["--tiny", "--stage", "candidates", "--out", str(out)], tmp_path)
    assert "片 1/3" not in p.stderr and "片 2/3" in p.stderr  # 写完的片不重算
    assert {q.name: q.read_bytes() for q in D.shard_paths(out)} == tiny_run["shards"]
    assert {f: (out / f).read_bytes() for f in DETERMINISTIC} == tiny_run["ref"]
    assert not list(d.glob("*.tmp"))
    # 分段跑：--stop-after-shards 1，三次写完
    seg = tmp_path / "seg"
    strip_run(tiny_run["out"], seg)
    for k in (1, 2):
        run(["--tiny", "--stage", "candidates", "--out", str(seg), "--stop-after-shards", "1"], tmp_path)
        m = json.loads((seg / "metrics.json").read_text())
        assert m["n_shards"]["done"] == k and m["complete"] is False
    run(["--tiny", "--stage", "candidates", "--out", str(seg)], tmp_path)
    assert {q.name: q.read_bytes() for q in D.shard_paths(seg)} == tiny_run["shards"]
    assert {f: (seg / f).read_bytes() for f in DETERMINISTIC} == tiny_run["ref"]
    assert len(list((seg / "logs").glob("run_*.json"))) == 3


def test_candidate_workers_give_identical_bytes(tiny_run, tmp_path):
    out = tmp_path / "w2"
    strip_run(tiny_run["out"], out)
    run(["--tiny", "--stage", "candidates", "--out", str(out), "--candidate-workers", "2"], tmp_path)
    assert {q.name: q.read_bytes() for q in D.shard_paths(out)} == tiny_run["shards"]
    assert {f: (out / f).read_bytes() for f in DETERMINISTIC} == tiny_run["ref"]  # 进程数不进运行身份
    assert json.loads(next((out / "logs").glob("run_*.json")).read_text())["candidate_workers"] == 2


def test_rerun_all_reads_proposals_without_the_model(tiny_run, tmp_path):
    out = tmp_path / "again"
    shutil.copytree(tiny_run["out"], out)
    p = run(["--tiny", "--out", str(out)], tmp_path)  # --stage all
    assert "出框已跑完" in p.stderr and "refexp_box/eval" not in p.stderr  # 不再跑模型
    assert {q.name: q.read_bytes() for q in D.shard_paths(out)} == tiny_run["shards"]
    assert {f: (out / f).read_bytes() for f in DETERMINISTIC} == tiny_run["ref"]


def test_mixing_is_refused(tiny_run, tmp_path):
    out = tmp_path / "mix"
    shutil.copytree(tiny_run["out"], out)
    k4 = tmp_path / "w2_k4.json"
    d = json.loads(W2_JSON.read_text())
    d["candidates"]["generators"] = d["candidates"]["generators"][:4]
    k4.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    p = run(["--tiny", "--stage", "candidates", "--out", str(out), "--w2-config", str(k4)], tmp_path, ok=False)
    assert p.returncode == 2 and "拒绝混写" in p.stderr  # 换了候选清单：身份不同
    p = run(["--tiny", "--stage", "candidates", "--out", str(out), "--n", "9"], tmp_path, ok=False)
    assert p.returncode == 2 and "拒绝混用" in p.stderr  # 换了 n：清单不同
    assert {q.name: q.read_bytes() for q in D.shard_paths(out)} == tiny_run["shards"]


def test_unfrozen_config_is_refused_without_tiny(tmp_path):
    common = ["--train-data-root", str(tmp_path / "train"), "--eval-data-root", str(tmp_path / "eval")]
    assert json.loads(W2_JSON.read_text())["frozen"] is False  # 入库的配置还没冻结（W2-a 调参之后由管理员冻结）
    for stage in ("candidates", "all"):
        out = tmp_path / f"0099_w2c_{stage}"
        p = run(["--stage", stage, "--out", str(out), *common], tmp_path, ok=False)
        assert p.returncode == 2 and "未冻结" in p.stderr and not out.exists(), stage  # 什么都不写
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import build_w2c_data as B
    finally:
        sys.path.pop(0)
    a = B.parse_args(["--stage", "propose", "--out", str(tmp_path / "0099_w2c_p"), *common])  # 出框不读候选清单：不受限
    assert a.model == "qwen3.5-9b" and a.n == 16000 and a.shard_size == 1000 and a.candidate_workers == 1
    frozen = tmp_path / "w2_frozen.json"
    frozen.write_text(json.dumps({**json.loads(W2_JSON.read_text()), "frozen": True}, ensure_ascii=False, indent=1) + "\n")
    a = B.parse_args(["--stage", "candidates", "--out", str(tmp_path / "0099_w2c_c"), "--w2-config", str(frozen), *common])
    assert a.stage == "candidates"
    for bad in (["--out", str(tmp_path / "w2c_no_number"), *common], ["--out", str(tmp_path / "0099_x"), "--shard-size", "4", *common],
                ["--out", str(tmp_path / "0099_x"), "--eval-data-root", "e"]):
        with pytest.raises(SystemExit):
            B.parse_args(["--stage", "propose", *bad])
