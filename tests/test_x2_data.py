"""DEV-0069：X-2 的训练数据（src/explore/x2_data.py、scripts/build_x2_data.py）。CPU，测试造的数据（tests/_x2_fakes：带 gt_rle 的 §5 假行）。

- 样条行：目标 = oracle_spline(gt, n, parts = 1) 的网格整数，fit_iou = 推理时同一套光栅化（框约束）与真值的 IoU；框 IoU < 0.5 的行不用；
- 修正的目标规则：差的 8 连通块、限在框内、按面积排序、effect gate（门槛同 guidance）、取前 m 个、符号；都过不了 → []；
- 起点的三档轮换（好 = 最好的、差 = 最差的、中 = 中间的，各约 1/3，确定）；第 2 轮 = 第 1 轮的目标按 grabcut + 软锁定施加后的 mask，
  目标按同样的规则（effect gate 用第 1 轮之后的承诺场）；第 1 轮为 [] 的不造第 2 轮；第 2 轮的比例；
- 输出与 manifest：条数、过滤原因、sha256；分段续跑、多进程与一次跑完逐字节相同；拒绝混写；
- 读取：参数、来源、sha256、目标文本的核对。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.executors.raster import box_to_mask, mask_to_rle, rle_to_mask
from src.explore import guidance as G
from src.explore import oracle_sim as OS
from src.explore import x2_data as XD
from src.explore import x2_prompts as XP
from src.w2.config import load_w2_config
from tests import _x2_fakes as XF

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "scripts" / "build_x2_data.py"
CFG_DOC = json.loads((REPO / "configs" / "x2_train.json").read_text(encoding="utf-8"))
PARAMS = XD.check_params({k: CFG_DOC[k] for k in ("spline", "correct")})
W2 = load_w2_config()


@pytest.fixture(scope="module")
def w2c_dir(tmp_path_factory):
    return XF.write_fake_w2c_data(tmp_path_factory.mktemp("x2w") / "w2c_data")


@pytest.fixture(scope="module")
def built(w2c_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("x2b") / "x2"
    man = XD.build(w2c_dir, out, data_root=XF.ROOT, params=PARAMS, tiny=True, log=lambda m: None)
    return out, man


def _files(out: Path) -> dict[str, bytes]:
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file() and "logs" not in p.parts}


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def test_params_from_repo_config_and_validation():
    assert PARAMS == {"spline": {"n": 16, "min_box_iou": 0.5},
                      "correct": {"m": 2, "min_box_iou": None, "start_salt": "x2_correct_start_v1", "round2_ratio": 0.5,
                                  "round2_salt": "x2_correct_round2_v1", "round2_apply": {"rule": "grabcut", "soft_lock": True}}}
    assert PARAMS == XD.check_params(XD.DEFAULT_PARAMS)
    for edit, msg in [(lambda d: d["spline"].update(n=3), "spline.n"), (lambda d: d["spline"].update(min_box_iou=1.5), "min_box_iou"),
                      (lambda d: d["correct"].update(m=0), "correct.m"), (lambda d: d["correct"].update(round2_ratio=-0.1), "round2_ratio"),
                      (lambda d: d["correct"].update(round2_salt="x2_correct_start_v1"), "不能相同"),
                      (lambda d: d["correct"].update(round2_apply={"rule": "magic", "soft_lock": True}), "round2_apply"),
                      (lambda d: d["correct"].pop("start_salt"), "start_salt")]:
        d = json.loads(json.dumps(XD.DEFAULT_PARAMS))
        edit(d)
        with pytest.raises(XD.X2DataError, match=msg):
            XD.check_params(d)


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------


def _ellipse(h=96, w=128, cx=60.5, cy=50.5, a=30, b=20):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) ** 2 / a ** 2 + (yy - cy) ** 2 / b ** 2) <= 1


def _row(gt, box, *, rid="refcoco_unc/train/1", cands=None, gt_box=None):
    h, w = gt.shape
    rows, cols = np.nonzero(gt)
    gb = gt_box or [float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)]
    cands = cands if cands is not None else [gt, box_to_mask(box, h, w), np.zeros_like(gt), gt]
    cc = []
    for i, m in enumerate(cands, start=1):
        cc.append({"index": i, "name": f"c{i}", "ok": True, "rle": mask_to_rle(m), "iou": float((m & gt).sum() / max((m | gt).sum(), 1))})
    return {"id": rid, "source": "refcoco", "split": "train", "image": "x.jpg", "image_size": [w, h], "expression": "the thing",
            "gt_box": gb, "gt_rle": mask_to_rle(gt), "proposal": {"box": list(box), "failure": None, "raw_text": ""}, "candidates": cc,
            "label": 1}


def test_spline_row_target_and_fit_ceiling():
    gt = _ellipse()
    box = [30.0, 30.0, 92.0, 72.0]
    r, why = XD.spline_row(_row(gt, box), gt, box, 0.9, 16)
    assert why is None and r["n"] == 16 and len(r["ctrl_q"]) == 16
    sp = OS.oracle_spline(gt, 16, parts=1, holes=False)
    assert r["ctrl_q"] == [[int(x), int(y)] for x, y in sp["loops_q"][0]["ctrl"]]
    assert r["target_text"] == XP.spline_text(r["ctrl_q"]) and XP.parse_spline(r["target_text"], n=16).value == r["ctrl_q"]
    fit = XP.spline_mask(r["ctrl_q"], box, gt.shape)
    assert r["fit_iou"] == pytest.approx((fit & gt).sum() / (fit | gt).sum()) and r["fit_iou"] > 0.95
    assert r["box_q"] == XP.box_q(box, 128, 96)
    # 拟合只看最大的连通域（填洞后）：多一个小块、挖一个洞，目标不变
    g2 = gt.copy()
    g2[5:9, 5:9] = True
    g2[48:52, 58:62] = False
    r2, _ = XD.spline_row(_row(g2, box), g2, box, 0.9, 16)
    assert r2["ctrl_q"] == r["ctrl_q"]
    # 框截掉一块：上限随之下降（采样点夹在框内）
    r3, _ = XD.spline_row(_row(gt, [30.0, 30.0, 70.0, 72.0]), gt, [30.0, 30.0, 70.0, 72.0], 0.6, 16)
    assert r3["fit_iou"] < 0.8


def test_sentence_job_filters():
    gt = _ellipse()
    good = _row(gt, [30.0, 30.0, 92.0, 72.0])
    res, _ = XD.sentence_job((good, str(XF.ROOT), PARAMS))
    assert res["spline"] is not None and res["spline_skip"] is None and res["correct"]
    bad_box = _row(gt, [0.0, 0.0, 40.0, 40.0])  # 与真值框 IoU < 0.5：样条不用，修正照做（min_box_iou = null）
    res, _ = XD.sentence_job((bad_box, str(XF.ROOT), PARAMS))
    assert res["spline"] is None and res["spline_skip"] == "box_iou_low" and len(res["correct"]) >= 1
    p2 = json.loads(json.dumps(PARAMS))
    p2["correct"]["min_box_iou"] = 0.5
    res, _ = XD.sentence_job((bad_box, str(XF.ROOT), p2))
    assert res["correct"] == [] and res["correct_skip"] == "box_iou_low"
    failed = dict(good, proposal={"box": None, "failure": "no_box", "raw_text": ""}, candidates=None, label=None)
    res, _ = XD.sentence_job((failed, str(XF.ROOT), PARAMS))
    assert res["spline"] is None and res["correct"] == [] and res["spline_skip"] == res["correct_skip"] == "proposal_failed"
    empty = dict(good, gt_rle=mask_to_rle(np.zeros_like(gt)))
    res, _ = XD.sentence_job((empty, str(XF.ROOT), PARAMS))
    assert res["spline_skip"] == res["correct_skip"] == "gt_empty"


def test_correction_targets_rules():
    h, w = 96, 128
    gt = np.zeros((h, w), bool)
    gt[20:70, 20:100] = True
    box = [10.0, 10.0, 110.0, 80.0]
    start = gt.copy()
    start[20:40, 20:50] = False  # 漏掉 600 px → "+"
    start[75:79, 40:90] = True  # 多出 200 px（在框内）→ "-"
    start[60:62, 95:97] = False  # 漏掉 4 px：过不了 effect gate（< 16 px）
    start[85:95, 0:20] = True  # 框外多出：改不了，不算
    tgt, info = XD.correction_targets(start, gt, None, box, 3)
    assert [t["sign"] for t in tgt] == ["+", "-"] and info["areas"] == [600, 200]
    assert tgt[0]["box"] == OS.quantize_box([20, 20, 50, 40], w, h) and tgt[1]["box"] == OS.quantize_box([40, 75, 90, 79], w, h)
    assert info["n_gated"] == 1 and info["n_components"] == 3
    one, _ = XD.correction_targets(start, gt, None, box, 1)
    assert one == tgt[:1]  # 按面积取前 m 个
    assert XD.correction_targets(gt, gt, None, box, 2)[0] == []  # 没有差：[]
    tiny = gt.copy()
    tiny[30:33, 30:33] = False  # 9 px：门槛之下
    assert XD.correction_targets(tiny, gt, None, box, 2)[0] == []
    # 承诺场传进 effect gate（guidance.achievable_px："+" 只算 Λ ≤ 0 的背景像素）：人为把漏掉那块的 Λ 设成 +1，"+" 就被 gate 去掉
    lock = np.zeros((h, w))
    lock[20:40, 20:50] = 1.0
    t2, i2 = XD.correction_targets(start, gt, lock, box, 3)
    assert [t["sign"] for t in t2] == ["-"] and i2["n_gated"] == 2
    # 目标里的每个框，经 correction_round 的约束框与 effect gate 都保留（与推理时的处理一致）
    img = np.zeros((h, w, 3), np.uint8)
    _, _, ci = G.correction_round(img, start, None, XP.proposals_abs(tgt, w, h), rule="direct", confine_box=box)
    assert ci["n_applied"] == len(tgt) and ci["n_gated"] == 0 and ci["n_outside_confine"] == 0


def test_start_choice_tiers_and_determinism():
    cands = [{"index": i, "name": f"c{i}", "iou": v} for i, v in enumerate([0.5, 0.9, 0.1, 0.5], start=1)]
    assert XD.iou_ranking(cands) == [1, 0, 3, 2]  # 并列取编号小的
    tiers = {}
    for k in range(3000):
        s = XD.start_choice(f"id/{k}", cands, "salt")
        assert s == XD.start_choice(f"id/{k}", cands, "salt")
        tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1
        if s["tier"] == "good":
            assert s["index"] == 2 and s["rank"] == 0
        elif s["tier"] == "bad":
            assert s["index"] == 3 and s["rank"] == 3
        else:
            assert s["index"] in (1, 4) and s["rank"] in (1, 2)
    assert all(abs(v / 3000 - 1 / 3) < 0.04 for v in tiers.values()) and set(tiers) == set(XD.TIERS)
    r2 = sum(XD.wants_round2(f"id/{k}", "s2", 0.5) for k in range(4000))
    assert abs(r2 / 4000 - 0.5) < 0.03 and not any(XD.wants_round2(f"id/{k}", "s2", 0.0) for k in range(100))


def test_round2_is_round1_applied_with_grabcut_and_soft_lock():
    rows = XF.fake_rows(W2)
    by = {r["id"]: r for r in rows}
    n2 = 0
    for r in rows:
        res, _ = XD.sentence_job((r, str(XF.ROOT), PARAMS))
        if len(res["correct"]) < 2:
            continue
        n2 += 1
        r1, r2 = res["correct"]
        assert (r1["round"], r2["round"], r1["sample_id"], r2["sample_id"]) == (1, 2, r["id"], r["id"]) and r1["target"]
        w, h = r["image_size"]
        img = XD.load_image(XF.ROOT, r["image"], r["image_size"])
        start = rle_to_mask(by[r["id"]]["candidates"][r1["start"]["index"] - 1]["rle"]).astype(bool)
        np.testing.assert_array_equal(rle_to_mask(r1["start_rle"]).astype(bool), start)
        m2, lk2, _ = G.correction_round(img, start, None, XP.proposals_abs(r1["target"], w, h), rule="grabcut", soft_lock=True,
                                        confine_box=r["proposal"]["box"])
        np.testing.assert_array_equal(rle_to_mask(r2["start_rle"]).astype(bool), m2)
        gt = rle_to_mask(r["gt_rle"]).astype(bool)
        assert r2["target"] == XD.correction_targets(m2, gt, lk2, r["proposal"]["box"], 2)[0]
        assert r2["applied"]["rule"] == "grabcut" and r2["applied"]["soft_lock"] is True
    assert n2 >= 5


# ---------------------------------------------------------------------------
# 输出、manifest、续跑
# ---------------------------------------------------------------------------


def test_build_outputs_and_manifest(built, w2c_dir):
    out, man = built
    rows = [json.loads(x) for p in sorted(w2c_dir.glob("*.jsonl")) for x in p.read_text().splitlines() if x.strip()]
    n_failed = sum(1 for r in rows if r["proposal"]["box"] is None)
    c = man["counts"]
    assert man["filters"]["n_input_rows"] == len(rows) == 39
    assert man["filters"]["spline_skipped"] == {"proposal_failed": n_failed}  # 假数据的框都贴着真值
    assert c["correct"]["round1"] == len(rows) - n_failed and c["spline"] == len(rows) - n_failed
    assert c["correct"]["round2"] + sum(man["filters"]["round2_not_made"].values()) == c["correct"]["round1"]
    assert set(man["filters"]["round2_not_made"]) <= {"not_drawn", "round1_empty"} and c["correct"]["round2"] > 0
    assert man["params"] == PARAMS and man["tools"]["effect_min_px"] == G.EFFECT_MIN_PX
    assert man["source"]["manifest_sha256"] is not None and [s["file"] for s in man["source"]["shards"]] == sorted(p.name for p in w2c_dir.glob("*.jsonl"))
    for f in man["files"]:
        p = out / "x2_data" / f["file"]
        assert XD._sha(p) == f["sha256"] and len(XD.read_jsonl(p)) == f["n"]
    sp = [r for f in man["files"] if f["kind"] == "spline" for r in XD.read_jsonl(out / "x2_data" / f["file"])]
    assert man["spline"]["fit_iou"]["mean"] == pytest.approx(np.mean([r["fit_iou"] for r in sp]))
    co = [r for f in man["files"] if f["kind"] == "correct" for r in XD.read_jsonl(out / "x2_data" / f["file"])]
    assert man["correct"]["target_empty"]["all"]["n_empty"] == sum(1 for r in co if not r["target"])
    assert all(len(r["target"]) <= 2 for r in co) and str(out) not in (out / "x2_data" / "manifest.json").read_text()
    assert set(man["counts"]["correct_by_tier"]["round1"]) == set(XD.TIERS)
    timing = XD.read_jsonl(out / "logs" / "timing.jsonl")
    assert len(timing) == len(rows) and all("wall_time_s" in t for t in timing)


def test_resume_segments_and_workers_bytewise_identical(built, w2c_dir, tmp_path):
    full, _ = built
    seg = tmp_path / "seg"
    assert XD.build(w2c_dir, seg, data_root=XF.ROOT, params=PARAMS, tiny=True, stop_after_shards=1, log=lambda m: None) is None
    # 中断：第二片写了一半（只有 .tmp），续跑时整片重算
    (seg / "x2_data" / (XD.out_name("spline", "shard_0001.jsonl") + ".tmp")).write_text("garbage\n")
    assert XD.build(w2c_dir, seg, data_root=XF.ROOT, params=PARAMS, tiny=True, resume=True, stop_after_shards=1, log=lambda m: None) is None
    XD.build(w2c_dir, seg, data_root=XF.ROOT, params=PARAMS, tiny=True, resume=True, log=lambda m: None)
    assert _files(seg) == _files(full)
    par = tmp_path / "par"
    XD.build(w2c_dir, par, data_root=XF.ROOT, params=PARAMS, tiny=True, workers=2, log=lambda m: None)
    assert _files(par) == _files(full)
    assert len(list((seg / "logs").glob("run_*.json"))) == 3
    # 拒绝混写：参数不同、没有 --resume、已写完的片被改
    p2 = json.loads(json.dumps(PARAMS))
    p2["correct"]["round2_ratio"] = 0.25
    with pytest.raises(XD.X2DataError, match="拒绝混写"):
        XD.build(w2c_dir, seg, data_root=XF.ROOT, params=p2, tiny=True, resume=True, log=lambda m: None)
    with pytest.raises(XD.X2DataError, match="--resume"):
        XD.build(w2c_dir, seg, data_root=XF.ROOT, params=PARAMS, tiny=True, log=lambda m: None)
    f = seg / "x2_data" / XD.out_name("correct", "shard_0000.jsonl")
    lines = f.read_text().splitlines(keepends=True)
    f.write_text("".join(lines[1:]))
    f.write_text("".join(reversed(lines)))  # 已写完的片被改乱：id 的顺序与输入对不上
    with pytest.raises(XD.X2DataError, match="对不上"):
        XD.build(w2c_dir, seg, data_root=XF.ROOT, params=PARAMS, tiny=True, resume=True, log=lambda m: None)


def test_ratio_changes_round2_count(w2c_dir, tmp_path, built):
    _, man = built
    p0 = json.loads(json.dumps(PARAMS))
    p0["correct"]["round2_ratio"] = 0.0
    m0 = XD.build(w2c_dir, tmp_path / "r0", data_root=XF.ROOT, params=p0, tiny=True, log=lambda m: None)
    p1 = json.loads(json.dumps(PARAMS))
    p1["correct"]["round2_ratio"] = 1.0
    m1 = XD.build(w2c_dir, tmp_path / "r1", data_root=XF.ROOT, params=p1, tiny=True, log=lambda m: None)
    assert m0["counts"]["correct"]["round2"] == 0 < man["counts"]["correct"]["round2"] < m1["counts"]["correct"]["round2"]
    assert m1["counts"]["correct"]["round2"] == m1["counts"]["correct"]["round1"] - m1["filters"]["round2_not_made"].get("round1_empty", 0)
    assert m0["counts"]["spline"] == m1["counts"]["spline"] == man["counts"]["spline"]


def test_input_checks(tmp_path):
    rows = XF.fake_rows(W2, n=5)
    rows[3]["split"] = "val"
    d = tmp_path / "w"
    XF.FK.write_rows(d, rows)
    with pytest.raises(XD.X2DataError, match="只读训练集"):
        XD.build(d, tmp_path / "o", data_root=XF.ROOT, params=PARAMS, tiny=True, log=lambda m: None)
    nm = tmp_path / "nm"
    XF.FK.write_rows(nm, XF.fake_rows(W2, n=3), manifest=False)
    with pytest.raises(XD.X2DataError, match="manifest"):
        XD.build(nm, tmp_path / "o2", data_root=XF.ROOT, params=PARAMS, tiny=False, log=lambda m: None)


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


def test_load_checks(built, w2c_dir, tmp_path):
    out, man = built
    src = XD.source_identity(w2c_dir, require_manifest=True)
    data = XD.load_x2_data(out, params=PARAMS, source=src, data_root=XF.ROOT)
    assert {k: len(v) for k, v in data.pools.items()} == {"spline": man["counts"]["spline"], "correct": man["counts"]["correct"]["total"]}
    assert all(data.rows[i]["kind"] == "spline" for i in data.pool("spline"))
    ident = data.identity()
    assert ident["params"] == PARAMS and ident["pools"]["spline"]["n"] == man["counts"]["spline"] and str(out) not in json.dumps(ident)
    p2 = json.loads(json.dumps(PARAMS))
    p2["spline"]["n"] = 12
    with pytest.raises(XD.X2DataError, match="参数"):
        XD.load_x2_data(out, params=p2, source=src)
    other = dict(src, manifest_sha256="0" * 64)
    with pytest.raises(XD.X2DataError, match="另一份 w2c_data"):
        XD.load_x2_data(out, params=PARAMS, source=other)
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    f = bad / "x2_data" / XD.out_name("spline", "shard_0000.jsonl")
    rows = XD.read_jsonl(f)
    rows[0]["target_text"] = rows[0]["target_text"].replace(", ", ",", 1)
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    with pytest.raises(XD.X2DataError, match="sha256"):
        XD.load_x2_data(bad, params=PARAMS, source=None)
    m = json.loads((bad / "x2_data" / "manifest.json").read_text())
    for e in m["files"]:
        if e["file"] == f.name:
            e["sha256"] = XD._sha(f)
    (bad / "x2_data" / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
    with pytest.raises(XD.X2DataError, match="不合格"):
        XD.load_x2_data(bad, params=PARAMS, source=None)


def test_cli_tiny_and_usage(w2c_dir, tmp_path, built):
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    out = tmp_path / "cli"
    p = subprocess.run([sys.executable, str(BUILD), "--tiny", "--w2c-data", str(w2c_dir), "--out", str(out), "--workers", "2"],
                       capture_output=True, text=True, cwd=REPO, env=env)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "样条 37 行" in p.stdout
    assert _files(out) == _files(built[0])
    p = subprocess.run([sys.executable, str(BUILD), "--w2c-data", str(w2c_dir), "--out", str(tmp_path / "x"), "--data-root", "/x"],
                       capture_output=True, text=True, cwd=REPO, env=env)
    assert p.returncode == 2 and "运行编号" in p.stderr
    p = subprocess.run([sys.executable, str(BUILD), "--tiny", "--w2c-data", str(w2c_dir), "--out", str(out)],
                       capture_output=True, text=True, cwd=REPO, env=env)
    assert p.returncode == 2 and "--resume" in p.stderr
