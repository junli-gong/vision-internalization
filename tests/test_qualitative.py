"""DEV-0076：scripts/figures/make_qualitative.py（定性图，预注册 U-5）。

- 常量与 PREREGISTRATION.md U-5 的原文一致；设置名、trace 的目录名与 X-2 的运行器一致；
- 抽样：同一输入两次结果相同；池 = default_rng(20260929 + k).choice(…, 40, replace=False)；随机例 = 池的前 3 句；失败例 = 池里 IoU
  最低的 2 句、同分按 id 字典序（合成的 predictions 构造同分、字典序与自然顺序不同、与随机例重合）；缺句、句数不够时报错；
- 仓库里的 RUN-0092 / RUN-0072 输出上 --select-only 跑通（25 条、正文 10 条，两次相同，--expect-selection 核对）；
- fixture 上的假 X-2 运行（含样条解析失败与提出失败）：全流程出图（PDF、300 dpi 的 PNG、caption、figures.json）；trace 与 manifest
  不符时不读、不出图；--collect-only 之后 --render-from 出同样的图；--out 不带运行编号时拒绝。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MQ = _load("_make_qualitative_under_test", "scripts/figures/make_qualitative.py")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------


def test_constants_match_prereg_u5():
    text = (REPO / "PREREGISTRATION.md").read_text(encoding="utf-8")
    u5 = text[text.index("### U-5."): text.index("### U-6.")]
    assert f"numpy.random.default_rng({MQ.POOL_SEED_BASE} + k)" in u5
    assert f"抽 {MQ.POOL_N} 句" in u5 and "不放回" in u5
    assert f"抽中顺序的前 {MQ.N_RANDOM} 句" in u5
    assert f"X-2 逐句 IoU 最低的 {MQ.N_FAILURE} 句；同分按 id 字典序" in u5
    assert (MQ.POOL_SEED_BASE, MQ.POOL_N, MQ.N_RANDOM, MQ.N_FAILURE) == (20260929, 40, 3, 2)
    assert "x2_ellipse_trimap" in u5 and (MQ.MAIN_SETTING, MQ.ELLIPSE_SETTING) == ("x2_spline_trimap", "x2_ellipse_trimap")


def test_names_match_x2_runner():
    from src.explore import x2_runner as XR

    assert MQ.TRACE_SETTING == XR.TRACE_SETTING and MQ.TRACE == XR.TRACE
    defs = XR.setting_defs(1)
    assert defs[MQ.MAIN_SETTING]["kind"] == "spline" and defs[MQ.MAIN_SETTING]["mode"] == "trimap"
    assert defs[MQ.ELLIPSE_SETTING]["kind"] == "ellipse" and defs[MQ.ELLIPSE_SETTING]["mode"] == "trimap"


# ---------------------------------------------------------------------------
# 抽样（合成数据）
# ---------------------------------------------------------------------------

SPLITS = [("test_refcoco_unc_testA", "refcoco_unc/testA"), ("test_refcoco_unc_testB", "refcoco_unc/testB"),
          ("test_refcoco+_unc_testA", "refcoco+_unc/testA"), ("test_refcoco+_unc_testB", "refcoco+_unc/testB"),
          ("test_refcocog_umd_test", "refcocog_umd/test")]


def _subsets(n: int = 300) -> dict:
    from src.data.subsets import population_digest, sort_ids

    out = {}
    for j, (split, _) in enumerate(SPLITS):
        ids = sort_ids(str(5 + 7 * i + j) for i in range(n))  # 位数不一（1–4 位），字典序与自然顺序不同
        out[split] = {"dataset": split.split("_")[1], "n": n, "sha256": population_digest(ids), "ids": ids}
    return {"splits": out}


def _rec(prefix: str, sid: str, split: str, iou: float) -> dict:
    inter = int(round(iou * 1000))
    return {"id": f"{prefix}/{sid}", "w2_split": split, "instruction": f"expr {sid}", "failure": None, "fallback": None,
            "score": {"inter": inter, "union": 1000, "iou": inter / 1000}}


def _preds(subsets: dict, iou_of) -> dict:
    out = {}
    for st in (MQ.MAIN_SETTING, MQ.ELLIPSE_SETTING):
        for split, prefix in SPLITS:
            out.setdefault(st, {})[split] = [_rec(prefix, s, split, iou_of(st, split, s)) for s in subsets["splits"][split]["ids"]]
    return out


def _default_iou(st, split, s):
    return (int(s) * 37 % 1000) / 1000 + (0.0 if st == MQ.MAIN_SETTING else 0.0005)


def test_draw_pool_is_numpy_choice():
    ids = [str(i) for i in range(300)]
    for k in range(5):
        pool = MQ.draw_pool(ids, k)
        idx = np.random.default_rng(20260929 + k).choice(300, size=40, replace=False)
        assert pool == [ids[i] for i in idx]  # 对 id 列表与对下标 choice 是同一组
        assert len(set(pool)) == 40
    assert MQ.draw_pool(ids, 0) != MQ.draw_pool(ids, 1)
    with pytest.raises(MQ.QualError):
        MQ.draw_pool(ids[:39], 0)


def test_selection_rules_and_determinism():
    sub = _subsets()
    a = MQ.select_examples(sub, _preds(sub, _default_iou))
    b = MQ.select_examples(sub, _preds(sub, _default_iou))
    assert a == b
    assert len(a["entries"]) == 25 and len(a["main"]) == 10
    for k, (split, prefix) in enumerate(SPLITS):
        s = a["splits"][k]
        ids = sub["splits"][split]["ids"]
        pool = [ids[i] for i in np.random.default_rng(20260929 + k).choice(len(ids), size=40, replace=False)]
        assert [p["sent_id"] for p in s["pool"]] == pool and s["k"] == k
        assert s["random"] == [f"{prefix}/{x}" for x in pool[:3]]
        lowest = sorted(pool, key=lambda x: (_default_iou(MQ.MAIN_SETTING, split, x), f"{prefix}/{x}"))[:2]
        assert s["failure"] == [f"{prefix}/{x}" for x in lowest]
        es = [e for e in a["entries"] if e["split"] == split]
        assert [(e["type"], e["rank"]) for e in es] == [("random", 1), ("random", 2), ("random", 3), ("failure", 1), ("failure", 2)]
        assert [e["key"] for e in es if e["main"]] == [f"k{k}_random1", f"k{k}_failure1"]
        assert {e["iou"][MQ.ELLIPSE_SETTING]["iou"] for e in es} <= {r["score"]["iou"] for r in _preds(sub, _default_iou)[MQ.ELLIPSE_SETTING][split]}


def test_failure_ties_broken_by_id_lexicographic():
    sub = _subsets()
    pools = {split: MQ.draw_pool(sub["splits"][split]["ids"], k) for k, (split, _) in enumerate(SPLITS)}
    # 每个划分：池里取 a、b 两句，a 的位数少、首位大（自然顺序 a 在前，字典序 b 在前），再加第三句 c 排在两者之后；三句 IoU 都是 0
    zeros = {}
    for split, pool in pools.items():
        pairs = [(a, b) for a in pool for b in pool if len(a) < len(b) and a[0] > b[0]]
        assert pairs, "池里没有字典序与自然顺序相反的一对"
        a, b = pairs[0]
        c = max((x for x in pool if x not in (a, b)), key=lambda x: (x[0], len(x), x))  # 首位 ≥ a 的首位 > b 的首位
        zeros[split] = (a, b, c)

    def iou(st, split, s):
        return 0.0 if (st == MQ.MAIN_SETTING and s in zeros[split]) else 0.5 + int(s) % 97 / 1000

    sel = MQ.select_examples(sub, _preds(sub, iou))
    for k, (split, prefix) in enumerate(SPLITS):
        a, b, c = zeros[split]
        want = sorted(f"{prefix}/{x}" for x in (a, b, c))[:2]  # 三句同为 0：按样本 id 的字典序取前两句
        assert sel["splits"][k]["failure"] == want
        assert want[0] == f"{prefix}/{b}"  # 不是自然顺序（自然顺序会先取 a）


def test_failure_may_overlap_random():
    sub = _subsets()
    first = {split: MQ.draw_pool(sub["splits"][split]["ids"], k)[0] for k, (split, _) in enumerate(SPLITS)}

    def iou(st, split, s):
        return 0.01 if s == first[split] else 0.5 + int(s) % 89 / 1000

    sel = MQ.select_examples(sub, _preds(sub, iou))
    for k, (split, prefix) in enumerate(SPLITS):
        e = {x["key"]: x for x in sel["entries"] if x["split"] == split}
        assert e[f"k{k}_random1"]["id"] == e[f"k{k}_failure1"]["id"] == f"{prefix}/{first[split]}"
        assert (e[f"k{k}_random1"]["also"], e[f"k{k}_failure1"]["also"]) == ("failure1", "random1")
    assert len(sel["entries"]) == 25  # 规则照写：重合也照样入选


def test_selection_errors():
    sub = _subsets()
    p = _preds(sub, _default_iou)
    p[MQ.ELLIPSE_SETTING]["test_refcoco_unc_testB"] = p[MQ.ELLIPSE_SETTING]["test_refcoco_unc_testB"][1:]
    with pytest.raises(MQ.QualError):
        MQ.select_examples(sub, p)
    small = _subsets(30)
    with pytest.raises(MQ.QualError):
        MQ.select_examples(small, _preds(small, _default_iou))
    assert len(MQ.select_examples(small, _preds(small, _default_iou), pool_n=None)["splits"][0]["pool"]) == 30
    bad = _subsets()
    bad["splits"]["test_refcoco_unc_testA"]["sha256"] = "0" * 64
    with pytest.raises(MQ.QualError):
        MQ.select_examples(bad, _preds(bad, _default_iou))


def test_check_expected():
    sub = _subsets()
    a = MQ.select_examples(sub, _preds(sub, _default_iou))
    assert MQ.check_expected(a, json.loads(json.dumps(a))) == []
    b = json.loads(json.dumps(a))
    b["entries"][3]["id"] = "x"
    assert MQ.check_expected(a, b) == ["entries 不同"]


# ---------------------------------------------------------------------------
# 仓库里的真实输出：--select-only
# ---------------------------------------------------------------------------


def _have_repo_preds() -> bool:
    return all((MQ.leaf(REPO / run, st, split) / MQ.PREDICTIONS).is_file()
               for run, st in ((MQ.XT.CANDS["C1 (X-2)"], MQ.MAIN_SETTING), (MQ.XT.CANDS["C1 (X-2)"], MQ.ELLIPSE_SETTING),
                               (MQ.XT.REF, MQ.W2C_SETTING))
               for split, _ in SPLITS)


@pytest.mark.skipif(not _have_repo_preds(), reason="仓库里没有 RUN-0092 / RUN-0072 的 predictions.jsonl")
def test_select_only_on_repo_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    o1, o2 = tmp_path / "0001_sel", tmp_path / "0002_sel"
    assert MQ.main(["--select-only", "--out", str(o1)]) == 0
    assert MQ.main(["--select-only", "--out", str(o2), "--expect-selection", str(o1 / MQ.SELECTION)]) == 0
    a, b = (json.loads((o / MQ.SELECTION).read_text()) for o in (o1, o2))
    assert a["entries"] == b["entries"] and a["splits"] == b["splits"]
    assert len(a["entries"]) == 25 and len(a["main"]) == 10
    assert [s["split"] for s in a["splits"]] == [s for s, _ in SPLITS] and all(s["pool_n"] == 40 for s in a["splits"])
    assert all(set(e["iou"]) == {MQ.MAIN_SETTING, MQ.ELLIPSE_SETTING, MQ.W2C_SETTING} for e in a["entries"])
    assert a["inputs"]["subsets"]["sha256"] == hashlib.sha256((REPO / "configs/w2_llmseg_subsets.json").read_bytes()).hexdigest()
    # 抽样规则逐条重算
    sub = json.loads((REPO / "configs/w2_llmseg_subsets.json").read_text())
    for k, s in enumerate(a["splits"]):
        ids = sub["splits"][s["split"]]["ids"]
        assert [p["sent_id"] for p in s["pool"]] == [ids[i] for i in np.random.default_rng(20260929 + k).choice(300, 40, replace=False)]
        assert s["failure"] == [p["id"] for p in sorted(s["pool"], key=lambda p: (p["x2_iou"], p["id"]))[:2]]
    exp = json.loads((o1 / MQ.SELECTION).read_text())
    exp["entries"][0]["rank"] = 9
    (tmp_path / "bad.json").write_text(json.dumps(exp))
    o3 = tmp_path / "0003_sel"
    assert MQ.main(["--select-only", "--out", str(o3), "--expect-selection", str(tmp_path / "bad.json")]) == 1
    assert not o3.exists()


# ---------------------------------------------------------------------------
# fixture 上的假 X-2 运行：全流程
# ---------------------------------------------------------------------------


def _fake_x2_run(root: Path) -> Path:
    """tests/fixtures/w2 的 tiny 划分上写一个假的 X-2 运行：两个设置的 predictions.jsonl、trace.jsonl 与 LARGE_FILES.manifest.tsv。
    refcoco_unc/testB/109 的样条解析失败（X-2 交 1 号候选，这里用框）；refcoco+_unc/testB/307 提出失败（没有 mask）。"""
    from src.e1.scoring import gt_mask
    from src.executors.raster import mask_to_rle
    from src.explore import x2_prompts as XP
    from src.explore.x2_tools import ellipse_ctrl_q
    from src.metrics.seg import mask_inter_union, sample_iou
    from src.w2.config import load_w2_config
    from src.w2.data import tiny_paths, tiny_splits, w2_items

    doc = tiny_splits(load_w2_config())
    run = root / "0092_fake_x2"
    man = ["path\tbytes\tsha256\tmtime_epoch"]
    for split, _ in SPLITS:
        preds = {MQ.MAIN_SETTING: [], MQ.ELLIPSE_SETTING: []}
        trace = []
        for s in w2_items("refseg", split, tiny_paths(), splits=doc):
            gt = gt_mask(s)
            h, w = gt.shape
            x0, y0, x1, y1 = s.gt["box"]
            box = [x0 + 1.0, y0 + 0.5, min(x1 + 2.0, w), min(y1 + 1.0, h)]
            ell = ellipse_ctrl_q(box, w, h, 16)
            ell_r = XP.spline_mask(ell, box, (h, w))
            sp_q = [[min(1000, q[0] + (7 if i % 2 else -5)), q[1]] for i, q in enumerate(ell)]
            fail_sp, fail_prop = s.id == "refcoco_unc/testB/109", s.id == "refcoco+_unc/testB/307"
            sp_r = None if fail_sp else XP.spline_mask(sp_q, box, (h, w))
            row = {"id": s.id, "box": None if fail_prop else box, "box_q": None, "failure": "no_box" if fail_prop else None}
            if not fail_prop:
                row["spline"] = {"call": {"failure": "malformed_json" if fail_sp else None}, "ctrl_q": None if fail_sp else sp_q}
                if not fail_sp:
                    row["spline"].update(raster_rle=mask_to_rle(sp_r), trimap_rle=mask_to_rle(sp_r), trimap_ok=True)
                row["ellipse"] = {"ctrl_q": ell, "raster_rle": mask_to_rle(ell_r), "trimap_rle": mask_to_rle(ell_r), "trimap_ok": True}
            trace.append(row)
            box_m = np.zeros((h, w), bool)
            box_m[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = True
            for st, m in ((MQ.MAIN_SETTING, None if fail_prop else (box_m if fail_sp else sp_r)),
                          (MQ.ELLIPSE_SETTING, None if fail_prop else ell_r)):
                sc = mask_inter_union(m, gt)
                preds[st].append({"id": s.id, "w2_split": split, "instruction": s.fields["expression"], "image_size": [w, h],
                                  "failure": "no_box" if fail_prop else None, "fallback": "parse_failure" if (fail_sp and st == MQ.MAIN_SETTING) else None,
                                  "answer": None if m is None else {"kind": "mask", "mask": {"rle": mask_to_rle(m), "area": int(m.sum())}},
                                  "score": {"inter": int(sc.inter), "union": int(sc.union), "iou": sample_iou(*sc)}})
        for st, rows in preds.items():
            p = MQ.leaf(run, st, split) / MQ.PREDICTIONS
            p.parent.mkdir(parents=True)
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        rel = f"{MQ.MODEL}/{MQ.TASK}/{MQ.TRACE_SETTING}/{split}/{MQ.TRACE}"
        tp = run / rel
        tp.parent.mkdir(parents=True)
        tp.write_text("".join(json.dumps(r) + "\n" for r in trace))
        man.append(f"{rel}\t{tp.stat().st_size}\t{hashlib.sha256(tp.read_bytes()).hexdigest()}\t0")
    (run / MQ.MANIFEST).write_text("\n".join(man) + "\n")
    return run


FIGS = ["fig_qual_main"] + [f"fig_qual_appendix_{k}_{s.removeprefix('test_').replace('+', 'plus')}" for k, (s, _) in enumerate(SPLITS)]


def _check_figures(out: Path):
    from PIL import Image

    for name in FIGS:
        pdf, png, cap = out / f"{name}.pdf", out / f"{name}.png", out / f"{name}_caption.tex"
        assert pdf.stat().st_size > 0 and png.stat().st_size > 0, name
        assert b"/Type3" not in pdf.read_bytes() and b"/Type /Font" in pdf.read_bytes()  # TrueType，无 Type 3
        with Image.open(png) as im:
            assert round(im.info["dpi"][0]) == 300
        t = cap.read_text()
        assert t.startswith("% Generated by scripts/figures/make_qualitative.py") and "\\caption{" in t
        assert "20260929" in t and "RUN-0092" in t


def test_full_run_on_fixture(tmp_path):
    pytest.importorskip("matplotlib")
    run = _fake_x2_run(tmp_path)
    out = tmp_path / "0099_qual"
    assert MQ.main(["--tiny", "--x2-run", str(run), "--w2c-run", "none", "--out", str(out)]) == 0
    _check_figures(out)
    sel = json.loads((out / MQ.SELECTION).read_text())
    figs = json.loads((out / MQ.FIGURES_JSON).read_text())
    assert sel["tiny"] is True and [s["pool_n"] for s in sel["splits"]] == [2, 2, 2, 2, 4]
    assert [len(f["keys"]) for n, f in figs["figures"].items() if n != "fig_qual_main"] == [4, 4, 4, 4, 5]
    assert len(figs["figures"]["fig_qual_main"]["keys"]) == 10
    by = {e["id"]: e for e in figs["entries"]}
    assert by["refcoco_unc/testB/109"]["spline"]["failure"] == "malformed_json" and by["refcoco_unc/testB/109"]["iou"]["spline_raster"] is None
    assert by["refcoco+_unc/testB/307"]["spline"]["proposal_failure"] == "no_box"
    assert by["refcoco+_unc/testB/307"]["iou"][MQ.MAIN_SETTING] == 0.0
    ok = by["refcoco_unc/testA/106"]
    assert 0 < ok["iou"]["spline_raster"] <= 1 and len(ok["image"]["sha256"]) == 64
    assert "gt_rle" not in ok and "rle" not in json.dumps(figs)  # figures.json 不含 mask
    assert set(figs["traces"]) == {s for s, _ in SPLITS} and all(v["manifest"] == "same" for v in figs["traces"].values())
    # 同一输入再跑一次：抽样与图的数字相同
    out2 = tmp_path / "0100_qual"
    assert MQ.main(["--tiny", "--x2-run", str(run), "--w2c-run", "none", "--out", str(out2),
                    "--expect-selection", str(out / MQ.SELECTION)]) == 0
    assert json.loads((out2 / MQ.FIGURES_JSON).read_text())["entries"] == figs["entries"]


def test_trace_manifest_mismatch_refused(tmp_path):
    run = _fake_x2_run(tmp_path)
    man = (run / MQ.MANIFEST).read_text().splitlines()
    parts = man[2].split("\t")
    parts[2] = "f" * 64
    man[2] = "\t".join(parts)
    (run / MQ.MANIFEST).write_text("\n".join(man) + "\n")
    out = tmp_path / "0101_qual"
    assert MQ.main(["--tiny", "--x2-run", str(run), "--w2c-run", "none", "--out", str(out), "--collect-only"]) == 2
    assert not list(out.glob("*.pdf")) and not (out / MQ.BUNDLE).exists()


def test_prediction_data_mismatch_refused(tmp_path):
    run = _fake_x2_run(tmp_path)
    p = MQ.leaf(run, MQ.ELLIPSE_SETTING, "test_refcoco_unc_testA") / MQ.PREDICTIONS
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]["score"]["inter"] += 1  # 记的交并与解码的 mask 对不上
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert MQ.main(["--tiny", "--x2-run", str(run), "--w2c-run", "none", "--out", str(tmp_path / "0102_qual"), "--collect-only"]) == 2


def test_collect_then_render(tmp_path):
    pytest.importorskip("matplotlib")
    run = _fake_x2_run(tmp_path)
    out = tmp_path / "0103_qual"
    assert MQ.main(["--tiny", "--x2-run", str(run), "--w2c-run", "none", "--out", str(out), "--collect-only"]) == 0
    assert not list(out.glob("*.pdf"))
    b = out / MQ.BUNDLE
    rows = json.loads((b / "entries.json").read_text())
    assert len(rows) == 21 and "path" not in rows[0]["image"]  # bundle 里不记 HPC 的路径
    assert sorted(p.name for p in (b / "images").iterdir()) == sorted({r["image"]["file"] for r in rows})
    out2 = tmp_path / "0104_qual"
    assert MQ.main(["--render-from", str(b), "--out", str(out2)]) == 0
    _check_figures(out2)
    full = tmp_path / "0105_qual"
    assert MQ.main(["--tiny", "--x2-run", str(run), "--w2c-run", "none", "--out", str(full)]) == 0
    assert json.loads((out2 / MQ.FIGURES_JSON).read_text())["entries"] == json.loads((full / MQ.FIGURES_JSON).read_text())["entries"]


def test_out_needs_run_number(tmp_path):
    for bad in ("qual", "76_qual", "qual_0076"):
        with pytest.raises(SystemExit) as e:
            MQ.parse_args(["--select-only", "--out", str(tmp_path / bad)])
        assert e.value.code == 2
    assert MQ.parse_args(["--select-only", "--out", str(tmp_path / "0076_qual")]).select_only
    assert re.fullmatch(MQ.RUN_DIR_NAME, "0076_qual")
