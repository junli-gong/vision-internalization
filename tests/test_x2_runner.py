"""DEV-0069：X-2 的推理运行器（src/explore/x2_runner.py、src/explore/x2_tools.py、scripts/run_x2.py、configs/x2_run.json）。

- 配置与设置表：20 个设置（主管线、诊断、两种消融），多轮设置的计算量配对；配置核对；
- 工具：椭圆消融的控制点、随机框（同数量、同大小、同符号、落在提出的框里，确定）；
- 一组样本的轨迹（假适配器，按提示给确定的回答）：描样条、两次挑选、两条修正轨迹（施加、提前停：[] 与解析失败）、随机框轨迹、椭圆；
  调用都关思考、单张图、生成上限按 x2_prompts；进程池与单进程逐位相同；
- 各设置交的 mask 与记账（调用次数、token）；样条解析失败交 1 号；提出失败没有 mask；
- 命令行 tiny 端到端（真的小模型 + X-2 训练的检查点；随机模型给不出框，同 tests/test_w2c_eval 换成合法的框后重跑）：各设置的输出
  是 W2 格式、scripts/gate_explore.py 能读；续跑逐字节相同；拒绝 test；devcheck3 的守卫；不是 X-2 的检查点拒绝；
- 运行器的进程不 import 学习型的视觉模块（LLM 本身的 torch / transformers 除外）。
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.e1.prompts import PROMPTS_PATH, load_prompt_config
from src.e1.scoring import gt_mask
from src.executors.raster import box_to_mask, rle_to_mask
from src.explore import guidance as G
from src.explore import oracle_sim as OS
from src.explore import x2_prompts as XP
from src.explore import x2_render as XR_
from src.explore import x2_runner as XR
from src.explore import x2_tools as XT
from src.w2 import runner as R
from src.w2.config import load_w2_config
from src.w2.data import tiny_paths, tiny_splits, w2_items
from src.w2.loop import candidates_row
from tests import _w2_fakes as WF
from tests import _x2_fakes as XF

REPO = Path(__file__).resolve().parents[1]
RUN_X2 = REPO / "scripts" / "run_x2.py"
TRAIN = REPO / "scripts" / "train_w2c.py"
BUILD = REPO / "scripts" / "build_x2_data.py"
BUILD_RP = REPO / "scripts" / "build_w2c_replay.py"
QWEN = "qwen3.5-0.8b"
W2 = load_w2_config()
RC = XR.load_run_config()
PROMPTS = load_prompt_config(PROMPTS_PATH)
FAKE_IMPL = R.load_impl("tests._w2_fakes", "tests._w2_fakes")
FAKES = ["--candidates-module", "tests._w2_fakes", "--render-module", "tests._w2_fakes"]


def run(script: Path, args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------
# 配置与设置
# ---------------------------------------------------------------------------


def test_run_config_and_settings_table(tmp_path):
    assert RC.rounds == 3 and RC.batching(tiny=False) == {"batch_size": 16, "chunk_size": 64, "max_batch_tokens": 48000}
    assert RC.batching(tiny=True)["chunk_size"] == 2 and RC.doc["correction"]["rule"] == "grabcut" and RC.doc["correction"]["soft_lock"]
    d = XR.setting_defs(3)
    assert list(d) == ["x2_select", "x2_select_cor1", "x2_select_cor2", "x2_select_cor3", "x2_spline_raster", "x2_spline_trimap",
                       "x2_select5", "x2_select5_cor1", "x2_select5_cor2", "x2_select5_cor3", "x2_oracle4", "x2_oracle5",
                       "x2_select_cor1_randbox", "x2_select_cor2_randbox", "x2_select_cor3_randbox", "x2_select5_cor1_randbox",
                       "x2_select5_cor2_randbox", "x2_select5_cor3_randbox", "x2_ellipse_raster", "x2_ellipse_trimap"]
    assert [k for k, v in d.items() if v["diagnostic"]] == ["x2_oracle4", "x2_oracle5"]
    assert {v["ablation"] for v in d.values()} == {None, "randbox", "ellipse"}
    assert all(v["multi_round"] == (v["kind"] in ("cor", "randbox")) for v in d.values())
    assert d["x2_select5"]["note"].startswith("挑选没在 K = 5 上训练过")
    for edit, msg in [(lambda c: c.update(rounds=0), "rounds"), (lambda c: c["correction"].update(rule="magic"), "correction"),
                      (lambda c: c["correction"].update(early_stop="never"), "early_stop"), (lambda c: c["select5"].update(fifth="x"), "select5"),
                      (lambda c: c["batching"].update(chunk_size=0), "batching")]:
        c = copy.deepcopy(dict(RC.doc))
        edit(c)
        p = tmp_path / "r.json"
        p.write_text(json.dumps(c, ensure_ascii=False))
        with pytest.raises(XR.X2RunError, match=msg):
            XR.load_run_config(p)


def test_checkpoint_must_be_x2(tmp_path):
    leaf = tmp_path / "w2c" / QWEN / "seed0"
    ck = leaf / "ckpt" / "final" / "model"
    ck.mkdir(parents=True)
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "w2c"}))
    with pytest.raises(XR.X2RunError, match="不是 X-2"):
        XR.checkpoint_x2_params(ck)
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "w2c", "x2": {"params": {"spline": {"n": 12}, "correct": {"m": 3}},
                                                                                "prompts": {"version": "old"}, "render": {"version": XR_.RENDER_VERSION}}}))
    with pytest.raises(XR.X2RunError, match="版本"):
        XR.checkpoint_x2_params(ck)
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "w2c", "x2": {"params": {"spline": {"n": 12}, "correct": {"m": 3}},
                                                                                "prompts": {"version": XP.PROMPT_VERSION},
                                                                                "render": {"version": XR_.RENDER_VERSION}}}))
    assert XR.checkpoint_x2_params(ck)["n"] == 12 and XR.checkpoint_x2_params(ck)["m"] == 3
    with pytest.raises(XR.X2RunError, match="ckpt/final/model"):
        XR.checkpoint_x2_params(tmp_path)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def test_ellipse_and_random_boxes():
    box = [20.0, 10.0, 100.0, 70.0]
    q = XT.ellipse_ctrl_q(box, 128, 96, 16)
    assert len(q) == 16 and all(isinstance(v, int) and 0 <= v <= 1000 for p in q for v in p)
    yy, xx = np.mgrid[0:96, 0:128] + 0.5
    ell = ((xx - 60) / 40) ** 2 + ((yy - 40) / 30) ** 2 <= 1
    m = XP.spline_mask(q, box, (96, 128))
    assert (m & ell).sum() / (m | ell).sum() > 0.95 and not (m & ~box_to_mask(box, 96, 128)).any()
    items = [{"sign": "+", "box": [100, 100, 300, 150]}, {"sign": "-", "box": [0, 0, 900, 900]}]
    bq = [50, 60, 500, 400]
    a = XT.random_items(items, bq, OS.rng_for("s", "id", "select", 1))
    assert a == XT.random_items(items, bq, OS.rng_for("s", "id", "select", 1))
    assert a != XT.random_items(items, bq, OS.rng_for("s", "id", "select", 2))
    assert [x["sign"] for x in a] == ["+", "-"]
    assert a[0]["box"][2] - a[0]["box"][0] == 200 and a[0]["box"][3] - a[0]["box"][1] == 50  # 同大小
    assert a[1]["box"] == [50, 60, 500, 400]  # 比提出的框大：贴左上、截到提出的框
    for k in range(50):
        r = XT.random_items(items[:1], bq, OS.rng_for("s", k))[0]["box"]
        assert bq[0] <= r[0] and r[2] <= bq[2] and bq[1] <= r[1] and r[3] <= bq[3]


# ---------------------------------------------------------------------------
# 一组样本的轨迹（假适配器）
# ---------------------------------------------------------------------------


def _samples():
    base = w2_items("refseg", "devcheck2", tiny_paths(), splits=tiny_splits(W2))
    words = ["plus", "empty", "junk"]
    return [dataclasses.replace(s, fields={**s.fields, "expression": f"the {w} thing {i}"}) for i, (s, w) in enumerate(zip(base, words))]


def _cand_rows(samples, *, fail_first: bool = False):
    rows = {}
    for i, s in enumerate(samples):
        gt = gt_mask(s)
        h, w = gt.shape
        x0, y0, x1, y1 = (float(v) for v in s.gt["box"])
        box = [x0 + 1, y0 + 1, x1 - 1, y1 - 1]
        if fail_first and i == 0:
            rows[s.id] = candidates_row(s.id, (w, h), {"mode": "box", "box": None, "failure": "no_box", "fg_points": [], "bg_points": []}, None, 4)
            continue
        ms = [box_to_mask(box, h, w), gt.copy(), box_to_mask([x0 + 4, y0 + 4, x1 - 4, y1 - 4], h, w), np.zeros((h, w), bool)]
        cands = [WF.Candidate(name=f"c{k}", mask=m, ok=True, info={}) for k, m in enumerate(ms, start=1)]
        rows[s.id] = candidates_row(s.id, (w, h), {"mode": "box", "box": box, "failure": None, "fg_points": [], "bg_points": []}, cands, 4)
    return rows


def _ctx(ad, pool, *, n=16, m=2):
    bat = RC.batching(tiny=True)
    return XR.ChunkContext(adapter=ad, w2=W2, rc=RC, impl=FAKE_IMPL, prompts=PROMPTS, n=n, m=m, batch_size=bat["batch_size"],
                           max_batch_tokens=bat["max_batch_tokens"], tiny=True, pool=pool)


@pytest.fixture(scope="module")
def traced():
    samples = _samples()
    rows = _cand_rows(samples)
    ad = XF.ScriptedX2Adapter(select="2")
    with XT.ToolPool(1) as pool:
        recs, tms = XR.process_chunk(_ctx(ad, pool), samples, rows)
    return samples, rows, recs, tms, ad


def test_trace_with_scripted_adapter(traced):
    samples, rows, recs, tms, ad = traced
    by = {r["id"]: r for r in recs}
    assert [r["id"] for r in recs] == [s.id for s in samples]
    for s in samples:
        t = by[s.id]
        w, h = rows[s.id]["image_size"]
        assert t["box_q"] == XP.box_q(rows[s.id]["proposal"]["box"], w, h)
        assert t["select4"]["answer"] == 2 and t["select5"]["answer"] == 2 and t["select5"]["fifth"] == XR.FIFTH_NAME
        sp = t["spline"]
        assert sp["call"]["failure"] is None and len(sp["ctrl_q"]) == 16 and sp["raster_rle"] and sp["trimap_rle"]
        np.testing.assert_array_equal(rle_to_mask(sp["raster_rle"]).astype(bool), XP.spline_mask(sp["ctrl_q"], t["box"], (h, w)))
        assert t["ellipse"]["ctrl_q"] == XT.ellipse_ctrl_q(t["box"], w, h, 16)
    plus, empty, junk = (by[s.id] for s in samples)
    for base in XR.BASES:
        assert [r["stop"] for r in empty["cor"][base]] == ["empty"] and empty["randbox"][base] == []
        assert [r["stop"] for r in junk["cor"][base]] == ["parse_failure"] and junk["cor"][base][0]["call"]["failure"] == "malformed_json"
        pr = plus["cor"][base]
        assert 1 <= len(pr) <= 3 and all(r["round"] == k + 1 for k, r in enumerate(pr))
        applied = [r for r in pr if r["stop"] is None]
        assert applied and all(r["mask_rle"] is not None and r["items"][0]["sign"] == "+" for r in applied)
        assert len(plus["randbox"][base]) == len(applied)
        for rb, r in zip(plus["randbox"][base], applied):
            assert [(i["sign"], i["box"][2] - i["box"][0], i["box"][3] - i["box"][1]) for i in rb["items"]] == \
                [(i["sign"], i["box"][2] - i["box"][0], i["box"][3] - i["box"][1]) for i in r["items"]]
    # 第 1 轮 = 起点上按 guidance 施加模型的框（grabcut、软锁定、约束框）
    s0 = samples[0]
    w, h = rows[s0.id]["image_size"]
    from src.w2.loop import effective_masks

    st = effective_masks(rows[s0.id])[1]  # 挑了 2 号
    img = XT.load_rgb(str(s0.images[0]))
    m1, _, _ = G.correction_round(img, st, None, XP.proposals_abs(plus["cor"]["select"][0]["items"], w, h), rule="grabcut", soft_lock=True,
                                  confine_box=plus["box"])
    np.testing.assert_array_equal(rle_to_mask(plus["cor"]["select"][0]["mask_rle"]).astype(bool), m1)
    # 调用：关思考、单张图（修正是叠加图）、生成上限
    sp_calls = [c for c in ad.calls if c["prompt"].startswith("The object")]
    co_calls = [c for c in ad.calls if "current mask" in c["prompt"]]
    assert len(sp_calls) == 3 and all(c["n_images"] == 1 and c["max_new_tokens"] == XP.max_new_tokens_spline(16) for c in sp_calls)
    assert all(c["n_images"] == 1 and c["max_new_tokens"] == XP.max_new_tokens_correct(2) for c in co_calls)
    assert all(c["image_sizes"][0] == XR_.overlay_size(w, h, W2.style) for c in co_calls)
    assert all(c["enable_thinking"] is False for c in ad.calls)
    sel = [c for c in ad.calls if "candidate masks" in c["prompt"]]
    assert sorted({c["n_images"] for c in sel}) == [5, 6]  # separate 布局加参照：K = 4 时 5 张、K = 5 时 6 张
    assert len(co_calls) == sum(len(t["cor"][b]) for t in recs for b in XR.BASES)
    assert {tm["id"] for tm in tms} == {s.id for s in samples} and all("select4" in tm["model_s"] for tm in tms)


def test_tool_pool_is_bitwise_identical(traced):
    samples, rows, recs, _, _ = traced
    with XT.ToolPool(2) as pool:
        recs2, _ = XR.process_chunk(_ctx(XF.ScriptedX2Adapter(select="2"), pool), samples, rows)
    assert recs2 == recs


def test_settings_masks_and_accounting(traced):
    samples, rows, recs, tms, _ = traced
    by, tby = {r["id"]: r for r in recs}, {t["id"]: t for t in tms}
    defs = XR.setting_defs(3)
    prop = {"n_prompt_tokens": 10, "n_generated_tokens": 5, "n_inserted_tokens": 0, "n_visual_tokens": 64, "n_image_pixels": 65536}
    from src.w2.loop import effective_masks

    for s in samples:
        t, row = by[s.id], rows[s.id]
        cands = effective_masks(row)
        out = {}
        for name, d in defs.items():
            rec, comp = XR.setting_record(name, d, s, t, tby[s.id], row, prop, 0.5, 0.25, "devcheck2")
            out[name] = rec
            assert rec["score"]["inter"] <= rec["score"]["union"] and rec["setting"] == name and comp.n_calls == rec["n_model_calls"]
            assert comp.wall_time_s >= comp.call_time_s >= 0.5
        sel = rle_to_mask(out["x2_select"]["answer"]["mask"]["rle"]).astype(bool)
        np.testing.assert_array_equal(sel, cands[1])
        assert out["x2_select"]["n_model_calls"] == 2 and out["x2_spline_raster"]["n_model_calls"] == 2
        assert out["x2_select5"]["n_model_calls"] == 3 and out["x2_oracle4"]["n_model_calls"] == 1 and out["x2_ellipse_trimap"]["n_model_calls"] == 1
        nc = len(t["cor"]["select"])
        for r in (1, 2, 3):
            assert out[f"x2_select_cor{r}"]["n_model_calls"] == 2 + min(r, nc)
            assert out[f"x2_select_cor{r}_randbox"]["n_model_calls"] == out[f"x2_select_cor{r}"]["n_model_calls"]
            assert out[f"x2_select5_cor{r}"]["n_model_calls"] == 3 + min(r, len(t["cor"]["select5"]))
        applied = [x for x in t["cor"]["select"] if x["mask_rle"] is not None]
        want1 = rle_to_mask(applied[0]["mask_rle"]).astype(bool) if applied else cands[1]
        np.testing.assert_array_equal(rle_to_mask(out["x2_select_cor1"]["answer"]["mask"]["rle"]).astype(bool), want1)
        np.testing.assert_array_equal(rle_to_mask(out["x2_spline_raster"]["answer"]["mask"]["rle"]).astype(bool),
                                      rle_to_mask(t["spline"]["raster_rle"]).astype(bool))
        assert out["x2_oracle4"]["score"]["iou"] >= out["x2_select"]["score"]["iou"]
        assert out["x2_oracle5"]["score"]["iou"] >= out["x2_oracle4"]["score"]["iou"]
        assert out["x2_select_cor3"]["tokens_by_stage"]["cor_select"]["calls"] == min(3, nc)
    # 样条解析失败：交 1 号（fallback = parse_failure）；第 5 个候选画空
    s = samples[0]
    t = json.loads(json.dumps(by[s.id]))
    t["spline"] = {"call": {**t["spline"]["call"], "failure": "truncated"}, "ctrl_q": None}
    rec, _ = XR.setting_record("x2_spline_trimap", defs["x2_spline_trimap"], s, t, tby[s.id], rows[s.id], prop, 0.0, 0.0, "devcheck2")
    assert rec["fallback"] == "parse_failure" and rec["info"]["spline_failure"] == "truncated"
    np.testing.assert_array_equal(rle_to_mask(rec["answer"]["mask"]["rle"]).astype(bool), effective_masks(rows[s.id])[0])
    o5 = XR.final_of("x2_oracle5", defs["x2_oracle5"], t, rows[s.id], gt_mask(s))
    assert o5["mask"] is not None
    # 提出失败：没有 mask，所有设置都记失败
    rows_f = _cand_rows(samples, fail_first=True)
    tf = dict(by[s.id], failure="no_box", box=None, box_q=None)
    for name, d in defs.items():
        rec, comp = XR.setting_record(name, d, s, tf, tby[s.id], rows_f[s.id], prop, 0.0, 0.0, "devcheck2")
        assert rec["failure"] == "no_box" and rec["answer"] is None and rec["score"]["inter"] == 0 and rec["n_model_calls"] == 1


# ---------------------------------------------------------------------------
# 命令行 tiny 端到端
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def x2_ckpt(tmp_path_factory):
    """tiny 的 X-2 检查点（命令行：假数据 → build_x2_data → 回放数据 → train_w2c --config configs/x2_train.json）。"""
    from tests.test_w2c_replay import inject_stops

    tmp = tmp_path_factory.mktemp("x2ck")
    w2c = XF.write_fake_w2c_data(tmp / "w2c_data")
    run(BUILD, ["--tiny", "--w2c-data", w2c, "--out", tmp / "x2"], tmp)
    run(BUILD_RP, ["--tiny", "--w2c-data", w2c, "--out", tmp / "rp"], tmp)
    shutil.copytree(tmp / "rp", tmp / "rpi")
    inject_stops(tmp / "rpi")
    run(TRAIN, ["--tiny", "--model", QWEN, "--config", REPO / "configs" / "x2_train.json", "--w2c-data", w2c, "--x2-data", tmp / "x2",
                "--replay-data", tmp / "rpi", "--out", tmp / "train"], tmp)
    return tmp / "train" / "w2c" / QWEN / "seed0" / "ckpt" / "final" / "model"


def _inject_boxes(leaf: Path) -> None:
    gt = {s.id: s.gt["box"] for s in w2_items("refexp_box", "devcheck2", tiny_paths(), splits=tiny_splits(W2))}
    recs = jsonl(leaf / "predictions.jsonl")
    for r in recs[1:]:  # 第一条保留随机模型的解析失败（提出失败的路径）
        x0, y0, x1, y1 = gt[r["id"]]
        r.update(failure=None, parsed={"box": [x0 + 1, y0 + 1, x1 - 1, y1]})
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def _load_gate_explore():
    spec = importlib.util.spec_from_file_location("_gate_explore_for_x2", REPO / "scripts" / "gate_explore.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_end_to_end_outputs_resume_and_gate_reader(x2_ckpt, tmp_path):
    tmp = tmp_path
    out = tmp / "ev"
    base = ["--tiny", "--model", QWEN, "--checkpoint", x2_ckpt, "--split", "devcheck2", "--out", out, *FAKES]
    run(RUN_X2, base, tmp)
    m = out / QWEN
    _inject_boxes(m / "refexp_box" / "propose_w2c" / "devcheck2")
    for d in ["candidates_w2c", XR.TRACE_SETTING, XR.SUMMARY_DIR, *XR.setting_defs(3)]:
        shutil.rmtree(m / "refseg" / d, ignore_errors=True)
    p = run(RUN_X2, base + ["--tool-workers", "2"], tmp)
    assert "x2_select=" in p.stdout and "出框" not in p.stderr  # 提出已写完：直接读
    ck_sha = __import__("hashlib").sha256((x2_ckpt / "manifest.json").read_bytes()).hexdigest()
    ge = _load_gate_explore()
    for name, d in XR.setting_defs(3).items():
        leaf = m / "refseg" / name / "devcheck2"
        recs = jsonl(leaf / "predictions.jsonl")
        assert len(recs) == 3 and recs[0]["failure"] is not None and all(r["failure"] is None for r in recs[1:])
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["w2"]["setting"] == name and cfg["w2"]["split"] == "devcheck2" and cfg["checkpoint"]["manifest_sha256"] == ck_sha
        assert cfg["x2"]["setting"] == name and cfg["x2"]["definition"] == json.loads(json.dumps(d)) and cfg["x2"]["n"] == 16
        met = json.loads((leaf / "metrics.json").read_text())
        assert met["complete"] is True and met["n"] == 3 and [h["name"] for h in met["headline"]] == ["cIoU", "gIoU"]
        comp = json.loads((leaf / "compute.json").read_text())
        assert comp["n_samples"] == 3 and comp["n_calls"] == sum(r["n_model_calls"] for r in recs)
        assert (leaf / "meta.json").exists() and (leaf / "README.md").exists()
        g = ge.load_dir(leaf, "候选")  # scripts/gate_explore.py 的读法
        assert g["identity"]["setting"] == name and g["identity"]["checkpoint"]["manifest_sha256"] == ck_sha and len(g["records"]) == 3
    summ = json.loads((m / "refseg" / XR.SUMMARY_DIR / "devcheck2.json").read_text())
    assert set(summ["settings"]) == set(XR.setting_defs(3)) and "x2_select_cor3" in summ["compute_pairing"]
    pair = summ["compute_pairing"]["x2_select_cor1"]["single_round_at_most_this_compute"]
    assert all(c <= summ["compute_pairing"]["x2_select_cor1"]["calls_per_sample"] for _, c in pair)
    tr = jsonl(m / "refseg" / XR.TRACE_SETTING / "devcheck2" / "trace.jsonl")
    assert len(tr) == 3 and tr[0]["failure"] is not None and all(t["spline"]["call"]["failure"] for t in tr[1:])  # 随机模型：样条失败
    before = {n: (m / "refseg" / n / "devcheck2" / "predictions.jsonl").read_bytes() for n in XR.setting_defs(3)}
    run(RUN_X2, base + ["--resume"], tmp)
    assert all((m / "refseg" / n / "devcheck2" / "predictions.jsonl").read_bytes() == b for n, b in before.items())
    p = run(RUN_X2, base, tmp, ok=False)  # 不加 --resume：拒绝覆盖
    assert p.returncode == 2 and "--resume" in p.stderr


def test_cli_guards(x2_ckpt, tmp_path):
    tmp = tmp_path
    base = ["--tiny", "--model", QWEN, "--checkpoint", x2_ckpt, *FAKES]
    p = run(RUN_X2, base + ["--split", "test_refcoco_unc_testA", "--out", tmp / "t"], tmp, ok=False)
    assert p.returncode == 2 and "test" in p.stderr and not (tmp / "t").exists()
    ex = ["--splits-file", REPO / "configs" / "explore_splits.json"]
    p = run(RUN_X2, base + ex + ["--split", "devcheck3_refcoco_unc_val", "--out", tmp / "d3"], tmp, ok=False)
    assert p.returncode == 2 and "G-EX" in p.stderr and not (tmp / "d3").exists()
    p = run(RUN_X2, base + ex + ["--split", "devcheck3_refcoco_unc_val", "--unseal-devcheck3", "G-X", "--out", tmp / "d3"], tmp, ok=False)
    assert p.returncode == 2 and not (tmp / "d3").exists()
    p = run(RUN_X2, base + ex + ["--split", "explore_val_refcoco_unc_val", "--unseal-devcheck3", "G-EX", "--out", tmp / "ev"], tmp, ok=False)
    assert p.returncode == 2 and "只配密封" in p.stderr
    p = run(RUN_X2, base + ["--split", "devcheck2", "--settings", "x2_magic", "--out", tmp / "s"], tmp, ok=False)
    assert p.returncode == 2 and "未知的设置" in p.stderr
    p = run(RUN_X2, ["--model", QWEN, "--checkpoint", x2_ckpt, "--split", "devtune800", "--out", tmp / "noid", "--data-root", "/x"], tmp,
            ok=False)
    assert p.returncode == 2 and "运行编号" in p.stderr
    # devcheck3 放行：门名进运行身份，放行记录追加进 unseal_devcheck3.jsonl
    out = tmp / "d3ok"
    run(RUN_X2, base + ex + ["--split", "devcheck3_refcoco_unc_val", "--unseal-devcheck3", "G-EX", "--out", out,
                             "--settings", "x2_select", "x2_select5_cor3"], tmp)
    leaf = out / QWEN / "refseg" / "x2_select" / "devcheck3_refcoco_unc_val"
    assert yaml.safe_load((leaf / "config.yaml").read_text())["w2"]["unseal"]["gate"] == "G-EX"
    assert len(jsonl(out / QWEN / "unseal_devcheck3.jsonl")) == 1
    assert not (out / QWEN / "refseg" / "x2_spline_raster").exists()  # --settings 只导出给定的


def test_runner_process_imports_no_learned_vision_modules():
    code = ("import json, sys; import src.explore.x2_runner, src.explore.x2_tools; import src.explore.guidance as G; "
            "allowed = {'torch', 'transformers'}; "
            "print(json.dumps(sorted(r for r in {m.split('.')[0] for m in sys.modules} if r in G.BANNED_MODULE_ROOTS and r not in allowed)))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert json.loads(r.stdout.strip().splitlines()[-1]) == []  # LLM 本身的 torch、transformers 之外，没有学习型的视觉模块
    import re

    for f in ("x2_runner.py", "x2_tools.py", "x2_data.py", "x2_prompts.py", "x2_render.py"):
        src = (REPO / "src" / "explore" / f).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(?:from|import)\s+(?:" + "|".join(G.BANNED_MODULE_ROOTS) + r")\b", src, re.M), f
