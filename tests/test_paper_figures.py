"""DEV-0052：scripts/figures/make_paper_figures.py（论文用图表，D103 §五）。

用小的假结果树跑一遍：
- 每个输出文件都生成（PDF、300 dpi 的 PNG、LaTeX 与 Markdown 的表、图注、data/*.json、README）；
- 读到的数字与假数据一致（E1 的读数与配对差、E2 的逐层读数与相对深度、E3 表的种子平均与标准差、干扰的 L/P/R）；
- 缺 9B（RUN-0056）或秩 128（RUN-0054）时不报错，README 与图注写明缺；目录出现后自动纳入；
- 秩 128 的 C3、E3b+C3 不回退到秩 32 的读数；被合并核对拒绝的种子在表注里列出、不进平均；
- JSON 与重算不一致时 README 报"不一致"。
- DEV-0063：--e3-prepend 把按 e5-v2 补评的目录放在最前，补评的种子（两臂）覆盖原目录，表注与干扰图注写明 e5-v2。
- DEV-0073：通过 G-EX 的 C1 = X-2、椭圆消融、C2 = X-3 进 W2 表与正文主表 tab_main；合并列取 w2_test_pooled.json（W2-a 缺与不缺
  两种情况用临时目录里的合成 JSON）；语言代价四方表 tab_lang_cost（与 _full）、计算表 tab_compute 与分析 JSON 逐项一致；
  E1、E2、E3、干扰图与 fig_w2_size 逐字节不变。
- DEV-0077：统一主表 tab_unified 与图 fig_unified、分档图 fig_bins、种子波动表 tab_seeds 与分析 JSON 逐项一致；跑坏的系统照样占一行；
  已有图表与 DEV-0077 动手前（a29e96c81）的逐字节相同，tab_main 只改了 caption。
- DEV-0078（D162）：四方表（与 _full）在 C1 列后加一列 C1 配方 3 个种子的范围，语言代价图在 C1 的点下面另画一行范围；范围与 seed_variance.json
  逐项一致（并钉住负责人答复 N46 的数）；表注、图注写明其余各列单种子、以大题量为准；tab_seeds 的表注加种子稳定性一句；新增 4B 补充表 tab_x2_4b
  与 x2_4b.json、seed_variance.json 逐项一致。其余图表与 DEV-0078 动手前（dfa4383e7）的逐字节相同，改了的几个只多出点名的列、行与表注。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytest.importorskip("matplotlib")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MPF = _load("_paperfig_under_test", "scripts/figures/make_paper_figures.py")
ITF = MPF.ITF

TT_TASKS = [t for _, t in ITF.L_ITEMS]
SIZE_IDX = {"0.8B": 0, "2B": 1, "4B": 2, "9B": 3}


# ---------------------------------------------------------------------------------------------
# 假数据（每个读数由一个简单公式给出，测试里独立算期望值）
# ---------------------------------------------------------------------------------------------


def ttext_value(size: str, arm: str, seed: int, task: str, rank: int = 32) -> float:
    i, k = SIZE_IDX[size], TT_TASKS.index(task) if task in TT_TASKS else 9
    base = 0.50 + 0.03 * i + 0.01 * k
    if arm == "base":
        return base
    if arm == "c3":
        return base + 0.02 + 0.004 * seed + (0.001 if rank == 128 else 0.0)
    if arm == "e3b_c3":  # 干扰随尺寸变小；秩 128 再小一半
        gap = (0.12 - 0.03 * i + 0.002 * seed) * (0.5 if rank == 128 else 1.0)
        return ttext_value(size, "c3", seed, task, rank) - gap
    if arm == "e3b":
        return 0.0
    raise KeyError(arm)


def grid_value(size: str, arm: str, seed: int, task: str, rank: int = 32) -> float:
    i = SIZE_IDX[size]
    if task == "refseg":
        e = 0.30 + 0.08 * i + 0.01 * seed
        return e if arm == "e3b" else e - (0.10 - 0.02 * i) * (0.5 if rank == 128 else 1.0)
    e = 0.40 - 0.05 * i + 0.003 * seed  # depth AbsRel，E3b+C3 更大 = 更差
    return e if arm == "e3b" else e + (0.05 - 0.01 * i) * (0.5 if rank == 128 else 1.0)


def _w(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _ci(v, h=0.01, n_resamples=1000):
    return {"low": v - h, "high": v + h, "level": 0.95, "n_resamples": n_resamples}


def _ttext_metrics(task: str, v: float) -> dict:
    if task == "ovd":
        return {"headline": [{"name": "F1@0.5", "scope": "coco_val2017", "value": v, "ci": _ci(v), "n": 500}]}
    name = {"refexp_box": "acc@0.5", "refexp_poly": "cIoU", "grid16": "gIoU"}.get(task, "accuracy")
    return {"headline": [{"name": name, "scope": "all", "value": v, "ci": _ci(v), "n": 100}]}


def _grid_metrics(task: str, v: float) -> dict:
    key = "cIoU" if task == "refseg" else "abs_rel"
    return {"headline": {"procedure": "single", "upsample": "guided", key: {"value": v, "ci": _ci(v), "n": 50}}}


def _merge_check(passed: bool, gpu: str) -> dict:
    fails = {"restricted": [] if passed else ["restricted/thinking_off：关思考时受限解码与 generate_batch 的 exact_rows 0.9300 < 0.95"],
             "merge": []}
    return {"pass": passed, "criteria_version": "v1", "gpu": gpu, "failures": fails}


def e1_values(model: str) -> dict:
    j = [m for _, m in MPF.QWEN + MPF.E1_REFERENCE + MPF.E1_EXCLUDED].index(model)
    return {"grounding": 0.70 + 0.02 * j, "own_box": 0.45 + 0.01 * j, "polygon": 0.0 if j < 4 else 0.05 * j,
            "grid16": 0.001 * j, "poly_structured": 0.10 + 0.05 * j, "poly_structured_min8": 0.08 + 0.05 * j,
            "valid": 0.2 + 0.01 * j}


def build_e1(root: Path) -> None:
    models = []
    for _lab, m in MPF.QWEN + MPF.E1_REFERENCE + MPF.E1_EXCLUDED:
        v = e1_values(m)
        base = root / MPF.E1_MAIN / m
        _w(base / "refexp_box" / "metrics.json",
           {"headline": [{"name": "acc@0.5", "scope": "all", "value": v["grounding"], "ci": _ci(v["grounding"], 0.003), "n": 56857}],
            "valid": {"rate": v["valid"]}})
        _w(base / "refexp_poly" / "metrics.json",
           {"headline": [{"name": "cIoU", "scope": "all", "value": v["polygon"], "ci": _ci(v["polygon"], 0.0), "n": 8000}]})
        _w(base / "grid16" / "metrics.json",
           {"headline": [{"name": "gIoU", "scope": "all", "value": v["grid16"], "ci": _ci(v["grid16"], 0.0005), "n": 500},
                         {"name": "cIoU", "scope": "all", "value": 0.9, "n": 500}]})
        rows = [{"readout": "own_box", "strict": {"cIoU": {"value": v["own_box"], "ci": _ci(v["own_box"], 0.03), "n": 200}}}]
        if m in dict((mm, 1) for _l, mm in MPF.QWEN):
            for r in ("poly_structured", "poly_structured_min8"):
                d = v[r] - v["own_box"]
                rows.append({"readout": r, "strict": {"cIoU": {"value": v[r], "ci": _ci(v[r], 0.02), "n": 200}},
                             "minus": "own_box", "paired_diff": {"cIoU": {"value": d, "ci": _ci(d, 0.04), "n": 200}}})
            _w(root / MPF.E1_POSTHOC2 / m / "refexp_poly__poly_structured" / "metrics.json",
               {"headline": [{"name": "cIoU", "scope": "all", "value": v["poly_structured"], "n": 200}]})
        models.append({"model": m, "tasks": {"refexp_poly": {"rows": rows}}})
    _w(root / MPF.E1_OWNBOX, {"kind": "ownbox", "models": models})


E2_LAYERS = {"qwen3.5-0.8b": list(range(0, 5)), "qwen3.5-2b": list(range(0, 5)), "qwen3.5-4b": list(range(0, 7)),
             "qwen3.5-9b": list(range(0, 9, 2))}
E2_METRIC = {"depth": ("abs_rel", "↓"), "normals": ("mean", "↓"), "semseg": ("miou", "↑"), "refseg": ("ciou", "↑")}


def e2_value(model: str, task: str, layer: int) -> float:
    return 0.2 + 0.01 * layer + 0.001 * list(E2_LAYERS).index(model) + 0.1 * list(E2_METRIC).index(task)


def build_e2(root: Path) -> None:
    summ = {"models": {}}
    for model, layers in E2_LAYERS.items():
        for task, (metric, better) in E2_METRIC.items():
            probes = {}
            for li in layers:
                v = e2_value(model, task, li)
                probes[f"L{li}/linear"] = {"kind": "linear", "feature": f"L{li}",
                                           "eval": {metric: v, metric + "_ci": {"low": v - 0.005, "high": v + 0.005}}}
                probes[f"L{li}/conv"] = {"kind": "conv", "feature": f"L{li}", "eval": {metric: v - 0.05}}
            _w(root / MPF.E2_RAW / model / task / "metrics.json",
               {"headline": [{"metric": metric, "better": better}], "layers": layers, "n_eval": 654, "probes": probes})
            summ["models"].setdefault(model, {})[task] = {"kinds": {"linear": {"curve": {f"L{li}": e2_value(model, task, li)
                                                                                         for li in layers}}}}
    _w(root / MPF.E2_SUMMARY, summ)


def _arm_tree(d: Path, size: str, model: str, arms_seeds: dict, gpu: str, rank: int = 32,
              rejected: tuple = (), grid_skip: tuple = ()) -> None:
    """arms_seeds = {臂: [种子…]}；c3、e3b_c3 写 T 文本，e3b_c3、e3b 写结构强制网格。"""
    for arm, seeds in arms_seeds.items():
        for s in seeds:
            if arm in ("c3", "e3b_c3"):
                tdir = d / "t_text" / arm / f"seed{s}" / model
                _w(tdir / "merge_check.json", _merge_check(True, gpu))
                for t in TT_TASKS + ["grid16"]:
                    _w(tdir / t / "metrics.json", _ttext_metrics(t, ttext_value(size, arm, s, t, rank)))
            if arm in ("e3b_c3", "e3b"):
                gdir = d / "eval" / arm / model / f"seed{s}"
                ok = (arm, s) not in rejected
                _w(gdir / "merge_check.json", _merge_check(ok, gpu))
                if ok and (arm, s) not in grid_skip:
                    for t in ("refseg", "depth"):
                        _w(gdir / f"{t}__structured" / "metrics.json", _grid_metrics(t, grid_value(size, arm, s, t, rank)))


def _base_dir_index(task: str, n_dirs: int) -> int:
    """同真的布局：2B 的基座在三个 E1 目录里——主评测（0027）、去偏（0030）、对调标记（0033）。"""
    if n_dirs == 1:
        return 0
    return 1 if task.endswith("__debiased") else 2 if task.endswith("__letter_swap") else 0


def _base_tree(dirs: list[Path], size: str) -> None:
    for t in TT_TASKS:
        f = dirs[_base_dir_index(t, len(dirs))] / t / "metrics.json"
        if not f.exists():  # 2B 的定位、多边形就是 E1 主评测本身（build_e1 已写），不覆盖
            _w(f, _ttext_metrics(t, ttext_value(size, "base", 0, t)))


def base_expected(size: str, task: str) -> float:
    if size == "2B" and task in ("refexp_box", "refexp_poly"):
        return e1_values("qwen3.5-2b")["grounding" if task == "refexp_box" else "polygon"]
    return ttext_value(size, "base", 0, task)


REJECTED_4B = (("e3b", 2),)


def build_e3(root: Path, with_9b: bool = False, with_r128: bool = False, r128_grid_skip: tuple = ()) -> None:
    for spec in MPF.E3_SIZES:
        if spec["size"] == "9B" and not with_9b:
            continue
        dirs = [root / d for d in spec["dirs"]]
        size, model = spec["size"], spec["model"]
        seeds = list(spec["seeds"])
        gpu = "NVIDIA B200" if spec["machine"] == "B200" else "NVIDIA A100-PCIE-40GB"
        if size == "2B":  # 三个目录：c3 种子 0 在第一个，E3b 种子 0 的网格在第二个，其余在第三个
            _arm_tree(dirs[0], size, model, {"c3": [0]}, gpu)
            _arm_tree(dirs[1], size, model, {"e3b": [0]}, gpu)
            _arm_tree(dirs[2], size, model, {"c3": [1, 2], "e3b_c3": seeds, "e3b": [1, 2]}, gpu)
        else:
            _arm_tree(dirs[0], size, model, {"c3": seeds, "e3b_c3": seeds, "e3b": seeds}, gpu,
                      rejected=REJECTED_4B if size == "4B" else ())
        _base_tree([root / d for d in spec["base"]], size)
    if with_r128:
        spec = MPF.E3_R128
        _arm_tree(root / spec["dirs"][0], spec["size"], spec["model"], {"c3": list(spec["seeds"]), "e3b_c3": list(spec["seeds"])},
                  "NVIDIA A100-PCIE-40GB", rank=128, grid_skip=r128_grid_skip)


def write_interference_json(root: Path) -> None:
    """同真的 docs/analysis/e3_interference_by_size.json：interference.py 对 0.8B、2B、4B 的输出。"""
    blocks = [ITF.size_block(s["size"], s["machine"], s["model"], [str(root / d) for d in s["dirs"]])
              for s in MPF.E3_SIZES if s["size"] != "9B"]
    _w(root / MPF.INTERFERENCE_JSON, {"blocks": blocks, "trend": {"L": "随尺寸缩小"}, "relative_trend_descriptive": {}})


def build_tree(root: Path, **kw) -> Path:
    build_e1(root)
    build_e2(root)
    build_e3(root, **kw)
    write_interference_json(root)
    return root


def run(root: Path, out: Path) -> Path:
    assert MPF.main(["--root", str(root), "--out", str(out)]) == 0
    return out


def data(out: Path, name: str) -> dict:
    return json.loads((out / "data" / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    root = build_tree(tmp_path_factory.mktemp("tree"))
    out = run(root, tmp_path_factory.mktemp("out"))
    return root, out


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    root = build_tree(tmp_path_factory.mktemp("tree_full"), with_9b=True, with_r128=True, r128_grid_skip=(("e3b_c3", 2),))
    out = run(root, tmp_path_factory.mktemp("out_full"))
    return root, out


# ---------------------------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------------------------


def test_labels_cover_section13_items():
    assert set(MPF.TTEXT_LABELS) == set(TT_TASKS)
    assert len(TT_TASKS) == 8


def test_all_outputs_written(base_run):
    from PIL import Image

    _root, out = base_run
    for name, width_in in (("fig_e1_gap", MPF.DOUBLE_COL_IN), ("fig_e2_layers", MPF.DOUBLE_COL_IN),
                           ("fig_interference_size", MPF.SINGLE_COL_IN)):
        pdf, png = out / f"{name}.pdf", out / f"{name}.png"
        assert pdf.stat().st_size > 1000 and pdf.read_bytes()[:5] == b"%PDF-"
        assert b"/Type3" not in pdf.read_bytes()  # TrueType 字体（会议要求无 Type 3）
        with Image.open(png) as im:
            assert abs(im.info["dpi"][0] - MPF.DPI) < 1
            assert im.size[0] == round(width_in * MPF.DPI)
        cap = (out / f"{name}_caption.tex").read_text(encoding="utf-8")
        assert cap.count("\\caption{") == 1
        assert (out / "data" / f"{name}.json").is_file()
    tex = (out / "tab_e3_comparison.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert (out / "tab_e3_comparison.md").read_text(encoding="utf-8").count("| Base |") == 3
    assert (out / "data" / "tab_e3_comparison.json").is_file()
    assert (out / "README.md").is_file()


def test_e1_numbers(base_run):
    _root, out = base_run
    d = data(out, "fig_e1_gap")
    for _lab, m in MPF.QWEN + MPF.E1_REFERENCE:
        v = e1_values(m)
        r = d["models"][m]["readouts"]
        for key in ("grounding", "own_box", "polygon", "grid16"):
            assert r[key]["value"] == pytest.approx(v[key])
        if m in dict((mm, 1) for _l, mm in MPF.QWEN):
            assert r["poly_structured"]["value"] == pytest.approx(v["poly_structured"])
            for readout in ("poly_structured", "poly_structured_min8"):
                p = d["paired"][m][readout]
                assert p["value"] == pytest.approx(v[readout] - v["own_box"])
        else:
            assert r["poly_structured"] is None  # InternVL 没有跑结构强制
    assert r["grounding"]["low"] == pytest.approx(v["grounding"] - 0.003)
    assert d["n_paired"] == 8
    assert d["n_paired_ci_excluding_zero"] == 8  # 假数据的配对差 ≤ −0.23，CI 半宽 0.04
    assert d["models"]["gemma-4-e2b-it"]["group"] == "excluded"
    assert all(c.startswith("一致") for c in d["checks"])
    cap = (out / "fig_e1_gap_caption.tex").read_text(encoding="utf-8")
    assert "n=56,857" in cap and "8 of 8 CIs exclude 0" in cap
    v = e1_values("gemma-4-e2b-it")["valid"]
    assert f"Gemma-4-E2B {100 * v:.0f}" in cap


def test_e2_numbers(base_run):
    _root, out = base_run
    d = data(out, "fig_e2_layers")
    for model, layers in E2_LAYERS.items():
        for task, (metric, _b) in E2_METRIC.items():
            rec = d["models"][model]["tasks"][task]
            assert rec["metric"] == metric
            assert rec["layers"] == layers
            assert rec["frac"] == pytest.approx([li / max(layers) for li in layers])
            assert rec["value"] == pytest.approx([e2_value(model, task, li) for li in layers])
            assert rec["low"] == pytest.approx([e2_value(model, task, li) - 0.005 for li in layers])
    assert all(c.startswith("一致") for c in d["checks"])
    assert "9B probed at every other layer" in (out / "fig_e2_layers_caption.tex").read_text(encoding="utf-8")


def _expected_cell(size, arm, task, seeds, grid=False, rank=32):
    f = grid_value if grid else ttext_value
    vals = [f(size, arm, s, task, rank) for s in seeds]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def test_e3_table_numbers(base_run):
    _root, out = base_run
    d = data(out, "tab_e3_comparison")
    sizes = {s["size"]: s for s in d["sizes"]}
    assert sizes["9B"]["present"] is False
    for size in ("0.8B", "2B", "4B"):
        rows = sizes[size]["rows"]
        for task in TT_TASKS:
            assert rows["base"][task]["mean"] == pytest.approx(base_expected(size, task))
            for arm in ("c3", "e3b_c3"):
                m, sd = _expected_cell(size, arm, task, [0, 1, 2])
                assert rows[arm][task]["mean"] == pytest.approx(m)
                assert rows[arm][task]["sd"] == pytest.approx(sd)
                assert rows[arm][task]["n"] == 3
            assert rows["e3b"][task]["n"] == 0  # E3b 没有 T 文本
        assert rows["c3"]["refseg"]["na"] and rows["base"]["depth"]["na"]
    # 4B 的 E3b 种子 2 被合并核对拒绝：网格只有种子 0、1，并在表注里列出
    m, sd = _expected_cell("4B", "e3b", "refseg", [0, 1], grid=True)
    cell = sizes["4B"]["rows"]["e3b"]["refseg"]
    assert (cell["n"], cell["mean"], cell["sd"]) == (2, pytest.approx(m), pytest.approx(sd))
    assert [(r["size"], r["arm"], r["seed"]) for r in d["rejected"]] == [("4B", "e3b", 2)]
    assert any("4B e3b seed 2: refseg structured grid（rejected by merge check）" in g for g in d["gaps"])
    md = (out / "tab_e3_comparison.md").read_text(encoding="utf-8")
    assert f"{m * 100:.1f} ± {sd * 100:.1f} (n=2)" in md
    assert "exact-row agreement 0.930 < 0.95" in md
    m, sd = _expected_cell("0.8B", "c3", "refexp_box", [0, 1, 2])
    assert f"{m * 100:.1f}$\\pm${sd * 100:.1f}" in (out / "tab_e3_comparison.tex").read_text(encoding="utf-8")
    assert "Missing (not yet available): Qwen3.5-9B (RUN-0056)" in md


def _expected_L(size, seeds, rank=32):
    per_seed = [statistics.mean(100 * (ttext_value(size, "c3", s, t, rank) - ttext_value(size, "e3b_c3", s, t, rank))
                                for t in TT_TASKS) for s in seeds]
    return per_seed


def test_interference_numbers_and_missing(base_run):
    _root, out = base_run
    d = data(out, "fig_interference_size")
    blocks = {b["size"]: b for b in d["blocks"]}
    assert set(blocks) == {"0.8B", "2B", "4B"}
    for size in ("0.8B", "2B", "4B"):
        b = blocks[size]
        assert [b["L"][str(s)] for s in (0, 1, 2)] == pytest.approx(_expected_L(size, [0, 1, 2]))
        assert b["means"]["L"]["mean"] == pytest.approx(statistics.mean(_expected_L(size, [0, 1, 2])))
        assert b["source"] == MPF.INTERFERENCE_JSON
    seeds_p = [0, 1]  # 4B 的 E3b 种子 2 被拒
    assert sorted(int(k) for k in blocks["4B"]["P_seg"]) == seeds_p
    for s in seeds_p:
        e, mix = grid_value("4B", "e3b", s, "refseg"), grid_value("4B", "e3b_c3", s, "refseg")
        assert blocks["4B"]["P_seg"][str(s)] == pytest.approx(e - mix)
        assert blocks["4B"]["R_seg"][str(s)] == pytest.approx((e - mix) / e)
        assert blocks["4B"]["P_depth"][str(s)] == pytest.approx(grid_value("4B", "e3b_c3", s, "depth") - grid_value("4B", "e3b", s, "depth"))
    assert all(c.startswith("一致") for c in d["checks"])
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "missing" in readme and "RUN-0056" in readme and "RUN-0054" in readme
    cap = (out / "fig_interference_size_caption.tex").read_text(encoding="utf-8")
    assert "Qwen3.5-9B (RUN-0056) not yet available" in cap
    assert "rank 128 (RUN-0054) not yet available" in cap
    assert "4B P\\_seg: 2 of 3 seeds; 4B P\\_depth: 2 of 3 seeds" in cap


def test_9b_and_r128_picked_up_when_present(full_run):
    _root, out = full_run
    t = data(out, "tab_e3_comparison")
    s9 = next(s for s in t["sizes"] if s["size"] == "9B")
    assert s9["present"] is True and s9["seeds"] == [0]
    assert s9["rows"]["c3"]["refexp_box"]["mean"] == pytest.approx(ttext_value("9B", "c3", 0, "refexp_box"))
    assert s9["rows"]["c3"]["refexp_box"]["sd"] is None
    md = (out / "tab_e3_comparison.md").read_text(encoding="utf-8")
    assert "| 9B | Base |" in md and "Missing (not yet available)" not in md
    d = data(out, "fig_interference_size")
    blocks = {b["size"]: b for b in d["blocks"]}
    assert set(blocks) == {"0.8B", "2B", "4B", "9B", "0.8B r128"}
    b9 = blocks["9B"]
    assert b9["source"].startswith("computed with scripts/mgr/interference.py")
    assert [b9["L"]["0"]] == pytest.approx(_expected_L("9B", [0]))
    br = blocks["0.8B r128"]
    assert br["rank"] == 128 and br["x_size"] == "0.8B"
    assert [br["L"][str(s)] for s in (0, 1, 2)] == pytest.approx(_expected_L("0.8B", [0, 1, 2], rank=128))
    # P：E3b 用秩 32，E3b+C3 用秩 128；秩 128 的种子 2 没有网格读数，不回退到秩 32 的 E3b+C3
    assert sorted(int(k) for k in br["P_seg"]) == [0, 1]
    for s in (0, 1):
        assert br["P_seg"][str(s)] == pytest.approx(grid_value("0.8B", "e3b", s, "refseg") -
                                                    grid_value("0.8B", "e3b_c3", s, "refseg", rank=128))
    readme = (out / "README.md").read_text(encoding="utf-8")
    missing = readme.split("## 缺的数据（missing）")[1].split("## 1.")[0]
    assert "RUN-0056" not in missing.split("E3 表里缺的格子")[0] and "RUN-0054" not in missing
    cap = (out / "fig_interference_size_caption.tex").read_text(encoding="utf-8")
    assert "not yet available" not in cap and "LoRA rank 128" in cap


def test_9b_dir_without_pairs_is_reported_missing(tmp_path):
    root = build_tree(tmp_path / "tree")
    spec = next(s for s in MPF.E3_SIZES if s["size"] == "9B")
    _arm_tree(root / spec["dirs"][0], "9B", spec["model"], {"c3": [0]}, "NVIDIA A100-PCIE-40GB")  # 只有 C3，还没有 E3b+C3
    out = run(root, tmp_path / "out")
    d = data(out, "fig_interference_size")
    assert "9B" not in {b["size"] for b in d["blocks"]}
    assert any("9B" in m and "还没有两臂配对的读数" in m for m in d["missing"])
    t = data(out, "tab_e3_comparison")
    assert any(g.startswith("E3 9B e3b_c3 seed 0") for g in t["gaps"])


def test_9b_with_language_pair_only_shows_L_and_names_missing_P(tmp_path):
    """RUN-0056 的顺序是先出 L 的两臂、后训 E3b：中间状态下 9B 只画 L，图注写明 P 还缺。"""
    root = build_tree(tmp_path / "tree")
    spec = next(s for s in MPF.E3_SIZES if s["size"] == "9B")
    _arm_tree(root / spec["dirs"][0], "9B", spec["model"], {"c3": [0], "e3b_c3": [0]}, "NVIDIA A100-PCIE-40GB")
    shutil.rmtree(root / spec["dirs"][0] / "eval")  # E3b+C3 的网格也还没评
    out = run(root, tmp_path / "out")
    b9 = next(b for b in data(out, "fig_interference_size")["blocks"] if b["size"] == "9B")
    assert [b9["L"]["0"]] == pytest.approx(_expected_L("9B", [0])) and not b9["P_seg"]
    cap = (out / "fig_interference_size_caption.tex").read_text(encoding="utf-8")
    assert "Qwen3.5-9B (RUN-0056): P\\_seg not yet available" in cap
    assert "Qwen3.5-9B (RUN-0056): L not yet available" not in cap


def test_interference_json_mismatch_is_flagged(tmp_path):
    root = build_tree(tmp_path / "tree")
    p = root / MPF.INTERFERENCE_JSON
    j = json.loads(p.read_text(encoding="utf-8"))
    j["blocks"][0]["L"]["0"] += 1.0
    p.write_text(json.dumps(j), encoding="utf-8")
    out = run(root, tmp_path / "out")
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "不一致：0.8B" in readme
    # 图上画的是 JSON 的值（任务书：数据取 interference.py 的 JSON 输出），不一致只报告
    d = data(out, "fig_interference_size")
    b = next(b for b in d["blocks"] if b["size"] == "0.8B")
    assert b["L"]["0"] == pytest.approx(_expected_L("0.8B", [0])[0] + 1.0)


REEVAL = "experiments/E3/0071_e3_reeval_e5v2_hpcgpu07"  # 补评目录（DEV-0063；运行编号是假的）


def test_e3_prepend_reeval_dir_overrides_and_is_marked(tmp_path):
    """DEV-0063（D102-2）：4B 的 E3b 种子 2 被拒；补评目录里有按 e5-v2 补评的 E3b 种子 2 与机内配对的 E3b+C3 种子 2。--e3-prepend 把它放在
    4B 目录列表的最前面：两臂的种子 2 从补评目录读（覆盖原来 E3b+C3 种子 2 的读数），表注与干扰图注写明 e5-v2；不给时与原来相同。"""
    root = build_tree(tmp_path / "tree")
    spec = next(s for s in MPF.E3_SIZES if s["size"] == "4B")
    new_vals = {("e3b", "refseg"): 0.71, ("e3b_c3", "refseg"): 0.55, ("e3b", "depth"): 0.21, ("e3b_c3", "depth"): 0.26}
    for (arm, task), v in new_vals.items():
        gdir = root / REEVAL / "eval" / arm / spec["model"] / "seed2"
        mc = _merge_check(True, "NVIDIA A100-PCIE-40GB")
        mc["criteria_version"] = "e5-v2"
        _w(gdir / "merge_check.json", mc)
        m = _grid_metrics(task, v)
        m["merge_criteria"] = {"version": "e5-v2"}
        _w(gdir / f"{task}__structured" / "metrics.json", m)
    sizes = MPF.prepend_dirs(MPF.E3_SIZES, {"4B": [REEVAL]})
    blocks = [ITF.size_block(s["size"], s["machine"], s["model"], [str(root / d) for d in s["dirs"]])
              for s in sizes if s["size"] != "9B"]  # 干扰的 JSON 按同样的目录顺序重出（模块说明）
    _w(root / MPF.INTERFERENCE_JSON, {"blocks": blocks, "trend": {"L": "随尺寸缩小"}, "relative_trend_descriptive": {}})
    out = tmp_path / "out"
    assert MPF.main(["--root", str(root), "--out", str(out), "--e3-prepend", f"4B={REEVAL}"]) == 0
    t = data(out, "tab_e3_comparison")
    s4 = next(s for s in t["sizes"] if s["size"] == "4B")
    assert s4["dirs"][0] == REEVAL and s4["prepended"] == [REEVAL]
    vals = [grid_value("4B", "e3b", s, "refseg") for s in (0, 1)] + [0.71]
    cell = s4["rows"]["e3b"]["refseg"]
    assert cell["n"] == 3 and cell["mean"] == pytest.approx(statistics.mean(vals))
    assert s4["rows"]["e3b_c3"]["refseg"]["seeds"]["2"] == pytest.approx(0.55)  # 机内配对的另一臂也从补评目录读
    assert t["rejected"] == [] and not any("4B e3b seed 2" in g for g in t["gaps"])
    assert sorted((r["arm"], r["task"]) for r in t["non_v1_grid"]) == sorted(new_vals)
    assert {(r["size"], r["seed"], r["criteria"], r["gpu"]) for r in t["non_v1_grid"]} == {("4B", 2, "e5-v2", "NVIDIA A100-PCIE-40GB")}
    md = (out / "tab_e3_comparison.md").read_text(encoding="utf-8")
    assert "re-evaluated under criteria e5-v2" in md and "4B E3b seed 2 (refseg/depth; criteria e5-v2" in md
    assert "4B E3b+C3 seed 2 (refseg/depth; criteria e5-v2" in md and "kept as v1 per D102-2" not in md
    d = data(out, "fig_interference_size")
    b4 = next(b for b in d["blocks"] if b["size"] == "4B")
    assert b4["P_seg"]["2"] == pytest.approx(0.71 - 0.55) and b4["P_depth"]["2"] == pytest.approx(0.26 - 0.21)
    assert b4["pair_criteria"]["refseg"]["2"] == "e5-v2" and "mismatched_criteria" not in b4
    assert all(c.startswith("一致") for c in d["checks"])
    cap = (out / "fig_interference_size_caption.tex").read_text(encoding="utf-8")
    assert "re-evaluated under merge-check criteria e5-v2, both arms (see Table E3): 4B seed 2" in cap
    assert "4B P\\_seg: 2 of 3 seeds" not in cap
    assert f"补评目录（--e3-prepend，排在最前、先到先得）`{REEVAL}`" in (out / "README.md").read_text(encoding="utf-8")
    # 不给 --e3-prepend：JSON 是按补评目录重出的，重算对不上 → 核对报不一致（提醒两边的目录顺序要一致）
    out2 = tmp_path / "out2"
    assert MPF.main(["--root", str(root), "--out", str(out2)]) == 0
    assert any(c.startswith("不一致：4B") for c in data(out2, "fig_interference_size")["checks"])
    t2 = data(out2, "tab_e3_comparison")
    assert "non_v1_grid" not in t2 and [(r["size"], r["arm"], r["seed"]) for r in t2["rejected"]] == [("4B", "e3b", 2)]
    # 用法错误
    for bad in ("4B", f"5B={REEVAL}", "4B=experiments/E3/nope"):
        with pytest.raises(SystemExit) as e:
            MPF.main(["--root", str(root), "--out", str(tmp_path / "o3"), "--e3-prepend", bad])
        assert e.value.code == 2


def test_size_value_parses_labels():
    assert MPF.size_value("0.8B") == 0.8 and MPF.size_value("9B") == 9.0
    with pytest.raises(ValueError):
        MPF.size_value("big")


# ---------------------------------------------------------------------------------------------
# DEV-0071：W2 一节（test 主表、尺寸、语言代价、计数分档）与干扰图 (d) 的离群读数。用仓库里的真数据出一遍。
# ---------------------------------------------------------------------------------------------

# docs/figures/ 现在是这样出的（README 的 E3 节：补评目录 0071 排在 2B、4B 的最前）；原有输出逐字节比对用同一组参数
REAL_ARGS = ["--e3-prepend", "2B=experiments/E3/0071_e3_reeval_e5v2_hpcgpu07",
             "--e3-prepend", "4B=experiments/E3/0071_e3_reeval_e5v2_hpcgpu07"]
W2_FIGS = ("fig_w2_size", "fig_lang_cost", "fig_count_bins")
W2_TABS = ("tab_w2_test", "tab_w2_test_giou", "tab_main", "tab_lang_cost", "tab_lang_cost_full", "tab_compute")  # 后四张：DEV-0073
# DEV-0073：E1、E2、E3、干扰图与 fig_w2_size（任务书没有要改它）与 docs/figures/ 里 DEV-0071 的输出逐字节相同
UNCHANGED = ["fig_e1_gap.pdf", "fig_e1_gap.png", "fig_e1_gap_caption.tex", "data/fig_e1_gap.json",
             "fig_e2_layers.pdf", "fig_e2_layers.png", "fig_e2_layers_caption.tex", "data/fig_e2_layers.json",
             "tab_e3_comparison.tex", "tab_e3_comparison.md", "data/tab_e3_comparison.json",
             "fig_interference_size.pdf", "fig_interference_size.png", "fig_interference_size_caption.tex",
             "data/fig_interference_size.json",
             "fig_w2_size.pdf", "fig_w2_size.png", "fig_w2_size_caption.tex", "data/fig_w2_size.json"]
SPLITS = ["test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA", "test_refcoco+_unc_testB",
          "test_refcocog_umd_test"]


def _repo_json(rel: str) -> dict:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    if not (REPO / MPF.W2_TEST_JSON).is_file():
        pytest.skip("仓库里没有 W2 的分析 JSON")
    out = tmp_path_factory.mktemp("real")
    assert MPF.main(["--out", str(out)] + REAL_ARGS) == 0
    return out


def test_split_labels_and_conventions():
    assert list(MPF.W2T.SPLITS) == SPLITS  # 划分与目录的约定取自 scripts/mgr/w2_test_table.py
    assert MPF.split_label("test_refcoco+_unc_testA") == "RefCOCO+ testA"
    assert MPF.split_label("test_refcocog_umd_test", short=True) == "refcocog test"
    assert list(MPF.CE.COARSE) == ["2–4", "5–7", "8–10"]
    with pytest.raises(ValueError):
        MPF.split_label("val_refcoco")


def test_real_new_outputs_written(real_run):
    from PIL import Image

    out = real_run
    for name in W2_FIGS:
        pdf, png = out / f"{name}.pdf", out / f"{name}.png"
        assert pdf.read_bytes()[:5] == b"%PDF-" and b"/Type3" not in pdf.read_bytes()
        with Image.open(png) as im:
            assert abs(im.info["dpi"][0] - MPF.DPI) < 1 and im.size[0] == round(MPF.SINGLE_COL_IN * MPF.DPI)
        assert (out / f"{name}_caption.tex").read_text(encoding="utf-8").count("\\caption{") == 1
        assert (out / "data" / f"{name}.json").is_file()
    for name in W2_TABS:
        tex = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert "\\toprule" in tex and "\\bottomrule" in tex and tex.count("\\caption{") == 1
        assert (out / f"{name}.md").is_file()
    for name in ("tab_w2_test", "tab_main", "tab_lang_cost", "tab_compute"):
        assert (out / "data" / f"{name}.json").is_file()
    readme = (out / "README.md").read_text(encoding="utf-8")
    for sec in ("## 5. W2 test 主表", "## 6. W2-a 随尺寸", "## 7. 语言代价", "## 8. 计数分档", "## 9. 附录的全 5,000 句表", "## 10. 语言代价四方对照表",
                "## 11. 计算记账表"):
        assert sec in readme
    assert readme.split("## 缺的数据（missing）")[1].split("\n## ")[0].strip() == "- 无"  # W2-a 的合并读数已在（RUN-0095）
    for name in W2_FIGS + W2_TABS:
        assert f"`{name}." in readme.split("## 文件")[1]
    # 来源（文件与运行编号）写进图注、表注与 README
    for name, runs in (("fig_w2_size", ["RUN-0072", "RUN-0073", "RUN-0081"]),
                       ("fig_lang_cost", ["RUN-0069", "RUN-0074", "RUN-0076", "RUN-0080", "RUN-0084", "RUN-0091"]),
                       ("fig_count_bins", ["RUN-0056", "RUN-0069", "RUN-0076", "RUN-0091"])):
        cap = (out / f"{name}_caption.tex").read_text(encoding="utf-8")
        assert all(r in cap for r in runs), name
    for name, runs in (("tab_w2_test", ["RUN-0072", "RUN-0076", "RUN-0078", "RUN-0092", "RUN-0095"]),
                       ("tab_main", ["RUN-0072", "RUN-0078", "RUN-0092", "RUN-0095"]), ("tab_compute", ["RUN-0072", "RUN-0078", "RUN-0092"]),
                       ("tab_lang_cost", ["RUN-0056", "RUN-0069", "RUN-0074", "RUN-0076", "RUN-0080", "RUN-0084", "RUN-0091"])):
        tex = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert all(r in tex for r in runs), name
    for src in (MPF.W2_TEST_JSON, MPF.W2_REPLAY_RUN, MPF.W2_REPLAY_MD, MPF.LLMSEG_COMPARE, MPF.LANG_COST_JSON, MPF.COUNT_ERRORS_JSON,
                MPF.X_TEST_JSON, MPF.W2_POOLED_JSON):
        assert src in readme
    # 逐划分的 gIoU 配对差与 LLMSeg 同句已由 5b3bb70ee 补进 x_test_table.json："分析里没有算的格子"一节不出现
    assert "## 分析里没有算的格子" not in readme


def test_real_existing_outputs_byte_identical(real_run):
    """仓库里 docs/figures/ 的每个文件（README 除外：生成时间与 commit）与用同一组参数重出的逐字节相同——提交的图表就是这份代码的输出。"""
    files = sorted(str(f.relative_to(REPO / "docs" / "figures")) for f in (REPO / "docs" / "figures").rglob("*") if f.is_file())
    assert set(UNCHANGED) <= set(files) and "tab_unified.tex" in files and "fig_bins.png" in files
    for f in files:
        if f != "README.md":
            assert (real_run / f).read_bytes() == (REPO / "docs" / "figures" / f).read_bytes(), f


PRE_DEV0077 = "a29e96c81"  # DEV-0077 动手前的 origin/main：那时 docs/figures/ 里的图表除 tab_main 的 caption 外都不该变
PRE_DEV0078 = "dfa4383e7"  # DEV-0078 动手前的 origin/main：除下面点名的几个外，docs/figures/ 逐字节不变
DEV0078_CHANGED = ("tab_lang_cost.tex", "tab_lang_cost.md", "tab_lang_cost_full.tex", "tab_lang_cost_full.md", "data/tab_lang_cost.json",
                   "fig_lang_cost.pdf", "fig_lang_cost.png", "fig_lang_cost_caption.tex", "data/fig_lang_cost.json",
                   "tab_seeds.tex", "tab_seeds.md")
DEV0078_NEW = ("tab_x2_4b.tex", "tab_x2_4b.md", "data/tab_x2_4b.json")


def _git_show(spec: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", spec], capture_output=True, timeout=60)
    return r.stdout if r.returncode == 0 else None


def test_real_outputs_unchanged_since_dev0077_base(real_run):
    """DEV-0077 任务书：已有图表除 tab_main 的 caption 外逐字节不变。对照 a29e96c81 里的 docs/figures/：其余文件逐字节相同；
    tab_main.tex 只有 caption 一行不同（新标题加一句"附录的全 5,000 句版，正文用 tab_unified"，原来的表注原样接在后面）；
    tab_main.md 只有标题一行不同、表注多了第一条。"""
    ls = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", PRE_DEV0077, "docs/figures"], capture_output=True,
                        text=True, timeout=60)
    if ls.returncode != 0 or not ls.stdout.strip():
        pytest.skip(f"没有 {PRE_DEV0077} 的 git 历史")
    files = [f.removeprefix("docs/figures/") for f in ls.stdout.split()]
    assert "tab_main.tex" in files and "fig_count_bins.png" in files and "tab_unified.tex" not in files
    for f in files:
        # DEV-0078 点名改的几个（四方表、语言代价图）由 test_real_outputs_unchanged_since_dev0078_base 对照 dfa4383e7 核对
        if f not in ("README.md", "tab_main.tex", "tab_main.md") + DEV0078_CHANGED:
            assert (real_run / f).read_bytes() == _git_show(f"{PRE_DEV0077}:docs/figures/{f}"), f
    old = _git_show(f"{PRE_DEV0077}:docs/figures/tab_main.tex").decode("utf-8").splitlines()
    new = (real_run / "tab_main.tex").read_text(encoding="utf-8").splitlines()
    diff = [(a, b) for a, b in zip(old, new) if a != b]
    assert len(old) == len(new) and len(diff) == 1
    a, b = diff[0]
    assert a.startswith("\\caption{Main result") and b.startswith("\\caption{Full W2 test, all 5,000 expressions (appendix")
    assert "Table~\\ref{tab:unified}" in b and b.endswith(a.split(". ", 1)[1])  # 原来的表注一字不改，接在新的第一句后面
    old = _git_show(f"{PRE_DEV0077}:docs/figures/tab_main.md").decode("utf-8").splitlines()
    new = (real_run / "tab_main.md").read_text(encoding="utf-8").splitlines()
    assert new[0] != old[0] and "appendix" in new[0]
    added = [x for x in new[1:] if x not in old]
    assert len(added) == 1 and added[0].startswith("- Appendix version on all 5,000 W2 test expressions") and "tab_unified" in added[0]
    assert [x for x in new[1:] if x != added[0]] == old[1:]


RANGE_HEAD = "C1 = X-2, range over 3 seeds"


def _drop_col(line: str, i: int, sep: str) -> str:
    """表的一行去掉第 i 格（Markdown 用 " | "，LaTeX 用 " & "）。"""
    if sep == " | ":
        cells = line.strip().strip("|").split("|")
        return "|" + "|".join(c for k, c in enumerate(cells) if k != i) + "|"
    body = line.removesuffix(" \\\\")
    cells = body.split(sep)
    return sep.join(c for k, c in enumerate(cells) if k != i) + " \\\\"


def test_real_outputs_unchanged_since_dev0078_base(real_run):
    """DEV-0078 任务书：其余图表除点名改动的以外逐字节不变。对照 dfa4383e7 的 docs/figures/：
    - 没点名的文件逐字节相同；新增的只有 tab_x2_4b 三个文件；
    - 四方表（与 _full）：每一行去掉"3 个种子的范围"那一列后与原来相同（.md 与 .tex），列格式多一个 c；表注原有各条一字不改（Data 一条只在末尾
      加种子范围的来源），另多两条；
    - tab_seeds：只多了表注的种子稳定性一条（.md 一行；.tex 的 caption 里同一句），表的数值与版式不变；
    - data/tab_lang_cost.json、data/fig_lang_cost.json 只多了 seed_range 一个键。"""
    ls = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", PRE_DEV0078, "docs/figures"], capture_output=True,
                        text=True, timeout=60)
    if ls.returncode != 0 or not ls.stdout.strip():
        pytest.skip(f"没有 {PRE_DEV0078} 的 git 历史")
    files = [f.removeprefix("docs/figures/") for f in ls.stdout.split()]
    assert set(DEV0078_CHANGED) <= set(files) and not set(DEV0078_NEW) & set(files)
    now = sorted(str(f.relative_to(real_run)) for f in real_run.rglob("*") if f.is_file())
    assert sorted(set(now) - set(files)) == sorted(DEV0078_NEW)
    for f in files:
        if f not in ("README.md",) + DEV0078_CHANGED:
            assert (real_run / f).read_bytes() == _git_show(f"{PRE_DEV0078}:docs/figures/{f}"), f
    for name in ("tab_lang_cost", "tab_lang_cost_full"):
        old = _git_show(f"{PRE_DEV0078}:docs/figures/{name}.md").decode("utf-8").splitlines()
        new = (real_run / f"{name}.md").read_text(encoding="utf-8").splitlines()
        head = _cells(new[2])
        i = head.index(RANGE_HEAD)
        assert head[i - 1] == "C1 = X-2 (ours)"
        old_rows, new_rows = [x for x in old if x.startswith("|")], [x for x in new if x.startswith("|")]
        assert len(old_rows) == len(new_rows) == 9
        assert new_rows[1] == old_rows[1] + "---:|"
        for a, b in zip(old_rows[:1] + old_rows[2:], new_rows[:1] + new_rows[2:]):
            assert _drop_col(b, i, " | ") == a, b
        old_notes, new_notes = [x for x in old if x.startswith("- ")], [x for x in new if x.startswith("- ")]
        assert old_notes[-1].startswith("- Data: ") and new_notes[-1].startswith(old_notes[-1][:-1] + "; the 3-seed range: seed_variance.json (")
        added = [x for x in new_notes if x not in old_notes[:-1] and not x.startswith("- Data: ")]
        assert [x for x in new_notes[:-1] if x not in added] == old_notes[:-1] and len(added) == 2
        assert added[0].startswith("- C1 = X-2 (ours) is training seed 0") and added[1].startswith("- Language cost is read on the language battery")
        assert new[0] == old[0]
        # .tex：表的每一行去掉同一列后与原来相同；列格式多一个 c；caption 一行之外都相同
        old_t = _git_show(f"{PRE_DEV0078}:docs/figures/{name}.tex").decode("utf-8").splitlines()
        new_t = (real_run / f"{name}.tex").read_text(encoding="utf-8").splitlines()
        assert len(old_t) == len(new_t)
        for a, b in zip(old_t, new_t):
            if a.startswith("\\caption{"):
                assert b.startswith("\\caption{") and "range over 3 seeds" in b
            elif a.startswith("\\begin{tabular}"):
                assert b == a.replace("c}", "cc}")
            elif " & " in a:
                assert _drop_col(b, i, " & ") == a, b
            else:
                assert a == b
    old = _git_show(f"{PRE_DEV0078}:docs/figures/tab_seeds.md").decode("utf-8").splitlines()
    new = (real_run / "tab_seeds.md").read_text(encoding="utf-8").splitlines()
    added = [x for x in new if x not in old]
    assert len(added) == 1 and added[0].startswith("- Segmentation on the development data is almost the same for the 3 seeds")
    assert [x for x in new if x != added[0]] == old
    old_t = _git_show(f"{PRE_DEV0078}:docs/figures/tab_seeds.tex").decode("utf-8").splitlines()
    new_t = (real_run / "tab_seeds.tex").read_text(encoding="utf-8").splitlines()
    diff = [(a, b) for a, b in zip(old_t, new_t) if a != b]
    assert len(old_t) == len(new_t) and len(diff) == 1 and diff[0][0].startswith("\\caption{")
    a, b = diff[0]
    sentence = MPF._tex_escape(added[0][2:])
    assert b.replace(" " + sentence, "", 1) == a  # caption 里只多了这一句
    for name in ("tab_lang_cost", "fig_lang_cost"):
        old_j = json.loads(_git_show(f"{PRE_DEV0078}:docs/figures/data/{name}.json"))
        new_j = data(real_run, name)
        assert set(new_j) - set(old_j) == {"seed_range"} and {k: v for k, v in new_j.items() if k != "seed_range"} == old_j, name


def test_real_w2_table_matches_analysis_json(real_run):
    out = real_run
    d = data(out, "tab_w2_test")
    j = _repo_json(MPF.W2_TEST_JSON)["models"]["qwen3.5-9b"]
    for metric in ("cIoU", "gIoU"):
        for key, _lab in MPF.W2_ROWS:
            for sp in SPLITS:
                p = d["rows"][metric][key][sp]
                if key == "replay":
                    m = _repo_json(f"{MPF.W2_REPLAY_RUN}/qwen3.5-9b/refseg/w2c/{sp}/metrics.json")
                    h = next(h for h in m["headline"] if h["name"] == metric)
                    assert (p["value"], p["low"], p["high"]) == (h["value"], h["ci"]["low"], h["ci"]["high"])
                else:
                    c = j["per_split"][sp][key][metric]
                    assert (p["value"], p["low"], p["high"]) == (c["value"], c["ci"][0], c["ci"][1])
    for key in ("w2c-w1f", "w2a-w1f"):
        for sp in SPLITS:
            p, c = d["pairs"][key][sp], j["pairs"][key][sp]
            assert (p["value"], p["low"], p["high"], p["n"]) == (c["diff"], c["ci"][0], c["ci"][1], 1000)
        p, c = d["pairs"][key]["pooled"], j["pairs_pooled"][key]
        assert (p["value"], p["low"], p["high"], p["n"]) == (c["diff"], c["ci"][0], c["ci"][1], 5000)
    # 回放 − W2-c：docs/analysis/w2c_replay.md §1（抽查两格）
    rp = d["pairs"]["replay-w2c"]
    assert (rp["pooled"]["value"], rp["pooled"]["low"], rp["pooled"]["high"], rp["pooled"]["n"]) == (0.001, -0.005, 0.007, 5000)
    assert (rp["test_refcoco+_unc_testA"]["value"], rp["test_refcoco+_unc_testA"]["low"]) == (0.013, 0.002)
    # LLMSeg：读数与 CI 取叶目录，合并读数与配对差取 compare.json
    cmp_ = _repo_json(MPF.LLMSEG_COMPARE)
    for metric in ("cIoU", "gIoU"):
        for sp in SPLITS + ["pooled"]:
            blk = cmp_["pooled"] if sp == "pooled" else cmp_["splits"][sp]
            for rd in ("rec", "seg"):
                c = blk[rd][metric]
                p = d["llmseg"]["pairs"][metric][rd][sp]
                assert (p["value"], p["low"], p["high"]) == (c["diff"], c["ci"]["low"], c["ci"]["high"])
                assert d["llmseg"]["rows"][metric][rd][sp]["value"] == pytest.approx(c["a"], abs=1e-12)
            assert d["llmseg"]["rows"][metric]["w2c"][sp]["value"] == blk["rec"][metric]["b"]
        leaf = _repo_json(f"{MPF.LLMSEG_RUN}/qwen3.5-9b/refseg/llmseg/{SPLITS[0]}/metrics.json")
        h = next(h for h in leaf["readouts"]["seg"]["headline"] if h["name"] == metric)
        assert d["llmseg"]["rows"][metric]["seg"][SPLITS[0]]["low"] == h["ci"]["low"]
    assert all(c.startswith("一致") for c in d["checks"])
    # 抽查表里的格子（×100，一位小数）
    md = (out / "tab_w2_test.md").read_text(encoding="utf-8")
    w2c = j["per_split"]["test_refcoco_unc_testA"]["w2c"]["cIoU"]
    assert f"| W2-c | {100 * w2c['value']:.1f} [{100 * w2c['ci'][0]:.1f}, {100 * w2c['ci'][1]:.1f}] |" in md
    pool = j["pairs_pooled"]["w2c-w1f"]
    assert f"{100 * pool['diff']:+.1f} [{100 * pool['ci'][0]:+.1f}, {100 * pool['ci'][1]:+.1f}] |" in md
    assert "| W2-c + replay − W2-c |" in md and "+0.1 [-0.5, +0.7] |" in md
    rec = cmp_["pooled"]["rec"]["cIoU"]
    assert f"{100 * rec['diff']:+.1f} [{100 * rec['ci']['low']:+.1f}, {100 * rec['ci']['high']:+.1f}] |" in md
    assert "| LLMSeg, before refinement |" in md and "| LLMSeg, after refinement (main) |" in md
    tex = (out / "tab_w2_test.tex").read_text(encoding="utf-8")
    assert f"${100 * w2c['value']:.1f}$\\,{{\\scriptsize $[{100 * w2c['ci'][0]:.1f}, {100 * w2c['ci'][1]:.1f}]$}}" in tex
    giou = (out / "tab_w2_test_giou.md").read_text(encoding="utf-8")
    assert "| LLMSeg after − W2-c |" in giou
    assert "| W2-c − W1-F |" not in giou and "| C1 − W2-c |" in giou  # gIoU 的配对差只有候选的四行（W2 的只算了 cIoU）
    assert "all 1,000 outputs per split fail to parse" in md
    assert "Per-system pooled cIoU is not part of the analysis" not in md and "from w2_test_pooled.json (10,000 resamples)" in md


def _fmt(v, signed=False):
    return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"


def _cell(v, lo, hi, signed=False):
    return f"{_fmt(v, signed)} [{_fmt(lo, signed)}, {_fmt(hi, signed)}]"


def test_real_w2_table_candidates_and_pooled_column(real_run):
    """DEV-0073：候选三行（逐划分取 RUN-0092 叶目录 metrics.json）、合并列（w2_test_pooled.json）、候选的配对差与 LLMSeg 同句（x_test_table.json）。"""
    out = real_run
    d = data(out, "tab_w2_test")
    xj, pj = _repo_json(MPF.X_TEST_JSON), _repo_json(MPF.W2_POOLED_JSON)
    for metric in ("cIoU", "gIoU"):
        for key, cand, setting, jkey, _lab in MPF.X_ROWS:
            for sp in SPLITS:
                m = _repo_json(f"{MPF.XT.CANDS[cand]}/qwen3.5-9b/refseg/{setting}/{sp}/metrics.json")
                h = next(h for h in m["headline"] if h["name"] == metric)
                p = d["rows"][metric][key][sp]
                assert (p["value"], p["low"], p["high"], p["n_resamples"]) == (h["value"], h["ci"]["low"], h["ci"]["high"], 1000)
        # 合并列：每一行都取 w2_test_pooled.json（名字按运行目录与设置对上）
        for key in d["rows"][metric]:
            e = pj["systems"][MPF.pooled_name(key)]
            p = d["rows"][metric][key]["pooled"]
            c = e[metric]
            assert (p["value"], p["low"], p["high"], p["n"], p["n_resamples"]) == (c[metric.lower()], c["ci"][0], c["ci"][1], 5000, 10000)
        # 配对差：x_test_table.json 的 main_minus_*（gIoU 取 g_ 键）
        pre = "" if metric == "cIoU" else "g_"
        for pk, cand, jk, _lab in MPF.X_PAIRS:
            c = xj["candidates"][cand]["pooled"][pre + jk]
            p = d["x_pairs"][metric][pk]["pooled"]
            assert (p["value"], p["low"], p["high"], p["n"]) == (c["diff"], c["ci"][0], c["ci"][1], 5000)
    for sp in SPLITS:  # 逐划分：cIoU 与 gIoU（gIoU 的 C1 − W1-F、LLMSeg 同句与 C1 − LLMSeg 由 5b3bb70ee 补上）
        blk = xj["candidates"][MPF.C1]["per_split"][sp]
        assert d["x_pairs"]["cIoU"]["c1-w2c"][sp]["value"] == blk["main_minus_w2c"]["diff"]
        p, c = d["x_pairs"]["gIoU"]["c1-w1f"][sp], blk["g_main_minus_w1f"]
        assert (p["value"], p["low"], p["high"], p["n"]) == (c["diff"], c["ci"][0], c["ci"][1], 1000)
        for k, jk in (("c1", "g_main_on_subset"), ("c1-rec", "g_main_minus_llmseg_rec"), ("c1-seg", "g_main_minus_llmseg_seg")):
            part = "rows" if k == "c1" else "pairs"
            p, c = d["llmseg"][part]["gIoU"][k][sp], blk["llmseg"][jk]
            assert (p["value"], p["low"], p["high"], p["n"]) == (c.get("giou", c.get("diff")), c["ci"][0], c["ci"][1], 300)
    assert d["gaps"] == []
    assert all(c.startswith("一致") for c in d["checks"])
    md = (out / "tab_w2_test.md").read_text(encoding="utf-8")
    # 抽查：C1 合并、C1 − W2-c 合并、C1 − 椭圆消融合并（×100，一位小数）
    c1 = pj["systems"]["C1 = X-2（主方法）"]["cIoU"]
    assert f"| C1 = X-2 (ours) |" in md and f"{_cell(c1['ciou'], *c1['ci'])} |" in md
    for lab, jk in (("C1 − W2-c", "main_minus_w2c"), ("C1 − in-box ellipse ablation", "main_minus_ablation")):
        c = xj["candidates"][MPF.C1]["pooled"][jk]
        row = next(r for r in md.splitlines() if r.startswith(f"| {lab} |"))
        assert row.endswith(f"| {_cell(c['diff'], *c['ci'], signed=True)} |"), row
    ll = xj["candidates"][MPF.C1]["pooled"]["llmseg"]
    row = next(r for r in md.splitlines() if r.startswith("| C1 − LLMSeg after |"))
    assert row.endswith(f"| {_cell(ll['main_minus_llmseg_rec']['diff'], *ll['main_minus_llmseg_rec']['ci'], signed=True)} |")
    giou = (out / "tab_w2_test_giou.md").read_text(encoding="utf-8")
    for lab, path in (("C1 − W1-F", ("g_main_minus_w1f",)), ("C1 − LLMSeg after", ("llmseg", "g_main_minus_llmseg_rec")),
                      ("C1 = X-2, same expressions", ("llmseg", "g_main_on_subset"))):
        row = next(r for r in giou.splitlines() if r.startswith(f"| {lab} |"))
        want = []
        for sp in SPLITS + ["pooled"]:
            blk = xj["candidates"][MPF.C1]["pooled"] if sp == "pooled" else xj["candidates"][MPF.C1]["per_split"][sp]
            for k in path:
                blk = blk[k]
            want.append(_cell(blk.get("giou", blk.get("diff")), *blk["ci"], signed="giou" not in blk))
        assert row == f"| {lab} | " + " | ".join(want) + " |", lab
    assert "—" not in giou.split("\n\n- ")[0] and "not computed in the analysis" not in giou


def test_real_tab_main_matches_analysis_json(real_run):
    """正文主表：逐划分只给读数、合并带 CI（w2_test_pooled.json）、末列每句调用 / 生成 token（x_test_table.json 的 compute），
    外部基线一块取 x_test_table.json 的 pooled.llmseg；不加粗（DEV-0073 审查），表注写明 C1 是事先定的主方法与 C2 − C1 对照 X-6 的预测。"""
    out = real_run
    d = data(out, "tab_main")
    xj, pj = _repo_json(MPF.X_TEST_JSON), _repo_json(MPF.W2_POOLED_JSON)
    md = (out / "tab_main.md").read_text(encoding="utf-8")
    tex = (out / "tab_main.tex").read_text(encoding="utf-8")
    lines = md.splitlines()
    keys = [k for g in MPF.MAIN_ROWS for k in g]
    assert [r for r in lines if r.startswith("| ") and not r.startswith("| *") and not r.startswith("| System")][:len(keys)] == \
           [next(r for r in lines if r.startswith(f"| {MPF.ROW_LABELS[k]} |")) for k in keys]
    pooled = {k: pj["systems"][MPF.pooled_name(k)]["cIoU"] for k in keys}
    comp ={"w1f": xj["compute_reference"]["W1-F"], "w2c": xj["compute_reference"]["W2-c"],
            "c1": xj["candidates"][MPF.C1]["compute"]["main"], "c1_ellipse": xj["candidates"][MPF.C1]["compute"]["ablation"],
            "c2": xj["candidates"][MPF.C2]["compute"]["main"]}
    for k in keys:
        row = next(r for r in lines if r.startswith(f"| {MPF.ROW_LABELS[k]} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) == 8
        for sp, c in zip(SPLITS, cells[1:6]):
            assert c == _fmt(d["rows"][k][sp]["value"]) and "[" not in c  # 逐划分只有读数
        v = pooled[k]
        assert cells[6] == _cell(v["ciou"], *v["ci"])  # 不加粗
        if k in comp:
            calls = f"{comp[k]['n_calls']:.1f}".removesuffix(".0")
            assert cells[7] == f"{calls} / {comp[k]['generated_tokens']:,.0f}"
        else:
            assert cells[7] == "—"
    assert "**" not in md and "\\best" not in tex and "\\textbf" not in tex and "Bold:" not in md
    # C1 合并（w2_test_pooled.json）；表注：C1 事先定为主方法、C2 是简化版、合并的 C2 − C1 在 X-6 预测的 ±2 点以内
    c1 = pj["systems"]["C1 = X-2（主方法）"]["cIoU"]
    c2 = pj["systems"]["C2 = X-3（消融）"]["cIoU"]
    assert d["rows"]["c1"]["pooled"]["value"] == c1["ciou"]
    gap = 100 * (c2["ciou"] - c1["ciou"])
    assert 0 < gap <= 100 * MPF.X6_C1_C2_MARGIN
    assert ("- C1 was designated the main method before any test run (preregistration X-5a, the project lead's decision); C2 is C1's "
            f"simplified version with the box-correction stream removed; pooled, C2 is {gap:.1f} points above C1, within the ±2 points "
            "predicted in preregistration X-6.") in md
    # 外部基线：同样 1,500 句
    ll = xj["candidates"][MPF.C1]["pooled"]["llmseg"]
    ext = d["external"]
    for key, src in (("llmseg_rec", ll["llmseg_rec"]), ("c1", ll["main_on_subset"]), ("c1-rec", ll["main_minus_llmseg_rec"])):
        assert (ext[key]["value"], ext[key]["low"], ext[key]["high"], ext[key]["n"]) == \
               (src.get("ciou", src.get("diff")), src["ci"][0], src["ci"][1], 1500)
    row = next(r for r in lines if r.startswith("| LLMSeg, after refinement (main) |"))
    lr = xj["compute_reference"]["LLMSeg（1,500 句）"]
    assert row == (f"| LLMSeg, after refinement (main) | | | | | | {_cell(ll['llmseg_rec']['ciou'], *ll['llmseg_rec']['ci'])} | "
                   f"{lr['n_calls']:.1f} / {lr['generated_tokens']:,.0f} |")
    row = next(r for r in lines if r.startswith("| C1 − LLMSeg, after refinement (points) |"))
    assert _cell(ll["main_minus_llmseg_rec"]["diff"], *ll["main_minus_llmseg_rec"]["ci"], signed=True) in row
    # 表注：C1 是什么、消融改了什么、配对 bootstrap 的口径、运行编号
    for s in ("a single LoRA", "16 control points", "seeds GrabCut as a trimap", "no learned parameters", "inscribed in the proposed box",
              "without the signed box-correction stream", "per-expression paired bootstrap of the ratio of sums, 95%, 10,000 resamples",
              "RUN-0092 (C1, ellipse ablation, C2)", "no compute record"):
        assert s in md, s


def test_real_tab_compute_matches_analysis_json(real_run):
    md = (real_run / "tab_compute.md").read_text(encoding="utf-8")
    xj = _repo_json(MPF.X_TEST_JSON)
    src = {"W1-F": xj["compute_reference"]["W1-F"], "W2-c": xj["compute_reference"]["W2-c"],
           MPF.ROW_LABELS["c1"]: xj["candidates"][MPF.C1]["compute"]["main"],
           MPF.ROW_LABELS["c1_ellipse"]: xj["candidates"][MPF.C1]["compute"]["ablation"],
           MPF.ROW_LABELS["c2"]: xj["candidates"][MPF.C2]["compute"]["main"],
           "LLMSeg (full pipeline)": xj["compute_reference"]["LLMSeg（1,500 句）"]}
    rows = [r for r in md.splitlines() if r.startswith("| ") and not r.startswith("| System")]
    assert [r.split("|")[1].strip() for r in rows] == list(src)
    for r in rows:
        lab = r.split("|")[1].strip()
        c = src[lab]
        want = [f"{c['n']:,}", f"{c['n_calls']:,.2f}", f"{c['generated_tokens']:,.1f}", f"{c['prompt_tokens']:,.0f}", f"{c['call_time_s']:,.2f}",
                "—" if lab.startswith("LLMSeg") else f"{c['wall_time_s']:,.2f}"]
        assert [x.strip() for x in r.strip("|").split("|")[1:]] == want, lab
    assert "accumulated under concurrent requests" in md


def test_real_tab_lang_cost_matches_analysis_json(real_run):
    """四方表：E3b+C3、W2-c、W2-c+回放、X-2；附录版另加 X-3。每格与 lang_cost_9b.json 逐项一致；基座读数按节点分开写。"""
    out = real_run
    j = _repo_json(MPF.LANG_COST_JSON)["systems"]
    md, full = (out / "tab_lang_cost.md").read_text(encoding="utf-8"), (out / "tab_lang_cost_full.md").read_text(encoding="utf-8")
    head = md.splitlines()[2]
    assert head == f"| Check | Reading | Base | E3b+C3 | W2-c | W2-c + replay | C1 = X-2 (ours) | {RANGE_HEAD} |"
    assert full.splitlines()[2] == head + " C2 = X-3 (ablation) |"  # 附录版：范围一列也紧跟 C1，在 C2 之前
    lt = data(out, "tab_lang_cost")
    x2 = next(s for s in lt["systems"] if s["name"] == "X-2（C1）")
    # 抽查：X-2 的大题量 MMLU-Pro 与计数（C1 列，第 7 格）
    for sec, key in (("battery", "mmlu_pro_letter.accuracy"), ("ttext", "count")):
        r, c = j["X-2（C1）"][sec][key], x2["cells"][f"{sec}/{key}"]
        assert (c["delta"], c["low"], c["high"], c["n"]) == (r["delta_pp"], r["ci95_pp"][0], r["ci95_pp"][1], r["n"])
    row = next(r for r in md.splitlines() if "MMLU-Pro, letter answers" in r)
    r = j["X-2（C1）"]["battery"]["mmlu_pro_letter.accuracy"]
    assert _cells(row)[6] == f"{r['delta_pp']:+.1f} [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]"
    row = next(r for r in md.splitlines() if "CountBench counting accuracy" in r)
    r = j["X-2（C1）"]["ttext"]["count"]
    assert _cells(row)[6] == f"{r['delta_pp']:+.1f} [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}]"
    # 健康检查：差与标红（†）；X-2 没有标红
    health = [r for r in md.splitlines() if r.startswith("|") and r.split("|")[2].strip().split(" (")[0] in ("MMLU-Pro", "GSM8K", "IFEval")]
    assert len(health) == 3
    for line in health:
        bench = {"MMLU-Pro": "mmlu_pro_letter.accuracy", "GSM8K": "gsm8k.flexible", "IFEval": "ifeval.prompt_level_strict"}[
            line.split("|")[2].strip().split(" (")[0]]
        cells = [c.strip() for c in line.strip("|").split("|")][3:]
        for name, c in zip(("E3b+C3", "W2-c", "W2-c+回放", "X-2（C1）"), cells):
            h = j[name]["health"][bench]
            assert c == f"{h['delta_pp']:+.1f}" + ("†" if h["flag"] else ""), (name, bench)
    assert not any(c["flag"] for k, c in x2["cells"].items() if k.startswith("health/"))
    # 基座读数只取 lang_cost_9b.json（battery.*.base，5b3bb70ee 补上）；表里每格的基座 = JSON
    for s in lt["systems"]:
        for key in ("mmlu_pro_letter.accuracy", "ifeval.prompt_level_strict"):
            assert s["cells"][f"battery/{key}"]["base"] == j[s["name"]]["battery"][key]["base"] is not None
    # 核对（审查第 2 条：留作测试）：lang_cost_9b.json 的大题量基座、系统读数、差与 CI = 来源 lb_compare 文件的同一行
    n_chk = 0
    for name, e in j.items():
        rows = {(r["reading"], r["metric"]): r for r in _repo_json(e["sources"]["battery"])["rows"]}
        for key, r in e["battery"].items():
            lb = rows[tuple(key.split(".", 1))]
            assert (r["base"], r["other"], r["delta_pp"], r["ci95_pp"], r["n"]) == \
                   (lb["base_pct"], lb["other_pct"], lb["delta_pp"], lb["ci95_pp"], lb["n"]), (name, key)
            n_chk += 1
    assert n_chk == 3 * len(j)
    # 基座：MMLU-Pro 两个节点相同 → 一个数；IFEval 两个节点不同 → 按节点分开写
    b06, b07 = j["W2-c"]["battery"], j["X-2（C1）"]["battery"]
    k = "ifeval.prompt_level_strict"
    row = next(r for r in md.splitlines() if "IFEval, prompt-level strict" in r)
    assert f"| {b06[k]['base']:.1f} (hpcgpu06) / {b07[k]['base']:.1f} (hpcgpu07) |" in row
    k = "mmlu_pro_letter.accuracy"
    assert b06[k]["base"] == b07[k]["base"]
    assert f"| {b06[k]['base']:.1f} |" in next(r for r in md.splitlines() if "MMLU-Pro, letter answers" in r)
    assert "C1 = X-2 (ours): health check and language battery against the base on hpcgpu07, T-text against the base on hpcgpu06" in md
    assert not lt["missing"]
    assert "C2 = X-3" not in md.split("\n- ")[0] and "X-3" in full


def _seed_readings() -> dict:
    """seed_variance.json 各种子的读数（相对各自节点基座的差）：{"<节>/<读数>": [(种子, 差, 标红)]}，独立于出图脚本取。"""
    sj = _repo_json(MPF.SEED_JSON)["seeds"]
    out: dict = {}
    for k in sorted(sj, key=int):
        for sec, key in (("battery", "mmlu_pro_letter.accuracy"), ("battery", "ifeval.prompt_level_strict"), ("ttext", "refexp_box"),
                         ("ttext", "count")):
            out.setdefault(f"{sec}/{key}", []).append((int(k), sj[k]["language"][sec][key]["delta_pp"], False))
        for key, h in sj[k]["health"]["rows"].items():
            out.setdefault(f"health/{key}", []).append((int(k), h["delta_pp"], h["flag"]))
    return out


def _expected_range(rows, mean=False, scale=1.0, signed=True) -> str:
    """范围一格的期望写法：最小与最大，离 0 近的一端在前；计数加均值，健康检查加标红的种子。"""
    vals = [v for _k, v, _f in rows]
    f = (lambda x: f"{scale * x:+.1f}") if signed else (lambda x: f"{scale * x:.1f}")
    a, b = (f(x) for x in sorted((min(vals), max(vals)), key=abs))
    s = a if a == b else f"{a} to {b}"
    extra = ([f"mean {f(sum(vals) / len(vals))}"] if mean else []) + \
        ([f"† seeds {' and '.join(str(k) for k, _v, fl in rows if fl)}"] if any(fl for _k, _v, fl in rows) else [])
    return s + (f" ({'; '.join(extra)})" if extra else "")


TABLE_ROWS = {"health/mmlu_pro_letter.accuracy": "MMLU-Pro (n=100)", "health/gsm8k.flexible": "GSM8K (n=50)",
              "health/ifeval.prompt_level_strict": "IFEval (n=50)", "battery/mmlu_pro_letter.accuracy": "MMLU-Pro, letter answers (n=2,000)",
              "battery/ifeval.prompt_level_strict": "IFEval, prompt-level strict (n=541)", "ttext/refexp_box": "Grounding Acc@0.5 (n=56,857)",
              "ttext/count": "CountBench counting accuracy (n=442)"}


def test_real_tab_lang_cost_seed_range(real_run):
    """DEV-0078（D162）：四方表与附录版的"3 个种子的范围"一列 = seed_variance.json 各种子读数的最小与最大（离 0 近的一端在前），计数另给均值，
    健康检查另给标红的种子；钉住负责人答复 N46 的数（MMLU-Pro −0.2 到 −2.3、IFEval 0.0 到 −3.5、定位 +3.4 到 +3.5、计数 −8.1 到 −16.1、
    均值 −10.9）。C1 列是 seed 0（来源路径对上，读数相同）。表注写明其余各列单种子、以大题量为准、seed 1、2 在 50 条 IFEval 上标红、大题量上三个种子
    都没有超过 3.5 点。"""
    out = real_run
    sr = _seed_readings()
    j = _repo_json(MPF.LANG_COST_JSON)["systems"]["X-2（C1）"]
    s0 = _repo_json(MPF.SEED_JSON)["seeds"]["0"]
    # C1 列 = seed 0：来源相同、读数相同
    assert {k: s0["language"]["sources"][k] for k in ("battery", "ttext")} == {k: j["sources"][k] for k in ("battery", "ttext")}
    for sec in ("battery", "ttext"):
        for key, r in j[sec].items():
            if key in s0["language"][sec]:
                assert (s0["language"][sec][key]["delta_pp"], s0["language"][sec][key]["ci95_pp"]) == (r["delta_pp"], r["ci95_pp"]), key
    assert {k: v["delta_pp"] for k, v in s0["health"]["rows"].items()} == {k: v["delta_pp"] for k, v in j["health"].items()}
    lt = data(out, "tab_lang_cost")
    assert lt["seed_range"]["c1_seed"] == 0 and lt["seed_range"]["seeds"] == [0, 1, 2] and not lt["seed_range"]["missing"]
    # 计数的三个种子与 count_errors_c1_seeds.json 的 all 档一致（同一批预测文件的另一份分析）
    ce = _repo_json("docs/analysis/count_errors_c1_seeds.json")["systems"]
    assert [round(100 * ce[n]["bins"]["all"]["vs_base"]["d_acc"], 6) for n in ("X-2（C1）", "X-2（C1）seed 1", "X-2（C1）seed 2")] == \
           [round(v, 6) for _k, v, _f in sr["ttext/count"]]
    pins = {"battery/mmlu_pro_letter.accuracy": "-0.2 to -2.3", "battery/ifeval.prompt_level_strict": "+0.0 to -3.5",
            "ttext/refexp_box": "+3.4 to +3.5", "ttext/count": "-8.1 to -16.1 (mean -10.9)",
            "health/ifeval.prompt_level_strict": "-4.0 to -12.0 († seeds 1 and 2)"}
    for name in ("tab_lang_cost", "tab_lang_cost_full"):
        md = (out / f"{name}.md").read_text(encoding="utf-8")
        i = _cells(md.splitlines()[2]).index(RANGE_HEAD)
        for k, lab in TABLE_ROWS.items():
            cell = _cells(next(r for r in md.splitlines() if f"| {lab} |" in r))[i]
            assert cell == _expected_range(sr[k], mean=k == "ttext/count"), (name, k)
            if k in pins:
                assert cell == pins[k], (name, k)
        notes = md.split("\n\n- ")[1]
        assert "C1 = X-2 (ours) is training seed 0, the checkpoint in the test tables; every other column is a single training seed." in notes
        assert "the end nearer zero first" in notes and "seeds 0 and 1 on hpcgpu07, seed 2 on hpcgpu06" in notes
        flagged = [(k, v) for k, v, fl in sr["health/ifeval.prompt_level_strict"] if fl]
        assert [k for k, _v in flagged] == [1, 2]
        drop = max(-v for k in ("battery/mmlu_pro_letter.accuracy", "battery/ifeval.prompt_level_strict") for _s, v, _f in sr[k])
        assert drop == 3.5
        assert ("Language cost is read on the language battery (D154, D162); the health-check flags are reported alongside only: the health "
                f"check flags C1's seeds 1 and 2 on IFEval (n=50: {MPF._signed(flagged[0][1])} and {MPF._signed(flagged[1][1])}; not flagged: "
                f"seed 0 −4.0), while on the language battery none of the 3 seeds loses more than {drop:.1f} points.") in md
        assert "the 3-seed range: seed_variance.json (RUN-0069, RUN-0080, RUN-0084, RUN-0091, RUN-0093, RUN-0094, RUN-0096)" in md
        tex = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert "$-8.1$ to $-16.1$\\,{\\scriptsize (mean $-10.9$)}" in tex and "$-4.0$ to $-12.0$\\,{\\scriptsize (\\dag{} seeds 1 and 2)}" in tex


def test_real_w2_size_matches_analysis_json(real_run):
    out = real_run
    d = data(out, "fig_w2_size")
    j = _repo_json(MPF.W2_TEST_JSON)["models"]
    assert [s["size"] for s in d["sizes"]] == ["2B", "4B", "9B"]
    cap = (out / "fig_w2_size_caption.tex").read_text(encoding="utf-8")
    for s in d["sizes"]:
        c = j[s["model"]]["pairs_pooled"]["w2a-w1f"]
        assert (s["pairs"]["w2a-w1f"]["value"], s["pairs"]["w2a-w1f"]["low"], s["pairs"]["w2a-w1f"]["high"]) == \
               (c["diff"], c["ci"][0], c["ci"][1])
        # 解析失败率：独立地从叶目录加一遍
        run = MPF.W2T.RUNS[s["model"]]
        pf = [_repo_json(f"{run}/{s['model']}/refseg/w2a/{sp}/metrics.json")["pick"]["parse_failure"] for sp in SPLITS]
        n, of = sum(x["n"] for x in pf), sum(x["of"] for x in pf)
        assert (s["parse_failure"]["w2a"]["n"], s["parse_failure"]["w2a"]["of"]) == (n, of)
        assert f"({n:,}/{of:,})" in cap
    s9 = d["sizes"][-1]
    assert s9["pairs"]["w2c-w1f"]["value"] == j["qwen3.5-9b"]["pairs_pooled"]["w2c-w1f"]["diff"]
    assert all(s["pairs"]["w2c-w1f"] is None for s in d["sizes"][:2])
    assert "see text" not in cap and "2B 48\\%" in cap


def test_real_lang_cost_matches_analysis_json(real_run, tmp_path):
    out = real_run
    d = data(out, "fig_lang_cost")
    j = _repo_json(MPF.LANG_COST_JSON)["systems"]
    # 缺省：四方对照的四个系统（DEV-0073 起含 X-2，T 文本由 RUN-0091 补齐）
    assert d["with_explore"] is False and [s["name"] for s in d["systems"]] == ["E3b+C3", "W2-c", "W2-c+回放", "X-2（C1）"]
    for s in d["systems"]:
        for sec, key, _lab in MPF.LANG_READOUTS:
            r, p = j[s["name"]][sec][key], s["values"][key]
            assert (p["value"], p["low"], p["high"]) == (r["delta_pp"], r["ci95_pp"][0], r["ci95_pp"][1])
    cap = (out / "fig_lang_cost_caption.tex").read_text(encoding="utf-8")
    assert "W2-c + replay: IFEval $-$6.0 (n=50)" in cap and "W2-c: MMLU-Pro $-$6.0 (n=100), IFEval $-$6.0 (n=50)" in cap
    assert "C1 = X-2 (ours), seed 0: none" in cap  # 健康检查的标红只写在图注：X-2（seed 0）没有
    # DEV-0078（D162）：C1 的点仍是 seed 0；另一行画 3 个种子的范围（每个种子一个空心点）= seed_variance.json 各种子的读数
    sr, rng = _seed_readings(), d["seed_range"]
    assert rng["c1_seed"] == 0 and rng["seeds"] == [0, 1, 2]
    for sec, key, _lab in MPF.LANG_READOUTS:
        r = rng["rows"][f"{sec}/{key}"]
        assert r["values"] == [v for _k, v, _f in sr[f"{sec}/{key}"]] and (r["low"], r["high"]) == (min(r["values"]), max(r["values"]))
        assert r["values"][0] == j["X-2（C1）"][sec][key]["delta_pp"]  # seed 0 = 图里 C1 的点
    assert MPF.range_ends(rng["rows"]["ttext/count"]) == (max(r for r in rng["rows"]["ttext/count"]["values"]),
                                                          min(r for r in rng["rows"]["ttext/count"]["values"]))
    assert MPF.range_ends(rng["rows"]["ttext/refexp_box"])[0] == min(rng["rows"]["ttext/refexp_box"]["values"])
    assert "C1 = X-2 (ours) is training seed 0, the checkpoint in the test tables; every other system is a single training seed." in cap
    assert "The thin violet line with open circles, just below C1's point, is C1 = X-2, range over 3 seeds" in cap
    assert f"counting: mean $-${-rng['rows']['ttext/count']['mean']:.1f} over the 3 seeds" in cap and "mean $-$10.9" in cap
    assert "the health check flags C1's seeds 1 and 2 on IFEval (n=50: $-$12.0 and $-$6.0" in cap
    assert "none of the 3 seeds loses more than 3.5 points" in cap and "Language cost is read on the language battery (D154, D162)" in cap
    assert "the 3-seed range: seed\\_variance.json (RUN-0069, RUN-0080, RUN-0084, RUN-0091, RUN-0093, RUN-0094, RUN-0096)" in cap
    assert "C1 = X-2 (ours): health check and language battery against the base on hpcgpu07, T-text against the base on hpcgpu06" in cap
    assert "X-3" not in cap
    # --with-explore：再加 X-3（C2，同一份 JSON，含 T 文本）与 X-3L（读它自己的 compare，没有 T 文本）
    lc = MPF.collect_lang_cost(REPO, with_explore=True)
    assert [s["name"] for s in lc["systems"]] == ["E3b+C3", "W2-c", "W2-c+回放", "X-2（C1）", "X-3（C2，消融）", "X-3L"] and not lc["missing"]
    by = {s["name"]: s for s in lc["systems"]}
    assert by["X-3（C2，消融）"]["values"]["count"]["value"] == j["X-3（C2，消融）"]["ttext"]["count"]["delta_pp"]
    for name, battery, _h, _node in MPF.EXPLORE_LANG:
        r = next(r for r in _repo_json(battery)["rows"] if (r["reading"], r["metric"]) == ("mmlu_pro_letter", "accuracy"))
        assert by[name]["values"]["mmlu_pro_letter.accuracy"]["value"] == r["delta_pp"] and by[name]["values"]["count"] is None
    assert [f["reading"] for f in by["X-3（C2，消融）"]["health_flags"]] == ["ifeval"] and by["X-2（C1）"]["health_flags"] == []
    assert MPF.main(["--out", str(tmp_path), "--with-explore"] + REAL_ARGS) == 0
    assert data(tmp_path, "fig_lang_cost")["with_explore"] is True
    cap = (tmp_path / "fig_lang_cost_caption.tex").read_text(encoding="utf-8")
    assert "Also shown: C2 = X-3 (ablation)" in cap and "C2 = X-3 (ablation): IFEval $-$10.0 (n=50)" in cap and "X-3L" in cap
    assert "range over 3 seeds" in cap and data(tmp_path, "fig_lang_cost")["seed_range"]["seeds"] == [0, 1, 2]
    assert "--with-explore 打开" in (tmp_path / "README.md").read_text(encoding="utf-8")
    cb = data(tmp_path, "fig_count_bins")  # 计数分档：--with-explore 时 X-3（C2）排在 X-2 后面
    assert [s["name"] for s in cb["systems"]] == ["base@0069", "W2-c", "W2-c+回放", "X-2（C1）", "X-3（C2）", "C3", "E3b+C3"]
    # 配色：C1 在两张图里同一个颜色（分类槽 7）
    assert MPF.W2_COLORS["X-2（C1）"] == MPF.SERIES_7 and MPF.W2_COLORS["X-3（C2，消融）"] == MPF.W2_COLORS["X-3（C2）"]


def test_real_count_bins_match_analysis_json(real_run):
    d = data(real_run, "fig_count_bins")
    j = _repo_json(MPF.COUNT_ERRORS_JSON)["systems"]
    assert [s["name"] for s in d["systems"]] == ["base@0069", "W2-c", "W2-c+回放", "X-2（C1）", "C3", "E3b+C3"]
    assert {s["name"]: s["paired_base"] for s in d["systems"]}["X-2（C1）"] == "base@0069"
    assert d["bases_identical"] is True
    for s in d["systems"]:
        for b in ("2–4", "5–7", "8–10"):
            assert s["acc"][b] == j[s["name"]]["bins"][b]["acc"]
            if s["name"] != "base@0069":
                vb = j[s["name"]]["bins"][b]["vs_base"]
                assert (s["delta"][b]["value"], s["delta"][b]["low"], s["delta"][b]["high"]) == (vb["d_acc"], vb["ci95"][0], vb["ci95"][1])
    assert {s["name"]: s["paired_base"] for s in d["systems"]}["E3b+C3"] == "base@0056"


def test_real_rseg_outlier_drawn_off_axis(real_run):
    """干扰图 (d)：秩 128 种子 2 的 R_seg 远在 r32 范围之外，画到轴外（箭头 + 数值）；其余点都在纵轴范围内，别的面板不动。"""
    d = data(real_run, "fig_interference_size")
    assert list(d["offscale"]) == ["R_seg"]
    off = d["offscale"]["R_seg"]
    r128 = next(b for b in d["blocks"] if b.get("rank") == 128)
    assert [(p["block"], p["kind"], p["seed"]) for p in off["points"]] == [("0.8B r128", "seed", 2)]
    assert off["points"][0]["value"] == r128["R_seg"]["2"] < off["ylim"][0]
    others = [v for b in d["blocks"] for v in b["R_seg"].values()] + [b["means"]["R_seg"]["mean"] for b in d["blocks"]]
    others.remove(r128["R_seg"]["2"])
    assert all(off["ylim"][0] < v < off["ylim"][1] for v in others)
    cap = (real_run / "fig_interference_size_caption.tex").read_text(encoding="utf-8")
    assert "In (d), a rank-128 reading lies far outside" in cap and "seed 2: R\\_seg = $-$3.28" in cap
    assert "箭头 + 数值" in (real_run / "README.md").read_text(encoding="utf-8")


def test_pooled_names_and_conventions():
    """合并列的系统名都能按（运行目录, 设置）对上 w2_test_pooled.SYSTEMS；候选与设置名取自 x_test_table.py；四方表的系统名 = lang_cost_table.SYSTEMS。"""
    assert {MPF.pooled_name(k) for k in MPF.ROW_LABELS} == {n for n, _r, _s in MPF.WTP.SYSTEMS}
    assert len({MPF.pooled_name(k) for k in MPF.ROW_LABELS}) == len(MPF.ROW_LABELS)
    assert MPF.row_run_setting("c1") == (MPF.XT.CANDS[MPF.C1], MPF.XT.MAIN) and MPF.row_run_setting("c1_ellipse")[1] == MPF.XT.ABL
    assert [n for n, _l in MPF.LANG_SYSTEMS + MPF.LANG_ABLATION] == list(MPF.LCT.SYSTEMS)
    assert set(MPF.LANG_LABELS) == set(MPF.LCT.SYSTEMS)


def _pooled_view(tmp_path: Path, w2a: dict | None, c1_ciou: float | None = None) -> Path:
    """仓库的视图：experiments 与 docs/analysis 的其他文件用符号链接，只把 w2_test_pooled.json 换成合成的（W2-a 缺或不缺）。"""
    root = tmp_path / "view"
    (root / "docs" / "analysis").mkdir(parents=True)
    (root / "experiments").symlink_to(REPO / "experiments")
    for f in (REPO / "docs" / "analysis").iterdir():
        if f.name != "w2_test_pooled.json":
            (root / "docs" / "analysis" / f.name).symlink_to(f)
    pj = _repo_json(MPF.W2_POOLED_JSON)
    pj["systems"]["W2-a"] = w2a if w2a is not None else {"status": "缺", "missing": list(SPLITS)}
    if c1_ciou is not None:
        pj["systems"]["C1 = X-2（主方法）"]["cIoU"] = {"n": 5000, "ciou": c1_ciou, "ci": [c1_ciou - 0.01, c1_ciou + 0.01]}
    _w(root / MPF.W2_POOLED_JSON, pj)
    return root


def test_w2a_pooled_missing_and_present(tmp_path):
    """任务书：w2_test_pooled.json 里 W2-a 缺与不缺两种情况都能出表（合成 JSON 放在临时目录）。缺时合并列写"—"并进 missing；
    不缺时读数取 JSON。另：把 C1 的合并读数改大，表注里 C2 − C1 的那句跟着变（低于、超出 ±2 点），与 x_test_table.json 的核对报不一致。"""
    # 缺
    root = _pooled_view(tmp_path / "a", None)
    w2 = MPF.collect_w2(root)
    assert w2["rows"]["cIoU"]["w2a"]["pooled"] is None and w2["rows"]["gIoU"]["w2a"]["pooled"] is None
    assert [m for m in w2["missing"] if "W2-a" in m] == [f"W2 合并列 W2-a（{MPF.W2_POOLED_JSON}：status 是 缺，缺 {', '.join(SPLITS)}）"]
    out = tmp_path / "a" / "out"
    out.mkdir()
    MPF.tab_main(w2, out)
    MPF.tab_w2(w2, out, "cIoU", "tab_w2_test")
    md = (out / "tab_main.md").read_text(encoding="utf-8")
    row = next(r for r in md.splitlines() if r.startswith("| W2-a |"))
    assert row.endswith("| — | — |") and "Missing: W2 合并列 W2-a" in md
    row = next(r for r in (out / "tab_w2_test.md").read_text(encoding="utf-8").splitlines() if r.startswith("| W2-a |"))
    assert row.endswith("| — |") and "[" in row.split("|")[2]  # 逐划分照常有读数与 CI
    # 不缺（合成的读数，C1 改成最高）
    root = _pooled_view(tmp_path / "b", {"status": "完成", "source": "experiments/W2/0095_w2a_scores",
                                         "cIoU": {"n": 5000, "ciou": 0.4321, "ci": [0.4200, 0.4400]},
                                         "gIoU": {"n": 5000, "giou": 0.4567, "ci": [0.4500, 0.4600]}}, c1_ciou=0.80)
    w2 = MPF.collect_w2(root)
    assert not [m for m in w2["missing"] if "W2-a" in m]
    assert (w2["rows"]["cIoU"]["w2a"]["pooled"]["value"], w2["rows"]["gIoU"]["w2a"]["pooled"]["value"]) == (0.4321, 0.4567)
    out = tmp_path / "b" / "out"
    out.mkdir()
    MPF.tab_main(w2, out)
    md = (out / "tab_main.md").read_text(encoding="utf-8")
    assert next(r for r in md.splitlines() if r.startswith("| W2-a |")).endswith("| 43.2 [42.0, 44.0] | — |")
    assert "| 80.0 [79.0, 81.0] |" in next(r for r in md.splitlines() if r.startswith("| C1 = X-2 (ours) |")) and "**" not in md
    c2 = _repo_json(MPF.W2_POOLED_JSON)["systems"]["C2 = X-3（消融）"]["cIoU"]["ciou"]
    assert f"pooled, C2 is {100 * (0.80 - c2):.1f} points below C1, outside the ±2 points predicted in preregistration X-6." in md
    assert any(c.startswith("不一致") and "c1 cIoU" in c for c in w2["checks"])


def _itf(main_vals: dict, extra_vals: dict) -> dict:
    blocks = []
    for size, vals in main_vals.items():
        blocks.append({"size": size, "x_size": size, "R_seg": vals, "means": {"R_seg": {"mean": statistics.mean(vals.values())}}})
    blocks.append({"size": "0.8B r128", "x_size": "0.8B", "rank": 128, "R_seg": extra_vals,
                   "means": {"R_seg": {"mean": statistics.mean(extra_vals.values())}}})
    return {"blocks": blocks}


def test_offscale_rule():
    main = {"0.8B": {0: 0.9, 1: 0.8}, "9B": {0: 0.1}}  # r32 范围 [0.1, 0.9]，跨度 0.8 → 窗口 [−0.7, 1.7]
    assert MPF.offscale(_itf(main, {0: 0.5, 1: -0.6}), "R_seg") is None  # 都在窗口内：面板与改动前相同
    off = MPF.offscale(_itf(main, {0: 0.5, 1: -3.0, 2: 0.7}), "R_seg")
    assert [(p["seed"], p["side"]) for p in off["points"]] == [(1, "below")]  # 平均 −0.6 在窗口内，不算离群
    assert off["ylim"][0] < -0.7 and off["ylim"][1] > 0.9
    off = MPF.offscale(_itf(main, {0: 2.5, 1: 2.6}), "R_seg")
    assert {p["kind"] for p in off["points"]} == {"seed", "mean"} and all(p["side"] == "above" for p in off["points"])
    assert MPF.offscale(_itf(main, {0: -3.0}), "P_seg") is None  # 只处理 (d)
    assert MPF.offscale({"blocks": _itf(main, {0: 0.5})["blocks"][:-1]}, "R_seg") is None  # 没有秩 128


def test_fake_tree_without_w2_data_reports_missing(base_run):
    """假的结果树里没有 W2 的分析 JSON：照常出全部文件，README 写明缺；干扰图没有离群读数时没有那一句、data 没有 offscale。"""
    _root, out = base_run
    for name in W2_FIGS:
        assert (out / f"{name}.png").is_file()
    for name in W2_TABS:
        assert (out / f"{name}.md").is_file()
    readme = (out / "README.md").read_text(encoding="utf-8")
    miss = readme.split("## 缺的数据（missing）")[1].split("## 1.")[0]
    assert f"W2 主表：{MPF.W2_TEST_JSON} 不在" in miss and f"语言代价：{MPF.LANG_COST_JSON} 不在" in miss
    assert "marked by an arrow" not in (out / "fig_interference_size_caption.tex").read_text(encoding="utf-8")
    assert "offscale" not in data(out, "fig_interference_size")
    # DEV-0077：统一主表、分档、种子波动的 JSON 都不在，照常出文件，README 写明缺
    for name in ("fig_unified", "fig_bins"):
        assert (out / f"{name}.png").is_file() and (out / f"{name}_caption.tex").is_file()
    for name in ("tab_unified", "tab_seeds"):
        assert (out / f"{name}.md").is_file() and (out / f"{name}.tex").is_file()
    for m in (f"统一主表：{MPF.UNIFIED_JSON} 不在", f"统一主表的文献读数：{MPF.SOTA_JSON} 不在", f"分档：{MPF.X_BINS_JSON} 不在",
              f"种子波动：{MPF.SEED_JSON} 不在", f"4B 补充表：{MPF.X2_4B_JSON} 不在"):
        assert m in miss, m
    # DEV-0078：4B 补充表照常出文件；没有种子的 JSON 时四方表不加范围一列（缺项照旧由"语言代价表"报）
    assert (out / "tab_x2_4b.md").is_file() and (out / "tab_x2_4b.tex").is_file() and (out / "data" / "tab_x2_4b.json").is_file()
    assert RANGE_HEAD not in (out / "tab_lang_cost.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# DEV-0077：统一主表（正文主表）、它的图、分档图、种子波动附录表。用仓库里的真数据出一遍，与分析 JSON 逐项对。
# ---------------------------------------------------------------------------------------------

U_FIGS = (("fig_unified", MPF.DOUBLE_COL_IN), ("fig_bins", MPF.SINGLE_COL_IN))
U_TABS = ("tab_unified", "tab_seeds")
LISA = "LISA-7B（xinlai/LISA-7B-v1）"
E3B = "9B E3b 像素 token 网格"


def _rows(md: str) -> list[str]:
    """Markdown 表的数据行（去掉表头、分隔行与块标题行）。"""
    return [r for r in md.split("\n\n- ")[0].splitlines() if r.startswith("| ") and not r.startswith("| *")][1:]


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def test_real_unified_outputs_written(real_run):
    from PIL import Image

    out = real_run
    for name, width_in in U_FIGS:
        pdf, png = out / f"{name}.pdf", out / f"{name}.png"
        assert pdf.read_bytes()[:5] == b"%PDF-" and b"/Type3" not in pdf.read_bytes()
        with Image.open(png) as im:
            assert abs(im.info["dpi"][0] - MPF.DPI) < 1 and im.size[0] == round(width_in * MPF.DPI)
        assert (out / f"{name}_caption.tex").read_text(encoding="utf-8").count("\\caption{") == 1
        assert (out / "data" / f"{name}.json").is_file()
    for name in U_TABS:
        tex = (out / f"{name}.tex").read_text(encoding="utf-8")
        assert "\\toprule" in tex and "\\bottomrule" in tex and tex.count("\\caption{") == 1 and f"\\label{{tab:{name[4:]}}}" in tex
        assert (out / f"{name}.md").is_file() and (out / "data" / f"{name}.json").is_file()
    assert "Needs \\usepackage{booktabs,graphicx,xcolor}." in (out / "tab_unified.tex").read_text(encoding="utf-8").splitlines()[0]
    readme = (out / "README.md").read_text(encoding="utf-8")
    for sec in ("## 12. 统一主表（正文主表）", "## 13. 统一主表的图", "## 14. 分档", "## 15. 种子波动（附录）"):
        assert sec in readme
    for name in [n for n, _w in U_FIGS] + list(U_TABS):
        assert f"`{name}." in readme.split("## 文件")[1]
    for src in (MPF.UNIFIED_JSON, MPF.SOTA_JSON, MPF.X_BINS_JSON, MPF.SEED_JSON):
        assert src in readme
    assert readme.split("## 缺的数据（missing）")[1].split("\n## ")[0].strip() == "- 无"
    # 运行编号写进表注与图注（任务书：RUN-0072、0078、0092、0095、0097、0100）
    runs = ["RUN-0072", "RUN-0078", "RUN-0092", "RUN-0095", "RUN-0097", "RUN-0100"]
    for f in ("tab_unified.tex", "tab_unified.md", "fig_unified_caption.tex"):
        txt = (out / f).read_text(encoding="utf-8")
        assert all(r in txt for r in runs), f
    assert all(r in (out / "fig_bins_caption.tex").read_text(encoding="utf-8") for r in ("RUN-0072", "RUN-0092", "RUN-0099"))


def test_real_tab_unified_matches_json(real_run):
    """正文主表：系统一个不少、顺序与组同 unified_table.json；抽查 X-2、LISA、E3b 的格子与一个文献格；不加粗；表注写明子集、照实报告、
    LISA 两处不是同一个权重、文献数字在完整 test 划分上且不参与配对。"""
    out = real_run
    uj, sj = _repo_json(MPF.UNIFIED_JSON), _repo_json(MPF.SOTA_JSON)
    md, tex = (out / "tab_unified.md").read_text(encoding="utf-8"), (out / "tab_unified.tex").read_text(encoding="utf-8")
    rows = _rows(md)
    n_sys = len(uj["systems"])
    assert n_sys == 12 and len(rows) == n_sys + len(sj["rows"])
    sys_rows, sota_rows = rows[:n_sys], rows[n_sys:]
    assert [_cells(r)[0] for r in sys_rows] == [MPF.UNIFIED_LABELS[e["name"]][0] for e in uj["systems"]]
    d = data(out, "tab_unified")
    assert [s["name"] for s in d["systems"]] == [e["name"] for e in uj["systems"]] and not d["missing"]
    # 组的标题按 JSON 里出现的顺序，各一块
    titles = [r.strip("| *") for r in md.splitlines() if r.startswith("| *")]
    groups = list(dict.fromkeys(e["group"] for e in uj["systems"]))
    assert titles[:len(groups)] == [MPF.UNIFIED_GROUPS[g][0] for g in groups] and titles[-1].startswith(MPF.UNIFIED_SOTA_LABEL)
    by = {e["name"]: e for e in uj["systems"]}
    for name in (uj["main"], LISA, E3B):
        e = by[name]
        cells = _cells(next(r for r in sys_rows if r.startswith(f"| {MPF.UNIFIED_LABELS[name][0]} |")))
        assert cells[1:6] == [_fmt(e["per_split"][sp]["cIoU"]["ciou"]) for sp in SPLITS]
        assert cells[6] == _cell(e["pooled"]["cIoU"]["ciou"], *e["pooled"]["cIoU"]["ci"])
        assert cells[7] == _fmt(e["pooled"]["gIoU"]["giou"])
        if name == uj["main"]:
            assert cells[8] == "—"
        else:
            mm = e["main_minus"]["pooled"]["cIoU"]
            assert cells[8] == _cell(mm["diff"], *mm["ci"], signed=True)
    # 文献格：DeRIS-L 在 RefCOCO+ testB（JSON 里已是 ×100），合并与配对差写"—"；.tex 里灰色
    der = next(r for r in sj["rows"] if r["name"] == "DeRIS-L")
    v = der["values"][sj["splits"].index("test_refcoco+_unc_testB")]
    cells = _cells(next(r for r in sota_rows if r.startswith("| DeRIS-L,")))
    assert cells[1 + SPLITS.index("test_refcoco+_unc_testB")] == f"{v:.1f}" and cells[6:] == ["—", "—", "—"]
    trow = next(r for r in tex.splitlines() if r.startswith("\\textcolor{gray}{DeRIS-L"))
    assert f"\\textcolor{{gray}}{{${v:.1f}$}}" in trow and trow.count("\\textcolor{gray}{--}") == 3
    assert sum(r.startswith("\\textcolor{gray}{") for r in tex.splitlines()) == len(sj["rows"])
    # 不加粗
    assert "**" not in md and "\\textbf" not in tex and "\\best" not in tex
    for s in ("hash rank", f"salt {_repo_json(uj['subset'])['salt']}", "300 per split", "the same expressions as LLMSeg's run",
              "Reported as run, nothing selected", "No entry is set in bold", "LISA-7B appears twice, and the two are not the same weights",
              "xinlai/LISA-7B-v1", "fine-tuned on RefCOCO/+/g", "image level", "on the full test splits, not on our 300-expression subsets",
              "never paired", "learned vision models", "all on hpcgpu06"):
        assert s in md, s
    for r in sj["rows"]:  # 各自的出处照录
        assert r["source"] in md, r["name"]
    # .tex：子集、照实报告、LISA、文献、运行编号在 caption；系统定义与文献出处在表下面（caption 太长 pdflatex 报错）
    cap = next(ln for ln in tex.splitlines() if ln.startswith("\\caption"))
    below = tex.split("\\begin{minipage}{\\textwidth}\\footnotesize")[1].split("\\end{minipage}")[0]
    for s in ("hash rank", "Reported as run, nothing selected", "not the same weights", "never paired", "RUN-0100", "under the table"):
        assert s in cap, s
    assert "seeds GrabCut as a trimap" in below and "Lai et al., LISA" in below and "seeds GrabCut" not in cap
    assert tex.index("\\end{tabular}") < tex.index("\\begin{minipage}") < tex.index("\\end{table*}")
    assert all(c.startswith("一致") for c in d["checks"]) and len(d["checks"]) == 4


def test_real_fig_unified_matches_json(real_run):
    """图：每个划分面板的横条 = JSON 的逐划分读数与 CI（10,000 次）；文献线只在划分面板；合并面板 = 合并读数与 CI；颜色按组、C1 槽 7。"""
    d = data(real_run, "fig_unified")
    uj, sj = _repo_json(MPF.UNIFIED_JSON), _repo_json(MPF.SOTA_JSON)
    assert list(d["panels"]) == SPLITS + ["pooled"]
    for col in SPLITS + ["pooled"]:
        bars = d["panels"][col]["bars"]
        assert [b["system"] for b in bars] == [e["name"] for e in uj["systems"]]
        for b, e in zip(bars, uj["systems"]):
            c = (e["pooled"] if col == "pooled" else e["per_split"][col])["cIoU"]
            assert (b["value"], b["low"], b["high"], b["n"], b["n_resamples"]) == \
                   (c["ciou"], c["ci"][0], c["ci"][1], 1500 if col == "pooled" else 300, 10000)
            assert b["color"] == (MPF.SERIES_7 if e["name"] == uj["main"] else MPF.UNIFIED_GROUP_COLORS[e["group"]])
        sota = d["panels"][col]["sota"]
        if col == "pooled":
            assert sota == []
        else:
            want = sorted((r["values"][sj["splits"].index(col)], r["name"]) for r in sj["rows"])
            assert [(x["value"], x["name"]) for x in sota] == want
            assert all(a["label_y"] > b["label_y"] for a, b in zip(sota, sota[1:]))  # 名字一行一个，自下而上
    # 配色：过 validate_palette.js 的那一组（紫、蓝、青、橙、深灰）；文献线中性灰
    assert {b["color"] for b in d["panels"]["pooled"]["bars"]} == {MPF.SERIES_7, MPF.SERIES_1, MPF.SERIES_3, MPF.SERIES_2, MPF.INK_2}
    assert MPF.UNIFIED_MAIN_COLOR == MPF.W2_COLORS["X-2（C1）"] and MPF.UNIFIED_GROUP_COLORS["本文的方法"] == MPF.W2_COLORS["W2-c"]
    assert MPF.SOTA_COLOR == MPF.MUTED
    cap = (real_run / "fig_unified_caption.tex").read_text(encoding="utf-8")
    assert "Table~\\ref{tab:unified}" in cap and "The pooled panel has none" in cap and "not the released v1 weights" in cap
    assert "10,000 resamples" in cap and "C1 = X-2 (ours) violet" in cap


def test_real_fig_bins_matches_json(real_run):
    """分档图：(a) 人与非人、(b) 形状复杂度四分位；每档的句数与两个配对差 = x_bins.json；图注写分档规则（切点）与每个划分里人的占比。"""
    d = data(real_run, "fig_bins")
    xj = _repo_json(MPF.X_BINS_JSON)
    assert [s["key"] for s in d["schemes"]] == ["(i) 人与非人", "(iii) 形状复杂度 c = P²/A"]
    a, b = d["schemes"]
    assert [x["label"] for x in a["bins"]] == ["Person", "Non-person"]
    q = [f"{x:.1f}" for x in xj["cuts"]["complexity"]]
    assert [x["label"] for x in b["bins"]] == [f"Q1: c ≤ {q[0]}", f"Q2: {q[0]} < c ≤ {q[1]}", f"Q3: {q[1]} < c ≤ {q[2]}", f"Q4: c > {q[2]}"]
    for sch in d["schemes"]:
        for x in sch["bins"]:
            r = xj["schemes"][sch["key"]][x["key"]]
            assert x["n"] == r["n"]
            for dk in ("x2_minus_w2c", "x2_minus_ellipse"):
                p = x["diffs"][dk]
                assert (p["value"], p["low"], p["high"], p["n"], p["n_resamples"]) == (r[dk]["diff"], *r[dk]["ci"], r[dk]["n"], 10000)
    person = xj["schemes"]["(i) 人与非人"]["人"]
    assert a["bins"][0]["diffs"]["x2_minus_w2c"]["value"] == person["x2_minus_w2c"]["diff"]
    assert sum(x["n"] for x in a["bins"]) == sum(x["n"] for x in b["bins"]) == xj["n"]
    cap = (real_run / "fig_bins_caption.tex").read_text(encoding="utf-8")
    assert f"cut points {q[0]}, {q[1]} and {q[2]}" in cap and "fixed before any system's bin readings were computed" in cap
    assert "preregistration U-6" in cap and "COCO category 1" in cap and "c = P$^2$/A" in cap
    for sp in SPLITS:
        assert f"{MPF.split_label(sp)} {100 * xj['person_share_by_split'][sp]['person']:.0f}\\%" in cap, sp


def test_real_tab_seeds_matches_json(real_run):
    """种子波动：主设置与椭圆消融的合并 cIoU、− W2-c 参照、− seed 0、跨种子均值 ± 标准差；语言读数（含计数）；健康检查的标红 †；
    表注写明只在开发数据与语言检查上、不是 test，以及运行编号。"""
    out = real_run
    sj = _repo_json(MPF.SEED_JSON)
    md, tex = (out / "tab_seeds.md").read_text(encoding="utf-8"), (out / "tab_seeds.tex").read_text(encoding="utf-8")
    rows = {(_cells(r)[0]): _cells(r)[1:] for r in _rows(md)}
    seeds = [sj["seeds"][k] for k in sorted(sj["seeds"], key=int)]
    for st, lab in MPF.SEED_SETTINGS:
        cells = rows[f"{lab}, pooled (n=2,300)"]
        assert cells[:3] == [_cell(e["settings"][st]["pooled"]["ciou"], *e["settings"][st]["pooled"]["ci"]) for e in seeds]
        a = sj["across_seeds"][st]
        assert cells[3] == f"{100 * a['mean']:.2f} ± {100 * a['sd']:.2f}"
        x = [e["settings"][st]["minus_w2c_pooled"] for e in seeds]
        assert rows[f"{lab} − W2-c reference"][:3] == [_cell(c["diff"], *c["ci"], signed=True) for c in x]
        x = [e["settings"][st].get("minus_seed0_pooled") for e in seeds]
        assert rows[f"{lab} − seed 0"][:3] == ["—"] + [_cell(c["diff"], *c["ci"], signed=True) for c in x[1:]]
    # 计数（T 文本）：三个种子的差与 CI
    cnt = [e["language"]["ttext"]["count"] for e in seeds]
    assert rows["CountBench counting accuracy (n=442)"][:3] == [f"{c['delta_pp']:+.1f} [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}]"
                                                              for c in cnt]
    mm = [e["language"]["battery"]["mmlu_pro_letter.accuracy"] for e in seeds]
    assert rows["MMLU-Pro, letter answers (n=2,000)"][:3] == [f"{c['delta_pp']:+.1f} [{c['ci95_pp'][0]:+.1f}, {c['ci95_pp'][1]:+.1f}]"
                                                            for c in mm]
    # 健康检查：差与标红 †
    h = [e["health"]["rows"]["ifeval.prompt_level_strict"] for e in seeds]
    assert rows["IFEval (n=50)"][:3] == [f"{c['delta_pp']:+.1f}" + ("†" if c["flag"] else "") for c in h]
    assert any(c["flag"] for c in h) and "\\textsuperscript{\\dag}" in tex
    for s in ("not on the test splits", "development data", "RUN-0080", "RUN-0093", "RUN-0094", "RUN-0082", "RUN-0084", "RUN-0091",
              "RUN-0096", "RUN-0069", "hpcgpu07", "hpcgpu06", "10,000 resamples"):
        assert s in md, s
    assert not data(out, "tab_seeds")["missing"]
    # DEV-0078（D162）：种子稳定性一句——分割几乎相同（三个种子的合并读数），差异全在语言的副作用上，列出计数的三个值与均值
    sp = [f"{100 * e['settings'][MPF.XT.MAIN]['pooled']['ciou']:.1f}" for e in seeds]
    cv = [c["delta_pp"] for c in cnt]
    want = (f"- Segmentation on the development data is almost the same for the 3 seeds (C1 recipe, pooled: {sp[0]}, {sp[1]} and {sp[2]}); "
            "the differences between seeds are all in the language side effects, mostly counting (CountBench counting accuracy, seeds 0, 1 "
            f"and 2: {MPF._signed(cv[0])}, {MPF._signed(cv[1])} and {MPF._signed(cv[2])} points; mean {MPF._signed(sum(cv) / 3)}).")
    assert want in md.splitlines() and md.splitlines().index(want) == md.splitlines().index(next(x for x in md.splitlines() if x.startswith("- "))) + 1
    assert "(C1 recipe, pooled: 73.0, 73.1 and 73.2)" in want and "−8.6, −16.1 and −8.1 points; mean −10.9" in want
    assert MPF._tex_escape(want[2:]) in tex


def test_real_tab_x2_4b_matches_json(real_run):
    """DEV-0078：4B 补充表（附录）。4B 一列 = x2_4b.json（分割三个设置的合并读数与 CI、样条 − 椭圆、大题量的差与 CI、健康检查的差与 †）；
    9B 一列 = 三个种子的范围（分割取 x2_4b.json 的 nine_b，与 seed_variance.json 的 across_seeds 相同；语言与健康检查取 seed_variance.json）；
    9B 的样条 − 椭圆 = 每个种子两个合并读数之差。表注写明补充实验、预注册 X-7、只在开发数据上、4B 一个种子、RUN-0098 与 0101。"""
    out = real_run
    xj, svj = _repo_json(MPF.X2_4B_JSON), _repo_json(MPF.SEED_JSON)
    md, tex = (out / "tab_x2_4b.md").read_text(encoding="utf-8"), (out / "tab_x2_4b.tex").read_text(encoding="utf-8")
    assert md.splitlines()[2] == "| Reading | Qwen3.5-4B (1 seed) | Qwen3.5-9B (range over 3 seeds) |"
    rows = {_cells(r)[0]: _cells(r)[1:] for r in _rows(md)}
    assert len(rows) == 4 + 2 + 3
    # 9B 的分割：x2_4b.json 的 nine_b 就是 seed_variance.json 的 across_seeds
    for st in ("x2_spline_trimap", "x2_ellipse_trimap", "x2_select"):
        assert xj["nine_b"][st] == svj["across_seeds"][st], st
    for st, lab in MPF.X4B_SETTINGS:
        p, nb = xj["settings"][st]["pooled"], xj["nine_b"][st]["pooled"]
        assert rows[f"{lab}, pooled (n=2,300)"] == [_cell(p["ciou"], *p["ci"]), _expected_range([(0, v, False) for v in nb], scale=100, signed=False)]
    sme = xj["spline_minus_ellipse"]
    d9 = [a - b for a, b in zip(xj["nine_b"]["x2_spline_trimap"]["pooled"], xj["nine_b"]["x2_ellipse_trimap"]["pooled"])]
    assert rows["C1 recipe (spline) − in-box ellipse"] == [_cell(sme["diff"], *sme["ci"], signed=True),
                                                         _expected_range([(0, v, False) for v in d9], scale=100)]
    # 抽查（任务书）：4B 样条、4B IFEval 的差与 CI、9B 三个种子的范围
    assert rows["C1 recipe (spline), pooled (n=2,300)"] == ["72.2 [70.8, 73.5]", "73.0 to 73.2"]
    assert rows["C1 recipe (spline) − in-box ellipse"] == ["+16.2 [+15.4, +17.0]", "+16.8 to +17.2"]
    b = xj["battery"]["ifeval.prompt_level_strict"]
    assert rows["IFEval, prompt-level strict (n=541)"] == [f"{b['delta_pp']:+.1f} [{b['ci95_pp'][0]:+.1f}, {b['ci95_pp'][1]:+.1f}]",
                                                          _expected_range(_seed_readings()["battery/ifeval.prompt_level_strict"])]
    assert rows["IFEval, prompt-level strict (n=541)"] == ["-5.7 [-8.7, -2.8]", "+0.0 to -3.5"]
    b = xj["battery"]["mmlu_pro_letter.accuracy"]
    assert rows["MMLU-Pro, letter answers (n=2,000)"] == [f"{b['delta_pp']:+.1f} [{b['ci95_pp'][0]:+.1f}, {b['ci95_pp'][1]:+.1f}]", "-0.2 to -2.3"]
    sr = _seed_readings()
    for k, lab in (("mmlu_pro_letter.accuracy", "MMLU-Pro (n=100)"), ("gsm8k.flexible", "GSM8K (n=50)"), ("ifeval.prompt_level_strict", "IFEval (n=50)")):
        h = xj["health"]["rows"][k]
        assert rows[lab] == [f"{h['delta_pp']:+.1f}" + ("†" if h["flag"] else ""), _expected_range(sr[f"health/{k}"])], lab
    assert rows["IFEval (n=50)"] == ["-10.0†", "-4.0 to -12.0 († seeds 1 and 2)"]
    # 题数、重抽次数取 4B 的 compare 文件
    hc, bc = _repo_json(MPF.X4.HEALTH), _repo_json(MPF.X4.BATTERY)
    assert [r["n"] for r in hc["rows"] if r["headline"]] == [100, 50, 50] and bc["statistics"]["bootstrap"]["n_resamples"] == 1000
    for s in ("Supplementary experiment (preregistration X-7", "Development data only, not the test splits", "Qwen3.5-4B: one training seed",
              "RUN-0098 (hpcgpu07)", "4B language battery RUN-0101 (hpcgpu07", "no interval", "10,000 resamples (seed 20260929)",
              "1,000 resamples", "base 47.4 and 83.0", "4B base 48.0, 86.0 and 88.0", "Language cost is read on the language battery",
              "Data: x2_4b.json, seed_variance.json"):
        assert s in md, s
    assert "T-text" not in md  # 这张表没有 T 文本
    assert "\\label{tab:x2-4b}" in tex and tex.count("\\caption{") == 1 and "\\textsuperscript{\\dag}" in tex
    d = data(out, "tab_x2_4b")
    assert not d["missing"] and d["runs"] == {"segmentation": ["0098"], "health": ["0098"], "battery": ["0101"]}
    assert d["nodes"] == {"health": "hpcgpu07", "battery": "hpcgpu07"} and d["node"] == "hpcgpu07"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "## 16. 4B 补充实验（附录）：`tab_x2_4b.tex` / `tab_x2_4b.md`（DEV-0078）" in readme and MPF.X2_4B_JSON in readme
    assert all(f"`{f}`" in readme.split("## 文件")[1] for f in DEV0078_NEW)


def test_range_cell_and_seed_range_rules():
    """范围的写法（离 0 近的一端在前；两端写出来相同时只写一个；均值与标红的种子）与 seed_range 的缺项：少一个种子、C1 的来源对不上。"""
    r = {"low": -16.06, "high": -8.14, "mean": -10.93, "flagged": None}
    assert MPF._range_cell(r, False) == "-8.1 to -16.1 (mean -10.9)"
    assert MPF._range_cell({"low": 3.41, "high": 3.52}, False) == "+3.4 to +3.5"
    assert MPF._range_cell({"low": -3.5, "high": 0.0}, False) == "+0.0 to -3.5"
    assert MPF._range_cell({"low": 0.7302, "high": 0.7304}, False, 100.0, False) == "73.0"
    assert MPF._range_cell({"low": -12.0, "high": -4.0, "flagged": [1, 2]}, True) == "$-4.0$ to $-12.0$\\,{\\scriptsize (\\dag{} seeds 1 and 2)}"
    assert MPF._range_cell({"low": -6.0, "high": -4.0, "flagged": [2]}, False) == "-4.0 to -6.0 († seed 2)"
    assert MPF._range_cell(None, False) == "—"
    sd = MPF.collect_seeds(REPO)
    full = MPF.seed_range(sd, _repo_json(MPF.LANG_COST_JSON)["systems"]["X-2（C1）"]["sources"])
    assert full["c1_seed"] == 0 and not full["missing"]
    sd2 = json.loads(json.dumps(sd))
    sd2["seeds"][1]["status"] = "缺"
    part = MPF.seed_range(sd2)
    assert part["seeds"] == [0, 2] and any("只用了 2 个种子" in m for m in part["missing"])
    bad = MPF.seed_range(sd, {"battery": "experiments/X/0999_x_hpcgpu07/lb.json", "ttext": "x"})
    assert bad["c1_seed"] is None and any("对不上" in m for m in bad["missing"])
    assert MPF.seed_range({"seeds": []})["missing"] == [f"C1 的种子范围：{MPF.SEED_JSON} 不在"]


def test_real_latex_compiles(real_run, tmp_path):
    """生成的每张表与每个图注都能用 pdflatex 编译（DEV-0077：tab_unified 的表注全放进 caption 时约 5,000 字符，pdflatex 量 caption 的单行宽度
    报 Dimension too large；系统定义与文献出处移到表下面之后通过）。没有 pdflatex 就跳过。"""
    exe = shutil.which("pdflatex")
    if exe is None:
        pytest.skip("没有 pdflatex")
    for f in real_run.iterdir():
        if f.suffix in (".tex", ".pdf"):
            shutil.copy(f, tmp_path / f.name)
    body = []
    for t in sorted(f.stem for f in real_run.glob("tab_*.tex")):
        body += [f"\\input{{{t}.tex}}", "\\clearpage"]
    for fg in sorted(f.name.removesuffix("_caption.tex") for f in real_run.glob("fig_*_caption.tex")):
        body += [f"\\begin{{figure*}}\\centering\\includegraphics[width=\\textwidth]{{{fg}.pdf}}\\input{{{fg}_caption.tex}}\\end{{figure*}}",
                 "\\clearpage"]
    assert any("tab_unified" in b for b in body) and any("fig_unified" in b for b in body)
    doc = ["\\documentclass{article}", "\\usepackage[utf8]{inputenc}", "\\usepackage[T1]{fontenc}",
           "\\usepackage{booktabs,graphicx,multirow,xcolor}", "\\usepackage[margin=1in]{geometry}", "\\begin{document}", *body,
           "\\end{document}"]
    (tmp_path / "all.tex").write_text("\n".join(doc), encoding="utf-8")
    r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", "all.tex"], cwd=tmp_path, capture_output=True, text=True,
                       timeout=300)
    log = (tmp_path / "all.log").read_text(encoding="utf-8", errors="replace") if (tmp_path / "all.log").is_file() else r.stdout
    assert r.returncode == 0, [ln for ln in log.splitlines() if ln.startswith("!") or ln.startswith("l.")][:6]
    assert len(next(ln for ln in (real_run / "tab_unified.tex").read_text(encoding="utf-8").splitlines() if ln.startswith("\\caption"))) < 3500


def _analysis_view(tmp_path: Path, replace: dict) -> Path:
    """仓库的视图：experiments、configs、docs/references 与 docs/analysis 的其他文件用符号链接，只把 replace 里的分析 JSON 换成合成的。"""
    root = tmp_path / "view"
    (root / "docs" / "analysis").mkdir(parents=True)
    for d in ("experiments", "configs"):
        (root / d).symlink_to(REPO / d)
    (root / "docs" / "references").symlink_to(REPO / "docs" / "references")
    for f in (REPO / "docs" / "analysis").iterdir():
        if f"docs/analysis/{f.name}" not in replace:
            (root / "docs" / "analysis" / f.name).symlink_to(f)
    for rel, obj in replace.items():
        _w(root / rel, obj)
    return root


def test_unified_failed_system_still_listed(tmp_path):
    """照实报告（预注册 U-1）：JSON 里某个系统没跑完（status 缺）也照样占一行、格子写"—"、进 missing 与表注；没有英文名的新系统用原名，
    系统数仍与 JSON 相同。"""
    uj = _repo_json(MPF.UNIFIED_JSON)
    i = next(k for k, e in enumerate(uj["systems"]) if e["name"] == E3B)
    uj["systems"][i] = {"group": uj["systems"][i]["group"], "name": E3B, "source": uj["systems"][i]["source"], "status": "缺"}
    uj["systems"].append({"group": "外部方法", "name": "NewSys", "source": "experiments/W2/0999_new_hpcgpu06", "status": "缺"})
    root = _analysis_view(tmp_path, {MPF.UNIFIED_JSON: uj})
    u = MPF.collect_unified(root)
    out = tmp_path / "out"
    out.mkdir()
    MPF.tab_unified(u, out)
    MPF.fig_unified(u, out)
    md = (out / "tab_unified.md").read_text(encoding="utf-8")
    rows = _rows(md)
    assert len(rows) == len(uj["systems"]) + len(_repo_json(MPF.SOTA_JSON)["rows"])
    assert _cells(next(r for r in rows if r.startswith(f"| {MPF.UNIFIED_LABELS[E3B][0]} |")))[1:] == ["—"] * 8
    assert _cells(next(r for r in rows if r.startswith("| NewSys |")))[1:] == ["—"] * 8
    assert any("E3b" in m and "status 是 缺" in m for m in u["missing"]) and any("NewSys 没有英文名" in m for m in u["missing"])
    assert "- Missing: " in md
    # 只有完成的系统进核对：E3b 不在新跑三项的核对里（20 格 = E1 与 LISA）
    assert next(c for c in u["checks"] if "新跑三项" in c).endswith("（20 格）")
