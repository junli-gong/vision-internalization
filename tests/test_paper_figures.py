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
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import statistics
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
    for sec in ("## 5. W2 test 主表", "## 6. W2-a 随尺寸", "## 7. 语言代价", "## 8. 计数分档", "## 9. 正文主表", "## 10. 语言代价四方对照表",
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
    """原有的 E1、E2、E3、干扰图与 fig_w2_size 的输出与仓库里 docs/figures/ 的逐字节相同（DEV-0073 不动它们）。"""
    for f in UNCHANGED:
        assert (real_run / f).read_bytes() == (REPO / "docs" / "figures" / f).read_bytes(), f


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
    assert head == "| Check | Reading | Base | E3b+C3 | W2-c | W2-c + replay | C1 = X-2 (ours) |"
    assert full.splitlines()[2] == head + " C2 = X-3 (ablation) |"
    lt = data(out, "tab_lang_cost")
    x2 = next(s for s in lt["systems"] if s["name"] == "X-2（C1）")
    # 抽查：X-2 的大题量 MMLU-Pro 与计数
    for sec, key in (("battery", "mmlu_pro_letter.accuracy"), ("ttext", "count")):
        r, c = j["X-2（C1）"][sec][key], x2["cells"][f"{sec}/{key}"]
        assert (c["delta"], c["low"], c["high"], c["n"]) == (r["delta_pp"], r["ci95_pp"][0], r["ci95_pp"][1], r["n"])
    row = next(r for r in md.splitlines() if "MMLU-Pro, letter answers" in r)
    r = j["X-2（C1）"]["battery"]["mmlu_pro_letter.accuracy"]
    assert row.endswith(f"| {r['delta_pp']:+.1f} [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}] |")
    row = next(r for r in md.splitlines() if "CountBench counting accuracy" in r)
    r = j["X-2（C1）"]["ttext"]["count"]
    assert row.endswith(f"| {r['delta_pp']:+.1f} [{r['ci95_pp'][0]:+.1f}, {r['ci95_pp'][1]:+.1f}] |")
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
    assert "C1 = X-2 (ours): none" in cap  # 健康检查的标红只写在图注：X-2 没有
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
