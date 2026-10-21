"""DEV-0052：scripts/figures/make_paper_figures.py（论文用图表，D103 §五）。

用小的假结果树跑一遍：
- 每个输出文件都生成（PDF、300 dpi 的 PNG、LaTeX 与 Markdown 的表、图注、data/*.json、README）；
- 读到的数字与假数据一致（E1 的读数与配对差、E2 的逐层读数与相对深度、E3 表的种子平均与标准差、干扰的 L/P/R）；
- 缺 9B（RUN-0056）或秩 128（RUN-0054）时不报错，README 与图注写明缺；目录出现后自动纳入；
- 秩 128 的 C3、E3b+C3 不回退到秩 32 的读数；被合并核对拒绝的种子在表注里列出、不进平均；
- JSON 与重算不一致时 README 报"不一致"。
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


def test_size_value_parses_labels():
    assert MPF.size_value("0.8B") == 0.8 and MPF.size_value("9B") == 9.0
    with pytest.raises(ValueError):
        MPF.size_value("big")
