"""DEV-0051：W1 的基线（src/w1/baselines.py）与对 E1 运行器的两处放宽。

- "自己的框"：光栅化与 scripts/summarize_e1_posthoc.own_box_record 的多边形对照逐项相同；手算的矩形上 IoU ≈ 1；解析失败的框是空 mask。
- W1-F：合成图上（暗底、亮矩形）框 → GrabCut 跑通，IoU 高、确定；框解析失败、GrabCut 报错各自记录。用假的 grabcut_mask（同签名、同
  规则）跑；真工作台合入之后另用 src.w1.workbench.grabcut_mask 跑同一组断言，并核对与工作台的函数是同一个。
- E1 的两处放宽（run_forced_poly_task 的 sample_subset / leaf，run_task 的 leaf）：不给新参数时，在 E1 原有的子集上，新代码与改动前
  （eb550b99 的 src/e1/posthoc_runner.py、src/e1/runner.py 原样，tests/fixtures/w1/ 里，sha256 钉住）在同一进程、同一个小模型上的
  确定输出（predictions.jsonl、metrics.json、config.yaml、README.md）逐字节相同；给了新参数时样本可以取 W1 的划分、输出写进给定目录，
  清单对不上照样报错。
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.adapters import registry
from src.adapters.image_policy import PixelBudget
from src.data.records import Sample
from src.e1 import posthoc_runner as PR
from src.e1 import runner as R
from src.e1.posthoc import load_posthoc_config
from src.e1.prompts import load_prompt_config
from src.e1.tasks import DataPaths, select
from src.w1 import baselines as B
from src.w1.config import load_w1_config
from src.w1.data import tiny_splits, w1_items
from src.w1.runner import W1RunSpec, w1_record
from tests._w1_fakes import FAKE_GRABCUT_ITERATIONS, fake_grabcut_mask

REPO = Path(__file__).resolve().parents[1]
QWEN = "qwen3.5-0.8b"
SNAP_POSTHOC = REPO / "tests" / "fixtures" / "w1" / "e1_posthoc_runner_eb550b99.py"
SNAP_POSTHOC_SHA256 = "e3cf8f9574a3382949660196e87a1eda1d63cb8b7df47f46da5dbcea0bab24c4"
SNAP_RUNNER = REPO / "tests" / "fixtures" / "w1" / "e1_runner_eb550b99.py"
SNAP_RUNNER_SHA256 = "c2bc70a6ac2b7f318d88c4210857ba137f5b77942ac7b65a7dc04e674653d9e7"
DETERMINISTIC = ("predictions.jsonl", "metrics.json", "config.yaml", "README.md")
CFG = load_w1_config()
HAS_WB = importlib.util.find_spec("src.w1.workbench") is not None


def _load(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # 快照里有 dataclass
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


@pytest.fixture(scope="module")
def summ():
    return _load(REPO / "scripts" / "summarize_e1_posthoc.py", "_w1_test_summarize")


@pytest.fixture(scope="module")
def tiny():
    from src.adapters.factory import make_adapter

    try:
        return make_adapter(QWEN, device="cpu", dtype=torch.float32, image_policy=PixelBudget(64), tiny=True, init_std=0.05)
    except OSError as e:
        pytest.fail(f"取不到 {QWEN} 的小文件：{e!r}", pytrace=False)


def _spec(tiny, task, subset, bs=4):
    entry = registry.get_model(QWEN)
    return B.e1_spec(entry, tiny, PixelBudget(64), task, subset, bs, tiny=True, limit=None)


# ---------------------------------------------------------------------------
# "自己的框"
# ---------------------------------------------------------------------------


def test_own_box_matches_summary_script(summ):
    gt = np.zeros((48, 64), dtype=bool)
    gt[8:24, 16:48] = True
    for rec in ({"id": "r/1", "failure": None, "text": "x", "raw_text": "x", "parsed": {"box": [16.0, 8.0, 48.0, 24.0]}},
                {"id": "r/2", "failure": None, "text": "y", "parsed": {"box": [0.0, 0.0, 30.5, 47.0]}},
                {"id": "r/3", "failure": "no_box", "text": "no", "parsed": None}):
        mine = B.own_box_record(rec, gt)
        assert mine == summ.own_box_record("refexp_poly", rec, gt)
    hit = B.own_box_record({"id": "r/1", "failure": None, "text": "x", "parsed": {"box": [16.0, 8.0, 48.0, 24.0]}}, gt)
    assert hit["score"]["iou"] == pytest.approx(1.0, abs=0.08)
    miss = B.own_box_record({"id": "r/3", "failure": "no_box", "text": "no", "parsed": None}, gt)
    assert (miss["score"]["inter"], miss["score"]["union"]) == (0, int(gt.sum()))


# ---------------------------------------------------------------------------
# W1-F
# ---------------------------------------------------------------------------


def _synthetic():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 40, (48, 64, 3), dtype=np.uint8)
    arr[12:36, 16:48] = (np.array([210, 60, 50]) + rng.integers(-10, 10, (24, 32, 3))).astype(np.uint8)
    poly = [[16.0, 12.0, 48.0, 12.0, 48.0, 36.0, 16.0, 36.0]]
    s = Sample(id="syn/x/1", task="refexp", images=("/nonexistent.png",), image_keys=("k",), fields={"expression": "red"},
               gt={"box": [16.0, 12.0, 48.0, 36.0], "mask": {"polygons": poly, "height": 48, "width": 64}}, source="syn", split="x")
    return s, Image.fromarray(arr)


def _grabcuts():
    out = [B.GrabCut(fake_grabcut_mask, FAKE_GRABCUT_ITERATIONS, "tests._w1_fakes.fake_grabcut_mask")]
    if HAS_WB:
        out.append(B.workbench_grabcut())
    return out


@pytest.mark.parametrize("gc", _grabcuts(), ids=lambda g: g.name)
def test_w1f_runs_on_synthetic_image(gc):
    s, im = _synthetic()
    box = {"id": s.id, "failure": None, "parsed": {"box": [12.0, 8.0, 52.0, 40.0]}}
    rec, t = B.w1f_episode(s, box, im, gc)
    assert rec["failure"] is None and rec["error"] is None and rec["score"]["iou"] > 0.8
    assert rec["box"] == [12.0, 8.0, 52.0, 40.0] and rec["answer"]["source"] == gc.name
    assert rec["answer"]["kind"] == "mask" and rec["answer"]["mask"]["area"] > 0 and len(t["tool_times"]) == 1
    again, _ = B.w1f_episode(s, box, im, gc)  # 迭代次数与种子固定：确定
    assert again["score"] == rec["score"] and again["answer"] == rec["answer"]
    # 框解析失败：空 mask，记框的失败原因，不调 GrabCut
    nob, t2 = B.w1f_episode(s, {"id": s.id, "failure": "no_box", "parsed": None}, im, gc)
    assert nob["failure"] == "no_box" and nob["score"]["inter"] == 0 and nob["answer"] is None and t2["tool_times"] == []
    # GrabCut 报错：退化的框（没有前景像素）与盖满全图的框（没有背景像素）都记 tool_error，按空 mask 计分
    for b in ([20.0, 10.0, 20.0, 30.0], [0.0, 0.0, 64.0, 48.0]):
        bad, _ = B.w1f_episode(s, {"id": s.id, "failure": None, "parsed": {"box": b}}, im, gc)
        assert bad["failure"] == B.FAILURE_TOOL_ERROR and "ValueError" in bad["error"] and bad["score"]["inter"] == 0


@pytest.mark.skipif(not HAS_WB, reason="DEV-0050（src/w1/workbench.py）还没合入")
def test_w1f_uses_the_workbench_function():
    from src.w1 import workbench as WB

    g = B.workbench_grabcut()
    assert g.fn is WB.grabcut_mask and g.iterations == WB.GRABCUT_ITERATIONS and g.name == CFG.baselines["w1f"]["function"]
    # 与工作台的 grabcut 工具同一个算法：同一个框，工具写出的 mask 与 W1-F 的逐位相同
    s, im = _synthetic()
    box = [12.0, 8.0, 52.0, 40.0]
    rec, _ = B.w1f_episode(s, {"id": s.id, "failure": None, "parsed": {"box": box}}, im, g)
    want = WB.grabcut_mask(np.asarray(im), box_abs=box, iterations=WB.GRABCUT_ITERATIONS)
    from src.executors.raster import mask_to_rle

    assert rec["answer"]["mask"]["rle"] == mask_to_rle(want)


# ---------------------------------------------------------------------------
# E1 的两处放宽：快照与新路径
# ---------------------------------------------------------------------------


def _same_outputs(a: Path, b: Path) -> None:
    for f in DETERMINISTIC:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_forced_poly_default_path_identical_to_eb550b99(tiny, tmp_path):
    assert hashlib.sha256(SNAP_POSTHOC.read_bytes()).hexdigest() == SNAP_POSTHOC_SHA256
    old = _load(SNAP_POSTHOC, "_e1_posthoc_runner_eb550b99")
    old.REPO_ROOT = PR.REPO_ROOT  # 快照按自己的位置推出的仓库根不对（只影响 meta.json 与配置文件的相对路径）
    prompts = load_prompt_config()
    posthoc = load_posthoc_config(prompts)
    paths = DataPaths.posthoc_fixtures(nyu_cache=tmp_path / "nyu")
    sel = select("refexp_poly", "think", paths)
    spec = _spec(tiny, "refexp_poly", "forced_poly")
    for mod, out in ((old, tmp_path / "old"), (PR, tmp_path / "new")):
        mod.run_forced_poly_task(tiny, spec, sel, prompts, posthoc, paths, out, variant="poly_structured", warmup=False,
                                 log=lambda m: None)
    leaf = Path(QWEN) / "refexp_poly__poly_structured"
    _same_outputs(tmp_path / "old" / leaf, tmp_path / "new" / leaf)
    cfg = yaml.safe_load((tmp_path / "new" / leaf / "config.yaml").read_text())
    assert cfg["posthoc"]["base_subset"] == "think"


def test_run_task_default_path_identical_to_eb550b99_and_leaf_override(tiny, tmp_path):
    assert hashlib.sha256(SNAP_RUNNER.read_bytes()).hexdigest() == SNAP_RUNNER_SHA256
    old = _load(SNAP_RUNNER, "_e1_runner_eb550b99")
    old.REPO_ROOT = R.REPO_ROOT
    prompts = load_prompt_config()
    paths = DataPaths.posthoc_fixtures(nyu_cache=tmp_path / "nyu")
    sel = select("refexp_box", "dev", paths)
    spec = _spec(tiny, "refexp_box", "dev", bs=2)
    old.run_task(tiny, spec, sel, prompts, paths, tmp_path / "old", warmup=False, log=lambda m: None)
    R.run_task(tiny, spec, sel, prompts, paths, tmp_path / "new", warmup=False, log=lambda m: None)
    leaf = Path(QWEN) / "refexp_box__dev"
    _same_outputs(tmp_path / "old" / leaf, tmp_path / "new" / leaf)
    other = tmp_path / "elsewhere" / "a" / "b"
    R.run_task(tiny, spec, sel, prompts, paths, tmp_path / "unused", warmup=False, log=lambda m: None, leaf=other)
    assert not (tmp_path / "unused").exists()
    _same_outputs(tmp_path / "new" / leaf, other)


def _w1_spec(split: str, setting: str) -> W1RunSpec:
    e = registry.get_model(QWEN)
    return W1RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=f"tiny({e.key})",
                     task="refseg", split=split, setting=setting, tiny=True, limit=None, image_policy={"tag": "default"})


def test_forced_poly_on_w1_split(tiny, tmp_path):
    prompts = load_prompt_config()
    posthoc = load_posthoc_config(prompts)
    paths = DataPaths.posthoc_fixtures(nyu_cache=tmp_path / "nyu")
    doc = tiny_splits(CFG)
    samples = w1_items("refseg", "devcheck", paths, splits=doc)
    ws = _w1_spec("devcheck", "baseline_poly_structured")
    sel = B.w1_selection("refexp_poly", "w1:devcheck", samples, w1_record(ws, doc, samples))
    spec = _spec(tiny, "refexp_poly", "forced_poly")
    leaf = tmp_path / "w1" / "leaf"
    m = PR.run_forced_poly_task(tiny, spec, sel, prompts, posthoc, paths, None, variant="poly_structured", warmup=False,
                                log=lambda m: None, sample_subset="w1:devcheck", leaf=leaf)
    assert m["n"] == len(samples) == 3 and m["complete"] is True
    cfg = yaml.safe_load((leaf / "config.yaml").read_text())
    assert cfg["posthoc"]["base_subset"] == "w1:devcheck" and cfg["data"]["selection_stats"]["w1"]["split"] == "devcheck"
    assert [r["id"] for r in R.read_jsonl_tolerant(leaf / "predictions.jsonl", repair=False)] == [s.id for s in samples]
    with pytest.raises(ValueError, match="思考子集以外"):  # sample_subset 不能拿来换回思考子集
        PR.run_forced_poly_task(tiny, spec, sel, prompts, posthoc, paths, tmp_path / "x", variant="poly_structured",
                                sample_subset="think")
    with pytest.raises(ValueError, match="样本取自"):  # 清单与 sample_subset 对不上
        PR.run_forced_poly_task(tiny, spec, sel, prompts, posthoc, paths, tmp_path / "y", variant="poly_structured",
                                sample_subset="w1:devtune")
    with pytest.raises(ValueError, match="样本取自"):  # 不给 sample_subset：照旧只认思考子集
        PR.run_forced_poly_task(tiny, spec, sel, prompts, posthoc, paths, tmp_path / "z", variant="poly_structured")
