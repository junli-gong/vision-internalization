"""DEV-0018：外部预测（LISA 式对照）的读入与评分，以及与训练数据的隔离（硬约束 1）。

隔离的两条腿：
1. 静态：src/ 与 scripts/ 下，只有 ALLOWED_READERS 里的文件可以 import src/e5/external.py 或引用它的标记文件名；
   生成器、E5 的协议 / 提示 / 评分、数据代码、以后的任何训练代码都不行（新文件自动纳入扫描）。
2. 动态：在打开文件、列目录、stat 的系统调用上设桩，跑一遍评测集的生成、提示渲染与基线评分（数据路径），
   核对没有任何路径落进外部预测目录；同一个桩下读外部预测本身会被记到（正对照，证明桩有效）。
"""

import builtins
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.e5 import external as EXT
from src.e5 import scoring as SC
from src.e5.prompts import load_prompt_config
from src.e5.protocol import E5_TASKS, eval_items
from src.e5.runner import RunSpec, make_sample, run_external_task
from src.executors.raster import mask_to_rle

REPO = Path(__file__).resolve().parents[1]


def _make_dir(root: Path, system="lisa-7b-v1") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / EXT.EXTERNAL_MARKER).write_text(json.dumps({"format_version": 1, "system": system}), encoding="utf-8")
    return root


def test_load_rle_png_failure_missing_extra(tmp_path):
    root = _make_dir(tmp_path / "ext")
    m1 = np.zeros((32, 48), bool)
    m1[3:9, 4:20] = True
    m2 = np.zeros((32, 48), bool)
    m2[10:20, 30:40] = True
    (root / "masks").mkdir()
    Image.fromarray((m2 * 255).astype(np.uint8)).save(root / "masks" / "b.png")
    lines = [{"id": "occlusion/test/0", "mask_rle": mask_to_rle(m1)}, {"id": "occlusion/test/1", "mask_png": "masks/b.png"},
             {"id": "occlusion/test/2", "failure": "no_mask"}, {"id": "occlusion/test/99", "failure": "x"}]
    (root / "occlusion.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    ids = [f"occlusion/test/{i}" for i in range(4)]
    got, st = EXT.load_task(root, "occlusion", ids, (32, 48))
    assert np.array_equal(got[ids[0]], m1) and np.array_equal(got[ids[1]], m2)
    assert got[ids[2]] == "no_mask" and got[ids[3]] == EXT.MISSING
    assert st["n_missing"] == 1 and st["n_extra_ignored"] == 1 and st["system"] == "lisa-7b-v1"
    with pytest.raises(EXT.ExternalPredictionError):
        EXT.load_task(root, "occlusion", ids, (32, 32))  # 尺寸不对
    (root / "imagery.jsonl").write_text(json.dumps(lines[0]) + "\n" + json.dumps(lines[0]) + "\n", encoding="utf-8")
    with pytest.raises(EXT.ExternalPredictionError):
        EXT.load_task(root, "imagery", ids, (32, 48))  # 重复 id
    (root / "physics.jsonl").write_text(json.dumps({"id": ids[0], "mask_png": "../outside.png"}) + "\n", encoding="utf-8")
    with pytest.raises(EXT.ExternalPredictionError):
        EXT.load_task(root, "physics", ids, (32, 48))  # PNG 路径不能出目录
    with pytest.raises(EXT.ExternalPredictionError):
        EXT.read_manifest(tmp_path)  # 没有标记文件


def test_external_run_scores_only(tmp_path):
    root = _make_dir(tmp_path / "ext")
    items = eval_items("rotation3d", "dev", 1)
    samples = [make_sample(it) for it in items]
    lines = [{"id": items[0].id, "mask_rle": mask_to_rle(samples[0].target_mask)},  # 完美
             {"id": items[1].id, "mask_rle": mask_to_rle(samples[1].baselines["copy"].mask)},  # 只会分割当前图像
             {"id": items[2].id, "failure": "lisa_error"}]  # 第 4 条缺失
    (root / "rotation3d.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    spec = RunSpec(system="external", model_key="lisa-7b-v1", task="rotation3d", subset="dev", n_per_bin=1, n_resamples=0,
                   external_dir=str(root))
    m = run_external_task(spec, tmp_path / "out")
    leaf = tmp_path / "out" / "external" / "lisa-7b-v1" / "rotation3d__dev"
    recs = [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines()]
    assert recs[0]["score"]["iou_full"] == 1.0
    assert recs[1]["score"]["iou_full"] == samples[1].meta["baseline_iou"]["copy"]["full"]
    assert recs[2]["failure"] == "external_failure" and recs[3]["failure"] == "external_missing"
    assert all(r["score"]["relief"] is None for r in recs)
    assert m["parse"]["n_failed"] == 2 and m["gate"]["judgment"]["status"] == "not_applicable"
    cfg = (leaf / "config.yaml").read_text()
    assert "marker_sha256" in cfg and (leaf / "compute.json").exists()
    bad = RunSpec(system="external", model_key="other", task="rotation3d", subset="dev", n_per_bin=1, n_resamples=0,
                  external_dir=str(root))
    with pytest.raises(ValueError):
        run_external_task(bad, tmp_path / "out2")


def test_export_inputs(tmp_path):
    samples = [make_sample(it) for it in eval_items("viewpoint", "dev", 1)[:2]] + [make_sample(eval_items("imagery", "dev", 1)[0])]
    p = EXT.export_inputs(samples, tmp_path / "inputs")
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["viewpoint/val/0", "viewpoint/val/1", "imagery/val/0"]
    assert rows[2]["image"] is None and rows[0]["target_description"] == samples[0].meta["target_description"]
    with Image.open(tmp_path / "inputs" / rows[0]["image"]) as im:
        assert np.array_equal(np.asarray(im), samples[0].image)


# ---------------------------------------------------------------------------
# 隔离（硬约束 1）
# ---------------------------------------------------------------------------

_REF = re.compile(r"(src\.e5\.external|from src\.e5 import[^\n]*\bexternal\b|src/e5/external|" + re.escape(EXT.EXTERNAL_MARKER)
                  + r"|EXTERNAL_MARKER)")


def test_static_only_allowed_files_reference_external_predictions():
    offenders = []
    for base in ("src", "scripts"):
        for path in sorted((REPO / base).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel in EXT.ALLOWED_READERS:
                continue
            if _REF.search(path.read_text(encoding="utf-8")):
                offenders.append(rel)
    assert not offenders, f"这些文件引用了外部预测（硬约束 1：只有 {EXT.ALLOWED_READERS} 可以）：{offenders}"
    # 允许的读者确实存在（清单没有写错）
    assert all((REPO / f).exists() for f in EXT.ALLOWED_READERS)
    assert not any(f.startswith(("src/synth", "src/data", "src/rl", "src/readout", "src/probes")) for f in EXT.ALLOWED_READERS)


class _Probe:
    """记下所有被打开 / 列出 / stat 的路径。"""

    def __init__(self, monkeypatch):
        self.paths: list[str] = []
        for mod, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "listdir"), (os, "scandir"), (os, "stat")):
            orig = getattr(mod, name)
            monkeypatch.setattr(mod, name, self._wrap(orig))

    def _wrap(self, orig):
        def f(path=".", *a, **k):
            if isinstance(path, (str, bytes, os.PathLike)):
                self.paths.append(os.path.abspath(os.fsdecode(path)))
            return orig(path, *a, **k)

        return f

    def touched(self, root: str) -> list[str]:
        """root 是设桩之前就解析好的绝对路径（在桩下再 resolve 会把 root 自己记进去）。"""
        return [p for p in self.paths if p == root or p.startswith(root + os.sep)]


def test_dynamic_data_path_never_reads_the_external_directory(tmp_path, monkeypatch):
    root = _make_dir(tmp_path / "external_predictions")
    for t in E5_TASKS:
        (root / f"{t}.jsonl").write_text("", encoding="utf-8")
    prompts = load_prompt_config()
    roots = {os.path.abspath(root), str(root.resolve())}  # macOS 的 /var 与 /private/var
    probe = _Probe(monkeypatch)
    for t in E5_TASKS:  # 数据路径：评测集的生成、提示渲染、基线评分
        samples = [make_sample(it) for it in eval_items(t, "dev", 1)[:2]]
        for s in samples:
            prompts.render("zeroshot", t, s.instruction, s.meta["target_description"])
            prompts.render("cot", t, s.instruction, s.meta["target_description"])
        SC.baseline_records(samples)
    assert [p for r in roots for p in probe.touched(r)] == []
    EXT.load_task(root, "occlusion", ["occlusion/val/0"], (512, 512))  # 正对照：读外部预测会被记到
    assert [p for r in roots for p in probe.touched(r)]
