"""DEV-0011：scripts/validate_batching.py（批处理核对，给服务器用）。

彩排（--tiny）端到端跑一条子进程（与服务器同一流程）；其余在进程内核对输入解析、分批规则与报告结构。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/validate_batching.py"


@pytest.fixture(scope="module")
def vb():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_validate_batching_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def test_parse_args_rules(vb):
    a = vb.parse_args(["--tiny", "--out", "x.json"])
    assert (a.device, a.dtype, a.batch_sizes, a.model) == ("cpu", "fp32", list(vb.TINY_BATCH_SIZES), None)
    a = vb.parse_args(["--model", "qwen3.5-2b", "--jsonl", "r.jsonl", "--batch-sizes", "8", "16", "--out", "x.json",
                       "--image-policy", "pixel_budget:1024"])
    assert (a.device, a.dtype, a.batch_sizes) == ("cuda", "bf16", [8, 16])
    for bad in (["--out", "x.json", "--model", "qwen3.5-2b"],  # 真模型缺 --jsonl / --batch-sizes
                ["--tiny", "--out", "x.json", "--batch-sizes", "0"],
                ["--tiny", "--out", "x.json", "--image-policy", "budget:3"],
                ["--tiny"]):
        with pytest.raises(SystemExit):
            vb.parse_args(bad)


def test_group_batches_keeps_order_within_same_flags(vb):
    rows = [{"enable_thinking": t, "max_new_tokens": n} for t, n in
            [(False, 16), (True, 16), (False, 16), (False, 8), (False, 16), (True, 16), (False, 16)]]
    assert vb.group_batches(rows, 2) == [[0, 2], [4, 6], [1, 5], [3]]
    assert vb.group_batches(rows, 8) == [[0, 2, 4, 6], [1, 5], [3]]
    assert sorted(i for b in vb.group_batches(rows, 3) for i in b) == list(range(len(rows)))


def test_first_diff(vb):
    assert vb.first_diff("abc", "abc") is None
    assert vb.first_diff("abc", "abd") == 2
    assert vb.first_diff("ab", "abc") == 2
    assert vb.first_diff((1, 2, 3), (1, 5)) == 1


def test_load_rows_validates_and_resolves_paths(vb, tmp_path):
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "a.png")
    good = [{"image": "a.png", "instruction": "x", "enable_thinking": False, "max_new_tokens": 4},
            {"id": "q", "image": ["a.png", "a.png"], "instruction": "y", "enable_thinking": True, "max_new_tokens": 2,
             "order": "image_first"},
            {"image": None, "instruction": "z", "enable_thinking": False, "max_new_tokens": 1}]
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in good) + "\n\n")
    rows = vb.load_rows(p)
    assert [r["id"] for r in rows] == ["0", "q", "2"]
    assert rows[0]["images"] == [str((tmp_path / "a.png").resolve())] and len(rows[1]["images"]) == 2
    assert rows[2]["images"] == [] and rows[1]["order"] == "image_first" and rows[0]["order"] == "instruction_first"
    assert len(vb.load_rows(p, limit=2)) == 2
    for bad in ({"instruction": "x", "enable_thinking": "no", "max_new_tokens": 4},
                {"instruction": "x", "enable_thinking": False, "max_new_tokens": 0},
                {"instruction": "x", "enable_thinking": False, "max_new_tokens": 4, "extra": 1},
                {"instruction": "x", "enable_thinking": False, "max_new_tokens": 4, "order": "sideways"},
                {"instruction": 3, "enable_thinking": False, "max_new_tokens": 4}):
        p.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ValueError):
            vb.load_rows(p)


def test_tiny_rehearsal_end_to_end(tmp_path):
    out = tmp_path / "vb.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--model", "gemma-4-e2b-it", "--out", str(out)],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert out.exists(), proc.stderr[-3000:]
    rep = json.loads(out.read_text())
    assert proc.returncode == 0 and rep["ok"] is True, rep.get("error")
    assert rep["model"]["image_policy"] == {"type": "SoftTokenBudget", "max_soft_tokens": 70}
    assert rep["warmup"]["counted"] is False and rep["jsonl"]["n_rows"] == 8
    assert set(rep["batched"]) == {"4", "8"}
    assert rep["batched"]["8"]["batch_row_counts"] == [6, 2]  # 按 (enable_thinking, max_new_tokens) 分组
    for b in rep["batched"].values():
        assert b["raw_text_equal_rate"] == 1.0 and b["token_ids_equal_rate"] == 1.0 and b["mismatches"] == []
        assert b["samples_per_s"] > 0 and b["tokens_per_s"] > 0
    assert rep["sequential"]["samples_per_s"] > 0
    assert len({tuple(map(tuple, r["grids_hw"])) for r in rep["rows"]}) >= 3  # 网格不止一种
    assert len({r["single"]["raw_text"] for r in rep["rows"]}) >= 4  # 小模型的输出随输入变化


def test_run_with_explicit_jsonl_in_process(vb, tmp_path):
    """--tiny 也能读给定的 JSONL（服务器上的路径），一行带两张图、一行没有图。"""
    rng = np.random.default_rng(0)
    for name, (h, w) in {"a": (90, 140), "b": (150, 80)}.items():
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(tmp_path / f"{name}.png")
    rows = [{"id": "x", "image": "a.png", "instruction": "Describe.", "enable_thinking": False, "max_new_tokens": 6},
            {"id": "y", "image": ["a.png", "b.png"], "instruction": "Compare them.", "enable_thinking": False,
             "max_new_tokens": 6},
            {"id": "z", "image": None, "instruction": "Hi", "enable_thinking": False, "max_new_tokens": 6}]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "vb.json"
    code = vb.main(["--tiny", "--model", "internvl3.5-2b-hf", "--jsonl", str(tmp_path / "rows.jsonl"),
                    "--batch-sizes", "3", "--image-policy", "tiles:true:6", "--out", str(out)])
    rep = json.loads(out.read_text())
    assert code == 0 and rep["ok"] is True
    assert rep["model"]["image_policy"] == {"type": "TilePolicy", "crop_to_patches": True, "max_patches": 6}
    assert [r["id"] for r in rep["rows"]] == ["x", "y", "z"] and rep["rows"][2]["n_image_pixels"] == 0
    assert rep["batched"]["3"]["raw_text_equal_rate"] == 1.0


def test_errors_are_written_before_exit(vb, tmp_path):
    out = tmp_path / "vb.json"
    (tmp_path / "rows.jsonl").write_text(json.dumps({"image": "missing.png", "instruction": "x",
                                                     "enable_thinking": False, "max_new_tokens": 2}) + "\n")
    code = vb.main(["--tiny", "--jsonl", str(tmp_path / "rows.jsonl"), "--out", str(out)])
    rep = json.loads(out.read_text())
    assert code == 2 and rep["ok"] is False and "missing.png" in rep["error"]
