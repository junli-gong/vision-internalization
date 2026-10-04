"""DEV-0035 §3(a)（D68-4(a)、D66-3）：E1 的事后读数接训练后的检查点——run_e1.py --subset debiased --checkpoint 与 --subset free
--checkpoint（--tiny：训练框架产出的小检查点，CPU fp32，与服务器同一流程的子进程）。

- 去偏读数：四个任务跑通；加载方式同 T 文本评测（先合并核对，再 load_trained(merge=True)），结果在 <out>/<标签>/<模型>/ 之下；
  config.yaml 另记 checkpoint（manifest 的 sha256），除 checkpoint 与图像口径（检查点自己的口径）之外与 base 的 config.yaml 相同；
  README 另记一行；运行记录里有合并核对的摘要；
- free：同一个标签下复用已通过的合并核对，config.yaml 记 checkpoint，记录里有网格 token 的计数；
- 格式强制仍只用于 base 的 Qwen：--subset forced 配 --checkpoint 是用法错误；
- base 运行：没有 checkpoint 键、README 没有检查点一行（base 的输出与改动前逐字节相同另在开发机上核对过，见完成总结）。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
QWEN = "qwen3.5-0.8b"
LETTER_TASKS = ("rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial")
LABEL = "t_text/e3b/seed0"


def run(script: str, args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, str(REPO / "scripts" / f"{script}.py"), *args], capture_output=True, text=True, cwd=REPO,
                       env=env, timeout=1200)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


@pytest.fixture(scope="module")
def ck(tmp_path_factory):
    d = tmp_path_factory.mktemp("posthoc_ck")
    run("train_e3b", ["--tiny", "--model", QWEN, "--supervision", "grid", "--seed", "0", "--out", str(d / "train")], d)
    return {"ck": d / "train" / "e3b" / QWEN / "seed0" / "ckpt" / "final" / "model", "tmp": d}


def test_debiased_and_free_on_checkpoint(ck, tmp_path):
    out, base_out = tmp_path / "o", tmp_path / "b"
    common = ["--tiny", "--model", QWEN, "--nyu-cache", str(tmp_path / "nyu")]
    p = run("run_e1", [*common, "--task", "all", "--subset", "debiased", "--checkpoint", str(ck["ck"]), "--checkpoint-label", LABEL,
                       "--out", str(out)], tmp_path)
    assert p.stdout.count("/debiased n=") == 4
    root = out / LABEL / QWEN
    sha = hashlib.sha256((ck["ck"] / "manifest.json").read_bytes()).hexdigest()
    mc = json.loads((root / "merge_check.json").read_text())
    assert mc["pass"] is True and list(mc["checks"]) == ["merge"]  # 与 T 文本评测同一种核对
    run("run_e1", [*common, "--task", "all", "--subset", "debiased", "--out", str(base_out)], tmp_path)
    for task in LETTER_TASKS:
        leaf = root / f"{task}__debiased"
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["checkpoint"]["manifest_sha256"] == sha and cfg["checkpoint"]["label"] == LABEL
        assert cfg["checkpoint"]["n_grid_tokens"] == 67 and cfg["checkpoint"]["merge_check"]["checks"] == ["merge"]
        bcfg = yaml.safe_load((base_out / QWEN / f"{task}__debiased" / "config.yaml").read_text())
        assert "checkpoint" not in bcfg
        drop = ("checkpoint", "image_policy")  # 检查点用它自己的图像口径（--tiny 时照记；真模型必须就是 E1 冻结的口径）
        assert {k: v for k, v in cfg.items() if k not in drop} == {k: v for k, v in bcfg.items() if k not in drop}, task
        m = json.loads((leaf / "metrics.json").read_text())
        assert m["complete"] is True and m["posthoc"]["kind"] == "debiased_letter"
        assert f"标签 {LABEL}" in (leaf / "README.md").read_text()
        assert "训练后的检查点" not in (base_out / QWEN / f"{task}__debiased" / "README.md").read_text()
        rl = json.loads((leaf / "logs" / "run_001.json").read_text())
        assert rl["merge_check"]["pass"] is True and rl["checkpoint_dir"] == str(ck["ck"])
        n_read = len((leaf / "readings.jsonl").read_text().splitlines())
        assert n_read == len((base_out / QWEN / f"{task}__debiased" / "readings.jsonl").read_text().splitlines())
    # free（固定 200 条的自由作答；这里是 fixture 的 think id）：同一个标签，复用已通过的合并核对
    p = run("run_e1", [*common, "--task", "rel_depth_nyu", "spatial_cvbench", "--subset", "free", "--checkpoint", str(ck["ck"]),
                       "--checkpoint-label", LABEL, "--out", str(out)], tmp_path)
    assert "复用已通过" in p.stderr and p.stdout.count("/free n=") == 2
    for task in ("rel_depth_nyu", "spatial_cvbench"):
        leaf = root / f"{task}__free"
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["checkpoint"]["manifest_sha256"] == sha and cfg["subset"] == "free"
        recs = [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines()]
        assert recs and all("n_grid_tokens" in r for r in recs)


def test_forced_refuses_checkpoint(ck, tmp_path):
    p = run("run_e1", ["--tiny", "--model", QWEN, "--task", "all", "--subset", "forced", "--checkpoint", str(ck["ck"]),
                       "--checkpoint-label", LABEL, "--out", str(tmp_path / "f")], tmp_path, ok=False)
    assert p.returncode == 2 and "只用于 base 的 Qwen" in p.stderr and not (tmp_path / "f").exists()
    p = run("run_e1", ["--tiny", "--model", QWEN, "--task", "all", "--subset", "debiased", "--checkpoint", str(ck["ck"]),
                       "--out", str(tmp_path / "g")], tmp_path, ok=False)
    assert p.returncode == 2 and "--checkpoint-label" in p.stderr  # 标签照样必给
