"""DEV-0025 验收：scripts/train_e3b.py --tiny（真实的小型随机模型，CPU，tests/fixtures 的小数据；与服务器同一流程的子进程）。

- 三家族 × 两种监督（grid = E3b，text = C3）跑通：输出目录 <out>/<变体>/<模型>/seed<k>/ 的文件齐全，最终检查点只有 LoRA、token 行与
  manifest，续跑状态完成后删掉；
- 分段运行（--stop-after-steps）后 --resume 与一次跑完逐字节相同；
- 命令行的拒绝条件：非空目录不加 --resume、种子不在配置里、真模型缺 --data-root 或目录名没有运行编号。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.e3.config import load_config, resolve_run

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "train_e3b.py"
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
DETERMINISTIC = ("config.yaml", "train_log.jsonl", "README.md", "train_summary.json", "tokens_init.json")
VARIANT = {"grid": "e3b", "text": "c3"}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO)
    if check and p.returncode != 0:
        raise AssertionError(f"train_e3b {args} 退出码 {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p


@pytest.mark.parametrize("supervision", ["grid", "text"])
def test_tiny_three_families_and_staged_resume(supervision, tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    base = ["--tiny", "--model", *FAMILIES, "--supervision", supervision, "--seed", "0"]
    p = run(*base, "--out", str(a))
    assert p.stdout.count("steps=") == 3
    p = run(*base, "--out", str(b), "--stop-after-steps", "3")
    assert "按 --stop-after-steps 停下" in p.stdout and not (b / VARIANT[supervision] / FAMILIES[1]).exists()
    for _ in FAMILIES:  # 第一个模型续跑完成后，后面的模型从头跑
        run(*base, "--out", str(b), "--resume")
    rc = resolve_run(load_config(), supervision=supervision, tiny=True)
    for key in FAMILIES:
        la, lb = a / VARIANT[supervision] / key / "seed0", b / VARIANT[supervision] / key / "seed0"
        for f in DETERMINISTIC:
            if f == "tokens_init.json" and supervision == "text":
                assert not (la / f).exists()
                continue
            assert (la / f).read_bytes() == (lb / f).read_bytes(), f"{key} {f} 分段续跑后与一次跑完不同"
        for leaf in (la, lb):
            assert {p.name for p in leaf.iterdir()} >= {"config.yaml", "meta.json", "train_log.jsonl", "accounting.json",
                                                         "compute.json", "README.md", "train_summary.json", "logs", "ckpt"}
            assert sorted(p.name for p in (leaf / "ckpt").iterdir()) == ["final"]  # 续跑状态完成后删掉
            files = sorted(str(p.relative_to(leaf / "ckpt" / "final" / "model")) for p in (leaf / "ckpt" / "final" / "model").rglob("*")
                           if p.is_file())
            assert files == ["lora/adapter_config.json", "lora/adapter_model.safetensors", "manifest.json",
                             "token_rows.safetensors", "tokens.json"]
        fa = (la / "ckpt" / "final" / "model")
        fb = (lb / "ckpt" / "final" / "model")
        for f in ("lora/adapter_model.safetensors", "token_rows.safetensors", "tokens.json"):
            assert (fa / f).read_bytes() == (fb / f).read_bytes(), f"{key} {f}"
        ident = yaml.safe_load((la / "config.yaml").read_text())
        assert ident["supervision"] == supervision and ident["training"]["steps"] == rc["steps"]
        assert (ident["tokens"] is None) == (supervision == "text")
        if supervision == "grid":
            assert len(ident["tokens"]["vocab"]) == 67 and ident["tokens"]["budget"]["n_total_planned"] <= 300
        log = [json.loads(l) for l in (la / "train_log.jsonl").read_text().splitlines()]
        assert len(log) == rc["steps"] and all(r["n_samples"] == rc["batch_size"] for r in log)
        runs = sorted((lb / "logs").glob("run_*.json"))
        assert len(runs) == (2 if key == FAMILIES[0] else 1)


def test_cli_rejections(tmp_path):
    base = ["--tiny", "--model", "qwen3.5-0.8b", "--supervision", "grid"]
    p = run(*base, "--seed", "7", "--out", str(tmp_path / "x"), check=False)
    assert p.returncode == 2 and "seeds" in p.stderr
    p = run("--model", "qwen3.5-0.8b", "--supervision", "grid", "--seed", "0", "--out", str(tmp_path / "0040_x"), check=False)
    assert p.returncode == 2 and "--data-root" in p.stderr
    p = run("--model", "qwen3.5-0.8b", "--supervision", "grid", "--seed", "0", "--data-root", str(tmp_path),
            "--out", str(tmp_path / "noid"), check=False)
    assert p.returncode == 2 and "运行编号" in p.stderr
    out = tmp_path / "y"
    run(*base, "--seed", "0", "--out", str(out), "--stop-after-steps", "1")
    p = run(*base, "--seed", "0", "--out", str(out), check=False)
    assert p.returncode == 2 and "--resume" in p.stderr
    assert list((out / "e3b" / "qwen3.5-0.8b" / "errors").glob("error_*.json"))
