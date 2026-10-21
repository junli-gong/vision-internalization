"""DEV-0051：scripts/run_w1.py 的命令行彩排（--tiny，子进程，与服务器同一流程）与 scripts/w1_gate.py。

- 回路（full / no_eye）用假适配器与假工作台（--adapter-factory、--workbench，只配 --tiny；DEV-0049 / 0050 合入前）：输出齐全，运行身份记下
  协议、提示配置与划分清单的 sha256、工具清单的 sha256；三个任务都跑通；--resume 从一组中间截断处续跑，与一次跑完逐字节相同；
  不给 --resume 而目录非空拒绝覆盖（退出码 2）。
- 基线用真实的小型随机 Qwen："自己的框"（框没跑就先按 E1 的协议跑框）、结构强制的多边形、计数的 W1-0；W1-F（框已写完，不再跑模型；
  要 DEV-0050 的 grabcut_mask，合入之前跳过）。同一个划分上各目录的 id 相同，能配对。w1_gate.py 在这些输出上跑通，结论标为彩排。
- 命令行规则：假适配器 / 假工作台只配 --tiny；划分与任务对不上、基线与任务对不上都报错。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_w1.py"
GATE = REPO / "scripts" / "w1_gate.py"
QWEN = "qwen3.5-0.8b"
FAKES = ["--adapter-factory", "tests._w1_fakes:make_policy_adapter", "--workbench", "tests._w1_fakes:FakeWorkbench"]
FILES = ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "README.md", "meta.json")
DETERMINISTIC = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")


def run(args, tmp: Path, *, ok: bool = True, script: Path = SCRIPT) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def w1(task, split, setting, out, *extra, tiny=True):
    return (["--tiny"] if tiny else []) + ["--model", QWEN, "--task", task, "--split", split, "--setting", setting,
                                           "--out", str(out), *extra]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def leaf(out: Path, task: str, setting: str, split: str) -> Path:
    return out / QWEN / task / setting / split


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w1cli")
    out = tmp / "out"
    run(w1("refseg", "devcheck", "full", out, "--chunk-size", "2", *FAKES), tmp)
    run(w1("refexp_box", "test_refcocog_umd_test", "full", out, *FAKES), tmp)
    run(w1("count", "test_countbench", "no_eye", out, *FAKES), tmp)
    run(w1("refseg", "devcheck", "baseline_ownbox", out), tmp)  # 框还没跑：先按 E1 的协议跑 W1-0 refexp_box
    run(w1("refseg", "devcheck", "baseline_poly_structured", out), tmp)
    run(w1("count", "test_countbench", "baseline_e1", out), tmp)
    return {"tmp": tmp, "out": out}


def test_loop_outputs_and_identity(runs):
    from src.w1.config import load_w1_config
    from src.w1.data import splits_sha256, tiny_splits

    lf = leaf(runs["out"], "refseg", "full", "devcheck")
    for f in FILES + ("logs/timing.jsonl", "logs/workbench.jsonl"):
        assert (lf / f).exists(), f
    recs = jsonl(lf / "predictions.jsonl")
    m = json.loads((lf / "metrics.json").read_text())
    assert m["n"] == len(recs) == 3 and m["complete"] is True and [h["name"] for h in m["headline"]] == ["cIoU", "gIoU"]
    cfg = yaml.safe_load((lf / "config.yaml").read_text())
    w = load_w1_config()
    assert cfg["prompts"]["config_sha256"] == w.sha256 and cfg["w1"]["splits_sha256"] == splits_sha256(tiny_splits(w))
    assert cfg["protocol"]["max_rounds"] == 6 and cfg["protocol"]["max_tool_calls"] == 8
    assert cfg["protocol"]["thinking_budget"] == w.data["tiny"]["protocol"]["thinking_budget"]  # --tiny 只缩小预算
    assert len(cfg["workbench"]["tools_sha256"]) == 64 and "submit_mask" in cfg["workbench"]["tools"]
    assert cfg["batching"] == {"batch_size": 2, "chunk_size": 2, "max_batch_tokens": 48000, "config_default": True}
    assert cfg["model"]["revision"] and cfg["w1"]["setting"] == "full"
    for r in recs:
        assert r["setting"] == "full" and r["w1_split"] == "devcheck" and r["rounds"] and "score" in r
        assert all("wall" not in k for k in r)
    comp = json.loads((lf / "compute.json").read_text())
    assert comp["n_samples"] == 3 and comp["n_rounds"] == sum(r["n_rounds"] for r in recs) and comp["image_pixels"] > 0
    timing = jsonl(lf / "logs/timing.jsonl")
    assert [t["chunk_index"] for t in timing] == [0, 0, 1]  # 组大小 2：两组
    for task, setting, split, n in (("refexp_box", "full", "test_refcocog_umd_test", 4), ("count", "no_eye", "test_countbench", 2)):
        lx = leaf(runs["out"], task, setting, split)
        mx = json.loads((lx / "metrics.json").read_text())
        assert mx["n"] == n and mx["complete"] is True
        tools = yaml.safe_load((lx / "config.yaml").read_text())["workbench"]["tools"]
        assert ("zoom" in tools) == (setting != "no_eye")  # A1 关掉了眼


def test_loop_resume_is_byte_identical(runs):
    lf = leaf(runs["out"], "refseg", "full", "devcheck")
    ref = {f: (lf / f).read_bytes() for f in DETERMINISTIC}
    lines = ref["predictions.jsonl"].splitlines(keepends=True)
    (lf / "predictions.jsonl").write_bytes(lines[0] + lines[1][:40])  # 截在第一组（两条）的中间
    (lf / "metrics.json").unlink()
    p = run(w1("refseg", "devcheck", "full", runs["out"], "--chunk-size", "2", "--resume", *FAKES), runs["tmp"])
    assert "退回到 0" in p.stderr  # 整组重跑
    for f in DETERMINISTIC:
        assert (lf / f).read_bytes() == ref[f], f
    assert len(list((lf / "logs").glob("run_*.json"))) == 2
    # 不给 --resume：拒绝覆盖
    bad = run(w1("refseg", "devcheck", "full", runs["out"], "--chunk-size", "2", *FAKES), runs["tmp"], ok=False)
    assert bad.returncode == 2 and "已存在且非空" in bad.stderr
    # 续跑换了组大小：身份不同，拒绝混写
    bad = run(w1("refseg", "devcheck", "full", runs["out"], "--chunk-size", "3", "--resume", *FAKES), runs["tmp"], ok=False)
    assert bad.returncode == 2 and "配置与已有的不同" in bad.stderr


def test_baselines_pair_with_the_loop(runs):
    out = runs["out"]
    ids = [r["id"] for r in jsonl(leaf(out, "refseg", "full", "devcheck") / "predictions.jsonl")]
    box = leaf(out, "refexp_box", "baseline_e1", "devcheck")
    assert [r["id"] for r in jsonl(box / "predictions.jsonl")] == ids  # "自己的框"先跑了框
    bcfg = yaml.safe_load((box / "config.yaml").read_text())
    assert bcfg["experiment"] == "E1" and bcfg["task"] == "refexp_box" and bcfg["generation"]["enable_thinking"] is False
    assert bcfg["data"]["selection_stats"]["w1"]["split"] == "devcheck" and bcfg["batch_protocol"]["batch_size"] == 4
    for setting in ("baseline_ownbox", "baseline_poly_structured"):
        lx = leaf(out, "refseg", setting, "devcheck")
        recs = jsonl(lx / "predictions.jsonl")
        assert [r["id"] for r in recs] == ids, setting
        assert all({"inter", "union", "iou"} <= set(r["score"]) for r in recs)
        m = json.loads((lx / "metrics.json").read_text())
        assert m["complete"] is True and m["headline"][0]["name"] == "cIoU", setting
    own = yaml.safe_load((leaf(out, "refseg", "baseline_ownbox", "devcheck") / "config.yaml").read_text())
    assert own["box_run"]["leaf"] == f"<out>/{QWEN}/refexp_box/baseline_e1/devcheck" and own["w1"]["setting"] == "baseline_ownbox"
    poly = yaml.safe_load((leaf(out, "refseg", "baseline_poly_structured", "devcheck") / "config.yaml").read_text())
    assert poly["posthoc"]["base_subset"] == "w1:devcheck" and poly["generation"]["readout"]["kind"] == "poly_structured"
    cnt = leaf(out, "count", "baseline_e1", "test_countbench")
    assert json.loads((cnt / "metrics.json").read_text())["headline"][0]["name"] == "accuracy"
    assert [r["id"] for r in jsonl(cnt / "predictions.jsonl")] == \
        [r["id"] for r in jsonl(leaf(out, "count", "no_eye", "test_countbench") / "predictions.jsonl")]


@pytest.mark.skipif(importlib.util.find_spec("src.w1.workbench") is None, reason="DEV-0050（src/w1/workbench.py）还没合入")
def test_w1f_reuses_the_box_run(runs):
    """W1-F：框已写完，直接读（不加载模型），框 → 工作台的 grabcut_mask；与回路同一批 id。"""
    out = runs["out"]
    box = leaf(out, "refexp_box", "baseline_e1", "devcheck")
    before = (box / "predictions.jsonl").read_bytes()
    p = run(w1("refseg", "devcheck", "w1f", out), runs["tmp"])
    assert "先跑框" not in p.stderr and (box / "predictions.jsonl").read_bytes() == before
    lx = leaf(out, "refseg", "w1f", "devcheck")
    recs = jsonl(lx / "predictions.jsonl")
    assert [r["id"] for r in recs] == [r["id"] for r in jsonl(leaf(out, "refseg", "full", "devcheck") / "predictions.jsonl")]
    cfg = yaml.safe_load((lx / "config.yaml").read_text())
    assert cfg["w1f"]["function"] == "src.w1.workbench.grabcut_mask" and cfg["box_run"]["leaf"].endswith("refexp_box/baseline_e1/devcheck")
    m = json.loads((lx / "metrics.json").read_text())
    assert m["complete"] is True and m["headline"][0]["name"] == "cIoU"


def test_gate_on_tiny_outputs(runs, tmp_path):
    out = runs["out"]
    args = ["--w1", str(leaf(out, "refseg", "full", "devcheck")), "--ownbox", str(leaf(out, "refseg", "baseline_ownbox", "devcheck")),
            "--poly", str(leaf(out, "refseg", "baseline_poly_structured", "devcheck")), "--out-json", str(tmp_path / "g.json")]
    p = run(args, runs["tmp"], script=GATE)
    doc = json.loads((tmp_path / "g.json").read_text())
    assert doc["gate_point"]["is_gate_point"] is False and doc["verdict"].startswith("彩排，不作数") and "彩排" in p.stdout
    assert doc["n"] == 3 and doc["decision"] in ("继续", "放弃")
    bad = run(["--w1", args[1], "--ownbox", args[1], "--poly", args[5]], runs["tmp"], script=GATE, ok=False)
    assert bad.returncode == 2 and "baseline_ownbox" in bad.stderr


@pytest.mark.parametrize("make, msg", [
    (lambda o: w1("refseg", "devcheck", "full", o, *FAKES, tiny=False) + ["--data-root", "/x"], "只配 --tiny"),
    (lambda o: w1("count", "devcheck", "full", o, *FAKES), "不用于 count"),
    (lambda o: w1("count", "test_countbench", "baseline_ownbox", o), "只用于"),
    (lambda o: w1("refseg", "devcheck", "full", o, tiny=False) + ["--data-root", "/x"], "运行编号"),
])
def test_cli_rules(tmp_path, make, msg):
    p = run(make(tmp_path / "no_run_number"), tmp_path, ok=False)
    assert p.returncode == 2 and msg in p.stderr
