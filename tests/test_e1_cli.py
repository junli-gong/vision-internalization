"""DEV-0014：scripts/run_e1.py 的 --tiny 彩排（真实的小型随机模型，CPU，tests/fixtures/e1 的数据；与服务器同一流程的子进程）。

- 每个任务一条端到端（三个家族各跑三个任务），输出文件齐全；
- 真实小模型上的断点续传逐字节一致（删掉最后几条与半行再 --resume）；
- 批处理验收在 CPU 上一致率 100%、raw_text 逐字相同 100%；
- 估算模式写出 estimate.json 与表；
- 命令行参数的规则（进程内）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_e1.py"
FILES = ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "README.md", "meta.json")
GROUPS = {  # 三个家族各跑三个任务，合起来覆盖全部 9 个
    "qwen3.5-0.8b": ["refexp_box", "grid16", "rel_depth_nyu"],
    "gemma-4-e2b-it": ["ovd", "count", "spatial_cvbench"],
    "internvl3.5-2b-hf": ["refexp_poly", "rel_depth_blink", "spatial_embspatial"],
}


@pytest.fixture(scope="module")
def cli():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_run_e1_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def run(args, tmp: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp)}
    p = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--nyu-cache", str(tmp / "nyu"), *args],
                       capture_output=True, text=True, cwd=REPO, env=env, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    return p


@pytest.mark.parametrize("model", sorted(GROUPS))
def test_tiny_end_to_end_each_task_and_resume_bytes(model, tmp_path):
    tasks = GROUPS[model]
    out = tmp_path / "run"
    run(["--model", model, "--task", *tasks, "--limit", "3", "--batch-size", "2", "--out", str(out)], tmp_path)
    for t in tasks:
        leaf = out / model / t
        for f in FILES:
            assert (leaf / f).exists(), (t, f)
        m = json.loads((leaf / "metrics.json").read_text())
        cfg = (leaf / "config.yaml").read_text()
        assert m["model"] == model and m["task"] == t and m["prompt_config_sha256"] in cfg
        assert m["parse"]["n"] == m["n"] >= 1 and m["diagnostics"]["image_pixels"]["min"] > 0
    # 续跑：删掉第一个任务的后几条（并留下半行），--resume 之后与原来逐字节相同
    t = tasks[0]
    leaf = out / model / t
    ref = {f: (leaf / f).read_bytes() for f in ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")}
    lines = ref["predictions.jsonl"].splitlines(keepends=True)
    if len(lines) >= 2:
        (leaf / "predictions.jsonl").write_bytes(lines[0] + lines[1][:30])
        run(["--model", model, "--task", t, "--limit", "3", "--batch-size", "2", "--out", str(out), "--resume"], tmp_path)
        for f, b in ref.items():
            assert (leaf / f).read_bytes() == b, f


def test_tiny_acceptance_all_tasks_cpu_agreement_is_total(tmp_path):
    out = tmp_path / "acc"
    run(["--acceptance", "--model", "qwen3.5-0.8b", "--task", "all", "--batch-size", "3", "--limit", "3", "--out", str(out)],
        tmp_path)
    base = out / "qwen3.5-0.8b" / "acceptance"
    doc = json.loads((base / "acceptance.json").read_text())
    assert doc["all_pass"] and len(doc["tasks"]) == 9 and doc["agreement"] == "bitwise"  # 默认口径（D32）
    for r in doc["tasks"]:
        assert r["answer_agreement"] == 1.0 and r["raw_text_equal_rate"] == 1.0, r["task"]
        assert r["diff"] in (0.0, None)
        for m in ("bitwise", "iou90"):
            assert r["agreement_by_mode"][m]["rate"] == 1.0 and r["agreement_by_mode"][m]["pass"], (r["task"], m)
        oc = r["agreement_by_mode"]["outcome"]
        assert (oc["rate"], oc["pass"]) == ((None, None) if r["task"] == "ovd" else (1.0, True)), r["task"]
    assert {m: v["all_pass"] for m, v in doc["all_pass_by_agreement"].items()} == {"bitwise": True, "iou90": True,
                                                                                   "outcome": True}
    assert doc["all_pass_by_agreement"]["outcome"]["not_evaluated"] == ["ovd"]  # 检测的开发子集没有 GT
    md = (base / "acceptance.md").read_text()
    assert md.count("\n| ") >= 10  # 表头 + 9 行
    assert "(a) 逐位 一致率" in md and "(b) IoU ≥ 0.9 一致率" in md and "(c) 按判定结果 一致率" in md
    assert (base / "bs1" / "qwen3.5-0.8b" / "count__dev" / "predictions.jsonl").exists()
    # 换口径重算：--resume 跳过已完成的两遍，只重新比较（不再生成）
    leaf3 = base / "bs3" / "qwen3.5-0.8b" / "refexp_box__dev"
    before = (leaf3 / "predictions.jsonl").read_bytes()
    for mode in ("iou90", "outcome"):
        run(["--acceptance", "--model", "qwen3.5-0.8b", "--task", "all", "--batch-size", "3", "--limit", "3", "--out",
             str(out), "--resume", "--agreement", mode], tmp_path)
        d = json.loads((base / "acceptance.json").read_text())
        assert d["agreement"] == mode and d["all_pass"] is True and d["tasks"][0]["agreement"] == mode
        assert d["not_evaluated"] == (["ovd"] if mode == "outcome" else [])
        assert "（本次采用）" in (base / "acceptance.md").read_text()
    assert (leaf3 / "predictions.jsonl").read_bytes() == before
    runs = sorted((leaf3 / "logs").glob("run_*.json"))
    assert len(runs) == 3 and all(json.loads(p.read_text())["n_remaining"] == 0 for p in runs[1:])


def test_tiny_count_with_countbenchqa_questions(tmp_path):
    """--count-questions countbenchqa：fixture 里落地的人工问句进指令（tests/fixtures/e1/root/countbenchqa_questions）。"""
    out = tmp_path / "cq"
    run(["--model", "gemma-4-e2b-it", "--task", "count", "--limit", "2", "--batch-size", "2", "--count-questions",
         "countbenchqa", "--out", str(out)], tmp_path)
    leaf = out / "gemma-4-e2b-it" / "count"
    recs = [json.loads(line) for line in (leaf / "predictions.jsonl").read_text().splitlines()]
    qs = json.loads((REPO / "tests/fixtures/e1/root/countbenchqa_questions/countbench_paired_questions.json").read_text())
    rows = [int(r["id"].rsplit("/", 1)[1]) for r in recs]
    assert [r["instruction"] for r in recs] == [qs[k]["question"] + " Answer with a single number." for k in rows]
    assert [r["extras"]["object_rule"] for r in recs] == ["countbenchqa", "countbenchqa_free_form"]  # 第 0、3 行
    assert "count_questions: countbenchqa" in (leaf / "config.yaml").read_text()


def test_tiny_estimate_writes_table(tmp_path):
    out = tmp_path / "est"
    run(["--estimate", "--model", "internvl3.5-2b-hf", "--task", "count", "rel_depth_blink", "--batch-sizes", "1", "2",
         "--estimate-n", "2", "--out", str(out)], tmp_path)
    base = out / "internvl3.5-2b-hf" / "estimate"
    doc = json.loads((base / "estimate.json").read_text())
    assert [r["task"] for r in doc["tasks"]] == ["count", "rel_depth_blink"]
    for r in doc["tasks"]:
        assert [b["batch_size"] for b in r["by_batch_size"]] == [1, 2] and r["best_batch_size"] in (1, 2)
        assert r["gpu_hours"]["eval"] is not None and r["n"]["eval"] >= 1
    assert "合计" in (base / "estimate.md").read_text()


def test_parse_args_rules(cli, monkeypatch):
    a = cli.parse_args(["--tiny", "--model", "qwen3.5-0.8b", "--task", "all", "--out", "x"])
    assert a.task == list(cli.TASKS) and (a.device, a.dtype, a.batch_size, a.map_ci_resamples) == ("cpu", "fp32", 4, 0)
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    a = cli.parse_args(["--model", "qwen3.5-2b", "--task", "count", "--out", "experiments/E1/0011_e1_main",
                        "--data-root", "/d"])
    assert (a.device, a.dtype, a.batch_size, a.map_ci_resamples) == ("cuda", "bf16", 16, cli.DEFAULT_MAP_CI_RESAMPLES)
    bad = [
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "nope", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--subset", "extra", "--policy", "qwen:1024", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "refexp_box", "--subset", "extra", "--out", "x"],  # extra 要配附加口径
        ["--tiny", "--model", "gemma-4-e2b-it", "--task", "refexp_box", "--subset", "extra", "--policy", "qwen:1024", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "refexp_box", "--policy", "qwen:1024", "--out", "x"],  # 主口径冻结
        ["--model", "qwen3.5-0.8b", "--task", "count", "--out", "experiments/E1/main", "--data-root", "/d"],  # 目录名要带编号
        ["--model", "qwen3.5-0.8b", "--task", "count", "--out", "experiments/E1/0011_x"],  # 没有数据目录
        ["--tiny", "--acceptance", "--model", "qwen3.5-0.8b", "--task", "count", "--batch-size", "1", "--out", "x"],
        ["--tiny", "--model", "unknown-model", "--task", "count", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--agreement", "outcome", "--out", "x"],  # 只用于验收
        ["--tiny", "--acceptance", "--model", "qwen3.5-0.8b", "--task", "count", "--agreement", "nope", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--count-questions", "manual", "--out", "x"],
    ]
    for argv in bad:
        with pytest.raises(SystemExit):
            cli.parse_args(argv)
    ok = cli.parse_args(["--tiny", "--model", "internvl3.5-2b-hf", "--task", "ovd", "--subset", "extra",
                         "--policy", "internvl:native", "--out", "x"])
    assert ok.policy == "internvl:native"
    assert (ok.agreement, ok.count_questions) == ("bitwise", "heuristic")  # 默认值：D32 的逐位；启发式问句
    acc = cli.parse_args(["--tiny", "--acceptance", "--model", "qwen3.5-0.8b", "--task", "all", "--agreement", "outcome",
                          "--count-questions", "countbenchqa", "--out", "x"])
    assert (acc.agreement, acc.count_questions) == ("outcome", "countbenchqa")
    from src.adapters.image_policy import INTERNVL_NATIVE_TILES, PixelBudget

    assert cli.resolve_policy("internvl:native", "internvl3.5") == INTERNVL_NATIVE_TILES
    assert cli.resolve_policy("qwen:1024", "qwen3.5") == PixelBudget(1024)
    with pytest.raises(ValueError):
        cli.resolve_policy("qwen:1024", "gemma4")


def test_error_is_written_before_exit(tmp_path):
    p = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--model", "qwen3.5-0.8b", "--task", "count",
                        "--prompts", str(tmp_path / "missing.json"), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, cwd=REPO, timeout=300)
    assert p.returncode == 2
    errs = list((tmp_path / "o" / "qwen3.5-0.8b" / "errors").glob("error_*.json"))
    doc = json.loads(errs[0].read_text()) if len(errs) == 1 else {}
    assert "FileNotFoundError" in doc["error"] and "missing.json" not in doc["argv"][0] and "load_prompt_config" in doc["traceback"]
