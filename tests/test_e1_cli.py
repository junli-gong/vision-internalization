"""DEV-0014：scripts/run_e1.py 的 --tiny 彩排（真实的小型随机模型，CPU，tests/fixtures/e1 的数据；与服务器同一流程的子进程）。

- 每个任务一条端到端（三个家族各跑三个任务），输出文件齐全；选择题与点对任务是受限字母读数（DEV-0017）；
- 真实小模型上的断点续传逐字节一致（删掉最后几条与半行再 --resume）；
- 批处理验收 v3（DEV-0017）：A1、A2 逐位相同，A1 与 B 之间按组 ±δ，CPU 上全部通过；
- 估算模式写出 estimate.json 与表；
- 命令行参数的规则（进程内），含冻结的批大小。
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
LETTER_TASKS = ("rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial")


@pytest.fixture(scope="module")
def cli():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_run_e1_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def test_docstring_frozen_batch_size_matches_config(cli):
    """文档字符串里写的冻结批大小 = configs/e1_prompts.json 的 batch_protocol.batch_size（D43 改为 64；DEV-0029 改文档时钉住）。"""
    from src.e1.prompts import load_prompt_config

    bs = load_prompt_config().batch_protocol()["batch_size"]
    assert f"--batch-size 默认取冻结值（{bs}，" in cli.__doc__


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
        assert m["valid"]["n"] == m["n"] and (m["three_columns"] or t == "ovd")
        recs = [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines()]
        if t in LETTER_TASKS:  # 受限字母读数：真分词器的字母表，1 个 token，映射回本题的字母（DEV-0017）
            assert m["readout"] == {"letter": m["n"]} and m["generation"]["max_new_tokens"] == 1 and m["valid"]["rate"] == 1.0
            for r in recs:
                assert r["n_generated_tokens"] == 1 and not r["truncated"] and r["parsed"]["route"] == "letter_token"
                assert r["parsed"]["letter"] in r["extras"]["letters"] and r["parsed"]["spelling"].strip() == r["parsed"]["letter"]
        else:
            assert m["readout"] == {"text": m["n"]}
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


def test_tiny_acceptance_v3_all_tasks(tmp_path):
    """验收 v3（DEV-0017）：A1、A2 用 --batch-size，B 用 --alt-batch-size；CPU fp32 上两遍逐位相同、各组的差为 0、全部通过；
    检测按开发图的完整标注判（fixture 的 coco_train2014_e1dev）；--resume 不再生成，只重新比较。"""
    out = tmp_path / "acc"
    args = ["--acceptance", "--model", "qwen3.5-0.8b", "--task", "all", "--batch-size", "3", "--alt-batch-size", "2",
            "--out", str(out)]
    run(args, tmp_path)
    base = out / "qwen3.5-0.8b" / "acceptance"
    doc = json.loads((base / "acceptance.json").read_text())
    assert doc["version"] == 3 and doc["all_pass"] is True and doc["rerun_all_identical"] is True
    assert doc["header"]["batch_size_a"] == 3 and doc["header"]["batch_size_b"] == 2 and doc["header"]["frozen"] is False
    from src.e1.prompts import TASKS

    assert [r["task"] for r in doc["tasks"]] == list(TASKS)
    for r in doc["tasks"]:
        assert r["rerun"]["identical"] and r["rerun"]["n_diff"] == 0 and r["rerun"]["file_identical"], r["task"]
        assert r["reference"]["agreement_rate"]["outcome"] == 1.0, r["task"]  # 检测也按判定结果比（有 GT）
    groups = {g["group"]: g for g in doc["groups"]}
    assert list(groups) == ["short_text", "letter", "polygon", "grid", "detection"]
    for g in groups.values():
        assert g["pass"] is True and g["diff"] == 0.0 and g["diff_ci"]["low"] == g["diff_ci"]["high"] == 0.0, g["group"]
    assert groups["letter"]["n"] == sum(r["n"] for r in doc["tasks"] if r["group"] == "letter")
    assert groups["detection"]["n"] == 2 and doc["groups_below_n_min"] == ["short_text", "letter", "polygon", "grid"]
    md = (base / "acceptance.md").read_text()
    assert "## 数值噪声底（A1 与 B 之间，按组）" in md and "| detection | ovd | F1@0.5 | 2 |" in md
    for p in ("A1", "A2", "B"):
        assert (base / p / "qwen3.5-0.8b" / "count__dev" / "predictions.jsonl").exists()
    a1 = base / "A1" / "qwen3.5-0.8b" / "refexp_box__dev"
    assert "batch_size: 3" in (a1 / "config.yaml").read_text()
    assert "batch_size: 2" in (base / "B" / "qwen3.5-0.8b" / "refexp_box__dev" / "config.yaml").read_text()
    before = (a1 / "predictions.jsonl").read_bytes()
    run(args + ["--resume"], tmp_path)  # 三遍都已完成：不再生成，只重新比较
    assert (a1 / "predictions.jsonl").read_bytes() == before
    runs = sorted((a1 / "logs").glob("run_*.json"))
    assert len(runs) == 2 and json.loads(runs[1].read_text())["n_remaining"] == 0
    assert json.loads((base / "acceptance.json").read_text())["all_pass"] is True


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
    # 真模型的批大小不在命令行里定死：读配置后取冻结值（resolve_batch_sizes）
    assert (a.device, a.dtype, a.batch_size, a.map_ci_resamples) == ("cuda", "bf16", None, cli.DEFAULT_MAP_CI_RESAMPLES)
    bad = [
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "nope", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--subset", "extra", "--policy", "qwen:1024", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "refexp_box", "--subset", "extra", "--out", "x"],  # extra 要配附加口径
        ["--tiny", "--model", "gemma-4-e2b-it", "--task", "refexp_box", "--subset", "extra", "--policy", "qwen:1024", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "refexp_box", "--policy", "qwen:1024", "--out", "x"],  # 主口径冻结
        ["--model", "qwen3.5-0.8b", "--task", "count", "--out", "experiments/E1/main", "--data-root", "/d"],  # 目录名要带编号
        ["--model", "qwen3.5-0.8b", "--task", "count", "--out", "experiments/E1/0011_x"],  # 没有数据目录
        ["--tiny", "--acceptance", "--model", "qwen3.5-0.8b", "--task", "count", "--batch-size", "1", "--out", "x"],
        ["--tiny", "--acceptance", "--model", "qwen3.5-0.8b", "--task", "count", "--alt-batch-size", "1", "--out", "x"],
        ["--tiny", "--acceptance", "--model", "qwen3.5-0.8b", "--task", "count", "--batch-size", "3", "--alt-batch-size", "3",
         "--out", "x"],  # 两种批组成必须不同
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--alt-batch-size", "3", "--out", "x"],  # 只用于验收
        ["--tiny", "--model", "unknown-model", "--task", "count", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--agreement", "outcome", "--out", "x"],  # v3 起没有这个参数
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--count-questions", "manual", "--out", "x"],
        ["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--subset", "free", "--out", "x"],  # free 只给选择题与点对
    ]
    for argv in bad:
        with pytest.raises(SystemExit):
            cli.parse_args(argv)
    ok = cli.parse_args(["--tiny", "--model", "internvl3.5-2b-hf", "--task", "ovd", "--subset", "extra",
                         "--policy", "internvl:native", "--out", "x"])
    assert ok.policy == "internvl:native"
    assert ok.count_questions == "countbenchqa"  # D36-2：默认人工问句
    acc = cli.parse_args(["--tiny", "--acceptance", "--model", "qwen3.5-0.8b", "--task", "all", "--count-questions",
                          "heuristic", "--out", "x"])
    assert (acc.batch_size, acc.alt_batch_size, acc.count_questions) == (4, 3, "heuristic")
    free = cli.parse_args(["--tiny", "--model", "qwen3.5-0.8b", "--task", "rel_depth_nyu", "spatial_cvbench", "--subset",
                           "free", "--out", "x"])
    assert free.subset == "free"


def test_frozen_batch_sizes_are_enforced_for_real_models(cli, monkeypatch):
    """D36-1、D43：真模型默认用冻结的批大小（64；验收 B 遍 48）；给别的值要 --allow-unfrozen-batch。"""
    from src.e1.prompts import load_prompt_config

    prompts = load_prompt_config()
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    base = ["--model", "qwen3.5-2b", "--task", "count", "--out", "experiments/E1/0011_e1_main", "--data-root", "/d"]
    assert cli.resolve_batch_sizes(cli.parse_args(base), prompts) == (64, None)
    assert cli.resolve_batch_sizes(cli.parse_args(["--acceptance", *base]), prompts) == (64, 48)
    with pytest.raises(ValueError, match="不是冻结的批大小 64"):
        cli.resolve_batch_sizes(cli.parse_args([*base, "--batch-size", "32"]), prompts)
    with pytest.raises(ValueError, match="composition_b"):
        cli.resolve_batch_sizes(cli.parse_args(["--acceptance", *base, "--alt-batch-size", "8"]), prompts)
    assert cli.resolve_batch_sizes(cli.parse_args([*base, "--batch-size", "32", "--allow-unfrozen-batch"]), prompts) == (32, None)
    with pytest.raises(ValueError, match="必须都 > 1 且不同"):
        cli.resolve_batch_sizes(cli.parse_args(["--acceptance", *base, "--alt-batch-size", "64", "--allow-unfrozen-batch"]),
                                prompts)
    tiny = cli.parse_args(["--tiny", "--model", "qwen3.5-0.8b", "--task", "count", "--batch-size", "7", "--out", "x"])
    assert cli.resolve_batch_sizes(tiny, prompts) == (7, None)  # --tiny 不受冻结值约束
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
