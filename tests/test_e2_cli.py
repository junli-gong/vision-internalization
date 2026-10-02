"""DEV-0023：scripts/run_e2.py 的 --tiny 彩排（真实的小型随机模型，CPU，tests/fixtures/e2 与 tests/fixtures/e1 的小数据；
与服务器同一流程的子进程）：三个家族 × 四个任务 × 两种探针跑通，分段运行后续跑与一次跑完逐字节相同；中途崩溃后从检查点续跑
也逐字节相同；--estimate；命令行的拒绝条件。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.e2.config import TASKS, curve_steps, load_config, resolve_task
from src.e2.data import E2Paths

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_e2.py"
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
DETERMINISTIC = ("config.yaml", "curves.jsonl", "metrics.json", "eval_samples.npz", "README.md")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO)
    if check and p.returncode != 0:
        raise AssertionError(f"run_e2 {args} 退出码 {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p


def _probe_weights(leaf: Path) -> dict:
    return torch.load(leaf / "ckpt" / "probes_final.pt", map_location="cpu", weights_only=False)["probes"]


@pytest.mark.parametrize("model", FAMILIES)
def test_tiny_all_tasks_and_resume_is_byte_identical(model, tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    base = ["--tiny", "--model", model, "--task", "all", "--reference", "post_proj"]
    run(*base, "--out", str(a))
    p = run(*base, "--out", str(b), "--stop-after-steps", "3")
    assert "按 --stop-after-steps 停下" in p.stdout and not (b / model / "semseg").exists()
    run(*base, "--out", str(b), "--resume")
    cfg = load_config()
    for task in TASKS:
        la, lb = a / model / task, b / model / task
        assert not (la / "ckpt" / "state.pt").exists() and (la / "ckpt" / "probes_final.pt").exists()  # 完成后删掉优化器状态
        for f in DETERMINISTIC:
            assert (la / f).read_bytes() == (lb / f).read_bytes(), f"{model} {task}/{f} 续跑后与一次跑完不同"
        wa, wb = _probe_weights(la), _probe_weights(lb)
        assert set(wa) == set(wb) and all(torch.equal(wa[n][k], wb[n][k]) for n in wa for k in wa[n])
        m = json.loads((la / "metrics.json").read_text())
        tc = resolve_task(cfg, task, tiny=True)
        assert m["complete"] and len(m["probes"]) == 2 * len(m["feature_keys"]) == 2 * len(m["layers"])
        assert m["layers"] == list(range(len(m["layers"])))  # --layers 默认 all（小模型 4 层 → 0..4）
        assert m["post_proj_check"]["equal"] is True
        for name, pr in m["probes"].items():
            h = m["headline"][0]["metric"]
            assert pr["eval"][h] is not None and pr["eval"][f"{h}_ci"]["low"] <= pr["eval"][h] <= pr["eval"][f"{h}_ci"]["high"]
        curves = [json.loads(l) for l in (la / "curves.jsonl").read_text().splitlines()]
        assert len(curves) == len(curve_steps(tc["steps"], tc["eval_every"])) * len(m["probes"])
        acc = json.loads((la / "accounting.json").read_text())
        n_val = m["n_val"]
        want = tc["feature_norm"]["n_samples"] + tc["steps"] * tc["batch_size"] + (len(curve_steps(tc["steps"], tc["eval_every"])) + 1) * n_val + m["n_eval"]
        assert acc["n_forward"] == want and acc["peak_mem_bytes"] is None and acc["wall_s"] > 0
        accb = json.loads((lb / "accounting.json").read_text())
        if task == "depth":  # 分了两次：两份运行记录，前向次数合计相同（统计量只算一次）
            assert accb["n_runs"] == 2 and accb["n_forward"] == want
        cy = yaml.safe_load((la / "config.yaml").read_text())
        assert cy["input"]["order"] == ("instruction_first" if task == "refseg" else "image_first")
        assert cy["data"]["train_exclusions_sha256"] and "dedupe_file" not in cy["data"]["train_pool"]
        assert "| L0 |" in (la / "README.md").read_text()
    # 已完成的再 --resume：直接返回，不新增运行记录
    run(*base, "--out", str(b), "--resume")
    assert len(list((b / model / "depth" / "logs").glob("run_*.json"))) == 2


def test_crash_between_checkpoints_then_resume_is_byte_identical(tmp_path):
    """第 3 步之后崩溃（检查点在第 2 步，第 3 步的曲线已经写了）：续跑截掉第 3 步的曲线、从第 2 步接着跑，结果与一次跑完相同。"""
    from src.adapters.factory import make_adapter
    from src.adapters import registry
    from src.e2.runner import RunSpec, run_task

    ad = make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
    e = registry.get_model("gemma-4-e2b-it")
    cfg = load_config()
    tc = resolve_task(cfg, "normals", tiny=True)
    assert tc["ckpt_every"] == 2 and tc["eval_every"] == 1
    spec = RunSpec(e.key, e.family, e.repo, e.revision, ad.model_key, True, {"type": "SoftTokenBudget"}, "fp32", "normals",
                   "every 2", (0, 2, 4), (), ad.num_layers(), ad.hidden_size())
    quiet = lambda _: None  # noqa: E731
    run_task(ad, cfg, spec, tc, E2Paths.fixtures(), tmp_path / "A", log=quiet)
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        run_task(ad, cfg, spec, tc, E2Paths.fixtures(), tmp_path / "B", log=quiet, _fail_after_step=3)
    lb = tmp_path / "B" / e.key / "normals"
    assert max(json.loads(l)["step"] for l in (lb / "curves.jsonl").read_text().splitlines()) == 3
    assert torch.load(lb / "ckpt" / "state.pt", weights_only=False)["step"] == 2
    with pytest.raises(FileExistsError, match="--resume"):
        run_task(ad, cfg, spec, tc, E2Paths.fixtures(), tmp_path / "B", log=quiet)
    run_task(ad, cfg, spec, tc, E2Paths.fixtures(), tmp_path / "B", log=quiet, resume=True, keep_optimizer_state=True)
    assert torch.load(lb / "ckpt" / "state.pt", weights_only=False)["step"] == tc["steps"]
    la = tmp_path / "A" / e.key / "normals"
    for f in DETERMINISTIC:
        assert (la / f).read_bytes() == (lb / f).read_bytes(), f
    run_log = json.loads((lb / "logs" / "run_001.json").read_text())
    assert run_log["status"] == "error" and run_log["steps_this_run"] == 3


def test_resume_refuses_a_different_configuration(tmp_path):
    base = ["--tiny", "--model", "internvl3.5-2b-hf", "--task", "refseg", "--out", str(tmp_path)]
    run(*base, "--stop-after-steps", "1")
    p = run(*base, "--resume", "--layers", "every", "2", check=False)
    assert p.returncode == 2 and "续跑的配置与已有的不同" in p.stderr
    p = run(*base, check=False)
    assert p.returncode == 2 and "--resume" in p.stderr


def test_tiny_estimate_writes_table(tmp_path):
    run("--tiny", "--estimate", "--model", "qwen3.5-0.8b", "--task", "all", "--out", str(tmp_path))
    doc = json.loads((tmp_path / "qwen3.5-0.8b" / "estimate" / "estimate.json").read_text())
    assert [r["task"] for r in doc["tasks"]] == list(TASKS)
    for r in doc["tasks"]:
        assert r["gpu_hours"] > 0 and r["measured"]["train_step_s"] > 0 and r["forwards"]["train"] == r["steps"] * r["batch_size"]
    assert doc["gpu_hours_total"] == pytest.approx(sum(r["gpu_hours"] for r in doc["tasks"]))
    assert "| depth |" in (tmp_path / "qwen3.5-0.8b" / "estimate" / "estimate.md").read_text()
    assert not (tmp_path / "qwen3.5-0.8b" / "depth").exists()  # 估算不写正式运行的目录


@pytest.mark.parametrize("args,msg", [
    (["--model", "qwen3.5-0.8b", "--task", "normals", "--data-root", "x", "--out", "0001_x"], "N20"),
    (["--model", "qwen3.5-0.8b", "--task", "depth", "--data-root", "x", "--out", "bad_name"], "运行编号"),
    (["--model", "qwen3.5-0.8b", "--task", "depth", "--out", "0001_x"], "--data-root"),
    (["--tiny", "--model", "qwen3.5-0.8b", "--task", "nope", "--out", "x"], "未知任务"),
    (["--tiny", "--model", "qwen3.5-0.8b", "--task", "depth", "depth", "--out", "x"], "重复"),
    (["--tiny", "--model", "unknown", "--task", "depth", "--out", "x"], "未登记"),
    (["--tiny", "--model", "qwen3.5-0.8b", "--task", "depth", "--layers", "every", "0", "--out", "x"], "k 必须"),
    (["--tiny", "--model", "qwen3.5-0.8b", "--task", "depth", "--reference", "post_proj", "post_proj", "--out", "x"], "重复"),
    (["--tiny", "--estimate", "--resume", "--model", "qwen3.5-0.8b", "--task", "depth", "--out", "x"], "--estimate"),
])
def test_cli_rejects(args, msg, monkeypatch, capsys):
    import importlib.util

    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    spec = importlib.util.spec_from_file_location("run_e2_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(SystemExit) as e:
        cli.parse_args(args)
    assert e.value.code == 2 and msg in capsys.readouterr().err


def test_cli_normals_gate_message():
    p = run("--model", "qwen3.5-0.8b", "--task", "normals", "--data-root", "x", "--out", "0001_x", check=False)
    assert p.returncode == 2 and "--enable-normals" in p.stderr
