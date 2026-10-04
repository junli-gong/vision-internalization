"""DEV-0023：scripts/run_e2.py 的 --tiny 彩排（真实的小型随机模型，CPU，tests/fixtures/e2 与 tests/fixtures/e1 的小数据；
与服务器同一流程的子进程）：三个家族 × 四个任务 × 两种探针跑通，分段运行后续跑与一次跑完逐字节相同；中途崩溃后从检查点续跑
也逐字节相同；--estimate；命令行的拒绝条件。DEV-0031：上采样走可分离矩阵乘（配置与运行记录里写明、fp32 审计进运行记录）；
法向不再需要 --enable-normals（D59-6、D63-4）；估算覆盖法向与 pre_proj，表头写明层、参考线、是否含法向，每行写明上采样实现。
DEV-0032（D64-4）：运行记录（logs/run_k.json 的 audit.precision）与估算的表头写明运行时的卷积与矩阵乘精度设置，只记录、不改，
不进运行身份（换了设置，config.yaml 与其余确定的输出逐字节不变）。"""

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
        N = len(m["layers"]) - 1
        # 特征键：L0..L{N} 是逐层曲线，L{N}_pre（最后一层归一化之前，配置默认加上；DEV-0024 I1）单列
        assert m["complete"] and len(m["probes"]) == 2 * len(m["feature_keys"]) == 2 * (len(m["layers"]) + 1)
        assert m["layers"] == list(range(len(m["layers"])))  # --layers 默认 all（小模型 4 层 → 0..4）
        assert m["curve_keys"] == [f"L{l}" for l in m["layers"]] and m["single_keys"] == [f"L{N}_pre"] == [m["pre_norm_last"]]
        assert m["probes"][f"L{N}_pre/linear"]["on_curve"] is False and m["probes"]["L0/linear"]["on_curve"] is True
        assert m["post_proj_check"]["equal"] is True
        assert m["positions"]["n_samples"] == m["n_eval"] and m["positions"]["unit"] == ("sample" if task == "refseg" else "pixel")
        for name, pr in m["probes"].items():
            h = m["headline"][0]["metric"]
            assert pr["eval"][h] is not None and pr["eval"][f"{h}_ci"]["low"] <= pr["eval"][h] <= pr["eval"][f"{h}_ci"]["high"]
            # 按光栅位置分开的标题指标（DEV-0024 I2）：两种划分、各区域都有，标题指标的键都在
            bp = pr["eval_by_position"]
            assert {s: list(v) for s, v in bp.items()} == {"half": ["top", "bottom"], "raster_quartile": ["q1", "q2", "q3", "q4"]}
            assert all(set(x["metric"] for x in m["headline"]) <= set(r) for v in bp.values() for r in v.values())
            if task != "refseg":  # 像素划分：每个区域都有像素
                assert all(r[h] is not None for v in bp.values() for r in v.values())
            else:  # 按质心：每条样本恰好落在一个区域里
                assert sum(r["n"] for r in bp["half"].values()) == m["n_eval"] - m["positions"]["n_unassigned"]
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
        assert cy["training"]["upsample"] == {"mode": "bilinear", "align_corners": False, "impl": "separable_matmul"}
        up_audit = json.loads((la / "logs" / "run_001.json").read_text())["audit"]["upsample"]
        assert up_audit["cpu"]["ok"] is True and up_audit["cpu"]["guarded_bwd_rel"] <= 1e-5  # DEV-0031 的 fp32 审计
        assert cy["data"]["train_exclusions_sha256"] and "dedupe_file" not in cy["data"]["train_pool"]
        dd = cy["data"]["dedupe"]  # 剔除清单的阈值与比较过的评测集（DEV-0024 B）
        assert dd["max_hamming"] == 8 and dd["legacy"] is False and set(dd["required_eval_sets"]) <= set(dd["eval_sets"])
        assert cy["layers"]["pre_norm_last"] == f"L{N}_pre"
        readme = (la / "README.md").read_text()
        assert "| L0 |" in readme and f"| L{N}_pre |" in readme and "## 按光栅位置分开" in readme and "| 探针 | 全图 | 上半 | 下半 |" in readme
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
    """--task all 含法向；--reference pre_proj 时 pre_proj 那组探针与视觉塔的额外前向都在实测里（DEV-0031）。"""
    run("--tiny", "--estimate", "--model", "qwen3.5-0.8b", "--task", "all", "--reference", "post_proj", "pre_proj",
        "--out", str(tmp_path))
    doc = json.loads((tmp_path / "qwen3.5-0.8b" / "estimate" / "estimate.json").read_text())
    assert [r["task"] for r in doc["tasks"]] == list(TASKS) and "normals" in TASKS
    h = doc["header"]
    assert h["includes_normals"] is True and h["tasks"] == list(TASKS) and h["references"] == ["post_proj", "pre_proj"]
    assert h["layer_spec"] == "all" and h["n_layers"] == 5 and h["pre_norm_last"] == "L4_pre"
    assert h["upsample_impls"] == ["separable_matmul"] and h["upsample_audit"]["cpu"]["ok"] is True
    for r in doc["tasks"]:
        assert r["gpu_hours"] > 0 and r["measured"]["train_step_s"] > 0 and r["forwards"]["train"] == r["steps"] * r["batch_size"]
        assert r["upsample_impl"] == "separable_matmul"
        # pre_proj：特征键与探针里都有（2 种探针），每次前向另有一次视觉塔前向（前向开销），外推进 forwards.vision_extra
        assert r["feature_keys"] == ["pre_proj", "L0", "L1", "L2", "L3", "L4", "L4_pre"]
        assert r["n_probes_pre_proj"] == 2 and r["n_probes"] == 2 * len(r["feature_keys"])
        fw = r["forwards"]
        assert r["measured"]["vision_extra_per_forward"] == 1.0
        assert fw["vision_extra"] == fw["norm"] + fw["train"] + fw["val"] + fw["eval"]
    assert doc["gpu_hours_total"] == pytest.approx(sum(r["gpu_hours"] for r in doc["tasks"]))
    md = (tmp_path / "qwen3.5-0.8b" / "estimate" / "estimate.md").read_text()
    assert "| depth | separable_matmul | 14（2） |" in md and "| normals | separable_matmul |" in md
    assert "- 层：all（5 个，另加 L4_pre）" in md and "- 参考线：post_proj、pre_proj" in md and "- 含法向：是" in md
    assert "| 任务 | 上采样实现 |" in md
    assert not (tmp_path / "qwen3.5-0.8b" / "depth").exists()  # 估算不写正式运行的目录
    # D64-4：表头写明运行时的精度设置（子进程里是 torch 的默认值）
    from src.e2.runner import PRECISION_KEYS

    pr = h["precision"]
    assert set(PRECISION_KEYS) <= set(pr) and pr["torch.backends.cudnn.deterministic"] is False
    assert pr["torch.backends.cudnn.benchmark"] is False and pr["torch.backends.cudnn.allow_tf32"] is True
    assert "- 精度设置（只记录、不改，D64-4）：cudnn.allow_tf32 = True，cudnn.deterministic = False" in md


def test_precision_recorded_but_not_in_identity(tmp_path):
    """D64-4：run_k.json 的 audit.precision 记下运行时的 cudnn.allow_tf32、deterministic、benchmark、cudnn.conv.fp32_precision 与
    作用域外的 cuda.matmul.fp32_precision；运行不改这些设置；换了设置，config.yaml 与其余确定的输出逐字节不变（不进运行身份）；
    新旧接口混用、读 allow_tf32 报错时记 null 与原因，不中断运行。"""
    from src.adapters import registry
    from src.adapters.factory import make_adapter
    from src.e2.runner import PRECISION_KEYS, RunSpec, backend_precision, run_task

    e = registry.get_model("qwen3.5-0.8b")
    cfg = load_config()
    tc = resolve_task(cfg, "depth", tiny=True)
    quiet = lambda _: None  # noqa: E731
    cd = torch.backends.cudnn
    saved = (cd.benchmark, cd.deterministic, cd.conv.fp32_precision)

    def go(out: Path) -> tuple[Path, dict]:
        ad = make_adapter(e.key, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
        spec = RunSpec(e.key, e.family, e.repo, e.revision, ad.model_key, True, {"type": "PixelBudget"}, "fp32", "depth",
                       "all", tuple(range(ad.num_layers() + 1)), (), ad.num_layers(), ad.hidden_size(), pre_norm_last=True)
        before = backend_precision()
        run_task(ad, cfg, spec, tc, E2Paths.fixtures(), out, log=quiet)
        assert backend_precision() == before  # 只记录，不改
        leaf = out / e.key / "depth"
        return leaf, json.loads((leaf / "logs" / "run_001.json").read_text())["audit"]["precision"]

    try:
        la, pa = go(tmp_path / "A")
        cd.benchmark, cd.deterministic = True, True
        lb, pb = go(tmp_path / "B")
        cd.conv.fp32_precision = "ieee"  # 新接口改 conv、rnn 不动：读旧接口的 allow_tf32 会报"新旧接口混用"
        lc, pc = go(tmp_path / "C")
    finally:
        cd.benchmark, cd.deterministic, cd.conv.fp32_precision = saved
    assert backend_precision()["torch.backends.cudnn.conv.fp32_precision"] == saved[2]
    for p in (pa, pb, pc):
        assert set(PRECISION_KEYS) <= set(p) and set(p["context"]) and p["torch"] == torch.__version__
    assert (pa["torch.backends.cudnn.benchmark"], pa["torch.backends.cudnn.deterministic"]) == (False, False)
    assert (pb["torch.backends.cudnn.benchmark"], pb["torch.backends.cudnn.deterministic"]) == (True, True)
    assert pa["torch.backends.cudnn.allow_tf32"] is True and pa["unavailable"] == {}
    assert pa["torch.backends.cudnn.conv.fp32_precision"] == saved[2] and pa["torch.backends.cuda.matmul.fp32_precision"] is not None
    assert pc["torch.backends.cudnn.conv.fp32_precision"] == "ieee" and pc["torch.backends.cudnn.allow_tf32"] is None
    assert "RuntimeError" in pc["unavailable"]["torch.backends.cudnn.allow_tf32"]
    for leaf in (lb, lc):
        for f in DETERMINISTIC:
            assert (la / f).read_bytes() == (leaf / f).read_bytes(), f
    # 运行身份的键与 DEV-0031 时相同（没有新增精度一类的键）
    cy = yaml.safe_load((la / "config.yaml").read_text())
    assert list(cy) == ["experiment", "framework", "model", "image_policy", "task", "input", "layers", "references", "feature_keys",
                        "training", "seed", "config", "data", "eval_limit"]
    assert "precision" not in (la / "config.yaml").read_text() and "allow_tf32" not in (la / "config.yaml").read_text()


@pytest.mark.parametrize("args,msg", [
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


def _cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_e2_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_runs_normals_without_the_old_license_flag(monkeypatch, capsys):
    """D59-6 / D63-4：法向的许可门撤掉。真模型的 normals 与 --task all 不加参数就能跑；--enable-normals 仍接受（兼容旧命令），
    不改变任何东西，帮助文本注明 D59-6；配置的 license_gate 写明许可已确认。"""
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    cli = _cli()
    base = ["--model", "qwen3.5-0.8b", "--data-root", "x", "--out", "0001_x"]
    assert cli.parse_args(["--task", "normals", *base]).task == ["normals"]
    a = cli.parse_args(["--task", "all", *base])
    assert a.task == list(TASKS) and not a.tiny and a.device == "cuda"
    b = cli.parse_args(["--task", "all", "--enable-normals", *base])
    assert {k: v for k, v in vars(b).items() if k != "enable_normals"} == {k: v for k, v in vars(a).items() if k != "enable_normals"}
    assert cli.parse_args(["--estimate", "--task", "all", *base]).task == list(TASKS)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "--enable-normals" in help_text and "D59-6" in help_text and "无作用" in help_text and "N20" not in help_text
    assert load_config().task("normals")["license_gate"].startswith("D59-6：许可已确认（Bae 2021 / GeoNet）")


def test_tiny_normals_without_flag(tmp_path):
    """--tiny 的法向不加 --enable-normals 跑通（真模型的同一条参数校验见上一条）。"""
    p = run("--tiny", "--model", "qwen3.5-0.8b", "--task", "normals", "--layers", "every", "4", "--out", str(tmp_path))
    assert "qwen3.5-0.8b normals" in p.stdout and (tmp_path / "qwen3.5-0.8b" / "normals" / "metrics.json").exists()
