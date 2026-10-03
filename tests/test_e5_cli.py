"""DEV-0018：scripts/run_e5.py 的 --tiny 彩排（真实的小型随机模型、CPU；与服务器同一流程的子进程）。

- 三个家族各跑每类任务 2 条，零样本与思维链两个系统，输出文件齐全、指标结构完整；
- 真实小模型上的断点续传逐字节一致（删掉最后一条与半行再 --resume）；
- 平凡基线、外部预测、导出输入三种不加载模型的命令；
- d 只在规范平台上生成（非规范平台上不加 --tiny 报错）；命令行参数的规则。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.e5 import runner as RN
from src.e5.protocol import E5_TASKS

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_e5.py"
FILES = ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "README.md", "meta.json")
MODELS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")


def run(args, tmp: Path, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp)}
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=900)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


ADAPTER_CLASSES = {"qwen3.5-0.8b": ("src.adapters.qwen35", "Qwen35Adapter"),
                   "gemma-4-e2b-it": ("src.adapters.gemma4", "Gemma4Adapter"),
                   "internvl3.5-2b-hf": ("src.adapters.internvl35", "InternVL35Adapter")}


def has_budgeted(model: str) -> bool:
    """这个家族的适配器有没有 generate_batch_budgeted（D59-5 的预算强制收尾；DEV-0027 实现）。"""
    mod, cls = ADAPTER_CLASSES[model]
    return callable(getattr(getattr(importlib.import_module(mod), cls), "generate_batch_budgeted", None))


@pytest.mark.parametrize("model", MODELS)
def test_tiny_each_family_each_task_two_samples(model, tmp_path):
    """真实小模型：零样本一律跑；思维链走 generate_batch_budgeted，DEV-0027 合入之前适配器没有它，这时核对思维链明确报错，
    合入之后同一条测试自动跑思维链（DEV-0028；假适配器上的完整核对见 tests/test_e5_budgeted.py）。"""
    out = tmp_path / "run"
    systems = ("zeroshot", "cot") if has_budgeted(model) else ("zeroshot",)
    run(["--tiny", "--system", *systems, "--model", model, "--task", "all", "--subset", "dev", "--limit", "2",
         "--out", str(out)], tmp_path)
    if not has_budgeted(model):
        p = run(["--tiny", "--system", "cot", "--model", model, "--task", "c", "--subset", "dev", "--limit", "1",
                 "--out", str(tmp_path / "cot")], tmp_path, ok=False)
        assert p.returncode == 2 and "generate_batch_budgeted" in p.stderr
    for system in systems:
        for t in E5_TASKS:
            leaf = out / system / model / f"{t}__dev"
            for f in FILES:
                assert (leaf / f).exists(), (system, t, f)
            recs = [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines()]
            assert [r["id"] for r in recs] == [f"{t}/val/0", f"{t}/val/1"]
            # --tiny 的上限 48：零样本是单段；思维链是预算强制收尾，思考段与作答段各截到 48，合计 ≤ 96
            # （补的思考结束标记不计入 n_generated_tokens，D60）。管理员修正（D61）：原断言只按单段写。
            cap = 48 if system == "zeroshot" else 2 * 48
            assert all(r["n_generated_tokens"] <= cap and r["prompt"] for r in recs)
            if t == "imagery":
                assert all(r["n_image_pixels"] == 0 and r["n_visual_tokens"] == 0 for r in recs)  # e 没有图像
            else:
                assert all(r["n_image_pixels"] > 0 for r in recs)
            m = json.loads((leaf / "metrics.json").read_text())
            assert m["n"] == 2 and m["system"] == system and m["model"] == model
            assert {"headline", "overall", "by_bin", "by_group", "parse", "gate", "baselines_headline"} <= set(m)
            assert ("relief_error" in m["overall"]) == (t in ("rotation3d", "viewpoint"))  # a、b 两个系统都给起伏
            cfg = (leaf / "config.yaml").read_text()
            assert "max_new_tokens_override: 48" in cfg
            assert f"path: {'generate_batch_budgeted' if system == 'cot' else 'generate_batch'}\n" in cfg
            assert f"enable_thinking: {'true' if system == 'cot' else 'false'}" in cfg
            assert ("budget" in m) == (system == "cot")


def test_tiny_resume_is_byte_identical(tmp_path):
    out = tmp_path / "run"
    args = ["--tiny", "--system", "zeroshot", "--model", "qwen3.5-0.8b", "--task", "a", "--subset", "dev", "--limit", "3",
            "--batch-size", "2", "--out", str(out)]
    run(args, tmp_path)
    leaf = out / "zeroshot" / "qwen3.5-0.8b" / "rotation3d__dev"
    pred = (leaf / "predictions.jsonl").read_bytes()
    metrics = (leaf / "metrics.json").read_bytes()
    lines = pred.splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_bytes(b"".join(lines[:2]) + lines[2][: len(lines[2]) // 2])  # 最后一批（1 条）删掉一半
    p = run(args, tmp_path, ok=False)
    assert p.returncode == 2 and "--resume" in p.stderr  # 目录非空：不加 --resume 拒绝
    run(args + ["--resume"], tmp_path)
    assert (leaf / "predictions.jsonl").read_bytes() == pred
    assert (leaf / "metrics.json").read_bytes() == metrics
    assert len(list((leaf / "logs").glob("run_*.json"))) == 2  # 首跑与续跑各一份；被拒的那次在占位之前就停了
    p = run(args[:-2] + ["--batch-size", "3", "--out", str(out), "--resume"], tmp_path, ok=False)
    assert p.returncode == 2  # 批大小是协议的一部分：续跑不能改


def test_baseline_external_and_export_commands(tmp_path):
    out = tmp_path / "run"
    run(["--tiny", "--system", "baseline", "--task", "all", "--subset", "dev", "--limit", "2", "--out", str(out)], tmp_path)
    names = {"rotation3d": {"copy", "copy_flat", "copy_convex", "instr_affine", "gt_affine_upper"},
             "viewpoint": {"copy", "translate", "copy_flat", "copy_convex", "gt_affine_upper"},
             "occlusion": {"visible", "convex_hull"}, "physics": {"copy", "const_velocity"}, "imagery": {"no_transform"}}
    for t, ns in names.items():
        got = {p.name for p in (out / "baseline").iterdir() if (p / f"{t}__dev").exists()}
        assert got == ns, t
        for n in ns:
            m = json.loads((out / "baseline" / n / f"{t}__dev" / "metrics.json").read_text())
            assert m["parse"]["n_failed"] == 0 and m["n"] == 2
    comp = json.loads((out / "baseline" / "copy" / "rotation3d__dev" / "compute.json").read_text())
    assert comp["n_samples"] == 2 and comp["generated_tokens"] == 0
    # 外部预测：只有标记文件、没有任何 mask → 全部记 external_missing
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "EXTERNAL_PREDICTIONS.json").write_text(json.dumps({"format_version": 1, "system": "lisa-7b-v1"}))
    run(["--tiny", "--system", "external", "--model", "lisa-7b-v1", "--external-dir", str(ext), "--task", "c", "--subset", "dev",
         "--limit", "2", "--out", str(out)], tmp_path)
    m = json.loads((out / "external" / "lisa-7b-v1" / "occlusion__dev" / "metrics.json").read_text())
    assert m["parse"]["failures"] == {"external_missing": 2}
    assert m["headline"]["mask_iou_current_grid"]["strict"] == 0.0  # 对当前 mask 的 IoU 也报（D59-5）
    readme = (out / "external" / "lisa-7b-v1" / "occlusion__dev" / "README.md").read_text()
    assert "对当前 mask 的 IoU" in readme
    run(["--export-inputs", str(tmp_path / "inputs"), "--task", "b", "e", "--subset", "dev", "--limit", "2"], tmp_path)
    assert len((tmp_path / "inputs" / "viewpoint" / "inputs.jsonl").read_text().splitlines()) == 2
    assert len(list((tmp_path / "inputs" / "viewpoint" / "images").glob("*.png"))) == 2


def test_physics_only_on_canonical_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(RN, "on_canonical_platform", lambda: False)
    spec = RN.RunSpec(system="baseline", model_key="copy", task="physics", subset="dev", n_per_bin=1, n_resamples=0)
    with pytest.raises(RuntimeError, match="规范平台"):
        RN.run_baseline_task(spec, tmp_path)
    RN.run_baseline_task(RN.RunSpec(**{**spec.__dict__, "tiny": True}), tmp_path)  # 彩排允许
    monkeypatch.setattr(RN, "on_canonical_platform", lambda: True)
    RN.run_baseline_task(spec, tmp_path / "canon")


def test_export_inputs_checks_the_platform_like_the_runner(cli, tmp_path, monkeypatch):
    """D40-7：--export-inputs 的 d 只在规范平台上导出（与运行同一个检查），非规范平台上只允许 --tiny；
    先核对全部任务、再导出，所以 d 不合格时一个文件也不写。"""
    monkeypatch.setattr(RN, "on_canonical_platform", lambda: False)
    out = tmp_path / "inputs"
    assert cli.main(["--export-inputs", str(out), "--task", "c", "d", "--subset", "dev", "--limit", "1"]) == 2
    assert not (out / "occlusion").exists() and not (out / "physics").exists()
    assert list((out / "errors").glob("error_*.json"))
    assert cli.main(["--export-inputs", str(out), "--task", "d", "--subset", "dev", "--limit", "1", "--tiny"]) == 0
    assert (out / "physics" / "inputs.jsonl").exists()
    monkeypatch.setattr(RN, "on_canonical_platform", lambda: True)
    assert cli.main(["--export-inputs", str(tmp_path / "canon"), "--task", "d", "--subset", "dev", "--limit", "1"]) == 0


@pytest.fixture(scope="module")
def cli():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_run_e5_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def test_argument_rules(cli):
    a = cli.parse_args(["--system", "zeroshot", "--model", "qwen3.5-0.8b", "--task", "all", "--out", "experiments/E5/0019_e5_x"])
    assert a.task == list(E5_TASKS) and a.n_per_bin == 100 and a.device == "cuda" and a.max_new_tokens is None
    a = cli.parse_args(["--system", "baseline", "--task", "a", "c", "--subset", "dev", "--out", "experiments/E5/0019_x"])
    assert a.task == ["rotation3d", "occlusion"] and a.n_per_bin == 2
    for bad in (["--system", "zeroshot", "--task", "a", "--out", "experiments/E5/0019_x"],  # 缺 --model
                ["--system", "zeroshot", "baseline", "--model", "qwen3.5-0.8b", "--task", "a", "--out", "experiments/E5/0019_x"],
                ["--system", "baseline", "--task", "a", "--out", "experiments/E5/no_run_id"],
                ["--system", "zeroshot", "--model", "qwen3.5-0.8b", "--task", "a", "--max-new-tokens", "8",
                 "--out", "experiments/E5/0019_x"],
                ["--system", "external", "--model", "lisa", "--task", "a", "--out", "experiments/E5/0019_x"],
                ["--task", "a", "--out", "x"]):
        with pytest.raises(SystemExit):
            cli.parse_args(bad)
