"""DEV-0036 的命令行彩排（--tiny：真实的小型随机模型，CPU fp32，tests/fixtures/e1_posthoc；与服务器同一流程的子进程）与汇总脚本。

- 对调标记（--subset letter_swap）：输出齐全；读数 = 2 × n、order_major；original 那张图就是主评测的渲染图（像素 sha256 = 点对配置的
  render_sha256），它的 argmax 与主评测的字母读数逐条相同（同 D68 的 s = 0 核对）；换标记的图另有 sha256；续跑逐字节相同；
  --subset letter_swap_dev 与另一个家族各跑一遍。
- 结构强制的多边形（--subset forced_poly）：两个目录；每条以停止 token 结束、点数在 [下限, 64]，全部有效；最长生成的上界 769 记进
  config.yaml；续跑逐字节相同；非 Qwen 拒绝运行。
- --mask-grid-tokens（D72-4）：带网格 token 的小检查点（网格 token 的 lm_head 行放大，不屏蔽时自由生成里有网格 token）加了选项后输出里
  没有网格 token；受限字母读数与不带网格 token 的检查点，加不加选项 predictions.jsonl 逐字节相同；config.yaml 记下选项。
- 命令行规则（--task all --subset free 展开为四个选择题类任务等）；汇总脚本 ownbox 与 estimate --posthoc 在小运行输出上跑通。
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_e1.py"
SUMMARY = REPO / "scripts" / "summarize_e1_posthoc.py"
FILES = ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "README.md", "meta.json")
QWEN = "qwen3.5-0.8b"
LETTER_TASKS = ("rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial")


def _load(name: str, path: Path):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load("_run_e1_dev0036_under_test", SCRIPT)


@pytest.fixture(scope="module")
def summ():
    return _load("_summarize_e1_posthoc_dev0036_cli", SUMMARY)


def run(args, tmp: Path, *, ok: bool = True, tiny: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    head = ["--tiny", "--nyu-cache", str(tmp / "nyu")] if tiny else []
    p = subprocess.run([sys.executable, str(SCRIPT), *head, *args], capture_output=True, text=True, cwd=REPO, env=env,
                       timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dev0036")
    ls, main, fp = tmp / "ls", tmp / "main", tmp / "fp"
    run(["--model", QWEN, "--task", "all", "--subset", "letter_swap", "--out", str(ls)], tmp)
    run(["--model", QWEN, "--task", "rel_depth_nyu", "--out", str(main)], tmp)
    run(["--model", QWEN, "--task", "all", "--subset", "forced_poly", "--out", str(fp)], tmp)
    return {"tmp": tmp, "ls": ls, "main": main, "fp": fp}


# ---------------------------------------------------------------------------
# 对调标记
# ---------------------------------------------------------------------------


def test_letter_swap_outputs(runs):
    from src.data.nyu_pairs import load_pairs
    from src.e1.posthoc import POSTHOC_PATH
    from src.runlog import sha256_file

    leaf = runs["ls"] / QWEN / "rel_depth_nyu__letter_swap"
    assert sorted(p.name for p in (runs["ls"] / QWEN).iterdir()) == ["rel_depth_nyu__letter_swap"]
    for f in FILES + ("readings.jsonl", "logs/timing.jsonl"):
        assert (leaf / f).exists(), f
    recs, reads = jsonl(leaf / "predictions.jsonl"), jsonl(leaf / "readings.jsonl")
    m = json.loads((leaf / "metrics.json").read_text())
    assert m["subset"] == "letter_swap" and m["n"] == len(recs) == 5 and m["complete"] is True
    assert m["posthoc"]["config_sha256"] == sha256_file(POSTHOC_PATH) and m["posthoc"]["kind"] == "letter_swap"
    assert len(reads) == 2 * len(recs) == m["posthoc"]["n_readings"]
    assert [r["order"] for r in reads] == [0] * 5 + [1] * 5  # order_major
    assert [r["view"] for r in reads] == ["original"] * 5 + ["labels_swapped"] * 5
    pairs = {f"nyuv2_pairs/{p['split']}/{p['id']}": p for p in load_pairs(REPO / "tests/fixtures/e1/nyu_pairs.json")["pairs"]}
    by = {}
    for r in reads:
        by.setdefault(r["id"], {})[r["view"]] = r
        assert r["instruction"] == recs[0]["instruction"]  # 提示不变
    for sid, v in by.items():
        assert v["original"]["image_sha256"] == pairs[sid]["render_sha256"]  # 第一张就是主评测的渲染图
        assert v["labels_swapped"]["image_sha256"] != v["original"]["image_sha256"]
        assert v["labels_swapped"]["perm"] == [1, 0] and v["original"]["perm"] == [0, 1]
    d = m["letter_swap"]
    assert [e["view"] for e in d["per_view"]] == ["original", "labels_swapped"] and d["random_level"] == 0.5
    for r in recs:
        assert r["readout"] == "letter_swap" and r["parsed"]["route"] == "letter_swap" and r["k"] == 2
        assert [o["view"] for o in r["orders"]] == ["original", "labels_swapped"]
    cfg = (leaf / "config.yaml").read_text()
    assert "subset: letter_swap" in cfg and "kind: letter_swap" in cfg and "layout: order_major" in cfg
    comp = json.loads((leaf / "compute.json").read_text())
    assert comp["n_calls"] == len(reads) and comp["n_rounds"] == comp["n_samples"] == len(recs)
    assert "对调标记" in (leaf / "README.md").read_text()


def test_letter_swap_view0_equals_main_letter_readout(runs):
    ls = jsonl(runs["ls"] / QWEN / "rel_depth_nyu__letter_swap" / "predictions.jsonl")
    main = {r["id"]: r for r in jsonl(runs["main"] / QWEN / "rel_depth_nyu" / "predictions.jsonl")}
    assert [r["id"] for r in ls] == list(main)
    for r in ls:
        assert r["instruction"] == main[r["id"]]["instruction"]
        assert r["parsed"]["original_letter"] == main[r["id"]]["parsed"]["letter"]
        assert r["score"]["original_correct"] == main[r["id"]]["score"]["correct"]


def test_letter_swap_resume_is_byte_identical(runs):
    leaf = runs["ls"] / QWEN / "rel_depth_nyu__letter_swap"
    names = ("readings.jsonl", "predictions.jsonl", "metrics.json", "README.md", "config.yaml")
    ref = {f: (leaf / f).read_bytes() for f in names}
    lines = ref["readings.jsonl"].splitlines(keepends=True)
    (leaf / "readings.jsonl").write_bytes(b"".join(lines[:7]) + lines[7][:30])  # 截在换标记那一段的第一批中间
    (leaf / "predictions.jsonl").unlink()
    run(["--model", QWEN, "--task", "rel_depth_nyu", "--subset", "letter_swap", "--out", str(runs["ls"]), "--resume"],
        runs["tmp"])
    for f in names:
        assert (leaf / f).read_bytes() == ref[f], f
    last = json.loads(sorted((leaf / "logs").glob("run_*.json"))[-1].read_text())
    assert last["trimmed_to_batch_boundary"] == {"from": 7, "to": 5}


def test_letter_swap_dev_other_family(tmp_path):
    out = tmp_path / "o"
    run(["--model", "internvl3.5-2b-hf", "--task", "all", "--subset", "letter_swap_dev", "--batch-size", "3", "--out", str(out)],
        tmp_path)
    leaf = out / "internvl3.5-2b-hf" / "rel_depth_nyu__letter_swap_dev"
    m = json.loads((leaf / "metrics.json").read_text())
    assert m["posthoc"]["base_subset"] == "dev" and m["readout"] == {"letter_swap": m["n"]} and m["n"] == 5
    assert len(jsonl(leaf / "readings.jsonl")) == 10


# ---------------------------------------------------------------------------
# 结构强制的多边形
# ---------------------------------------------------------------------------


def test_forced_poly_outputs(runs):
    base = runs["fp"] / QWEN
    assert sorted(p.name for p in base.iterdir()) == ["refexp_poly__poly_structured", "refexp_poly__poly_structured_min8"]
    for name, mn in (("refexp_poly__poly_structured", 3), ("refexp_poly__poly_structured_min8", 8)):
        leaf = base / name
        for f in FILES:
            assert (leaf / f).exists(), (name, f)
        m = json.loads((leaf / "metrics.json").read_text())
        recs = jsonl(leaf / "predictions.jsonl")
        assert m["subset"] == "forced_poly" and m["posthoc"]["base_subset"] == "think" and m["n"] == len(recs) == 5
        assert m["generation"]["enable_thinking"] is False and m["generation"]["max_new_tokens"] == 1024
        g = m["generation"]["forced_poly"]["grammar"]
        assert g == {"min_points": mn, "max_points": 64, "coord_max": 1000, "stop_ids": g["stop_ids"], "max_tokens": 769}
        assert m["valid"]["rate"] == 1.0 and m["parse"]["failure_rate"] == 0.0 and m["truncation"]["rate"] == 0.0
        for r in recs:
            assert r["stopped"] is True and r["failure"] is None and r["text"].startswith("[[") and r["text"].endswith("]]")
            n = r["forced_poly"]["n_points"]
            assert mn <= n <= 64 and len(r["parsed"]["polygons"]) == 1 and len(r["parsed"]["polygons"][0]) == n
        nr = m["forced_poly"]["near_rectangle"]
        assert nr["n"] == 5 and nr["min_area_ratio"] == 0.95 and nr["n_points"] == 4
        assert "结构强制的多边形" in (leaf / "README.md").read_text()


def test_forced_poly_resume_is_byte_identical(runs):
    leaf = runs["fp"] / QWEN / "refexp_poly__poly_structured"
    names = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")
    ref = {f: (leaf / f).read_bytes() for f in names}
    lines = ref["predictions.jsonl"].splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_bytes(b"".join(lines[:4]) + lines[4][:40])  # 批大小 4：第二批写了一半
    shutil.rmtree(runs["fp"] / QWEN / "refexp_poly__poly_structured_min8")
    run(["--model", QWEN, "--task", "refexp_poly", "--subset", "forced_poly", "--out", str(runs["fp"]), "--resume"],
        runs["tmp"])
    for f in names:
        assert (leaf / f).read_bytes() == ref[f], f


def test_forced_poly_refuses_non_qwen(tmp_path):
    out = tmp_path / "o"
    p = run(["--model", "gemma-4-e2b-it", "--task", "all", "--subset", "forced_poly", "--out", str(out)], tmp_path, ok=False)
    assert p.returncode == 2
    errs = list((out / "gemma-4-e2b-it" / "errors").glob("error_*.json"))
    assert len(errs) == 1 and "只做" in json.loads(errs[0].read_text())["error"]


# ---------------------------------------------------------------------------
# 命令行规则
# ---------------------------------------------------------------------------


def test_parse_args_dev0036(cli, tmp_path):
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--subset", "free", "--out", "x"])
    assert a.task == list(LETTER_TASKS)  # D70-5：free 自动展开为四个选择题与点对任务
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--out", "x"])
    assert len(a.task) == 9  # 主评测照旧
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--subset", "letter_swap", "--out", "x"])
    assert a.task == ["rel_depth_nyu"]
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--subset", "forced_poly", "--out", "x"])
    assert a.task == ["refexp_poly"]
    ck = tmp_path / "ck"
    ck.mkdir()
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--subset", "eval", "--checkpoint", str(ck),
                        "--checkpoint-label", "c1p", "--mask-grid-tokens", "--out", "x"])
    assert a.mask_grid_tokens is True
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--subset", "letter_swap", "--checkpoint", str(ck),
                        "--checkpoint-label", "c1p", "--out", "x"])
    assert a.task == ["rel_depth_nyu"]  # 对调标记可以配检查点（同 debiased）
    bad = [
        ["--tiny", "--model", QWEN, "--task", "count", "--subset", "letter_swap", "--out", "x"],
        ["--tiny", "--model", QWEN, "--task", "grid16", "--subset", "forced_poly", "--out", "x"],
        ["--tiny", "--model", QWEN, "--task", "all", "--subset", "forced_poly", "--checkpoint", str(ck),
         "--checkpoint-label", "c1p", "--out", "x"],
        ["--tiny", "--model", QWEN, "--task", "all", "--mask-grid-tokens", "--out", "x"],  # 没有 --checkpoint
        ["--tiny", "--model", QWEN, "--task", "all", "--subset", "think", "--checkpoint", str(ck), "--checkpoint-label", "c",
         "--mask-grid-tokens", "--out", "x"],
        ["--tiny", "--model", QWEN, "--task", "all", "--subset", "debiased", "--checkpoint", str(ck), "--checkpoint-label", "c",
         "--mask-grid-tokens", "--out", "x"],
    ]
    for argv in bad:
        with pytest.raises(SystemExit):
            cli.parse_args(argv)


# ---------------------------------------------------------------------------
# --mask-grid-tokens（D72-4）
# ---------------------------------------------------------------------------


def _make_checkpoint(root: Path, *, grid: bool) -> Path:
    """C1′ 那样的检查点（没有 LoRA）：grid = True 时加 E3 的网格 token，并把它们的 lm_head（与输入嵌入）行设成放大的随机值——
    不屏蔽时自由生成里几乎必然出现网格 token；grid = False 时是不带网格 token 的检查点（随机 LoRA）。"""
    from src.adapters.factory import make_adapter
    from src.adapters.image_policy import parse_policy
    from src.e3.config import load_config
    from src.e3.tokens import add_vocab

    cfg = load_config()
    pol = parse_policy(cfg.doc["tiny"]["image_policies"]["qwen3.5"])
    ad = make_adapter(QWEN, device="cpu", dtype=torch.float32, image_policy=pol, tiny=True, init_std=0.05)
    if not grid:
        ad.inject_lora(r=4, lora_alpha=8, init_lora_weights=False)
        with torch.no_grad():
            for n, p in ad.model.named_parameters():
                if "lora_B" in n:
                    p.mul_(1e-3)
        return ad.save_trained(root)
    tok = add_vocab(ad, cfg.vocab)
    ids = torch.tensor(sorted(tok.grid_ids))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for w in {id(x): x for x in (ad.model.get_input_embeddings().weight, ad.lm_head_weight())}.values():
            w[ids] = torch.randn(len(ids), w.shape[1], generator=g) * 3.0
    return ad.save_trained(root)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    d = tmp_path_factory.mktemp("mask_ck")
    return {"grid": _make_checkpoint(d / "grid", grid=True), "plain": _make_checkpoint(d / "plain", grid=False), "tmp": d}


def _pred_bytes(out: Path, label: str, leaf: str) -> bytes:
    return (out / label / QWEN / leaf / "predictions.jsonl").read_bytes()


def test_mask_grid_tokens_on_grid_checkpoint(ckpts, tmp_path):
    common = ["--model", QWEN, "--checkpoint", str(ckpts["grid"]), "--checkpoint-label", "c1p", "--limit", "2"]
    a, b = tmp_path / "a", tmp_path / "b"
    run([*common, "--task", "rel_depth_nyu", "--subset", "free", "--out", str(a)], tmp_path)
    run([*common, "--task", "rel_depth_nyu", "--subset", "free", "--mask-grid-tokens", "--out", str(b)], tmp_path)
    ra = jsonl(a / "c1p" / QWEN / "rel_depth_nyu__free" / "predictions.jsonl")
    rb = jsonl(b / "c1p" / QWEN / "rel_depth_nyu__free" / "predictions.jsonl")
    assert sum(r["n_grid_tokens"] for r in ra) > 0  # 不屏蔽：网格 token 冒出来（测试有效）
    assert all(r["n_grid_tokens"] == 0 for r in rb)  # 屏蔽之后一个都没有
    cfg = yaml.safe_load((b / "c1p" / QWEN / "rel_depth_nyu__free" / "config.yaml").read_text())
    assert cfg["mask_grid_tokens"]["enabled"] is True and cfg["mask_grid_tokens"]["n_ids"] == 67
    assert cfg["mask_grid_tokens"]["applied"] is True
    assert "mask_grid_tokens" not in yaml.safe_load((a / "c1p" / QWEN / "rel_depth_nyu__free" / "config.yaml").read_text())
    m = json.loads((b / "c1p" / QWEN / "rel_depth_nyu__free" / "metrics.json").read_text())
    assert m["diagnostics"]["grid_tokens_in_output"]["n_rows"] == 0
    # 受限字母读数（eval）：只在字母上取，不挂处理器——加不加选项逐字节相同，config 记 applied = false
    run([*common, "--task", "rel_depth_nyu", "--out", str(a)], tmp_path)
    run([*common, "--task", "rel_depth_nyu", "--mask-grid-tokens", "--out", str(b)], tmp_path)
    assert _pred_bytes(a, "c1p", "rel_depth_nyu") == _pred_bytes(b, "c1p", "rel_depth_nyu")
    cfg = yaml.safe_load((b / "c1p" / QWEN / "rel_depth_nyu" / "config.yaml").read_text())
    assert cfg["mask_grid_tokens"]["applied"] is False


def test_mask_grid_tokens_noop_without_grid_tokens(ckpts, tmp_path):
    common = ["--model", QWEN, "--checkpoint", str(ckpts["plain"]), "--checkpoint-label", "c3", "--limit", "2",
              "--task", "rel_depth_nyu", "--subset", "free"]
    a, b = tmp_path / "a", tmp_path / "b"
    run([*common, "--out", str(a)], tmp_path)
    run([*common, "--mask-grid-tokens", "--out", str(b)], tmp_path)
    assert _pred_bytes(a, "c3", "rel_depth_nyu__free") == _pred_bytes(b, "c3", "rel_depth_nyu__free")
    ca = yaml.safe_load((a / "c3" / QWEN / "rel_depth_nyu__free" / "config.yaml").read_text())
    cb = yaml.safe_load((b / "c3" / QWEN / "rel_depth_nyu__free" / "config.yaml").read_text())
    import hashlib

    from src.e1.grid_mask import RULE

    assert cb.pop("mask_grid_tokens") == {"enabled": True, "n_ids": 0, "ids_sha256": hashlib.sha256(b"").hexdigest(),
                                          "rule": RULE, "basis": "D72-4（DEV-0036）", "applied": False}
    assert ca == cb  # 除了记下选项，运行身份不变


# ---------------------------------------------------------------------------
# 汇总脚本：ownbox 与 estimate --posthoc
# ---------------------------------------------------------------------------


def _write(leaf: Path, recs: list[dict]) -> None:
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


def test_ownbox_cli_on_tiny_outputs(runs, summ, tmp_path):
    """主评测的框是合成的（id、指令按 fixture 样本渲染；框取 GT 框），多边形读数取真实的小运行输出（结构强制的两个变体，另把
    poly_structured 拷一份当主评测），网格的主评测合成。核对：对上 5 条、自己的框近矩形占比 = 1、配对差的方向。"""
    from src.e1.prompts import load_prompt_config
    from src.e1.scoring import gt_mask
    from src.e1.tasks import DataPaths, select

    prompts = load_prompt_config()
    paths = DataPaths.posthoc_fixtures(nyu_cache=tmp_path / "nyu")
    main, think = tmp_path / "main", tmp_path / "think"
    boxes = {}
    for task in ("refexp_poly", "grid16"):
        for s in select(task, "think", paths).samples:
            h, w = gt_mask(s).shape
            x0, y0, x1, y1 = s.gt["box"]
            boxes[s.id] = {"id": s.id, "instruction": prompts.render("refexp_box", "qwen3.5", s).text, "image_size": [w, h],
                           "failure": None, "text": "[...]", "parsed": {"box": [x0, y0, x1, y1]}}
    _write(main / QWEN / "refexp_box", list(boxes.values()))
    _write(think / QWEN / "refexp_box__think", list(boxes.values())[:2])
    shutil.copytree(runs["fp"] / QWEN / "refexp_poly__poly_structured", main / QWEN / "refexp_poly")
    grid = [{"id": s.id, "failure": "wrong_row_count", "text": "1", "parsed": None,
             "score": {"inter": 0, "union": int(gt_mask(s).sum()), "iou": 0.0}} for s in select("grid16", "think", paths).samples]
    _write(main / QWEN / "grid16", grid)
    out = tmp_path / "o.json"
    assert summ.main(["ownbox", "--tiny", "--main", str(main), "--think", str(think), "--forced-poly", str(runs["fp"]),
                      "--out-json", str(out), "--out-md", str(tmp_path / "o.md")]) == 0
    doc = json.loads(out.read_text())
    (m,) = doc["models"]
    poly, gridc = m["tasks"]["refexp_poly"], m["tasks"]["grid16"]
    assert poly["n_matched"] == poly["n_ids"] == 5 and gridc["n_matched"] == 5
    rows = {r["readout"]: r for r in poly["rows"]}
    assert set(rows) == {"own_box", "main", "poly_structured", "poly_structured_min8"}
    assert rows["own_box"]["near_rectangle"]["share_of_valid"] == 1.0  # 框当多边形：恰好 4 点的矩形
    assert rows["own_box"]["strict"]["cIoU"]["value"] > 0.3  # GT 框光栅化后与 GT mask 的重合
    assert rows["main"]["paired_diff"]["cIoU"] == rows["poly_structured"]["paired_diff"]["cIoU"]  # main 是它的拷贝
    assert "think" in poly["missing"] and "prefill" in poly["missing"]  # 没给的读数照报"缺"
    grows = {r["readout"]: r for r in gridc["rows"]}
    assert grows["main"]["paired_diff"]["gIoU"]["value"] < 0  # 全部失败的网格读数比自己的框差
    assert poly["think_box"]["n_intersection"] == 2 and poly["think_box"]["row"]["readout"] == "own_box_think"
    md = (tmp_path / "o.md").read_text()
    assert "模型自己的框" in md and "own_box_think" in md


def test_estimate_posthoc_rows(summ, tmp_path):
    root = tmp_path / "0030"
    leaf = root / QWEN / "rel_depth_nyu__debiased"
    _write(leaf, [{"id": f"n/{i}", "k": 2} for i in range(10)])
    (leaf / "logs").mkdir()
    (leaf / "logs" / "run_001.json").write_text(json.dumps({"wall_s": 30.0}))
    pre = root / QWEN / "refexp_poly__forced_prefill"
    _write(pre, [{"id": f"p/{i}", "n_generated_tokens": 20 if i else 100} for i in range(40)])
    (pre / "logs").mkdir()
    (pre / "logs" / "run_001.json").write_text(json.dumps({"wall_s": 5.0}))
    (pre / "logs" / "timing.jsonl").write_text("".join(
        json.dumps({"id": f"p/{i}", "run": 1, "batch_index": 0, "batch_wall_time_s": 2.0}) + "\n" for i in range(40)))
    rows = summ.dev0036_estimate_rows(root, 200, 64)
    ls = next(r for r in rows if r["kind"] == "letter_swap")
    assert ls["n_readings"] == 20 and ls["estimate_s"] == 30.0
    fp = next(r for r in rows if r["kind"] == "forced_poly")
    assert fp["step_s_median"] == pytest.approx(2.0 / 100) and fp["n_batches"] == 4
    assert fp["upper_s"] == pytest.approx(2 * 4 * 769 * 0.02)
