"""DEV-0033：E1 事后读数的命令行彩排（--tiny：真实的小型随机模型，CPU fp32，tests/fixtures/e1_posthoc；与服务器同一流程的子进程）
与汇总脚本 scripts/summarize_e1_posthoc.py。

- 去偏读数（--subset debiased，四个任务）：输出齐全；读数次数 = Σk、order_major；s = 0 的答案与主评测的字母读数逐条相同
  （NYU 点对、EmbSpatial 两个 fixture 的数据相同，批组成相同）；续跑（读数截在一批中间、留半行）之后逐字节相同；另两个家族与
  --subset debiased_dev 各跑一遍。
- 格式强制（--subset forced，Qwen）：三个目录；网格两种读法每条恰好 271 个 token、不判截断，结构强制全部有效；预填的输入多出预填的
  token、raw_text 以 "[[" 开头；续跑逐字节相同；非 Qwen 拒绝运行（退出码 2，报错先落盘）。
- 命令行规则；汇总脚本的三个子命令（validity、forced 在合成数据上逐数核对，并在真实的小运行输出上跑通；estimate 的算法）。
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

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_e1.py"
SUMMARY = REPO / "scripts" / "summarize_e1_posthoc.py"
LETTER_TASKS = ("rel_depth_nyu", "rel_depth_blink", "spatial_cvbench", "spatial_embspatial")
FILES = ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "README.md", "meta.json")
QWEN = "qwen3.5-0.8b"


def _load(name: str, path: Path):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


@pytest.fixture(scope="module")
def cli():
    return _load("_run_e1_posthoc_under_test", SCRIPT)


@pytest.fixture(scope="module")
def summ():
    return _load("_summarize_e1_posthoc_under_test", SUMMARY)


def run(args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--nyu-cache", str(tmp / "nyu"), *args],
                       capture_output=True, text=True, cwd=REPO, env=env, timeout=1200)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """一次性跑好：Qwen 的去偏读数（四个任务）、主评测（NYU、EmbSpatial 的字母读数；多边形、网格前 5 句）、格式强制。"""
    tmp = tmp_path_factory.mktemp("posthoc")
    deb, main, forced = tmp / "deb", tmp / "main", tmp / "forced"
    run(["--model", QWEN, "--task", "all", "--subset", "debiased", "--out", str(deb)], tmp)
    run(["--model", QWEN, "--task", "rel_depth_nyu", "spatial_embspatial", "--out", str(main)], tmp)
    run(["--model", QWEN, "--task", "refexp_poly", "grid16", "--limit", "5", "--out", str(main)], tmp)
    run(["--model", QWEN, "--task", "all", "--subset", "forced", "--out", str(forced)], tmp)
    return {"tmp": tmp, "deb": deb, "main": main, "forced": forced}


# ---------------------------------------------------------------------------
# 去偏读数
# ---------------------------------------------------------------------------


def test_debiased_outputs(runs):
    from src.e1.posthoc import POSTHOC_PATH
    from src.runlog import sha256_file

    for t in LETTER_TASKS:
        leaf = runs["deb"] / QWEN / f"{t}__debiased"
        for f in FILES + ("readings.jsonl", "logs/timing.jsonl"):
            assert (leaf / f).exists(), (t, f)
        recs, reads = jsonl(leaf / "predictions.jsonl"), jsonl(leaf / "readings.jsonl")
        m = json.loads((leaf / "metrics.json").read_text())
        assert m["subset"] == "debiased" and m["n"] == len(recs) >= 1 and m["complete"] is True
        assert m["posthoc"]["config_sha256"] == sha256_file(POSTHOC_PATH) and m["posthoc"]["base_subset"] == "eval"
        assert len(reads) == sum(r["k"] for r in recs) == m["posthoc"]["n_readings"]
        orders = [r["order"] for r in reads]
        assert orders == sorted(orders)  # order_major
        for r in recs:
            assert r["readout"] == "letter_debiased" and r["failure"] is None and not r["truncated"]
            assert len(r["orders"]) == r["k"] == len(r["extras"]["letters"]) and abs(sum(r["mean_probs"]) - 1) < 1e-5
            assert r["parsed"]["letter"] == r["extras"]["letters"][max(range(r["k"]), key=lambda j: (r["mean_probs"][j], -j))]
        d = m["debiased"]
        assert [e["order"] for e in d["per_order"]] == list(range(max(r["k"] for r in recs)))
        assert d["original_letter"]["headline"][0]["n"] == m["n"] and "paired_diff" in d
        cfg = (leaf / "config.yaml").read_text()
        assert "subset: debiased" in cfg and "kind: letter_debiased" in cfg and "layout: order_major" in cfg
        comp = json.loads((leaf / "compute.json").read_text())
        assert comp["n_calls"] == len(reads) and comp["n_rounds"] == comp["n_samples"] == len(recs)
        assert comp["generated_tokens"] == len(reads)
    # BLINK 与 CV-Bench 的指令确实被换了位置（事后 fixture 的真实格式提示）
    for t in ("rel_depth_blink", "spatial_cvbench"):
        reads = jsonl(runs["deb"] / QWEN / f"{t}__debiased" / "readings.jsonl")
        by = {}
        for r in reads:
            by.setdefault(r["id"], []).append(r["instruction"])
        assert all(len(set(v)) == len(v) >= 2 for v in by.values())


def test_order0_equals_main_letter_readout(runs):
    """NYU 点对、EmbSpatial：两个 fixture 的数据相同、批组成相同（--tiny 批大小 4）；s = 0 的 argmax 就是主评测的字母读数。"""
    for t in ("rel_depth_nyu", "spatial_embspatial"):
        deb = jsonl(runs["deb"] / QWEN / f"{t}__debiased" / "predictions.jsonl")
        main = {r["id"]: r for r in jsonl(runs["main"] / QWEN / t / "predictions.jsonl")}
        assert [r["id"] for r in deb] == list(main)
        for r in deb:
            assert r["instruction"] == main[r["id"]]["instruction"]
            assert r["parsed"]["original_letter"] == main[r["id"]]["parsed"]["letter"]
            assert r["score"]["original_correct"] == main[r["id"]]["score"]["correct"]


def test_debiased_resume_is_byte_identical(runs):
    leaf = runs["deb"] / QWEN / "rel_depth_nyu__debiased"
    names = ("readings.jsonl", "predictions.jsonl", "metrics.json", "README.md", "config.yaml")
    ref = {f: (leaf / f).read_bytes() for f in names}
    lines = ref["readings.jsonl"].splitlines(keepends=True)
    assert len(lines) == 10  # 5 对 × 2 种顺序；批大小 4：s=0 [0,4) [4,5)，s=1 [5,9) [9,10)
    (leaf / "readings.jsonl").write_bytes(b"".join(lines[:7]) + lines[7][:25])  # 截在 s = 1 的第一批中间，留半行
    (leaf / "predictions.jsonl").unlink()
    run(["--model", QWEN, "--task", "rel_depth_nyu", "--subset", "debiased", "--out", str(runs["deb"]), "--resume"],
        runs["tmp"])
    for f in names:
        assert (leaf / f).read_bytes() == ref[f], f
    last = json.loads(sorted((leaf / "logs").glob("run_*.json"))[-1].read_text())
    assert last["n_done_before"] == 5 and last["trimmed_to_batch_boundary"] == {"from": 7, "to": 5}
    p = run(["--model", QWEN, "--task", "rel_depth_nyu", "--subset", "debiased", "--out", str(runs["deb"])], runs["tmp"],
            ok=False)
    assert p.returncode == 2  # 目录非空又没有 --resume：拒绝覆盖


@pytest.mark.parametrize("model, args", [
    ("gemma-4-e2b-it", ["--task", "all", "--subset", "debiased_dev"]),
    ("internvl3.5-2b-hf", ["--task", "rel_depth_blink", "spatial_cvbench", "--subset", "debiased", "--batch-size", "2"]),
])
def test_debiased_other_families(model, args, tmp_path):
    out = tmp_path / "o"
    run(["--model", model, *args, "--out", str(out)], tmp_path)
    sub = args[args.index("--subset") + 1]
    leaves = sorted((out / model).iterdir())
    assert leaves and all(p.name.endswith(f"__{sub}") for p in leaves)
    for leaf in leaves:
        m = json.loads((leaf / "metrics.json").read_text())
        assert m["posthoc"]["base_subset"] == ("dev" if sub == "debiased_dev" else "eval")
        assert m["readout"] == {"letter_debiased": m["n"]} and m["n"] >= 1


# ---------------------------------------------------------------------------
# 格式强制
# ---------------------------------------------------------------------------


def test_forced_outputs(runs):
    from src.e1.posthoc import GRID_FIXED_TOKENS

    base = runs["forced"] / QWEN
    assert sorted(p.name for p in base.iterdir()) == ["grid16__forced_restricted", "grid16__forced_structured",
                                                      "refexp_poly__forced_prefill"]
    main_poly = {r["id"]: r for r in jsonl(runs["main"] / QWEN / "refexp_poly" / "predictions.jsonl")}
    for name in ("grid16__forced_restricted", "grid16__forced_structured", "refexp_poly__forced_prefill"):
        leaf = base / name
        for f in FILES:
            assert (leaf / f).exists(), (name, f)
        m = json.loads((leaf / "metrics.json").read_text())
        recs = jsonl(leaf / "predictions.jsonl")
        assert m["subset"] == "forced" and m["posthoc"]["base_subset"] == "think" and m["n"] == len(recs) == 5
        assert m["generation"]["enable_thinking"] is False
        if name.startswith("grid16"):
            assert m["generation"]["max_new_tokens"] == GRID_FIXED_TOKENS
            for r in recs:
                assert r["n_generated_tokens"] == GRID_FIXED_TOKENS and r["stopped"] is False and r["truncated"] is False
                assert r["forced"]["fixed_length"] is True and r["failure"] != "truncated"
        else:
            assert m["generation"]["max_new_tokens"] == 1024
            for r in recs:
                assert r["raw_text"].startswith("[[") and r["forced"]["answer_prefix"] == "[["
                assert r["forced"]["answer_prefix_ids"]
                assert r["n_prompt_tokens"] == main_poly[r["id"]]["n_prompt_tokens"] + len(r["forced"]["answer_prefix_ids"])
                assert r["instruction"] == main_poly[r["id"]]["instruction"]  # 提示不变
    s = json.loads((base / "grid16__forced_structured" / "metrics.json").read_text())
    assert s["valid"]["rate"] == 1.0 and s["parse"]["failure_rate"] == 0.0  # 结构强制必然是合法网格
    assert "事后诊断" in (base / "grid16__forced_structured" / "README.md").read_text()


def test_forced_resume_is_byte_identical(runs):
    leaf = runs["forced"] / QWEN / "grid16__forced_structured"
    names = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")
    ref = {f: (leaf / f).read_bytes() for f in names}
    lines = ref["predictions.jsonl"].splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_bytes(b"".join(lines[:2]) + lines[2][:40])
    run(["--model", QWEN, "--task", "grid16", "--subset", "forced", "--out", str(runs["forced"]), "--resume"], runs["tmp"])
    for f in names:
        assert (leaf / f).read_bytes() == ref[f], f


def test_forced_refuses_non_qwen(tmp_path):
    out = tmp_path / "o"
    p = run(["--model", "gemma-4-e2b-it", "--task", "grid16", "--subset", "forced", "--out", str(out)], tmp_path, ok=False)
    assert p.returncode == 2
    errs = list((out / "gemma-4-e2b-it" / "errors").glob("error_*.json"))
    assert len(errs) == 1 and "格式强制只做" in json.loads(errs[0].read_text())["error"]


# ---------------------------------------------------------------------------
# 命令行规则
# ---------------------------------------------------------------------------


def test_parse_args_posthoc_rules(cli, monkeypatch):
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--subset", "debiased", "--out", "x"])
    assert a.task == list(LETTER_TASKS)
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "all", "--subset", "forced", "--out", "x"])
    assert a.task == ["refexp_poly", "grid16"]
    a = cli.parse_args(["--tiny", "--model", QWEN, "--task", "spatial_cvbench", "--subset", "debiased_dev", "--out", "x"])
    assert a.subset == "debiased_dev"
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    bad = [
        ["--tiny", "--model", QWEN, "--task", "count", "--subset", "debiased", "--out", "x"],
        ["--tiny", "--model", QWEN, "--task", "rel_depth_nyu", "--subset", "forced", "--out", "x"],
        ["--tiny", "--model", QWEN, "--task", "grid16", "--subset", "debiased", "--out", "x"],
        ["--tiny", "--acceptance", "--model", QWEN, "--task", "all", "--subset", "debiased", "--out", "x"],
        ["--tiny", "--estimate", "--model", QWEN, "--task", "all", "--subset", "forced", "--out", "x"],
        ["--tiny", "--model", QWEN, "--task", "all", "--subset", "debiased", "--policy", "qwen:1024", "--out", "x"],
        ["--model", QWEN, "--task", "all", "--subset", "debiased", "--out", "experiments/E1/debiased", "--data-root", "/d"],
    ]
    for argv in bad:
        with pytest.raises(SystemExit):
            cli.parse_args(argv)
    from src.e1.prompts import load_prompt_config

    real = cli.parse_args(["--model", "qwen3.5-4b", "--task", "all", "--subset", "debiased", "--data-root", "/d",
                           "--out", "experiments/E1/0031_e1_debiased"])
    assert cli.resolve_batch_sizes(real, load_prompt_config()) == (64, None)  # 冻结的批大小


# ---------------------------------------------------------------------------
# 汇总脚本
# ---------------------------------------------------------------------------

TH = {"near_random_margin": 0.05, "free_valid_rate_min": 0.9, "free_minus_debiased_min": 0.2}


def _deb(i, k, correct, orig):
    letters = [chr(65 + j) for j in range(k)]
    return {"id": f"s/{i}", "k": k, "score": {"correct": correct, "original_correct": orig},
            "parsed": {"letter": "A", "original_letter": "B" if i % 2 else "A"}, "extras": {"letters": letters}}


def _free(i, correct, ok=True):
    return {"id": f"s/{i}", "failure": None if ok else "no_choice", "text": "A" if ok else "hmm",
            "score": {"correct": correct}}


def test_validity_cell_rules(summ):
    deb = [_deb(i, 2, i < 5, i < 4) for i in range(20)]  # free 的 10 条上 5 条对 → 0.5（= 随机 0.5）
    free = [_free(i, i < 9) for i in range(10)]  # 有效率 1.0，严格 0.9
    c = summ.validity_cell("rel_depth_nyu", deb, free, TH)
    assert c["debiased_on_free"] == 0.5 and c["random_level_on_free"] == 0.5 and c["free_valid_rate"] == 1.0
    assert c["free_strict"] == pytest.approx(0.9) and c["free_minus_debiased"] == pytest.approx(0.4)
    assert c["rule1_near_random"] and c["rule2_free_valid_and_ahead"] and c["flag_not_reflecting_ability"]
    assert c["debiased_eval"]["value"] == 0.25 and c["original_letter_eval"]["value"] == 0.2
    assert c["debiased_minus_original_eval"]["value"] == pytest.approx(0.05)
    deb2 = [_deb(i, 2, i < 6, i < 4) for i in range(20)]  # 0.6 > 0.5 + 0.05：不接近随机
    assert not summ.validity_cell("rel_depth_nyu", deb2, free, TH)["flag_not_reflecting_ability"]
    free2 = [_free(i, i < 9, ok=i != 0) for i in range(10)]  # 有效率 0.9（≥ 0.9 仍算）
    assert summ.validity_cell("rel_depth_nyu", deb, free2, TH)["rule2_free_valid_and_ahead"]
    free3 = [_free(i, i < 9, ok=i > 1) for i in range(10)]  # 有效率 0.8
    assert not summ.validity_cell("rel_depth_nyu", deb, free3, TH)["rule2_free_valid_and_ahead"]
    free4 = [_free(i, i < 7) for i in range(10)]  # 0.7 − 0.5 = 0.2：按 ≥ 取，满足
    assert summ.validity_cell("rel_depth_nyu", deb, free4, TH)["rule2_free_valid_and_ahead"]
    main = [{"id": f"s/{i}", "parsed": {"letter": "A"}} for i in range(20)]
    agree = summ.validity_cell("rel_depth_nyu", deb, free, TH, main)["order0_vs_main_letter"]
    assert agree["n"] == 20 and agree["n_agree"] == 10 and agree["rate"] == 0.5
    with pytest.raises(KeyError):
        summ.validity_cell("rel_depth_nyu", deb[:5], free, TH)


def _seg(i, inter, union, ok=True):
    return {"id": f"r/{i}", "failure": None if ok else "malformed", "text": "x",
            "score": {"inter": inter, "union": union, "iou": (inter / union) if union else 0.0}}


def test_forced_cell_paired(summ):
    forced = [_seg(0, 5, 10), _seg(1, 0, 10, ok=False), _seg(2, 8, 10)]
    main = [_seg(2, 2, 10), _seg(0, 0, 10, ok=False), _seg(1, 0, 10, ok=False)]  # 顺序不同：按 id 配对
    c = summ.forced_cell("refexp_poly", forced, {"main": main, "think": None})
    f = c["forced"]
    assert f["strict"]["cIoU"]["value"] == pytest.approx(13 / 30) and f["valid_rate"]["value"] == pytest.approx(2 / 3)
    assert f["conditional"]["cIoU"]["value"] == pytest.approx(13 / 20) and f["n_valid"] == 2
    d = c["vs"]["main"]["paired_diff"]
    assert d["cIoU"]["value"] == pytest.approx(13 / 30 - 2 / 30) and d["valid_rate"]["value"] == pytest.approx(1 / 3)
    assert d["gIoU"]["value"] == pytest.approx((0.5 + 0.8) / 3 - 0.2 / 3)
    assert d["conditional_both_valid"]["n"] == 1 and d["conditional_both_valid"]["gIoU"]["value"] == pytest.approx(0.6)
    assert c["vs"]["think"] == {"missing": True}
    with pytest.raises(KeyError):
        summ.forced_cell("refexp_poly", forced, {"main": main[:1]})


def test_summary_cli_on_tiny_outputs(runs, summ, tmp_path):
    # free：合成（id 取 fixture 的 think 子集 = 去偏读数的全部 eval id 的子集），其余是真实的小运行输出
    free_root = tmp_path / "free"
    for t in LETTER_TASKS:
        deb = jsonl(runs["deb"] / QWEN / f"{t}__debiased" / "predictions.jsonl")
        leaf = free_root / QWEN / f"{t}__free"
        leaf.mkdir(parents=True)
        leaf.joinpath("predictions.jsonl").write_text("".join(
            json.dumps(_free(0, True) | {"id": r["id"]}) + "\n" for r in deb))
    out = tmp_path / "v.json"
    assert summ.main(["validity", "--debiased", str(runs["deb"]), "--free", str(free_root), "--main", str(runs["main"]),
                      "--out-json", str(out), "--out-md", str(tmp_path / "v.md")]) == 0
    doc = json.loads(out.read_text())
    cells = {c["task"]: c for c in doc["cells"]}
    assert set(cells) == set(LETTER_TASKS)
    for t in ("rel_depth_nyu", "spatial_embspatial"):
        assert cells[t]["order0_vs_main_letter"]["rate"] == 1.0
    assert "有效性检查" in (tmp_path / "v.md").read_text()
    think = tmp_path / "think"
    for t in ("refexp_poly", "grid16"):
        shutil.copytree(runs["main"] / QWEN / t, think / QWEN / f"{t}__think")
    assert summ.main(["forced", "--forced", str(runs["forced"]), "--main", str(runs["main"]), "--think", str(think),
                      "--out-json", str(tmp_path / "f.json"), "--out-md", str(tmp_path / "f.md")]) == 0
    fdoc = json.loads((tmp_path / "f.json").read_text())
    assert sorted((c["model"], c["variant"]) for c in fdoc["cells"]) == [(QWEN, "prefill"), (QWEN, "restricted"),
                                                                        (QWEN, "structured")]
    for c in fdoc["cells"]:
        assert c["n"] == 5 and set(c["vs"]) == {"main", "think"}
        # think 是 main 的拷贝：两个对照的配对差相同
        assert c["vs"]["main"]["paired_diff"]["gIoU"] == c["vs"]["think"]["paired_diff"]["gIoU"]
    assert summ.main(["estimate", "--main", str(runs["main"]), "--out-json", str(tmp_path / "e.json")]) == 0
    est = json.loads((tmp_path / "e.json").read_text())
    rows = [r for r in est["rows"] if r["kind"] == "debiased"]
    assert {r["task"] for r in rows} == {"rel_depth_nyu", "spatial_embspatial"}
    for r in rows:
        assert r["factor"] == r["sum_k"] / r["n"] and r["debiased_wall_s"] == pytest.approx(r["letter_wall_s"] * r["factor"])
