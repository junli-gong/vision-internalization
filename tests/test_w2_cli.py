"""DEV-0059：scripts/run_w2.py 的命令行彩排（--tiny，子进程，与服务器同一流程）与 scripts/w2_gate.py。

- 小型随机 Qwen（CPU）+ tests/fixtures 的小数据；候选与渲染用 tests/_w2_fakes 的假实现（--candidates-module、--render-module，只配
  --tiny；DEV-0058 合入之前）。随机模型给不出合法的框，所以先跑 E1 refexp_box（"提出"），再把它的记录换成合法的框（第一条保留解析失败），
  后面的设置都直接读这个框的运行（不再跑框）——这也核对了"已写完就直接读"。
- 全部设置 rc = 0：refseg 的 w2a（真的 chat_turn_batch）、random_pick、oracle_pick、w1f、ownbox、baseline_e1；refexp_box 与 count 的 w2a
  （按输出类型路由到 E1 单轮）。候选只生成一次，四种设置共用；oracle ≥ 其余；random 与规则一致；w1f 交 1 号。
- 挑选的解析与回退（--adapter-factory 的脚本化适配器）：编号、none、乱写、超出范围。
- 续跑逐字节相同；冻结保护（devcheck2、test 拒跑，退出码 2，什么都不写；devtune800 不受限；--tiny 不受限）；w2_gate 在 devcheck2 的彩排上跑通，
  结论标为彩排。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_w2.py"
GATE = REPO / "scripts" / "w2_gate.py"
QWEN = "qwen3.5-0.8b"
FAKES = ["--candidates-module", "tests._w2_fakes", "--render-module", "tests._w2_fakes"]
SCRIPTED = ["--adapter-factory", "tests._w2_fakes:make_scripted_select_adapter"]
GENS = [{"type": "grabcut_default"}, {"type": "box_scale", "percent": 20}, {"type": "box_scale", "percent": -20}, {"type": "box"}]
DETERMINISTIC = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")


def run(args, tmp: Path, *, ok: bool = True, script: Path = SCRIPT) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def w2(task, split, setting, out, cfg, *extra, tiny=True):
    return (["--tiny"] if tiny else []) + ["--model", QWEN, "--task", task, "--split", split, "--setting", setting, "--out", str(out),
                                           "--w2-config", str(cfg), *extra]


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def leaf(out: Path, task: str, setting: str, split: str) -> Path:
    return out / QWEN / task / setting / split


def inject_boxes(box_leaf: Path, split: str) -> None:
    """把 E1 refexp_box 的记录换成合法的框（GT 框内缩 1 像素；第一条保留原来的解析失败），只改 failure 与 parsed。"""
    from src.w2.data import tiny_paths, tiny_splits, w2_items

    gt = {s.id: s.gt["box"] for s in w2_items("refexp_box", split, tiny_paths(), splits=tiny_splits())}
    recs = jsonl(box_leaf / "predictions.jsonl")
    assert recs[0]["failure"] is not None  # 随机模型给不出框
    for r in recs[1:]:
        x0, y0, x1, y1 = gt[r["id"]]
        r.update(failure=None, parsed={"box": [x0 + 1, y0 + 1, x1 - 1, y1]})
    (box_leaf / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w2cli")
    cfg = tmp / "w2_k4.json"
    d = json.loads((REPO / "configs" / "w2.json").read_text())
    d["candidates"]["generators"] = GENS
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    out = tmp / "out"
    for split in ("devtune800", "devcheck2"):
        run(w2("refexp_box", split, "baseline_e1", out, cfg), tmp)
        inject_boxes(leaf(out, "refexp_box", "baseline_e1", split), split)
    out_s = tmp / "out_scripted"
    shutil.copytree(out / QWEN / "refexp_box", out_s / QWEN / "refexp_box")
    logs = {}
    logs["w2a"] = run(w2("refseg", "devtune800", "w2a", out, cfg, "--chunk-size", "2", *FAKES), tmp)
    for s in ("random_pick", "oracle_pick", "w1f"):
        logs[s] = run(w2("refseg", "devtune800", s, out, cfg, *FAKES), tmp)
    logs["ownbox"] = run(w2("refseg", "devtune800", "ownbox", out, cfg), tmp)
    logs["baseline_e1"] = run(w2("refseg", "devtune800", "baseline_e1", out, cfg), tmp)
    logs["box_w2a"] = run(w2("refexp_box", "devtune800", "w2a", out, cfg), tmp)
    logs["count_w2a"] = run(w2("count", "test_countbench", "w2a", out, cfg), tmp)
    for s in ("w2a", "w1f"):
        run(w2("refseg", "devcheck2", s, out, cfg, *FAKES), tmp)
    run(w2("refseg", "devtune800", "w2a", out_s, cfg, *FAKES, *SCRIPTED), tmp)
    return {"tmp": tmp, "out": out, "out_s": out_s, "cfg": cfg, "logs": logs}


def test_all_settings_and_outputs(runs):
    out = runs["out"]
    ids = None
    for s in ("w2a", "random_pick", "oracle_pick", "w1f", "ownbox", "baseline_e1"):
        lx = leaf(out, "refseg", s, "devtune800")
        for f in ("predictions.jsonl", "metrics.json", "config.yaml", "README.md", "meta.json"):
            assert (lx / f).exists(), (s, f)
        m = json.loads((lx / "metrics.json").read_text())
        assert m.get("complete", True) is True and m["n"] == 5, s
        assert m["headline"][0]["name"] == "cIoU", s
        got = [r["id"] for r in jsonl(lx / "predictions.jsonl")]
        ids = ids or got
        assert got == ids, s
    for s in ("random_pick", "oracle_pick", "w1f", "ownbox"):
        assert "先跑 E1" not in runs["logs"][s].stderr  # 框已写完：直接读，不跑模型
    for task, split in (("refexp_box", "devtune800"), ("count", "test_countbench")):
        lx = leaf(out, task, "w2a", split)
        src = leaf(out, task, "baseline_e1", split)
        assert (lx / "predictions.jsonl").read_bytes() == (src / "predictions.jsonl").read_bytes()  # 按输出类型路由：E1 单轮原样
        cfg = yaml.safe_load((lx / "config.yaml").read_text())
        assert cfg["routing"]["output_kind"] in ("box", "count") and cfg["routing"]["source"]["leaf"].endswith(f"{task}/baseline_e1/{split}")
        assert json.loads((lx / "metrics.json").read_text())["complete"] is True


def test_candidates_are_generated_once_and_shared(runs):
    out = runs["out"]
    cand = leaf(out, "refseg", "candidates", "devtune800")
    rows = jsonl(cand / "candidates.jsonl")
    assert len(rows) == 5 and rows[0]["candidates"] is None and all(len(r["candidates"]) == 4 for r in rows[1:])
    assert all(set(c) == {"index", "name", "ok", "info", "mask"} and "elapsed_s" in c["info"] for r in rows[1:] for c in r["candidates"])
    assert len(list((cand / "logs").glob("run_*.json"))) == 1  # 只生成了一次
    shas = set()
    for s in ("w2a", "random_pick", "oracle_pick", "w1f"):
        c = yaml.safe_load((leaf(out, "refseg", s, "devtune800") / "config.yaml").read_text())
        assert c["candidates"]["leaf"] == f"<out>/{QWEN}/refseg/candidates/devtune800" and c["candidates"]["k"] == 4
        assert c["propose"]["mode"] == "box" and c["propose"]["run"]["leaf"].endswith("refexp_box/baseline_e1/devtune800")
        shas.add(c["candidates"]["content_sha256"])
    assert len(shas) == 1
    recs = {s: {r["id"]: r for r in jsonl(leaf(out, "refseg", s, "devtune800") / "predictions.jsonl")}
            for s in ("w2a", "random_pick", "oracle_pick", "w1f")}
    salt = json.loads(runs["cfg"].read_text())["random_pick"]["salt"]
    from src.w2.loop import random_index

    for r in rows[1:]:
        i = r["id"]
        assert recs["w1f"][i]["final_index"] == 1 and recs["w1f"][i]["answer"]["mask"]["rle"] == r["candidates"][0]["mask"]["rle"]
        assert recs["random_pick"][i]["final_index"] == random_index(salt, i, 4)
        best = recs["oracle_pick"][i]
        assert best["final_index"] == best["best_index"] and best["final_is_best"]
        for s in ("w2a", "random_pick", "w1f"):
            assert recs[s][i]["score"]["iou"] <= best["score"]["iou"]
            assert [c["iou"] for c in recs[s][i]["candidates"]] == [c["iou"] for c in best["candidates"]]
    first = rows[0]["id"]
    for s in recs:
        assert recs[s][first]["failure"] == rows[0]["proposal"]["failure"] and recs[s][first]["n_candidates"] == 0
    om = json.loads((leaf(out, "refseg", "oracle_pick", "devtune800") / "metrics.json").read_text())
    assert om["pick"]["final_is_best"]["rate"] == 1.0 and om["pick"]["final_is_best"]["of"] == 4
    wm = json.loads((leaf(out, "refseg", "w1f", "devtune800") / "metrics.json").read_text())
    assert wm["pick"]["final_is_1"]["rate"] == 1.0


def test_w2a_with_the_real_chat_turn_batch(runs):
    out = runs["out"]
    lx = leaf(out, "refseg", "w2a", "devtune800")
    recs = jsonl(lx / "predictions.jsonl")
    cfg = yaml.safe_load((lx / "config.yaml").read_text())
    assert cfg["select"]["method"] == "chat_turn_batch" and cfg["select"]["thinking_budget"] == 8  # --tiny 只缩小预算
    assert cfg["render"]["module"] == "tests._w2_fakes" and cfg["batching"] == {"batch_size": 2, "chunk_size": 2, "max_batch_tokens": 48000}
    assert cfg["w2_config"]["frozen"] is False and len(cfg["w2_config"]["config_sha256"]) == 64
    for r in recs[1:]:
        s = r["select"]
        n_cells = len(GENS) + (1 if json.loads(runs["cfg"].read_text())["render"]["style"].get("reference") else 0)  # 参照格随配置的 style（D117）
        assert s["method"] == "chat_turn_batch" and s["images"][0]["size"] == [48 * n_cells, 36] and s["n_visual_tokens"] > 0
        assert r["fallback"] in (None, "none", "parse_failure") and r["final_index"] == (r["pick"] or 1)
        assert r["n_model_calls"] == 2 and r["n_tool_calls"] == 1 and all("wall" not in k for k in r)
    m = json.loads((lx / "metrics.json").read_text())
    assert set(m["pick"]) >= {"final_is_best", "final_is_1", "none", "parse_failure"} and m["n_with_candidates"] == 4
    comp = json.loads((lx / "compute.json").read_text())
    assert comp["n_samples"] == 5 and comp["n_calls"] == sum(r["n_model_calls"] for r in recs) and comp["image_pixels"] > 0
    t = jsonl(lx / "logs" / "timing.jsonl")
    assert [x["chunk_index"] for x in t] == [0, 0, 1, 1, 2] and all(x["wall_time_s"] >= x["select_wall_s"] for x in t)


def test_scripted_selection_parse_and_fallback(runs):
    lx = leaf(runs["out_s"], "refseg", "w2a", "devtune800")
    got = {r["id"].rsplit("/", 1)[1]: (r["pick"], r["fallback"], r["final_index"], (r.get("select") or {}).get("parse_failure"))
           for r in jsonl(lx / "predictions.jsonl")}
    assert got == {"100": (None, None, None, None),  # 提出失败：没有候选，不挑
                   "102": (None, "none", 1, None), "103": (None, "parse_failure", 1, "no_choice"), "104": (3, None, 3, None),
                   "105": (None, "parse_failure", 1, "out_of_range")}
    m = json.loads((lx / "metrics.json").read_text())
    assert (m["pick"]["none"]["n"], m["pick"]["parse_failure"]["n"], m["pick"]["final_is_1"]["n"], m["pick"]["none"]["of"]) == (1, 2, 3, 4)
    assert m["select"]["parse_failure"] == {"no_choice": 1, "out_of_range": 1}
    assert yaml.safe_load((lx / "config.yaml").read_text())["model"]["adapter_model_key"] == "fake-select"


def test_resume_is_byte_identical(runs):
    out, cfg = runs["out"], runs["cfg"]
    lx = leaf(out, "refseg", "w2a", "devtune800")
    cand = leaf(out, "refseg", "candidates", "devtune800")
    ref = {f: (lx / f).read_bytes() for f in DETERMINISTIC}
    cand_ref = (cand / "candidates.jsonl").read_bytes()
    lines = ref["predictions.jsonl"].splitlines(keepends=True)
    (lx / "predictions.jsonl").write_bytes(b"".join(lines[:3]) + lines[3][:50])  # 截在第二组（两条）的中间
    (lx / "metrics.json").unlink()
    p = run(w2("refseg", "devtune800", "w2a", out, cfg, "--chunk-size", "2", "--resume", *FAKES), runs["tmp"])
    assert "退回到 2" in p.stderr
    for f in DETERMINISTIC:
        assert (lx / f).read_bytes() == ref[f], f
    assert (cand / "candidates.jsonl").read_bytes() == cand_ref and len(list((cand / "logs").glob("run_*.json"))) == 1
    bad = run(w2("refseg", "devtune800", "w2a", out, cfg, "--chunk-size", "2", *FAKES), runs["tmp"], ok=False)
    assert bad.returncode == 2 and "已存在且非空" in bad.stderr
    bad = run(w2("refseg", "devtune800", "w2a", out, cfg, "--chunk-size", "3", "--resume", *FAKES), runs["tmp"], ok=False)
    assert bad.returncode == 2 and "配置与已有的不同" in bad.stderr
    # 派生的设置整份重写：--resume 之后逐字节相同
    lr = leaf(out, "refseg", "random_pick", "devtune800")
    before = (lr / "predictions.jsonl").read_bytes()
    run(w2("refseg", "devtune800", "random_pick", out, cfg, "--resume", *FAKES), runs["tmp"])
    assert (lr / "predictions.jsonl").read_bytes() == before


def test_freeze_guard(runs, tmp_path):
    cfg = runs["cfg"]
    for split in ("devcheck2", "test_refcoco_unc_testA"):
        out = tmp_path / f"0001_{split}"
        p = run(w2("refseg", split, "w1f", out, cfg, "--data-root", str(tmp_path / "nodata"), tiny=False), tmp_path, ok=False)
        assert p.returncode == 2 and "协议未冻结" in p.stderr and not out.exists(), split
    p = run(w2("count", "test_countbench", "baseline_e1", tmp_path / "0001_c", cfg, "--data-root", "/x", tiny=False), tmp_path, ok=False)
    assert p.returncode == 2 and "协议未冻结" in p.stderr
    # devtune800 不受限（之后因为没有数据才失败）；冻结的配置能过保护
    p = run(w2("refseg", "devtune800", "w1f", tmp_path / "0002_tune", cfg, "--data-root", str(tmp_path / "nodata"), tiny=False),
            tmp_path, ok=False)
    assert p.returncode == 2 and "协议未冻结" not in p.stderr
    frozen = tmp_path / "w2_frozen.json"
    d = json.loads(cfg.read_text())
    d["frozen"] = True
    frozen.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    p = run(w2("refseg", "devcheck2", "w1f", tmp_path / "0003_dc2", frozen, "--data-root", str(tmp_path / "nodata"), tiny=False),
            tmp_path, ok=False)
    assert p.returncode == 2 and "协议未冻结" not in p.stderr
    # --tiny 不受限：devcheck2 在 fixture 上跑通了（fixture 里），记的是未冻结的配置
    c = yaml.safe_load((leaf(runs["out"], "refseg", "w2a", "devcheck2") / "config.yaml").read_text())
    assert c["w2_config"]["frozen"] is False and c["w2"]["split"] == "devcheck2"


def test_gate_on_tiny_outputs(runs, tmp_path):
    out = runs["out"]
    args = ["--main-name", "W2-a", "--main", str(leaf(out, "refseg", "w2a", "devcheck2")), "--w1f", str(leaf(out, "refseg", "w1f", "devcheck2")),
            "--out-dir", str(tmp_path / "g")]
    p = run(args, runs["tmp"], script=GATE)
    doc = json.loads((tmp_path / "g" / "gate.json").read_text())
    assert doc["gate_point"]["is_gate_point"] is False and doc["verdict"].startswith("彩排，不作数") and "彩排" in p.stdout
    assert doc["n"] == 3 and doc["level"] == 0.975 and doc["n_resamples"] == 10_000
    assert not doc["gate_point"]["checks"]["w2_config_frozen"] and not doc["gate_point"]["checks"]["not_tiny"]
    assert (tmp_path / "g" / "gate.md").exists()
    bad = run(["--main-name", "W2-a", "--main", args[4], "--w1f", args[4]], runs["tmp"], script=GATE, ok=False)
    assert bad.returncode == 2 and "w1f" in bad.stderr


@pytest.mark.parametrize("make, msg", [
    (lambda o, c: w2("refseg", "devtune800", "w2a", o, c, *FAKES, tiny=False) + ["--data-root", "/x"], "只配 --tiny"),
    (lambda o, c: w2("count", "devtune800", "w2a", o, c), "不用于 count"),
    (lambda o, c: w2("count", "test_countbench", "ownbox", o, c), "只用于"),
    (lambda o, c: w2("refexp_box", "devtune800", "oracle_pick", o, c), "只用于"),
    (lambda o, c: w2("refseg", "devtune800", "w2a", o, c, tiny=False) + ["--data-root", "/x"], "运行编号"),
])
def test_cli_rules(runs, tmp_path, make, msg):
    p = run(make(tmp_path / "no_run_number", runs["cfg"]), tmp_path, ok=False)
    assert p.returncode == 2 and msg in p.stderr


def test_box_points_mode_end_to_end(runs, tmp_path):
    """propose.mode = box_points（调参选项）：提出另跑一次（真的小模型给不出合法的 JSON，全部解析失败、没有候选）；候选缓存是
    candidates_box_points；w1f 仍用 E1 协议的框（mode = box 的候选缓存）。"""
    cfg = tmp_path / "w2_bp.json"
    d = json.loads(runs["cfg"].read_text())
    d["propose"]["mode"] = "box_points"
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    out = tmp_path / "out_bp"
    shutil.copytree(runs["out"] / QWEN / "refexp_box", out / QWEN / "refexp_box")
    run(w2("refseg", "devtune800", "w2a", out, cfg, *FAKES, *SCRIPTED), tmp_path)
    pl = leaf(out, "refseg", "propose_box_points", "devtune800")
    recs = jsonl(pl / "predictions.jsonl")
    assert len(recs) == 5 and all(r["failure"] is not None for r in recs)
    assert all(r["instruction"].startswith("Locate the thing") and "point_2d" in r["instruction"] for r in recs)
    pc = yaml.safe_load((pl / "config.yaml").read_text())
    assert pc["generation"]["enable_thinking"] is False and pc["propose"]["max_points"] == 3
    w = yaml.safe_load((leaf(out, "refseg", "w2a", "devtune800") / "config.yaml").read_text())
    assert w["propose"]["mode"] == "box_points" and w["candidates"]["leaf"].endswith("refseg/candidates_box_points/devtune800")
    run(w2("refseg", "devtune800", "w1f", out, cfg, *FAKES), tmp_path)
    f = yaml.safe_load((leaf(out, "refseg", "w1f", "devtune800") / "config.yaml").read_text())
    assert f["propose"]["mode"] == "box" and f["candidates"]["leaf"].endswith("refseg/candidates/devtune800")
    p = run(w2("refseg", "devtune800", "random_pick", out, cfg, *FAKES), tmp_path)  # 提出已写完：不再跑
    assert len(list((pl / "logs").glob("run_*.json"))) == 1 and "propose_box_points" not in p.stderr


def test_box_points_with_valid_points_in_process(tmp_path):
    """box_points 的整条路径（进程内、假适配器给出合法的框与点）：解析 → 候选（points_box 用到前景点）→ 挑选。"""
    from src.e1.prompts import PROMPTS_PATH, load_prompt_config
    from src.w2 import runner as R
    from src.w2.config import load_w2_config
    from src.w2.data import tiny_paths, tiny_splits, w2_items
    from tests import _w2_fakes as F

    d = json.loads((REPO / "configs" / "w2.json").read_text())
    d["propose"]["mode"] = "box_points"
    d["candidates"]["generators"] = [{"type": "grabcut_default"}, {"type": "points_box"}, {"type": "box"}]
    (tmp_path / "c.json").write_text(json.dumps(d, ensure_ascii=False))
    cfg = load_w2_config(tmp_path / "c.json")
    doc = tiny_splits(cfg)
    paths = tiny_paths()
    samples = w2_items("refseg", "devtune800", paths, splits=doc)
    prompts = load_prompt_config(PROMPTS_PATH)
    pts = ('```json\n[{"bbox_2d": [100, 100, 800, 800], "label": "x"}, {"point_2d": [300, 300], "label": "foreground"}, '
           '{"point_2d": [500, 600]}, {"point_2d": [950, 950], "label": "background"}]\n```')
    ad = F.ScriptedSelectAdapter({"Locate the thing": pts, "covers the thing 100": "3", "covers the thing 102": "2"}, default="none")
    spec = R.W2RunSpec(model_key=QWEN, family="qwen3.5", repo="Qwen/x", revision="r" * 8, adapter_model_key="fake-select", task="refseg",
                       split="devtune800", setting="w2a", tiny=True, limit=None, image_policy={"tag": "default"})
    out = tmp_path / "out"
    common = dict(splits_doc=doc, prompts=prompts)
    pdir, precs = R.ensure_points_run(lambda: (None, None, ad), spec, cfg, samples, paths, out, batch_size=2, **common)
    assert all(r["failure"] is None and len(r["parsed"]["fg_points"]) == 2 and len(r["parsed"]["bg_points"]) == 1 for r in precs)
    assert ad.generate_calls[0]["enable_thinking"] is False and ad.generate_calls[0]["max_new_tokens"] == 256
    impl = R.load_impl("tests._w2_fakes", "tests._w2_fakes")
    cl, rows = R.ensure_candidates(spec, cfg, samples, out, mode="box_points", prop_dir=pdir, prop_recs=precs, impl=impl, chunk_size=2,
                                   **common)
    for s in samples:
        r = rows[s.id]
        assert r["proposal"]["fg_points"] and [c["name"] for c in r["candidates"]] == ["grabcut_default", "points_box", "box"]
        assert r["candidates"][1]["ok"] is True and r["candidates"][1]["mask"]["area"] < r["candidates"][2]["mask"]["area"]
    m = R.run_candidate_setting(ad, spec, cfg, samples, paths, out, mode="box_points", prop_dir=pdir, prop_recs=precs, cand_leaf=cl,
                                cand_rows=rows, impl=impl, batch_size=2, chunk_size=2, max_batch_tokens=None, resume=False, **common)
    recs = {r["id"].rsplit("/", 1)[1]: r for r in jsonl(leaf(out, "refseg", "w2a", "devtune800") / "predictions.jsonl")}
    assert (recs["100"]["final_index"], recs["102"]["final_index"], recs["103"]["fallback"]) == (3, 2, "none")
    assert m["pick"]["none"]["n"] == 3 and m["complete"] is True


def _strip_elapsed(lines):
    out = []
    for r in lines:
        for c in r["candidates"] or []:
            c["info"] = {k: v for k, v in c["info"].items() if k != "elapsed_s"}
        out.append(r)
    return out


def test_candidate_workers_give_identical_content(tmp_path):
    """补充 C：候选生成用 3 个进程（spawn）与单进程的 content_sha256 相同，candidates.jsonl 除 info 里的耗时外逐条相同；
    组大小 2 时跨组按样本顺序写回。生成器含真的 GrabCut（W1-F 的同一条路径）。"""
    from src.e1.prompts import PROMPTS_PATH, load_prompt_config
    from src.w2 import runner as R
    from src.w2.config import load_w2_config
    from src.w2.data import tiny_paths, tiny_splits, w2_items
    from src.w2.loop import candidates_digest_rows
    from tests import _w2_fakes as F

    d = json.loads((REPO / "configs" / "w2.json").read_text())
    d["propose"]["mode"] = "box_points"
    d["candidates"]["generators"] = [{"type": "grabcut_default"}, {"type": "points_box"}, {"type": "box_scale", "percent": 20}, {"type": "box"}]
    (tmp_path / "c.json").write_text(json.dumps(d, ensure_ascii=False))
    cfg = load_w2_config(tmp_path / "c.json")
    doc = tiny_splits(cfg)
    paths = tiny_paths()
    samples = w2_items("refseg", "devtune800", paths, splits=doc)
    prompts = load_prompt_config(PROMPTS_PATH)
    pts = '[{"bbox_2d": [80, 90, 820, 860]}, {"point_2d": [300, 300]}, {"point_2d": [500, 600]}]'
    ad = F.ScriptedSelectAdapter({"Locate the thing": pts}, default="1")
    spec = R.W2RunSpec(model_key=QWEN, family="qwen3.5", repo="Qwen/x", revision="r" * 8, adapter_model_key="fake-select", task="refseg",
                       split="devtune800", setting="w2a", tiny=True, limit=None, image_policy={"tag": "default"})
    impl = R.load_impl("tests._w2_fakes")
    got = {}
    for workers in (1, 3):
        out = tmp_path / f"out{workers}"
        common = dict(splits_doc=doc, prompts=prompts)
        pdir, precs = R.ensure_points_run(lambda: (None, None, ad), spec, cfg, samples, paths, out, batch_size=2, **common)
        cl, rows = R.ensure_candidates(spec, cfg, samples, out, mode="box_points", prop_dir=pdir, prop_recs=precs, impl=impl, chunk_size=2,
                                       workers=workers, **common)
        got[workers] = (candidates_digest_rows([rows[s.id] for s in samples]), _strip_elapsed(jsonl(cl / "candidates.jsonl")), cl)
    assert got[1][0] == got[3][0] and got[1][1] == got[3][1]
    assert [r["id"] for r in got[3][1]] == [s.id for s in samples]
    assert all(r["candidates"][0]["ok"] and r["candidates"][0]["info"]["error"] is None for r in got[3][1])  # 子进程里真的跑了 GrabCut
    assert all(c["mask"]["area"] > 0 for r in got[3][1] for c in r["candidates"][1:])
    run3 = json.loads((got[3][2] / "logs" / "run_001.json").read_text())
    assert run3["candidate_workers"] == 3
    assert yaml.safe_load((got[1][2] / "config.yaml").read_text())["candidates"] == \
        yaml.safe_load((got[3][2] / "config.yaml").read_text())["candidates"]  # 不进运行身份


def test_candidate_workers_cli(runs, tmp_path):
    """命令行的 --candidate-workers 3：与彩排里单进程生成的候选内容相同（运行身份里的 content_sha256 相同）。"""
    out = tmp_path / "out_w3"
    shutil.copytree(runs["out"] / QWEN / "refexp_box", out / QWEN / "refexp_box")
    run(w2("refseg", "devtune800", "w1f", out, runs["cfg"], *FAKES, "--candidate-workers", "3"), tmp_path)
    a = yaml.safe_load((leaf(out, "refseg", "w1f", "devtune800") / "config.yaml").read_text())["candidates"]["content_sha256"]
    b = yaml.safe_load((leaf(runs["out"], "refseg", "w1f", "devtune800") / "config.yaml").read_text())["candidates"]["content_sha256"]
    assert a == b
    bad = run(w2("refseg", "devtune800", "w1f", out, runs["cfg"], *FAKES, "--candidate-workers", "0"), tmp_path, ok=False)
    assert bad.returncode == 2 and "candidate-workers" in bad.stderr
