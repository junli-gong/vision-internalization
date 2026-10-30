"""DEV-0074：scripts/run_u_hands.py 与 src/unified/hands.py（统一主表的两只"手"；预注册 U-2；D154）。

- 放行（D154）：缺编号、编号不对（D147、小写、带空格）时退出码 2、什么都不写（--tiny 也一样）；--out 不带运行编号、系统与参数不配、划分
  不对也是用法错误，什么都不写。给 D154 时放行：运行身份（W2 叶目录与 raw 的 config.yaml）记批准编号，<out>/<模型>/test_unlock.jsonl
  每个划分一行。
- --tiny 端到端（子进程，同服务器）：两个系统都写出 W2 的叶目录，scripts/mgr/w2_test_table 的 metrics / preds 直接读得出；逐句计分按 RLE
  独立重算（rle_to_mask → mask_inter_union / sample_iou，GT = gt_mask）逐个相同，也与 raw 里原运行器自己的逐句交并相同；标题 =
  src/w1/scoring.headline；compute.json 是 raw 的原样；续跑与一次跑完逐字节相同；不加 --resume 拒绝覆盖。
- 子集取样：与 scripts/run_llmseg.py 的取样逐 id、逐划分相同（同一个进程里截获它交给 run_split 的样本）。
- 协议与原读数相同：E1 结构强制多边形的 raw 运行身份（generation、prompt、image_policy、task、subset）与 RUN-0033 的 9B
  refexp_poly__poly_structured、W1 RUN-0063 的 9B baseline_poly_structured 逐项相同，正式运行的批大小 = 它们的 64；E3b 的设置
  （单轮、结构强制、guided、批 64）与 RUN-0056 的 9B refseg__structured 逐项相同，配置文件的 sha256 相同，合并核对的 n 与检查项相同。
- 失败按空 mask 计：解析失败的记录没有 answer，交 0、并 |GT|、IoU 0，进 failures；W2 计分与 raw 对不上时报错。
- run_e1.py、eval_e3b.py 与 src/e1、src/e3、src/w2 没有改动（本任务只新增文件）；它们原有输出的逐字节快照由 tests/test_w1_baselines.py
  （run_forced_poly_task 的默认路径）与 tests/test_e3_eval.py（evaluate 的快照）钉住。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.e1.posthoc import load_posthoc_config
from src.e1.prompts import load_prompt_config
from src.e1.scoring import gt_mask
from src.e3 import evaluate as EV
from src.e3.config import load_config
from src.e3.merge_check import settings_from_config
from src.executors.raster import rle_to_mask
from src.llmseg import runner as LR
from src.metrics.seg import mask_inter_union, sample_iou
from src.unified import hands as H
from src.w1.scoring import headline
from src.w2.data import tiny_paths, tiny_splits, w2_items
from src.w2.runner import W2RunSpec

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_u_hands.py"
QWEN = "qwen3.5-0.8b"
SPLIT = "test_refcoco_unc_testA"  # tiny 上 2 句
DETERMINISTIC = ("predictions.jsonl", "metrics.json", "config.yaml", "README.md", "compute.json")
RUN0033 = REPO / "experiments/E1/0033_e1_posthoc2/qwen3.5-9b/refexp_poly__poly_structured/config.yaml"
RUN0063 = REPO / "experiments/W1/0063_w1_devcheck_hpcgpu07/qwen3.5-9b/refseg/baseline_poly_structured/devcheck/config.yaml"
RUN0056 = REPO / "experiments/E3/0056_hpc_9b_eval_hpcgpu06/eval/e3b/qwen3.5-9b/seed0/refseg__structured"


def _load(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


@pytest.fixture(scope="module")
def ru():
    return _load(SCRIPT, "_run_u_hands_under_test")


@pytest.fixture(scope="module")
def w2t():
    return _load(REPO / "scripts" / "mgr" / "w2_test_table.py", "_w2_test_table_for_u_hands")


def run(args, tmp: Path, *, ok: bool = True, script: Path = SCRIPT) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(TMPDIR=str(tmp), HF_HUB_OFFLINE="1")
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-4000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def leaf(out: Path, system: str, split: str = SPLIT) -> Path:
    return out / QWEN / "refseg" / H.SETTINGS[system] / split


def tiny_subset_ids() -> dict[str, list[str]]:
    """--tiny 的子集（同 run_llmseg 的 --tiny：tiny_splits 上按同一规则现算）里每个划分的样本 id。"""
    doc = tiny_splits()
    sub = LR.build_subsets(doc, splits=LR.test_splits(), source={"path": "tiny"})
    return {sp: [s.id for s in LR.subset_samples(w2_items("refseg", sp, tiny_paths(), splits=doc), sub["splits"][sp]["ids"])]
            for sp in LR.test_splits()}


# ---------------------------------------------------------------------------
# 常量与放行
# ---------------------------------------------------------------------------


def test_u_hands_constants():
    assert H.TEST_APPROVAL == "D154" and H.TEST_UNLOCK_LOG == "test_unlock.jsonl"
    assert H.SETTINGS == {"e1_poly_structured": "u_e1_poly_structured", "e3b_grid": "u_e3b_grid"}
    posthoc = load_posthoc_config(load_prompt_config())
    fp = posthoc.forced_poly
    assert fp["variants"][H.E1_VARIANT]["min_points"] == 3 and fp["enable_thinking"] is False  # D72-3：至少 3 个点，关思考
    assert "qwen3.5-9b" in fp["models"]
    s = EV.resolve_eval(load_config(), tiny=False, k_values=list(H.E3B_K_VALUES), decode=H.E3B_DECODE)
    assert s["upsample"]["primary"] == "guided" and s["k_values"] == [1] and s["n_rounds"] == 1 and s["n_samples"] == 0  # D83-3：单轮、guided
    assert EV.decode_mode(s) == "structured" and EV.merge_criteria_version(s) == "v1"
    assert H.check_approval("D154") == {"approval": "D154", "rule": H.UNLOCK_RULE}
    for bad in (None, "D147", "d154", " D154", "D154 ", "D1540", ""):
        with pytest.raises(H.TestApprovalError):
            H.check_approval(bad)


@pytest.fixture(scope="module")
def fake_ck(tmp_path_factory):
    """parse_args 只看检查点目录：拒绝的用例不用训练。"""
    d = tmp_path_factory.mktemp("fakeck") / "model"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"build_spec": {"key": QWEN, "tiny": True}}), encoding="utf-8")
    return d


def test_u_hands_refusals_write_nothing(ru, fake_ck, tmp_path, capsys):
    out = tmp_path / "0999_u_hands"
    e1 = ["--tiny", "--system", "e1_poly_structured", "--model", QWEN, "--split", SPLIT, "--out", str(out)]
    e3 = ["--tiny", "--system", "e3b_grid", "--model", QWEN, "--split", SPLIT, "--out", str(out), "--checkpoint", str(fake_ck),
          "--seed", "0"]
    cases = []
    for base in (e1, e3):
        cases += [base, base + ["--test-approval", "D147"], base + ["--test-approval", "d154"], base + ["--test-approval", " D154"],
                  base + ["--test-approval", "D1540"]]
    real = ["--system", "e1_poly_structured", "--model", QWEN, "--split", SPLIT, "--test-approval", "D154", "--data-root", str(tmp_path)]
    cases += [
        real + ["--out", str(tmp_path / "u_hands_no_number")],  # --out 没有运行编号（真运行）
        real + ["--out", str(tmp_path / "99_u_hands")],
        e1 + ["--test-approval", "D154", "--checkpoint", str(fake_ck)],  # 零样本的 E1 不配检查点
        e1 + ["--test-approval", "D154", "--seed", "0"],
        e3[:-4] + ["--test-approval", "D154"],  # e3b 没有 --checkpoint / --seed
        e3[:-2] + ["--test-approval", "D154"],
        e3 + ["--test-approval", "D154", "--split", "devtune800"],  # 不是 test 划分
        e1 + ["--test-approval", "D154", "--model", "internvl3.5-2b-hf"],  # 结构强制的多边形只做 Qwen3.5
        e1 + ["--test-approval", "D154", "--limit", "0"],
    ]
    for argv in cases:
        with pytest.raises(SystemExit) as ei:
            ru.main(argv)
        assert ei.value.code == 2, argv
        assert not out.exists() and not (tmp_path / "u_hands_no_number").exists() and not (tmp_path / "99_u_hands").exists(), argv
    err = capsys.readouterr().err
    assert "--test-approval D154" in err and "恰是 D154" in err and "运行编号" in err
    # 放行的参数合格时 parse_args 通过，放行记录就是运行身份里的那一份
    a = ru.parse_args(e1 + ["--test-approval", "D154"])
    assert a.test_unlock == {"approval": "D154", "rule": H.UNLOCK_RULE}
    # 直接调用 cmd 时也再核对一次
    a.test_approval = "D147"
    with pytest.raises(H.TestApprovalError):
        ru.cmd(a, [])
    assert not out.exists()


# ---------------------------------------------------------------------------
# --tiny 端到端
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e1_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("u_e1")
    out = tmp / "out"
    args = ["--tiny", "--system", "e1_poly_structured", "--model", QWEN, "--split", "all", "--test-approval", "D154", "--out", str(out)]
    p = run(args, tmp)
    return out, args, p, tmp


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """训练框架（scripts/train_e3b.py --tiny）产出的小检查点（Qwen）。"""
    d = tmp_path_factory.mktemp("u_e3b_train")
    run(["--tiny", "--model", QWEN, "--supervision", "grid", "--seed", "0", "--out", str(d)], d, script=REPO / "scripts" / "train_e3b.py")
    ck = d / "e3b" / QWEN / "seed0" / "ckpt" / "final" / "model"
    assert (ck / "manifest.json").is_file()
    return ck


@pytest.fixture(scope="module")
def e3_run(tmp_path_factory, tiny_ckpt):
    tmp = tmp_path_factory.mktemp("u_e3b")
    out = tmp / "out"
    args = ["--tiny", "--system", "e3b_grid", "--model", QWEN, "--split", SPLIT, "--test-approval", "D154", "--checkpoint", str(tiny_ckpt),
            "--seed", "0", "--out", str(out)]
    p = run(args, tmp)
    return out, args, p, tmp


def _check_leaf(lf: Path, system: str, split: str, want_ids: list[str], w2t) -> list[dict]:
    for f in ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "meta.json", "README.md"):
        assert (lf / f).is_file(), f
    recs = jsonl(lf / "predictions.jsonl")
    assert [r["id"] for r in recs] == want_ids
    m = json.loads((lf / "metrics.json").read_text(encoding="utf-8"))
    assert m["complete"] is True and m["n"] == m["n_expected"] == len(want_ids) and m["setting"] == H.SETTINGS[system]
    assert m["headline"] == json.loads(json.dumps(headline("refseg", recs)))  # W2 的标题（src/w1/scoring.headline）
    assert sum(m["failures"].values()) == len(recs) and m["n_failed"] == sum(1 for r in recs if r["failure"] is not None)
    # scripts/mgr/w2_test_table 的读法（W2 主表、x_test_table 都用它）
    got = w2t.metrics(lf)
    assert got["complete"] is True and got["n"] == len(recs) and got["cIoU"]["value"] == m["headline"][0]["value"]
    p = w2t.preds(lf)
    assert list(p) == want_ids and all(p[r["id"]]["inter"] == r["score"]["inter"] and p[r["id"]]["union"] == r["score"]["union"] for r in recs)
    # 逐句计分：按 RLE 独立重算（W2 的同一组函数），失败按空 mask
    samples = {s.id: s for s in w2_items("refseg", split, tiny_paths(), splits=tiny_splits())}
    for r in recs:
        gt = gt_mask(samples[r["id"]])
        if r["failure"] is None:
            mk = rle_to_mask(r["answer"]["mask"]["rle"])
            assert int(mk.sum()) == r["answer"]["mask"]["area"]
            st = mask_inter_union(mk, gt)
        else:
            assert r["answer"] is None
            st = mask_inter_union(None, gt)
        assert r["score"] == {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)}
        assert r["w2_split"] == split and r["system"] == system and r["n_model_calls"] == 1
    cfg = yaml.safe_load((lf / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["w2"]["test_unlock"] == {"approval": "D154", "rule": H.UNLOCK_RULE} and cfg["w2"]["split"] == split
    assert cfg["w2"]["setting"] == H.SETTINGS[system] and cfg["system"] == system and cfg["w2"]["subset"]["salt"] == "w2_llmseg_v1"
    raw = lf / m["raw"]["leaf"]
    assert (lf / "compute.json").read_bytes() == (raw / "compute.json").read_bytes()
    assert m["raw"]["predictions_sha256"] == hashlib.sha256((raw / "predictions.jsonl").read_bytes()).hexdigest()
    return recs


def test_u_hands_e1_tiny_end_to_end(e1_run, w2t):
    out, _, p, _ = e1_run
    ids = tiny_subset_ids()
    assert p.stdout.count("U e1_poly_structured") == 5
    for sp in LR.test_splits():
        lf = leaf(out, "e1_poly_structured", sp)
        recs = _check_leaf(lf, "e1_poly_structured", sp, ids[sp], w2t)
        raw = jsonl(lf / "raw" / "predictions.jsonl")
        assert [r["score"] for r in raw] == [r["score"] for r in recs]  # 与原运行器自己的逐句计分相同
        rcfg = yaml.safe_load((lf / "raw" / "config.yaml").read_text(encoding="utf-8"))
        assert rcfg["posthoc"]["variant"] == "poly_structured" and rcfg["posthoc"]["base_subset"] == f"w2_llmseg_subsets:{sp}"
        assert rcfg["generation"]["enable_thinking"] is False and rcfg["generation"]["forced_poly"]["grammar"]["min_points"] == 3
        assert rcfg["data"]["selection_stats"]["w2"]["test_unlock"]["approval"] == "D154"
        assert all(r["forced_poly"]["n_points"] >= 3 for r in raw)
        m = json.loads((lf / "metrics.json").read_text(encoding="utf-8"))
        assert m["raw"]["headline"] == m["headline"]  # E1 的标题与 W2 的同一个函数（metric_of）
    log = jsonl(out / QWEN / "test_unlock.jsonl")
    assert [x["split"] for x in log] == LR.test_splits()
    assert all(x["approval"] == "D154" and x["system"] == "e1_poly_structured" and x["setting"] == "u_e1_poly_structured"
               and x["checkpoint_manifest_sha256"] is None and "--test-approval" in x["argv"] for x in log)


def test_u_hands_e1_resume_identical_and_refuses_overwrite(e1_run):
    out, args, _, tmp = e1_run
    before = {sp: {f: (leaf(out, "e1_poly_structured", sp) / f).read_bytes() for f in DETERMINISTIC} for sp in LR.test_splits()}
    p = run(args, tmp, ok=False)  # 不加 --resume：拒绝覆盖
    assert p.returncode == 2 and "已存在且非空" in p.stderr
    run(args + ["--resume"], tmp)
    for sp in LR.test_splits():
        lf = leaf(out, "e1_poly_structured", sp)
        for f in DETERMINISTIC:
            assert (lf / f).read_bytes() == before[sp][f], (sp, f)
        assert sorted(x.name for x in (lf / "logs").glob("derive_*.json")) == ["derive_001.json", "derive_002.json"]
    assert len(jsonl(out / QWEN / "test_unlock.jsonl")) == 10  # 每次跑、每个划分一行；被拒覆盖的那次在加载模型之前就停了，不记


def test_u_hands_e3b_tiny_end_to_end_and_resume(e3_run, tiny_ckpt, w2t):
    out, args, p, tmp = e3_run
    lf = leaf(out, "e3b_grid", SPLIT)
    recs = _check_leaf(lf, "e3b_grid", SPLIT, tiny_subset_ids()[SPLIT], w2t)
    sha = hashlib.sha256((tiny_ckpt / "manifest.json").read_bytes()).hexdigest()
    cfg = yaml.safe_load((lf / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["e3b_grid"]["checkpoint"]["manifest_sha256"] == sha and cfg["e3b_grid"]["seed"] == 0
    assert cfg["e3b_grid"]["upsample"] == "guided" and cfg["e3b_grid"]["eval"]["decode"]["mode"] == "structured"
    assert cfg["e3b_grid"]["eval"]["k_values"] == [1] and cfg["raw"]["dir"] == "raw/eval/e3b/qwen3.5-0.8b/seed0/refseg__structured"
    raw = lf / cfg["raw"]["dir"]
    rcfg = yaml.safe_load((raw / "config.yaml").read_text(encoding="utf-8"))
    assert rcfg["checkpoint"]["manifest_sha256"] == sha and rcfg["eval"]["decode"]["mode"] == "structured"
    assert rcfg["data"]["stats"]["w2"]["test_unlock"]["approval"] == "D154" and rcfg["data"]["n"] == len(recs)
    scores = jsonl(raw / "scores.jsonl")
    for r, sc in zip(recs, scores):  # 与 E3b 评测自己的单轮、guided 逐句交并相同
        g = sc["procs"]["single"]["guided"]
        assert (r["score"]["inter"], r["score"]["union"], r["score"]["iou"]) == (g["inter"], g["union"], g["iou"])
    m = json.loads((lf / "metrics.json").read_text(encoding="utf-8"))
    assert m["raw"]["headline"]["cIoU"]["value"] == pytest.approx(m["headline"][0]["value"], rel=1e-12)
    assert m["e3b_grid"]["checkpoint_manifest_sha256"] == sha and m["e3b_grid"]["decode"] == "structured"
    assert (out / QWEN / "e3b_merge_check.json").is_file()
    log = jsonl(out / QWEN / "test_unlock.jsonl")
    assert len(log) == 1 and log[0]["checkpoint_manifest_sha256"] == sha and log[0]["system"] == "e3b_grid" and log[0]["split"] == SPLIT
    before = {f: (lf / f).read_bytes() for f in DETERMINISTIC}
    p2 = run(args + ["--resume"], tmp)
    assert "复用已通过" in p2.stderr  # 合并核对复用
    for f in DETERMINISTIC:
        assert (lf / f).read_bytes() == before[f], f


def test_u_hands_subset_ids_same_as_run_llmseg(e1_run, monkeypatch, tmp_path):
    """scripts/run_llmseg.py 在同一个 --tiny 设置下交给 run_split 的样本，与本运行器每个划分的 predictions 逐 id 相同。"""
    rl = _load(REPO / "scripts" / "run_llmseg.py", "_run_llmseg_for_u_hands")
    got: dict[str, list[str]] = {}

    def fake_run_split(adapter, spec, samples, out, **kw):
        got[spec.split] = [s.id for s in samples]
        return {"n": len(samples), "complete": True, "readouts": {}, "calls_per_image": None}

    monkeypatch.setattr(LR, "run_split", fake_run_split)
    argv = ["--tiny", "--model", QWEN, "--split", "all", "--out", str(tmp_path / "llmseg"), "--adapter-factory",
            "tests._llmseg_fakes:make_grid_adapter"]
    assert rl.cmd(rl.parse_args(argv), argv) == 0
    out = e1_run[0]
    assert list(got) == LR.test_splits()
    for sp, ids in got.items():
        assert [r["id"] for r in jsonl(leaf(out, "e1_poly_structured", sp) / "predictions.jsonl")] == ids


def test_u_hands_real_subset_config(ru):
    """真运行的子集：入库清单按规则重算一致；5 个 W2 test 划分各 300 句（预注册 U-1）；运行器的划分清单就是它的。"""
    assert LR.check_subsets() == []
    doc = LR.load_subsets()
    assert list(doc["splits"]) == LR.test_splits() and all(e["n"] == 300 for e in doc["splits"].values())
    assert hashlib.sha256(LR.SUBSETS_PATH.read_bytes()).hexdigest().startswith("dfd53ce8")  # 预注册 U-1 记的 sha256
    a = ru.parse_args(["--system", "e1_poly_structured", "--model", "qwen3.5-9b", "--split", "all", "--test-approval", "D154",
                       "--data-root", "/nonexistent", "--out", "experiments/W2/0999_u_hands_x"])
    splits_doc, subsets_doc, _, files = ru.load_docs(a)
    assert subsets_doc == doc and files == [REPO / "configs" / "w2_splits.json", LR.SUBSETS_PATH]


# ---------------------------------------------------------------------------
# 协议与原读数相同
# ---------------------------------------------------------------------------


def test_u_hands_e1_protocol_same_as_e1_and_w1(e1_run):
    for ref_path in (RUN0033, RUN0063):
        if not ref_path.is_file():
            pytest.skip(f"{ref_path} 不在（稀疏检出）")
        ref = yaml.safe_load(ref_path.read_text(encoding="utf-8"))
        got = yaml.safe_load((leaf(e1_run[0], "e1_poly_structured") / "raw" / "config.yaml").read_text(encoding="utf-8"))
        for k in ("task", "subset", "generation", "prompt", "image_policy", "experiment", "framework"):
            assert got[k] == ref[k], (ref_path, k)
        for k in ("kind", "config", "config_sha256", "version", "variant"):
            assert got["posthoc"][k] == ref["posthoc"][k], (ref_path, k)
        assert ref["batch_protocol"]["batch_size"] == load_prompt_config().batch_protocol()["batch_size"] == 64  # 正式运行用的批大小


def test_u_hands_e3b_protocol_same_as_run0056():
    if not (RUN0056 / "config.yaml").is_file():
        pytest.skip("RUN-0056 的评测目录不在（稀疏检出）")
    ref = yaml.safe_load((RUN0056 / "config.yaml").read_text(encoding="utf-8"))
    cfg = load_config()
    s = EV.resolve_eval(cfg, tiny=False, k_values=list(H.E3B_K_VALUES), decode=H.E3B_DECODE)
    assert json.loads(json.dumps(EV.identity_settings(s))) == ref["eval"]
    assert ref["config"]["sha256"] == cfg.sha256 and ref["config"]["path"] == cfg.path
    assert ref["checkpoint"]["manifest_sha256"].startswith("316d25c7")  # 预注册 U-2 记的检查点
    run_log = json.loads((RUN0056 / "logs" / "run_001.json").read_text(encoding="utf-8"))
    mc = run_log["merge_check"]
    assert settings_from_config(cfg.doc, tiny=False, purpose="grid_eval") == {"n": mc["n"], "checks": mc["checks"]}
    assert mc["criteria_version"] == "v1" and mc["pass"] is True
    assert run_log["checkpoint_dir"].endswith("/ckpt/hpcgpu06/0056/e3b/qwen3.5-9b/seed0/ckpt/final/model")


# ---------------------------------------------------------------------------
# 失败按空 mask 计；对不上就报错
# ---------------------------------------------------------------------------


def _spec(split: str) -> W2RunSpec:
    return W2RunSpec(model_key=QWEN, family="qwen3.5", repo="r", revision="v", adapter_model_key="a", task="refseg", split=split,
                     setting="u_e1_poly_structured", tiny=True, limit=None, image_policy={}, test_unlock={"approval": "D154"})


def _tok() -> dict:
    return {"n_prompt_tokens": 10, "n_generated_tokens": 5, "n_visual_tokens": 3, "n_image_pixels": 100}


def test_u_hands_failures_scored_as_empty_and_mismatch_refused():
    samples = w2_items("refseg", SPLIT, tiny_paths(), splits=tiny_splits())[:2]
    gts = [gt_mask(s) for s in samples]
    h, w = gts[0].shape
    full = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]
    raw = [{"id": samples[0].id, "image_size": [w, h], "failure": "no_polygon", "parsed": None, "stopped": True, "truncated": False,
            "score": {"inter": 0, "union": int(gts[0].sum()), "iou": 0.0},
            "forced_poly": {"n_points": 3, "near_rect": False, "area_ratio": None}, **_tok()}]
    st = mask_inter_union(np.ones_like(gts[1]), gts[1])
    raw.append({"id": samples[1].id, "image_size": list(gts[1].shape[::-1]), "failure": None, "parsed": {"polygons": [full]},
                "stopped": True, "truncated": False, "score": {"inter": st.inter, "union": st.union, "iou": sample_iou(*st)},
                "forced_poly": {"n_points": 4, "near_rect": True, "area_ratio": 1.0}, **_tok()})
    recs = H.e1_records(_spec(SPLIT), samples, raw)
    assert recs[0]["answer"] is None and recs[0]["failure"] == "no_polygon"
    assert recs[0]["score"] == {"inter": 0, "union": int(gts[0].sum()), "iou": 0.0}
    assert recs[1]["answer"]["mask"]["area"] == gts[1].size and recs[1]["score"]["inter"] == int(gts[1].sum())
    m = H.aggregate(recs)
    assert m["failures"] == {"no_polygon": 1, "ok": 1} and m["n_failed"] == 1 and m["empty_mask"]["n"] == 0
    assert m["compute_per_sentence"]["generated_tokens"] == 5.0
    bad = [dict(raw[0]), {**raw[1], "score": {**raw[1]["score"], "inter": raw[1]["score"]["inter"] - 1}}]
    with pytest.raises(RuntimeError, match="不同"):
        H.e1_records(_spec(SPLIT), samples, bad)
    with pytest.raises(RuntimeError, match="id 不同"):
        H.e1_records(_spec(SPLIT), samples, raw[::-1])
    # E3b：解析不出网格 → 没有 mask，按空 mask 计；全 0 的网格不是失败，记进 empty_mask
    n = 4
    zero = "00" * (n * n // 8)
    e3raw = [{"id": samples[0].id, "stage": "round1", "called": True, "grid": None, "failure": "bad_length", "stopped": True, **_tok()},
             {"id": samples[1].id, "stage": "round1", "called": True, "grid": zero, "failure": None, "stopped": True, **_tok()}]
    u1 = int(gts[1].sum())
    e3sc = [{"id": samples[0].id, "procs": {"single": {"valid": False, "guided": {"inter": 0, "union": int(gts[0].sum()), "iou": 0.0,
                                                                                   "biou": 0.0}, "grid": {"cell_iou": 0.0}}}},
            {"id": samples[1].id, "procs": {"single": {"valid": True, "guided": {"inter": 0, "union": u1, "iou": 0.0 if u1 else 1.0,
                                                                                  "biou": 0.0}, "grid": {"cell_iou": 0.0}}}}]
    r3 = H.e3b_records(_spec(SPLIT), samples, e3raw, e3sc, n=n, upsample="guided", threshold=0.5)
    assert r3[0]["answer"] is None and r3[0]["failure"] == "bad_length" and r3[0]["score"]["iou"] == 0.0
    assert r3[1]["failure"] is None and r3[1]["answer"]["mask"]["area"] == 0
    m3 = H.aggregate(r3)
    assert m3["failures"] == {"bad_length": 1, "ok": 1} and m3["empty_mask"]["n"] == 1
    e3sc_bad = [e3sc[0], {"id": samples[1].id, "procs": {"single": {"valid": True, "guided": {"inter": 1, "union": u1, "iou": 1 / u1,
                                                                                               "biou": 0.0}}}}]
    with pytest.raises(RuntimeError, match="不同"):
        H.e3b_records(_spec(SPLIT), samples, e3raw, e3sc_bad, n=n, upsample="guided", threshold=0.5)
