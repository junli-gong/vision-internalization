"""DEV-0067：scripts/gate_explore.py（G-EX，D126-4 的判定，规则写死）。

- 合成数据（构造的 devcheck3 清单，三层 200/200/100 句）上 (i)、(ii) 的四种组合，以及 (ii) 无法判定（没给健康检查、健康检查不完整）；
- 缺句、多句、重复、划分的 sha256 不对、参照不是 W2-c、清单待生成、没给门名或门名不对 → 拒绝判定（退出码 2）；
- 读数与 CI 即 scripts/w2_gate.decide（同一份数据上逐位相同；与 src/stats 用 0.975、10,000 次现算的相同）；记录的先后顺序不影响结果；
- 健康检查的标红即 src/health/scoring.compare（scripts/health_check.py --compare）：恰好低 5 个点不标、低 6 个点标；--compare 的输出
  作输入时结论相同；
- 判定点核对：不同机器、--tiny 的清单 → "彩排，不作数"。
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src import constants as C
from src.data import explore_splits as EX
from src.data.subsets import dump_config, population_digest
from src.health.scoring import compare as health_compare
from src.stats import paired_bootstrap_ci, ratio_of_sums
from tests.test_health_scoring import _report

REPO = Path(__file__).resolve().parents[1]
MODEL = "qwen3.5-9b"


def _load(name, path):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load("_gate_explore_under_test", REPO / "scripts" / "gate_explore.py")


@pytest.fixture(scope="module")
def w2_gate():
    return _load("_w2_gate_for_compare", REPO / "scripts" / "w2_gate.py")


# ---------------------------------------------------------------------------
# 构造的清单与运行目录
# ---------------------------------------------------------------------------


def toy_manifest(tmp: Path, *, tiny: bool = False) -> tuple[Path, dict]:
    """三层各 400 / 400 / 300 句的构造总体（refcoco 与 refcoco+ 共用图），按 explore_splits 的规则取 devcheck3 与 explore_val。"""
    pops = {"refcoco_unc_val": [(str(10 * i + j), f"coco:{i}") for i in range(200) for j in range(2)],
            "refcoco+_unc_val": [(str(10 * i + j + 5), f"coco:{i}") for i in range(200) for j in range(2)],
            "refcocog_umd_val": [(str(10 * i + j), f"coco:{i}") for i in range(300, 600) for j in range(1)]}
    e1 = {k: {"ids": [], "sha256": population_digest([])} for k in pops}
    splits, groups = EX.build_groups(pops, [], e1, tiny=tiny)
    eval_doc = {"populations": {k: {} for k in pops}, "subsets": {}}
    doc = EX.assemble(tiny=tiny, sources={}, eval_doc=eval_doc,
                      w2_images={"splits": {}, "n_images": 0, "images_sha256": population_digest([]), "images": []},
                      populations={}, splits=splits, groups=groups, checks={})
    path = tmp / "explore_splits.json"
    path.write_text(dump_config(doc), encoding="utf-8")
    return path, doc


def scores(ids, seed=0):
    rng = np.random.default_rng(seed)
    union = rng.integers(500, 1500, len(ids))
    frac = np.clip(rng.normal(0.6, 0.15, len(ids)), 0, 1)
    return {i: (int(f * u), int(u)) for i, f, u in zip(ids, frac, union)}


def recs_of(sc: dict) -> list[dict]:
    return [{"id": i, "score": {"inter": a, "union": b, "iou": (a / b if b else 1.0)}} for i, (a, b) in sc.items()]


def scaled(sc: dict, factor: float, only=None) -> dict:
    keys = set(sc) if only is None else set(only)
    return {i: ((min(b, int(round(a * factor))) if i in keys else a), b) for i, (a, b) in sc.items()}


def make_leaf(root: Path, doc: dict, split: str, recs: list[dict], *, setting="w2c", checkpoint=None, host="hpcgpu06", gate="G-EX",
              tiny=False, limit=None, frozen=True, model=MODEL, split_sha=None, w2=True) -> Path:
    leaf = root / model / "refseg" / setting / split
    leaf.mkdir(parents=True)
    if w2:
        w2sec = {"task": "refseg", "split": split, "setting": setting, "split_sha256": split_sha or doc["splits"][split]["sha256"],
                 "n_samples": len(recs), "limit": limit}
        if gate is not None:
            w2sec["unseal"] = {"gate": gate, "decision": EX.SEAL_DECISION}
        cfg = {"experiment": "W2", "w2": w2sec, "model": {"key": model, "revision": "r" * 40, "tiny": tiny},
               "w2_config": {"config": "configs/w2.json", "config_sha256": "a" * 64, "frozen": frozen}, "propose": {"mode": "box"}}
        if checkpoint is not None:
            cfg["checkpoint"] = checkpoint
        (leaf / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        (leaf / "metrics.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
    if host is not None:
        (leaf / "meta.json").write_text(json.dumps({"hostname": host}), encoding="utf-8")
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return leaf


CK = {"manifest_sha256": "c" * 64, "dir": "x"}


def side(root: Path, doc: dict, sc: dict, **kw) -> list[Path]:
    split_of = EX.split_of_sample(doc, EX.DEVCHECK3)
    out = []
    for nm in EX.group_split_names(EX.DEVCHECK3):
        out.append(make_leaf(root, doc, nm, recs_of({i: v for i, v in sc.items() if split_of[i] == nm}), **kw))
    return out


def health_files(tmp: Path, cand_counts: dict, *, host_b="hpcgpu06", host_c="hpcgpu06", limit=None) -> tuple[Path, Path]:
    base = {**_report({"mmlu_pro": 60, "gsm8k": 30, "ifeval": 25}, key=MODEL), "ok": True, "run": {"meta": {"hostname": host_b}}}
    cand = {**_report(cand_counts, key=MODEL), "ok": True, "run": {"meta": {"hostname": host_c}}, "limit": limit}
    pb, pc = tmp / "health_base.json", tmp / "health_cand.json"
    pb.write_text(json.dumps(base), encoding="utf-8")
    pc.write_text(json.dumps(cand), encoding="utf-8")
    return pb, pc


NOFLAG = {"mmlu_pro": 55, "gsm8k": 28, "ifeval": 25}  # 恰好低 5 个点：不标
FLAG = {"mmlu_pro": 54, "gsm8k": 30, "ifeval": 25}  # 低 6 个点：标红


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gx")
    path, doc = toy_manifest(tmp)
    ids = EX.group_ids(doc, EX.DEVCHECK3)
    ref = scores(ids, seed=1)
    ref_dirs = side(tmp / "ref", doc, ref, checkpoint=CK)
    variants = {"better": scaled(ref, 1.15), "worse": scaled(ref, 0.85), "close": scaled(ref, 0.9, only=ids[:20])}
    cand_dirs = {k: side(tmp / f"cand_{k}", doc, v, setting="w2c_x", checkpoint={"manifest_sha256": "d" * 64}) for k, v in variants.items()}
    (tmp / "hn").mkdir()
    (tmp / "hf").mkdir()
    hn, hf = health_files(tmp / "hn", NOFLAG), health_files(tmp / "hf", FLAG)
    return {"tmp": tmp, "path": path, "doc": doc, "ids": ids, "ref": ref, "ref_dirs": ref_dirs, "cand": cand_dirs, "variants": variants,
            "health_noflag": hn, "health_flag": hf}


def ev(gate, w, cand, health=None, **kw):
    hb, hc = health if health is not None else (None, None)
    return gate.evaluate(w["ref_dirs"], w["cand"][cand] if isinstance(cand, str) else cand, "C1", splits_file=w["path"],
                         unseal=kw.pop("unseal", "G-EX"), health_base=hb, health_candidate=hc, **kw)


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------


def test_rule_constants(gate, w2_gate):
    assert (gate.LEVEL, gate.N_RESAMPLES, gate.NONINFERIORITY_LOW, gate.GATE_N) == (0.975, 10_000, -0.02, 500)
    assert gate.W2G.decide.__code__.co_code == w2_gate.decide.__code__.co_code  # 同一个函数（scripts/w2_gate.py），不复制
    assert gate.REFERENCE_SETTING == "w2c" and gate.GROUP == "devcheck3"


def test_four_combinations(gate, world):
    w = world
    d = ev(gate, w, "better", w["health_noflag"])  # (i) 过、(ii) 过
    assert d["rules"]["i"]["pass"] and d["rules"]["ii"]["result"] == "通过" and d["decision"] == "通过" and d["ci_low"] > 0
    d = ev(gate, w, "better", w["health_flag"])  # (i) 过、(ii) 不过（标红）→ 通过
    assert d["rules"]["i"]["pass"] and d["rules"]["ii"]["result"] == "不通过" and d["decision"] == "通过"
    assert d["health"]["flagged"] == ["mmlu_pro_letter"]
    d = ev(gate, w, "close", w["health_noflag"])  # (i) 不过、(ii) 过 → 通过
    assert not d["rules"]["i"]["pass"] and -0.02 < d["ci_low"] <= 0 and d["rules"]["ii"]["result"] == "通过" and d["decision"] == "通过"
    d = ev(gate, w, "close", w["health_flag"])  # (i) 不过、(ii) 不过（标红）→ 不通过
    assert not d["rules"]["i"]["pass"] and d["rules"]["ii"]["result"] == "不通过" and d["decision"] == "不通过"
    d = ev(gate, w, "worse", w["health_noflag"])  # (i) 不过、(ii) 不过（下界 ≤ −0.02）→ 不通过
    assert d["ci_low"] < -0.02 and not d["health"]["any_flag"] and d["rules"]["ii"]["result"] == "不通过" and d["decision"] == "不通过"
    assert d["n"] == 500 and d["gate_point"]["is_gate_point"] and d["verdict"] == "不通过"


def test_rule_ii_undecidable_without_or_with_incomplete_health(gate, world, tmp_path):
    w = world
    d = ev(gate, w, "close")
    assert d["rules"]["ii"]["result"] == "无法判定" and d["decision"] == "不通过" and "没给健康检查" in d["rules"]["ii"]["why"]
    d = ev(gate, w, "better")
    assert d["rules"]["ii"]["result"] == "无法判定" and d["decision"] == "通过"  # (i) 照样能过
    hb, hc = health_files(tmp_path, NOFLAG, host_c="other")
    d = ev(gate, w, "close", (hb, hc))
    assert d["rules"]["ii"]["result"] == "无法判定" and "同机" in d["rules"]["ii"]["why"] and d["decision"] == "不通过"
    hb, hc = health_files(tmp_path, NOFLAG, limit=3)  # --limit 的健康检查（题目一样多，limit 记的不是 None）
    d = ev(gate, w, "close", (hb, hc))
    assert d["rules"]["ii"]["result"] == "无法判定" and "--limit" in d["rules"]["ii"]["why"]


def test_ci_is_w2_gate_decide(gate, w2_gate, world):
    w = world
    d = ev(gate, w, "close", w["health_noflag"])
    ids = w["ids"]
    cand = recs_of({i: w["variants"]["close"][i] for i in ids})
    ref = recs_of({i: w["ref"][i] for i in ids})
    dd = w2_gate.decide(cand, ref)
    assert (d["ci_low"], d["ci_high"], d["diff"], d["diff_exact"]) == (dd["ci_low"], dd["ci_high"], dd["diff"], dd["diff_exact"])
    assert (d["ciou"]["candidate"], d["ciou"]["reference"]) == (dd["ciou"]["main"], dd["ciou"]["w1f"])
    pa = np.array([[r["score"]["inter"], r["score"]["union"]] for r in cand], dtype=np.float64)
    pb = np.array([[r["score"]["inter"], r["score"]["union"]] for r in ref], dtype=np.float64)
    ci = paired_bootstrap_ci(pa, pb, statistic=ratio_of_sums, n_resamples=10_000, level=0.975, seed=C.BOOTSTRAP_SEED)
    assert (d["ci_low"], d["ci_high"]) == (ci.low, ci.high)
    g = d["diagnostics"]["giou"]
    assert g["level"] == 0.975 and g["n_resamples"] == 10_000 and set(d["diagnostics"]["ciou_by_split"]) == set(EX.group_split_names("devcheck3"))


def test_record_order_does_not_matter(gate, world, tmp_path):
    w = world
    split_of = EX.split_of_sample(w["doc"], EX.DEVCHECK3)
    dirs = []
    for nm in EX.group_split_names(EX.DEVCHECK3):
        rs = recs_of({i: v for i, v in w["variants"]["close"].items() if split_of[i] == nm})
        random.Random(7).shuffle(rs)
        dirs.append(make_leaf(tmp_path / "shuf", w["doc"], nm, rs, setting="w2c_x", checkpoint={"manifest_sha256": "d" * 64}))
    a, b = ev(gate, w, list(reversed(dirs)), w["health_noflag"]), ev(gate, w, "close", w["health_noflag"])
    assert (a["ci_low"], a["ci_high"], a["diff"]) == (b["ci_low"], b["ci_high"], b["diff"])


# ---------------------------------------------------------------------------
# 输入核对
# ---------------------------------------------------------------------------


def test_refuses_bad_inputs(gate, world, tmp_path):
    w = world
    doc = w["doc"]
    split_of = EX.split_of_sample(doc, EX.DEVCHECK3)
    names = EX.group_split_names(EX.DEVCHECK3)

    def cand_dirs(root, sc, drop=(), dup=None, extra=None, **kw):
        out = []
        for nm in names:
            rs = recs_of({i: v for i, v in sc.items() if split_of[i] == nm and i not in drop})
            if dup and nm == names[0]:
                rs.append(rs[0])
            if extra and nm == names[0]:
                rs.append({"id": extra, "score": {"inter": 1, "union": 2, "iou": 0.5}})
            out.append(make_leaf(root, doc, nm, rs, setting="w2c_x", **kw))
        return out

    sc = w["variants"]["close"]
    cases = [
        (cand_dirs(tmp_path / "miss", sc, drop={w["ids"][0]}), "缺 1 句"),
        (cand_dirs(tmp_path / "dup", sc, dup=True), "重复"),
        (cand_dirs(tmp_path / "extra", sc, extra="refcoco_unc/val/999999", w2=False), "多 1 句"),
        (cand_dirs(tmp_path / "wrongsplit", sc, extra="refcocog_umd/val/" + doc["splits"][names[2]]["ids"][0]), "不属于它"),
        (cand_dirs(tmp_path / "sha", sc, split_sha="0" * 64), "sha256"),
        (cand_dirs(tmp_path / "two", sc)[:2], "缺"),
    ]
    for dirs, msg in cases:
        with pytest.raises(ValueError, match=msg):
            ev(gate, w, dirs)
    # 参照不是 W2-c
    bad_ref = side(tmp_path / "ref_w2a", doc, w["ref"], setting="w2a")
    with pytest.raises(gate.GateInputError, match="w2c"):
        gate.evaluate(bad_ref, w["cand"]["close"], "C1", splits_file=w["path"], unseal="G-EX")
    plain = side(tmp_path / "ref_plain", doc, w["ref"], w2=False)
    with pytest.raises(gate.GateInputError, match="没有 W2 的身份"):
        gate.evaluate(plain, w["cand"]["close"], "C1", splits_file=w["path"], unseal="G-EX")
    # 门名
    for u in (None, "G-W2"):
        with pytest.raises(EX.SealedSplitError):
            ev(gate, w, "close", unseal=u)
    # 待生成的清单
    with pytest.raises(gate.GateInputError, match="还没生成"):
        gate.evaluate(w["ref_dirs"], w["cand"]["close"], "C1", splits_file=EX.EXPLORE_SPLITS_PATH, unseal="G-EX")


def test_candidate_without_w2_identity_is_accepted(gate, world, tmp_path):
    """候选可以是别的运行器的输出：只要 W2 格式的 predictions.jsonl（与 meta.json 的主机名，判定点要同机）。"""
    w = world
    dirs = side(tmp_path / "plain", w["doc"], w["variants"]["better"], w2=False)
    d = ev(gate, w, dirs, w["health_noflag"])
    assert d["decision"] == "通过" and d["gate_point"]["is_gate_point"]


# ---------------------------------------------------------------------------
# 健康检查与判定点
# ---------------------------------------------------------------------------


def test_health_flag_reuses_health_check_compare(gate, world, tmp_path):
    w = world
    for hb_hc, want in ((w["health_noflag"], []), (w["health_flag"], ["mmlu_pro_letter"])):
        st = gate.health_status(*map(str, hb_hc))
        res = health_compare(*map(str, hb_hc))
        assert st["flagged"] == res["flagged"] == want and st["any_flag"] == res["any_flag"] and st["valid"]
        out = tmp_path / f"cmp_{len(want)}.json"
        out.write_text(json.dumps(res), encoding="utf-8")  # scripts/health_check.py --compare --out 写的就是 compare 的结果
        st2 = gate.health_status(compare_json=str(out))
        assert st2["flagged"] == want and st2["valid"] and "同机未核对" in st2["source"]["note"]
        d1 = ev(gate, w, "close", hb_hc)
        d2 = gate.evaluate(w["ref_dirs"], w["cand"]["close"], "C1", splits_file=w["path"], unseal="G-EX", health_compare_json=str(out))
        assert d1["rules"]["ii"]["result"] == d2["rules"]["ii"]["result"]
    with pytest.raises(gate.GateInputError, match="二选一"):
        gate.health_status(str(w["health_flag"][0]), str(w["health_flag"][1]), str(out))
    # 候选的健康检查用的检查点与候选的运行不同
    hb, hc = w["health_noflag"]
    rep = json.loads(Path(hc).read_text())
    rep["checkpoint"] = {"manifest_sha256": "e" * 64}
    bad = tmp_path / "hc_other_ck.json"
    bad.write_text(json.dumps(rep))
    with pytest.raises(gate.GateInputError, match="检查点"):
        ev(gate, w, "close", (hb, bad))


def test_gate_point_checks(gate, world, tmp_path):
    w = world
    other_host = side(tmp_path / "h2", w["doc"], w["variants"]["better"], setting="w2c_x", host="hpcgpu07",
                      checkpoint={"manifest_sha256": "d" * 64})
    d = ev(gate, w, other_host, w["health_noflag"])
    assert not d["gate_point"]["checks"]["same_host"] and d["verdict"].startswith("彩排，不作数") and d["decision"] == "通过"
    no_gate = side(tmp_path / "ng", w["doc"], w["variants"]["better"], setting="w2c_x", gate=None, checkpoint={"manifest_sha256": "d" * 64})
    d = ev(gate, w, no_gate, w["health_noflag"])
    assert not d["gate_point"]["checks"]["unsealed_with_G-EX"] and not d["gate_point"]["is_gate_point"]
    tpath, tdoc = toy_manifest(tmp_path, tiny=True)
    ids = EX.group_ids(tdoc, EX.DEVCHECK3)
    ref = scores(ids, seed=3)
    rd = side(tmp_path / "tref", tdoc, ref, checkpoint=CK)
    cd = side(tmp_path / "tcand", tdoc, scaled(ref, 1.2), setting="w2c_x")
    d = gate.evaluate(rd, cd, "C1", splits_file=tpath, unseal="G-EX")
    assert d["n"] == 8 and not d["gate_point"]["checks"]["manifest_not_tiny"] and not d["gate_point"]["checks"]["n_is_500"]
    assert d["verdict"].startswith("彩排，不作数")


def test_cli(gate, world, tmp_path):
    w = world
    hb, hc = w["health_noflag"]
    args = ["--candidate-name", "C1", "--unseal-devcheck3", "G-EX", "--splits-file", str(w["path"]),
            "--reference", *map(str, w["ref_dirs"]), "--candidate", *map(str, w["cand"]["close"]),
            "--health-base", str(hb), "--health-candidate", str(hc), "--out-dir", str(tmp_path / "g")]
    assert gate.main(args) == 0
    doc = json.loads((tmp_path / "g" / "gate_explore.json").read_text(encoding="utf-8"))
    md = (tmp_path / "g" / "gate_explore.md").read_text(encoding="utf-8")
    assert doc["decision"] == "通过" and doc["rules"]["ii"]["result"] == "通过" and doc["unseal"]["gate"] == "G-EX"
    assert "结论：通过" in md and "(ii)" in md and "gIoU" in md
    assert gate.main(args[:2] + args[4:]) == 2  # 没给门名
    bad = args.copy()
    bad[bad.index("--candidate") + 1] = str(tmp_path / "nowhere")
    assert gate.main(bad) == 2


def test_rule_boundaries_are_strict(gate, monkeypatch):
    """(i) 下界 > 0、(ii) 下界 > −0.02 都是严格不等号：恰好等于时不通过。"""
    base = {"n": 500, "ciou": {"main": 0.6, "w1f": 0.6}, "diff": 0.0, "diff_exact": "0/1", "ci_high": 0.03, "level": 0.975,
            "n_resamples": 10_000, "seed": C.BOOTSTRAP_SEED}
    ok_health = {"given": True, "valid": True, "any_flag": False, "flagged": [], "problems": []}
    for low, want_i, want_ii in ((0.0, False, "通过"), (1e-12, True, "通过"), (-0.02, False, "不通过"), (-0.0199, False, "通过")):
        monkeypatch.setattr(gate.W2G, "decide", lambda a, b, low=low: {**base, "ci_low": low})
        d = gate.decide([], [], ok_health)
        assert (d["rules"]["i"]["pass"], d["rules"]["ii"]["result"]) == (want_i, want_ii), low
        assert d["decision"] == ("通过" if want_i or want_ii == "通过" else "不通过")
    d = gate.decide([], [], {**ok_health, "any_flag": True, "flagged": ["ifeval"]})
    assert d["rules"]["ii"]["result"] == "不通过" and "ifeval" in d["rules"]["ii"]["why"]
