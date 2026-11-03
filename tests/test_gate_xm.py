"""DEV-0081：X-M 的去留判定（scripts/gate_xm.py；D165 第五节 = 预注册 XM-5，F1 口径 XM-3a）。

- 四条原文照录（与负责人原文、预注册逐字相同）、阈值与 bootstrap 的次数、水平、种子写死；
- 指标桩：Rex-Omni 宏平均 F1 的手算例子；按图像重抽的配对 bootstrap 与 src.stats 的参考实现逐位相同（只经 f1_rexomni 这个公开名字）；
  (3) 的逐句配对与 scripts/mgr/x_test_table.paired、src.stats 逐位相同；
- 第 4 条：d170（预注册 XM-5a，缺省、唯一进结论）= X-M − 同节点基座，MMLU-Pro 下界 > −5.7、IFEval > −8.1；d165 原文 = X-M − X-2
  seed 0、下界 > −2，总是照算但不进结论；都由 lang_battery 运行的逐题结果（手造）逐题配对，10,000 次、不取整；lb_compare 只作诊断；
- 同节点（判定集三个叶目录、指代分割两边、两次 lang_battery）与提示版本（检查点记的 = 判定集叶目录记的）的判定点核对，不过就"不作数"；
- 合成的输入：四条各自通过与不通过；全部通过时 verdict = src/xm/data.XM_GATE_PASS、判定点全真，DEV-0079 的 check_xm_gate 认它；
- 彩排标记；缺输入、门名不对、对或句子的集合不对、检查点对不上时拒绝（退出码 2，什么都不写）；
- 端到端：DEV-0079 的假数据根上的真划分与密封守卫。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src import constants as C
from src import stats as S
from src.xm import data as XD
from src.xm import metrics as M
from tests import _xm_train_fakes as XF
from tests._xm_fakes import build_fake_root

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GX = _load(REPO / "scripts" / "gate_xm.py", "_gate_xm")
XM_SHA = hashlib.sha256((REPO / "configs" / "xm_train.json").read_bytes()).hexdigest()
X2_SHA = hashlib.sha256((REPO / "configs" / "x2_train.json").read_bytes()).hexdigest()
REF = "b" * 64
N_MMLU, N_IFEVAL = 2000, 541
BASE_MMLU = [k % 2 == 0 for k in range(N_MMLU)]  # 同节点基座：MMLU-Pro 50%
XM_MMLU = [k % 2 == 0 or k % 10 == 1 for k in range(N_MMLU)]  # X-M：60%，逐题只多不少 → +10 点
X2_MMLU = BASE_MMLU  # X-2 seed 0（D165 原文的诊断用）：缺省同基座
BASE_IFEVAL = [k % 3 == 0 for k in range(N_IFEVAL)]  # IFEval 181 / 541
X2_IFEVAL = BASE_IFEVAL
X2_MMLU_HIGH = [k % 2 == 0 or k % 10 in (1, 3) for k in range(N_MMLU)]  # 70%：X-M − X-2 = −10 点（原文不通过）


def _drop(v, n):
    """把前 n 个对的改成错（X-M 比 X-2 差 n 题）。"""
    out, k = list(v), 0
    for i, x in enumerate(out):
        if x and k < n:
            out[i], k = False, k + 1
    return out


def make_inputs(tmp: Path, *, xm_seg="good", box_seg="mid", xm_det="good", base_det="mid", shrink=0.0, xm_mmlu=XM_MMLU,
                xm_ifeval=BASE_IFEVAL, x2_mmlu=X2_MMLU, x2_ifeval=X2_IFEVAL, n_images=500, dec_hosts=("hpcgpu06",) * 3,
                refseg_hosts=("hpcgpu07", "hpcgpu07"), lb_hosts=("hpcgpu07", "hpcgpu07", "hpcgpu07"), ref_seed=0, tiny=False, limit=None,
                lb_limit=None, unseal=XD.GATE, leaf_pv=XF.PROMPT_VERSION, ck_pv=XF.PROMPT_VERSION, lb_compare=False, prompts_frozen=True) -> dict:
    """lb_hosts = (X-M、基座、X-2 seed 0) 三次 lang_battery 的节点。"""
    pairs = XF.decision_pairs(n_images)
    ckd, cand = XF.write_xm_checkpoint(tmp / "train", config_sha256=XM_SHA, prompt_version=ck_pv)
    ck = XF.checkpoint(cand, config_sha256=XM_SHA)
    rk = XF.checkpoint(REF, config_sha256=X2_SHA, seed=ref_seed)
    lv = tmp / "leaves"
    main = XF.write_xm_leaf(lv / "xm_main", GX.XM_MAIN, pairs, det=xm_det, seg=xm_seg, ck=ck, host=dec_hosts[0], tiny=tiny, limit=limit,
                            unseal=unseal, prompt_version=leaf_pv, prompts_frozen=prompts_frozen)
    box = XF.write_xm_leaf(lv / "xm_box", GX.XM_BOX, pairs, det=xm_det, seg=box_seg, ck=ck, host=dec_hosts[1], unseal=unseal,
                           prompt_version=leaf_pv, prompts_frozen=prompts_frozen)
    base = XF.write_xm_leaf(lv / "base_det", GX.BASE_DET, pairs, det=base_det, seg=None, ck=None, host=dec_hosts[2], unseal=unseal,
                            prompt_version=leaf_pv, prompts_frozen=prompts_frozen)
    xm_ref = [XF.write_refseg_leaf(lv / "xm_refseg" / sp, sp, n, ck=ck, shrink=shrink, host=refseg_hosts[0])
              for sp, n in GX.REFSEG_SPLITS.items()]
    x2_ref = [XF.write_refseg_leaf(lv / "x2_refseg" / sp, sp, n, ck=rk, host=refseg_hosts[1]) for sp, n in GX.REFSEG_SPLITS.items()]
    lbx = XF.write_lb_run(tmp / "lb" / "xm", ck=cand, mmlu=xm_mmlu, ifeval=xm_ifeval, host=lb_hosts[0], limit=lb_limit)
    lbb = XF.write_lb_run(tmp / "lb" / "base", ck=None, mmlu=BASE_MMLU, ifeval=BASE_IFEVAL, host=lb_hosts[1])
    lb2 = XF.write_lb_run(tmp / "lb" / "x2", ck=REF, mmlu=x2_mmlu, ifeval=x2_ifeval, host=lb_hosts[2])
    kw = {"xm_checkpoint": str(ckd), "xm_main": str(main), "xm_box": str(box), "base_det": str(base),
          "xm_refseg": [str(p) for p in xm_ref], "x2_refseg": [str(p) for p in x2_ref], "lb_xm": str(lbx), "lb_base": str(lbb),
          "lb_x2": str(lb2)}
    if lb_compare:  # 诊断：故意给一个"不通过"的 lb_compare，结论不受影响
        kw["lb_compare"] = str(XF.write_lb_compare(tmp / "lbc" / "lb_compare_xm.json", base_ck=REF, other_ck=cand, mmlu_ci=(-9.0, -7.0)))
    return {"pairs": pairs, "ckdir": ckd, "cand": cand, "kw": kw}


def cli_args(inp: dict, out: Path, *extra: str, gate: str | None = XD.GATE) -> list[str]:
    k = inp["kw"]
    args = ([] if gate is None else ["--gate", gate]) + [
        "--data-root", str(out.parent), "--xm-checkpoint", k["xm_checkpoint"], "--xm-main", k["xm_main"], "--xm-box-grabcut", k["xm_box"],
        "--base-det", k["base_det"], "--xm-refseg", *k["xm_refseg"], "--x2-refseg", *k["x2_refseg"], "--lb-xm", k["lb_xm"],
        "--lb-base", k["lb_base"], "--lb-x2", k["lb_x2"], "--out-dir", str(out)]
    if "lb_compare" in k:
        args += ["--lb-compare", k["lb_compare"]]
    return args + list(extra)


@pytest.fixture
def fake_pairs(monkeypatch):
    """xm_decision 的对：换掉数据层的读取，守卫照用 DEV-0079 的 require_unsealed。"""
    box = {"pairs": None}

    def load_pairs(split, data_root, *, gate=None, xm_gate=None, splits_path=None, exclusions_path=None):
        XD.require_unsealed(split, gate)
        assert split == XD.XM_DECISION
        return box["pairs"]

    monkeypatch.setattr(XD, "load_pairs", load_pairs)
    return box


def run_gate(fake_pairs, inp: dict, **kw) -> dict:
    fake_pairs["pairs"] = inp["pairs"]
    return GX.evaluate(gate=kw.pop("gate", XD.GATE), data_root="unused", **inp["kw"], **kw)


# ---------------------------------------------------------------------------
# 规则照录
# ---------------------------------------------------------------------------


def test_rules_verbatim_and_constants():
    directive = (REPO / "docs" / "lead_answers" / "2026-11-03-xm-directive.md").read_text(encoding="utf-8")
    prereg = (REPO / "PREREGISTRATION.md").read_text(encoding="utf-8").replace("**", "")
    xm5 = prereg[prereg.index("### XM-5"): prereg.index("### XM-6")]
    for k, text in GX.RULES.items():
        assert f"{k}. {text}" in directive and f"{k}. {text}" in xm5, k
    assert dict(GX.THRESHOLDS) == {"1": 0.0, "2": 0.0, "3": -0.02, "4": -2.0}
    assert dict(GX.D170_THRESHOLDS) == {"mmlu_pro": -5.7, "ifeval": -8.1}  # XM-5a：−3.7 − 2、−6.1 − 2
    assert "−5.7" in GX.RULE_4_D170 and "−8.1" in GX.RULE_4_D170
    if "#### XM-5a" in prereg:  # 预注册 XM-5a（D170）在 main 上；这个 worktree 的基线还没有它
        xm5a = prereg[prereg.index("#### XM-5a"):]
        head = GX.RULE_4_D170.split("：MMLU-Pro：")[0]
        assert head in xm5a and "MMLU-Pro：CI 下界 > −5.7" in xm5a and "IFEval：CI 下界 > −8.1" in xm5a
    assert (GX.N_RESAMPLES, GX.LEVEL, GX.SEED) == (10_000, 0.95, C.BOOTSTRAP_SEED)
    assert GX.CRITERION4_VARIANTS == ("d170",) and GX.PASS == XD.XM_GATE_PASS == "通过"
    assert GX.DECISION_N_IMAGES == 500 and GX.REFSEG_N == 2300
    assert {k: (v[0], v[2], v[3]) for k, v in GX.LB_ROWS.items()} == {"mmlu_pro": ("mmlu_pro_letter", "accuracy", 2000),
                                                                      "ifeval": ("ifeval", "prompt_level_strict", 541)}


def test_gate_uses_metrics_only_through_its_public_names():
    """src/xm/metrics（DEV-0080）只经 M.table、M.point、M.paired_bootstrap 与 M.PRIMARY 用到；不用 src/xm/prompts。"""
    import re

    src = (REPO / "scripts" / "gate_xm.py").read_text(encoding="utf-8")
    code = src.split('"""', 2)[2]  # 去掉模块说明
    assert set(re.findall(r"\bM\.(\w+)", code)) == {"table", "point", "paired_bootstrap", "PRIMARY"}
    assert "src.xm.prompts" not in code and "from src.xm import prompts" not in code
    assert re.findall(r"^from src\.xm import (\w+) as (\w+)", code, flags=re.M) == [("data", "XD"), ("metrics", "M")]
    assert M.PRIMARY == "f1_rexomni" and M.DECISION_N_RESAMPLES == GX.N_RESAMPLES


# ---------------------------------------------------------------------------
# 指标（DEV-0080 的 src/xm/metrics；它自己的测试在 tests/test_xm_metrics.py）
# ---------------------------------------------------------------------------


def test_f1_criterion_is_metrics_paired_bootstrap_and_order_free():
    """(1)(2) 的读数就是 src/xm/metrics.table → point / paired_bootstrap（主读数，10,000 次）；行的先后不影响（按 id 排序后配对）。"""
    pairs = XF.decision_pairs(60)
    a = [{"id": p.id, "image_key": f"coco2017:{p.image}", "category_id": p.category_id, "det": XF.eval_counts(p.id, "good")} for p in pairs]
    b = [{"id": p.id, "image_key": f"coco2017:{p.image}", "category_id": p.category_id, "det": XF.eval_counts(p.id, "mid")} for p in pairs]
    rng = np.random.default_rng(0)
    a_shuf = [a[i] for i in rng.permutation(len(a))]
    got = GX.f1_criterion("2", "box_f1_rexomni", GX.DET, GX.pair_rows(a_shuf, GX.DET, "a"), GX.pair_rows(b, GX.DET, "b"), "a", "b")
    ta, tb = M.table(sorted(a, key=lambda r: r["id"]), "det"), M.table(sorted(b, key=lambda r: r["id"]), "det")
    ci = M.paired_bootstrap(ta, tb, M.PRIMARY, n_resamples=10_000, level=0.95, seed=C.BOOTSTRAP_SEED)
    assert (got["diff"], got["ci_low"], got["ci_high"]) == (ci.estimate, ci.low, ci.high)
    assert (got["candidate"], got["control"]) == (M.point(ta, M.PRIMARY), M.point(tb, M.PRIMARY)) and got["candidate"] == 1.0
    assert got["n_images"] == 60 and got["statistic"] == "f1_rexomni" and got["iou"] == "box"
    with pytest.raises(GX.GateInputError, match="seg"):  # base_det 的 seg 是 null
        GX.pair_rows([{**a[0], "seg": None}], GX.SEG, "base_det")


def test_paired_ciou_ci_matches_x_test_table_and_src_stats():
    xt = _load(REPO / "scripts" / "mgr" / "x_test_table.py", "_x_test_table_for_gate_xm")
    rng = np.random.default_rng(5)
    u = rng.integers(100, 5000, size=300).astype(np.float64)
    A = np.stack([np.floor(u * rng.random(300)), u], 1)
    B = np.stack([np.floor(u * rng.random(300)), u], 1)
    got = GX.paired_ciou_ci(A, B)
    ref = S.paired_bootstrap_ci(A, B, statistic=S.ratio_of_sums, n_resamples=10_000, level=0.95, seed=C.BOOTSTRAP_SEED)
    assert (got.estimate, got.low, got.high) == (ref.estimate, ref.low, ref.high)
    ids = [f"s{i:04d}" for i in range(300)]  # x_test_table.paired 按 id 排序后配对
    da = {i: {"inter": A[k, 0], "union": A[k, 1]} for k, i in enumerate(ids)}
    db = {i: {"inter": B[k, 0], "union": B[k, 1]} for k, i in enumerate(ids)}
    p = xt.paired(da, db)
    assert (p["diff"], p["ci"]) == (got.estimate, [got.low, got.high])


# ---------------------------------------------------------------------------
# 第 4 条：d170（XM-5a，进结论）= X-M − 同节点基座；d165 原文 = X-M − X-2 seed 0（诊断）；逐题配对，10,000 次
# ---------------------------------------------------------------------------


def _ref_pp(a, b, n=10_000):
    ci = S.paired_bootstrap_ci(np.array(a, float), np.array(b, float), n_resamples=n, level=0.95, seed=C.BOOTSTRAP_SEED)
    return 100.0 * ci.low, 100.0 * ci.high


def test_criterion4_d170_hand_fixture(fake_pairs, tmp_path, capsys):
    inp = make_inputs(tmp_path, lb_compare=True)
    fake_pairs["pairs"] = inp["pairs"]
    out = tmp_path / "g"
    assert GX.main(cli_args(inp, out)) == 0  # 不给 --criterion4：缺省 d170
    doc = json.loads((out / "gate_xm.json").read_text(encoding="utf-8"))
    c4 = doc["criteria"]["4"]
    assert doc["criterion4"] == "d170" and c4["variant"] == "d170" and c4["control_name"] == "同节点基座" and "XM-5a" in c4["source"]
    mm, ie = c4["rows"]["mmlu_pro"], c4["rows"]["ifeval"]
    assert (mm["n"], mm["control_pct"], mm["xm_pct"]) == (2000, 50.0, 60.0) and mm["diff_pp"] == pytest.approx(10.0, abs=1e-9)
    assert (mm["ci_low_pp"], mm["ci_high_pp"]) == _ref_pp(XM_MMLU, BASE_MMLU)  # 逐题配对、10,000 次、不取整
    assert (mm["ci_low_pp"], mm["ci_high_pp"]) != _ref_pp(XM_MMLU, BASE_MMLU, 1000)  # 不是 lb_compare 的 1,000 次
    assert 8.0 < mm["ci_low_pp"] < 10.0 < mm["ci_high_pp"] < 12.0
    assert (ie["n"], ie["diff_pp"], ie["ci_low_pp"], ie["ci_high_pp"]) == (541, 0.0, 0.0, 0.0)  # 逐题一样：差 0、CI [0, 0]
    assert (mm["threshold"], ie["threshold"]) == (-5.7, -8.1) and c4["pass"] and c4["bootstrap"]["n_resamples"] == 10_000
    # D165 原文：总是照算、写出来，不进结论
    d = doc["diagnostics"]["criterion4_d165"]
    assert d["counts_toward_verdict"] is False and d["control_name"] == "X-2 seed 0" and d["threshold"] == -2.0 and d["same_node"] is True
    assert (d["rows"]["mmlu_pro"]["ci_low_pp"], d["rows"]["mmlu_pro"]["ci_high_pp"]) == _ref_pp(XM_MMLU, X2_MMLU)
    assert d["would_pass"] is True and d["result_if_counted"] == "满足"
    # lb_compare 只作诊断：它给的是"不通过"的读数，结论照样按逐题配对的 10,000 次
    assert doc["diagnostics"]["lb_compare"]["rows"][0]["ci95_pp"] == [-9.0, -7.0] and doc["decision"] == "通过"
    assert doc["inputs"]["lb_compare"]["role"] == "诊断，不入结论" and doc["inputs"]["lb_x2"]["role"].startswith("D165")
    md = (out / "gate_xm.md").read_text(encoding="utf-8")
    assert "第 4 条，d170（进结论）" in md and "第 4 条原文，d165（诊断，不入结论）" in md and "若计入" in md


@pytest.mark.parametrize("edit", [
    {"x2_mmlu": X2_MMLU_HIGH},  # X-2 seed 0 比 X-M 高 10 点：原文的下界远低于 −2
    {"xm_mmlu": _drop(BASE_MMLU, 80), "x2_mmlu": BASE_MMLU},  # X-M 比基座、X-2 都低 4 点：d170 下界约 −4.9 > −5.7，原文 < −2
])
def test_d165_reported_but_does_not_affect_verdict(fake_pairs, tmp_path, edit):
    doc = run_gate(fake_pairs, make_inputs(tmp_path, **edit))
    d = doc["diagnostics"]["criterion4_d165"]
    assert d["would_pass"] is False and d["rows"]["mmlu_pro"]["ci_low_pp"] < -2.0 and d["result_if_counted"] == "不满足"
    assert doc["criteria"]["4"]["pass"] is True and doc["criteria"]["4"]["rows"]["mmlu_pro"]["ci_low_pp"] > -5.7
    assert doc["decision"] == doc["verdict"] == XD.XM_GATE_PASS and doc["rehearsal"] is False


@pytest.mark.parametrize("edit,name", [
    ({"xm_mmlu": _drop(BASE_MMLU, 140), "x2_mmlu": _drop(BASE_MMLU, 140)}, "mmlu_pro"),  # −7 点：下界约 −7.9 ≤ −5.7
    ({"xm_ifeval": _drop(BASE_IFEVAL, 55), "x2_ifeval": _drop(BASE_IFEVAL, 55)}, "ifeval"),  # 约 −10.2 点 ≤ −8.1
])
def test_d170_fails_on_language_drop_even_if_d165_would_pass(fake_pairs, tmp_path, edit, name):
    doc = run_gate(fake_pairs, make_inputs(tmp_path, **edit))
    r = doc["criteria"]["4"]["rows"][name]
    assert r["ci_low_pp"] <= GX.D170_THRESHOLDS[name] and not r["pass"] and doc["decision"] == doc["verdict"] == "不通过"
    assert {k for k, c in doc["criteria"].items() if not c["pass"]} == {"4"}
    assert doc["diagnostics"]["criterion4_d165"]["would_pass"] is True  # X-2 seed 0 一样低：原文会通过，但它不进结论


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def test_all_four_pass_and_check_xm_gate_accepts(fake_pairs, tmp_path, capsys):
    inp = make_inputs(tmp_path)
    fake_pairs["pairs"] = inp["pairs"]
    out = tmp_path / "gate"
    assert GX.main(cli_args(inp, out)) == 0
    doc = json.loads((out / "gate_xm.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == XD.XM_GATE_PASS and doc["decision"] == "通过" and doc["pass"] is True and doc["rehearsal"] is False
    assert doc["checks"] and all(doc["checks"].values()) and doc["is_gate_point"] is True
    assert {"decision_same_node", "refseg_same_node", "lb_same_node", "prompt_version_matches"} <= set(doc["checks"])
    assert doc["nodes"] == {"decision": {"xm_main": "hpcgpu06", "xm_box_grabcut": "hpcgpu06", "base_det": "hpcgpu06"},
                            "refseg_xm": ["hpcgpu07"], "refseg_x2": ["hpcgpu07"], "lb_xm": "hpcgpu07", "lb_base": "hpcgpu07",
                            "lb_x2": "hpcgpu07"}
    assert doc["candidate"]["manifest_sha256"] == inp["cand"] and doc["reference"]["manifest_sha256"] == REF
    assert doc["prompt_versions"] == {"checkpoint": XF.PROMPT_VERSION, "xm_main": XF.PROMPT_VERSION, "xm_box_grabcut": XF.PROMPT_VERSION,
                                      "base_det": XF.PROMPT_VERSION}
    assert all(doc["criteria"][k]["pass"] and doc["criteria"][k]["result"] == "满足" for k in "1234")
    c1, c2, c3 = (doc["criteria"][k] for k in "123")
    assert c1["metric"] == "mask_f1_rexomni" and c2["metric"] == "box_f1_rexomni" and c3["metric"] == "ciou"
    assert c1["ci_low"] > 0 and c2["ci_low"] > 0 and c3["ci_low"] > -0.02 and c3["n"] == 2300
    assert c1["n_images"] == 500 and c1["bootstrap"]["n_resamples"] == 10_000 and c1["bootstrap"]["unit"] == "image"
    assert set(doc["inputs"]) >= {"xm_checkpoint", "xm_main", "xm_box_grabcut", "base_det", "xm_refseg", "x2_refseg", "lb_xm", "lb_base", "lb_x2",
                                  "configs"}
    assert len(doc["inputs"]["lb_xm"]["sha256"]["predictions.jsonl"]) == 64 and doc["inputs"]["lb_compare"] is None
    assert "通过" in (out / "gate_xm.md").read_text(encoding="utf-8") and "通过" in capsys.readouterr().out
    gp = XD.check_xm_gate(out / "gate_xm.json", checkpoint=inp["ckdir"])  # DEV-0079 的放行认它
    assert gp.verdict == XD.XM_GATE_PASS and gp.checkpoint_manifest_sha256 == inp["cand"]


@pytest.mark.parametrize("edit,failing", [
    ({"box_seg": "good"}, "1"),  # 描样条与框交给 GrabCut 一样好：差 0，下界不 > 0
    ({"xm_seg": "bad"}, "1"),
    ({"base_det": "good"}, "2"),
    ({"shrink": 0.1}, "3"),  # cIoU 约 −0.07
])
def test_each_criterion_can_fail(fake_pairs, tmp_path, edit, failing):
    doc = run_gate(fake_pairs, make_inputs(tmp_path, **edit))
    assert {k for k, c in doc["criteria"].items() if not c["pass"]} == {failing}
    assert doc["decision"] == doc["verdict"] == "不通过" and doc["pass"] is False and doc["rehearsal"] is False


@pytest.mark.parametrize("edit,check", [
    ({"dec_hosts": ("hpcgpu06", "hpcgpu06", "hpcgpu07")}, "decision_same_node"),  # (a)
    ({"refseg_hosts": ("hpcgpu06", "hpcgpu07")}, "refseg_same_node"),  # (b) X-M 的指代分割在 06，X-2 seed 0 的在 07
    ({"lb_hosts": ("hpcgpu07", "hpcgpu06", "hpcgpu07")}, "lb_same_node"),  # (c) X-M 的大题量在 07，基座在 06
    ({"lb_hosts": (None, None, None)}, "lb_same_node"),  # 读不到节点：不算同节点
    ({"leaf_pv": "xm-prompts-tuned"}, "prompt_version_matches"),  # 判定集叶目录的提示与训练时不同
    ({"ck_pv": None}, "prompt_version_matches"),  # 检查点没记提示版本
    ({"prompts_frozen": False}, "decision_prompts_frozen"),  # 判定集叶目录用的不是冻结的提示（D168）
    ({"tiny": True}, "not_tiny"),
    ({"limit": 5}, "no_limit"),
    ({"n_images": 499}, "decision_n_images_is_500"),
    ({"ref_seed": 1}, "reference_is_x2_train_seed0"),
    ({"lb_limit": 100}, "lb_full"),
    ({"unseal": None}, "decision_unsealed_with_G-XM"),
])
def test_gate_point_checks_make_rehearsal(fake_pairs, tmp_path, edit, check):
    doc = run_gate(fake_pairs, make_inputs(tmp_path, **edit))
    assert doc["checks"][check] is False and doc["rehearsal"] is True and check in doc["rehearsal_reasons"]
    assert doc["verdict"].startswith(GX.REHEARSAL) and doc["verdict"] != XD.XM_GATE_PASS


def test_lb_node_from_run_identity_without_meta_json(fake_pairs, tmp_path):
    inp = make_inputs(tmp_path)
    for side in ("xm", "base", "x2"):  # 只留 lang_battery.json 的 run.meta.hostname
        (tmp_path / "lb" / side / "meta.json").unlink()
    doc = run_gate(fake_pairs, inp)
    assert doc["checks"]["lb_same_node"] is True and doc["nodes"]["lb_xm"] == doc["nodes"]["lb_base"] == "hpcgpu07"


def test_d165_pair_on_another_node_is_only_a_diagnostic(fake_pairs, tmp_path):
    """X-2 seed 0 的大题量在别的节点：只记在 D165 原文的诊断里（same_node = false），判定点照样全真。"""
    doc = run_gate(fake_pairs, make_inputs(tmp_path, lb_hosts=("hpcgpu07", "hpcgpu07", "hpcgpu06")))
    assert doc["diagnostics"]["criterion4_d165"]["same_node"] is False and doc["checks"]["lb_same_node"] is True
    assert all(doc["checks"].values()) and doc["verdict"] == XD.XM_GATE_PASS


def test_rehearsal_flag(fake_pairs, tmp_path):
    inp = make_inputs(tmp_path)
    fake_pairs["pairs"] = inp["pairs"]
    out = tmp_path / "g"
    assert GX.main(cli_args(inp, out, "--rehearsal")) == 0
    doc = json.loads((out / "gate_xm.json").read_text(encoding="utf-8"))
    assert doc["rehearsal"] is True and doc["verdict"] == f"{GX.REHEARSAL}（规则给出：通过）" and all(doc["checks"].values())
    assert doc["rehearsal_reasons"] == ["--rehearsal"] and "彩排" in (out / "gate_xm.md").read_text(encoding="utf-8")
    with pytest.raises(XD.XMGateError, match="彩排"):
        XD.check_xm_gate(out / "gate_xm.json", checkpoint=inp["ckdir"])
    assert GX.main(cli_args(inp, tmp_path / "s", "--splits", str(tmp_path / "other_splits.json"))) == 0  # 别的清单：只能彩排
    assert json.loads((tmp_path / "s" / "gate_xm.json").read_text())["checks"]["decision_splits_is_repo_config"] is False


def _refuse(fake_pairs, inp, out, capsys, *extra, gate=XD.GATE, msg):
    fake_pairs["pairs"] = inp["pairs"]
    assert GX.main(cli_args(inp, out, *extra, gate=gate)) == 2
    assert msg in capsys.readouterr().err
    assert not (out / "gate_xm.json").exists()


def test_refusals(fake_pairs, tmp_path, capsys):
    inp = make_inputs(tmp_path)
    out = tmp_path / "out"
    k = inp["kw"]
    _refuse(fake_pairs, inp, out, capsys, gate=None, msg="G-XM")  # 不给门名
    _refuse(fake_pairs, inp, out, capsys, gate="G-EX", msg="G-XM")
    for bad, msg in ((dict(k, lb_xm=str(tmp_path / "missing")), "不在"), (dict(k, base_det=str(tmp_path / "nope")), "不在"),
                     (dict(k, xm_checkpoint=str(tmp_path)), "ckpt/final/model")):
        _refuse(fake_pairs, {**inp, "kw": bad}, out, capsys, msg=msg)
    # 没写完
    m = Path(k["xm_box"]) / "metrics.json"
    m.write_text(json.dumps({"complete": False}))
    _refuse(fake_pairs, inp, out, capsys, msg="没有写完")
    m.write_text(json.dumps({"complete": True}))
    # 对的集合不是 xm_decision 的全部（多一对）
    _refuse(fake_pairs, {**inp, "pairs": inp["pairs"] + [XF.decision_pairs(501)[-1]]}, out, capsys, msg="对的集合")
    # 检查点对不上
    alt = make_inputs(tmp_path / "alt")
    assert alt["cand"] != inp["cand"]
    _refuse(fake_pairs, {**inp, "kw": dict(k, xm_checkpoint=alt["kw"]["xm_checkpoint"])}, out, capsys, msg="--xm-checkpoint")
    bd = XF.write_xm_leaf(tmp_path / "bd", GX.BASE_DET, inp["pairs"], det="mid", seg=None, ck=XF.checkpoint(inp["cand"], config_sha256=XM_SHA))
    _refuse(fake_pairs, {**inp, "kw": dict(k, base_det=str(bd))}, out, capsys, msg="基座零样本")
    _refuse(fake_pairs, {**inp, "kw": dict(k, base_det=alt["kw"]["xm_box"])}, out, capsys, msg="xm_decision")  # 设置不对
    _refuse(fake_pairs, {**inp, "kw": dict(k, xm_box=alt["kw"]["xm_box"])}, out, capsys, msg="xm_box_grabcut")
    _refuse(fake_pairs, {**inp, "kw": dict(k, lb_xm=alt["kw"]["lb_xm"])}, out, capsys, msg="--lb-xm")
    lb_wrong = XF.write_lb_run(tmp_path / "lbw", ck="c" * 64, mmlu=X2_MMLU, ifeval=X2_IFEVAL)
    _refuse(fake_pairs, {**inp, "kw": dict(k, lb_x2=str(lb_wrong))}, out, capsys, msg="--lb-x2")
    _refuse(fake_pairs, {**inp, "kw": dict(k, lb_base=str(lb_wrong))}, out, capsys, msg="--lb-base")  # 基座一边带了检查点
    # 两次 lang_battery 的题目不同；没跑完的运行
    lb_short = XF.write_lb_run(tmp_path / "lbs", ck=REF, mmlu=X2_MMLU[:1000], ifeval=X2_IFEVAL)
    _refuse(fake_pairs, {**inp, "kw": dict(k, lb_x2=str(lb_short))}, out, capsys, msg="题目或数据不同")
    lb_bad = XF.write_lb_run(tmp_path / "lbb", ck=REF, mmlu=X2_MMLU, ifeval=X2_IFEVAL, ok=False)
    _refuse(fake_pairs, {**inp, "kw": dict(k, lb_x2=str(lb_bad))}, out, capsys, msg="一次跑完")
    # 指代分割：少一个划分 → 两边句子集合不同；设置不对
    _refuse(fake_pairs, {**inp, "kw": dict(k, xm_refseg=k["xm_refseg"][:-1])}, out, capsys, msg="句子集合不同")
    _refuse(fake_pairs, {**inp, "kw": dict(k, xm_refseg=[k["xm_main"]] + k["xm_refseg"][1:])}, out, capsys, msg="W2")
    # 进结论的第 4 条只有 d170（XM-5a）：d165 原文只作诊断，不能选来定结论；必填的输入缺了：argparse 退出码 2
    for v in ("d165", "b"):
        with pytest.raises(SystemExit) as e:
            GX.main(cli_args(inp, out, "--criterion4", v))
        assert e.value.code == 2
    with pytest.raises(SystemExit) as e:  # --lb-base 必填
        GX.main([a for a in cli_args(inp, out) if a not in ("--lb-base", k["lb_base"])])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        GX.main(["--gate", XD.GATE, "--data-root", "x", "--out-dir", str(out)])
    assert e.value.code == 2 and not (out / "gate_xm.json").exists()


def test_end_to_end_on_dev0079_fake_splits(tmp_path, capsys):
    """DEV-0079 的假数据根与 tiny 划分：真的 load_pairs（密封守卫 G-XM）；3 张图，所以只能彩排。"""
    root = tmp_path / "root"
    build_fake_root(root)
    sp = tmp_path / "xm_splits.json"
    sp.write_text(XD.dump_config(XD.build_splits_doc(root, tiny=True)), encoding="utf-8")
    pairs = XD.load_pairs(XD.XM_DECISION, root, gate=XD.GATE, splits_path=sp)
    with pytest.raises(XD.SealedSplitError):
        XD.load_pairs(XD.XM_DECISION, root, splits_path=sp)
    inp = make_inputs(tmp_path / "in", n_images=1)
    ck = XF.checkpoint(inp["cand"], config_sha256=XM_SHA)
    lv = tmp_path / "dec"
    kw = dict(inp["kw"], xm_main=str(XF.write_xm_leaf(lv / "m", GX.XM_MAIN, pairs, det="good", seg="good", ck=ck)),
              xm_box=str(XF.write_xm_leaf(lv / "b", GX.XM_BOX, pairs, det="good", seg="mid", ck=ck)),
              base_det=str(XF.write_xm_leaf(lv / "d", GX.BASE_DET, pairs, det="mid", seg=None, ck=None)))
    out = tmp_path / "gate"
    args = ["--data-root", str(root), "--splits", str(sp), "--xm-checkpoint", kw["xm_checkpoint"], "--xm-main", kw["xm_main"],
            "--xm-box-grabcut", kw["xm_box"], "--base-det", kw["base_det"], "--xm-refseg", *kw["xm_refseg"], "--x2-refseg", *kw["x2_refseg"],
            "--lb-xm", kw["lb_xm"], "--lb-base", kw["lb_base"], "--lb-x2", kw["lb_x2"], "--out-dir", str(out)]
    assert GX.main(args) == 2 and "G-XM" in capsys.readouterr().err  # 不给门名：拒绝
    assert GX.main(["--gate", XD.GATE, *args]) == 0
    doc = json.loads((out / "gate_xm.json").read_text(encoding="utf-8"))
    assert doc["decision_split"]["n_pairs"] == len(pairs) and doc["decision_split"]["splits_sha256"] == XD.file_sha256(sp)
    assert doc["rehearsal"] is True and {"decision_n_images_is_500", "decision_splits_is_repo_config"} <= set(doc["rehearsal_reasons"])


def test_end_to_end_on_real_runner_leaves(fake_pairs, tmp_path, monkeypatch):
    """DEV-0080 的运行器（src/xm/runner.run_split + tests/_xm_run_fakes 的脚本化假适配器）真写出的判定集叶目录：gate_xm 读得懂
    （行、身份、提示版本与冻结、节点）、两个系统能配对。两张图，所以只能彩排。"""
    import src.runlog as RL
    from src.e1.prompts import PROMPTS_PATH, load_prompt_config
    from src.xm import prompts as XMP
    from src.xm import runner as XR
    from tests import _xm_run_fakes as RF

    monkeypatch.setattr(RL, "git_info", lambda root: {"commit": "test", "dirty": None, "diff_shortstat": None, "n_untracked": None})
    inp = make_inputs(tmp_path / "in", n_images=1)
    root = tmp_path / "data"
    pairs = XR.pairs_from_json(RF.write_fixture(root, split=XD.XM_DECISION))
    model = {"key": "qwen3.5-9b", "family": "qwen3.5", "repo": "x", "revision": "r", "adapter_model_key": "fake", "tiny": False}
    unseal = {"gate": XD.GATE, "decision": XD.SEAL_DECISION}
    ck = {"manifest_sha256": inp["cand"], "training": {"config_sha256": XM_SHA, "seed": 0}, "xm": {"prompts_version": XMP.PROMPT_VERSION}}
    out = tmp_path / "0001_xm_decision"
    for system in ("xm", "base"):
        ad = RF.ScriptedXMAdapter(RF.gt_answers(pairs, root), system=system)
        XR.run_split(ad, pairs, out, system=system, split=XD.XM_DECISION, model=model, image_policy={"tag": "test"},
                     checkpoint=ck if system == "xm" else None, prompts=load_prompt_config(PROMPTS_PATH), data_root=root, tiny=True,
                     list_batch_size=2, resume=False, unseal=unseal, log=lambda m: None)
    leaf = out / "qwen3.5-9b" / "xm"
    fake_pairs["pairs"] = pairs
    kw = dict(inp["kw"], xm_main=str(leaf / GX.XM_MAIN / XD.XM_DECISION), xm_box=str(leaf / GX.XM_BOX / XD.XM_DECISION),
              base_det=str(leaf / GX.BASE_DET / XD.XM_DECISION))
    doc = GX.evaluate(gate=XD.GATE, data_root="unused", **kw)
    assert doc["decision_split"]["n_pairs"] == len(pairs) and doc["criteria"]["1"]["n_images"] == 2
    assert doc["prompt_versions"] == {k: "xm-prompts-v1" for k in ("checkpoint", "xm_main", "xm_box_grabcut", "base_det")}
    for k in ("prompt_version_matches", "decision_prompts_frozen", "decision_same_node", "decision_unsealed_with_G-XM", "no_limit"):
        assert doc["checks"][k] is True, k
    assert doc["rehearsal"] is True and "decision_n_images_is_500" in doc["rehearsal_reasons"]
    assert doc["criteria"]["2"]["candidate"] == doc["criteria"]["2"]["control"] == 1.0  # 两个假适配器都答真值框
    assert doc["criteria"]["1"]["statistic"] == "f1_rexomni" and doc["criteria"]["1"]["bootstrap"]["n_resamples"] == 10_000
