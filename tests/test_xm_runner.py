"""DEV-0080：X-M 的运行器（src/xm/runner.py、src/xm/gate.py、scripts/run_xm.py）。

- 一个划分的全流程（假适配器，按提示给确定的回答）：调用 1 → NMS → 调用 2 → 逐实例贴边；各设置的叶目录与读数；检测在 xm_* 之间相同；
  样条逐条失败退回椭圆；调用 1 失败的对；crowd 上的框忽略；基座按 E1 的 boxes 解析；记账；续跑逐字节相同；进程池逐位相同；
- ReasonSeg：对 → 样本、忽略区域的计分；
- 密封与 gate（进程内调 parse_args，另有一条子进程核对退出码 2、什么都不写）；
- 命令行 tiny 端到端（真的小模型）：基座两个设置、X-M 四个设置（X-2 的 tiny 检查点改装成 X-M 的）、ReasonSeg 的 --system x2；
- X-2 的代码、配置不动（与 git HEAD 逐字节相同）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.e1.prompts import PROMPTS_PATH, load_prompt_config
from src.executors.raster import mask_to_rle, rle_to_mask
from src.xm import data as XD
from src.xm import gate as XG
from src.xm import metrics as XMM
from src.xm import prompts as XMP
from src.xm import runner as XR
from tests import _xm_run_fakes as F
from tests.test_x2_runner import x2_ckpt  # noqa: F401 —— tiny 的 X-2 检查点（pytest 的 fixture）

REPO = Path(__file__).resolve().parents[1]
RUN_XM = REPO / "scripts" / "run_xm.py"
QWEN = "qwen3.5-0.8b"
PROMPTS = load_prompt_config(PROMPTS_PATH)
MODEL = {"key": "fake", "family": "qwen3.5", "repo": "x", "revision": "r", "adapter_model_key": "fake", "tiny": True}
W, H = F.W, F.H


def _load_run_xm():
    spec = importlib.util.spec_from_file_location("_run_xm_for_tests", RUN_XM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RX = _load_run_xm()


def run(args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(RUN_XM), *map(str, args)], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture(scope="module")
def fast_meta():
    """meta.json 的 git 信息在这个 worktree 里很慢（删掉的 experiments/ 让 git diff 要十几秒）：进程内的测试换成常量。"""
    import src.runlog as RL

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RL, "git_info", lambda root: {"commit": "test", "dirty": None, "diff_shortstat": None, "n_untracked": None})
        yield


def _pairs(root: Path, split: str = "xm_tune") -> list[XD.XMPair]:
    pairs = XR.pairs_from_json(F.write_fixture(root, split=split))
    a_disk, a_sq, b_disk = pairs
    short = dataclasses.replace(a_disk, id=a_disk.id + "/short", category_id=3, category="red disk short")
    junk = dataclasses.replace(a_sq, id=a_sq.id + "/junk", category_id=4, category="blue square junk")
    return [a_disk, a_sq, b_disk, short, junk]


CROWD_BOX = [89.0, 24.0, 111.0, 50.0]  # 图 B 的 crowd 区域（3 个小圆盘）的外接框


def _run(tmp: Path, system: str, out_name: str = "0001_t", *, tool_workers: int = 1, resume: bool = False, pairs=None, ad=None):
    root = tmp / "data"
    pairs = pairs or _pairs(root)
    ad = ad or F.ScriptedXMAdapter(F.gt_answers(pairs, root), system=system, duplicate_first=True, extra_box=CROWD_BOX)
    summ = XR.run_split(ad, pairs, tmp / out_name, system=system, split="xm_tune", model=MODEL, image_policy={"tag": "test"},
                        checkpoint={"manifest_sha256": "abc"} if system == "xm" else None, prompts=PROMPTS, data_root=root, tiny=True,
                        list_batch_size=2, resume=resume, tool_workers=tool_workers)
    return pairs, ad, summ, tmp / out_name / "fake" / "xm"


@pytest.fixture(scope="module")
def xm_run(tmp_path_factory, fast_meta):
    return _run(tmp_path_factory.mktemp("xmrun"), "xm")


def test_trace_and_calls(xm_run):
    pairs, ad, summ, base = xm_run
    tr = {t["id"]: t for t in jsonl(base / "xm_trace" / "xm_tune" / "trace.jsonl")}
    a_disk, a_sq, b_disk, short, junk = pairs
    for p in (a_disk, a_sq, b_disk, short):
        t = tr[p.id]
        n = len(p.instances)
        assert t["nms"] == {"n_in": n + 2, "kept": list(range(n)) + [n + 1]}  # 末尾重复的第 1 个框被 NMS 去掉，crowd 上的框留下
        assert len(t["boxes"]) == n + 1 and t["boxes_q"] == [XMP.XP.box_q(b, W, H) for b in t["boxes"]]
        assert t["trace"]["call"]["failure"] is None and len(t["instances"]) == n + 1
    assert tr[short.id]["trace"]["failures"][0] == "wrong_count" and tr[short.id]["trace"]["splines"][0] is None
    assert tr[junk.id]["list"]["call"]["failure"] == "malformed_json" and tr[junk.id]["boxes"] == [] and tr[junk.id]["trace"] is None
    lists = [c for c in ad.calls if c["prompt"].startswith("Find every instance")]
    traces = [c for c in ad.calls if c["prompt"].startswith("Each numbered box")]
    assert len(lists) == 5 and len(traces) == 4 and all(c["enable_thinking"] is False and c["n_images"] == 1 for c in ad.calls)
    assert all(c["max_new_tokens"] == XMP.max_new_tokens_list() and c["batch"] <= 2 for c in lists)
    # 调用 2：组内（tiny 的组 = 2 个对）按框数升序分批（批大小 2），生成上限取批内各对 64 + 200 × 框数 的最大值
    ks = [len(tr[p.id]["boxes"]) for p in (a_disk, a_sq, b_disk, short)]  # 3、2、2、3
    assert [c["prompt"].split("(")[1].split(" ")[0] for c in traces] == ["2", "3", "2", "3"]  # 每组内框数少的先（"(k listed)"）
    want = [XMP.max_new_tokens_trace(max(ks[0:2]))] * 2 + [XMP.max_new_tokens_trace(max(ks[2:4]))] * 2
    assert [c["max_new_tokens"] for c in traces] == want


def test_settings_outputs_and_metrics(xm_run):
    pairs, _, summ, base = xm_run
    a_disk, a_sq, b_disk, short, junk = pairs
    tr = {t["id"]: t for t in jsonl(base / "xm_trace" / "xm_tune" / "trace.jsonl")}
    recs = {s: {r["id"]: r for r in jsonl(base / s / "xm_tune" / "predictions.jsonl")} for s in XR.settings_of("xm")}
    mets = {s: json.loads((base / s / "xm_tune" / "metrics.json").read_text()) for s in XR.settings_of("xm")}
    assert set(summ["settings"]) == set(XR.settings_of("xm"))
    for s in XR.settings_of("xm"):
        leaf = base / s / "xm_tune"
        assert all((leaf / f).exists() for f in ("predictions.jsonl", "metrics.json", "compute.json", "config.yaml", "meta.json", "README.md"))
        assert mets[s]["complete"] is True and mets[s]["det"] == mets["xm_spline_trimap"]["det"]  # 检测：四个设置相同
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["xm"]["setting"] == s and cfg["metrics"]["version"] == XMM.METRICS_VERSION and cfg["prompts"]["version"] == XMP.PROMPT_VERSION
        comp = json.loads((leaf / "compute.json").read_text())
        assert comp["n_calls"] == sum(r["n_model_calls"] for r in recs[s].values()) and comp["n_samples"] == 5
    # 检测的计数：真值框都配上；图 A 上 crowd 位置的框是 FP，图 B 上它落在 crowd 区域里，忽略
    for p in (a_disk, a_sq, short):
        d = recs["xm_spline_trimap"][p.id]["det"]
        assert d["tp"][0] == len(p.instances) and d["fp"][0] == 1 and d["fn"][0] == 0 and d["ignored"][0] == 0
    d = recs["xm_spline_trimap"][b_disk.id]["det"]
    assert d["tp"][0] == 1 and d["fp"][0] == 0 and d["ignored"][0] == 1
    j = recs["xm_spline_trimap"][junk.id]
    assert j["failure"] == "malformed_json" and j["instances"] == [] and j["det"]["fn"][0] == len(junk.instances) and j["n_model_calls"] == 1
    # 各设置交的 mask
    for p in pairs[:4]:
        t = tr[p.id]
        for k, e in enumerate(recs["xm_box_grabcut"][p.id]["instances"]):
            assert e["mask"]["rle"] == t["instances"][k]["box_gc"]["rle"] and e["fallback"] is None
        for k, e in enumerate(recs["xm_ellipse_trimap"][p.id]["instances"]):
            assert e["mask"]["rle"] == t["instances"][k]["ellipse"]["trimap_rle"]
        for k, e in enumerate(recs["xm_spline_trimap"][p.id]["instances"]):
            inst = t["instances"][k]
            want = inst["box_gc"]["rle"] if inst["spline"] is None else inst["spline"]["trimap_rle"]  # D168：退回框 → 缺省 GrabCut
            assert e["mask"]["rle"] == want and e["fallback"] == (None if inst["spline"] else "box_grabcut")
            assert e["mask"]["area"] == int(rle_to_mask(want).sum())
        assert recs["xm_spline_trimap"][p.id]["n_model_calls"] == 2 and recs["xm_box_grabcut"][p.id]["n_model_calls"] == 1
    sr = recs["xm_spline_trimap"][short.id]["instances"][0]
    assert sr["spline_failure"] == "wrong_count" and sr["fallback"] == "box_grabcut"
    assert sr["mask"]["rle"] == recs["xm_box_grabcut"][short.id]["instances"][0]["mask"]["rle"]  # 失败的实例与对照相同（判定第 1 条中性）
    f = mets["xm_spline_trimap"]["failures"]
    assert f["spline_failure"]["wrong_count"] == 1 and f["fallback"]["box_grabcut"] == 1 and f["list_failure"]["malformed_json"] == 1
    assert yaml.safe_load((base / "xm_spline_trimap" / "xm_tune" / "config.yaml").read_text())["post"]["spline_failure_fallback"] == "box_grabcut"
    assert f["nms_suppressed"] == 4 and "spline_failure" not in mets["xm_box_grabcut"]["failures"]
    # 读数 = src/xm/metrics.readout(predictions)（只由 predictions.jsonl 算）
    rows = list(recs["xm_spline_trimap"].values())
    again = XMM.readout(rows, "seg")
    assert again["f1_rexomni"]["value"] == mets["xm_spline_trimap"]["seg"]["f1_rexomni"]["value"]
    assert mets["xm_spline_trimap"]["seg"]["f1_rexomni"]["ci"]["n_resamples"] == 1000
    sm = json.loads((base / "xm_summary" / "xm_tune__xm_trace.json").read_text())
    assert sm["settings"]["xm_spline_trimap"]["det"]["f1_rexomni"] == mets["xm_spline_trimap"]["det"]["f1_rexomni"]["value"]


def test_ellipse_fallback_option(tmp_path, fast_meta, monkeypatch):
    """可选的退回（框内椭圆）：常量换成 "ellipse" 后，失败的实例交椭圆的 trimap / 光栅化，运行身份记它。"""
    from src.xm import post as PO

    monkeypatch.setattr(PO, "SPLINE_FAILURE_FALLBACK", "ellipse")
    pairs = _pairs(tmp_path / "data")
    short = pairs[3]
    _, _, _, base = _run(tmp_path, "xm", pairs=[short])
    t = jsonl(base / "xm_trace" / "xm_tune" / "trace.jsonl")[0]
    assert t["trace"]["failures"][0] == "wrong_count"
    for s, key in (("xm_spline_trimap", "trimap_rle"), ("xm_spline_raster", "raster_rle")):
        e = jsonl(base / s / "xm_tune" / "predictions.jsonl")[0]["instances"]
        assert e[0]["fallback"] == "ellipse" and e[0]["mask"]["rle"] == t["instances"][0]["ellipse"][key]
        assert all(x["fallback"] is None for x in e[1:])
        assert yaml.safe_load((base / s / "xm_tune" / "config.yaml").read_text())["post"]["spline_failure_fallback"] == "ellipse"
    assert json.loads((base / "xm_spline_trimap" / "xm_tune" / "metrics.json").read_text())["failures"]["fallback"]["ellipse"] == 1


def test_resume_pool_and_base(xm_run, tmp_path, fast_meta):
    pairs, _, _, base = xm_run
    out = base.parents[1]
    before = {s: (base / s / "xm_tune" / "predictions.jsonl").read_bytes() for s in XR.settings_of("xm")}
    _run(out.parent, "xm", out.name, resume=True, pairs=pairs)
    assert all((base / s / "xm_tune" / "predictions.jsonl").read_bytes() == b for s, b in before.items())
    with pytest.raises(FileExistsError):
        _run(out.parent, "xm", out.name, resume=False, pairs=pairs)
    # 进程池（2 个）：轨迹逐字节相同
    _run(out.parent, "xm", "0002_pool", tool_workers=2, pairs=pairs)
    assert (out.parent / "0002_pool" / "fake" / "xm" / "xm_trace" / "xm_tune" / "trace.jsonl").read_bytes() == \
        (base / "xm_trace" / "xm_tune" / "trace.jsonl").read_bytes()
    # 基座：E1 的 boxes 解析（Qwen 的原生格式）给出同样的框，检测计数与 X-M 的相同
    _, ad, summ, bb = _run(out.parent, "base", "0003_base", pairs=pairs)
    assert set(summ["settings"]) == {"base_det", "base_box_grabcut"}
    tb = {t["id"]: t for t in jsonl(bb / "base_trace" / "xm_tune" / "trace.jsonl")}
    tx = {t["id"]: t for t in jsonl(base / "xm_trace" / "xm_tune" / "trace.jsonl")}
    for p in pairs:
        assert tb[p.id]["boxes_q"] == tx[p.id]["boxes_q"] and tb[p.id]["list"]["parser"] == XR.BASE_PARSER
        assert set(tb[p.id]["instances"][0]) == {"box_gc"} if tb[p.id]["instances"] else True
    assert not [c for c in ad.calls if c["prompt"].startswith("Each numbered box")]  # 基座不调调用 2
    bd = {r["id"]: r for r in jsonl(bb / "base_det" / "xm_tune" / "predictions.jsonl")}
    xd = {r["id"]: r for r in jsonl(base / "xm_box_grabcut" / "xm_tune" / "predictions.jsonl")}
    assert all(bd[i]["det"] == xd[i]["det"] and bd[i]["seg"] is None for i in bd)
    m = json.loads((bb / "base_det" / "xm_tune" / "metrics.json").read_text())
    assert m["seg"] is None and m["det"]["f1_rexomni"]["value"] is not None
    bg = {r["id"]: r for r in jsonl(bb / "base_box_grabcut" / "xm_tune" / "predictions.jsonl")}
    assert all(bg[i]["seg"] == xd[i]["seg"] for i in bg)  # 同样的框 → 同样的缺省 GrabCut
    with pytest.raises(XR.XMRunError):
        XR.system_of(["xm_spline_trimap", "base_det"])


def test_not_exhaustive_and_reasonseg(tmp_path):
    pairs = XR.pairs_from_json(F.write_fixture(tmp_path, split="reasonseg_val"))
    p = dataclasses.replace(pairs[1], meta={XR.NOT_EXHAUSTIVE_KEY: [2, 7]})
    assert XR.not_exhaustive(p) and not XR.not_exhaustive(pairs[0])
    t = {"boxes": [[0.0, 0.0, 5.0, 5.0]]}
    assert XR.det_eval(p, t)["fp"][0] == 0 and XR.det_eval(pairs[1], t)["fp"][0] == 1
    assert XR.pairs_from_json(_dump(tmp_path / "p.json", [p]))[0].meta == p.meta  # meta 往返
    # ReasonSeg：一个实例 → 样本；忽略区域逐像素不计
    rs = dataclasses.replace(pairs[2], category="the thing that is red")
    s = XR.reasonseg_samples([rs], tmp_path)[0]
    assert s.fields["expression"] == "the thing that is red" and s.gt["box"] == list(rs.instances[0].box_xyxy)
    empty = dataclasses.replace(rs, id=rs.id + "/e", instances=())
    assert XR.reasonseg_samples([empty], tmp_path)[0].gt["box"] == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(XR.XMRunError):
        XR.reasonseg_samples([pairs[0]], tmp_path)  # 两个实例
    gt = rle_to_mask(rs.instances[0].rle)
    pred = gt.copy()
    pred[:, :20] = True  # 多出一条
    ig = np.zeros_like(gt)
    ig[:, :20] = True
    rs_ig = dataclasses.replace(rs, meta={XR.IGNORE_KEY: mask_to_rle(ig)})
    rec = {"id": rs.id, "answer": {"mask": {"rle": mask_to_rle(pred)}}, "score": {"inter": int((pred & gt).sum()), "union": int((pred | gt).sum())}}
    out = XR.reasonseg_readout([rec], [rs_ig])
    assert out["without_ignore"]["cIoU"] < 1.0 and out["with_ignore"]["cIoU"] == pytest.approx(1.0) and out["n_with_ignore_region"] == 1


def _dump(path: Path, pairs) -> Path:
    path.write_text(json.dumps([XR.pair_to_json(p) for p in pairs]), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 密封与 gate
# ---------------------------------------------------------------------------


def _gate_json(path: Path, manifest_sha: str, **edit) -> Path:
    doc = {"verdict": XD.XM_GATE_PASS, "rehearsal": False, "checks": {"n_is_500": True, "not_tiny": True}, "candidate": {"manifest_sha256": manifest_sha}}
    doc.update(edit)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def _fake_ckpt(tmp: Path) -> tuple[Path, str]:
    ck = tmp / "ck" / "ckpt" / "final" / "model"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "manifest.json").write_text('{"x": 1}', encoding="utf-8")
    return ck, hashlib.sha256(b'{"x": 1}').hexdigest()


def test_guards_in_process(tmp_path, capsys, monkeypatch):
    pj = F.write_fixture(tmp_path / "d")
    ck, sha = _fake_ckpt(tmp_path)
    good = _gate_json(tmp_path / "g_ok.json", sha)
    tiny = ["--tiny", "--model", QWEN, "--pairs-json", pj]

    def refused(args, msg):
        out = tmp_path / "0009_refused"
        with pytest.raises(SystemExit) as e:
            RX.parse_args([*map(str, args), "--out", str(out)])
        assert e.value.code == 2 and msg in capsys.readouterr().err and not out.exists(), (args, msg)

    refused([*tiny, "--split", "xm_decision"], "G-XM")
    refused([*tiny, "--split", "xm_decision", "--gate", "G-EX"], "G-XM")
    refused([*tiny, "--split", "xm_tune", "--gate", "G-XM"], "只配密封")
    refused([*tiny, "--split", "xm_final"], "--xm-gate")
    refused([*tiny, "--split", "lvis_sub", "--checkpoint", ck, "--xm-gate", tmp_path / "nope.json"], "读不了")
    for name, edit, msg in [("v", {"verdict": "判定通过"}, "结论"), ("r", {"rehearsal": True}, "彩排"), ("c", {"checks": {"a": True, "b": False}}, "判定点"),
                            ("e", {"checks": {}}, "判定点"), ("m", {"candidate": {"manifest_sha256": "0" * 64}}, "不是本次的"),
                            ("k", {"candidate": {}}, "不是本次的")]:
        refused([*tiny, "--split", "xm_final", "--checkpoint", ck, "--xm-gate", _gate_json(tmp_path / f"g_{name}.json", sha, **edit)], msg)
    refused([*tiny, "--split", "xm_tune", "--xm-gate", good], "只配")
    refused([*tiny, "--split", "reasonseg_val", "--checkpoint", ck], "--system")
    refused([*tiny, "--split", "reasonseg_val", "--checkpoint", ck, "--system", "xm"], "--xm-gate")
    refused([*tiny, "--split", "reasonseg_val", "--checkpoint", ck, "--system", "x2", "--xm-gate", good], "只配")
    refused([*tiny, "--split", "xm_tune", "--system", "x2"], "只配")
    refused([*tiny, "--split", "xm_tune", "--setting", "xm_spline_trimap"], "--checkpoint")
    refused([*tiny, "--split", "xm_tune", "--setting", "base_det", "--checkpoint", ck], "不配检查点")
    refused([*tiny, "--split", "xm_tune", "--setting", "base_det", "xm_box_grabcut"], "一次只跑一个系统")
    refused([*tiny, "--split", "xm_tune", "--setting", "magic"], "未知的设置")
    refused(["--model", QWEN, "--split", "xm_tune", "--pairs-json", pj, "--data-root", tmp_path], "只配 --tiny")
    # 冻结的提示（D168：xm-prompts-v1 已冻结）：非 tiny 的密封 / 报告划分放行；FROZEN = False 时拒绝（tiny 不查）
    assert XMP.FROZEN is True
    real = ["--model", QWEN, "--data-root", str(tmp_path)]
    for split, extra in (("xm_decision", ["--gate", "G-XM"]), ("xm_final", ["--xm-gate", str(good)]), ("lvis_sub", ["--xm-gate", str(good)])):
        a = RX.parse_args([*real, "--split", split, *extra, "--out", str(tmp_path / "0017_frozen")])
        assert a.split == split and not (tmp_path / "0017_frozen").exists()
    monkeypatch.setattr(XMP, "FROZEN", False)
    refused(["--model", QWEN, "--split", "xm_decision", "--gate", "G-XM", "--data-root", tmp_path], "冻结")
    refused(["--model", QWEN, "--split", "xm_final", "--xm-gate", good, "--data-root", tmp_path], "冻结")
    RX.parse_args([*real, "--split", "xm_tune", "--out", str(tmp_path / "0018_tune")])  # 调参集不要求冻结
    RX.parse_args([*map(str, tiny), "--split", "xm_decision", "--gate", "G-XM", "--out", str(tmp_path / "0019_tiny")])  # tiny 不查
    monkeypatch.setattr(XMP, "FROZEN", True)
    with pytest.raises(SystemExit):
        RX.parse_args([*map(str, tiny), "--split", "xm_tune", "--out", str(tmp_path / "xm_no_run_id")])
    assert "运行编号" in capsys.readouterr().err
    # 合格的放行：基座的设置不比检查点（记 checkpoint_compared = false）；X-M 的比
    a = RX.parse_args([*map(str, tiny), "--split", "xm_final", "--xm-gate", str(good), "--out", str(tmp_path / "0010_ok")])
    g = RX.check_gates(a, "base")
    assert g["xm_gate"]["identity"]["checkpoint_compared"] is False and g["xm_gate"]["identity"]["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest()
    a = RX.parse_args([*map(str, tiny), "--split", "xm_final", "--checkpoint", str(ck), "--xm-gate", str(good), "--out", str(tmp_path / "0011_ok")])
    assert RX.check_gates(a, "xm")["xm_gate"]["identity"]["checkpoint_compared"] is True
    a = RX.parse_args([*map(str, tiny), "--split", "xm_decision", "--gate", "G-XM", "--out", str(tmp_path / "0012_ok")])
    assert XG.seal_identity(RX.check_gates(a, "base")["unseal"]) == {"gate": "G-XM", "decision": XG.SEAL_DECISION}
    assert XG.GATE_PASS == XD.XM_GATE_PASS == "通过"


def test_guard_subprocess_exit_code(tmp_path):
    pj = F.write_fixture(tmp_path / "d")
    out = tmp_path / "0013_sealed"
    p = run(["--tiny", "--model", QWEN, "--pairs-json", pj, "--split", "xm_decision", "--out", out], tmp_path, ok=False)
    assert p.returncode == 2 and "G-XM" in p.stderr and not out.exists()


# ---------------------------------------------------------------------------
# 命令行 tiny 端到端
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    """DEV-0079 的假数据根（COCO / LVIS / ReasonSeg 的官方格式）与 tiny 的划分清单：命令行经 load_pairs 读，与真运行同一条路径。"""
    from tests._xm_fakes import build_fake_root

    d = tmp_path_factory.mktemp("xmroot")
    build_fake_root(d / "root")
    sp = d / "xm_splits.json"
    sp.write_text(XD.dump_config(XD.build_splits_doc(d / "root", tiny=True)), encoding="utf-8")
    return d / "root", sp


def test_cli_tiny_base_xm_final_with_gate(fake_root, tmp_path):
    root, sp = fake_root
    ck, sha = _fake_ckpt(tmp_path)
    good = _gate_json(tmp_path / "g.json", sha)
    out = tmp_path / "0014_base_final"
    p = run(["--tiny", "--model", QWEN, "--data-root", root, "--splits-file", sp, "--split", "xm_final", "--xm-gate", good, "--out", out], tmp_path)
    assert "base_det" in p.stdout
    gp = XG.check_xm_gate(good, checkpoint=None)["pass"]
    want = [q.id for q in XD.load_pairs("xm_final", root, xm_gate=gp, splits_path=sp)]
    m = out / QWEN
    for s in ("base_det", "base_box_grabcut"):
        leaf = m / "xm" / s / "xm_final"
        recs = jsonl(leaf / "predictions.jsonl")
        assert [r["id"] for r in recs] == want and json.loads((leaf / "metrics.json").read_text())["complete"] is True
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["xm"]["xm_gate"]["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest() and cfg["checkpoint"] is None
        assert cfg["xm"]["xm_gate"]["checkpoint_compared"] is False
        assert cfg["image_policy"]["tag"] == "default" and cfg["generation"]["list_parser"] == XR.BASE_PARSER
    assert len(jsonl(m / XG.UNLOCK_LOG)) == 1 and not (m / "xm" / "xm_spline_trimap").exists()
    with pytest.raises(XD.XMGateError):  # load_pairs 自己也要 gate（DEV-0079）
        XD.load_pairs("xm_final", root, splits_path=sp)


@pytest.fixture(scope="module")
def xm_ckpt(x2_ckpt, tmp_path_factory):  # noqa: F811
    """X-2 的 tiny 检查点改装成 X-M 的：复制训练叶目录，config.yaml 加 xm 一节（DEV-0081 的约定：prompts.version、n）。"""
    src_leaf = Path(x2_ckpt).parents[2]
    leaf = tmp_path_factory.mktemp("xmck") / "w2c" / QWEN / "seed0"
    shutil.copytree(src_leaf, leaf)
    cfg = yaml.safe_load((leaf / "config.yaml").read_text())
    cfg[XR.XM_CHECKPOINT_SECTION] = {"prompts": {"version": XMP.PROMPT_VERSION}, "n": XMP.SPLINE_N}
    (leaf / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return leaf / "ckpt" / "final" / "model"


def test_cli_tiny_xm_decision(xm_ckpt, x2_ckpt, tmp_path):  # noqa: F811
    with pytest.raises(XR.XMRunError, match="没有 xm"):
        XR.checkpoint_xm_params(x2_ckpt)
    pj = F.write_fixture(tmp_path / "d", split="xm_decision")
    out = tmp_path / "0015_xm_decision"
    p = run(["--tiny", "--model", QWEN, "--checkpoint", xm_ckpt, "--pairs-json", pj, "--split", "xm_decision", "--gate", "G-XM",
             "--out", out, "--tool-workers", "2"], tmp_path)
    assert "xm_spline_trimap" in p.stdout
    m = out / QWEN
    ck_sha = hashlib.sha256((Path(xm_ckpt) / "manifest.json").read_bytes()).hexdigest()
    for s in XR.settings_of("xm"):
        leaf = m / "xm" / s / "xm_decision"
        cfg = yaml.safe_load((leaf / "config.yaml").read_text())
        assert cfg["checkpoint"]["manifest_sha256"] == ck_sha and cfg["xm"]["unseal"] == {"gate": "G-XM", "decision": XG.SEAL_DECISION}
        assert cfg["checkpoint"]["xm"]["prompts_version"] == XMP.PROMPT_VERSION and len(jsonl(leaf / "predictions.jsonl")) == 3
    assert len(jsonl(m / XG.UNSEAL_LOG)) == 1 and (m / "xm_merge_check.json").exists()
    # X-2 的检查点（叶目录 config.yaml 没有 xm 一节）：拒绝
    p = run(["--tiny", "--model", QWEN, "--checkpoint", x2_ckpt, "--pairs-json", pj, "--split", "xm_decision", "--gate", "G-XM", "--out", out],
            tmp_path, ok=False)
    assert p.returncode == 2


def test_cli_tiny_reasonseg_x2(x2_ckpt, fake_root, tmp_path):  # noqa: F811
    root, sp = fake_root
    rs = XD.load_pairs("reasonseg_val", root, splits_path=sp)
    assert any((q.meta or {}).get(XR.IGNORE_KEY) for q in rs)  # 假数据里 rs_b 有 ignore 区域
    out = tmp_path / "0016_reasonseg_x2"
    run(["--tiny", "--model", QWEN, "--checkpoint", x2_ckpt, "--data-root", root, "--splits-file", sp, "--split", "reasonseg_val", "--system", "x2",
         "--out", out, "--candidates-module", "tests._w2_fakes", "--render-module", "tests._w2_fakes"], tmp_path)
    m = out / QWEN
    leaf = m / "refseg" / "x2_spline_trimap" / "reasonseg_val"
    recs = jsonl(leaf / "predictions.jsonl")
    assert [r["id"] for r in recs] == [q.id for q in rs]
    assert not (m / "refseg" / "x2_select" / "reasonseg_val").exists()  # 只导出 x2_spline_trimap
    rd = json.loads((m / "reasonseg_readout" / "reasonseg_val.json").read_text())
    assert rd["system"] == "x2" and rd["settings"]["x2_spline_trimap"]["with_ignore"]["n"] == len(rs)
    assert rd["settings"]["x2_spline_trimap"]["without_ignore"] == XMM.refseg_readout(recs)
    assert rd["settings"]["x2_spline_trimap"]["n_with_ignore_region"] >= 1
    assert not (m / XG.UNLOCK_LOG).exists()


def test_x2_code_and_configs_unchanged():
    """X-2 的代码与配置与 git HEAD 逐字节相同（X-2 的输出由 tests/test_x2_*.py 钉住）。"""
    paths = ["src/explore", "scripts/run_x2.py", "scripts/gate_explore.py", "configs/x2_run.json", "configs/x2_train.json", "configs/x3_train.json",
             "configs/w2.json", "src/w2", "src/e1"]
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=REPO, capture_output=True)
    if r.returncode not in (0, 1):
        pytest.skip("没有 git")
    assert r.returncode == 0, "X-2 的代码或配置被改了"
