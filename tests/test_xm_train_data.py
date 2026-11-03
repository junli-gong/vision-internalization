"""DEV-0081：X-M 的 COCO 训练样本（src/xm/train_data.py、scripts/build_xm_data.py），输入是 DEV-0079 的训练目标（src/xm/targets.py）。

- 每条每实例样条样本至多 12 个框，按阅读顺序切开；提示只列这一段的框，目标是它们的样条；列框的目标是全部框；两者与 DEV-0079 记的
  box_text / spline_text 逐字相同；
- 抽对：带 crowd 的对整对剔除（D167-4）、样条不可用的对只进列框流；其余按 hash 排序依次取，直到两条流都够 ceil(steps × 每步条数 × 1.2)；
  缺省 2,000 × 6 × 1.2 = 14,400；
- 生成：manifest 逐字节可复现、续跑与一次跑完逐字节相同；读取的核对；只读训练集；
- 端到端：DEV-0079 的假数据根 → load_pairs("train") → targets.build → build_xm_data → load_xm_data；
- 命令行：--out 不带运行编号时拒绝。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest

from src.data.subsets import hash_rank
from src.xm import data as XD
from src.xm import exclusions as XE
from src.xm import prompts as XMP
from src.xm import targets as XT
from src.xm import train_data as XTD
from tests import _xm_train_fakes as XF
from tests._xm_fakes import build_fake_root

REPO = Path(__file__).resolve().parents[1]
CFG_DOC = json.loads((REPO / "configs" / "xm_train.json").read_text(encoding="utf-8"))
BUILD = REPO / "scripts" / "build_xm_data.py"


def _load_build():
    spec = importlib.util.spec_from_file_location("_build_xm_data", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BX = _load_build()
PARAMS = XTD.params_from_config(CFG_DOC)


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _outs(run: Path, kind: str) -> list[dict]:
    return [r for f in sorted((run / XTD.DATA_DIR).glob(f"{kind}_*.jsonl")) for r in jsonl(f)]


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------


def test_fake_targets_are_dev0079_rows():
    params = XT.check_params(XT.DEFAULT_PARAMS)
    rows = XF.target_rows()
    assert all(XT.row_problems(r, params) == [] and XTD.train_problems(r) == [] for r in rows)
    assert [r["n_kept"] for r in rows if r["truncated"]] == [30]


def test_chunk_ranges():
    assert XTD.chunk_ranges(30, 12) == [(0, 12), (12, 24), (24, 30)]
    assert XTD.chunk_ranges(12, 12) == [(0, 12)]
    assert XTD.chunk_ranges(13, 12) == [(0, 12), (12, 13)]
    assert XTD.chunk_ranges(1, 12) == [(0, 1)]
    with pytest.raises(ValueError):
        XTD.chunk_ranges(0, 12)


def test_rows_prompts_and_targets():
    rows = XF.target_rows()
    t = next(r for r in rows if r["n_kept"] == 25)
    d = XTD.det_row(t)
    assert d["id"] == t["id"] + "#det" and d["pair_id"] == t["id"] and d["kind"] == "coco_det" and d["n_boxes"] == 25
    assert d["target_text"] == XMP.format_boxes(t["boxes_q"]) == t["box_text"] and json.loads(d["target_text"]) == t["boxes_q"]
    assert (d["width"], d["height"]) == tuple(t["image_size"])
    ins = XTD.inst_rows(t, 12)
    assert [len(r["boxes"]) for r in ins] == [12, 12, 1] and [r["box_start"] for r in ins] == [0, 12, 24]
    assert [r["id"] for r in ins] == [f"{t['id']}#inst{k}" for k in range(3)] and all(r["n_chunks"] == 3 for r in ins)
    assert XMP.format_splines(t["ctrl_q"]) == t["spline_text"]  # 与 DEV-0079 的写法（X-2 的 spline_text 套一层列表）逐字相同
    for r in ins:
        s, e = r["box_start"], r["box_start"] + len(r["boxes"])
        assert r["boxes"] == t["boxes_q"][s:e] and r["splines"] == t["ctrl_q"][s:e] and r["fit_iou"] == t["fit_iou"][s:e]
        assert r["target_text"] == XMP.format_splines(t["ctrl_q"][s:e]) == XT.spline_list_text(t["ctrl_q"][s:e])
        assert XT.parse_spline_list(r["target_text"]) == t["ctrl_q"][s:e]
        # 提示只列这一段的框：训练时是 call2_prompt(类别, 这一段)，不是整对的全部框（只经公开的函数比，不依赖提示的措辞）
        assert XMP.call2_prompt(r["category"], r["boxes"]) != XMP.call2_prompt(t["category"], t["boxes_q"])
    t30 = next(r for r in rows if r["truncated"])  # 33 个实例截到 30：3 段 12 + 12 + 6
    assert [len(r["boxes"]) for r in XTD.inst_rows(t30, 12)] == [12, 12, 6]
    t12 = next(r for r in rows if r["n_kept"] == 12 and r["spline_ok"])
    assert [len(r["boxes"]) for r in XTD.inst_rows(t12, 12)] == [12]
    bad = next(r for r in rows if not r["spline_ok"])
    with pytest.raises(ValueError, match="只进列框流"):
        XTD.inst_rows(bad, 12)
    assert XTD.det_row(bad)["n_boxes"] == bad["n_kept"]
    with pytest.raises(XTD.XMDataError, match="box_text"):  # 两边的写法不一致就报错
        XTD.det_row({**t, "box_text": t["box_text"].replace(", ", ",")})
    with pytest.raises(XTD.XMDataError, match="spline_text"):
        XTD.inst_rows({**t, "spline_text": t["spline_text"].replace(", ", ",")}, 12)
    assert "traffic light" in XMP.call1_prompt("traffic light")


@pytest.mark.parametrize("edit", [
    lambda r: r.update(id="coco2017/val2017/1/1"),
    lambda r: r.update(split="xm_tune"),
    lambda r: r.update(source="lvis_v1"),
])
def test_train_problems(edit):
    r = dict(XF.target_rows()[0])
    edit(r)
    assert any("只读训练集" in p for p in XTD.train_problems(r))


def test_sizing_from_config():
    assert XTD.sizing(2000, {"coco_det": 6, "coco_inst": 6}, 0.2)["need"] == {"coco_det": 14400, "coco_inst": 14400}
    s = XTD.sizing_from_config(CFG_DOC, tiny=False)
    assert (s["steps"], s["per_step"], s["need"]) == (2000, {"coco_det": 6, "coco_inst": 6}, {"coco_det": 14400, "coco_inst": 14400})
    assert XTD.sizing_from_config(CFG_DOC, tiny=True)["need"] == {"coco_det": 10, "coco_inst": 10}  # ceil(4 × 2 × 1.2)
    assert XTD.sizing_from_config(CFG_DOC, tiny=False, steps=1500)["need"]["coco_det"] == 10800
    assert PARAMS == {"pair_salt": "xm_train_pairs_v1", "margin": 0.2, "inst_max_boxes": 12}
    with pytest.raises(XTD.XMDataError, match="inst_max_boxes"):
        XTD.check_params({"pair_salt": "s", "margin": 0.2, "inst_max_boxes": 31})


def test_select_pairs_drops_crowd_and_follows_hash_order():
    rows = XF.target_rows()
    info = {r["id"]: {"n_kept": r["n_kept"], "spline_ok": r["spline_ok"], "n_crowd": r["n_crowd"]} for r in rows}
    need = {"coco_det": 8, "coco_inst": 12}
    sel, rec = XTD.select_pairs(info, need=need, salt="s1", max_boxes=12, tiny=False)
    crowd = {r["id"] for r in rows if r["n_crowd"]}
    assert len(crowd) == 2 and not sel & crowd and rec["n_crowd_dropped"] == 2 and rec["n_eligible_pairs"] == 18
    ranked = hash_rank([r["id"] for r in rows if not r["n_crowd"]], "xm_train_pairs", "s1")
    got, k = {"coco_det": 0, "coco_inst": 0}, 0
    while got["coco_det"] < 8 or got["coco_inst"] < 12:
        got["coco_det"] += 1
        got["coco_inst"] += math.ceil(info[ranked[k]]["n_kept"] / 12) if info[ranked[k]]["spline_ok"] else 0
        k += 1
    assert sel == set(ranked[:k]) and rec["rows"] == got and rec["satisfied"] and rec["n_selected_pairs"] == k
    assert rec["n_selected_spline_unusable"] == sum(1 for p in sel if not info[p]["spline_ok"])
    other, _ = XTD.select_pairs(info, need=need, salt="s2", max_boxes=12, tiny=False)
    assert other != sel  # 换盐换样本
    with pytest.raises(XTD.XMDataError, match="不够"):
        XTD.select_pairs(info, need={"coco_det": 19, "coco_inst": 1}, salt="s1", max_boxes=12, tiny=False)
    s_all, r_all = XTD.select_pairs(info, need={"coco_det": 19, "coco_inst": 1}, salt="s1", max_boxes=12, tiny=True)
    assert s_all == set(info) - crowd and r_all["satisfied"] is False


# ---------------------------------------------------------------------------
# 生成、续跑、读取
# ---------------------------------------------------------------------------


def _tree(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file() and "logs" not in p.parts}


def test_build_resume_and_load(tmp_path, capsys):
    rows = XF.target_rows()
    tg = XF.write_targets(tmp_path / "targets", rows)
    a = tmp_path / "a"
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(a), "--data-root", str(XF.ROOT)]) == 0
    assert "剔除带 crowd 的 2 对" in capsys.readouterr().out
    man = json.loads((a / XTD.DATA_DIR / XTD.MANIFEST).read_text())
    assert man["sizing"]["need"] == {"coco_det": 10, "coco_inst": 10} and man["selection"]["satisfied"]
    assert man["counts"]["crowd_pairs_dropped"] == 2 and man["selection"]["n_available_pairs"] == 20
    det, inst = _outs(a, "coco_det"), _outs(a, "coco_inst")
    assert len(det) == man["counts"]["coco_det"] >= 10 and len(inst) == man["counts"]["coco_inst"] >= 10
    crowd = {r["id"] for r in rows if r["n_crowd"]}
    no_spline = {r["id"] for r in rows if not r["spline_ok"]}
    assert not {r["pair_id"] for r in det + inst} & crowd  # 带 crowd 的对：两条流都没有
    assert {r["pair_id"] for r in inst} == {r["pair_id"] for r in det} - no_spline  # 样条不可用的对：只进列框流
    assert man["counts"]["pairs_det_only"] == len({r["pair_id"] for r in det} & no_spline)
    assert all(len(r["boxes"]) <= 12 for r in inst)
    assert sum(len(r["boxes"]) for r in inst) == sum(r["n_boxes"] for r in det if r["pair_id"] not in no_spline)  # 切段不丢框
    n_in = len(list((tg / XT.DATA_DIR).glob("targets_*.jsonl")))
    assert len(list((a / XTD.DATA_DIR).glob("*.jsonl"))) == 2 * n_in  # 一个输入分片对应一对输出分片
    # 再生成一次（另一个目录）：除 logs/ 外逐字节相同；续跑：删掉一对输出，--resume 补回同样的字节
    b = tmp_path / "b"
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(b)]) == 0
    assert _tree(a) == _tree(b)
    before = _tree(b)
    for f in (b / XTD.DATA_DIR).glob("*_targets_001.jsonl"):
        f.unlink()
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(b)]) == 2  # 已存在：要 --resume
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(b), "--resume"]) == 0
    assert _tree(b) == before and len(list((b / "logs").glob("run_*.json"))) == 2
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(b), "--resume", "--steps", "9"]) == 2  # 参数不同：拒绝混写
    # 读取
    x = XTD.load_xm_data(a, params=PARAMS, data_root=XF.ROOT)
    assert {k: len(v) for k, v in x.pools.items()} == {"coco_det": len(det), "coco_inst": len(inst)}
    assert all(x.rows[i]["kind"] == k for k, p in x.pools.items() for i in p)
    ident = x.identity()
    assert ident["manifest_sha256"] == hashlib.sha256((a / XTD.DATA_DIR / XTD.MANIFEST).read_bytes()).hexdigest()
    assert ident["selection"]["n_crowd_dropped"] == 2 and str(a) not in json.dumps(ident)
    with pytest.raises(XTD.XMDataError, match="参数"):
        XTD.load_xm_data(a, params={**PARAMS, "inst_max_boxes": 10})
    with pytest.raises(XTD.XMDataError, match="不够"):
        XTD.load_xm_data(a, params=PARAMS, need={"coco_det": 10_000, "coco_inst": 1})
    with pytest.raises(XTD.XMDataError, match="图像不在"):
        XTD.load_xm_data(a, params=PARAMS, data_root=tmp_path)
    # 改一行的目标文本（同时改 manifest 的 sha256）：重写核对不过
    c = tmp_path / "c"
    shutil.copytree(a, c)
    f = sorted((c / XTD.DATA_DIR).glob("coco_inst_*.jsonl"))[0]
    rs = jsonl(f)
    rs[0]["target_text"] = rs[0]["target_text"].replace(", ", ",")
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    m = json.loads((c / XTD.DATA_DIR / XTD.MANIFEST).read_text())
    for e in m["files"]:
        if e["file"] == f.name:
            e["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    (c / XTD.DATA_DIR / XTD.MANIFEST).write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    with pytest.raises(XTD.XMDataError, match="目标文本"):
        XTD.load_xm_data(c, params=PARAMS)
    # 训练目标被改过：sha256 与 DEV-0079 的 manifest 不符
    t0 = sorted((tg / XT.DATA_DIR).glob("targets_*.jsonl"))[0]
    t0.write_text(t0.read_text().replace('"n_crowd": 0', '"n_crowd": 0 ', 1))
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(tmp_path / "d")]) == 2
    assert "sha256" in capsys.readouterr().err


def test_end_to_end_with_dev0079_targets(tmp_path):
    """DEV-0079 的假数据根 → 划分与去重清单 → load_pairs("train") → targets.build → build_xm_data → load_xm_data。"""
    root = tmp_path / "root"
    build_fake_root(root)
    sp, ep = tmp_path / "xm_splits.json", tmp_path / "xm_train_exclusions.json"
    sp.write_text(XD.dump_config(XD.build_splits_doc(root, tiny=True)), encoding="utf-8")
    ep.write_text(XD.dump_config(XE.build_exclusions(root, splits_path=sp, tiny=True, workers=1)), encoding="utf-8")
    pairs = XD.load_pairs("train", root, splits_path=sp, exclusions_path=ep)
    tg = tmp_path / "tg"
    XT.build(pairs, tg, params=XT.DEFAULT_PARAMS, source=XT.source_identity(pairs, splits_path=sp, exclusions_path=ep, train_json_sha256="x"),
             tiny=True, shard_pairs=3, log=lambda m: None)
    trows, _ = XT.load_targets(tg)
    out = tmp_path / "xm"
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(out), "--data-root", str(root)]) == 0
    man = json.loads((out / XTD.DATA_DIR / XTD.MANIFEST).read_text())
    assert man["selection"]["n_available_pairs"] == len(trows) == len(pairs)
    assert man["counts"]["crowd_pairs_dropped"] == sum(1 for r in trows if r["n_crowd"])
    det, inst = _outs(out, "coco_det"), _outs(out, "coco_inst")
    by_id = {r["id"]: r for r in trows}
    for r in det:
        t = by_id[r["pair_id"]]
        assert t["n_crowd"] == 0 and r["target_text"] == t["box_text"] and r["boxes"] == t["boxes_q"]
    for pid in {r["pair_id"] for r in inst}:
        t = by_id[pid]
        rs = [r for r in inst if r["pair_id"] == pid]
        assert [len(r["boxes"]) for r in rs] == [e - s for s, e in XTD.chunk_ranges(t["n_kept"], 12)]
        assert "".join(r["target_text"][1:-1] + ", " for r in rs)[:-2] == t["spline_text"][1:-1]  # 各段拼起来就是整对的目标
    big = [t for t in trows if t["truncated"] and t["id"] in {r["pair_id"] for r in inst}]
    assert all([len(r["boxes"]) for r in inst if r["pair_id"] == t["id"]] == [12, 12, 6] for t in big)
    x = XTD.load_xm_data(out, params=PARAMS, data_root=root)
    assert len(x.pool("coco_det")) == len(det) and len(x.pool("coco_inst")) == len(inst)


def test_cli_refuses(tmp_path, capsys):
    tg = XF.write_targets(tmp_path / "targets", XF.target_rows())
    with pytest.raises(SystemExit) as e:  # 末级目录名不带运行编号
        BX.main(["--targets", str(tg), "--out", str(tmp_path / "xm_data")])
    assert e.value.code == 2 and "运行编号" in capsys.readouterr().err
    assert BX.main(["--targets", str(tg), "--out", str(tmp_path / "0105_xm_data")]) == 2  # 非 --tiny：要 14,400 条，20 对不够
    assert "不够" in capsys.readouterr().err
    assert BX.main(["--tiny", "--targets", str(tmp_path / "nope"), "--out", str(tmp_path / "x")]) == 2  # 没有训练目标
    assert "build_targets" in capsys.readouterr().err
    bad = XF.target_rows()
    bad[3] = {**bad[3], "split": "xm_tune"}  # 混进评测划分的行：拒绝
    tb = XF.write_targets(tmp_path / "bad", bad)
    assert BX.main(["--tiny", "--targets", str(tb), "--out", str(tmp_path / "y")]) == 2
    assert "只读训练集" in capsys.readouterr().err


def test_frozen_prompts_recorded_and_required(tmp_path, monkeypatch):
    """训练样本记的是冻结的 xm-prompts-v1（D168）；提示没冻结时非 --tiny 拒绝生成。"""
    fi = XTD.format_identity()
    assert (fi["prompt_version"], fi["prompts_frozen"], fi["max_boxes"], fi["n_ctrl"]) == ("xm-prompts-v1", True, 30, 16)
    assert fi["templates_sha256"] == XMP.identity()["templates_sha256"]
    tg = XF.write_targets(tmp_path / "targets", XF.target_rows())
    assert BX.main(["--tiny", "--targets", str(tg), "--out", str(tmp_path / "a")]) == 0
    assert json.loads((tmp_path / "a" / XTD.DATA_DIR / XTD.MANIFEST).read_text())["format"] == fi
    monkeypatch.setattr(XMP, "FROZEN", False)
    with pytest.raises(XTD.XMDataError, match="冻结"):
        XTD.build(tg, tmp_path / "0106_xm_data", params=PARAMS, size=XTD.sizing_from_config(CFG_DOC, tiny=False), tiny=False)
    with pytest.raises(XTD.XMDataError, match="目标写法"):  # 提示的口径变了：已生成的数据不能再读
        XTD.load_xm_data(tmp_path / "a", params=PARAMS)


def test_train_data_modules_import_no_learned_modules():
    """样本的生成、提示与指标的桩只用不含学习的模块（同 tests/test_xm_targets.py 的核对）。"""
    import subprocess
    import sys

    from src.explore import guidance as G

    code = ("import json, sys; import src.xm.train_data, src.xm.prompts, src.xm.metrics; import src.explore.guidance as G; "
            "print(json.dumps({'loaded': G.loaded_learned_modules(), 'roots': sorted({m.split('.')[0] for m in sys.modules})}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True)
    doc = json.loads(r.stdout.strip().splitlines()[-1])
    assert doc["loaded"] == [] and not set(doc["roots"]) & set(G.BANNED_MODULE_ROOTS)
