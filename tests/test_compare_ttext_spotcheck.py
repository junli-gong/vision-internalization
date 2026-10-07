"""DEV-0039 §2：scripts/compare_ttext_spotcheck.py（冻结对照的 T 文本抽查比对，D72-4）。

两边都是真的 run_e1 --tiny 输出（Qwen 小模型、tests/fixtures/e1 的小数据，批大小 4）：基座跑全量，抽查跑 --limit 4（前 4 条 = 第一批）。
不读 experiments/ 的真实大文件。
- 同一份输出与自己比、抽查（前 N 条）与基座比 → 相同（退出码 0），JSON 与 MD 都写出、同样的输入逐字节相同；检测的 F1 有 GT 时才算；
- 改一条原始输出 → 不完全相同（退出码 1），列出那条 id 与解析结果的一致率；
- id 对不上（顺序换了、多于基座、抽查不完整）→ 不可比（退出码 2）；
- 提示配置的 sha256 不同 → 不可比；已裁定等价的一对（D60：5de1aec7… 与 f3353de0…）在 eval 上照收并记下，指令原文不同仍不可比；
- 其余运行身份（图像口径、批配置）不同 → 不可比；字母表只要求抽查那边是基座的子集；
- N 不在基座全量运行的批边界上（含字母集合一变就另起一批的情形）→ 不可比，报出最近的边界；
- 只有 predictions.jsonl.gz 时照读，LARGE_FILES.sha256 对不上 → 不可比；
- 事后读数（没有 raw_text 的记录、读法不是 text / letter）→ 不可比；多边形的 cIoU 与配对差另有单元测试（多边形上限 2048 token，
  小模型在 CPU 上跑太慢，不进整合的那两遍 run_e1）。
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.e1 import prompts as P

REPO = Path(__file__).resolve().parents[1]
MODEL = "qwen3.5-0.8b"
TASKS = ("refexp_box", "grid16", "count", "ovd", "rel_depth_nyu", "spatial_cvbench")  # 多边形（上限 2048 token）太慢，cIoU 另有单元测试
LIMIT = 4  # = --tiny 的批大小：前 4 条恰好是第一批


def _script():
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_compare_ttext_spotcheck_under_test", REPO / "scripts" / "compare_ttext_spotcheck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


CT = _script()


def run_e1(tasks, *args: str) -> None:
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "run_e1.py"), "--tiny", "--model", MODEL, "--task", *tasks, *args],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr[-3000:]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """基座（全量）与抽查（--limit 4）两份 run_e1 --tiny 输出的模型目录。检测的 fixture 只有 4 张图（--limit 4 就是全量），这个任务的
    抽查直接拷基座的叶目录（CPU 上一遍约 30 秒，确定的输出与重跑相同）。"""
    root = tmp_path_factory.mktemp("spot")
    run_e1(TASKS, "--out", str(root / "base"))
    run_e1([t for t in TASKS if t != "ovd"], "--out", str(root / "spot"), "--limit", str(LIMIT))
    assert len(_recs(root / "base" / MODEL / "ovd")) == LIMIT
    shutil.copytree(root / "base" / MODEL / "ovd", root / "spot" / MODEL / "ovd")
    return root / "base" / MODEL, root / "spot" / MODEL


def _copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _recs(leaf: Path) -> list[dict]:
    return [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines() if x.strip()]


def _write_recs(leaf: Path, recs: list[dict], *, fix_ids: bool = True) -> None:
    (leaf / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    if fix_ids:  # 让 config.yaml 的条数与 id 指纹跟着 predictions 走（模拟一次真的这样跑出来的运行）
        cfg = _cfg(leaf)
        cfg["data"]["n_samples"] = len(recs)
        cfg["data"]["sample_ids_sha256"] = hashlib.sha256("\n".join(r["id"] for r in recs).encode("utf-8")).hexdigest()
        _write_cfg(leaf, cfg)


def _cfg(leaf: Path) -> dict:
    return yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))


def _write_cfg(leaf: Path, cfg: dict) -> None:
    (leaf / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _main(base: Path, spot: Path, out: Path, *extra: str) -> tuple[int, dict]:
    code = CT.main(["--base", str(base), "--spot", str(spot), "--out", str(out), *extra])
    return code, json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 相同
# ---------------------------------------------------------------------------


def test_identical_self_and_prefix(runs, tmp_path):
    base, spot = runs
    code, rep = _main(base, base, tmp_path / "self.json")
    assert code == 0 and rep["verdict"] == "相同"
    assert rep["tasks_compared"] == list(TASKS) and not rep["tasks_only_in_base"] and not rep["tasks_only_in_spot"]
    code, rep = _main(base, spot, tmp_path / "a" / "prefix.json", "--tiny")  # 输出目录不存在时建好
    assert code == 0 and rep["verdict"] == "相同", rep
    md = (tmp_path / "a" / "prefix.md").read_text(encoding="utf-8")
    assert md.startswith("# 冻结对照的 T 文本抽查：相同") and "## 规则" in md and "批" in md
    for t in TASKS:
        r = rep["tasks"][t]
        n_base = len(_recs(base / t))
        assert r["n"] == min(LIMIT, n_base) and r["identical"] is True and r["differing_ids"] == []
        for k in ("raw_identical", "parsed_identical", "score_identical"):
            assert r[k] == {"n": r["n"], "rate": 1.0}, (t, k)
        assert r["batch"]["n_on_batch_boundary"] and r["batch"]["n_base"] == n_base and r["batch"]["batch_size"] == 4
        assert r["prompt_config"]["equivalence"] is None and r["base_limit"] is None
        assert r["spot_limit"] == (None if t == "ovd" else LIMIT)  # 检测的抽查是拷的基座（fixture 见上）
        m = r["metric"]
        assert m["base"] == m["spot"], t
        if m["diff"] is not None:
            assert m["diff"]["estimate"] == 0.0
    assert rep["tasks"]["ovd"]["metric"]["base"] is not None  # --tiny：检测的 GT 取 tests/fixtures/e1
    assert rep["tasks"]["refexp_box"]["metric"]["name"] == "acc@0.5" and rep["tasks"]["grid16"]["metric"]["name"] == "gIoU"
    assert rep["tasks"]["spatial_cvbench"]["batch"]["readout"] == "letter"
    # 没有 GT 时检测的指标记 null 并注明；同样的输入，报告逐字节相同
    code, rep2 = _main(base, spot, tmp_path / "nogt.json")
    assert code == 0 and rep2["tasks"]["ovd"]["metric"]["base"] is None and "--data-root" in rep2["tasks"]["ovd"]["metric"]["note"]
    _main(base, spot, tmp_path / "nogt2.json")
    assert (tmp_path / "nogt.json").read_bytes() == (tmp_path / "nogt2.json").read_bytes()
    assert (tmp_path / "nogt.md").read_bytes() == (tmp_path / "nogt2.md").read_bytes()
    # --task 只比指定的任务
    code, rep3 = _main(base, spot, tmp_path / "two.json", "--task", "count", "grid16")
    assert code == 0 and rep3["tasks_compared"] == ["grid16", "count"]


# ---------------------------------------------------------------------------
# 不完全相同
# ---------------------------------------------------------------------------


def test_one_raw_output_changed(runs, tmp_path):
    base, spot = runs
    s = _copy(spot, tmp_path / "spot")
    recs = _recs(s / "grid16")
    recs[2]["raw_text"] += " "  # 只改原始输出，解析结果不动
    _write_recs(s / "grid16", recs, fix_ids=False)
    recs = _recs(s / "count")
    recs[0]["raw_text"] = "999"
    recs[0]["parsed"] = {"count": 999}
    _write_recs(s / "count", recs, fix_ids=False)
    code, rep = _main(base, s, tmp_path / "r.json")
    assert code == 1 and rep["verdict"] == "不完全相同"
    assert rep["summary"]["tasks_not_identical"] == ["grid16", "count"]
    g = rep["tasks"]["grid16"]
    assert g["differing_ids"] == [recs_id(spot / "grid16", 2)] and g["n_differing"] == 1
    assert g["raw_identical"] == {"n": 3, "rate": 0.75} and g["parsed_identical"]["rate"] == 1.0
    c = rep["tasks"]["count"]
    assert c["differing_ids"] == [recs_id(spot / "count", 0)] and c["parsed_identical"]["n"] == c["n"] - 1
    assert all(rep["tasks"][t]["identical"] for t in TASKS if t not in ("grid16", "count"))
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## 原始输出不同的条目" in md and recs_id(spot / "grid16", 2) in md and "解析结果一致率" in md


def recs_id(leaf: Path, i: int) -> str:
    return _recs(leaf)[i]["id"]


def test_differing_ids_capped_at_20(tmp_path):
    """列出的不同 id 每个任务最多 20 条（MAX_LISTED_IDS），条数照实报。"""
    assert CT.MAX_LISTED_IDS == 20
    base = tmp_path / "b" / "count"
    base.mkdir(parents=True)
    recs = [{"id": f"x/{i}", "instruction": "q", "extras": {}, "image_size": [4, 4], "readout": "text", "raw_text": "1",
             "failure": None, "parsed": {"count": 1}, "score": {"gt": 1, "correct": True}} for i in range(24)]
    _write_recs_with_cfg(base, recs)
    spot = _copy(base.parent, tmp_path / "s") / "count"
    for r in recs:
        r["raw_text"] = "1 "
    _write_recs(spot, recs, fix_ids=False)
    rep = CT.compare(base.parent, spot.parent)
    r = rep["tasks"]["count"]
    assert rep["verdict"] == "不完全相同" and r["n_differing"] == 24 and r["differing_ids"] == [f"x/{i}" for i in range(20)]
    assert "共 24 条，列前 20 条" in CT.markdown(rep)


def _write_recs_with_cfg(leaf: Path, recs: list[dict], *, batch_size: int = 4, readout: str = "text") -> None:
    """合成的叶目录：最小的 config.yaml（比对只读这些键）加 predictions。"""
    cfg = {"model": {"key": MODEL}, "task": "count", "subset": "eval", "image_policy": {"tag": "default"},
           "generation": {"readout": {"kind": readout, "parse": "count"}}, "prompt": {"config_sha256": "ab", "entry": {}},
           "data": {"n_samples": len(recs), "has_gt": True,
                    "sample_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in recs).encode("utf-8")).hexdigest()},
           "limit": None, "batch_size": batch_size, "batch_protocol": {"batch_size": batch_size}}
    _write_cfg(leaf, cfg)
    _write_recs(leaf, recs, fix_ids=False)


# ---------------------------------------------------------------------------
# 不可比：条目、提示配置、其余运行身份、批边界、大文件
# ---------------------------------------------------------------------------


def _na(base: Path, spot: Path, out: Path, match: str, *extra: str) -> dict:
    code, rep = _main(base, spot, out, *extra)
    assert code == 2 and rep["verdict"] == "不可比", rep
    assert match in rep["reason"], rep["reason"]
    assert (out.with_suffix(".md")).read_text(encoding="utf-8").startswith("# 冻结对照的 T 文本抽查：不可比")
    return rep


def test_id_mismatch_is_error(runs, tmp_path):
    base, spot = runs
    s = _copy(spot, tmp_path / "s1")
    recs = _recs(s / "refexp_box")
    recs[1], recs[2] = recs[2], recs[1]  # 顺序换了：不是基座清单的前 N 条
    _write_recs(s / "refexp_box", recs)
    _na(base, s, tmp_path / "a.json", "不是基座评测清单的前 4 条")
    s = _copy(spot, tmp_path / "s2")
    recs = _recs(s / "grid16")
    recs[3]["id"] = "refcoco/val/does-not-exist"
    _write_recs(s / "grid16", recs)
    _na(base, s, tmp_path / "b.json", "第 3 条")
    s = _copy(spot, tmp_path / "s3")  # 抽查不完整：predictions 比 config.yaml 记的少
    _write_recs(s / "count", _recs(s / "count")[:1], fix_ids=False)
    _na(base, s, tmp_path / "c.json", "不完整")
    s = _copy(spot, tmp_path / "s4")  # 抽查多于基座
    recs = _recs(base / "count")
    _write_recs(s / "count", recs + [dict(recs[0], id="countbench/all/extra")])
    _na(base, s, tmp_path / "d.json", "多于基座")
    s = _copy(spot, tmp_path / "s5")  # id 相同、输入不同（指令原文）
    recs = _recs(s / "count")
    recs[0]["instruction"] = recs[0]["instruction"] + " Please."
    _write_recs(s / "count", recs, fix_ids=False)
    _na(base, s, tmp_path / "e.json", "输入")
    _na(base, spot, tmp_path / "f.json", "不是两边都有的任务目录", "--task", "refexp_poly")


def test_prompt_config_sha_mismatch_is_error(runs, tmp_path):
    base, spot = runs
    s = _copy(spot, tmp_path / "s")
    cfg = _cfg(s / "grid16")
    cfg["prompt"]["config_sha256"] = "0" * 64
    _write_cfg(s / "grid16", cfg)
    _na(base, s, tmp_path / "a.json", "提示配置的 sha256 不同")


def test_prompt_config_equivalent_pair_d60(runs, tmp_path):
    """RUN-0027 用 5de1aec7…，之后的代码是 f3353de0…：D60 裁定两者在 eval、dev、free、extra 上等价，照收并记进报告；子集不在范围内、
    或指令原文不同，仍不可比。"""
    old, new = CT.EQUIVALENT_PROMPT_CONFIGS[0]["sha256"]
    base, spot = runs
    b, s = _copy(base, tmp_path / "b"), _copy(spot, tmp_path / "s")
    for leaf, sha in ((b, old), (s, new)):
        for t in TASKS:
            cfg = _cfg(leaf / t)
            cfg["prompt"]["config_sha256"] = sha
            _write_cfg(leaf / t, cfg)
    code, rep = _main(b, s, tmp_path / "ok.json")
    assert code == 0 and rep["verdict"] == "相同"
    eq = rep["summary"]["prompt_config_equivalence"]
    assert (eq["base_sha256"], eq["spot_sha256"]) == (old, new) and "D60" in eq["basis"]
    assert all(rep["tasks"][t]["prompt_config"]["equivalence"] == eq for t in TASKS)
    assert "已裁定等价" in (tmp_path / "ok.md").read_text(encoding="utf-8")
    with pytest.raises(CT.NotComparable, match="sha256 不同"):
        CT.prompt_equivalence(old, new, "think")  # 思考子集改过（D59-1），不在等价范围内
    recs = _recs(s / "count")
    recs[1]["instruction"] = recs[1]["instruction"].replace("How many", "Count the")
    _write_recs(s / "count", recs, fix_ids=False)
    _na(b, s, tmp_path / "bad.json", "输入")


def test_equivalence_table_pins():
    """等价表里的两份配置：旧的就是存档的 RUN-0027 配置（tests/fixtures/e1/e1_prompts_5de1aec7.json），新的就是现在的 E1 提示配置；
    等价的子集与 tests/test_e1_d59.py 核对的子集相同。现在的配置一改，这里就失败：要先确认新配置与 5de1aec7… 在这些子集上仍然等价
    （test_e1_d59），再在 EQUIVALENT_PROMPT_CONFIGS 里加一条，否则抽查对 RUN-0027 会判不可比。"""
    from tests import test_e1_d59 as D59

    old, new = CT.EQUIVALENT_PROMPT_CONFIGS[0]["sha256"]
    assert old == D59.OLD_SHA256 == hashlib.sha256(D59.OLD_PATH.read_bytes()).hexdigest()
    assert new == P.load_prompt_config().sha256
    assert tuple(CT.EQUIVALENT_PROMPT_CONFIGS[0]["subsets"]) == tuple(D59.SAME_SUBSETS)


def test_other_identity_mismatch_is_error(runs, tmp_path):
    base, spot = runs
    for name, key, val, match in (("img", "image_policy", {"tag": "default", "type": "PixelBudget", "pixel_budget_tokens": 64},
                                   "image_policy"),
                                  ("bs", "batch_size", 8, "batch_size"),
                                  ("model", "model", {"key": "qwen3.5-2b"}, "model")):
        s = _copy(spot, tmp_path / name)
        cfg = _cfg(s / "count")
        cfg[key] = val
        _write_cfg(s / "count", cfg)
        _na(base, s, tmp_path / f"{name}.json", match)
    s = _copy(spot, tmp_path / "gen")
    cfg = _cfg(s / "count")
    cfg["generation"]["max_new_tokens"] = 7
    _write_cfg(s / "count", cfg)
    _na(base, s, tmp_path / "gen.json", "generation")


def test_metric_ciou_paired_diff():
    """多边形的标题 cIoU（累计交 / 累计并）：两边各自的值与 CI，配对差 = 抽查 − 基座（ratio_of_sums 的配对 bootstrap）。"""
    rb = [{"score": {"inter": i, "union": 10}, "failure": None} for i in (5, 6, 7, 8)]
    rs = [{"score": {"inter": i, "union": 10}, "failure": None} for i in (5, 6, 7, 4)]
    m = CT.task_metric("refexp_poly", rb, rs, None)
    assert m["name"] == "cIoU" and m["base"]["value"] == pytest.approx(0.65) and m["spot"]["value"] == pytest.approx(0.55)
    assert m["diff"]["estimate"] == pytest.approx(-0.1) and m["diff"]["low"] <= -0.1 <= m["diff"]["high"]
    assert CT.task_metric("ovd", rb, rs, None)["base"] is None  # 检测没有 GT：null 并注明


def test_letter_table_subset_rule():
    ab = {"letters": ["A", "B"], "entries": {"A": [{"spelling": "A", "token_id": 32}], "B": [{"spelling": "B", "token_id": 33}]},
          "skipped": []}
    abc = copy.deepcopy(ab)
    abc["letters"].append("C")
    abc["entries"]["C"] = [{"spelling": "C", "token_id": 34}]
    assert CT._letter_table_ok(abc, ab) and CT._letter_table_ok(ab, ab) and CT._letter_table_ok(None, None)
    assert not CT._letter_table_ok(ab, abc)  # 抽查的字母多于基座
    bad = copy.deepcopy(ab)
    bad["entries"]["A"][0]["token_id"] = 99
    assert not CT._letter_table_ok(ab, bad) and not CT._letter_table_ok(ab, None)


def test_batch_boundary_rule(runs, tmp_path):
    # 单元：字母集合一变就另起一批（与 runner.batch_plan 同一个函数），边界不在批大小的整数倍上
    recs = [{"extras": {"letters": lt}} for lt in (["A", "B"],) * 3 + (["A", "B", "C"],) * 2 + (["A", "B"],) * 6]
    assert CT.batch_boundaries("letter", recs, 4) == [3, 5, 9, 11]
    assert CT.batch_boundaries("text", recs, 4) == [4, 8, 11]
    # 整合：抽查只有 3 条（批大小 4）→ 前 3 条的批组成与全量不同 → 不可比，报出最近的边界
    base, spot = runs
    s = _copy(spot, tmp_path / "s")
    _write_recs(s / "refexp_box", _recs(s / "refexp_box")[:3])
    rep = _na(base, s, tmp_path / "a.json", "不在基座全量运行的批边界上")
    assert "最近的边界是 0 与 4" in rep["reason"]
    # 合成的字母任务：N = 4 是批大小的整数倍，但字母集合在第 3 条变了，边界是 3、5 → 不可比；N = 5 可比
    b = tmp_path / "lb" / "count"
    b.mkdir(parents=True)
    letters = [["A", "B"]] * 3 + [["A", "B", "C"]] * 2 + [["A", "B"]] * 3
    full = [{"id": f"q/{i}", "instruction": "q", "extras": {"letters": lt}, "image_size": [4, 4], "readout": "letter",
             "raw_text": "A", "failure": None, "parsed": {"letter": "A"}, "score": {"gt": "A", "correct": True}}
            for i, lt in enumerate(letters)]
    _write_recs_with_cfg(b, full, readout="letter")
    for n, ok in ((4, False), (5, True), (3, True)):
        sp = tmp_path / f"ls{n}" / "count"
        sp.mkdir(parents=True)
        _write_recs_with_cfg(sp, full[:n], readout="letter")
        if ok:
            assert CT.compare(b.parent, sp.parent)["verdict"] == "相同"
        else:
            with pytest.raises(CT.NotComparable, match="最近的边界是 3 与 5"):
                CT.compare(b.parent, sp.parent)


def test_gz_predictions_and_large_files(runs, tmp_path):
    """基座只有 predictions.jsonl.gz（RUN-0027 的定位就是这样）：照读，有 LARGE_FILES.sha256 就核对解压后的 sha256 与字节数。"""
    base, spot = runs
    b = _copy(base, tmp_path / "b")
    leaf = b / "refexp_box"
    raw = (leaf / "predictions.jsonl").read_bytes()
    (leaf / "predictions.jsonl.gz").write_bytes(gzip.compress(raw))
    (leaf / "predictions.jsonl").unlink()
    gz_sha = hashlib.sha256((leaf / "predictions.jsonl.gz").read_bytes()).hexdigest()
    (leaf / "LARGE_FILES.sha256").write_text(f"{hashlib.sha256(raw).hexdigest()}  {len(raw)}  predictions.jsonl  {gz_sha}  0\n")
    code, rep = _main(b, spot, tmp_path / "ok.json")
    assert code == 0 and rep["verdict"] == "相同" and "refexp_box" in rep["tasks_compared"]
    (leaf / "LARGE_FILES.sha256").write_text(f"{'0' * 64}  {len(raw)}  predictions.jsonl  {gz_sha}  0\n")
    _na(b, spot, tmp_path / "bad.json", "LARGE_FILES.sha256")


def test_posthoc_records_refused(tmp_path):
    """事后读数（去偏字母读数等）的记录没有 raw_text：不比，判不可比（否则 None == None 会被当成相同）。"""
    recs = [{"id": f"q/{i}", "instruction": "q", "extras": {"letters": ["A", "B"]}, "image_size": [4, 4], "readout": "letter",
             "raw_text": "A", "failure": None, "parsed": {"letter": "A"}, "score": {"gt": "A", "correct": True}} for i in range(4)]
    b = tmp_path / "b" / "count"
    b.mkdir(parents=True)
    _write_recs_with_cfg(b, recs, readout="letter")
    s = _copy(b.parent, tmp_path / "s") / "count"
    _write_recs(s, [{k: v for k, v in r.items() if k != "raw_text"} for r in recs], fix_ids=False)
    with pytest.raises(CT.NotComparable, match="raw_text"):
        CT.compare(b.parent, s.parent)
    for leaf in (b, s):
        cfg = _cfg(leaf)
        cfg["generation"]["readout"]["kind"] = "letter_debiased"
        _write_cfg(leaf, cfg)
    _write_recs(s, recs, fix_ids=False)
    with pytest.raises(CT.NotComparable, match="letter_debiased"):
        CT.compare(b.parent, s.parent)


def test_cli_usage(runs, tmp_path):
    base, spot = runs
    with pytest.raises(SystemExit) as e:
        CT.main(["--base", str(base), "--spot", str(spot), "--out", str(tmp_path / "x.md")])
    assert e.value.code == 2
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "compare_ttext_spotcheck.py"), "--base", str(base), "--spot", str(spot),
                        "--out", str(tmp_path / "cli.json"), "--tiny"], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0 and "判定：相同" in p.stdout, p.stderr[-2000:]
