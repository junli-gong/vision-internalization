"""DEV-0059：W2 的划分（src/w2/data.py、configs/w2_splits.json）。

- 规则（纯函数 select_devcheck2，构造的总体）：按图像排除（同图的其他句子也去掉）、hash_rank 的顺序、与 W1 不重叠（按句、按图）、不够就报错。
- fixture（tests/fixtures/w2：e1_posthoc 的 RefCOCO 加两张 val 图）上的整套生成：经 E1 的加载器（subset='eval'）、W1 的 tiny 划分。
- 入库的清单：devtune800 = W1 的 devtune + devcheck 原样，test 与计数同 W1；devcheck2 待服务器生成（或已生成时自洽）；--check 通过。
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.subsets import dump_config, hash_rank, population_digest
from src.w1.data import load_splits as load_w1_splits
from src.w2 import data as D
from src.w2.config import load_w2_config

REPO = Path(__file__).resolve().parents[1]
CFG = load_w2_config()


def pop(n_images=40, per_image=5):
    """构造的总体：图 i 上有 per_image 句，sent_id = 1000 i + j。"""
    return [(str(1000 * i + j), f"coco:{i}") for i in range(n_images) for j in range(per_image)]


def test_exclusion_is_by_image_and_ranking_is_hash_rank():
    p = pop()
    excl = ["3000", "7004", "7001"]  # 图 3 与图 7 上的几句
    sel = D.select_devcheck2(p, excl, n=20, salt="w2_dev_v1", purpose="w2_devcheck")
    imgs = {s: k for s, k in p}
    assert not {imgs[s] for s in sel["ids"]} & {"coco:3", "coco:7"}  # 同图的其他句子也排除
    pool = [s for s, k in p if k not in ("coco:3", "coco:7")]
    assert sel["ids"] == sorted(hash_rank(pool, "w2_devcheck", "w2_dev_v1")[:20], key=int)
    assert sel["excluded_images"] == ["coco:3", "coco:7"]
    st = sel["steps"]
    assert (st["population"]["n"], st["exclude"]["n_sentences"], st["exclude"]["n_images"], st["exclude"]["n_sentences_removed"],
            st["after_exclusion"]["n"], st["selected"]["n"]) == (200, 3, 2, 10, 190, 20)
    assert st["selected"]["sha256"] == population_digest(sel["ids"]) and st["after_exclusion"]["sha256"] == population_digest(pool)
    assert sel["image_of"] == {s: imgs[s] for s in sel["ids"]}
    other = D.select_devcheck2(p, excl, n=20, salt="w2_dev_v1", purpose="other")
    assert other["ids"] != sel["ids"]  # purpose 生效


def test_selection_errors():
    p = pop(4, 3)
    with pytest.raises(D.SplitsError, match="不够"):
        D.select_devcheck2(p, ["0"], n=10, salt="s", purpose="p")
    with pytest.raises(D.SplitsError, match="不在总体里"):
        D.select_devcheck2(p, ["999999"], n=2, salt="s", purpose="p")
    with pytest.raises(D.SplitsError, match="重复"):
        D.select_devcheck2(p + [p[0]], [], n=2, salt="s", purpose="p")


def test_fixture_build_follows_the_rule():
    doc = D.tiny_splits(CFG)
    w1 = doc["splits"]["devtune800"]
    e = doc["splits"]["devcheck2"]
    assert w1["ids"] == ["100", "102", "103", "104", "105"]  # W1 的 tiny devtune + devcheck（图 901、902）
    assert e["status"] == "built" and e["n"] == CFG.splits["tiny"]["devcheck2_n"] == 3
    recs = D.population_records(D.tiny_paths())
    img = {str(s.fields["sent_id"]): s.image_keys[0] for s in recs}
    assert "101" not in img  # E1 的开发子集已被加载器剔除
    pool = [s for s in img if img[s] not in {img[i] for i in w1["ids"]}]
    assert sorted(pool, key=int) == ["110", "111", "112", "113", "114", "115"]
    assert e["ids"] == sorted(hash_rank(pool, "w2_devcheck", "w2_dev_v1")[:3], key=int)
    assert not set(e["ids"]) & set(w1["ids"]) and not {img[s] for s in e["ids"]} & {img[s] for s in w1["ids"]}
    assert e["image_of"] == {s: img[s] for s in e["ids"]}
    assert e["construction"]["exclude"]["images"] == ["coco:901", "coco:902"]
    assert D.check_devcheck2(doc, CFG) == []
    # 篡改：把 W1 用过的图上的一句塞进 devcheck2 → 自洽核对报错
    bad = copy.deepcopy(doc)
    bad["splits"]["devcheck2"]["ids"][0] = "103"
    bad["splits"]["devcheck2"]["image_of"] = {**{k: v for k, v in e["image_of"].items() if k != e["ids"][0]}, "103": "coco:901"}
    probs = D.check_devcheck2(bad, CFG)
    assert any("重叠" in p for p in probs) and any("排除的图上" in p for p in probs) and any("sha256" in p for p in probs)
    # 样本按清单取出，与 W1 的取法相同
    samples = D.w2_items("refseg", "devcheck2", D.tiny_paths(), splits=doc)
    assert [s.fields["sent_id"] for s in samples] == [int(i) for i in e["ids"]]


def test_repo_splits_reuse_w1_and_check_passes():
    doc = D.load_splits()
    w1 = load_w1_splits()
    dt = doc["splits"]["devtune800"]
    ids = sorted(set(w1["splits"]["devtune"]["ids"]) | set(w1["splits"]["devcheck"]["ids"]), key=int)
    assert dt["n"] == 800 and dt["ids"] == ids and dt["sha256"] == population_digest(ids)
    for k in ("test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA", "test_refcoco+_unc_testB",
              "test_refcocog_umd_test", "devtune_countbench", "test_countbench"):
        assert doc["splits"][k] == w1["splits"][k], k
    assert set(doc["splits"]) == {"devtune800", "devcheck2", *[k for k in w1["splits"] if k.startswith("test") or "count" in k]}
    e = doc["splits"]["devcheck2"]
    if e["status"] == "pending":  # 服务器生成之前
        assert e["ids"] is None and "--build" in e["build_command"]
        assert dump_config(D.build_skeleton(CFG)) == (REPO / "configs" / "w2_splits.json").read_text(encoding="utf-8")
        with pytest.raises(D.SplitsError, match="还没生成"):
            D.w2_items("refseg", "devcheck2", D.tiny_paths(), splits=doc)
    else:
        assert D.check_devcheck2(doc, CFG) == [] and e["n"] == 500
        assert e["construction"]["population"]["n"] == 10634
        assert e["construction"]["population"]["sha256"].startswith("1783dd7f")
    p = subprocess.run([sys.executable, "-m", "src.w2.data", "--check"], cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_cli_build_and_check_on_fixtures(tmp_path):
    out = tmp_path / "w2_splits.json"
    run = lambda *a: subprocess.run([sys.executable, "-m", "src.w2.data", *a], cwd=REPO, capture_output=True, text=True)  # noqa: E731
    p = run("--build", "--tiny", "--out", str(out))
    assert p.returncode == 0, p.stderr
    assert "排除 devtune800 的 5 句所在的 2 张图" in p.stdout and "取 3 句" in p.stdout
    doc = json.loads(out.read_text())
    assert doc == D.tiny_splits(CFG)
    assert run("--check", "--tiny", "--splits", str(out)).returncode == 0  # 整份重算
    assert run("--check", "--splits", str(out)).returncode == 0  # 不要数据：自洽
    doc["splits"]["devcheck2"]["ids"][0] = "112"
    out.write_text(dump_config(doc))
    assert run("--check", "--tiny", "--splits", str(out)).returncode == 1
    assert run("--check", "--splits", str(out)).returncode == 1
    assert run("--build", "--tiny", "--out", str(out)).returncode == 2  # 不覆盖内容不同的文件
    assert run("--build", "--out", str(out)).returncode == 2  # 没有 --data-root


def test_needs_frozen():
    assert D.needs_frozen("devcheck2") and D.needs_frozen("test_refcoco_unc_testA") and D.needs_frozen("test_countbench")
    assert not D.needs_frozen("devtune800") and not D.needs_frozen("devtune_countbench")
