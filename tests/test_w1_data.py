"""DEV-0051：W1 的划分（src/w1/data.py、configs/w1_splits.json）。

- 入库的清单就是重算的结果（逐字节），文件与各划分的 sha256 钉住；
- devtune 300 句与 devcheck 500 句不重叠、都在 E1 的 poly:refcoco_unc_val 里，恰是同一哈希排序的第 1–300 与第 301–800 名；
- test 划分与 E1 的 poly 子集逐 id 相同；计数的两个划分对上 E1 的开发子集与评测总体；
- w1_items 在 fixture 上按清单取样本（加载器的顺序），对不上就报错。
"""

from __future__ import annotations

import copy
import hashlib

import pytest

from src.data.eval_subsets import load_eval_subsets
from src.data.subsets import dump_config, hash_rank, load_dev_subsets, population_digest, sort_ids
from src.e1.tasks import DataPaths
from src.w1.config import load_w1_config
from src.w1.data import W1_SPLITS_PATH, build_from_files, load_splits, split_names, splits_sha256, tiny_splits, w1_items

W1_SPLITS_SHA256 = "a1371851555e854fac6f9d0e267f19a23602ee9cf31d5c303ef08432d1f7f0f7"
SPLIT_SHA256 = {
    "devtune": "21e50b3bb13e25be281d268543d8df5801226f828640e44f99e2aa502bb6635c",
    "devcheck": "e947f25f58c98ad907dbbef18e22dd285a0c3456acbd65baaa758cf56a102d0f",
    "test_refcoco_unc_testA": "ffb87b4634bbfaf792477fc067467a56d72fe3e07d0e66a432a58dfe537fb5c7",
    "test_refcoco_unc_testB": "f4fdc21a69a99dff84af2b861b7de0f5589896db17eace8a36e884ebe98fc273",
    "test_refcoco+_unc_testA": "18782733cda29f6f8b5fc2cbc7b4b573c1c29e322290f2017660473a8a632778",
    "test_refcoco+_unc_testB": "69c4229fbdbf35ee036c664c593942eb3ac5a72edc9f34b458fedbe69aeb6791",
    "test_refcocog_umd_test": "d09c23cd439827d3951bede4e52dbfca9cafaa622025f0966cb1d0cd5f0edd8b",
    "devtune_countbench": "3559c5aec49d23d3729f578c4ee0f39cacfe0e3e6fcd1f9a743708a03c509cd5",
    "test_countbench": "dad53f96fc19e90fe8be3bf4b85096329ff480ddefd0abcf9fe9c1c2152e73d4",
}
DOC = load_splits()
E1 = load_eval_subsets()


def test_file_is_the_rebuilt_config_and_pinned():
    raw = W1_SPLITS_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == W1_SPLITS_SHA256 == splits_sha256(DOC)
    assert dump_config(build_from_files()) == raw.decode("utf-8")
    assert {k: v["sha256"] for k, v in DOC["splits"].items()} == SPLIT_SHA256
    for k, v in DOC["splits"].items():
        if v["ids"] is not None:
            assert population_digest(v["ids"]) == v["sha256"] and len(v["ids"]) == v["n"] and v["ids"] == sort_ids(v["ids"]), k


def test_dev_splits_are_disjoint_ranked_slices():
    tune, check = DOC["splits"]["devtune"]["ids"], DOC["splits"]["devcheck"]["ids"]
    assert (len(tune), len(check)) == (300, 500) and not set(tune) & set(check)
    pop = E1["subsets"]["poly:refcoco_unc_val"]["ids"]
    assert len(pop) == 1000 and set(tune) | set(check) <= set(pop)
    ranked = hash_rank(pop, "w1_dev", "w1_dev_v1")
    assert set(tune) == set(ranked[:300]) and set(check) == set(ranked[300:800])
    for name in ("devtune", "devcheck"):
        e = DOC["splits"][name]
        assert (e["dataset"], e["split_by"], e["split"], e["tasks"]) == ("refcoco", "unc", "val", ["refseg", "refexp_box"])


def test_test_splits_equal_e1_poly_subsets():
    keys = load_w1_config().splits["test"]
    assert len(keys) == 5
    for key in keys:
        e = DOC["splits"][f"test_{key}"]
        assert e["ids"] == E1["subsets"][f"poly:{key}"]["ids"] and e["n"] == 1000, key


def test_count_splits_match_e1():
    dev = load_dev_subsets()["subsets"]["countbench"]
    e = DOC["splits"]["devtune_countbench"]
    assert e["ids"] == sort_ids(dev["ids"]) and e["n"] == 49 and e["tasks"] == ["count"]
    t = DOC["splits"]["test_countbench"]
    assert (t["n"], t["sha256"], t["ids"]) == (442, E1["populations"]["countbench"]["sha256"], None)
    assert split_names("count", DOC) == ["devtune_countbench", "test_countbench"]
    assert split_names("refseg", DOC) == split_names("refexp_box", DOC) == ["devtune", "devcheck"] + [
        f"test_{k}" for k in load_w1_config().splits["test"]]


def test_w1_items_on_fixture():
    doc = tiny_splits()
    assert doc["tiny"] is True and (doc["splits"]["devtune"]["n"], doc["splits"]["devcheck"]["n"]) == (2, 3)
    assert not set(doc["splits"]["devtune"]["ids"]) & set(doc["splits"]["devcheck"]["ids"])
    paths = DataPaths.posthoc_fixtures()
    seg = w1_items("refseg", "devcheck", paths, splits=doc)
    box = w1_items("refexp_box", "devcheck", paths, splits=doc)
    assert [s.id for s in seg] == [s.id for s in box] == [f"refcoco_unc/val/{i}" for i in doc["splits"]["devcheck"]["ids"]]
    t = w1_items("refseg", "test_refcocog_umd_test", paths, splits=doc)
    assert [str(s.fields["sent_id"]) for s in t] == doc["splits"]["test_refcocog_umd_test"]["ids"]
    cnt = w1_items("count", "test_countbench", DataPaths.fixtures(), splits=doc)
    assert len(cnt) == 2 and all("countbenchqa_question" in s.fields for s in cnt)
    assert [s.fields["row"] for s in w1_items("count", "devtune_countbench", DataPaths.fixtures(), splits=doc)] == [1]
    with pytest.raises(ValueError, match="只用于"):
        w1_items("count", "devcheck", paths, splits=doc)
    bad = copy.deepcopy(doc)
    bad["splits"]["devcheck"]["ids"] = ["100", "999999"]
    with pytest.raises(KeyError, match="找不到"):
        w1_items("refseg", "devcheck", paths, splits=bad)
    bad = copy.deepcopy(doc)
    bad["splits"]["test_countbench"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="指纹"):
        w1_items("count", "test_countbench", DataPaths.fixtures(), splits=bad)
