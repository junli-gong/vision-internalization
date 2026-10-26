"""DEV-0067：探索用的划分（src/data/explore_splits.py、configs/explore_splits.json）与密封守卫（scripts/run_w2.py --splits-file、
--unseal-devcheck3）。

- 规则（纯函数 select_layer、e1_subsets_on、build_groups，构造的总体）：按图排除（同图的其他句子也去掉；跨数据集）、按句排除、哈希排序
  用带前缀的样本 id、devcheck3 不够就报错、explore_val 不够就取全部、两组按图不相交；E1 子集经 population 链间接引用也算。
- fixture（tests/fixtures/explore）上的整套生成：经 E1 的加载器、W2 的 tiny 划分、W2-c 训练池的 fixture；--check（不要数据、整份重算）；
  改坏清单的各处都能查出来。
- 入库的清单：待生成时是骨架，与按入库清单重算的逐字节相同（生成后不要数据的核对通过）；configs/w2_splits.json 不动。
- 守卫：没给门名、门名不对时拒绝（--tiny 也拒绝，什么都不写）；放行时门名进运行身份、时间进日志；devcheck3 要冻结的配置；
  run_w2.py 默认（不给 --splits-file）时运行身份与原来逐字节相同（DEV-0067 改动之前的代码在开发机上对照过，这里钉住键与取值）。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.data import explore_splits as EX
from src.data import loaders as L
from src.data.eval_subsets import EVAL_SUBSETS_PATH, load_eval_subsets
from src.data.subsets import dump_config, hash_rank, population_digest, sort_ids
from src.w2 import data as W2D

REPO = Path(__file__).resolve().parents[1]
RUN_W2 = REPO / "scripts" / "run_w2.py"
W2C_TRAIN_FIXTURE = REPO / "tests" / "fixtures" / "w2c" / "root"
QWEN = "qwen3.5-0.8b"


def pop(n_images=30, per_image=4, base=0):
    """构造的总体：图 i 上有 per_image 句，sent_id = 1000 i + j + base。"""
    return [(str(1000 * i + j + base), f"coco:{i}") for i in range(n_images) for j in range(per_image)]


# ---------------------------------------------------------------------------
# 常量与约定
# ---------------------------------------------------------------------------


def test_rule_constants():
    assert EX.GATE == "G-EX"
    d3, ev = EX.GROUPS[EX.DEVCHECK3], EX.GROUPS[EX.EXPLORE_VAL]
    assert (d3["n"], d3["salt"], d3["purpose"], d3["allow_fewer"], d3["sealed"]) == ((200, 200, 100), "devcheck3_v1", "devcheck3", False, True)
    assert (ev["n"], ev["salt"], ev["purpose"], ev["allow_fewer"], ev["sealed"]) == ((400, 400, 200), "explore_val_v1", "explore_val", True, False)
    assert ev["exclude_images_of_groups"] == (EX.DEVCHECK3,) and d3["exclude_images_of_groups"] == ()
    assert EX.group_target_n(EX.DEVCHECK3) == 500 and EX.group_target_n(EX.EXPLORE_VAL) == 1000
    names = [EX.population_name(*p) for p in EX.POPULATIONS]
    assert names == ["refcoco_unc_val", "refcoco+_unc_val", "refcocog_umd_val"]
    assert set(names) <= set(load_eval_subsets(EVAL_SUBSETS_PATH)["populations"])
    assert EX.group_split_names(EX.DEVCHECK3) == ["devcheck3_refcoco_unc_val", "devcheck3_refcoco+_unc_val", "devcheck3_refcocog_umd_val"]


def test_sample_id_matches_the_loader():
    paths = EX.tiny_paths()
    for ds, sb, sp in EX.POPULATIONS:
        recs = L.load_refcoco(paths.root, ds, sp, subset="eval", dev_config=paths.dev_subsets).records
        assert recs and all(s.id == EX.sample_id(ds, sb, sp, s.fields["sent_id"]) for s in recs)


def test_w2_splits_untouched():
    """configs/w2_splits.json 的 sha256 是 W2 冻结配置身份的一部分：本任务不动它。"""
    assert hashlib.sha256((REPO / "configs" / "w2_splits.json").read_bytes()).hexdigest() == \
        "9dd22dff2ad422ec5d51106bb6e40227ec51afa27130f2a968c945d69055c0f1"


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def test_select_layer_excludes_by_image_and_sentence_and_ranks_prefixed_ids():
    p = pop()
    imgs = dict(p)
    sel = EX.select_layer(p, prefix="refcoco_unc/val/", exclude_images=["coco:3", "coco:7", "coco:999"], exclude_sentences=["5000", "5001", "77"],
                          n=20, salt="devcheck3_v1", purpose="devcheck3", allow_fewer=False)
    assert not {imgs[s] for s in sel["ids"]} & {"coco:3", "coco:7"}  # 同图的其他句子也去掉
    assert not {"5000", "5001"} & set(sel["ids"])
    pool = [s for s, k in p if k not in ("coco:3", "coco:7") and s not in ("5000", "5001")]
    ranked = hash_rank(["refcoco_unc/val/" + s for s in pool], "devcheck3", "devcheck3_v1")[:20]
    assert sel["ids"] == sort_ids(r.split("/")[-1] for r in ranked)  # 哈希的是带前缀的样本 id
    assert sel["ids"] != sort_ids(hash_rank(pool, "devcheck3", "devcheck3_v1")[:20])  # 与不带前缀的不同
    st = sel["steps"]
    assert (st["population"]["n"], st["exclude_images"]["n_images_listed"], st["exclude_images"]["n_images_hit"],
            st["exclude_images"]["n_sentences_removed"], st["exclude_images"]["after"]["n"]) == (120, 3, 2, 8, 112)
    assert (st["exclude_sentences"]["n_sentences_listed"], st["exclude_sentences"]["n_in_population"],
            st["exclude_sentences"]["n_sentences_removed"], st["exclude_sentences"]["after"]["n"]) == (3, 2, 2, 110)
    assert st["exclude_sentences"]["after"]["sha256"] == population_digest(pool)
    assert (st["selected"]["n"], st["selected"]["n_requested"], st["selected"]["all_taken"]) == (20, 20, False)
    assert st["selected"]["sha256"] == population_digest(sel["ids"]) and sel["image_of"] == {s: imgs[s] for s in sel["ids"]}
    other = EX.select_layer(p, prefix="refcoco_unc/val/", exclude_images=[], exclude_sentences=[], n=20, salt="explore_val_v1",
                            purpose="explore_val", allow_fewer=False)
    assert other["ids"] != EX.select_layer(p, prefix="refcoco_unc/val/", exclude_images=[], exclude_sentences=[], n=20,
                                           salt="devcheck3_v1", purpose="devcheck3", allow_fewer=False)["ids"]


def test_select_layer_shortage():
    p = pop(4, 3)
    with pytest.raises(EX.SplitsError, match="不够"):
        EX.select_layer(p, prefix="x/", exclude_images=["coco:0"], exclude_sentences=[], n=10, salt="s", purpose="p", allow_fewer=False)
    sel = EX.select_layer(p, prefix="x/", exclude_images=["coco:0"], exclude_sentences=[], n=10, salt="s", purpose="p", allow_fewer=True)
    assert len(sel["ids"]) == 9 and sel["steps"]["selected"]["all_taken"] and sel["steps"]["selected"]["n_requested"] == 10
    with pytest.raises(EX.SplitsError, match="重复"):
        EX.select_layer(p + [p[0]], prefix="x/", exclude_images=[], exclude_sentences=[], n=2, salt="s", purpose="p", allow_fewer=False)


def test_e1_subsets_resolve_population_chains():
    doc = {"populations": {"A": {}, "B": {}},
           "subsets": {"poly:A": {"population": "A", "ids": ["1", "2"]}, "think:x": {"population": "poly:A", "ids": ["2", "9"]},
                       "think8192:x": {"population": "think:x", "ids": ["9", "10"]}, "poly:B": {"population": "B", "ids": ["1"]}}}
    assert EX.e1_subsets_on(doc, "A") == ["poly:A", "think:x", "think8192:x"]  # 经链间接引用的也算
    x = EX.e1_sentences_on(doc, "A")
    assert x["ids"] == ["1", "2", "9", "10"] and x["n"] == 4 and x["sha256"] == population_digest(["1", "2", "9", "10"])
    assert EX.e1_subsets_on(doc, "B") == ["poly:B"]
    bad = copy.deepcopy(doc)
    bad["subsets"]["think:x"]["population"] = "nope"
    with pytest.raises(EX.SplitsError, match="既不是总体"):
        EX.e1_subsets_on(bad, "A")
    bad["subsets"]["think:x"]["population"] = "think8192:x"
    with pytest.raises(EX.SplitsError, match="环"):
        EX.e1_subsets_on(bad, "A")


def test_e1_subsets_in_the_real_config():
    doc = load_eval_subsets(EVAL_SUBSETS_PATH)
    on = EX.e1_subsets_on(doc, "refcoco_unc_val")
    assert {"poly:refcoco_unc_val", "grid16:refcoco_unc_val", "think:refexp_box", "think8192:refexp_box", "think:refexp_poly",
            "think8192:refexp_poly", "think:grid16", "think8192:grid16", "extra:refcoco_unc_val"} == set(on)
    assert EX.e1_subsets_on(doc, "refcoco+_unc_val") == ["poly:refcoco+_unc_val"]
    assert EX.e1_subsets_on(doc, "refcocog_umd_val") == ["poly:refcocog_umd_val"]


def _toy_groups(tiny=True, n=40):
    """三个总体：refcoco 与 refcoco+ 共用图 0…n−1（句号不同），refcocog 用图 n−10…2n−11；W2 的图 0–4；E1 的子集各 2 句。"""
    pops = {"refcoco_unc_val": pop(n, 3, base=0), "refcoco+_unc_val": pop(n, 3, base=500),
            "refcocog_umd_val": [(str(1000 * i + j + 700), f"coco:{i}") for i in range(n - 10, 2 * n - 10) for j in range(2)]}
    e1 = {k: {"ids": [v[20][0], v[21][0]], "sha256": "x"} for k, v in pops.items()}
    return pops, e1, EX.build_groups(pops, [f"coco:{i}" for i in range(5)], e1, tiny=tiny)


def test_build_groups_disjoint_across_groups_and_datasets():
    pops, e1, (splits, groups) = _toy_groups(tiny=False, n=150)
    assert [splits[x]["n"] for x in EX.group_split_names(EX.DEVCHECK3)] == [200, 200, 100] and groups[EX.DEVCHECK3]["n"] == 500
    imgs = {g: {i for e in splits.values() if e["group"] == g for i in e["image_of"].values()} for g in EX.GROUPS}
    assert not imgs[EX.DEVCHECK3] & imgs[EX.EXPLORE_VAL]  # 按图不相交，跨数据集（refcoco 与 refcoco+、refcocog 共用图）
    assert all(int(k.split(":")[1]) >= 5 for g in imgs.values() for k in g)  # W2 的图都排除了
    for name, e in splits.items():
        pname = e["construction"]["population"]["name"]
        assert not set(e["ids"]) & set(e1[pname]["ids"]), name
        if e["group"] == EX.EXPLORE_VAL:  # devcheck3 的图全在 explore_val 的按图排除里
            assert e["construction"]["exclude_images"]["n_images_of_groups"] == len(imgs[EX.DEVCHECK3])
    ev = [splits[x] for x in EX.group_split_names(EX.EXPLORE_VAL)]
    assert all(e["construction"]["selected"]["all_taken"] or e["n"] == e["construction"]["selected"]["n_requested"] for e in ev)
    assert groups[EX.EXPLORE_VAL]["n"] == sum(e["n"] for e in ev) and groups[EX.EXPLORE_VAL]["n_target"] == 1000


def test_build_groups_sizes_and_shortage(monkeypatch):
    pops, e1, (splits, groups) = _toy_groups(tiny=True)
    d3 = [splits[n] for n in EX.group_split_names(EX.DEVCHECK3)]
    assert [e["n"] for e in d3] == list(EX.GROUPS[EX.DEVCHECK3]["tiny_n"])
    ev = [splits[n] for n in EX.group_split_names(EX.EXPLORE_VAL)]
    assert [e["n"] for e in ev[:2]] == [6, 6]
    g = ev[2]
    assert g["construction"]["selected"]["all_taken"] is (g["n"] < 50)  # refcocog：不够 50 就取全部
    assert groups[EX.DEVCHECK3]["ids"] == sort_ids(EX.sample_id(e["dataset"], e["split_by"], e["split"], s) for e in d3 for s in e["ids"])
    monkeypatch.setitem(EX.GROUPS[EX.DEVCHECK3], "tiny_n", (1000, 3, 2))
    with pytest.raises(EX.SplitsError, match="不够 1000 句"):
        _toy_groups(tiny=True)


# ---------------------------------------------------------------------------
# fixture 上的整套生成与核对
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_doc():
    return EX.build_with_data(EX.tiny_paths(), tiny=True, train_root=W2C_TRAIN_FIXTURE)


def test_fixture_build_follows_the_rule(tiny_doc):
    doc = tiny_doc
    assert EX.status(doc) == EX.STATUS_BUILT and doc["kind"] == EX.KIND and doc["tiny"] is True
    sp = doc["splits"]
    assert [sp[n]["n"] for n in EX.group_split_names(EX.DEVCHECK3)] == [3, 3, 2]
    ev = [sp[n] for n in EX.group_split_names(EX.EXPLORE_VAL)]
    assert [e["n"] for e in ev[:2]] == [6, 6] and ev[2]["construction"]["selected"]["all_taken"] and ev[2]["n"] < 50
    # fixture：W2 的 tiny 划分在 901–906 上；refcocog 的 450、451 在 coco:903（W2 的 test_refcoco_unc_testA 的图）上；
    # E1 的 poly / extra 子集里有 120、121、122、320、420
    all_imgs = {i for e in sp.values() for i in e["image_of"].values()}
    assert not all_imgs & {f"coco:{i}" for i in range(901, 907)}
    g = sp["devcheck3_refcocog_umd_val"]["construction"]
    assert g["exclude_images"]["n_sentences_removed"] == 5 and g["exclude_images"]["n_images_hit"] == 3  # 400–402 在 901、902，450–451 在 903
    assert "903" not in str(doc["exclusion"]["images_of_w2_splits"]["splits"])  # 只记指纹
    assert "coco:903" in doc["exclusion"]["images_of_w2_splits"]["images"]
    used = {(e["dataset"], s) for e in sp.values() for s in e["ids"]}
    assert not used & {("refcocog", "450"), ("refcocog", "451"), ("refcoco", "120"), ("refcoco", "121"), ("refcoco", "122"),
                       ("refcoco+", "320"), ("refcocog", "420")}
    assert set(doc["exclusion"]["images_of_w2_splits"]["splits"]) == {"devtune800", "devcheck2", "test_refcoco_unc_testA",
                                                                       "test_refcoco_unc_testB", "test_refcoco+_unc_testA",
                                                                       "test_refcoco+_unc_testB", "test_refcocog_umd_test"}
    assert all(p["check"]["status"] == "一致" for p in doc["populations"].values())
    tp = doc["checks"]["train_pool"]
    assert tp["status"] == "核对过" and tp["pool"]["n_sentences"] > 0 and all(v["n_images"] == 0 for v in tp["overlap"].values())
    assert tp["exclusions"]["file"] == "dedupe/train_exclusions.json"
    assert EX.check_doc(doc) == []
    assert dump_config(EX.build_with_data(EX.tiny_paths(), tiny=True, train_root=W2C_TRAIN_FIXTURE)) == dump_config(doc)  # 可复现


def test_w2_items_reads_the_explore_splits(tiny_doc):
    paths = EX.tiny_paths()
    ids = []
    for name in EX.group_split_names(EX.DEVCHECK3):
        for task in W2D.REFCOCO_TASKS:
            samples = W2D.w2_items(task, name, paths, splits=tiny_doc)
            assert [str(s.fields["sent_id"]) for s in samples] == tiny_doc["splits"][name]["ids"]
        ids += [s.id for s in samples]
    assert sort_ids(ids) == EX.group_ids(tiny_doc, EX.DEVCHECK3)
    assert W2D.split_names("refseg", tiny_doc) == list(tiny_doc["splits"])


def test_train_pool_overlap_is_an_error():
    with pytest.raises(EX.SplitsError, match="训练池"):
        EX.train_pool_check(W2C_TRAIN_FIXTURE, EX.tiny_paths(), {EX.DEVCHECK3: {"coco:2001"}, EX.EXPLORE_VAL: set()}, tiny=True)


def _mut(doc, fn):
    d = copy.deepcopy(doc)
    fn(d)
    return EX.check_doc(d)


def test_check_catches_tampering(tiny_doc):
    d3 = "devcheck3_refcoco_unc_val"
    e3 = "explore_val_refcoco_unc_val"

    def swap_in_explore_id(d):  # devcheck3 的一句换成 explore_val 的一句（连同它的图）
        s = d["splits"][e3]["ids"][0]
        e = d["splits"][d3]
        old = e["ids"][0]
        e["ids"] = sort_ids(e["ids"][1:] + [s])
        e["image_of"] = {**{k: v for k, v in e["image_of"].items() if k != old}, s: d["splits"][e3]["image_of"][s]}

    def on_w2_image(d):
        d["splits"][d3]["image_of"][d["splits"][d3]["ids"][0]] = "coco:903"

    def e1_sentence(d):
        e = d["splits"]["devcheck3_refcoco+_unc_val"]
        e["ids"] = sort_ids(e["ids"][1:] + ["320"])
        e["image_of"] = {k: e["image_of"].get(k, "coco:911") for k in e["ids"]}

    def drop_excluded_image(d):
        w = d["exclusion"]["images_of_w2_splits"]
        w["images"] = [i for i in w["images"] if i != "coco:901"]

    def bad_group(d):
        d["groups"][EX.DEVCHECK3]["ids"] = d["groups"][EX.DEVCHECK3]["ids"][1:]

    def nonzero_check(d):
        d["checks"]["vs_e1_subsets_by_sentence"][EX.DEVCHECK3] = 1

    def wrong_salt(d):
        d["splits"][d3]["construction"]["selected"]["salt"] = "other"

    for fn, msg in ((swap_in_explore_id, "sha256"), (on_w2_image, "按图排除的图上"), (e1_sentence, "E1 的评测子集"),
                    (drop_excluded_image, "条数或 sha256"), (bad_group, "组 devcheck3"), (nonzero_check, "checks"), (wrong_salt, "规则")):
        problems = _mut(tiny_doc, fn)
        assert problems and any(msg in p for p in problems), (fn.__name__, problems)


def test_cli_build_and_check(tmp_path):
    out = tmp_path / "explore_splits.json"
    run = lambda *a: EX.main(list(a))  # noqa: E731
    assert run("--build", "--tiny", "--train-data-root", str(W2C_TRAIN_FIXTURE), "--out", str(out)) == 0
    assert run("--build", "--tiny", "--train-data-root", str(W2C_TRAIN_FIXTURE), "--out", str(out)) == 0  # 同内容可重写
    assert run("--check", "--splits", str(out)) == 0  # 不要数据的部分
    assert run("--check", "--tiny", "--train-data-root", str(W2C_TRAIN_FIXTURE), "--splits", str(out)) == 0  # 整份重算
    assert run("--check", "--tiny", "--splits", str(out)) == 1  # 生成时给了训练池、核对时没给：内容不同
    text = out.read_text(encoding="utf-8")
    out.write_text(text.replace('"salt": "devcheck3_v1"', '"salt": "devcheck3_v2"', 1), encoding="utf-8")
    assert run("--check", "--splits", str(out)) == 1
    assert run("--build", "--tiny", "--out", str(out)) == 2  # 已存在且内容不同：不覆盖


# ---------------------------------------------------------------------------
# 入库的清单（骨架或已生成）
# ---------------------------------------------------------------------------


def test_manifest_in_repo():
    """入库的清单：待生成时是骨架（与按入库清单重算的逐字节相同）；服务器生成、管理员放进来之后，不要数据的核对通过。"""
    doc = EX.load_explore_splits()
    assert isinstance(doc, W2D.SplitsDoc) and doc.path == EX.EXPLORE_SPLITS_PATH and doc["tiny"] is False
    assert doc["groups"][EX.DEVCHECK3]["n_target"] == 500 and doc["groups"][EX.DEVCHECK3]["gate"] == "G-EX"
    assert doc["sources"]["w2_splits"]["sha256"] == hashlib.sha256((REPO / "configs" / "w2_splits.json").read_bytes()).hexdigest()
    assert EX.main(["--check"]) == 0
    if EX.status(doc) == EX.STATUS_BUILT:
        assert EX.check_doc(doc) == [] and doc["groups"][EX.DEVCHECK3]["n"] == 500
        assert doc["checks"]["train_pool"]["status"] == "核对过"  # 生成时要给 --train-data-root
        return
    assert EX.check_doc(doc) == ["pending"]
    assert EX.EXPLORE_SPLITS_PATH.read_text(encoding="utf-8") == dump_config(EX.build_skeleton())
    assert set(doc["splits"]) == {EX.split_name(g, EX.population_name(*p)) for g in EX.GROUPS for p in EX.POPULATIONS}
    for name, e in doc["splits"].items():
        assert e["status"] == "pending" and e["ids"] is None and e["build_command"] == EX.BUILD_COMMAND
        assert e["sealed"] is name.startswith("devcheck3_")
        with pytest.raises(W2D.SplitsError, match="还没生成"):
            W2D.w2_items("refseg", name, EX.tiny_paths(), splits=doc)
    with pytest.raises(EX.SplitsError, match="还没生成"):
        EX.group_ids(doc, EX.DEVCHECK3)


# ---------------------------------------------------------------------------
# 密封守卫
# ---------------------------------------------------------------------------


def test_guard_function():
    for s in ("devcheck3", "devcheck3_refcoco_unc_val", "devcheck3_refcoco+_unc_val", "devcheck3_refcocog_umd_val"):
        assert EX.is_sealed(s) and EX.needs_frozen(s)
        with pytest.raises(EX.SealedSplitError, match="D126"):
            EX.require_unsealed(s, None)
        for wrong in ("G-W2", "g-ex", "G-EX ", ""):
            with pytest.raises(EX.SealedSplitError, match="G-EX"):
                EX.require_unsealed(s, wrong)
        rec = EX.require_unsealed(s, "G-EX", now="2026-11-02T00:00:00+00:00")
        assert rec == {"gate": "G-EX", "decision": EX.SEAL_DECISION, "unsealed_utc": "2026-11-02T00:00:00+00:00"}
        assert EX.seal_identity(rec) == {"gate": "G-EX", "decision": EX.SEAL_DECISION}  # 身份里不含时间
    assert EX.require_unsealed("devcheck3", "G-EX")["unsealed_utc"].endswith("+00:00")
    for s in ("explore_val", "explore_val_refcoco_unc_val", "devcheck2", "devtune800", "test_refcoco_unc_testA"):
        assert not EX.is_sealed(s) and EX.require_unsealed(s, None) is None and not EX.needs_frozen(s)
    assert EX.is_sealed("renamed", {"sealed": True})  # 条目标了 sealed 也算
    with pytest.raises(EX.SealedSplitError):
        EX.require_unsealed("renamed", None, entry={"sealed": True})
    assert EX.seal_identity(None) is None


def test_w2_record_default_unchanged_and_explore_records_its_file():
    """run_w2.py 默认（configs/w2_splits.json）时运行身份的 W2 一节与 DEV-0067 之前相同：键、splits_config、没有 unseal。"""
    from src.data.records import Sample
    from src.w2 import runner as R

    doc = W2D.load_splits(W2D.W2_SPLITS_PATH)
    assert type(doc) is dict
    spec = R.W2RunSpec(model_key="m", family="qwen3.5", repo="x", revision="r", adapter_model_key="m", task="refseg", split="devcheck2",
                       setting="w2a", tiny=False, limit=None, image_policy={})
    samples = [Sample(id=f"refcoco_unc/val/{i}", task="refexp", images=("x.jpg",), image_keys=("coco:1",), phash=(None,), fields={}, gt={},
                      source="refcoco", split="val") for i in doc["splits"]["devcheck2"]["ids"]]
    rec = R.w2_record(spec, doc, samples)
    assert list(rec) == ["task", "split", "setting", "splits_config", "splits_sha256", "split_sha256", "split_n", "n_samples", "limit",
                         "sample_ids_sha256"]
    assert rec["splits_config"] == "configs/w2_splits.json" and R._splits_files(doc) == [W2D.W2_SPLITS_PATH]
    assert rec["splits_sha256"] == hashlib.sha256(W2D.W2_SPLITS_PATH.read_bytes()).hexdigest()
    ex = EX.load_explore_splits()
    assert R._splits_files(ex) == [EX.EXPLORE_SPLITS_PATH]
    sp = spec.__class__(**{**spec.__dict__, "split": "devcheck3_refcoco_unc_val", "unseal": {"gate": "G-EX", "decision": "d"}})
    fake = W2D.SplitsDoc(copy.deepcopy(dict(ex)))
    fake.path = EX.EXPLORE_SPLITS_PATH
    fake["splits"]["devcheck3_refcoco_unc_val"].update(n=1, sha256="s")
    rec = R.w2_record(sp, fake, samples[:1])
    assert rec["splits_config"] == "configs/explore_splits.json" and rec["unseal"] == {"gate": "G-EX", "decision": "d"}
    assert rec["splits_sha256"] == hashlib.sha256(dump_config(fake).encode("utf-8")).hexdigest()


def _run(args, tmp: Path, ok=True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, str(RUN_W2), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def _w2(task, split, setting, out, *extra, tiny=True):
    return (["--tiny"] if tiny else []) + ["--model", QWEN, "--task", task, "--split", split, "--setting", setting, "--out", str(out), *extra]


EXPLORE = ["--splits-file", "configs/explore_splits.json"]


def test_run_w2_refuses_sealed_without_the_gate(tmp_path):
    from tests._w2_draft import W2_DRAFT_PATH  # 未冻结的草稿

    for extra, msg in (([], "没有给"), (["--unseal-devcheck3", "G-W2"], "给的是 'G-W2'")):
        out = tmp_path / f"0001_x{len(extra)}"
        p = _run(_w2("refexp_box", "devcheck3_refcoco_unc_val", "baseline_e1", out, *EXPLORE, *extra), tmp_path, ok=False)
        assert p.returncode == 2 and "密封的 devcheck3" in p.stderr and msg in p.stderr and not out.exists()
    p = _run(_w2("refexp_box", "explore_val_refcoco_unc_val", "baseline_e1", tmp_path / "0002_y", *EXPLORE, "--unseal-devcheck3", "G-EX"),
             tmp_path, ok=False)
    assert p.returncode == 2 and "只配密封的 devcheck3" in p.stderr
    out = tmp_path / "0003_unfrozen"  # 真运行：devcheck3 也要冻结的配置
    p = _run(_w2("refseg", "devcheck3_refcoco_unc_val", "w1f", out, *EXPLORE, "--unseal-devcheck3", "G-EX", "--w2-config", str(W2_DRAFT_PATH),
                 "--data-root", str(tmp_path / "nodata"), tiny=False), tmp_path, ok=False)
    assert p.returncode == 2 and "协议未冻结" in p.stderr and not out.exists()


def test_run_w2_unsealed_records_gate_and_time(tmp_path):
    out = tmp_path / "out"
    _run(_w2("refexp_box", "devcheck3_refcoco_unc_val", "baseline_e1", out, *EXPLORE, "--unseal-devcheck3", "G-EX"), tmp_path)
    leaf = out / QWEN / "refexp_box" / "baseline_e1" / "devcheck3_refcoco_unc_val"
    cfg = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
    w2 = cfg["data"]["selection_stats"]["w2"]
    assert w2["unseal"] == {"gate": "G-EX", "decision": EX.SEAL_DECISION} and w2["split"] == "devcheck3_refcoco_unc_val"
    assert w2["split_sha256"] == EX.tiny_splits()["splits"]["devcheck3_refcoco_unc_val"]["sha256"]
    log = [json.loads(x) for x in (out / QWEN / "unseal_devcheck3.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == 1 and log[0]["gate"] == "G-EX" and log[0]["unsealed_utc"] and "--unseal-devcheck3" in log[0]["argv"]
    ids = [json.loads(x)["id"] for x in (leaf / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    want = [i for i in EX.group_ids(EX.tiny_splits(), EX.DEVCHECK3) if i.startswith("refcoco_unc/")]
    assert ids == want
    # 续跑：身份里不含时间，所以同一个目录能续跑；日志再追加一条
    _run(_w2("refexp_box", "devcheck3_refcoco_unc_val", "baseline_e1", out, *EXPLORE, "--unseal-devcheck3", "G-EX", "--resume"), tmp_path)
    assert len((out / QWEN / "unseal_devcheck3.jsonl").read_text(encoding="utf-8").splitlines()) == 2
