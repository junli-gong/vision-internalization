"""DEV-0023：E2 的数据接线（训练加载器的去重清单、留出、训练顺序、评测集）与配置（层的展开、--tiny 的覆盖）。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.e2.config import TASKS, curve_steps, expand_layers, load_config, parse_layer_spec, resolve_task
from src.e2.data import (
    E2Paths,
    batch_indices,
    check_dedupe_coverage,
    eval_dedupe_sets,
    group_key,
    load_eval_set,
    load_nyu_test,
    load_train_pool,
    split_pool,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_e2.py"


@pytest.fixture(scope="module")
def paths():
    return E2Paths.fixtures()


@pytest.fixture()
def no_dedupe_paths(tmp_path):
    """fixture 训练根的副本，删掉去重剔除清单。"""
    root = tmp_path / "root"
    shutil.copytree(E2Paths.fixtures().train_root, root)
    (root / "dedupe" / "train_exclusions.json").unlink()
    base = E2Paths.fixtures()
    return E2Paths(train_root=root, eval_root=base.eval_root, dev_subsets=base.dev_subsets)


# ---------------------------------------------------------------------------
# 硬约束 7：没有去重清单不训练
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_train_pool_requires_dedupe_exclusions(task, no_dedupe_paths):
    with pytest.raises(FileNotFoundError, match="硬约束 7"):
        load_train_pool(task, no_dedupe_paths)


def test_cli_fails_without_dedupe_exclusions(no_dedupe_paths, tmp_path):
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, str(SCRIPT), "--tiny", "--model", "qwen3.5-0.8b", "--task", "semseg",
                        "--data-root", str(no_dedupe_paths.train_root), "--eval-root", str(no_dedupe_paths.eval_root),
                        "--dev-subsets", str(no_dedupe_paths.dev_subsets), "--out", str(out)],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 2, p.stderr[-2000:]
    errs = list((out / "qwen3.5-0.8b" / "errors").glob("error_*.json"))
    assert errs and "硬约束 7" in json.loads(errs[0].read_text())["error"]
    assert not (out / "qwen3.5-0.8b" / "semseg").exists()  # 没开始就停下：叶目录都没建


def test_exclusions_are_applied(paths):
    ade = load_train_pool("semseg", paths)
    assert ade.stats["n_dedupe_excluded"] == 1 and len(ade) == 5
    assert "ade20k/train/ADE_train_00000006" not in {s.id for s in ade.records}
    nn = load_train_pool("normals", paths)
    assert [s.fields["idx"] for s in nn.records] == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# DEV-0024：去重清单的覆盖（小问题 A、B）与 NYU 深度训练集的场景复查（小问题 C）
# ---------------------------------------------------------------------------


def _copy_root(tmp_path, edit_excl=None):
    root = tmp_path / "root"
    shutil.copytree(E2Paths.fixtures().train_root, root)
    p = root / "dedupe" / "train_exclusions.json"
    if edit_excl is not None:
        doc = json.loads(p.read_text())
        p.write_text(json.dumps(edit_excl(doc)))
    base = E2Paths.fixtures()
    return E2Paths(train_root=root, eval_root=base.eval_root, dev_subsets=base.dev_subsets)


def test_fixture_exclusions_cover_every_task_and_train_set(paths):
    from src.data.train_dedupe import TRAIN_SETS, exclusions_meta

    meta = exclusions_meta(paths.train_root / "dedupe" / "train_exclusions.json")
    assert meta["legacy"] is False and meta["max_hamming"] == 8 and sorted(meta["train_sets"]) == sorted(TRAIN_SETS)
    assert {t: eval_dedupe_sets(t) for t in TASKS} == {"depth": ("nyuv2/test",), "semseg": ("ade20k/val",),
                                                       "normals": ("nyu_normals/test",), "refseg": ("refcoco/eval",)}
    for t in TASKS:  # 四个训练集都有键，四个任务的评测集都比较过
        cov = check_dedupe_coverage(t, load_train_pool(t, paths).stats["dedupe_file"], real_model=True)
        assert cov["required_eval_sets"] == list(eval_dedupe_sets(t)) and cov["max_hamming"] == 8 and cov["legacy"] is False


@pytest.mark.parametrize("task", TASKS)
def test_train_pool_refuses_a_missing_train_set_key(task, tmp_path):
    from src.e2.data import load_train_pool as ltp

    set_name = {"depth": "nyu_depth_train/train", "semseg": "ade20k/train", "normals": "nyu_normals/train",
                "refseg": "refcoco_train/train"}[task]

    def drop(doc):
        doc["sets"].pop(set_name)
        doc["train_sets"].remove(set_name)
        return doc

    with pytest.raises(ValueError, match="没有和评测集比较过"):
        ltp(task, _copy_root(tmp_path, drop))


def test_dedupe_coverage_checks(tmp_path):
    p = _copy_root(tmp_path, lambda d: {**d, "eval_sets": [s for s in d["eval_sets"] if s != "ade20k/val"]})
    pool = load_train_pool("semseg", p)
    with pytest.raises(ValueError, match="ade20k/val"):
        check_dedupe_coverage("semseg", pool.stats["dedupe_file"], real_model=False)
    assert check_dedupe_coverage("depth", pool.stats["dedupe_file"], real_model=True)["eval_sets"]  # 别的任务不受影响
    legacy = _copy_root(tmp_path / "old", lambda d: {"schema": 1, "created_utc": "t", "max_hamming": 8, "rule": "r", "sets": d["sets"]})
    f = load_train_pool("depth", legacy).stats["dedupe_file"]
    with pytest.raises(ValueError, match="旧格式"):
        check_dedupe_coverage("depth", f, real_model=True)
    assert check_dedupe_coverage("depth", f, real_model=False)["legacy"] is True  # --tiny 照读
    with pytest.raises(ValueError, match="硬约束 7"):
        check_dedupe_coverage("depth", None, real_model=True)
    assert check_dedupe_coverage("depth", None, real_model=False)["file_sha256"] is None


def test_task_run_refuses_legacy_exclusions_for_real_models(tmp_path):
    """真模型（spec.tiny = False）的 TaskRun.load_data 开跑前就拒绝旧格式的剔除清单。"""
    import torch

    from src.adapters.factory import make_adapter
    from src.e2.runner import RunSpec, TaskRun

    legacy = _copy_root(tmp_path, lambda d: {"schema": 1, "created_utc": "t", "max_hamming": 8, "rule": "r", "sets": d["sets"]})
    cfg = load_config()
    tc = resolve_task(cfg, "depth", tiny=True)
    ad = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, tiny=True)
    for tiny in (False, True):
        spec = RunSpec("internvl3.5-2b-hf", "internvl3.5", "r", "0" * 40, "tiny", tiny, {}, "fp32", "depth", "all", (0,), (),
                       ad.num_layers(), ad.hidden_size())
        run = TaskRun(ad, cfg, spec, tc, legacy)
        if tiny:
            run.load_data()
            assert run.dedupe["legacy"] is True
        else:
            with pytest.raises(ValueError, match="旧格式"):
                run.load_data()


def test_nyu_depth_train_rechecks_official_test_scenes(tmp_path):
    """小问题 C：load_nyu_depth_train 复查 D11-2，基础场景属于官方 test 场景、或清单的 base_scene 与场景名对不上，都报错。"""
    from src.data.land.nyu_depth_train import TEST_SCENES

    p = _copy_root(tmp_path)
    pairs = p.train_root / "nyu_depth_train" / "pairs_train.jsonl"
    rows = [json.loads(l) for l in pairs.read_text().splitlines()]
    assert not {r["base_scene"] for r in rows} & set(TEST_SCENES)
    assert "kitchen_0002" in TEST_SCENES
    bad = [dict(rows[0], scene="kitchen_0002a", base_scene="kitchen_0002")] + rows[1:]
    pairs.write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(ValueError, match="官方 test 场景"):
        load_train_pool("depth", p)
    pairs.write_text("".join(json.dumps(r) + "\n" for r in [dict(rows[0], base_scene="office_0003")] + rows[1:]))
    with pytest.raises(ValueError, match="base_scene"):
        load_train_pool("depth", p)


# ---------------------------------------------------------------------------
# 留出与训练顺序
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASKS)
def test_holdout_is_by_group_and_disjoint(task, paths):
    pool = load_train_pool(task, paths).records
    for hold in ({"min_items": 1, "n_val": 2}, {"min_items": 3, "n_val": 2}):
        sp = split_pool(task, pool, {"group": "x", **hold}, n_stats=3)
        tg = {group_key(task, s) for s in sp.train}
        vg = {group_key(task, s) for s in sp.val}
        assert tg.isdisjoint(set(sp.info["held_groups"])) and vg <= set(sp.info["held_groups"])
        assert sp.info["n_held_items"] >= hold["min_items"] and 1 <= len(sp.val) <= hold["n_val"]
        assert {s.id for s in sp.stats} <= {s.id for s in sp.train} and len(sp.stats) == min(3, len(sp.train))
        assert len(sp.train) + sp.info["n_held_items"] == len(pool)
        again = split_pool(task, pool, {"group": "x", **hold}, n_stats=3)
        assert [s.id for s in again.val] == [s.id for s in sp.val] and [s.id for s in again.stats] == [s.id for s in sp.stats]
    if task == "depth":  # 同一个基础场景的两个目录（bedroom_0001a/b）不跨划分
        assert {s.fields["base_scene"] for s in sp.val}.isdisjoint({s.fields["base_scene"] for s in sp.train})
    with pytest.raises(ValueError, match="训练集为空"):
        split_pool(task, pool, {"group": "x", "min_items": 10_000, "n_val": 2}, n_stats=1)


def test_batch_order_is_a_permutation_per_epoch_and_stateless():
    n, bs = 7, 3
    seq = [i for step in range(7) for i in batch_indices(n, step, bs, seed=1, task="depth")]
    for ep in range(3):
        assert sorted(seq[ep * n:(ep + 1) * n]) == list(range(n))
    assert batch_indices(n, 5, bs, seed=1, task="depth") == seq[15:18]
    assert seq != [i for step in range(7) for i in batch_indices(n, step, bs, seed=2, task="depth")]
    assert batch_indices(n, 0, bs, seed=1, task="depth") != batch_indices(n, 0, bs, seed=1, task="semseg")


# ---------------------------------------------------------------------------
# 评测集
# ---------------------------------------------------------------------------


def test_eval_sets_on_fixtures(paths):
    nyu = load_nyu_test(paths.eval_root)
    assert [s.id for s in nyu.records] == ["nyuv2/test/1"] and nyu.records[0].gt["key"] == "depth_raw"
    with np.load(nyu.records[0].gt["depth_npz"]) as z:
        assert z["depth_raw"].shape == (480, 640)
    assert len(load_eval_set("semseg", paths)) == 3
    assert [s.fields["idx"] for s in load_eval_set("normals", paths).records] == [1, 2]
    ref = load_eval_set("refseg", paths, eval_subset="eval")
    assert ref.stats["n_dev_excluded"] == 1 and len(ref) == 5  # E1 的开发子集（sent 101）去掉
    assert len(load_eval_set("refseg", paths, eval_subset="all")) == 6


def test_eval_set_size_is_checked_for_real_configs(paths):
    """正式配置（非 --tiny）的 n_eval_expected 与 fixture 的条数不符：拒绝开跑（不抽样）。"""
    import torch

    from src.adapters.factory import make_adapter
    from src.e2.runner import RunSpec, TaskRun

    cfg = load_config()
    tc = resolve_task(cfg, "semseg", tiny=False)
    tc["feature_norm"]["n_samples"] = 1
    tc["holdout"] = {**tc["holdout"], "min_items": 1, "n_val": 1}
    ad = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, tiny=True)
    spec = RunSpec("internvl3.5-2b-hf", "internvl3.5", "r", "0" * 40, "tiny", True, {}, "fp32", "semseg", "all", (0,), (),
                   ad.num_layers(), ad.hidden_size())
    with pytest.raises(ValueError, match="评测集有 3 条，配置写的是 2000"):
        TaskRun(ad, cfg, spec, tc, paths).load_data()


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_layer_specs():
    assert expand_layers("all", 4) == [0, 1, 2, 3, 4]
    nine_b = expand_layers("every 2", 32)
    assert nine_b == list(range(0, 33, 2)) and len(nine_b) == 17  # 9B：32 层，每隔一层（简报 §8.3）
    assert expand_layers("every 2", 35) == list(range(0, 35, 2)) + [35]  # 奇数层的模型补上最后一层
    assert expand_layers("every 2", 35, always_include_last=False)[-1] == 34
    assert parse_layer_spec(["every", "1"]) == "all" and parse_layer_spec("every 3") == "every 3"
    for bad in (["every"], ["every", "0"], ["some"], ["every", "x"]):
        with pytest.raises(ValueError):
            parse_layer_spec(bad)
    cfg = load_config()
    assert cfg.default_layer_spec("qwen3.5-9b") == "every 2" and cfg.default_layer_spec("qwen3.5-0.8b") == "all"
    assert curve_steps(3000, 250)[-1] == 3000 and len(curve_steps(3000, 250)) == 12 and curve_steps(5, 2) == [2, 4, 5]


def test_config_comment_numbers():
    """configs/e2_probes.json 注释里的数：步数 × 批相对各训练集的 epoch 数（数据规模取 DEV-0022 落地 SPEC 钉住的值）、
    probe_chunk_max_elements = 2^27。"""
    from src.data.land import ade20k, nyu_depth_train, nyu_normals

    cfg = load_config()
    assert cfg.doc["probe_chunk_max_elements"] == 2**27
    t = {k: resolve_task(cfg, k, tiny=False) for k in TASKS}
    seen = {k: t[k]["steps"] * t[k]["batch_size"] for k in TASKS}
    assert seen["depth"] == seen["semseg"] == seen["refseg"] == 24_000
    assert 1.0 <= seen["depth"] / nyu_depth_train.SPEC["expect"]["n_kept"] < 1.1  # 约 1 个 epoch
    assert 1.15 <= seen["semseg"] / ade20k.SPEC["expect"]["train"] < 1.25  # 约 1.2 个 epoch
    n_normals = nyu_normals.SPEC["expect"]["train"][0] - 50 - t["normals"]["holdout"]["min_items"]  # 去掉 E1 开发子集与留出
    assert 10.5 <= seen["normals"] / n_normals < 11.5  # 约 11 个 epoch


def test_resolved_task_parameters_and_tiny_overrides():
    cfg = load_config()
    real = {t: resolve_task(cfg, t, tiny=False) for t in TASKS}
    assert {t: (r["steps"], r["batch_size"]) for t, r in real.items()} == {
        "depth": (3000, 8), "semseg": (3000, 8), "normals": (1000, 8), "refseg": (3000, 8)}
    assert real["semseg"]["out_channels"] == 150 and real["normals"]["out_channels"] == 3
    assert real["depth"]["order"] == "image_first" and real["refseg"]["order"] == "instruction_first"
    assert real["depth"]["optim"]["lr"] == {"linear": 1e-3, "conv": 5e-4} and real["depth"]["feature_norm"]["n_samples"] == 256
    assert real["depth"]["probes"]["conv"]["hidden"] == 256 and real["depth"]["bootstrap_resamples"] == 1000
    tiny = resolve_task(cfg, "depth", tiny=True)
    assert (tiny["steps"], tiny["batch_size"], tiny["optim"]["warmup_steps"], tiny["n_eval_expected"]) == (4, 2, 1, None)
    assert tiny["optim"]["lr"] == real["depth"]["optim"]["lr"] and tiny["loss"] == real["depth"]["loss"]
