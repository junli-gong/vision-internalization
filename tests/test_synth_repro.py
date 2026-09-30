"""可复现：样本 = 纯函数(任务, 划分, 序号, 生成器版本)，以及跨机器逐位复现的核对（D24，生成器版本 2）。

同机
- 同一输入两次生成逐字节相同；不同序号不同；与全局随机状态、字符串哈希随机化无关（子进程换 PYTHONHASHSEED）。
- d 的 60 个钉住 case 在本进程与一个新进程里各生成一次，逐个相同（任何平台都跑，包括开发机）。
- 把 math / numpy 的三角函数、np.mean、pycocotools 换成抛异常的桩，再生成样本：摘要不变，
  证明这些随平台变化的实现不在数据路径上（开发机上唯一能做的跨平台检查，见 lessons "跨机器差异"）。

钉住的摘要：tests/data/synth_digests_v2.json（由 scripts/synth_pin_digests.py 生成），
三个任务 × 三个划分 × 序号 0–19，档位按默认轮转。
- c（occlusion）、e（imagery），共 120 个：在任何平台上都必须逐位相同。
- d（physics），60 个：只在规范平台（src.synth.constants.CANONICAL_PLATFORM = x86_64 Linux）上比对，
  其他平台跳过（D24：Chipmunk 的浮点行为随平台不同，d 的数据只在规范平台上生成）。
  文件里 d 的 status 为 pending_canonical 时，清单是开发机的占位值：规范平台上不同记为 xfail，
  由管理员在下一个 RUN 之后用服务器上的值替换（常量级修改）；status 为 canonical 以后，不同即失败。

失败时的处理
- c、e 在开发机上不同：生成器的输出变了。确认改动是有意的，就把 SYNTH_GENERATOR_VERSION 加一，
  再运行 `python scripts/synth_pin_digests.py write` 重写整个文件（d 的状态随之回到 pending_canonical，等服务器重新填入）。
- c、e 只在别的平台上不同（开发机上相同）：跨平台不能逐位复现，是缺陷。不要改摘要，上报管理员；
  用 `python scripts/synth_pin_digests.py check` 列出不同的 case。
- d 在规范平台上不同（status 为 canonical）：先看 generated_on 里的 pymunk 版本与本机是否相同。
  版本不同，是依赖变了；版本相同，是生成器的输出变了。两种都上报管理员，不要自行改摘要。
- 生成器版本与文件里的 generator_version 不同：改了版本号却没有重新钉住，按第一条处理。
"""

import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.synth import generate
from src.synth import dmath
from src.synth import shapes as S
from src.synth.constants import CANONICAL_PLATFORM, SPLITS, SYNTH_GENERATOR_VERSION, TASKS

REPO = Path(__file__).resolve().parents[1]
DIGEST_FILE = REPO / "tests" / "data" / "synth_digests_v2.json"
PINNED = json.loads(DIGEST_FILE.read_text(encoding="utf-8"))
INDICES_PER_SPLIT = 20  # 任务书 DEV-0009 §5
ANY_PLATFORM_TASKS = ("occlusion", "imagery")


def _cases(task: str) -> list[tuple[str, str, int]]:
    return [(task, key.split("/")[0], int(key.split("/")[1])) for key in PINNED["tasks"][task]["digests"]]


def _digests(cases) -> dict[str, str]:
    return {f"{s}/{i}": generate(t, s, i).digest() for t, s, i in cases}


def _mismatches(task: str) -> list[str]:
    want = PINNED["tasks"][task]["digests"]
    got = _digests(_cases(task))
    return [k for k in want if got[k] != want[k]]


def _on_canonical_platform() -> bool:
    return (platform.machine(), platform.system()) == CANONICAL_PLATFORM


# 子进程与桩测试用的小清单：每个任务每个划分的前 3 个序号
SMALL = [(t, s, i) for t in TASKS for s in SPLITS for i in range(3)]


# ---------------------------------------------------------------------------
# 同机
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["occlusion", "physics", "imagery"])
def test_same_input_gives_identical_bytes(task):
    for split, index in (("train", 5), ("val", 6), ("test", 7)):
        a, b = generate(task, split, index), generate(task, split, index)
        assert a.digest() == b.digest()
        assert a.instruction == b.instruction and a.meta == b.meta
        assert a.target_mask.tobytes() == b.target_mask.tobytes()
        assert a.object_masks.tobytes() == b.object_masks.tobytes()
        if a.image is not None:
            assert a.image.tobytes() == b.image.tobytes()


@pytest.mark.parametrize("task", ["occlusion", "physics", "imagery"])
def test_different_index_gives_different_sample(task):
    digests = {generate(task, "train", i).digest() for i in range(8)}
    assert len(digests) == 8
    a, b = generate(task, "train", 20, bin=0), generate(task, "train", 21, bin=0)
    assert not np.array_equal(a.target_mask, b.target_mask)


def test_independent_of_global_random_state():
    ref = [generate(t, s, i).digest() for t, s, i in SMALL[:3]]
    np.random.seed(12345)
    random.seed(999)
    np.random.random(17)
    assert [generate(t, s, i).digest() for t, s, i in SMALL[:3]] == ref


def _digests_in_subprocess(cases, hash_seed: str) -> list[str]:
    code = "from src.synth import generate\n" f"for t, s, i in {cases!r}:\n" "    print(generate(t, s, i).digest())\n"
    env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_fresh_processes_with_different_hash_seeds_agree():
    outs = [_digests_in_subprocess(SMALL, seed) for seed in ("0", "4242")]
    here = [generate(t, s, i).digest() for t, s, i in SMALL]
    assert outs[0] == outs[1] == here


def test_physics_same_machine_determinism():
    """任务书 DEV-0009 §5：d 在同一台机器上两次生成相同（本进程一次、新进程一次；任何平台都跑）。"""
    cases = _cases("physics")
    here = [generate(t, s, i).digest() for t, s, i in cases]
    assert _digests_in_subprocess(cases, "7") == here


def test_generation_does_not_touch_platform_dependent_math(monkeypatch):
    """把随平台变化的实现换成抛异常的桩，清空缓存后重新生成：c、e 的摘要仍等于钉住值，d 的摘要不变。

    桩：math 与 numpy 的三角函数、np.mean / np.average、pycocotools（src.executors.raster 的多边形光栅化也走它）。
    """
    cases_ce = [c for c in SMALL if c[0] in ANY_PLATFORM_TASKS]
    cases_d = [c for c in SMALL if c[0] == "physics"]
    want_d = [generate(t, s, i).digest() for t, s, i in cases_d]

    def poison(name):
        def f(*args, **kwargs):
            raise AssertionError(f"{name} 不应出现在 E5 生成器的数据路径上")

        return f

    for name in ("sin", "cos", "tan", "asin", "acos", "atan", "atan2", "hypot"):
        monkeypatch.setattr(math, name, poison(f"math.{name}"))
    for name in ("sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "hypot", "mean", "average"):
        monkeypatch.setattr(np, name, poison(f"numpy.{name}"))
    from pycocotools import mask as mask_utils

    import src.executors.raster as exec_raster

    for name in ("frPyObjects", "merge", "decode", "encode"):
        monkeypatch.setattr(mask_utils, name, poison(f"pycocotools.mask.{name}"))
    monkeypatch.setattr(exec_raster, "polygons_to_mask", poison("src.executors.raster.polygons_to_mask"))
    dmath._rot_cs_cached.cache_clear()  # 缓存里的值要在桩下重新算一遍
    S._unit_polygon_cached.cache_clear()
    try:
        got_ce = {(t, s, i): generate(t, s, i).digest() for t, s, i in cases_ce}
        got_d = [generate(t, s, i).digest() for t, s, i in cases_d]
    finally:
        dmath._rot_cs_cached.cache_clear()
        S._unit_polygon_cached.cache_clear()
    for t, s, i in cases_ce:
        assert got_ce[(t, s, i)] == PINNED["tasks"][t]["digests"][f"{s}/{i}"], (t, s, i)
    assert got_d == want_d


# ---------------------------------------------------------------------------
# 钉住的摘要
# ---------------------------------------------------------------------------


def test_digest_file_matches_generator_version_and_coverage():
    assert PINNED["generator_version"] == SYNTH_GENERATOR_VERSION, "改了生成器版本却没有重新钉住：见模块说明"
    keys = [f"{s}/{i}" for s in SPLITS for i in range(INDICES_PER_SPLIT)]
    assert set(PINNED["tasks"]) == set(TASKS)
    for task in TASKS:
        block = PINNED["tasks"][task]
        assert list(block["digests"]) == keys, task
        assert block["compare_on"] == ("any_platform" if task in ANY_PLATFORM_TASKS else "canonical_platform")
    d = PINNED["tasks"]["physics"]
    assert d["status"] in ("canonical", "pending_canonical")
    assert (d["canonical_platform"]["machine"], d["canonical_platform"]["system"]) == CANONICAL_PLATFORM


@pytest.mark.parametrize("task", ANY_PLATFORM_TASKS)
def test_pinned_digests_any_platform(task):
    bad = _mismatches(task)
    assert not bad, (
        f"{task}：{len(bad)}/{len(PINNED['tasks'][task]['digests'])} 个钉住的摘要不同（{platform.machine()} {platform.system()}）："
        f"{bad[:12]}。处理见 tests/test_synth_repro.py 模块说明"
    )


def test_pinned_digests_physics_on_canonical_platform():
    block = PINNED["tasks"]["physics"]
    if not _on_canonical_platform():
        pytest.skip(
            f"D24：d 依赖 pymunk（Chipmunk）的浮点行为，钉住的摘要只在规范平台 {CANONICAL_PLATFORM} 上比对；"
            f"本机是 {(platform.machine(), platform.system())}。同机确定性见 test_physics_same_machine_determinism"
        )
    import pymunk

    bad = _mismatches("physics")
    recorded = block["generated_on"].get("pymunk")
    note = f"文件里的 pymunk {recorded}，本机 {pymunk.version}"
    if block["status"] != "canonical":
        if bad:
            pytest.xfail(
                f"d 清单待服务器填入（status={block['status']}，D24）：{len(bad)}/60 与占位值不同（{note}）。"
                "在规范平台上运行 `python scripts/synth_pin_digests.py write --only physics` 填入"
            )
        return
    assert not bad, f"d：{len(bad)}/60 个钉住的摘要在规范平台上不同（{note}）：{bad[:12]}。处理见模块说明"
