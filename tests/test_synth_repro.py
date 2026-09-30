"""可复现：样本 = 纯函数(任务, 划分, 序号, 生成器版本)。

- 同一输入两次生成逐字节相同；不同序号不同；与全局随机状态、字符串哈希随机化无关（子进程换 PYTHONHASHSEED）。
- 钉住的摘要（开发机 arm64 macOS 上生成）：
  * 在开发机上红 = 生成器的输出变了：把 src/synth/constants.py 的 SYNTH_GENERATOR_VERSION 加一，再更新这里的摘要；
  * 只在服务器（x86_64 Linux）上红 = 跨机器不能逐位复现：不要改摘要，上报管理员（d 依赖 Chipmunk 的编译浮点行为，
    c、e 只依赖 IEEE 初等运算与量化，预期两台机器相同）。
"""

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.synth import generate

REPO = Path(__file__).resolve().parents[1]

PINNED_CE = {
    ("occlusion", "train", 0): "f8336d559de7d67752d1881a3840dbaad7f2687058f87e24709b8f64d548a0a0",
    ("occlusion", "val", 1): "20f7be078a60955872ba3fc08e0f2fbb7c72f0d925cfc5cff2124316c13aa05c",
    ("occlusion", "test", 2): "019931825e2b2a7957c0d69dd3d46fc9931c115695fdfff4da3abb974f5cc246",
    ("imagery", "train", 0): "d61cc558399a54c8eb9fc01f1324c0cde38628184a5094987226caba447d9ad8",
    ("imagery", "val", 1): "770c74a1b120b5b3d5902959d19ca1c42df425839a32aa1035dbfa4509f76b42",
    ("imagery", "test", 2): "19303b5b3aab28b1f15962950aeb31ed5874c3ba3a9d2b92613dba29c4617f6a",
}
PINNED_D = {
    ("physics", "train", 0): "ad7f37cf80412f8b7bdf3b605de2d096e25250c7236b013e3bf3eb214f82bdf7",
    ("physics", "val", 1): "f5fd8fb9896b1c6f9ee873dd2305919adeedb697ed7c189cef4f79d88caf0aa1",
    ("physics", "test", 2): "7628065e65af781b85e2fbf1fa18019fc6f15020e36a5f7d9935cb456444b148",
    ("physics", "train", 3): "6b3df26c0da8a164b71824c3ed46a34b26564434ffa91631071b6930a4c0f77c",  # k = 40
}
CASES = sorted(PINNED_CE) + sorted(PINNED_D)


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
    ref = [generate(t, s, i).digest() for t, s, i in CASES[:3]]
    np.random.seed(12345)
    random.seed(999)
    np.random.random(17)
    assert [generate(t, s, i).digest() for t, s, i in CASES[:3]] == ref


def test_fresh_processes_with_different_hash_seeds_agree():
    code = (
        "from src.synth import generate\n"
        f"for t, s, i in {CASES!r}:\n"
        "    print(generate(t, s, i).digest())\n"
    )
    outs = []
    for seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(REPO))
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.split())
    here = [generate(t, s, i).digest() for t, s, i in CASES]
    assert outs[0] == outs[1] == here


def test_pinned_digests_occlusion_and_imagery():
    got = {c: generate(*c).digest() for c in PINNED_CE}
    assert got == PINNED_CE


def test_pinned_digests_physics():
    got = {c: generate(*c).digest() for c in PINNED_D}
    assert got == PINNED_D
