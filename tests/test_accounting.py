import json

import pytest

from src.accounting import (
    COMPUTE_JSON_FIELDS,
    COMPUTE_SCHEMA_VERSION,
    Compute,
    merge,
    read_compute_json,
    write_compute_json,
)
from src.adapters.base import GenerationResult


def _gen(n_gen, n_prompt, n_vis, t):
    return GenerationResult("x" * n_gen, tuple(range(n_gen)), n_gen, n_prompt, n_vis, t)


def test_call_and_round_counting():
    # 样本 1：叠加反馈 2 轮，第 2 轮分 3 块
    s1 = (
        Compute.from_generation(_gen(5, 100, 64, 0.5), new_round=True)
        + Compute.from_generation(_gen(3, 120, 16, 0.2), new_round=True)
        + Compute.from_generation(_gen(4, 121, 16, 0.3), new_round=False)
        + Compute.from_generation(_gen(2, 119, 16, 0.1), new_round=False)
    ).as_sample(wall_time_s=1.2)
    assert (s1.n_samples, s1.n_calls, s1.n_rounds) == (1, 4, 2)
    assert (s1.generated_tokens, s1.prompt_tokens, s1.visual_tokens) == (14, 460, 112)
    assert s1.call_time_s == pytest.approx(1.1) and s1.wall_time_s == 1.2
    # 样本 2：单轮
    s2 = Compute.call(generated_tokens=7, prompt_tokens=90, visual_tokens=64, call_time_s=0.4, new_round=True).as_sample(
        wall_time_s=0.45
    )
    total = merge([s1, s2])
    assert (total.n_samples, total.n_calls, total.n_rounds, total.generated_tokens) == (2, 5, 3, 21)
    assert total.per_sample_mean()["n_calls"] == 2.5
    assert total.per_sample_mean()["visual_tokens"] == 88.0
    assert total.per_sample_mean()["wall_time_s"] == pytest.approx(0.825)
    assert merge([]) == Compute()


def test_concurrent_tiles_call_time_vs_wall_time():
    """addendum-1 第 8 条：4 个块并发、各 2 s → 调用耗时 8 s，墙钟约 2 s。"""
    tiles = [Compute.call(generated_tokens=10, prompt_tokens=50, visual_tokens=16, call_time_s=2.0, new_round=(k == 0)) for k in range(4)]
    s = merge(tiles).as_sample(wall_time_s=2.05)
    assert s.call_time_s == 8.0 and s.wall_time_s == 2.05
    assert (s.n_calls, s.n_rounds) == (4, 1)


def test_wall_time_only_at_sample_level_and_validation():
    c = Compute.call(generated_tokens=1, prompt_tokens=1, visual_tokens=0, call_time_s=0.3, new_round=True)
    assert c.wall_time_s == 0.0
    with pytest.raises(TypeError):
        c.as_sample()  # 墙钟必填
    s = c.as_sample(wall_time_s=0.3)
    with pytest.raises(ValueError):
        s.as_sample(wall_time_s=0.3)
    with pytest.raises(ValueError):
        Compute(wall_time_s=1.0)  # 调用级记录不能带墙钟
    with pytest.raises(ValueError):
        Compute(generated_tokens=-1)
    with pytest.raises(ValueError):
        Compute(n_calls=1.5)
    with pytest.raises(ValueError):
        Compute(call_time_s=-0.1)
    assert Compute().per_sample_mean() is None


def test_compute_json_fixed_fields_roundtrip(tmp_path):
    c = merge(
        [
            Compute.call(generated_tokens=g, prompt_tokens=10 * g, visual_tokens=3, call_time_s=0.25, new_round=True).as_sample(
                wall_time_s=0.3
            )
            for g in (1, 2, 6)
        ]
    )
    p = tmp_path / "compute.json"
    write_compute_json(p, c)
    d = json.loads(p.read_text())
    assert tuple(d) == COMPUTE_JSON_FIELDS
    assert COMPUTE_SCHEMA_VERSION == 2 and d["schema_version"] == 2
    assert d["generated_tokens"] == 9 and d["n_samples"] == 3
    assert d["call_time_s"] == pytest.approx(0.75) and d["wall_time_s"] == pytest.approx(0.9)
    assert read_compute_json(p) == c
    d["schema_version"] = 1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        read_compute_json(p)
    d["schema_version"] = COMPUTE_SCHEMA_VERSION
    d["extra"] = 1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        read_compute_json(p)
