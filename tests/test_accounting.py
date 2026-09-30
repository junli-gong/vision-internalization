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
    ).as_sample()
    assert (s1.n_samples, s1.n_calls, s1.n_rounds) == (1, 4, 2)
    assert (s1.generated_tokens, s1.prompt_tokens, s1.visual_tokens) == (14, 460, 112)
    assert s1.wall_time_s == pytest.approx(1.1)
    # 样本 2：单轮
    s2 = Compute.call(generated_tokens=7, prompt_tokens=90, visual_tokens=64, wall_time_s=0.4, new_round=True).as_sample()
    total = merge([s1, s2])
    assert (total.n_samples, total.n_calls, total.n_rounds, total.generated_tokens) == (2, 5, 3, 21)
    assert total.per_sample_mean()["n_calls"] == 2.5
    assert total.per_sample_mean()["visual_tokens"] == 88.0
    assert merge([]) == Compute()


def test_as_sample_only_once_and_validation():
    s = Compute.call(generated_tokens=1, prompt_tokens=1, visual_tokens=0, wall_time_s=0.0, new_round=True).as_sample()
    with pytest.raises(ValueError):
        s.as_sample()
    with pytest.raises(ValueError):
        Compute(generated_tokens=-1)
    with pytest.raises(ValueError):
        Compute(n_calls=1.5)
    with pytest.raises(ValueError):
        Compute(wall_time_s=-0.1)
    assert Compute().per_sample_mean() is None


def test_compute_json_fixed_fields_roundtrip(tmp_path):
    c = merge(
        [Compute.call(generated_tokens=g, prompt_tokens=10 * g, visual_tokens=3, wall_time_s=0.25, new_round=True).as_sample() for g in (1, 2, 6)]
    )
    p = tmp_path / "compute.json"
    write_compute_json(p, c)
    d = json.loads(p.read_text())
    assert tuple(d) == COMPUTE_JSON_FIELDS
    assert d["schema_version"] == COMPUTE_SCHEMA_VERSION
    assert d["generated_tokens"] == 9 and d["n_samples"] == 3
    assert read_compute_json(p) == c
    d["schema_version"] = COMPUTE_SCHEMA_VERSION + 1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        read_compute_json(p)
    d["schema_version"] = COMPUTE_SCHEMA_VERSION
    d["extra"] = 1
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        read_compute_json(p)
