"""DEV-0011：统一构造入口 src/adapters/factory.make_adapter 与图像口径 src/adapters/image_policy。

两个脚本（冒烟、健康检查）改为调用它之后行为不变：它们的端到端测试在 test_adapter_*.py 与 test_health_run.py 里照旧通过；
这里另外核对分派、口径的类型检查、小模型参数、以及脚本里不再有各自的构造代码。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

from src.adapters import factory, gemma4, image_policy, internvl35, qwen35, registry
from src.adapters.factory import make_adapter
from src.adapters.image_policy import (
    E1_IMAGE_POLICIES,
    INTERNVL_NATIVE_TILES,
    FixedSize,
    PixelBudget,
    SoftTokenBudget,
    TilePolicy,
    parse_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILY_KEYS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
ADAPTER_CLASSES = {registry.FAMILY_QWEN35: qwen35.Qwen35Adapter, registry.FAMILY_GEMMA4: gemma4.Gemma4Adapter,
                   registry.FAMILY_INTERNVL35: internvl35.InternVL35Adapter}


def test_e1_policies_are_the_d28_3_values():
    assert E1_IMAGE_POLICIES == {
        registry.FAMILY_QWEN35: PixelBudget(300),
        registry.FAMILY_GEMMA4: SoftTokenBudget(280),
        registry.FAMILY_INTERNVL35: TilePolicy(crop_to_patches=False, max_patches=1),
    }
    assert INTERNVL_NATIVE_TILES == TilePolicy(True, 12)
    assert image_policy.E1_PIXEL_BUDGET_TOKENS == 300 and image_policy.E1_SUBSET_PIXEL_BUDGET_TOKENS == 1024
    assert qwen35.PIXELS_PER_LLM_TOKEN == image_policy.QWEN_PIXELS_PER_TOKEN == 32


@pytest.mark.parametrize("key", FAMILY_KEYS)
def test_tiny_default_policy_is_e1(key):
    ad = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True)
    family = registry.get_model(key).family
    assert type(ad) is ADAPTER_CLASSES[family]
    assert ad.image_policy == E1_IMAGE_POLICIES[family]
    assert ad.model_key == f"tiny({key})" and ad.dtype == torch.float32


def test_policy_type_is_checked_per_family():
    with pytest.raises(TypeError, match="PixelBudget"):
        make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=SoftTokenBudget(280), tiny=True)
    with pytest.raises(TypeError, match="SoftTokenBudget"):
        make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, image_policy=PixelBudget(300), tiny=True)
    with pytest.raises(TypeError, match="TilePolicy"):
        make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, image_policy=(False, 1), tiny=True)
    with pytest.raises(ValueError):  # Gemma 的档位由适配器校验
        make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, image_policy=SoftTokenBudget(100), tiny=True)
    with pytest.raises(KeyError):
        make_adapter("qwen3.5-1b", device="cpu", dtype=torch.float32, tiny=True)
    with pytest.raises(TypeError):
        make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.int64, tiny=True)


def test_family_kw_rules():
    with pytest.raises(TypeError, match="tie_word_embeddings"):
        make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32, tiny=True, tie_word_embeddings=False)
    with pytest.raises(TypeError, match="真模型"):  # 真模型不接受任何 family_kw，在下载权重之前就报错
        make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, seed=1)
    ad = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, image_policy=FixedSize(64, 96), tiny=True,
                      tie_word_embeddings=False)
    assert ad.tied_word_embeddings is False and ad.image_size == (64, 96)
    iv = make_adapter("internvl3.5-2b-hf", device="cpu", dtype=torch.float32, tiny=True)
    assert iv.tied_word_embeddings is False  # 与此前 health_check / 冒烟的默认相同


def test_init_std_reinitializes_deterministically():
    a = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05, seed=3)
    b = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True, init_std=0.05, seed=3)
    c = make_adapter("qwen3.5-0.8b", device="cpu", dtype=torch.float32, tiny=True)
    wa, wb, wc = (x.model.get_input_embeddings().weight for x in (a, b, c))
    assert torch.equal(wa, wb) and not torch.equal(wa, wc)
    assert abs(float(wa.detach().std()) - 0.05) < 0.005
    assert a.model.get_output_embeddings().weight is wa  # 共享关系不变
    with pytest.raises(ValueError):
        factory.reinit_parameters(c.model, std=0, seed=0)


def test_parse_policy():
    assert parse_policy("pixel_budget:300") == PixelBudget(300)
    assert parse_policy("fixed:480x640") == FixedSize(480, 640)
    assert parse_policy("soft_tokens:280") == SoftTokenBudget(280)
    assert parse_policy("tiles:single") == TilePolicy(False, 1)
    assert parse_policy("tiles:native") == INTERNVL_NATIVE_TILES
    assert parse_policy("tiles:true:6") == TilePolicy(True, 6)
    for bad in ("pixel_budget:x", "fixed:480", "tiles:maybe:3", "budget:300", "fixed:470x640"):
        with pytest.raises(ValueError):
            parse_policy(bad)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", REPO_ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scripts_construct_adapters_only_through_the_factory(monkeypatch):
    """冒烟与健康检查不再各写一份构造代码：源码里不出现小模型与加载的构造调用，构造经 make_adapter。"""
    monkeypatch.setattr(sys, "path", list(sys.path))
    forbidden = ("load_pinned_components(", "build_tiny_model(", "make_tiny_adapter(", ".load(")
    srcs = {
        "scripts/health_check.py": (REPO_ROOT / "scripts/health_check.py").read_text(),
        "scripts/smoke_adapter.py": (REPO_ROOT / "scripts/smoke_adapter.py").read_text(),
        "gemma4.smoke_setup": inspect.getsource(gemma4.smoke_setup),
        "internvl35._smoke_setup": inspect.getsource(internvl35._smoke_setup),
    }
    for name, src in srcs.items():
        assert not [f for f in forbidden if f in src], name
        assert "make_adapter(" in src, name


def test_health_check_vision_settings_are_unchanged():
    """健康检查报告里的 vision_settings_unused 与改用 factory 之前逐项相同。"""
    hc = _load_script("health_check")
    assert hc.vision_policy(registry.FAMILY_QWEN35, tiny=False).as_dict() == {"image_size": [480, 640]}
    assert hc.vision_policy(registry.FAMILY_GEMMA4, tiny=True).as_dict() == {"max_soft_tokens": 70}
    assert hc.vision_policy(registry.FAMILY_GEMMA4, tiny=False).as_dict() == {"max_soft_tokens": 280}
    assert hc.vision_policy(registry.FAMILY_INTERNVL35, tiny=False).as_dict() == dict(internvl35.SMOKE_POLICY)
