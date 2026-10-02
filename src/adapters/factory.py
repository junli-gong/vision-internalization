"""统一构造入口（DEV-0011）：注册键 → 对应家族的适配器。脚本（冒烟、健康检查、批处理核对）与 E1 框架都从这里构造，
按家族分派只在这一处。

    make_adapter(key, *, device, dtype, image_policy=None, tiny=False, **family_kw)

- key：注册键（src/adapters/registry.py）。真模型按固定 revision 加载（load）；tiny=True 时构造小型随机模型（只下载 config、
  tokenizer、处理器、模板等小文件，不下载权重），config 取 key 的固定 revision，model_key 记为 "tiny(<key>)"。
- dtype：torch 浮点 dtype（真模型的加载 dtype；小模型的参数 dtype）。
- image_policy：src/adapters/image_policy.py 的策略对象，必须是该家族接受的类型（Qwen：FixedSize / PixelBudget；Gemma：
  SoftTokenBudget；InternVL：TilePolicy），类型不对就报 TypeError。不给时取 E1 冻结的口径 E1_IMAGE_POLICIES（D28-3）。
- family_kw：只给小模型用（tiny=True）：Qwen 与 InternVL 接受 tie_word_embeddings（默认分别为 True / False，与此前各脚本相同），
  三个家族都接受 seed（默认 0）与 init_std（默认 None = 保持 HF 的随机初始化）。真模型不接受任何 family_kw；不认识的键报 TypeError。
  init_std（批处理核对用）：把全部二维及以上的浮点参数重新取成 N(0, init_std²)（种子 seed）。HF 默认初始化下小模型的 greedy 输出
  几乎与输入无关（Qwen 小模型每步都生成同一个换行 token），批量与逐条"逐个相同"就成了空话；取 0.05 左右，输出随指令与图变化。
- 构造方式记在适配器的 build_spec 上（DEV-0020）：{key, tiny, family_kw, image_policy}，save_trained 写进 manifest，
  load_trained(directory, device=…) 按它重建同一个 base（真模型按固定 revision 加载；小模型按同样的种子重建），再恢复检查点。
"""

from __future__ import annotations

from typing import Any

import torch

from src.adapters import checkpoint, registry
from src.adapters.base import ModelAdapter
from src.adapters.image_policy import (
    E1_IMAGE_POLICIES,
    FixedSize,
    ImagePolicy,
    SoftTokenBudget,
    TilePolicy,
    check_policy,
    parse_policy,
)

# 小模型接受的 family_kw 与默认值（与此前 scripts/smoke_adapter.py、scripts/health_check.py 里各自的写法相同）
_TINY_KW_DEFAULTS: dict[str, dict[str, Any]] = {
    registry.FAMILY_QWEN35: {"tie_word_embeddings": True, "seed": 0, "init_std": None},
    registry.FAMILY_GEMMA4: {"seed": 0, "init_std": None},
    registry.FAMILY_INTERNVL35: {"tie_word_embeddings": False, "seed": 0, "init_std": None},
}


def default_image_policy(family: str) -> ImagePolicy:
    """不给 image_policy 时用的口径：E1 冻结的那一套（D28-3）。"""
    try:
        return E1_IMAGE_POLICIES[family]
    except KeyError:
        raise ValueError(f"未知家族 {family!r}") from None


def reinit_parameters(model: Any, *, std: float, seed: int) -> None:
    """小模型专用：全部二维及以上的浮点参数按 named_parameters 的顺序重取 N(0, std²)（CPU 生成器，种子 seed）。
    共享的参数（tied lm_head）只出现一次，共享关系不变。"""
    if isinstance(std, bool) or not isinstance(std, (int, float)) or not std > 0:
        raise ValueError(f"init_std 必须是正数，收到 {std!r}")
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.is_floating_point() and p.dim() >= 2:
                p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) * float(std)).to(p.dtype))


def make_adapter(key: str, *, device: str | torch.device, dtype: torch.dtype, image_policy: ImagePolicy | None = None,
                 tiny: bool = False, **family_kw: Any) -> ModelAdapter:
    """按注册键构造适配器并加载模型（tiny=True 时挂小型随机模型）。规则见模块说明。"""
    entry = registry.get_model(key)
    family = entry.family
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError(f"dtype 必须是浮点 torch.dtype，收到 {dtype!r}")
    if not isinstance(tiny, bool):
        raise TypeError(f"tiny 必须是 bool，收到 {tiny!r}")
    policy = check_policy(family, default_image_policy(family) if image_policy is None else image_policy)
    kw: dict[str, Any] = {}
    if tiny:
        unknown = sorted(set(family_kw) - set(_TINY_KW_DEFAULTS[family]))
        if unknown:
            raise TypeError(f"{family} 的小模型不接受 {unknown}；可用：{sorted(_TINY_KW_DEFAULTS[family])}")
        kw = {**_TINY_KW_DEFAULTS[family], **family_kw}
    elif family_kw:
        raise TypeError(f"真模型不接受 family_kw：{sorted(family_kw)}")

    def finish(model: Any) -> Any:
        if kw.get("init_std") is not None:
            reinit_parameters(model, std=kw["init_std"], seed=kw["seed"])
        return model

    spec = {"key": key, "tiny": tiny, "family_kw": dict(kw), "image_policy": checkpoint.policy_spec(policy)}

    def done(ad: ModelAdapter) -> ModelAdapter:
        ad.build_spec = spec  # DEV-0020：检查点重建 base 用
        return ad

    if family == registry.FAMILY_QWEN35:
        from src.adapters import qwen35

        size_kw: dict[str, Any] = ({"image_size": (policy.h, policy.w)} if isinstance(policy, FixedSize)
                                   else {"image_policy": policy})
        if tiny:
            comps = qwen35.load_pinned_components(key)
            model = finish(qwen35.build_tiny_model(comps[0], tie_word_embeddings=kw["tie_word_embeddings"], dtype=dtype,
                                                   seed=kw["seed"]))
            return done(qwen35.make_tiny_adapter(comps, **size_kw, device=device, model=model, source_key=key))
        ad = qwen35.Qwen35Adapter(device=device, **size_kw)
        ad.load(key, dtype)
        return done(ad)

    if family == registry.FAMILY_GEMMA4:
        from src.adapters import gemma4

        assert isinstance(policy, SoftTokenBudget)
        if tiny:
            comps = gemma4.load_pinned_components(key)
            model = finish(gemma4.build_tiny_model(comps.config, dtype=dtype, seed=kw["seed"],
                                                   generation_config=comps.generation_config))
            return done(gemma4.make_tiny_adapter(comps, max_soft_tokens=policy.max_soft_tokens, device=device,
                                                 model=model, source_key=key))
        ad = gemma4.Gemma4Adapter(device=device, max_soft_tokens=policy.max_soft_tokens)
        ad.load(key, dtype)
        return done(ad)

    if family == registry.FAMILY_INTERNVL35:
        from src.adapters import internvl35

        assert isinstance(policy, TilePolicy)
        tile_kw = {"crop_to_patches": policy.crop_to_patches, "max_patches": policy.max_patches}
        if tiny:
            comps = internvl35.load_pinned_components(key)
            model = finish(internvl35.build_tiny_model(comps[0], tie_word_embeddings=kw["tie_word_embeddings"],
                                                       dtype=dtype, seed=kw["seed"]))
            return done(internvl35.make_tiny_adapter(comps, **tile_kw, device=device, model=model, source_key=key))
        ad = internvl35.InternVL35Adapter(device=device, **tile_kw)
        ad.load(key, dtype)
        return done(ad)

    raise ValueError(f"make_adapter 还不支持 {family} 家族（{key}）")  # 注册表新增家族时在这里加一支


def load_trained(directory: Any, *, device: str | torch.device, dtype: torch.dtype | None = None,
                 merge: bool = True) -> tuple[ModelAdapter, "checkpoint.TrainedCheckpoint"]:
    """按检查点 manifest 里的 build_spec 重建 base（真模型按固定 revision 加载，小模型按同样的种子与 init_std 重建；dtype 不给时取
    manifest 记的 dtype），再调 adapter.load_trained（restore_tokens → 挂 LoRA → merge=True 时在内存里合并）。返回 (适配器, 检查点内容)。"""
    man = checkpoint.read_manifest(directory)
    spec = man.get("build_spec")
    if not spec:
        raise checkpoint.CheckpointError("检查点没有记录构造方式（适配器不是 make_adapter 构造的）：请自己构造同一个 base，"
                                         "再调 adapter.load_trained")
    dt = dtype if dtype is not None else getattr(torch, man["dtype"])
    ad = make_adapter(spec["key"], device=device, dtype=dt, image_policy=parse_policy(spec["image_policy"]),
                      tiny=bool(spec["tiny"]), **dict(spec["family_kw"]))
    return ad, ad.load_trained(directory, merge=merge)


__all__ = ["default_image_policy", "load_trained", "make_adapter", "reinit_parameters"]
