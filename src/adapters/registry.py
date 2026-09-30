"""模型注册表：模型键 → (家族, 仓库, revision) 的唯一登记处（D9-1、D9-2）。

- 变体按 D9-1：Qwen3.5 用后训练版，Gemma 4 用 `-it`，InternVL3.5 用 `-HF`。
- revision 抄自 `docs/model_notes.md` §2（RUN-0001 实测登记的 sha），tests/test_adapter_qwen35.py
  逐条与该表比对。换 revision 要新开 D 条（D9-2）。
- 下载与加载一律带 `revision=`：适配器只接受这里登记的键，不接受裸仓库名。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

FAMILY_QWEN35 = "qwen3.5"
FAMILY_GEMMA4 = "gemma4"
FAMILY_INTERNVL35 = "internvl3.5"
FAMILIES: tuple[str, ...] = (FAMILY_QWEN35, FAMILY_GEMMA4, FAMILY_INTERNVL35)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelEntry:
    """一个登记的模型：键、家族、HF 仓库、固定的 revision（完整 commit sha）。"""

    key: str
    family: str
    repo: str
    revision: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.key}: 未知家族 {self.family!r}，应为 {FAMILIES} 之一")
        if not _SHA_RE.match(self.revision):
            raise ValueError(f"{self.key}: revision 必须是 40 位小写十六进制 commit sha，收到 {self.revision!r}")


_ENTRIES: tuple[ModelEntry, ...] = (
    ModelEntry("qwen3.5-0.8b", FAMILY_QWEN35, "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    ModelEntry("qwen3.5-2b", FAMILY_QWEN35, "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ModelEntry("qwen3.5-4b", FAMILY_QWEN35, "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ModelEntry("qwen3.5-9b", FAMILY_QWEN35, "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ModelEntry("gemma-4-e2b-it", FAMILY_GEMMA4, "google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    ModelEntry("gemma-4-e4b-it", FAMILY_GEMMA4, "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
    ModelEntry(
        "internvl3.5-2b-hf", FAMILY_INTERNVL35, "OpenGVLab/InternVL3_5-2B-HF", "3f301ffcf3dcbb47893afae6650ea3e78d96fb6d"
    ),
    ModelEntry(
        "internvl3.5-4b-hf", FAMILY_INTERNVL35, "OpenGVLab/InternVL3_5-4B-HF", "6bd4487402110ef9889ba50eb7aefeb302526fed"
    ),
)

MODELS: Mapping[str, ModelEntry] = MappingProxyType({e.key: e for e in _ENTRIES})
if len(MODELS) != len(_ENTRIES):
    raise RuntimeError("注册表里有重复的模型键")


def get_model(key: str) -> ModelEntry:
    """按键取登记项；未登记的键报错并列出全部已登记的键。"""
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"未登记的模型键 {key!r}；已登记：{sorted(MODELS)}") from None


def keys_for_family(family: str) -> tuple[str, ...]:
    """某家族的全部登记键，按登记顺序（即尺寸从小到大）。"""
    if family not in FAMILIES:
        raise ValueError(f"未知家族 {family!r}，应为 {FAMILIES} 之一")
    return tuple(e.key for e in _ENTRIES if e.family == family)
