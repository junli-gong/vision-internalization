"""E3a 读出的输入（DEV-0034 §1）：读出模块只吃这里给的张量（ReadoutInputs），这里是它和模型之间唯一的接口。

- full / c1（llm_inputs）：一次 adapter.training_forward（教师强制版的前向，目标为空：只取隐藏状态；带梯度，训练时梯度经它流回 LoRA
  与查询嵌入；评测时在 no_grad 里调）。HiddenRequest 要全部层（层号 0..num_layers()）在查询区域上的隐藏状态；语义分割的行另要
  类别名所在的那一段（第一个类别名的开头到最后一个的结尾，一个区间 "classes"，各层一次 index_select），每个类别名的平均由
  class_pool 矩阵在读出里做。没有别的东西：不取图像 token 的隐藏状态、不取像素。
- c2（vision_inputs）：adapter.vision_features（投影之后、进入 LLM 之前的图像 token，[gh·gw, H]，行主序）排成 [H, gh, gw]，
  F.interpolate(bilinear, align_corners=False, antialias=True) 缩放到 grid × grid（无学习）；no_grad（视觉塔冻结）。
  类别名的表示：同一条提示里类别名 token 的输入嵌入（c2_class_side，第 0 层，不经过 LLM 的任何 decoder 层）。
类别名的区间用适配器的 class_name_spans（拼法见 src/adapters/training.CLASS_NAME_SEPARATOR）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.adapters.base import ModelInputs, Span
from src.adapters.training import HiddenRequest

CLASS_REGION = "classes"


@dataclass
class ReadoutInputs:
    """读出模块的输入（model.READOUT_INPUTS 的三样）与记账用的计数。query [S, B, Q, H]；classes [S, B, T, H] 与 class_pool
    [B, C, T]（只有语义分割的批才有）。prompt_lengths：每行送进 LLM 的提示长度（c2 不跑 LLM，记 0）。"""

    query: torch.Tensor
    classes: torch.Tensor | None = None
    class_pool: torch.Tensor | None = None
    prompt_lengths: tuple[int, ...] = ()

    def args(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return self.query, self.classes, self.class_pool

    def permuted(self, perm: Sequence[int]) -> "ReadoutInputs":
        """泄漏检查（打乱）：第 b 行换成第 perm[b] 行的查询隐藏状态；类别名的表示不动（它在图像之前，与图无关）。"""
        idx = torch.as_tensor(list(perm), dtype=torch.long, device=self.query.device)
        return replace(self, query=self.query.index_select(1, idx))

    def detached(self) -> "ReadoutInputs":
        """切断读出输入与模型之间的梯度（tests 的像素通路检查用）。"""
        return replace(self, query=self.query.detach(), classes=None if self.classes is None else self.classes.detach())


def class_pool_matrix(spans: Sequence[Span]) -> tuple[Span, torch.Tensor]:
    """各类别名的 token 区间 → (覆盖全部类别名的一个区间, [C, T] 的平均矩阵：第 c 行在第 c 个名字的 token 上取 1/长度)。"""
    if not spans:
        raise ValueError("没有类别名区间")
    start, end = spans[0].start, spans[-1].end
    if any(b.start < a.end for a, b in zip(spans, spans[1:])):
        raise ValueError("类别名区间没有按顺序排列或有重叠")
    pool = torch.zeros(len(spans), end - start, dtype=torch.float32)
    for c, s in enumerate(spans):
        if len(s) == 0:
            raise ValueError(f"第 {c} 个类别名的区间为空")
        pool[c, s.start - start: s.end - start] = 1.0 / len(s)
    return Span(start, end), pool


class ClassSpans:
    """类别名区间的缓存：同一条指令（同一串 token）的区间只算一次（class_name_spans 要解码再编码，按行重算没有必要）。
    命中时核对这一行指令位置上的 token 与第一次相同，不同就重算。"""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self._cache: dict[tuple[int, ...], tuple[int, list[Span]]] = {}

    def spans(self, ad: Any, inputs: ModelInputs) -> list[Span]:
        ids = inputs.tensors["input_ids"][0]
        ins = tuple(int(t) for t in ids[inputs.instruction_span.start: inputs.instruction_span.end].tolist())
        hit = self._cache.get(ins)
        if hit is not None:
            base, rel = hit
            off = inputs.instruction_span.start - base
            return [Span(s.start + off, s.end + off) for s in rel]
        sp = list(ad.class_name_spans(inputs, self.names))
        self._cache[ins] = (inputs.instruction_span.start, sp)
        return sp


def llm_inputs(ad: Any, inputs_list: Sequence[ModelInputs], query_embeds: torch.Tensor, *, layers: Sequence[int],
               class_spans: ClassSpans | None = None, enable_thinking: bool) -> ReadoutInputs:
    """full / c1：各层在查询位置（与语义分割的类别名区间）上的隐藏状态（规则见模块说明）。query_embeds 是 fp32 的主副本，这里转成
    模型 dtype。不强制 no_grad。"""
    items = list(inputs_list)
    spans = pools = None
    if class_spans is not None:
        per_row = [class_pool_matrix(class_spans.spans(ad, inp)) for inp in items]
        if len({p.shape for _, p in per_row}) != 1 or any(not torch.equal(p, per_row[0][1]) for _, p in per_row):
            raise RuntimeError("同一批里各行的类别名区间结构不同（指令应当逐字相同）")
        spans = tuple({CLASS_REGION: sp} for sp, _ in per_row)
        pools = torch.stack([p for _, p in per_row]).to(ad.device)
    req = HiddenRequest(layers=tuple(int(l) for l in layers), regions=("query",), spans=spans)
    out = ad.training_forward(items, [[] for _ in items], enable_thinking=enable_thinking, return_hidden=req,
                              query_embeds=query_embeds.to(ad.dtype))
    q = torch.stack([torch.stack(out.hidden["query"][l]) for l in req.layers])
    c = torch.stack([torch.stack(out.hidden[CLASS_REGION][l]) for l in req.layers]) if spans is not None else None
    return ReadoutInputs(query=q, classes=c, class_pool=pools, prompt_lengths=tuple(int(n) for n in out.prompt_lengths))


def resize_grid(feats: torch.Tensor, hw: tuple[int, int], grid: int) -> torch.Tensor:
    """[gh·gw, H] 的行主序网格 → [grid·grid, H]（双线性、抗混叠；同尺寸时原样）。"""
    gh, gw = int(hw[0]), int(hw[1])
    if feats.shape[0] != gh * gw:
        raise ValueError(f"视觉特征 {tuple(feats.shape)} 与网格 {gh}×{gw} 不符")
    x = feats.float().t().reshape(1, -1, gh, gw)
    if (gh, gw) != (grid, grid):
        x = F.interpolate(x, size=(grid, grid), mode="bilinear", align_corners=False, antialias=True)
    return x.reshape(x.shape[1], grid * grid).t()


def vision_inputs(ad: Any, inputs_list: Sequence[ModelInputs], *, grid: int,
                  class_side: tuple[torch.Tensor, torch.Tensor] | None = None) -> ReadoutInputs:
    """c2：视觉特征缩放到 grid × grid（规则见模块说明）。class_side = c2_class_side 的结果（语义分割的批才给）。"""
    rows = []
    with torch.no_grad():
        for inp in inputs_list:
            if len(inp.images) != 1:
                raise ValueError(f"c2 只接受单图输入，收到 {len(inp.images)} 张")
            rows.append(resize_grid(ad.vision_features(inp), inp.images[0].grid_hw, grid))
    q = torch.stack(rows)[None]  # [1, B, Q, H]
    c = pool = None
    if class_side is not None:
        emb, pm = class_side
        b = q.shape[1]
        c = emb[None, None].expand(1, b, *emb.shape).to(q.device)
        pool = pm[None].expand(b, *pm.shape).to(q.device)
    return ReadoutInputs(query=q, classes=c, class_pool=pool, prompt_lengths=tuple(0 for _ in inputs_list))


def c2_class_side(ad: Any, instruction: str, names: Sequence[str], *, order: str) -> tuple[torch.Tensor, torch.Tensor]:
    """c2 的类别名表示：语义分割指令（不带图）里类别名那一段 token 的输入嵌入 [T, H]（fp32，冻结）与平均矩阵 [C, T]。"""
    inp = ad.build_inputs([], instruction, order)
    span, pool = class_pool_matrix(ad.class_name_spans(inp, list(names)))
    ids = inp.tensors["input_ids"][0, span.start: span.end]
    with torch.no_grad():
        emb = ad.model.get_input_embeddings()(ids[None].to(ad.device))[0].float()
    return emb, pool


__all__ = ["CLASS_REGION", "ClassSpans", "ReadoutInputs", "c2_class_side", "class_pool_matrix", "llm_inputs", "resize_grid",
           "vision_inputs"]
