"""E3 的新增 token 表、LoRA 与检查点（DEV-0020 第二部分；D21-2、D34-3、D36-7、D41-2）。

三个适配器都继承 CheckpointSupport；家族相关的部分由钩子提供（见类说明）。

**新增 token 的表与核对式恢复**
- token_table()：名字 → id（grid token，按 id 升序）、查询占位符的登记（id、是否已分配行、是否在 tokenizer 里、个数与位置模式）、
  每一行在输入嵌入 / lm_head（Gemma 另有 PLE）里的 sha256（行先转成 bf16 再取原始字节：与模型 dtype 无关，fp32 训练、bf16 评测
  也对得上）、家族、模型键、repo 与 revision、tokenizer / 嵌入的规模事实。
- restore_tokens(table, rows)：**只核对、不初始化**。只用于刚挂好的适配器（没加过 token、没注入过查询、没注入 LoRA）：
  先按本适配器的分配规则推算这些名字会拿到的 id，与表不同就报错（什么都不改）；相同再登记（不写均值；需要扩表时新行先填 0）、
  把保存的行写回去，写回后逐行重算哈希，与表不同就报错；最后按表恢复查询登记（个数与位置模式）。
  这条路径取代"重新挂适配器再 add_grid_tokens"：后者会把训练过的行重置为均值（D21-2）。

**LoRA**
- lora_target_modules()：只含 LLM 的线性层（`model.language_model.` 下的全部 nn.Linear），不含视觉塔、投影层与 lm_head；
  每个线性层的名字后缀必须在本家族的表里（LORA_TARGET_SUFFIXES），出现表外的线性层就报错（transformers 改了结构时不静默漏掉）。
  Qwen3.5 的表就是 env_report §7 的 language_model_linear_only（含 GDN 的 in_proj_qkv / z / b / a 与 out_proj）；
  Gemma 4、InternVL3.5 按各自的模块名补齐（Gemma 含 PLE 的三个投影；InternVL 的视觉塔也有 q/k/v_proj，按前缀排除）。
- 顺序约束：inject_lora 必须在 add_grid_tokens、inject_query_tokens（以及 restore_tokens）之后；之后再调它们报 TokenOrderError。
- inject_lora 用 peft 的 get_peft_model 原地包装（适配器的 model 仍是同一个对象，前向与生成照常走它，LoRA 生效）；peft 会把
  base 的参数全部设为不求梯度。merge_lora 在内存里 merge_and_unload，合并后的权重不落盘（D34）。
- 新增 token 行的训练开关 trainable_token_rows（DEV-0021；规则见 TRAINABLE_ROWS_RULE）：inject_lora 之后调用，只让这些
  grid token 的嵌入行（不共享时连同 lm_head 行）求梯度（权重上的梯度掩码钩子），行留在原权重里，保存、恢复、合并照旧。

**合并核对**（简报 §12；DEV-0021）：merge_consistency（单批、单个思考开关）；多批与两种思考开关用 merge_side（一侧的生成与
教师强制隐藏状态）+ compare_merge_sides（逐 run 比较；分歧处与 argmax 不同处的 logit 差），scripts/check_training_interfaces.py
用它们做评测前的标准核对。

**检查点目录**（save_trained 写；load_trained 读；只存这几样，不存合并后的权重，D34）
- `lora/adapter_config.json`、`lora/adapter_model.safetensors`：peft 格式（LoraConfig.save_pretrained +
  get_peft_model_state_dict，不含嵌入层）；没注入 LoRA（如 C1：冻结 LLM）时没有这个子目录。
- `tokens.json`（token_table）、`token_rows.safetensors`（embed、不共享时的 lm_head、Gemma 的 per_layer；行按 token_table 的 row_ids）。
- `query_embeds.safetensors`：注入了查询 token 时必须给（[n_query, hidden]）。
- `extra.safetensors`：调用方给的张量（E3a 的解码器、逐层混合权重等，名字 → 张量），可选。
- `manifest.json`：最后写；base 的 repo / revision / 模型键 / 家族 / dtype / 构造方式（factory 记下的 build_spec）、base 指纹
  （各参数前 FINGERPRINT_ELEMS 个元素的 bf16 字节的 sha256，attach 时算）、LoRA 配置摘要、各文件的 sha256 与字节数、版本。
  读取时逐个核对文件的 sha256 与字节数，目录里多出或缺少文件都报错（以 "." 开头的文件不算，如 macOS 的 .DS_Store）。
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.adapters import registry
from src.adapters.image_policy import FixedSize, PixelBudget, SoftTokenBudget, TilePolicy, parse_policy

CHECKPOINT_FORMAT = "visint-e3-checkpoint/1"
TOKEN_TABLE_FORMAT = "visint-token-table/1"
MANIFEST = "manifest.json"
TOKENS_JSON = "tokens.json"
TOKEN_ROWS = "token_rows.safetensors"
LORA_DIR = "lora"
LORA_CONFIG = f"{LORA_DIR}/adapter_config.json"
LORA_WEIGHTS = f"{LORA_DIR}/adapter_model.safetensors"
QUERY_EMBEDS = "query_embeds.safetensors"
EXTRA = "extra.safetensors"
QUERY_EMBEDS_KEY = "query_embeds"
FINGERPRINT_ELEMS = 4096  # base 指纹：每个参数取前这么多个元素
LM_PREFIX = "model.language_model."  # 三个家族的 LLM 都在这个前缀下（lora_targets_for 核对）

# 各家族 LoRA 目标的名字后缀（LLM 内全部 nn.Linear 的后缀；表外的线性层报错）
LORA_TARGET_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    registry.FAMILY_QWEN35: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                             "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
    registry.FAMILY_GEMMA4: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                             "per_layer_input_gate", "per_layer_projection", "per_layer_model_projection"),
    registry.FAMILY_INTERNVL35: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}


FROZEN_PROBE_ROWS = 512  # trainable_token_rows：冻结行抽样核对的行数（均匀取，另加全部其余新增行）
TRAINABLE_ROWS_RULE = (
    "trainable_token_rows(names_or_ids)：在 inject_lora 之后调用（peft 注入时把 base 的参数全部设为不求梯度）。输入嵌入（不共享"
    "时另有 lm_head）的权重改为求梯度，并在权重上挂一个梯度掩码钩子：反向传播得到的梯度在这些行以外一律置 0（torch.where，"
    "不受 NaN 影响）；共享权重时嵌入与 lm_head 是同一个 Parameter，只挂一次，嵌入查表与 lm_head 两条路径的梯度先相加再过掩码。"
    "行仍在原来的权重张量里，所以生成、restricted_logits、token_table / token_rows、save_trained、restore_tokens、merge_lora "
    "都不用改，看到的就是训练后的值。代价与约束：梯度与优化器状态是整个 [V, hidden] 的稠密张量；冻结行梯度为 0 时 Adam / SGD 的"
    "更新恰为 0（逐位不变），但 AdamW 的 weight decay 不看梯度，会改所有行，所以这些参数必须放进 weight_decay = 0 的参数组。"
    "save_trained 会核对冻结行的抽样（FROZEN_PROBE_ROWS 行加全部其余新增行）与打开时逐位相同，不同就报错。"
)


def _frozen_probe_rows(n_rows: int, trainable: Sequence[int], new_rows: Sequence[int]) -> list[int]:
    """冻结行的抽样：在 [0, n_rows) 上均匀取 FROZEN_PROBE_ROWS 行，加上全部其余新增行（grid token 与查询占位符），去掉要训练的行。"""
    even = torch.linspace(0, n_rows - 1, steps=min(FROZEN_PROBE_ROWS, n_rows)).round().long().tolist()
    skip = set(int(i) for i in trainable)
    return sorted((set(even) | set(int(i) for i in new_rows)) - skip)


class CheckpointError(RuntimeError):
    """检查点目录或 manifest 不合规：文件缺失 / 多出 / 哈希不符、base 对不上、格式不认识。"""


class TokenRestoreError(CheckpointError):
    """restore_tokens 核对失败：id 与表不同、行的哈希与表不同、适配器不是刚挂好的状态。"""


class LoraStateError(RuntimeError):
    """LoRA 的状态不对：重复注入、没注入就合并、合并后再保存等。"""


# ---------------------------------------------------------------------------
# 哈希与指纹
# ---------------------------------------------------------------------------


def _bf16_bytes(t: torch.Tensor) -> Any:
    return t.detach().to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy()


def row_sha256(rows: torch.Tensor) -> list[str]:
    """[n, d] 的行 → 每行的 sha256（行先转成 bf16，取原始字节）。"""
    if rows.dim() != 2:
        raise ValueError(f"rows 必须是二维 [n, d]，收到 {tuple(rows.shape)}")
    return [hashlib.sha256(r.tobytes()).hexdigest() for r in _bf16_bytes(rows)]


def base_fingerprint(model: Any) -> str:
    """base 权重的指纹：按 named_parameters 的顺序，每个参数的名字与前 FINGERPRINT_ELEMS 个元素（bf16 字节）一起取 sha256。
    LoRA 的参数跳过、`.base_layer` 去掉，所以注入 LoRA 前后相同；嵌入表的前几行是基础词表，扩表不影响它。在 attach 时算。"""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if "lora_" in name:
            continue
        h.update(name.replace(".base_layer", "").encode())
        h.update(_bf16_bytes(p.reshape(-1)[:FINGERPRINT_ELEMS]).tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def policy_spec(policy: Any) -> str:
    """图像口径 → image_policy.parse_policy 认得的写法（parse_policy(policy_spec(p)) == p，测试钉住）。"""
    if isinstance(policy, FixedSize):
        spec = f"fixed:{policy.h}x{policy.w}"
    elif isinstance(policy, PixelBudget):
        spec = f"pixel_budget:{policy.n_tokens}"
    elif isinstance(policy, SoftTokenBudget):
        spec = f"soft_tokens:{policy.max_soft_tokens}"
    elif isinstance(policy, TilePolicy):
        spec = f"tiles:{'true' if policy.crop_to_patches else 'false'}:{policy.max_patches}"
    else:
        raise TypeError(f"认不出图像口径 {policy!r}")
    if parse_policy(spec) != policy:
        raise RuntimeError(f"图像口径 {policy!r} 写成 {spec!r} 后解析不回原样")
    return spec


def lora_targets_for(model: Any, family: str) -> list[str]:
    """LLM（LM_PREFIX 下）的全部 nn.Linear 的名字（注入 LoRA 之后也给出原来的名字：去掉 .base_layer、跳过 lora_A/B）；
    每个的后缀必须在 LORA_TARGET_SUFFIXES[family] 里，否则报错。"""
    suffixes = set(LORA_TARGET_SUFFIXES[family])
    out: list[str] = []
    for name, mod in model.named_modules():
        if not name.startswith(LM_PREFIX) or ".lora_" in name or not isinstance(mod, torch.nn.Linear):
            continue
        base = name[: -len(".base_layer")] if name.endswith(".base_layer") else name
        if base.rsplit(".", 1)[-1] not in suffixes:
            raise RuntimeError(f"LLM 里的线性层 {base} 不在 {family} 的 LoRA 目标表 {sorted(suffixes)} 里")
        out.append(base)
    if not out:
        raise RuntimeError(f"{LM_PREFIX} 下没有找到线性层")
    return out


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _listed_files(d: Path) -> set[str]:
    return {p.relative_to(d).as_posix() for p in d.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(d).parts)}


def read_manifest(directory: str | os.PathLike) -> dict[str, Any]:
    """读 manifest 并核对目录：每个登记的文件存在、字节数与 sha256 相同，没有多出的文件（以 "." 开头的不算）。"""
    d = Path(directory)
    path = d / MANIFEST
    if not path.is_file():
        raise CheckpointError(f"{d} 里没有 {MANIFEST}")
    man = json.loads(path.read_text())
    if man.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"不认识的检查点格式 {man.get('format')!r}（期望 {CHECKPOINT_FORMAT!r}）")
    listed = man.get("files") or {}
    actual = _listed_files(d) - {MANIFEST}
    if actual != set(listed):
        raise CheckpointError(f"目录里的文件与 manifest 不符：多出 {sorted(actual - set(listed))}，"
                              f"缺少 {sorted(set(listed) - actual)}")
    for rel, info in listed.items():
        p = d / rel
        if p.stat().st_size != info["bytes"] or file_sha256(p) != info["sha256"]:
            raise CheckpointError(f"{rel} 的字节数或 sha256 与 manifest 不符")
    return man


@dataclass
class TrainedCheckpoint:
    """load_trained 的结果：manifest、查询嵌入（没有查询时为 None；在适配器的设备上）、extra 张量（CPU）。"""

    manifest: dict[str, Any]
    query_embeds: torch.Tensor | None = None
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


def _save_tensors(tensors: Mapping[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().to("cpu").contiguous().clone() for k, v in tensors.items()}, str(path),
              metadata={"format": "pt"})


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


@dataclass
class MergeSide:
    """merge_side 的结果，每个 run（一批输入 × 一种思考开关）一项：generate_batch 的结果（generate=False 时为 None）、
    教师强制的目标（每行一个 token 列表）、目标隐藏状态 target_hidden [N, hidden]（模型所在设备，已 detach）与每行的偏移。"""

    generations: list[list[Any]] | None
    targets: list[list[list[int]]]
    hidden: list[torch.Tensor]
    offsets: list[tuple[int, ...]]

    def targets_are_own_generations(self, k: int) -> bool:
        """第 k 个 run 的教师强制目标就是本侧自己生成的序列（这时分歧位置上的 logits 对两侧都有效）。"""
        return self.generations is not None and self.targets[k] == [list(r.token_ids) for r in self.generations[k]]


def merge_side(ad: Any, runs: Sequence[tuple[Sequence[Any], bool]], *, max_new_tokens: int,
               allowed_ids: Sequence[int] | None = None, targets: Sequence[Sequence[Sequence[int]]] | None = None,
               generate: bool = True) -> MergeSide:
    """合并核对的一侧（DEV-0021）：对每个 run = (inputs, enable_thinking) 做 generate_batch（allowed_ids 给了就受限；
    generate=False 时不生成），再以 targets[k]（不给时取本侧生成的序列）做 training_forward，记下目标隐藏状态。"""
    if not generate and targets is None:
        raise ValueError("generate=False 时必须给 targets")
    if targets is not None and len(targets) != len(runs):
        raise ValueError(f"targets 有 {len(targets)} 项，runs 有 {len(runs)} 项")
    allowed = None if allowed_ids is None else sorted(int(i) for i in allowed_ids)
    gens_all: list[list[Any]] | None = [] if generate else None
    all_targets, hidden, offsets = [], [], []
    for k, (inputs, thinking) in enumerate(runs):
        inputs = list(inputs)
        gens = None
        if generate:
            kw: dict[str, Any] = {} if allowed is None else {"allowed_token_ids": allowed}
            gens = ad.generate_batch(inputs, enable_thinking=thinking, max_new_tokens=max_new_tokens, **kw)
            gens_all.append(gens)  # type: ignore[union-attr]
        tgt = [list(t) for t in targets[k]] if targets is not None else [list(r.token_ids) for r in gens]
        with torch.no_grad():
            tf = ad.training_forward(inputs, tgt, enable_thinking=thinking)
        all_targets.append(tgt)
        hidden.append(tf.target_hidden.detach())
        offsets.append(tuple(tf.target_offsets))
    return MergeSide(gens_all, all_targets, hidden, offsets)


def _logits_and_columns(ad: Any, allowed_ids: Sequence[int] | None) -> tuple[Any, Any, Any]:
    """(logits 函数, 列 → token id, token id → 列)。logits 只用 training.project（唯一的算法）：全词表时 lm_head 先升到 fp32
    一次（每块不再重复转换）；allowed_ids 给了就只取这些行（与 restricted_logits 相同：先 index_select 再 project）。"""
    from src.adapters.training import check_allowed_ids, project

    weight, cap = ad.lm_head_weight().detach(), ad._logit_softcap()
    if allowed_ids is None:
        w = weight.float()
        return (lambda h: project(h, w, softcap=cap)), (lambda c: int(c)), (lambda t: int(t))
    allowed = sorted(int(i) for i in allowed_ids)
    idx = check_allowed_ids(allowed, int(weight.shape[0]), device=weight.device)
    w = weight.index_select(0, idx).float()
    col = {t: k for k, t in enumerate(allowed)}
    return (lambda h: project(h, w, softcap=cap)), (lambda c: allowed[int(c)]), (lambda t: col[int(t)])


def compare_teacher_forced(h_a: torch.Tensor, h_b: torch.Tensor, logits_fn: Any, col_to_id: Any, *,
                           chunk_size: int = 256) -> dict[str, Any]:
    """同一组目标上的两份目标隐藏状态 → logits 的比较（按 chunk_size 分块，每块的 logits 用完即弃）：
    max_abs_hidden、max_abs_logit、argmax_agreement，以及 argmax 不同的位置 disagreements =
    [(位置 n, a 的 argmax, b 的 argmax, gap, top2_gap)]：gap = max(L_a[a] − L_a[b], L_b[b] − L_b[a])（两侧各自对自己的选择领先
    多少，取大者），top2_gap = 两侧前两名之差的较大者。gap ≥ top2_gap；对方的选择恰是第二名时两者相同。"""
    from src.adapters.training import top2_gap

    if h_a.shape != h_b.shape:
        raise ValueError(f"两份隐藏状态的形状不同：{tuple(h_a.shape)} 与 {tuple(h_b.shape)}")
    n = int(h_a.shape[0])
    max_logit, agree, dis = 0.0, 0, []
    with torch.no_grad():
        max_h = float((h_a.float() - h_b.float()).abs().max()) if n else 0.0
        for s in range(0, n, chunk_size):
            la, lb = logits_fn(h_a[s: s + chunk_size]), logits_fn(h_b[s: s + chunk_size])
            max_logit = max(max_logit, float((la - lb).abs().max()))
            aa, ab = la.argmax(-1), lb.argmax(-1)
            bad = (aa != ab).nonzero().flatten()
            agree += int(la.shape[0]) - int(bad.numel())
            if bad.numel():
                la_m, lb_m, aa_m, ab_m = la[bad], lb[bad], aa[bad, None], ab[bad, None]
                gap = torch.maximum(la_m.gather(1, aa_m) - la_m.gather(1, ab_m),
                                    lb_m.gather(1, ab_m) - lb_m.gather(1, aa_m)).squeeze(1)
                t2 = torch.maximum(top2_gap(la_m), top2_gap(lb_m))
                cols = zip(bad.tolist(), aa_m[:, 0].tolist(), ab_m[:, 0].tolist(), gap.tolist(), t2.tolist())
                dis += [(s + i, col_to_id(x), col_to_id(y), g, t) for i, x, y, g, t in cols]  # 一次搬到 CPU，不逐项同步
    return {"n_tokens": n, "max_abs_hidden": max_h, "max_abs_logit": max_logit,
            "argmax_agreement": agree / n if n else 1.0, "disagreements": dis}


def compare_merge_sides(ad: Any, a: MergeSide, b: MergeSide, *, allowed_ids: Sequence[int] | None = None,
                        chunk_size: int = 256) -> list[dict[str, Any]]:
    """合并核对两侧的逐 run 比较（DEV-0021；logits 用 ad 的 lm_head，两侧的 lm_head 必须相同——LoRA 不含 lm_head，新增行由
    token 表核对）。b 的教师强制目标必须与 a 的相同。每个 run 一项：
    - generate（两侧都生成了时）：generation_agreement 的各项，另加 divergence_gaps / divergence_top2_gaps：每行第一个分歧处
      （a 的目标就是 a 自己生成的序列时，该位置的教师强制 logits 对两侧都有效）两侧 logits 上 |L[a 的 token] − L[b 的 token]|
      的较大者、前两名之差的较大者；没有分歧的行为 None。
    - teacher_forcing：compare_teacher_forced 的各项；disagreements 的位置换成 (行, 行内位置)。"""
    from src.adapters.training import generation_agreement, top2_gap

    if len(a.targets) != len(b.targets) or a.targets != b.targets:
        raise ValueError("两侧的教师强制目标不同：b 要用 a 的目标（merge_side(targets=a.targets)）")
    logits_fn, col_to_id, id_to_col = _logits_and_columns(ad, allowed_ids)
    out = []
    for k in range(len(a.targets)):
        offs = a.offsets[k]
        tf = compare_teacher_forced(a.hidden[k], b.hidden[k], logits_fn, col_to_id, chunk_size=chunk_size)

        rows_of = [bisect.bisect_right(offs, n) - 1 for n, *_ in tf["disagreements"]]
        tf["disagreements"] = [(r, n - offs[r], x, y, g, t2) for r, (n, x, y, g, t2) in zip(rows_of, tf["disagreements"])]
        rep: dict[str, Any] = {"teacher_forcing": tf}
        if a.generations is not None and b.generations is not None:
            ga, gb = a.generations[k], b.generations[k]
            gen = generation_agreement(ga, gb)
            gaps: list[float | None] = [None] * len(ga)
            top2: list[float | None] = [None] * len(ga)
            if a.targets_are_own_generations(k):
                pos, xs, ys, rows = [], [], [], []
                for r, d in enumerate(gen["first_divergence"]):
                    if d is not None and d < len(ga[r].token_ids) and d < len(gb[r].token_ids):
                        rows.append(r)
                        pos.append(offs[r] + d)
                        xs.append(id_to_col(ga[r].token_ids[d]))
                        ys.append(id_to_col(gb[r].token_ids[d]))
                if rows:
                    with torch.no_grad():
                        p = torch.tensor(pos, dtype=torch.long, device=a.hidden[k].device)
                        la, lb = logits_fn(a.hidden[k].index_select(0, p)), logits_fn(b.hidden[k].index_select(0, p))
                        x = torch.tensor(xs, dtype=torch.long, device=la.device)[:, None]
                        y = torch.tensor(ys, dtype=torch.long, device=la.device)[:, None]
                        g = torch.maximum((la.gather(1, x) - la.gather(1, y)).abs(),
                                          (lb.gather(1, x) - lb.gather(1, y)).abs()).squeeze(1)
                        t2 = torch.maximum(top2_gap(la), top2_gap(lb))
                    for i, r in enumerate(rows):
                        gaps[r], top2[r] = float(g[i]), float(t2[i])
            rep["generate"] = {**gen, "divergence_gaps": gaps, "divergence_top2_gaps": top2}
        out.append(rep)
    return out


def merge_consistency(ad: Any, inputs_list: Sequence[Any], *, max_new_tokens: int, enable_thinking: bool,
                      allowed_ids: Sequence[int] | None = None, chunk_size: int = 256) -> dict[str, Any]:
    """LoRA 合并前后的一致性读数（简报 §12"合并 LoRA"；D21-2：GPU 上写成一致率）。**会把适配器的 LoRA 合并**。

    - generate：合并前后 generate_batch（allowed_ids 给了就受限）的 token 序列：exact_rows、token_agreement（第一个分歧之前的
      token 数 / 两者较长者，按行求和再比）、first_divergence；另有分歧处的 logit 差（compare_merge_sides）。
    - teacher_forcing：合并前生成的序列当目标，合并前后 training_forward 的目标隐藏状态最大绝对差（max_abs_hidden）、logits 的
      最大绝对差（max_abs_logit；allowed_ids 给了用受限的行，否则全词表按 chunk_size 分块）、argmax 一致率与不一致的位置。
    before / after 是两次 generate_batch 的原始结果。多批、两种思考开关一起核对时用 merge_side + compare_merge_sides
    （scripts/check_training_interfaces.py）。
    """
    if ad.lora_state != "injected":
        raise LoraStateError("merge_consistency 要在注入 LoRA、未合并时调用")
    runs = [(list(inputs_list), enable_thinking)]
    before = merge_side(ad, runs, max_new_tokens=max_new_tokens, allowed_ids=allowed_ids)
    ad.merge_lora()
    after = merge_side(ad, runs, max_new_tokens=max_new_tokens, allowed_ids=allowed_ids, targets=before.targets)
    (rep,) = compare_merge_sides(ad, before, after, allowed_ids=allowed_ids, chunk_size=chunk_size)
    return {"n_rows": len(runs[0][0]), "before": before.generations[0], "after": after.generations[0], **rep}


# ---------------------------------------------------------------------------
# 三个适配器共用的实现
# ---------------------------------------------------------------------------


class CheckpointSupport:
    """token_table、restore_tokens、lora_target_modules、inject_lora、merge_lora、save_trained、load_trained 的公共实现
    （规则见模块说明）。

    家族钩子：`_token_registry()`（grid token、查询占位符、规模事实）、`_token_state_fresh()`、`_restore_registration(table)`
    （先核对 id 再登记，不初始化）、`_row_tables()`（[(种类, 权重)]，含 lm_head；共享时 lm_head 就是嵌入）。
    trainable_token_rows 只用 `_row_tables()` 与 grid_token_ids，不需要新的家族钩子。
    """

    _lora_state: str = "none"  # none / injected / merged
    _peft: Any = None
    build_spec: dict[str, Any] | None = None  # factory.make_adapter 记下的构造方式（load_trained 重建 base 用）
    _trainable_row_ids: tuple[int, ...] | None = None  # trainable_token_rows 打开的行
    _row_grad_hooks: tuple[Any, ...] | list[Any] = ()  # 梯度掩码钩子的 handle（再调 trainable_token_rows 时先移除）
    _frozen_row_probes: tuple[Any, ...] | list[Any] = ()  # (参数, 抽样的冻结行 id, 打开时的值)：save_trained 核对

    # ------------------------------------------------------------------
    # 身份
    # ------------------------------------------------------------------

    def _identity(self) -> dict[str, Any]:
        """家族、模型键、repo / revision（小模型取它的 config 来源）、是否小模型。"""
        key = self.model_key
        tiny = key is not None and key.startswith("tiny(") and key.endswith(")")
        src = key[5:-1] if tiny else key
        entry = registry.MODELS.get(src) if src is not None else None
        return {"family": self.family, "model_key": key, "tiny": tiny,
                "repo": entry.repo if entry else None, "revision": entry.revision if entry else None}

    # ------------------------------------------------------------------
    # 新增 token 的表
    # ------------------------------------------------------------------

    def token_table(self) -> dict[str, Any]:
        self._require_loaded()
        reg = self._token_registry()
        grid = sorted(reg["grid_tokens"], key=lambda g: g["id"])
        q = reg["query"]
        ids = sorted([g["id"] for g in grid] + ([q["id"]] if q["row_allocated"] else []))
        idx = torch.tensor(ids, dtype=torch.long, device=self._device)
        tables = {}
        with torch.no_grad():
            for kind, w in self._row_tables():
                tables[kind] = {"dim": int(w.shape[1]), "sha256": row_sha256(w.index_select(0, idx)) if ids else []}
        return {
            "format": TOKEN_TABLE_FORMAT,
            **self._identity(),
            "tied_word_embeddings": bool(self.tied_word_embeddings),
            "vocab": reg["vocab"],
            "grid_tokens": grid,
            "n_grid_tokens": len(grid),
            "query": q,
            "row_ids": ids,
            "row_tables": tables,
            "row_hash": "sha256 of the row's bfloat16 bytes",
        }

    def token_rows(self) -> dict[str, torch.Tensor]:
        """token_table 的 row_ids 对应的行（CPU 拷贝，模型 dtype）：embed、不共享时的 lm_head、Gemma 的 per_layer。"""
        self._require_loaded()
        ids = self.token_table()["row_ids"]
        idx = torch.tensor(ids, dtype=torch.long, device=self._device)
        out = {}
        with torch.no_grad():
            for kind, w in self._row_tables():
                if kind == "lm_head" and self.tied_word_embeddings:
                    continue
                out[kind] = w.index_select(0, idx).detach().to("cpu").clone()
        return out

    def restore_tokens(self, table: Mapping[str, Any], rows: Mapping[str, torch.Tensor]) -> None:
        self._require_loaded()
        self._check_tokens_mutable("restore_tokens")
        if table.get("format") != TOKEN_TABLE_FORMAT:
            raise TokenRestoreError(f"不认识的 token 表格式 {table.get('format')!r}")
        mine = self._identity()
        bad = {k: (table.get(k), v) for k, v in mine.items() if table.get(k) != v}
        if table.get("tied_word_embeddings") != bool(self.tied_word_embeddings):
            bad["tied_word_embeddings"] = (table.get("tied_word_embeddings"), bool(self.tied_word_embeddings))
        if bad:
            raise TokenRestoreError(f"token 表与本适配器的 base 不符（表, 本适配器）：{bad}")
        if not self._token_state_fresh():
            raise TokenRestoreError("restore_tokens 只用于刚挂好的适配器（还没加过 token、没注入过查询）")
        kinds = dict(self._row_tables())
        if set(table["row_tables"]) != set(kinds):
            raise TokenRestoreError(f"表里的行种类 {sorted(table['row_tables'])} 与本适配器 {sorted(kinds)} 不同")
        want_rows = {k for k in kinds if not (k == "lm_head" and self.tied_word_embeddings)}
        if set(rows) != want_rows:
            raise TokenRestoreError(f"给的行 {sorted(rows)} 与应有的 {sorted(want_rows)} 不同")
        ids = list(table["row_ids"])
        for k in want_rows:
            if tuple(rows[k].shape) != (len(ids), kinds[k].shape[1]):
                raise TokenRestoreError(f"{k} 的行形状 {tuple(rows[k].shape)}，期望 ({len(ids)}, {kinds[k].shape[1]})")
        self._restore_registration(table)  # 先核对 id（不符就报错、什么都不改），再登记；不初始化
        kinds = dict(self._row_tables())  # 扩表后权重是新的张量
        idx = torch.tensor(ids, dtype=torch.long, device=self._device)
        with torch.no_grad():
            for k in want_rows:
                w = kinds[k]
                src = rows[k].to(device=w.device, dtype=w.dtype)
                w[idx] = src
                if rows[k].dtype == w.dtype and not torch.equal(w.index_select(0, idx).cpu(), rows[k].cpu()):
                    raise TokenRestoreError(f"{k} 的行写回后与给的行不同")
            for k, w in kinds.items():
                got = row_sha256(w.index_select(0, idx)) if ids else []
                if got != list(table["row_tables"][k]["sha256"]):
                    n_bad = sum(a != b for a, b in zip(got, table["row_tables"][k]["sha256"]))
                    raise TokenRestoreError(f"{k} 的行写回后有 {n_bad} 行的哈希与表不同")
        q = table["query"]
        if q.get("n") is not None:
            self.inject_query_tokens(int(q["n"]), q["position_mode"])

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    @property
    def lora_state(self) -> str:
        """"none"（没注入）、"injected"（已注入、未合并）、"merged"（已在内存里合并）。"""
        return self._lora_state

    @property
    def peft_model(self) -> Any:
        """注入后的 PeftModel（没注入或已合并时为 None）。它包着的就是适配器的 model（同一个对象）。"""
        return self._peft

    def lora_target_modules(self) -> list[str]:
        self._require_loaded()
        return lora_targets_for(self._model, self.family)

    def inject_lora(self, *, r: int, lora_alpha: float, lora_dropout: float = 0.0, init_lora_weights: Any = True) -> None:
        """注入 LoRA（目标 = lora_target_modules()，bias 不训练）。必须在 add_grid_tokens / inject_query_tokens 之后（之后它们报错）。
        init_lora_weights=True 是 peft 的默认（B 为 0，注入后输出不变）；测试用 False 得到非零的随机 LoRA。"""
        self._require_loaded()
        if self._lora_state != "none":
            raise LoraStateError(f"LoRA 已经是 {self._lora_state} 状态，不能再注入")
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            raise ValueError(f"r 必须是正整数，收到 {r!r}")
        if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, (int, float)) or not lora_alpha > 0:
            raise ValueError(f"lora_alpha 必须是正数，收到 {lora_alpha!r}")
        if isinstance(lora_dropout, bool) or not isinstance(lora_dropout, (int, float)) or not 0 <= lora_dropout < 1:
            raise ValueError(f"lora_dropout 必须在 [0, 1)，收到 {lora_dropout!r}")
        from peft import LoraConfig

        cfg = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=float(lora_dropout),
                         target_modules=self.lora_target_modules(), bias="none", init_lora_weights=init_lora_weights)
        self._attach_lora(cfg)

    def _attach_lora(self, cfg: Any) -> None:
        from peft import get_peft_model

        # 核对以"实际包上 LoRA 的模块 = lora_target_modules()"为准：peft 会把很长的目标名单压缩成最短的后缀集合写回配置
        # （_find_minimal_target_modules，保证只匹配这些模块），所以从 adapter_config.json 读回来的 target_modules 不是全名
        targets = self.lora_target_modules()
        ident = self._identity()
        cfg.base_model_name_or_path = ident["repo"]
        cfg.revision = ident["revision"]
        training = self._model.training
        peft_model = get_peft_model(self._model, cfg)
        if peft_model.base_model.model is not self._model:
            raise LoraStateError("peft 没有原地包装适配器的 model")
        wrapped = sorted(n for n, m in self._model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A)
        if wrapped != sorted(targets):
            raise LoraStateError(f"包上 LoRA 的模块（{len(wrapped)} 个）与目标（{len(targets)} 个）不同")
        self._model.train(training)
        self._peft = peft_model
        self._lora_state = "injected"

    def merge_lora(self) -> None:
        if self._lora_state != "injected":
            raise LoraStateError(f"只有注入后、未合并的 LoRA 才能合并（当前 {self._lora_state}）")
        merged = self._peft.merge_and_unload()
        if merged is not self._model:
            raise LoraStateError("merge_and_unload 没有返回适配器的 model")
        left = [n for n, _ in self._model.named_modules() if "lora_" in n]
        if left:
            raise LoraStateError(f"合并后还有 LoRA 模块：{left[:3]}")
        self._peft = None
        self._lora_state = "merged"

    def lora_parameter_count(self) -> int:
        """注入的 LoRA 参数个数（A 与 B）。"""
        if self._lora_state != "injected":
            raise LoraStateError("没有注入的 LoRA")
        return sum(p.numel() for n, p in self._model.named_parameters() if "lora_" in n)

    # ------------------------------------------------------------------
    # 新增 token 行的训练开关（DEV-0021）
    # ------------------------------------------------------------------

    @property
    def trainable_token_row_ids(self) -> tuple[int, ...] | None:
        """trainable_token_rows 打开的行（升序）；没打开时为 None。"""
        return self._trainable_row_ids

    def trainable_token_rows(self, names_or_ids: Sequence[str | int]) -> list[torch.nn.Parameter]:
        """让这些 grid token 的行求梯度，其余行冻结（规则见 TRAINABLE_ROWS_RULE）。返回要交给优化器的参数（输入嵌入；不共享时
        另有 lm_head；共享时两者是同一个张量，只返回一次）。这些参数要放进 weight_decay = 0 的参数组（见规则）。
        必须在 inject_lora 之后、合并之前调用；再调一次就换成新的行集合。"""
        self._require_loaded()
        if self._lora_state != "injected":
            raise LoraStateError(f"trainable_token_rows 要在 inject_lora 之后、合并之前调用（当前 LoRA 状态：{self._lora_state}）："
                                 "peft 注入时会把 base 的参数全部设为不求梯度")
        ids = self._resolve_grid_rows(names_or_ids)
        tables = dict(self._row_tables())
        emb, head = tables["embed"], tables["lm_head"]
        if (emb is head) != bool(self.tied_word_embeddings):
            raise RuntimeError(f"tied_word_embeddings = {self.tied_word_embeddings}，但嵌入与 lm_head "
                               f"{'是' if emb is head else '不是'}同一个张量")
        for h in self._row_grad_hooks:
            h.remove()
        params = [emb] if emb is head else [emb, head]
        hooks, probes = [], []
        trainable = torch.tensor(ids, dtype=torch.long)
        new_rows = self.token_table()["row_ids"]
        for w in params:
            mask = torch.zeros((w.shape[0], 1), dtype=torch.bool, device=w.device)
            mask[trainable.to(w.device)] = True
            w.requires_grad_(True)
            hooks.append(w.register_hook(lambda g, m=mask: torch.where(m, g, torch.zeros((), dtype=g.dtype, device=g.device))))
            probe = _frozen_probe_rows(int(w.shape[0]), ids, new_rows)
            idx = torch.tensor(probe, dtype=torch.long, device=w.device)
            probes.append((w, idx, w.detach().index_select(0, idx).clone()))
        self._row_grad_hooks = hooks
        self._frozen_row_probes = probes
        self._trainable_row_ids = tuple(ids)
        return params

    def _resolve_grid_rows(self, names_or_ids: Sequence[str | int]) -> list[int]:
        """grid token 的名字或 id → 升序的 id（非空、不重复、都是本适配器加过的 grid token）。"""
        if isinstance(names_or_ids, (str, bytes, int)) or torch.is_tensor(names_or_ids):
            raise TypeError("names_or_ids 必须是 grid token 名字或 id 的序列（单个也放进列表）")
        items = list(names_or_ids)
        if not items:
            raise ValueError("names_or_ids 为空")
        grid = self.grid_token_ids
        valid = set(grid.values())
        ids = []
        for x in items:
            if isinstance(x, str):
                if x not in grid:
                    raise ValueError(f"{x!r} 不是本适配器加过的 grid token（已加 {len(grid)} 个）")
                ids.append(grid[x])
            elif isinstance(x, int) and not isinstance(x, bool):
                if x not in valid:
                    raise ValueError(f"id {x} 不是 grid token 的行（只有新增的 grid token 行能打开训练）")
                ids.append(int(x))
            else:
                raise TypeError(f"names_or_ids 的项必须是名字或 id，收到 {x!r}")
        if len(set(ids)) != len(ids):
            raise ValueError("names_or_ids 里有重复的 token")
        return sorted(ids)

    def _check_frozen_rows(self) -> None:
        """trainable_token_rows 之后，冻结行的抽样（含其余新增行）必须与打开时逐位相同；否则多半是优化器对这个参数做了
        weight decay（AdamW 的衰减不看梯度，会改所有行）。"""
        for w, idx, rows in self._frozen_row_probes:
            if not torch.equal(w.detach().index_select(0, idx), rows):
                raise CheckpointError("trainable_token_rows 之外的嵌入 / lm_head 行变了：这些参数必须放进 weight_decay = 0 的参数组"
                                      "（梯度掩码只让冻结行的梯度为 0，AdamW 的 weight decay 仍会改它们）")

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------

    def save_trained(self, directory: str | os.PathLike, *, extra: Mapping[str, torch.Tensor] | None = None,
                     query_embeds: torch.Tensor | None = None) -> Path:
        """写检查点目录（文件清单见模块说明），返回目录。目录必须不存在或为空。合并后的权重不存（合并后调用报错）。"""
        self._require_loaded()
        if self._lora_state == "merged":
            raise LoraStateError("LoRA 已经合并：合并后的权重不存盘（D34），检查点要在合并之前存")
        self._check_frozen_rows()
        d = Path(directory)
        if d.exists() and (not d.is_dir() or any(d.iterdir())):
            raise CheckpointError(f"{d} 已存在且不是空目录")
        table = self.token_table()
        q = table["query"]
        if q.get("n") is not None:
            if query_embeds is None:
                raise CheckpointError("注入了查询 token，必须给 query_embeds")
            want = (int(q["n"]), self.hidden_size())
            if not torch.is_tensor(query_embeds) or tuple(query_embeds.shape) != want:
                raise CheckpointError(f"query_embeds 的形状应为 {want}")
        elif query_embeds is not None:
            raise CheckpointError("没有注入查询 token，却给了 query_embeds")
        extra = dict(extra or {})
        for k, v in extra.items():
            if not isinstance(k, str) or not k or not torch.is_tensor(v):
                raise CheckpointError(f"extra 必须是 名字 → 张量，收到 {k!r}: {type(v).__name__}")
        d.mkdir(parents=True, exist_ok=True)
        (d / TOKENS_JSON).write_text(json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True))
        rows = self.token_rows()
        _save_tensors(rows, d / TOKEN_ROWS)
        lora = None
        if self._lora_state == "injected":
            from peft import get_peft_model_state_dict

            cfg = self._peft.peft_config["default"]
            (d / LORA_DIR).mkdir()
            cfg.save_pretrained(str(d / LORA_DIR))
            state = get_peft_model_state_dict(self._peft, save_embedding_layers=False)
            _save_tensors(state, d / LORA_WEIGHTS)
            targets = self.lora_target_modules()
            lora = {"r": int(cfg.r), "lora_alpha": float(cfg.lora_alpha), "lora_dropout": float(cfg.lora_dropout),
                    "n_targets": len(targets),
                    "targets_sha256": hashlib.sha256("\n".join(sorted(targets)).encode()).hexdigest(),
                    "n_parameters": self.lora_parameter_count(), "n_tensors": len(state)}
        if query_embeds is not None:
            _save_tensors({QUERY_EMBEDS_KEY: query_embeds}, d / QUERY_EMBEDS)
        if extra:
            _save_tensors(extra, d / EXTRA)
        import peft
        import transformers

        files = {rel: {"sha256": file_sha256(d / rel), "bytes": (d / rel).stat().st_size} for rel in sorted(_listed_files(d))}
        manifest = {
            "format": CHECKPOINT_FORMAT,
            **self._identity(),
            "dtype": str(self.dtype).removeprefix("torch."),
            "base_fingerprint": self._base_fingerprint,
            "build_spec": self.build_spec,
            "image_policy": policy_spec(self.image_policy),
            "lora": lora,
            "trainable_token_rows": None if self._trainable_row_ids is None else list(self._trainable_row_ids),
            "query": {"n": q.get("n"), "position_mode": q.get("position_mode")},
            "extra_keys": sorted(extra),
            "files": files,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        }
        tmp = d / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True))
        tmp.replace(d / MANIFEST)
        return d

    def load_trained(self, directory: str | os.PathLike, *, merge: bool = True) -> TrainedCheckpoint:
        """从检查点恢复（规则见模块说明）：base（没挂模型时按 manifest 的模型键与固定 revision 加载；小模型请用
        factory.load_trained 或先挂好同一个小模型）→ 核对 base（家族、模型键、revision、指纹）→ restore_tokens → 挂 LoRA
        （merge=True 时在内存里合并）。返回 manifest、查询嵌入与 extra 张量。"""
        if not isinstance(merge, bool):
            raise TypeError(f"merge 必须是 bool，收到 {merge!r}")
        d = Path(directory)
        man = read_manifest(d)
        if self._model is None:
            if man.get("tiny"):
                raise CheckpointError("小模型的检查点：请用 factory.load_trained，或先挂好同一个小模型再调 load_trained")
            self.load(man["model_key"], getattr(torch, man["dtype"]))
        mine = self._identity()
        bad = {k: (man.get(k), v) for k, v in mine.items() if man.get(k) != v}
        if man.get("base_fingerprint") != self._base_fingerprint:
            bad["base_fingerprint"] = (man.get("base_fingerprint"), self._base_fingerprint)
        if man.get("image_policy") != policy_spec(self.image_policy):
            bad["image_policy"] = (man.get("image_policy"), policy_spec(self.image_policy))
        if bad:
            raise CheckpointError(f"检查点的 base 与本适配器不符（检查点, 本适配器）：{bad}")
        if self._lora_state != "none":
            raise LoraStateError("load_trained 要在注入 LoRA 之前调用")
        table = json.loads((d / TOKENS_JSON).read_text())
        self.restore_tokens(table, _load_tensors(d / TOKEN_ROWS))
        if man.get("lora") is not None:
            from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

            self._attach_lora(LoraConfig.from_pretrained(str(d / LORA_DIR)))
            saved = _load_tensors(d / LORA_WEIGHTS)
            res = set_peft_model_state_dict(self._peft, saved)
            if getattr(res, "unexpected_keys", None):
                raise CheckpointError(f"LoRA 权重里有本模型没有的键：{res.unexpected_keys[:3]}")
            now = get_peft_model_state_dict(self._peft, save_embedding_layers=False)
            if set(now) != set(saved) or any(not torch.equal(now[k].cpu(), saved[k].to(now[k].dtype)) for k in saved):
                raise CheckpointError("LoRA 权重没有原样载入")
            if merge:
                self.merge_lora()
        qe = None
        if (d / QUERY_EMBEDS).is_file():
            qe = _load_tensors(d / QUERY_EMBEDS)[QUERY_EMBEDS_KEY].to(self._device)
        extra = _load_tensors(d / EXTRA) if (d / EXTRA).is_file() else {}
        return TrainedCheckpoint(manifest=man, query_embeds=qe, extra=extra)
