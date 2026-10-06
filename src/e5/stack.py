"""E5 主比较的起点与"起点 + E5 LoRA"的叠加（DEV-0037 §3、§4；configs/e5_train.json 的 starts 与 start_lora）。

**起点**（StartRef；--init）：
- base_warmup：DEV-0035 的网格格式预热检查点（scripts/train_e3b.py --supervision warmup；LoRA + 67 个网格 token 行）；
- e3b：E3b 检查点（--supervision grid；LoRA + 67 个网格 token 行）；
- e3a：E3a 检查点（scripts/train_e3a.py --variant full 的 ckpt/final/model；LoRA，没有网格 token，表里只有查询占位符的行）；
- base：基座，没有检查点（诊断）。
起点检查点的核对（resolve_start）：manifest 的构造方式与 --model、--tiny 相同；带网格 token 的起点（from_checkpoint）必须恰好是 E3b 词表的
67 个；e3a 不能带网格 token；三者都必须有 LoRA。检查点在训练叶目录 <叶>/ckpt/final/model 下时，另读 <叶>/config.yaml 的 variant 核对
（预热与 E3b 的检查点结构相同，防止弄混）；找不到就记 null。
配种子（DEV-0038，D77-3；check_seed_pairing）：E5 的第 i 个种子用 E3 的第 i 个种子训练的起点。起点的种子取 <叶>/config.yaml 的 seed
（检查点 manifest 不记种子，src/adapters 不改）；不等于 E5 的种子、或找不到叶目录的 config.yaml 时抛 StartError，除非调用方明确放开
（scripts/train_e5.py --allow-seed-mismatch）；结果（match / mismatch / unverifiable / not_applicable）进运行身份。

**起点 LoRA 的处理**（merge_then_new_lora）：起点的 LoRA 先在内存里合并进基座（merge_start_lora：peft 的 LoraConfig.from_pretrained +
get_peft_model + set_peft_model_state_dict + merge_and_unload，与适配器的 load_trained(merge=True) 走同一串调用，只是不改适配器的 LoRA
状态，所以之后还能注入一份新的 LoRA），再：
- from_checkpoint：restore_tokens(起点的 token 表与行)（网格 token 行取起点训练过的值）；
- add：add_vocab(适配器, E3b 词表)（均值初始化，与 E3b 开训时逐位相同；e3a 起点的查询占位符行不恢复，E5 不用查询 token）。
然后 torch.manual_seed(种子) → 注入新 LoRA（src/e5/train.py）。合并在 manual_seed 之前做（peft 建 LoRA 层时会用随机数），所以同一个种子下
各起点的新 LoRA 初始值逐位相同。
开训前的两项核对（不过就拒绝训练）：
1. 起点检查点的合并核对（start_merge_check）：scripts/check_training_interfaces.py --checkpoint（src/e3/merge_check.run_merge_check），
   判据 v1（D50-3），检查项同它自己的评测：带网格 token 的是 grid_eval（merge + restricted），e3a 是 t_text（merge）。同一个检查点的结果
   放在 <out>/train/<起点>/<模型>/start_checks/ 下，各数据量档与种子复用。
2. 合并后的权重 = 检查点合并评测时的权重：合并起点 LoRA 之后（加网格 token、注入新 LoRA 之前）的权重指纹（stack_fingerprint：
   LoRA 作用的全部线性层的权重，原始字节）必须与 factory.load_trained(起点, merge=True) 的指纹相同（reference_fingerprint，同一台机器、
   同一 dtype；结果也放在 start_checks/ 下复用）。

**E5 检查点**（D34：不存合并后的整模型）：<ckpt>/final/model/ = save_trained（新 LoRA + 网格 token 行 + manifest），起点有检查点时另在
extra.safetensors 里存起点 manifest 的 sha256（EXTRA_KEY，uint8[32]）；<ckpt>/final/e5_stack.json（STACK_FILE）记起点的种类、路径、manifest
的 sha256、叠加后的权重指纹与当时的设备 / dtype、训练集与种子。评测时按顺序合并（load_stacked）：按 manifest 的构造方式建 base →
merge_start_lora(起点) → 核对起点 manifest 的 sha256 与 extra 里的相同（必须）、权重指纹与训练时相同（CPU 同平台时必须相同；GPU 上
跨进程的 GEMM 不保证逐位相同，只记录）→ load_trained(E5 检查点, merge=False)（restore_tokens 恢复 E5 训练后的网格 token 行，挂上 E5 LoRA；评测前的合并核对之后
再 merge_lora）。
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import yaml

from src.adapters import factory
from src.adapters.checkpoint import (
    EXTRA,
    LORA_CONFIG,
    LORA_DIR,
    LORA_WEIGHTS,
    MANIFEST,
    TOKEN_ROWS,
    TOKENS_JSON,
    base_fingerprint,
    file_sha256,
    policy_spec,
    read_manifest,
)
from src.adapters.image_policy import parse_policy
from src.e3.merge_check import run_merge_check, settings_from_config
from src.e3.tokens import GridTokens, add_vocab, attach_vocab
from src.e5.train_config import REPO_ROOT, E5TrainConfig

STACK_FILE = "e5_stack.json"
STACK_FORMAT = "visint-e5-stack/1"
EXTRA_KEY = "e5_start_manifest_sha256"
START_CHECKS = "start_checks"
REFERENCE_FORMAT = "visint-e5-start-reference/1"
_IDENT_KEYS = ("family", "model_key", "tiny", "repo", "revision")


class StartError(RuntimeError):
    """起点检查点不合规，或叠加后的模型与检查点合并评测时的不一致：拒绝训练 / 评测。"""


@dataclass(frozen=True)
class StartRef:
    kind: str
    checkpoint: Path | None
    grid_tokens: str  # from_checkpoint / add
    manifest_sha256: str | None = None
    manifest: Mapping[str, Any] | None = None
    n_grid_tokens: int = 0
    leaf_variant: str | None = None
    expected_variant: str | None = None
    leaf_config: str | None = None
    leaf_seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint is not None

    def identity(self) -> dict[str, Any]:
        """进运行身份的部分（不含路径：路径与机器有关，只进运行记录与 e5_stack.json）。"""
        m = self.manifest or {}
        return {"kind": self.kind, "grid_tokens": self.grid_tokens, "manifest_sha256": self.manifest_sha256,
                "lora": m.get("lora"), "build_spec": m.get("build_spec"), "image_policy": m.get("image_policy"),
                "dtype": m.get("dtype"), "n_grid_tokens": self.n_grid_tokens, "leaf_variant": self.leaf_variant,
                "expected_variant": self.expected_variant, "leaf_seed": self.leaf_seed}

    def record(self) -> dict[str, Any]:
        return {**self.identity(), "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
                "leaf_config": self.leaf_config}


def _leaf_config(ckpt: Path) -> tuple[dict[str, Any] | None, str | None]:
    """检查点在 <叶>/ckpt/final/model 下时，读 <叶>/config.yaml（训练的运行身份）；找不到返回 (None, None)。"""
    p = ckpt.resolve()
    if p.name != "model" or p.parent.name != "final" or p.parent.parent.name != "ckpt":
        return None, None
    cfg = p.parents[2] / "config.yaml"
    if not cfg.is_file():
        return None, None
    return (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}), str(cfg)


def _leaf_variant(ckpt: Path) -> tuple[str | None, str | None]:
    """检查点在 <叶>/ckpt/final/model 下时，读 <叶>/config.yaml 的 variant。"""
    doc, cfg = _leaf_config(ckpt)
    if doc is None:
        return None, None
    v = doc.get("variant")
    return (None if v is None else str(v)), cfg


def _leaf_seed(ckpt: Path) -> int | None:
    """起点检查点训练时的种子：<叶>/config.yaml 的 seed（检查点 manifest 不记种子；DEV-0038）。找不到为 None。"""
    doc, _ = _leaf_config(ckpt)
    s = None if doc is None else doc.get("seed")
    return None if s is None or isinstance(s, bool) else int(s)


def resolve_start(cfg: E5TrainConfig, kind: str, checkpoint: str | Path | None, *, model_key: str, tiny: bool) -> StartRef:
    """起点的种类 + 检查点目录 → StartRef（核对规则见模块说明）。"""
    spec = cfg.start(kind)
    if not spec["checkpoint"]:
        if checkpoint is not None:
            raise StartError(f"起点 {kind} 没有检查点，却给了 --start-checkpoint {checkpoint}")
        return StartRef(kind=kind, checkpoint=None, grid_tokens=str(spec["grid_tokens"]))
    if checkpoint is None:
        raise StartError(f"起点 {kind} 要给 --start-checkpoint（{spec['role']}）")
    d = Path(checkpoint)
    man = read_manifest(d)  # 逐个核对文件的 sha256 与字节数
    bs = man.get("build_spec") or {}
    if bs.get("key") != model_key or bool(bs.get("tiny")) != bool(tiny):
        raise StartError(f"起点检查点的构造方式（key {bs.get('key')!r}，tiny {bs.get('tiny')}）与 --model {model_key!r}、--tiny {tiny} 不符")
    if man.get("lora") is None:
        raise StartError(f"起点 {kind} 的检查点没有 LoRA（{d}）：C1′、C1、C2 不是 E5 的起点")
    table = json.loads((d / TOKENS_JSON).read_text(encoding="utf-8"))
    names = sorted(g["name"] for g in table["grid_tokens"])
    if spec["grid_tokens"] == "from_checkpoint":
        if names != sorted(cfg.e3b.vocab.names):
            raise StartError(f"起点 {kind} 要带 E3b 词表的 {len(cfg.e3b.vocab.names)} 个网格 token，检查点里有 {len(names)} 个")
    elif names:
        raise StartError(f"起点 {kind} 不该带网格 token（检查点里有 {len(names)} 个）")
    variant, leaf_cfg = _leaf_variant(d)
    want = str(spec["variant"])
    if variant is not None and variant != want:
        raise StartError(f"起点 {kind} 的检查点来自变体 {variant!r}（{leaf_cfg}），应为 {want!r}")
    return StartRef(kind=kind, checkpoint=d, grid_tokens=str(spec["grid_tokens"]),
                    manifest_sha256=file_sha256(d / MANIFEST), manifest=man, n_grid_tokens=len(names), leaf_variant=variant,
                    expected_variant=want, leaf_config=leaf_cfg, leaf_seed=_leaf_seed(d))


SEED_PAIRING_RULE = ("D77-3：E5 的第 i 个种子用 E3 的第 i 个种子训练的起点检查点；起点的种子取它训练叶目录 config.yaml 的 seed"
                     "（manifest 不记种子）；不符或核对不了就拒绝训练，--allow-seed-mismatch 放开（记进运行身份）")


def check_seed_pairing(start: StartRef, seed: int, *, allow_mismatch: bool) -> dict[str, Any]:
    """配种子（DEV-0038，D77-3）：起点检查点的种子必须等于 E5 的运行种子。返回进运行身份的记录；不符或核对不了时，没有
    allow_mismatch 就抛 StartError。base 起点没有检查点，不适用。"""
    rec: dict[str, Any] = {"seed": int(seed), "start_seed": start.leaf_seed, "allow_mismatch": bool(allow_mismatch),
                           "rule": SEED_PAIRING_RULE}
    if not start.has_checkpoint:
        return {**rec, "status": "not_applicable"}
    if start.leaf_seed is None:
        status = "unverifiable"
        msg = (f"起点检查点 {start.checkpoint} 的种子核对不了：它不在 <训练叶目录>/ckpt/final/model 下，或叶目录的 config.yaml "
               "没有 seed")
    elif start.leaf_seed != int(seed):
        status = "mismatch"
        msg = f"起点检查点来自种子 {start.leaf_seed}（{start.leaf_config}），E5 的 --seed 是 {seed}：E5 的第 i 个种子要用 E3 的第 i 个种子"
    else:
        return {**rec, "status": "match"}
    if not allow_mismatch:
        raise StartError(f"{msg}（D77-3 配种子）；确要这样就加 --allow-seed-mismatch（记进运行身份）")
    return {**rec, "status": status}


# ---------------------------------------------------------------------------
# 合并起点 LoRA（不改适配器的 LoRA 状态）
# ---------------------------------------------------------------------------


def check_same_base(ad: Any, man: Mapping[str, Any], where: str) -> None:
    """检查点的 base 与适配器相同：家族、模型键、是否小模型、repo、revision、图像口径、base 指纹（适配器刚挂好、还没改过权重时算）。"""
    mine = {k: v for k, v in ad.token_table().items() if k in _IDENT_KEYS}
    bad = {k: (man.get(k), v) for k, v in mine.items() if man.get(k) != v}
    if man.get("image_policy") != policy_spec(ad.image_policy):
        bad["image_policy"] = (man.get("image_policy"), policy_spec(ad.image_policy))
    fp = base_fingerprint(ad.model)
    if man.get("base_fingerprint") != fp:
        bad["base_fingerprint"] = (man.get("base_fingerprint"), fp)
    if bad:
        raise StartError(f"{where} 的 base 与适配器不符（检查点, 适配器）：{bad}")


def merge_start_lora(ad: Any, ckpt: str | Path) -> dict[str, Any]:
    """起点检查点的 LoRA 合并进适配器的基座权重（模块说明）。适配器必须刚挂好（没有 LoRA、没加过 token）。返回摘要。"""
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    d = Path(ckpt)
    man = read_manifest(d)
    if ad.lora_state != "none":
        raise StartError(f"合并起点 LoRA 要在注入任何 LoRA 之前（当前 LoRA 状态 {ad.lora_state}）")
    if ad.grid_token_ids:
        raise StartError("合并起点 LoRA 要在加网格 token 之前")
    check_same_base(ad, man, f"起点检查点 {d}")
    if man.get("lora") is None or not (d / LORA_CONFIG).is_file():
        raise StartError(f"起点检查点 {d} 没有 LoRA")
    model = ad.model
    training = model.training
    targets = sorted(ad.lora_target_modules())
    cfg = LoraConfig.from_pretrained(str(d / LORA_DIR))
    peft_model = get_peft_model(model, cfg)
    if peft_model.base_model.model is not model:
        raise StartError("peft 没有原地包装适配器的 model")
    wrapped = sorted(n for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in m.lora_A)
    if wrapped != targets:
        raise StartError(f"包上起点 LoRA 的模块（{len(wrapped)} 个）与 lora_target_modules()（{len(targets)} 个）不同")
    saved = load_file(str(d / LORA_WEIGHTS))
    res = set_peft_model_state_dict(peft_model, saved)
    if getattr(res, "unexpected_keys", None):
        raise StartError(f"起点 LoRA 权重里有本模型没有的键：{res.unexpected_keys[:3]}")
    now = get_peft_model_state_dict(peft_model, save_embedding_layers=False)
    if set(now) != set(saved) or any(not torch.equal(now[k].cpu(), saved[k].to(now[k].dtype)) for k in saved):
        raise StartError("起点 LoRA 权重没有原样载入")
    merged = peft_model.merge_and_unload()
    if merged is not model:
        raise StartError("merge_and_unload 没有返回适配器的 model")
    left = [n for n, _ in model.named_modules() if "lora_" in n]
    if left:
        raise StartError(f"合并后还有 LoRA 模块：{left[:3]}")
    model.train(training)
    if ad.lora_state != "none":
        raise StartError(f"合并起点 LoRA 之后适配器的 LoRA 状态变成了 {ad.lora_state}")
    lora = dict(man["lora"])
    return {"checkpoint": str(d), "manifest_sha256": file_sha256(d / MANIFEST), "n_targets": len(targets), "lora": lora}


FINGERPRINT_ELEMS = 4096


def stack_fingerprint(ad: Any) -> str:
    """LoRA 作用的全部线性层（lora_target_modules()，名字排序）的权重：每个取前 FINGERPRINT_ELEMS 个元素的原始字节（模型 dtype，
    不舍入），连同名字、dtype 与形状一起取 sha256。合并起点 LoRA 改的正是这些权重；必须在注入新 LoRA 之前算（之后名字里多出
    base_layer）。"""
    if ad.lora_state == "injected":
        raise StartError("stack_fingerprint 要在没有挂着未合并的 LoRA 时算（none 或 merged）")
    mods = dict(ad.model.named_modules())
    h = hashlib.sha256()
    for name in sorted(ad.lora_target_modules()):
        w = mods[name].weight.detach()
        h.update(f"{name}|{w.dtype}|{tuple(w.shape)}".encode())
        h.update(w.reshape(-1)[:FINGERPRINT_ELEMS].to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def setup_start(ad: Any, start: StartRef, cfg: E5TrainConfig) -> tuple[GridTokens, dict[str, Any]]:
    """训练开头（注入新 LoRA 之前）：合并起点 LoRA，网格 token 按起点的种类恢复或新加。返回 (网格 token 布局, 摘要)。"""
    info: dict[str, Any] = {"kind": start.kind, "merged": None}
    if start.has_checkpoint:
        info["merged"] = merge_start_lora(ad, start.checkpoint)
    info["fingerprint"] = stack_fingerprint(ad)
    if start.grid_tokens == "from_checkpoint":
        from safetensors.torch import load_file

        table = json.loads((start.checkpoint / TOKENS_JSON).read_text(encoding="utf-8"))
        ad.restore_tokens(table, load_file(str(start.checkpoint / TOKEN_ROWS)))  # 只核对、不初始化；行取起点训练过的值
        tokens = attach_vocab(ad, cfg.e3b.vocab)
        info["tokens"] = "restore_tokens（起点检查点的网格 token 行）"
    else:
        tokens = add_vocab(ad, cfg.e3b.vocab)
        info["tokens"] = "add_grid_tokens（基础词表行的均值，与 E3b 开训时相同）"
    return tokens, info


def device_key(ad: Any) -> dict[str, Any]:
    dev = torch.device(ad.device)
    return {"device_type": dev.type, "dtype": str(ad.dtype).removeprefix("torch."),
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None}


# ---------------------------------------------------------------------------
# 开训前的两项核对（模块说明）
# ---------------------------------------------------------------------------


def start_checks_dir(out: str | Path, kind: str, model_key: str) -> Path:
    return Path(out) / "train" / kind / model_key / START_CHECKS


def start_merge_check(start: StartRef, cfg: E5TrainConfig, *, model_key: str, tiny: bool, device: str, dtype: str,
                      out_dir: Path, data_root: str | Path | None, log: Callable[[str], None]) -> dict[str, Any] | None:
    """起点检查点的合并核对（D50 v1；检查项同它自己的评测）。base 返回 None；不通过抛 MergeCheckFailed。"""
    if not start.has_checkpoint:
        return None
    purpose = "grid_eval" if start.grid_tokens == "from_checkpoint" else "t_text"
    st = settings_from_config(cfg.e3b.doc, tiny=tiny, purpose=purpose)
    out = out_dir / f"merge_check_{start.manifest_sha256[:16]}.json"
    e3b_path = Path(cfg.e3b.path) if Path(cfg.e3b.path).is_absolute() else REPO_ROOT / cfg.e3b.path
    summ = run_merge_check(start.checkpoint, model_key=model_key, tiny=tiny, n=st["n"], checks=st["checks"], out_json=out,
                           device=device, dtype=dtype, data_root=data_root, dev_subsets=None, e3b_config=e3b_path, log=log)
    return {**summ, "purpose": purpose}


def reference_fingerprint(start: StartRef, *, device: str, dtype: torch.dtype, out_dir: Path,
                          log: Callable[[str], None]) -> dict[str, Any]:
    """factory.load_trained(起点, merge=True) 的权重指纹（检查点合并评测时的权重）；同一个检查点、同一台机器与 dtype 的结果复用。"""
    out = out_dir / f"reference_{start.manifest_sha256[:16]}.json"
    if out.is_file():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("manifest_sha256") == start.manifest_sha256 and old.get("format") == REFERENCE_FORMAT:
            key = old.get("device_key") or {}
            want_dev = torch.device(device).type
            if key.get("device_type") == want_dev and key.get("dtype") == str(dtype).removeprefix("torch.") and \
                    key.get("gpu") == (torch.cuda.get_device_name(torch.device(device)) if want_dev == "cuda" else None):
                log(f"[起点] 复用 {out}")
                return old
    ad, _ = factory.load_trained(start.checkpoint, device=device, dtype=dtype, merge=True)
    doc = {"format": REFERENCE_FORMAT, "manifest_sha256": start.manifest_sha256, "fingerprint": stack_fingerprint(ad),
           "lora_state": ad.lora_state, "device_key": device_key(ad),
           "rule": "factory.load_trained(起点, merge=True) 之后 src/e5/stack.stack_fingerprint：检查点合并评测时的权重"}
    del ad
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


# ---------------------------------------------------------------------------
# E5 检查点：标记、e5_stack.json、按顺序合并的加载
# ---------------------------------------------------------------------------


def marker_extra(start: StartRef) -> dict[str, torch.Tensor] | None:
    """save_trained 的 extra：起点 manifest 的 sha256（uint8[32]）；base 起点为 None。"""
    if not start.has_checkpoint:
        return None
    return {EXTRA_KEY: torch.tensor(list(bytes.fromhex(start.manifest_sha256)), dtype=torch.uint8)}


def read_marker(model_dir: str | Path) -> str | None:
    from safetensors.torch import load_file

    p = Path(model_dir) / EXTRA
    if not p.is_file():
        return None
    t = load_file(str(p)).get(EXTRA_KEY)
    return None if t is None else bytes(t.to(torch.uint8).tolist()).hex()


def write_stack_file(final_dir: Path, doc: Mapping[str, Any]) -> None:
    (final_dir / STACK_FILE).write_text(json.dumps({"format": STACK_FORMAT, **doc}, indent=1, ensure_ascii=False) + "\n",
                                        encoding="utf-8")


def read_stack_file(final_dir: str | Path) -> dict[str, Any]:
    p = Path(final_dir) / STACK_FILE
    if not p.is_file():
        raise StartError(f"{final_dir} 里没有 {STACK_FILE}：不是 E5 训练的检查点（--checkpoint 给 <训练叶目录>/ckpt/final）")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("format") != STACK_FORMAT:
        raise StartError(f"{p} 的格式 {doc.get('format')!r} 不是 {STACK_FORMAT}")
    return doc


@dataclass
class Stacked:
    adapter: Any
    stack: dict[str, Any]
    manifest: dict[str, Any]
    model_dir: Path
    start_checkpoint: Path | None
    fingerprint: dict[str, Any]


def load_stacked(final_dir: str | Path, *, device: str, dtype: torch.dtype, start_override: str | Path | None = None,
                 log: Callable[[str], None] | None = None) -> Stacked:
    """评测的加载（模块说明）：base → 合并起点 LoRA → 核对 → load_trained(E5 检查点, merge=False)。返回时 E5 LoRA 已注入、未合并。"""
    log = log or (lambda m: None)
    fd = Path(final_dir)
    doc = read_stack_file(fd)
    model_dir = fd / doc["model_dir"]
    man = read_manifest(model_dir)
    spec = man.get("build_spec") or {}
    if not spec:
        raise StartError(f"{model_dir} 的 manifest 没有构造方式")
    ad = factory.make_adapter(spec["key"], device=device, dtype=dtype, image_policy=parse_policy(spec["image_policy"]),
                              tiny=bool(spec["tiny"]), **dict(spec["family_kw"]))
    st = doc["start"]
    marker = read_marker(model_dir)
    start_ck = None
    if st["checkpoint"] is not None or st["manifest_sha256"] is not None:
        start_ck = Path(start_override) if start_override is not None else Path(st["checkpoint"])
        sha = file_sha256(start_ck / MANIFEST)
        if sha != st["manifest_sha256"] or marker != sha:
            raise StartError(f"起点检查点 {start_ck} 的 manifest sha256 {sha[:16]}… 与 E5 检查点记下的不同"
                             f"（e5_stack.json {str(st['manifest_sha256'])[:16]}…，extra {str(marker)[:16]}…）")
        merge_start_lora(ad, start_ck)
    elif marker is not None:
        raise StartError("E5 检查点带起点标记，e5_stack.json 却说没有起点")
    fp = stack_fingerprint(ad)
    trained = doc.get("fingerprint") or {}
    same_platform = trained.get("device_key") == device_key(ad)
    match = fp == trained.get("value")
    # CPU 上合并逐位可复现：同平台必须相同。GPU 上跨进程的 GEMM 不保证逐位相同，起点的身份已由 manifest sha256 与 extra 的标记钉住、
    # 合并走同一串调用，这里只记录（训练开头在同一个进程里另有与 factory.load_trained(merge=True) 的严格比较）
    if same_platform and not match and torch.device(ad.device).type == "cpu":
        raise StartError(f"叠加后的权重指纹 {fp[:16]}… 与训练时 {str(trained.get('value'))[:16]}… 不同（同一设备类型、dtype）")
    if not match:
        log(f"[叠加] 权重指纹与训练时不同（训练 {trained.get('device_key')}，本机 {device_key(ad)}）：只记录")
    ad.load_trained(model_dir, merge=False)
    if man.get("lora") is not None and ad.lora_state != "injected":
        raise StartError(f"load_trained(merge=False) 之后 LoRA 状态是 {ad.lora_state}")
    return Stacked(adapter=ad, stack=doc, manifest=man, model_dir=model_dir, start_checkpoint=start_ck,
                   fingerprint={"value": fp, "trained": trained.get("value"), "match": match, "same_platform": same_platform,
                                "device_key": device_key(ad)})


__all__ = ["EXTRA_KEY", "SEED_PAIRING_RULE", "STACK_FILE", "STACK_FORMAT", "StartError", "StartRef", "Stacked", "check_same_base",
           "check_seed_pairing", "device_key",
           "load_stacked", "marker_extra", "merge_start_lora", "read_marker", "read_stack_file", "reference_fingerprint",
           "resolve_start", "setup_start", "stack_fingerprint", "start_checks_dir", "start_merge_check", "write_stack_file"]
