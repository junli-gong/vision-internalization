"""Rex-Omni 的模型一侧（DEV-0083）：权重清单与核对、分词器的重建、处理器、真模型与 tiny 随机模型、一批生成。

**权重**：configs/rexomni_weights.json（repo、revision、逐文件字节数与 sha256；大文件的 sha256 取自 HF 的 LFS 元数据，小文件现算，
并与 HF 树上的 git blob 哈希逐个对上）。verify_weights 运行前核对：文件恰好是清单里的、字节数相同、full_hash 时 sha256 也相同。

**分词器**（build_tokenizer）：Rex-Omni 只发了慢分词器的文件（vocab.json、merges.txt、tokenizer_config.json，没有 tokenizer.json）。
它把 1,000 个坐标 token <0>…<999> 放在 id 150643–151642，**顶掉** vocab.json 里同 id 的 1,000 个罕见 BPE token（tokenizer_config.json
的 added_tokens_decoder），<|endoftext|> 等 22 个特殊 token 在 151643–151664。官方评测用慢分词器（inference_text_prompt.py:527
tokenizer_mode="slow"、wrapper.py:170 use_fast=False，transformers 4.51.3），慢分词器按 added_tokens_decoder 的 id 编解码。
transformers 5 只有 tokenizers 后端，直接读这些文件时把 1,022 个新增 token 接在 vocab 后面（<0> 成了 151643、<|im_end|> 成了 152645，
与模型的嵌入行整体错开 1,000，2026-11-04 在 transformers 5.17.0 上实测）。所以这里照慢分词器的语义重建：
- vocab 里那 1,000 个 id 换成坐标 token，产出被顶掉的 token 的 1,000 条 merge 去掉（tokenizers 要求 merge 的结果在 vocab 里）；
- 预分词、NFC、ByteLevel 同 transformers 的 Qwen2Tokenizer；
- 新增 token 按 id 升序加入（已在 vocab 里的复用原 id，其余接在后面），之后逐个核对：1,022 个 token 的 id 都等于 added_tokens_decoder 的、
  config.json 的 eos / pad / 图像与视觉 token id 都对得上、词表不超过 config.vocab_size（check_tokenizer，入口断言）。
- 与慢分词器的差别只在含有被顶掉的那 1,000 个罕见 BPE token 的文本上（慢分词器把它们编成坐标 token 的 id，本身就是撞号）。提示与类名里
  没有这些 token。

**处理器**：官方 Qwen2_5_VLProcessor（图像与文本路径都是官方代码），图像处理器固定用 PIL 后端（官方 use_fast=False；transformers 5 里是
Qwen2VLImageProcessorPil），min/max pixels 照官方。开发机没有 torchvision、视频处理器构造不出来，构造期间临时跳过那一项类型检查、用
BaseVideoProcessor() 占位（同 src/adapters/qwen35.reference_processor；本运行器不送视频）。

**入口断言**（每批生成之前，任何机器上都会触发，抛 RexOmniInputError 并报出两个值）：
- 每张图的 image_grid_thw 等于 protocol.expected_grid（smart_resize，min/max pixels 照官方）；
- 每行 <|image_pad|> 的个数等于这张图的视觉 token 数；
- 输入张量与模型在同一个 device（check_devices；开发机上用 meta 设备做可观测代理）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.rexomni import protocol as P

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_MANIFEST_PATH = REPO_ROOT / "configs" / "rexomni_weights.json"
# chat_template.json（rev 0e5693d2）里 chat_template 字符串的 sha256；真模型加载时核对，tiny 用同一个模板（TINY_CHAT_TEMPLATE）
CHAT_TEMPLATE_SHA256 = "a0bc6f6fc7a29a80017a433e8f03a1cc1236e838a944a2d034295a60c4f2fddb"
# 同一个字符串（照 chat_template.json 原样；Qwen2-VL 的标准模板）。tiny 的处理器用它；tests/test_rexomni.py 钉住 sha256。
TINY_CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}"
    "{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# transformers 的 Qwen2Tokenizer（models/qwen2/tokenization_qwen2.py 的 PRETOKENIZE_REGEX，与慢分词器的 pat 相同）
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
CONFIG_TOKEN_IDS: Mapping[str, str] = {"eos_token_id": "<|im_end|>", "pad_token_id": "<|endoftext|>", "image_token_id": "<|image_pad|>",
                                       "video_token_id": "<|video_pad|>", "vision_start_token_id": "<|vision_start|>",
                                       "vision_end_token_id": "<|vision_end|>", "vision_token_id": "<|vision_pad|>"}

# tiny（开发机彩排）：小型随机 Qwen2.5-VL；分词器取登记的 Qwen3.5-0.8B（HF 缓存里有，不下载权重）再加 1,000 个坐标 token
TINY_TOKENIZER_KEY = "qwen3.5-0.8b"
TINY_TEXT: Mapping[str, Any] = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 2,
                                "num_key_value_heads": 1, "max_position_embeddings": 4096, "rope_theta": 1000000.0,
                                "rope_scaling": {"type": "default", "rope_type": "default", "mrope_section": [4, 6, 6]},
                                "tie_word_embeddings": True}
TINY_VISION: Mapping[str, Any] = {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
                                  "patch_size": P.PATCH_SIZE, "spatial_merge_size": P.MERGE_SIZE, "temporal_patch_size": 2,
                                  "window_size": 112, "fullatt_block_indexes": [0], "in_channels": 3, "hidden_act": "silu",
                                  "tokens_per_second": 2}
TINY_MAX_NEW_TOKENS = 24  # 随机模型不会自己停，开发机上只生成这么多（进运行身份）


class RexOmniError(RuntimeError):
    """用法、身份、数据或环境不合格（运行器以退出码 2 退出）。"""


class WeightsError(RexOmniError):
    """权重目录与 configs/rexomni_weights.json 不符。"""


class TokenizerError(RexOmniError):
    """重建的分词器与官方文件的 id 对不上。"""


class RexOmniInputError(RexOmniError):
    """入口断言：送进模型的输入与约定不符（网格、视觉 token 数、device）。"""


# ---------------------------------------------------------------------------
# 权重
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_weights_manifest(path: str | Path = WEIGHTS_MANIFEST_PATH) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("repo") != P.WEIGHTS_REPO or doc.get("revision") != P.WEIGHTS_REVISION or not doc.get("files"):
        raise WeightsError(f"{path}：repo / revision 不是 {P.WEIGHTS_REPO}@{P.WEIGHTS_REVISION}，或没有 files")
    for f in doc["files"]:
        if set(f) != {"path", "bytes", "sha256"} or len(str(f["sha256"])) != 64:
            raise WeightsError(f"{path}：文件条目 {f} 不合格")
    return doc


def weights_total_bytes(doc: Mapping[str, Any]) -> int:
    return int(sum(int(f["bytes"]) for f in doc["files"]))


def verify_weights(directory: str | Path, doc: Mapping[str, Any], *, full_hash: bool = True) -> dict[str, Any]:
    """权重目录（HF 快照目录或同内容的本地目录）必须恰好是清单里的文件（隐藏文件除外），字节数相同；full_hash 时 sha256 也逐个相同。
    返回运行身份里的一节。"""
    d = Path(directory)
    if not d.is_dir():
        raise WeightsError(f"权重目录 {d} 不存在")
    want = {f["path"]: f for f in doc["files"]}
    have = sorted(p.relative_to(d).as_posix() for p in d.rglob("*")
                  if (p.is_file() or p.is_symlink()) and not any(part.startswith(".") for part in p.relative_to(d).parts))
    if sorted(want) != have:
        raise WeightsError(f"{d}：文件 {have} 与 configs/rexomni_weights.json 的 {sorted(want)} 不同")
    for rel, f in sorted(want.items()):
        p = d / rel
        if p.stat().st_size != int(f["bytes"]):
            raise WeightsError(f"{p}：{p.stat().st_size} 字节，清单是 {f['bytes']}")
        if full_hash:
            h = sha256_file(p)
            if h != f["sha256"]:
                raise WeightsError(f"{p}：sha256 {h[:16]}… 与清单的 {f['sha256'][:16]}… 不同")
    return {"repo": doc["repo"], "revision": doc["revision"], "dir": str(d), "n_files": len(want), "total_bytes": weights_total_bytes(doc),
            "sha256_verified": bool(full_hash), "manifest_sha256": sha256_file(WEIGHTS_MANIFEST_PATH) if WEIGHTS_MANIFEST_PATH.is_file() else None}


def snapshot_dir(doc: Mapping[str, Any], *, download: bool) -> Path:
    """HF 快照目录（HF_HUB_CACHE 下；只取清单里的文件）。download = False 时只读本地缓存（HF_HUB_OFFLINE）。"""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(doc["repo"], revision=doc["revision"], allow_patterns=[f["path"] for f in doc["files"]],
                                  local_files_only=not download))


# ---------------------------------------------------------------------------
# 分词器
# ---------------------------------------------------------------------------


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def build_tokenizer(model_dir: str | Path) -> Any:
    """照慢分词器的语义重建（模块说明）。返回 transformers 的分词器（tokenizers 后端），左侧补齐。"""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    d = Path(model_dir)
    vocab: dict[str, int] = _read_json(d / "vocab.json")
    cfg = _read_json(d / "tokenizer_config.json")
    added = {int(k): v for k, v in cfg["added_tokens_decoder"].items()}
    inv = {i: t for t, i in vocab.items()}
    removed = set()
    for i in sorted(added):
        if i in inv:  # 坐标 token 顶掉同 id 的 BPE token（慢分词器的 added_tokens_decoder 优先）
            removed.add(inv[i])
            del vocab[inv[i]]
            vocab[added[i]["content"]] = i
    lines = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
    body = lines[1:] if lines and lines[0].startswith("#version") else lines
    merges = [tuple(x.split(" ")) for x in body if x]
    keep = [m for m in merges if m[0] not in removed and m[1] not in removed and (m[0] + m[1]) not in removed]
    tk = Tokenizer(BPE(vocab=vocab, merges=keep, dropout=None, unk_token=None, continuing_subword_prefix="", end_of_word_suffix="",
                       fuse_unk=False, byte_fallback=False))
    tk.decoder = decoders.ByteLevel()
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    for i in sorted(added):
        a = added[i]
        t = AddedToken(a["content"], single_word=bool(a.get("single_word")), lstrip=bool(a.get("lstrip")), rstrip=bool(a.get("rstrip")),
                       normalized=bool(a.get("normalized")), special=bool(a.get("special")))
        (tk.add_special_tokens if a.get("special") else tk.add_tokens)([t])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token=None, eos_token=cfg.get("eos_token"), pad_token=cfg.get("pad_token"),
                                  unk_token=None, clean_up_tokenization_spaces=False, padding_side=P.PADDING_SIDE,
                                  model_max_length=int(cfg.get("model_max_length", P.MAX_MODEL_LEN)))
    tok._rexomni_rebuild = {"n_added": len(added), "n_vocab_overridden": len(removed), "n_merges": len(merges), "n_merges_kept": len(keep)}
    return tok


def check_tokenizer(tok: Any, *, added_tokens: Mapping[int, str], config: Mapping[str, Any]) -> dict[str, Any]:
    """入口断言：每个新增 token 的 id 等于 added_tokens（tokenizer_config.json 的 added_tokens_decoder）；config 的特殊 token id
    对得上；词表不超过 config 的 vocab_size；坐标 token 往返。返回进运行身份的一节。"""
    bad = [(c, i, tok.convert_tokens_to_ids(c)) for i, c in sorted(added_tokens.items()) if tok.convert_tokens_to_ids(c) != i]
    if bad:
        raise TokenizerError(f"{len(bad)} 个新增 token 的 id 与官方不同，例如 {bad[:3]}（transformers 5 直接读慢分词器文件的错位？）")
    text_cfg = config.get("text_config") or {}
    for key, content in CONFIG_TOKEN_IDS.items():
        want = config.get(key, text_cfg.get(key))
        if want is None:
            continue
        got = tok.convert_tokens_to_ids(content)
        if int(want) != got:
            raise TokenizerError(f"config.{key} = {want}，分词器里 {content} 是 {got}")
    vs = config.get("vocab_size", text_cfg.get("vocab_size"))
    if vs is not None and len(tok) > int(vs):
        raise TokenizerError(f"分词器 {len(tok)} 个 token，多于模型的 vocab_size {vs}")
    sample = P.format_boxes("person", [[0, 35, 980, 987], [646, 0, 999, 940]])
    ids = tok(sample, add_special_tokens=False).input_ids
    back = tok.decode(ids, skip_special_tokens=P.SKIP_SPECIAL_TOKENS, clean_up_tokenization_spaces=P.CLEAN_UP_TOKENIZATION_SPACES)
    zero = tok.convert_tokens_to_ids("<0>")
    if back != sample or ids.count(zero) != 2:
        raise TokenizerError(f"坐标 token 往返不对：{sample!r} → {ids} → {back!r}")
    return {"n_tokens": len(tok), "n_added_checked": len(added_tokens), "coord_token_ids": [zero, tok.convert_tokens_to_ids(f"<{P.BIN_MAX}>")],
            "eos_token_id": tok.convert_tokens_to_ids(P.EOS_TEXT), "rebuild": dict(getattr(tok, "_rexomni_rebuild", {}) or {})}


# ---------------------------------------------------------------------------
# 处理器
# ---------------------------------------------------------------------------


def pil_image_processor_class() -> type:
    """PIL 后端的 Qwen2-VL 图像处理器（官方 use_fast=False）：transformers 5 是 Qwen2VLImageProcessorPil；4.x 的慢处理器叫
    Qwen2VLImageProcessor（5 里同名的是 torchvision 后端，不用）。"""
    try:
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

        return Qwen2VLImageProcessorPil
    except ImportError:
        import transformers

        if int(transformers.__version__.split(".")[0]) >= 5:
            raise
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

        return Qwen2VLImageProcessor


def check_image_processor(ip: Any) -> dict[str, Any]:
    size = getattr(ip, "size", None)
    lo = getattr(size, "shortest_edge", None) if size is not None and not isinstance(size, dict) else (size or {}).get("shortest_edge")
    hi = getattr(size, "longest_edge", None) if size is not None and not isinstance(size, dict) else (size or {}).get("longest_edge")
    lo = lo if lo is not None else getattr(ip, "min_pixels", None)
    hi = hi if hi is not None else getattr(ip, "max_pixels", None)
    got = (lo, hi, int(ip.patch_size), int(ip.merge_size))
    want = (P.MIN_PIXELS, P.MAX_PIXELS, P.PATCH_SIZE, P.MERGE_SIZE)
    if got != want:
        raise RexOmniError(f"图像处理器的 (min_pixels, max_pixels, patch, merge) = {got}，官方是 {want}")
    return {"class": f"{type(ip).__module__}.{type(ip).__name__}", "min_pixels": lo, "max_pixels": hi, "patch_size": int(ip.patch_size),
            "merge_size": int(ip.merge_size), "resample": int(getattr(ip, "resample", -1))}


def build_processor(tokenizer: Any, image_processor: Any, chat_template: str) -> Any:
    """官方 Qwen2_5_VLProcessor（模块说明：视频处理器占位）。"""
    from unittest import mock

    from transformers import Qwen2_5_VLProcessor
    from transformers.processing_utils import ProcessorMixin
    from transformers.video_processing_utils import BaseVideoProcessor

    with mock.patch.object(ProcessorMixin, "check_argument_for_proper_class", lambda self, name, arg: type(arg)):
        return Qwen2_5_VLProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=BaseVideoProcessor(),
                                   chat_template=chat_template)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


def model_device(model: Any) -> Any:
    return next(model.parameters()).device


def check_devices(device: Any, tensors: Mapping[str, Any]) -> None:
    """入口断言：每个输入张量都在模型的 device 上（开发机上用 meta 设备测）。"""
    bad = {k: str(v.device) for k, v in tensors.items() if hasattr(v, "device") and str(v.device) != str(device)}
    if bad:
        raise RexOmniInputError(f"输入张量不在模型的 device {device} 上：{bad}")


@dataclass
class Engine:
    """一个加载好的 Rex-Omni（或 tiny 随机模型）：generate 一批（图像, 所问的类别）。"""

    model: Any
    processor: Any
    tokenizer: Any
    info: dict[str, Any]
    tiny: bool = False
    max_new_tokens_cap: int | None = None  # tiny 时 TINY_MAX_NEW_TOKENS
    stop_strings: tuple[str, ...] = P.STOP_STRINGS
    _calls: int = field(default=0, repr=False)
    _stop: Any = field(default=None, repr=False)

    def stopping_criteria(self) -> Any:
        """停止串（transformers 的 StopStringCriteria，按解码后的文本认，同 vLLM 的 stop）：只建一次（每次建要约 0.45 s，15 万词表）。"""
        if self._stop is None:
            from transformers import StoppingCriteriaList, StopStringCriteria

            self._stop = StoppingCriteriaList([StopStringCriteria(tokenizer=self.tokenizer, stop_strings=list(self.stop_strings))])
        return self._stop

    def generate(self, items: Sequence[tuple[Any, Sequence[str]]]) -> list[dict[str, Any]]:
        """items：[(PIL RGB 图, 所问的类别（按提示里的顺序；joint 是该图的全部正类别，per_category 只有一个）)]。一批生成（左侧补齐，贪心，
        repetition_penalty 1.05，停止串与 eos），每条返回
        {text（截掉停止串，同 vLLM）, raw_text, stop_reason（eos | stop_string | max_new_tokens）, n_prompt_tokens, n_generated_tokens,
        n_visual_tokens, image_grid_thw, max_new_tokens, prompt, wall_s（本批墙钟 / 条数）, batch_wall_s, batch_size}。"""
        import torch

        if not items:
            return []
        for _, cats in items:
            if isinstance(cats, str) or not cats:
                raise RexOmniInputError(f"所问的类别必须是非空的类别列表，收到 {cats!r}")
        prompts = [P.detect_prompt(list(cats)) for _, cats in items]
        texts = [self.processor.apply_chat_template(P.messages(list(cats)), tokenize=False, add_generation_prompt=True) for _, cats in items]
        images = [img for img, _ in items]
        t0 = time.perf_counter()
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        grids = [tuple(int(v) for v in g) for g in inputs["image_grid_thw"].tolist()]
        pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        for k, (img, g) in enumerate(zip(images, grids)):
            want = P.expected_grid(img.size[0], img.size[1])
            if g != want:
                raise RexOmniInputError(f"第 {k} 张图 {img.size}：image_grid_thw {g}，smart_resize 应是 {want}")
            n_pad = int((inputs["input_ids"][k] == pad_id).sum())
            if n_pad != P.n_visual_tokens(g):
                raise RexOmniInputError(f"第 {k} 条：<|image_pad|> {n_pad} 个，网格 {g} 应是 {P.n_visual_tokens(g)} 个")
        dev = model_device(self.model)
        inputs = inputs.to(dev)
        check_devices(dev, dict(inputs))
        attn = inputs["attention_mask"]
        n_prompt = [int(v) for v in attn.sum(dim=1).tolist()]
        mnt = P.max_new_tokens_for(int(attn.shape[1]))
        if self.max_new_tokens_cap is not None:
            mnt = min(mnt, int(self.max_new_tokens_cap))
        eos = self.tokenizer.convert_tokens_to_ids(P.EOS_TEXT)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=mnt, do_sample=False, num_beams=1, repetition_penalty=P.REPETITION_PENALTY,
                                      eos_token_id=eos, pad_token_id=eos, stopping_criteria=self.stopping_criteria(),
                                      temperature=None, top_p=None, top_k=None)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        wall = time.perf_counter() - t0
        res = []
        plen = int(inputs["input_ids"].shape[1])
        for k in range(len(items)):
            gen = out[k, plen:].tolist()
            first_eos = gen.index(eos) if eos in gen else len(gen)
            raw = self.tokenizer.decode(gen[:first_eos], skip_special_tokens=P.SKIP_SPECIAL_TOKENS,
                                        clean_up_tokenization_spaces=P.CLEAN_UP_TOKENIZATION_SPACES)
            text, hit = P.truncate_at_stop(raw)
            if hit is not None:
                reason = "stop_string"
            elif first_eos < len(gen):
                reason = "eos"
            else:
                reason = "max_new_tokens"
            res.append({"prompt": prompts[k], "raw_text": raw, "text": text, "stop_reason": reason, "stop_string": hit,
                        "n_prompt_tokens": n_prompt[k], "n_generated_tokens": first_eos + int(reason == "eos"),
                        "n_visual_tokens": P.n_visual_tokens(grids[k]), "image_grid_thw": list(grids[k]), "max_new_tokens": mnt,
                        "wall_s": wall / len(items), "batch_wall_s": wall, "batch_size": len(items)})
        self._calls += 1
        return res


def load_model_weights(model_dir: str | Path, *, dtype: str) -> tuple[Any, dict[str, Any]]:
    """Qwen2_5_VLForConditionalGeneration.from_pretrained（wrapper.py:144–151；sdpa）。Rex-Omni 的权重是 transformers 4.51 的键名
    （model.layers.*、visual.*、lm_head），transformers 5 加载时改名（2026-11-04 用同格式的小检查点实测：0 个缺失、0 个多余、逐个相等）。
    入口断言：缺失、多余、形状不符的键一个都不许有（否则那些参数是随机初始化的，输出静默变坏）；参数的 dtype 必须是要的。"""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    td = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model, li = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir), dtype=td, attn_implementation=P.ATTN_IMPLEMENTATION,
                                                                    output_loading_info=True)
    bad = {k: list(li.get(k) or [])[:5] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if li.get(k)}
    if bad:
        raise RexOmniError(f"{model_dir} 的权重没有完整加载（这些参数会是随机的）：{bad}")
    got = next(model.parameters()).dtype
    if got != td:
        raise RexOmniError(f"模型参数的 dtype {got} 不是 {td}")
    return model, {k: len(li.get(k) or []) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")}


def load_real(model_dir: str | Path, *, device: str, dtype: str = P.TORCH_DTYPE, weights: Mapping[str, Any] | None = None) -> Engine:
    """真模型：官方的 Qwen2_5_VLForConditionalGeneration（wrapper.py:141–151），bf16，sdpa；处理器与分词器见模块说明。"""
    d = Path(model_dir)
    cfg = _read_json(d / "config.json")
    tcfg = _read_json(d / "tokenizer_config.json")
    pre = _read_json(d / "preprocessor_config.json")
    if (pre.get("min_pixels"), pre.get("max_pixels")) != (P.MIN_PIXELS, P.MAX_PIXELS):
        raise RexOmniError(f"preprocessor_config.json 的 min/max pixels {pre.get('min_pixels')}/{pre.get('max_pixels')} 与官方评测的不同")
    chat = _read_json(d / "chat_template.json")["chat_template"]
    if hashlib.sha256(chat.encode("utf-8")).hexdigest() != CHAT_TEMPLATE_SHA256:
        raise RexOmniError("chat_template.json 的模板与 rev 0e5693d2 的不同")
    tok = build_tokenizer(d)
    tok_info = check_tokenizer(tok, added_tokens={int(k): v["content"] for k, v in tcfg["added_tokens_decoder"].items()}, config=cfg)
    ip = pil_image_processor_class().from_pretrained(str(d), min_pixels=P.MIN_PIXELS, max_pixels=P.MAX_PIXELS)
    ip_info = check_image_processor(ip)
    proc = build_processor(tok, ip, chat)
    model, load_info = load_model_weights(d, dtype=dtype)
    model = model.to(device).eval()
    info = {"kind": "real", "dir": str(d), "weights": dict(weights) if weights else None, "dtype": dtype, "device": str(model_device(model)),
            "attn_implementation": P.ATTN_IMPLEMENTATION, "tokenizer": tok_info, "image_processor": ip_info,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256, "model_class": type(model).__name__, "loading": load_info}
    return Engine(model=model, processor=proc, tokenizer=tok, info=info)


def tiny_tokenizer() -> Any:
    """tiny 的分词器：登记的 Qwen3.5-0.8B（固定 revision，HF 缓存），加 1,000 个坐标 token（特殊 token，接在词表后面）。"""
    from tokenizers import AddedToken
    from transformers import AutoTokenizer

    from src.adapters import registry  # 只为取分词器的来源（repo 与 revision）

    e = registry.get_model(TINY_TOKENIZER_KEY)
    tok = AutoTokenizer.from_pretrained(e.repo, revision=e.revision)
    tok.add_tokens([AddedToken(f"<{i}>", special=True, normalized=False) for i in range(P.BIN_MAX + 1)], special_tokens=True)
    tok.padding_side = P.PADDING_SIDE
    tok._rexomni_source = {"repo": e.repo, "revision": e.revision}
    return tok


def tiny_config(tok: Any) -> Any:
    from transformers import Qwen2_5_VLConfig

    ids = {k: tok.convert_tokens_to_ids(c) for k, c in CONFIG_TOKEN_IDS.items()}
    return Qwen2_5_VLConfig(model_type="qwen2_5_vl", vocab_size=len(tok), **dict(TINY_TEXT), **ids,
                            vision_config=dict(TINY_VISION))  # 初始化用 Qwen2.5-VL 的缺省 initializer_range


def build_tiny(*, device: str = "cpu", dtype: str = "float32", seed: int = 0) -> Engine:
    """tiny：小型随机 Qwen2.5-VL（2 层、hidden 64，视觉塔 1 层），PIL 图像处理器（官方的 min/max pixels），官方的聊天模板。"""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration

    tok = tiny_tokenizer()
    cfg = tiny_config(tok)
    tok_info = check_tokenizer(tok, added_tokens={tok.convert_tokens_to_ids(f"<{i}>"): f"<{i}>" for i in range(P.BIN_MAX + 1)},
                               config=cfg.to_dict())
    ip = pil_image_processor_class()(min_pixels=P.MIN_PIXELS, max_pixels=P.MAX_PIXELS, patch_size=P.PATCH_SIZE, merge_size=P.MERGE_SIZE,
                                     temporal_patch_size=2)
    ip_info = check_image_processor(ip)
    proc = build_processor(tok, ip, TINY_CHAT_TEMPLATE)
    td = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    torch.manual_seed(seed)
    model = Qwen2_5_VLForConditionalGeneration(cfg).to(dtype=td, device=device).eval()
    info = {"kind": "tiny", "tokenizer_source": getattr(tok, "_rexomni_source", None), "seed": seed, "dtype": dtype, "device": str(model_device(model)),
            "text": dict(TINY_TEXT), "vision": dict(TINY_VISION), "max_new_tokens_cap": TINY_MAX_NEW_TOKENS, "tokenizer": tok_info,
            "image_processor": ip_info, "chat_template_sha256": hashlib.sha256(TINY_CHAT_TEMPLATE.encode("utf-8")).hexdigest(),
            "model_class": type(model).__name__}
    return Engine(model=model, processor=proc, tokenizer=tok, info=info, tiny=True, max_new_tokens_cap=TINY_MAX_NEW_TOKENS)


__all__ = [
    "CHAT_TEMPLATE_SHA256", "Engine", "RexOmniError", "RexOmniInputError", "TINY_CHAT_TEMPLATE", "TINY_MAX_NEW_TOKENS", "TokenizerError",
    "WEIGHTS_MANIFEST_PATH", "WeightsError", "build_processor", "build_tiny", "build_tokenizer", "check_devices", "check_image_processor",
    "check_tokenizer", "load_model_weights", "load_real", "load_weights_manifest", "pil_image_processor_class", "snapshot_dir", "verify_weights",
    "weights_total_bytes",
]
