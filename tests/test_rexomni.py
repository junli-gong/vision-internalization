"""DEV-0083：Rex-Omni 外部参照（src/rexomni/、scripts/run_rexomni.py）。

- 解析：官方格式的样例字符串（rex_omni/parser.py:39、inference_text_prompt.py:136 的示例），多框、空、格式错、越界、截断、停止串、
  不合法的框、同名组的键序、类名归一化、别的类名、100 个的上限；
- 坐标换算：网格 → 像素 → 网格（官方 int 截断，差至多 1）、像素 → 网格 → 像素（误差不超过一格）；smart_resize 与真处理器的网格；
- 出处：给了官方仓库的签出（REXOMNI_UPSTREAM_DIR）时，CITATIONS 逐条逐字核对；协议常量与引文一致；
- 分词器的重建（合成的小词表：顶掉 vocab 的新增 token 复用原 id、merge 去掉、错位时 check_tokenizer 拒绝）；给了真模型文件
  （REXOMNI_MODEL_DIR）时核对 1,022 个 id；
- 权重清单与 verify_weights；入口断言（device、网格）；
- 划分与 gate 的拒绝（xm_decision、缺 --xm-gate、彩排的 gate……）；
- 端到端：假模型（官方格式的真值框）在合成对上 F1 = 1、重复框记 FP、crowd 上的框忽略；tiny 随机 Qwen2.5-VL 的命令行跑通、叶目录能被
  src/xm/metrics.table 读、续跑逐字节相同、身份不同拒绝续跑；经 load_pairs 的 xm_final（gate 放行）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.rexomni import model as RM
from src.rexomni import protocol as P
from src.rexomni import runner as RR
from src.xm import data as XD
from src.xm import gate as XG
from src.xm import metrics as XMM
from tests import _rexomni_fakes as RF
from tests import _xm_run_fakes as F

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "scripts" / "run_rexomni.py"
W, H = 640, 480


def _load_cli():
    spec = importlib.util.spec_from_file_location("_run_rexomni_for_tests", RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLI = _load_cli()


def run(args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(RUN), *map(str, args)], capture_output=True, text=True, cwd=REPO, env=env, timeout=900)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def boxes_for(text: str, cat: str = "person", w: int = W, h: int = H, max_dets: int = XMM.MAX_DETS):
    ext, info = P.parse_prediction(P.truncate_at_stop(text)[0])
    b, q, st = P.select_boxes(ext, cat, w, h, max_dets=max_dets)
    return b, q, {**info, **st}


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

OFFICIAL_EXAMPLE = "<|object_ref_start|>person<|object_ref_end|><|box_start|><0><35><980><987>, <646><0><999><940><|box_end|>"  # parser.py:39


def test_parse_official_example_multi_box():
    b, q, st = boxes_for(OFFICIAL_EXAMPLE + "<|im_end|>")
    assert q == [[0, 35, 980, 987], [646, 0, 999, 940]]
    assert b[0] == [(0 / 999.0) * W, (35 / 999.0) * H, (980 / 999.0) * W, (987 / 999.0) * H]
    assert b[1][2] == W and b[1][1] == 0.0  # 网格 999 = 原图的右边
    assert st["n_groups"] == 1 and st["n_boxes_parsed"] == 2 and not st["incomplete"] and st["n_not_box"] == 0
    # 没有收尾的 <|im_end|> 也一样；<|im_end|> 之后的东西不看
    assert boxes_for(OFFICIAL_EXAMPLE)[1] == q
    assert boxes_for(OFFICIAL_EXAMPLE + "<|im_end|>" + OFFICIAL_EXAMPLE.replace("<0><35>", "<1><36>"))[1] == q


def test_parse_empty_and_no_group():
    for text in ("", "<|im_end|>", "There are none.", "None"):
        b, q, st = boxes_for(text)
        assert b == [] and st["n_groups"] == 0
    # 有组、框列表为空：模型说"没有"，不是解析失败
    b, _, st = boxes_for("<|object_ref_start|>person<|object_ref_end|><|box_start|><|box_end|><|im_end|>")
    assert b == [] and st["n_groups"] == 1


def test_parse_malformed():
    # inference_text_prompt.py:136 docstring 里的方括号写法：没有 <数字> token，官方的解析认不出框（组在，框为空）
    doc = ("<|object_ref_start|>cell phone<|object_ref_end|><|box_start|>[[298, 143, 365, 279]]<|box_end|>, <|object_ref_start|>person"
           "<|object_ref_end|><|box_start|>[[1, 10, 419, 636], [263, 18, 419, 298]]<|box_end|><|im_end|>")
    b, _, st = boxes_for(doc)
    assert b == [] and st["n_groups"] == 2 and st["n_coord_lists"] == 0
    # 3 个数：不是框；6 个数：前 4 个是合法的框，剩下 2 个是点（丢掉）；点、多边形都不进框
    ext, _ = P.parse_prediction("<|object_ref_start|>person<|object_ref_end|><|box_start|><1><2><3>, <1><2><3><4><5><6>, <7><8><|box_end|>")
    assert ext == {"person": [[1, 2, 3], [1, 2, 3, 4], [5, 6], [7, 8]]}
    b, q, st = P.select_boxes(ext, "person", W, H, max_dets=100)
    assert q == [[1, 2, 3, 4]] and st["n_not_box"] == 3 and st["n_coord_lists"] == 4
    # 不合法的框（x1 ≤ x0）：官方把剩下的数整串当坐标；恰好 4 个时仍是"框"（宽为负，必然 FP），记 n_inverted
    b, q, st = boxes_for("<|object_ref_start|>person<|object_ref_end|><|box_start|><500><500><100><900><|box_end|>")
    assert q == [[500, 500, 100, 900]] and st["n_inverted"] == 1
    ext, _ = P.parse_prediction("<|object_ref_start|>person<|object_ref_end|><|box_start|><500><500><100><900><1><2><3><4><|box_end|>")
    assert ext == {"person": [[500, 500, 100, 900, 1, 2, 3, 4]]}  # 不合法之后整段吞掉（:253–267），不是框


def test_parse_out_of_grid_not_clipped():
    b, q, st = boxes_for("<|object_ref_start|>person<|object_ref_end|><|box_start|><0><0><1500><999><|box_end|>")
    assert q == [[0, 0, 1500, 999]] and st["n_out_of_grid"] == 1
    assert b[0][2] == pytest.approx(1500 / 999.0 * W) and b[0][2] > W  # 官方不截到图内


def test_parse_incomplete_and_stop_string():
    # 截断（只有一个组、没有 <|box_end|>）：走 incomplete 那条路，末尾不完整的坐标（2 个数 = 点）丢掉
    b, q, st = boxes_for("<|object_ref_start|>person<|object_ref_end|><|box_start|><0><35><980><987>, <646><0><99")
    assert q == [[0, 35, 980, 987]] and st["incomplete"] and st["n_not_box"] == 1
    # 两个组都没收尾：不走 incomplete（官方只认恰好一个 object_ref），什么都没有
    b, _, st = boxes_for("<|object_ref_start|>person<|object_ref_end|><|box_start|><1><2><3><4>, <|object_ref_start|>person<|object_ref_end|>"
                         "<|box_start|><5><6><7><8>")
    assert b == [] and st["n_object_ref"] == 2 and st["n_groups"] == 0
    # 停止串（vLLM 的 <0><0><0><0>）：输出截在停止串之前、不含它；之后是截断的组，末尾的逗号去掉
    raw = "<|object_ref_start|>person<|object_ref_end|><|box_start|><10><10><20><20>, <0><0><0><0>, <0><0><0><0>"
    text, hit = P.truncate_at_stop(raw)
    assert hit == "<0><0><0><0>" and text.endswith("<20>, ")
    assert boxes_for(raw)[1] == [[10, 10, 20, 20]]


def test_select_category_matching_and_order():
    # 类名归一化（lower、"_" → " "，convert_coco_lvis_to_standard_format.py:13/:96）；别的类名的框丢掉并记数
    t = ("<|object_ref_start|>Traffic_Light<|object_ref_end|><|box_start|><1><1><9><9><|box_end|>, <|object_ref_start|>car<|object_ref_end|>"
         "<|box_start|><2><2><8><8><|box_end|>")
    b, q, st = boxes_for(t, "traffic light")
    assert q == [[1, 1, 9, 9]] and st["n_other_category"] == 1
    # 同名的组按出现顺序接起来，但按 dict 的键序取（官方先 result[类名].extend，再按 extracted.items() 转换）：
    # "Person" 的两组在 "person" 之前
    t = ("<|object_ref_start|>Person<|object_ref_end|><|box_start|><1><1><9><9><|box_end|>, <|object_ref_start|>person<|object_ref_end|>"
         "<|box_start|><2><2><8><8><|box_end|>, <|object_ref_start|>Person<|object_ref_end|><|box_start|><3><3><7><7><|box_end|>")
    assert boxes_for(t)[1] == [[1, 1, 9, 9], [3, 3, 7, 7], [2, 2, 8, 8]]
    # 最多 100 个（XM-3a），保持输出顺序；不去重
    many = [[i, i, i + 50, i + 50] for i in range(105)] + [[0, 0, 50, 50]]
    b, q, st = boxes_for(P.format_boxes("person", many))
    assert q == [list(x) for x in many[:100]] and st["n_capped"] == 6 and st["n_boxes_parsed"] == 106


def test_prompt_and_messages():
    assert P.detect_prompt(["person"]) == "Detect person. Output the bounding box coordinates in [x0, y0, x1, y1] format."
    assert P.detect_prompt(["potted plant", "tv"]) == "Detect potted plant, tv. Output the bounding box coordinates in [x0, y0, x1, y1] format."
    with pytest.raises(ValueError):
        P.detect_prompt([""])
    m = P.messages(["car"])
    assert m[0] == {"role": "system", "content": "You are a helpful assistant"} and m[1]["content"][0]["type"] == "image"
    # 每次只问一个类别（任务书）：身份与 README 写明这不是官方的每图多类别协议，不许把读数说成论文数字的复现
    ident = P.identity("per_category")
    assert ident["query_mode"] == "per_category" and "不是官方协议的复现" in ident["deviations"][0] and ident["category_order"] is None
    j = P.identity("joint")
    assert j["query_mode"] == "joint" and "category_id 升序" in j["category_order"] and "67" in j["deviations"][0]
    assert all(d in j["deviations"] for d in P.COMMON_DEVIATIONS) and "sdpa" in " ".join(P.COMMON_DEVIATIONS)
    with pytest.raises(ValueError):
        P.identity("per_image")


# ---------------------------------------------------------------------------
# 坐标换算与预处理
# ---------------------------------------------------------------------------


def test_coordinate_round_trip():
    import random

    rng = random.Random(0)
    n_off = 0
    for _ in range(500):
        w, h = rng.randint(20, 2000), rng.randint(20, 2000)
        xs, ys = sorted(rng.sample(range(1000), 2)), sorted(rng.sample(range(1000), 2))
        bins = [xs[0], ys[0], xs[1], ys[1]]
        px = P.bins_to_xyxy(bins, w, h)
        back = P.xyxy_to_bins(px, w, h)  # 官方 convert_boxes_to_normalized_bins：int 截断，浮点误差下可能少 1
        assert all(a - b in (0, 1) for a, b in zip(bins, back)), (bins, back)
        n_off += sum(a != b for a, b in zip(bins, back))
        assert [round(v / s * P.BIN_DIVISOR) for v, s in zip(px, (w, h, w, h))] == bins  # 换算本身可逆
        # 像素 → 网格 → 像素：误差不超过一格（原图尺寸 / 999）
        box = [rng.uniform(0, w / 2), rng.uniform(0, h / 2), rng.uniform(w / 2, w), rng.uniform(h / 2, h)]
        rt = P.bins_to_xyxy(P.xyxy_to_bins(box, w, h), w, h)
        assert all(abs(a - b) <= s / P.BIN_DIVISOR + 1e-9 for a, b, s in zip(box, rt, (w, h, w, h)))
    assert n_off > 0  # 截断确实会少 1（所以假模型的框比真值小不到一格）


def test_smart_resize_and_grid():
    # 2026-11-04 用 rev 0e5693d2 的 preprocessor_config.json 与 transformers 5.17.0 的 Qwen2VLImageProcessorPil 实测的 image_grid_thw
    assert P.expected_grid(640, 480) == (1, 34, 46) and P.expected_grid(500, 375) == (1, 26, 36)
    assert P.expected_grid(3000, 2000) == (1, 82, 122) and P.expected_grid(20, 20) == (1, 8, 8)
    assert P.n_visual_tokens((1, 34, 46)) == 391
    assert (P.MIN_PIXELS, P.MAX_PIXELS) == (12544, 2007040)  # = preprocessor_config.json
    h, w = P.smart_resize(2000, 3000)
    assert h * w <= P.MAX_PIXELS and h % 28 == 0 and w % 28 == 0
    assert P.max_new_tokens_for(391 + 40) == 4096 - 431 and P.max_new_tokens_for(10) == 4086 and P.max_new_tokens_for(5000) == 1


def test_citations_well_formed_and_constants():
    assert len(P.CITATIONS) >= 50 and len({(f, n) for f, n, _, _ in P.CITATIONS}) == len(P.CITATIONS)
    snippets = {(f, n): s for f, n, s, _ in P.CITATIONS}
    assert P.SYSTEM_PROMPT in snippets[("evaluation/inference_text_prompt.py", 12)]
    assert P.QUESTION_TEMPLATE in snippets[("evaluation/inference_text_prompt.py", 58)]
    assert P.STOP_STRINGS[0] in snippets[("evaluation/inference_text_prompt.py", 539)]
    assert str(P.REPETITION_PENALTY) in snippets[("evaluation/inference_text_prompt.py", 546)]
    assert str(P.MAX_NEW_TOKENS) in snippets[("evaluation/scrpts/eval_coco.sh", 16)]
    assert "16 * 28 * 28" in snippets[("evaluation/scrpts/eval_coco.sh", 17)] and "2560 * 28 * 28" in snippets[("evaluation/scrpts/eval_coco.sh", 18)]
    assert P.GROUP_RE.pattern in snippets[("evaluation/inference_text_prompt.py", 161)]
    assert P.INCOMPLETE_RE.pattern in snippets[("evaluation/inference_text_prompt.py", 174)]
    assert P.COORD_RE.pattern in snippets[("evaluation/inference_text_prompt.py", 209)]
    assert len(P.UPSTREAM_COMMIT) == 40 and len(P.WEIGHTS_REVISION) == 40


@pytest.mark.skipif(not os.environ.get("REXOMNI_UPSTREAM_DIR"), reason="没有给官方仓库的签出（REXOMNI_UPSTREAM_DIR，commit 6508981c）")
def test_citations_verbatim_in_upstream():
    root = Path(os.environ["REXOMNI_UPSTREAM_DIR"])
    for f, n, snippet, _ in P.CITATIONS:
        lines = (root / f).read_text(encoding="utf-8").splitlines()
        assert snippet in lines[n - 1], f"{f}:{n} 不是 {snippet!r}：{lines[n - 1]!r}"


# ---------------------------------------------------------------------------
# 分词器
# ---------------------------------------------------------------------------


def _mini_tokenizer_dir(root: Path) -> tuple[Path, dict[int, str], dict]:
    """合成的小号"慢分词器文件"：256 个 ByteLevel 字符 + 3 条 merge；新增 token 里 <0> 顶掉 id 258（merge 产出的 "xy"）。"""
    from tokenizers import pre_tokenizers

    alpha = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alpha)}
    vocab.update({"de": 256, "te": 257, "xy": 258})
    merges = ["#version: 0.2", "d e", "t e", "x y"]
    added = {258: "<0>", 259: "<1>", 260: "<|im_end|>", 261: "<|endoftext|>", 262: "<|image_pad|>"}
    root.mkdir(parents=True, exist_ok=True)
    (root / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (root / "merges.txt").write_text("\n".join(merges) + "\n", encoding="utf-8")
    tc = {"added_tokens_decoder": {str(i): {"content": c, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                                            "special": True} for i, c in added.items()},
          "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "model_max_length": 4096}
    (root / "tokenizer_config.json").write_text(json.dumps(tc), encoding="utf-8")
    return root, added, {"eos_token_id": 260, "pad_token_id": 261, "image_token_id": 262, "vocab_size": 300}


def test_tokenizer_rebuild_reuses_overridden_ids(tmp_path):
    d, added, cfg = _mini_tokenizer_dir(tmp_path / "tok")
    tok = RM.build_tokenizer(d)
    info = RM.check_tokenizer(tok, added_tokens=added, config=cfg)
    assert info["rebuild"] == {"n_added": 5, "n_vocab_overridden": 1, "n_merges": 3, "n_merges_kept": 2}
    assert tok.convert_tokens_to_ids("<0>") == 258 and tok.convert_tokens_to_ids("<|im_end|>") == 260 and len(tok) == 263
    assert tok.decode([258, 259], skip_special_tokens=False) == "<0><1>"
    assert tok("de te", add_special_tokens=False).input_ids == [256, vocab_id(tok, "Ġ"), 257]
    assert 258 not in tok("xy", add_special_tokens=False).input_ids  # "xy" 的 merge 去掉了（它的 id 现在是 <0>）
    assert tok.padding_side == "left" and tok.pad_token == "<|endoftext|>"


def vocab_id(tok, piece: str) -> int:
    return tok.convert_tokens_to_ids(piece)


def test_check_tokenizer_rejects_shifted_ids(tmp_path):
    """transformers 5 直接读慢分词器文件时的错位（新增 token 接在词表后面）：check_tokenizer 必须拒绝。"""
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    d, added, cfg = _mini_tokenizer_dir(tmp_path / "tok")
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tk = Tokenizer(BPE(vocab=vocab, merges=[("d", "e"), ("t", "e"), ("x", "y")]))
    tk.add_special_tokens([AddedToken(c, special=True) for _, c in sorted(added.items())])  # 接在 258 之后：<0> = 259
    bad = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    with pytest.raises(RM.TokenizerError, match="id 与官方不同"):
        RM.check_tokenizer(bad, added_tokens=added, config=cfg)
    good = RM.build_tokenizer(d)
    with pytest.raises(RM.TokenizerError, match="image_token_id"):
        RM.check_tokenizer(good, added_tokens=added, config={**cfg, "image_token_id": 999})
    with pytest.raises(RM.TokenizerError, match="vocab_size"):
        RM.check_tokenizer(good, added_tokens=added, config={**cfg, "vocab_size": 100})


@pytest.mark.skipif(not os.environ.get("REXOMNI_MODEL_DIR"), reason="没有给 Rex-Omni 的模型小文件（REXOMNI_MODEL_DIR，rev 0e5693d2）")
def test_real_tokenizer_files():
    d = Path(os.environ["REXOMNI_MODEL_DIR"])
    tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    tok = RM.build_tokenizer(d)
    info = RM.check_tokenizer(tok, added_tokens={int(k): v["content"] for k, v in tc["added_tokens_decoder"].items()}, config=cfg)
    assert info["n_added_checked"] == 1022 and info["coord_token_ids"] == [150643, 151642] and info["n_tokens"] == 151665
    assert info["rebuild"]["n_vocab_overridden"] == 1000
    chat = json.loads((d / "chat_template.json").read_text(encoding="utf-8"))["chat_template"]
    assert chat == RM.TINY_CHAT_TEMPLATE


@pytest.mark.skipif(not os.environ.get("REXOMNI_MODEL_DIR"), reason="没有给 Rex-Omni 的模型小文件（REXOMNI_MODEL_DIR，rev 0e5693d2）")
def test_load_real_path_with_tiny_weights(monkeypatch):
    """真模型那条路（官方的小文件：config、分词器、预处理、聊天模板），只把 from_pretrained 换成同 config 缩小的随机模型。"""
    from PIL import Image
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    d = Path(os.environ["REXOMNI_MODEL_DIR"])
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))

    def fake_from_pretrained(path, dtype=None, attn_implementation=None, output_loading_info=False):
        assert Path(path) == d and attn_implementation == P.ATTN_IMPLEMENTATION and output_loading_info
        c = {**cfg, **RM.TINY_TEXT, "vision_config": {**cfg["vision_config"], **RM.TINY_VISION}}
        return Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**c)).to(dtype), {"missing_keys": [], "unexpected_keys": []}

    monkeypatch.setattr(Qwen2_5_VLForConditionalGeneration, "from_pretrained", staticmethod(fake_from_pretrained))
    eng = RM.load_real(d, device="cpu", dtype="float32")
    eng.max_new_tokens_cap = 6
    g = eng.generate([(Image.new("RGB", (640, 480), (90, 90, 90)), ["person"]), (Image.new("RGB", (500, 375)), ["traffic light", "car"])])
    assert [x["image_grid_thw"] for x in g] == [[1, 34, 46], [1, 26, 36]] and g[0]["n_visual_tokens"] == 391
    assert eng.info["tokenizer"]["coord_token_ids"] == [150643, 151642] and eng.info["tokenizer"]["eos_token_id"] == 151645
    assert eng.tokenizer.convert_tokens_to_ids("<|image_pad|>") == cfg["image_token_id"] == 151655
    text = eng.processor.apply_chat_template(P.messages(["person"]), tokenize=False, add_generation_prompt=True)
    assert text.startswith("<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>")


def _legacy_checkpoint(d: Path, *, drop: str | None = None):
    """tiny 的 Qwen2.5-VL 存成 transformers 4.51 的键名（model.layers.*、visual.*、lm_head；Rex-Omni 的 model.safetensors.index.json 就是
    这样），可选去掉一个键。"""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen2_5_VLForConditionalGeneration

    cfg = RM.tiny_config(RM.tiny_tokenizer())
    torch.manual_seed(0)
    m = Qwen2_5_VLForConditionalGeneration(cfg)
    sd = {}
    for k, v in m.state_dict().items():
        nk = "model." + k[len("model.language_model."):] if k.startswith("model.language_model.") else (
            "visual." + k[len("model.visual."):] if k.startswith("model.visual.") else k)
        if nk != drop:
            sd[nk] = v.contiguous().clone()
    d.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(d / "model.safetensors"), metadata={"format": "pt"})
    cfg.save_pretrained(str(d))
    return m


def test_load_model_weights_legacy_keys_and_guard(tmp_path):
    import torch

    ref = _legacy_checkpoint(tmp_path / "ok")
    m, info = RM.load_model_weights(tmp_path / "ok", dtype="float32")
    assert info == {"missing_keys": 0, "unexpected_keys": 0, "mismatched_keys": 0}
    assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), m.state_dict().values()))
    _legacy_checkpoint(tmp_path / "bad", drop="model.norm.weight")
    with pytest.raises(RM.RexOmniError, match="没有完整加载"):
        RM.load_model_weights(tmp_path / "bad", dtype="float32")


def test_chat_template_pinned():
    assert hashlib.sha256(RM.TINY_CHAT_TEMPLATE.encode("utf-8")).hexdigest() == RM.CHAT_TEMPLATE_SHA256


# ---------------------------------------------------------------------------
# 权重清单
# ---------------------------------------------------------------------------


def test_weights_manifest():
    doc = RM.load_weights_manifest()
    assert doc["repo"] == P.WEIGHTS_REPO and doc["revision"] == P.WEIGHTS_REVISION
    paths = {f["path"] for f in doc["files"]}
    assert paths == {"LICENSE", "added_tokens.json", "chat_template.json", "config.json", "generation_config.json", "merges.txt",
                     "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                     "preprocessor_config.json", "tokenizer_config.json", "vocab.json"}
    assert RM.weights_total_bytes(doc) == doc["total_bytes"] == 8_136_991_717  # 8.14 GB（xm_refs_notes）


def test_weights_cmd_offline_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    p = run(["weights", "--verify"], tmp_path, ok=False)
    doc = json.loads(p.stdout)
    assert p.returncode == 1 and doc["ok"] is False and doc["total_gb"] == 8.137


def test_verify_weights(tmp_path):
    files = {"config.json": b'{"a": 1}', "model-00001-of-00002.safetensors": b"\x00" * 10}
    doc = {"repo": P.WEIGHTS_REPO, "revision": P.WEIGHTS_REVISION,
           "files": [{"path": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for k, v in files.items()]}
    d = tmp_path / "w"
    d.mkdir()
    for k, v in files.items():
        (d / k).write_bytes(v)
    (d / ".cache").mkdir()
    (d / ".cache" / "x").write_text("隐藏文件不算")
    assert RM.verify_weights(d, doc)["n_files"] == 2
    (d / "config.json").write_bytes(b'{"a": 2}')
    with pytest.raises(RM.WeightsError, match="sha256"):
        RM.verify_weights(d, doc)
    assert RM.verify_weights(d, doc, full_hash=False)["sha256_verified"] is False
    (d / "config.json").write_bytes(b'{"a": 22}')
    with pytest.raises(RM.WeightsError, match="字节"):
        RM.verify_weights(d, doc, full_hash=False)
    (d / "config.json").write_bytes(files["config.json"])
    (d / "README.md").write_text("多一个")
    with pytest.raises(RM.WeightsError, match="不同"):
        RM.verify_weights(d, doc)


# ---------------------------------------------------------------------------
# 入口断言
# ---------------------------------------------------------------------------


def test_check_devices_on_meta():
    import torch

    RM.check_devices(torch.device("cpu"), {"input_ids": torch.zeros(2), "n": 3})
    with pytest.raises(RM.RexOmniInputError, match="meta"):
        RM.check_devices(torch.device("meta"), {"input_ids": torch.zeros(2), "pixel_values": torch.zeros(1, device="meta")})


@pytest.fixture(scope="module")
def tiny_engine():
    return RM.build_tiny()


def test_engine_entry_assertions(tiny_engine, monkeypatch):
    import torch
    from PIL import Image
    from transformers.feature_extraction_utils import BatchFeature

    img = Image.new("RGB", (128, 96), (200, 30, 30))
    g = tiny_engine.generate([(img, ["red disk"])])[0]
    assert g["image_grid_thw"] == list(P.expected_grid(128, 96)) and g["n_generated_tokens"] <= RM.TINY_MAX_NEW_TOKENS
    assert g["prompt"] == P.detect_prompt(["red disk"]) and g["stop_reason"] in ("eos", "stop_string", "max_new_tokens")
    # 网格与 smart_resize 不符
    monkeypatch.setattr(P, "expected_grid", lambda w, h: (1, 1, 1))
    with pytest.raises(RM.RexOmniInputError, match="image_grid_thw"):
        tiny_engine.generate([(img, ["red disk"])])
    monkeypatch.undo()
    # 输入没搬到模型的 device（模型在 meta、输入留在 cpu）：进模型之前就拒绝
    monkeypatch.setattr(BatchFeature, "to", lambda self, *a, **k: self)
    monkeypatch.setattr(RM, "model_device", lambda m: torch.device("meta"))
    with pytest.raises(RM.RexOmniInputError, match="device"):
        tiny_engine.generate([(img, ["red disk"])])


def test_engine_bookkeeping_after_generate(tiny_engine, monkeypatch):
    """生成之后的记账：eos 收尾、停止串（vLLM 语义：截在停止串之前）、到上限没停；批内按 eos 补齐（wrapper.py:838 pad = eos）。"""
    import torch
    from PIL import Image
    from transformers import StopStringCriteria

    tok = tiny_engine.tokenizer
    eos = tok.convert_tokens_to_ids(P.EOS_TEXT)
    seqs = [tok(P.format_boxes("red disk", [[1, 2, 300, 400]]).replace(P.EOS_TEXT, ""), add_special_tokens=False).input_ids + [eos],
            tok("<|object_ref_start|>red disk<|object_ref_end|><|box_start|><10><10><20><20>, <0><0><0><0>", add_special_tokens=False).input_ids,
            tok("<|object_ref_start|>red disk<|object_ref_end|><|box_start|><1><1>", add_special_tokens=False).input_ids]
    width = max(len(s) for s in seqs)

    def fake_generate(**kw):
        ids = kw["input_ids"]
        assert kw["do_sample"] is False and kw["repetition_penalty"] == P.REPETITION_PENALTY
        (crit,) = kw["stopping_criteria"]
        assert type(crit).__name__ == "StopStringCriteria" and crit is tiny_engine.stopping_criteria()[0]
        assert kw["pad_token_id"] == eos == kw["eos_token_id"] and kw["max_new_tokens"] == RM.TINY_MAX_NEW_TOKENS
        gen = torch.tensor([s + [eos] * (width - len(s)) for s in seqs])
        return torch.cat([ids, gen], dim=1)

    monkeypatch.setattr(tiny_engine.model, "generate", fake_generate)
    monkeypatch.setattr(tiny_engine, "max_new_tokens_cap", RM.TINY_MAX_NEW_TOKENS)
    img = Image.new("RGB", (128, 96))
    out = tiny_engine.generate([(img, ["red disk"])] * 3)
    a, b, c = out
    assert a["stop_reason"] == "eos" and a["n_generated_tokens"] == len(seqs[0]) and boxes_for(a["raw_text"], "red disk", 128, 96)[1] == [[1, 2, 300, 400]]
    assert b["stop_reason"] == "stop_string" and b["n_generated_tokens"] == len(seqs[1]) and b["text"].endswith("<20>, ")
    assert boxes_for(b["raw_text"], "red disk", 128, 96)[1] == [[10, 10, 20, 20]]
    assert c["stop_reason"] == "eos" and c["n_generated_tokens"] == len(seqs[2]) + 1  # 批内补齐的 eos 就是它的收尾
    assert a["n_prompt_tokens"] == b["n_prompt_tokens"] and a["batch_size"] == 3
    # 到上限没停（没有 eos）：max_new_tokens
    seqs[:] = [tok("<|box_start|>" + "<5>" * 30, add_special_tokens=False).input_ids[: RM.TINY_MAX_NEW_TOKENS]]
    width = len(seqs[0])
    (d,) = tiny_engine.generate([(img, ["red disk"])])
    assert d["stop_reason"] == "max_new_tokens" and d["n_generated_tokens"] == RM.TINY_MAX_NEW_TOKENS
    # transformers 的 stop_strings 认得出由特殊 token（坐标 token）拼成的停止串
    crit = StopStringCriteria(tokenizer=tok, stop_strings=list(P.STOP_STRINGS))
    for s, want in (("<|box_start|><10><0><0><0><0>", True), ("<|box_start|><10><0><0><0>", False), ("<0><0><0><10>", False)):
        assert bool(crit(torch.tensor([tok(s, add_special_tokens=False).input_ids]), None)[0]) is want, s


def test_engine_real_generate_halts_at_stop_string(tiny_engine, monkeypatch, tmp_path):
    """真的 transformers generate：输出头换成总选 <0> 的，第 4 个 <0> 时停（停止串），文本截在停止串之前（为空）。"""
    import torch
    from PIL import Image

    zero = tiny_engine.tokenizer.convert_tokens_to_ids("<0>")
    vocab = tiny_engine.model.config.get_text_config().vocab_size

    class Head(torch.nn.Module):
        def forward(self, h):
            out = torch.zeros(*h.shape[:-1], vocab, dtype=h.dtype)
            out[..., zero] = 10.0
            return out

    monkeypatch.setattr(tiny_engine.model, "lm_head", Head())
    (g,) = tiny_engine.generate([(Image.new("RGB", (128, 96)), ["red disk"])])
    assert g["stop_reason"] == "stop_string" and g["n_generated_tokens"] == 4 and g["raw_text"] == "<0><0><0><0>" and g["text"] == ""
    row = RR.pair_record(_pairs(tmp_path / "d")[0], g)
    assert row["failure"] == RR.FAIL_EMPTY and row["stop_reason"] == "stop_string" and not row["truncated"]


def test_tiny_engine_info(tiny_engine):
    info = tiny_engine.info
    assert info["kind"] == "tiny" and info["tokenizer"]["n_added_checked"] == 1000
    assert info["image_processor"]["min_pixels"] == P.MIN_PIXELS and info["image_processor"]["max_pixels"] == P.MAX_PIXELS
    assert info["image_processor"]["class"].endswith("Qwen2VLImageProcessorPil") and info["chat_template_sha256"] == RM.CHAT_TEMPLATE_SHA256
    text = tiny_engine.processor.apply_chat_template(P.messages(["car"]), tokenize=False, add_generation_prompt=True)
    assert text == ("<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    "Detect car. Output the bounding box coordinates in [x0, y0, x1, y1] format.<|im_end|>\n<|im_start|>assistant\n")


# ---------------------------------------------------------------------------
# 划分与 gate
# ---------------------------------------------------------------------------


def _gate_json(path: Path, **edit) -> Path:
    doc = {"verdict": XD.XM_GATE_PASS, "rehearsal": False, "checks": {"n_is_500": True, "not_tiny": True},
           "candidate": {"manifest_sha256": "a" * 64}}
    doc.update(edit)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def test_split_and_gate_refusals(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    pj = F.write_fixture(tmp_path / "d")
    good = _gate_json(tmp_path / "g_ok.json")
    tiny = ["run", "--tiny", "--pairs-json", str(pj)]

    def refused(args, msg, out_name="0009_rexomni_refused"):
        out = tmp_path / out_name
        with pytest.raises(SystemExit) as e:
            CLI.parse_args([*map(str, args), "--out", str(out)])
        assert e.value.code == 2 and msg in capsys.readouterr().err and not out.exists(), (args, msg)

    refused([*tiny, "--split", "xm_decision"], "密封")
    refused([*tiny, "--split", "xm_decision", "--xm-gate", good], "密封")
    refused([*tiny, "--split", "reasonseg_val"], "只做检测")
    refused([*tiny, "--split", "xm_final"], "--xm-gate")
    refused([*tiny, "--split", "lvis_sub"], "--xm-gate")
    refused([*tiny, "--split", "xm_final", "--xm-gate", tmp_path / "nope.json"], "读不了")
    for name, edit, msg in [("r", {"rehearsal": True}, "彩排"), ("v", {"verdict": "判定通过"}, "结论"),
                            ("c", {"checks": {"a": True, "b": False}}, "判定点"), ("e", {"checks": {}}, "判定点"),
                            ("k", {"candidate": {}}, "候选")]:
        refused([*tiny, "--split", "xm_final", "--xm-gate", _gate_json(tmp_path / f"g_{name}.json", **edit)], msg)
    refused([*tiny, "--split", "xm_tune", "--xm-gate", good], "只配")
    refused(["run", "--split", "xm_tune", "--pairs-json", pj, "--data-root", tmp_path], "只配 --tiny")
    refused(["run", "--split", "xm_tune"], "--data-root")
    refused([*tiny, "--split", "xm_tune"], "_rexomni", out_name="0009_tiny")  # 运行目录名必须含 _rexomni
    refused([*tiny, "--split", "xm_tune"], "运行编号", out_name="rexomni_tiny")
    refused([*tiny, "--split", "xm_tune", "--batch-size", "0"], "正整数")
    # 合格：xm_final 的 gate 走基座那条路（不比 manifest）
    a = CLI.parse_args([*tiny, "--split", "xm_final", "--xm-gate", str(good), "--out", str(tmp_path / "0010_rexomni_ok")])
    g = CLI.check_gates(a)
    assert g["identity"]["checkpoint_compared"] is False and g["identity"]["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest()
    a = CLI.parse_args(["run", "--split", "xm_tune", "--data-root", str(tmp_path), "--out", str(tmp_path / "0011_rexomni_tune")])
    assert CLI.check_gates(a) is None and a.device == "cuda" and a.dtype == "bfloat16" and a.batch_size == 1
    with pytest.raises(RR.RexOmniRunError, match="密封"):
        RR.check_split("xm_decision")


def test_guard_subprocess_exit_code(tmp_path):
    pj = F.write_fixture(tmp_path / "d")
    out = tmp_path / "0013_rexomni_sealed"
    p = run(["run", "--tiny", "--pairs-json", pj, "--split", "xm_decision", "--out", out], tmp_path, ok=False)
    assert p.returncode == 2 and "密封" in p.stderr and not out.exists()


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fast_meta():
    """meta.json 的 git 信息在这个 worktree 里可能很慢：进程内的测试换成常量（同 tests/test_xm_runner.py）。"""
    import src.runlog as RL

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RL, "git_info", lambda root: {"commit": "test", "dirty": None, "diff_shortstat": None, "n_untracked": None})
        yield


CROWD_BOX = [89.0, 24.0, 111.0, 50.0]  # tests/_xm_run_fakes 图 B 的 crowd 区域的外接框


def _pairs(root: Path):
    pairs = [XD.pair_from_dict(d) for d in json.loads(F.write_fixture(root).read_text(encoding="utf-8"))]
    a_disk, a_sq, b_disk = pairs
    junk = dataclasses.replace(a_sq, id=a_sq.id + "/junk", category_id=4, category="blue square junk")
    silent = dataclasses.replace(a_disk, id=a_disk.id + "/silent", category_id=5, category="red disk silent")
    return [a_disk, a_sq, b_disk, junk, silent]


def test_call_units_order_and_contiguity(tmp_path):
    a_disk, a_sq, b_disk = _pairs(tmp_path / "d")[:3]
    units = RR.call_units([a_sq, a_disk, b_disk], P.QUERY_JOINT)  # 输入里方块（2）在圆盘（1）之前
    assert [[p.id for p in u] for u, _ in units] == [[a_sq.id, a_disk.id], [b_disk.id]]  # 行序 = 输入顺序
    assert [c for _, c in units] == [["red disk", "blue square"], ["red disk"]]  # 提示里按 category_id 升序
    assert [c for _, c in RR.call_units([a_sq, a_disk, b_disk], P.QUERY_PER_CATEGORY)] == [["blue square"], ["red disk"], ["red disk"]]
    with pytest.raises(RR.RexOmniRunError, match="不相邻"):
        RR.call_units([a_disk, b_disk, a_sq], P.QUERY_JOINT)
    with pytest.raises(RR.RexOmniRunError, match="同一个类别"):
        RR.call_units([a_disk, dataclasses.replace(a_disk, id="x")], P.QUERY_JOINT)
    assert RR.batch_bounds(units, 1) == [(0, 1, 2), (1, 2, 3)] and RR.batch_bounds(units, 2) == [(0, 2, 3)]


def test_select_joint_dispatch_and_unasked():
    text = P.format_groups([("red disk", [[1, 1, 9, 9], [2, 2, 8, 8]]), ("Blue_Square", [[3, 3, 7, 7]]), ("green triangle", [[0, 0, 5, 5]]),
                            ("red disk", [[4, 4, 6, 6]]), ("dog", [[1, 2, 3, 4], [5, 6, 7, 8]])])
    assert text.count(", <|object_ref_start|>") == 4 and text.endswith("<|box_end|><|im_end|>")
    ext, info = P.parse_prediction(text)
    assert info["n_groups"] == 5
    sel, n_unasked = P.select_joint(ext, ["red disk", "blue square"], 100, 100, max_dets=100)
    assert sel["red disk"][1] == [[1, 1, 9, 9], [2, 2, 8, 8], [4, 4, 6, 6]] and sel["blue square"][1] == [[3, 3, 7, 7]]
    assert n_unasked == 3 and sel["red disk"][2]["n_other_category"] == 0
    # 没回答的所问类别：空；只问一个类别时与 select_boxes 相同
    sel, n_unasked = P.select_joint(ext, ["red disk", "cat"], 100, 100, max_dets=100)
    assert sel["cat"][1] == [] and n_unasked == 4
    assert P.select_joint(ext, ["red disk"], 100, 100, max_dets=100)[0]["red disk"][:2] == P.select_boxes(ext, "red disk", 100, 100,
                                                                                                         max_dets=100)[:2]
    with pytest.raises(ValueError, match="重名"):
        P.select_joint(ext, ["red disk", "Red_Disk"], 100, 100, max_dets=100)
    assert P.detect_prompt(["red disk", "blue square"]) == ("Detect red disk, blue square. Output the bounding box coordinates in "
                                                            "[x0, y0, x1, y1] format.")


def test_scripted_joint_end_to_end(tmp_path, fast_meta):
    root = tmp_path / "data"
    pairs = _pairs(root)[:3]
    eng = RF.ScriptedRexOmniEngine(F.gt_answers(pairs, root), other_phrase="green triangle")
    out = tmp_path / "0025_rexomni_joint"
    m = RR.run_split(eng, pairs, out, split="xm_tune", setting=RR.SETTING_JOINT, data_root=root, batch_size=1, n_resamples=100)
    leaf = RR.leaf_dir(out, "xm_tune", RR.SETTING_JOINT)
    assert leaf == out / "rex-omni" / "xm" / "rexomni_det_joint" / "xm_tune"
    rows = jsonl(leaf / "predictions.jsonl")
    assert [r["id"] for r in rows] == [p.id for p in pairs]
    assert [c["categories"] for c in eng.calls] == [["red disk", "blue square"], ["red disk"]]  # 每图一次
    for p, r in zip(pairs, rows):
        assert r["det"]["tp"][0] == len(p.instances) and r["det"]["fp"][0] == 0 and "n_other_category" not in r["parse"]
        assert r["parse"]["n_unasked_category"] == 1 and r["query_mode"] == "joint"
    assert rows[0]["tokens"] == rows[1]["tokens"] and rows[0]["raw"] == rows[1]["raw"]  # 同一次调用
    s = m["summary"]
    assert s["n_calls"] == 2 and s["n_unasked_category"] == 2 and s["pairs_per_call"]["max"] == 2 and m["det"]["f1_rexomni"]["value"] == 1.0
    assert m["smoke_line"]["pass"] is True and m["smoke_line"]["value"] == m["det"]["f1_rexomni_exact"]["value"]
    tm = jsonl(leaf / "timing.jsonl")
    assert [t["call_index"] for t in tm] == [0, 0, 1] and tm[0]["wall_s"] == pytest.approx(tm[0]["call_wall_s"] / 2)
    cfg = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["query_mode"] == "joint" and cfg["protocol"]["category_order"] == P.JOINT_ORDER and cfg["data"]["n_calls"] == 2
    assert "Rex-Omni 自己的协议" in (leaf / "README.md").read_text(encoding="utf-8")
    # 续跑按批的边界（这里一批 = 一张图）：截掉最后一行，退回到第 1 张图之后，整张图重跑
    full = (leaf / "predictions.jsonl").read_bytes()
    lines = full.decode("utf-8").splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_text(lines[0], encoding="utf-8")
    eng.calls.clear()
    RR.run_split(eng, pairs, out, split="xm_tune", setting=RR.SETTING_JOINT, data_root=root, batch_size=1, resume=True, n_resamples=100)
    assert (leaf / "predictions.jsonl").read_bytes() == full and [c["categories"] for c in eng.calls] == [["red disk", "blue square"],
                                                                                                         ["red disk"]]


def test_smoke_line():
    def met(split, setting, v):
        return {"split": split, "setting": setting, "det": {"f1_rexomni_exact": {"value": v}}}

    assert RR.smoke_line(met("xm_tune", RR.SETTING_JOINT, 0.61))["pass"] is True
    assert RR.smoke_line(met("xm_tune", RR.SETTING_JOINT, 0.60))["pass"] is True
    assert RR.smoke_line(met("xm_tune", RR.SETTING_JOINT, 0.5999))["pass"] is False
    assert RR.smoke_line(met("xm_tune", RR.SETTING_JOINT, None))["pass"] is False
    assert RR.smoke_line(met("xm_tune", RR.SETTING_PER_CATEGORY, 0.1)) is None and RR.smoke_line(met("xm_final", RR.SETTING_JOINT, 0.1)) is None
    assert RR.SMOKE_THRESHOLD == 0.60 and CLI.SMOKE_FAIL_RC == 3


def test_scripted_end_to_end(tmp_path, fast_meta):
    root = tmp_path / "data"
    pairs = _pairs(root)
    eng = RF.ScriptedRexOmniEngine(F.gt_answers(pairs, root))
    out = tmp_path / "0020_rexomni_scripted"
    m = RR.run_split(eng, pairs, out, split="xm_tune", setting=RR.SETTING_PER_CATEGORY, data_root=root, batch_size=2, n_resamples=200)
    leaf = RR.leaf_dir(out, "xm_tune", RR.SETTING_PER_CATEGORY)
    assert leaf == out / "rex-omni" / "xm" / "rexomni_det" / "xm_tune"
    for f in ("predictions.jsonl", "timing.jsonl", "metrics.json", "config.yaml", "meta.json", "README.md"):
        assert (leaf / f).is_file(), f
    rows = jsonl(leaf / "predictions.jsonl")
    assert [r["id"] for r in rows] == [p.id for p in pairs] and all(r["seg"] is None for r in rows)
    by = {r["id"]: r for r in rows}
    # 真值框经网格往返（差不到一格），全部配上
    for p in pairs[:3]:
        assert by[p.id]["det"]["tp"][0] == len(p.instances) and by[p.id]["det"]["fp"][0] == 0 and by[p.id]["failure"] is None
    assert by[pairs[3].id]["failure"] == RR.FAIL_NO_GROUP and by[pairs[4].id]["failure"] == RR.FAIL_EMPTY
    assert m["summary"]["n_parse_failures"] == 2 and m["complete"] and m["det"]["n_pairs"] == 5
    # 叶目录能被 src/xm/metrics.table 读，读数与 metrics.json 相同
    tb = XMM.table(rows, "det")
    assert XMM.point(tb, "f1_rexomni") == m["det"]["f1_rexomni"]["value"]
    # 4 类（圆盘 2 对、方块 1 对、junk 与 silent 各 1 对）：P 只在有预测的 2 类上平均 = 1，R = (1 + 1 + 0 + 0) / 4
    assert m["det"]["f1_rexomni"]["value"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert [b for b in by[pairs[0].id]["instances"]][0]["bins"] == P.xyxy_to_bins(pairs[0].instances[0].box_xyxy, F.W, F.H)
    cfg = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["protocol"]["query_mode"] == "per_category" and cfg["setting"] == "rexomni_det" and cfg["system"] == "rex-omni"
    assert cfg["data"]["pair_ids_sha256"] == RR.pairs_digest(pairs) and cfg["batching"]["batch_size"] == 2
    assert "不是" in (leaf / "README.md").read_text(encoding="utf-8")
    assert [c["batch"] for c in eng.calls] == [2, 2, 2, 2, 1]
    # 重复框记 FP、crowd 上的框忽略、别的类名的框丢掉；与上一份按同一批对配对 bootstrap
    eng2 = RF.ScriptedRexOmniEngine(F.gt_answers(pairs, root), duplicate_first=True, extra_box=CROWD_BOX, other_phrase="green triangle")
    out2 = tmp_path / "0021_rexomni_scripted"
    RR.run_split(eng2, pairs, out2, split="xm_tune", setting=RR.SETTING_PER_CATEGORY, data_root=root, batch_size=1, n_resamples=200)
    rows2 = jsonl(RR.leaf_dir(out2, "xm_tune", RR.SETTING_PER_CATEGORY) / "predictions.jsonl")
    b2 = {r["id"]: r for r in rows2}
    assert b2[pairs[2].id]["det"]["ignored"][0] == 1 and b2[pairs[2].id]["det"]["fp"][0] == 1  # 图 B：crowd 上的框忽略、重复的框 FP
    assert b2[pairs[0].id]["parse"]["n_other_category"] == 1 and b2[pairs[0].id]["det"]["fp"][0] == 2  # 重复 + 图 A 上没有 crowd 的那个框
    ci = XMM.paired_bootstrap(XMM.table(rows, "det"), XMM.table(rows2, "det"), "f1_rexomni", n_resamples=200)
    assert ci.low <= ci.high


def test_image_key_and_not_exhaustive_same_as_xm_runner(tmp_path):
    from src.xm import runner as XR  # 测试里可以 import；Rex-Omni 一侧不 import 它

    pairs = _pairs(tmp_path / "d")
    lv = [dataclasses.replace(pairs[0], meta={"not_exhaustive": True}), dataclasses.replace(pairs[0], meta={"not_exhaustive": False}),
          dataclasses.replace(pairs[0], meta={"not_exhaustive_category_ids": [pairs[0].category_id]}), dataclasses.replace(pairs[0], meta={})]
    for p in [*pairs, *lv]:
        assert RR.image_key(p) == XR.image_key(p) and RR.not_exhaustive(p) == XR.not_exhaustive(p)
        assert RR.det_eval(p, [list(i.box_xyxy) for i in p.instances]) == XR.det_eval(p, {"boxes": [list(i.box_xyxy) for i in p.instances]})


def test_resume_and_identity(tmp_path, fast_meta):
    root = tmp_path / "data"
    pairs = _pairs(root)
    eng = RF.ScriptedRexOmniEngine(F.gt_answers(pairs, root))
    out = tmp_path / "0022_rexomni_resume"
    RR.run_split(eng, pairs, out, split="xm_tune", setting=RR.SETTING_PER_CATEGORY, data_root=root, batch_size=2, n_resamples=None)
    leaf = RR.leaf_dir(out, "xm_tune", RR.SETTING_PER_CATEGORY)
    full = (leaf / "predictions.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        RR.run_split(eng, pairs, out, split="xm_tune", setting=RR.SETTING_PER_CATEGORY, data_root=root, batch_size=2, n_resamples=None)
    # 截掉最后 2 行再续跑：退回到批的边界（5 → 3 行 → 退到 2 行），整批重跑，结果逐字节相同
    lines = full.decode("utf-8").splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_text("".join(lines[:3]) + lines[3][:20], encoding="utf-8")
    eng.calls.clear()
    RR.run_split(eng, pairs, out, split="xm_tune", setting=RR.SETTING_PER_CATEGORY, data_root=root, batch_size=2, resume=True, n_resamples=None)
    assert (leaf / "predictions.jsonl").read_bytes() == full and [c["batch"] for c in eng.calls] == [2, 2, 1]  # 每条一项
    assert len(jsonl(leaf / "timing.jsonl")) == 5 and len(list((leaf / "logs").glob("run_*.json"))) == 2
    # 身份不同（批大小、对）：拒绝续跑
    with pytest.raises(RR.RexOmniRunError, match="身份"):
        RR.run_split(eng, pairs, out, split="xm_tune", setting=RR.SETTING_PER_CATEGORY, data_root=root, batch_size=1, resume=True, n_resamples=None)
    with pytest.raises(RR.RexOmniRunError, match="身份"):
        RR.run_split(eng, pairs[:4], out, split="xm_tune", setting=RR.SETTING_PER_CATEGORY, data_root=root, batch_size=2, resume=True, n_resamples=None)
    with pytest.raises(RR.RexOmniRunError, match="_rexomni"):
        RR.run_split(eng, pairs, tmp_path / "0023_other", split="xm_tune", setting=RR.SETTING_JOINT, data_root=root)
    with pytest.raises(RR.RexOmniRunError, match="split"):
        RR.run_split(eng, pairs, tmp_path / "0024_rexomni", split="xm_final", setting=RR.SETTING_PER_CATEGORY, data_root=root)
    with pytest.raises(RR.RexOmniRunError, match="设置"):
        RR.run_split(eng, pairs, tmp_path / "0026_rexomni", split="xm_tune", setting="rexomni_det_per_image", data_root=root)


def test_cli_tiny_end_to_end(tmp_path):
    """tiny 的命令行，缺省两个设置都跑（先 joint 再 per_category，模型只加载一次），叶目录能被 src/xm/metrics.table 读、两个设置可以配对。"""
    pj = F.write_fixture(tmp_path / "d")
    out = tmp_path / "0030_rexomni_tiny"
    p = run(["run", "--tiny", "--split", "xm_tune", "--pairs-json", pj, "--out", out, "--batch-size", "2"], tmp_path)
    assert "Rex-Omni rexomni_det_joint xm_tune n=3 calls=2" in p.stdout and "Rex-Omni rexomni_det xm_tune n=3 calls=3" in p.stdout
    assert p.stdout.index("rexomni_det_joint") < p.stdout.index("Rex-Omni rexomni_det xm_tune")
    assert "冒烟线 f1_rexomni_exact" in p.stdout and "--tiny 只记不拦" in p.stdout  # 随机模型过不了线，但 tiny 不拦（退出码 0）
    want = [d["id"] for d in json.loads(pj.read_text(encoding="utf-8"))]
    tables = {}
    for setting, n_calls in ((RR.SETTING_JOINT, 2), (RR.SETTING_PER_CATEGORY, 3)):
        leaf = RR.leaf_dir(out, "xm_tune", setting)
        rows = jsonl(leaf / "predictions.jsonl")
        assert [r["id"] for r in rows] == want and all(r["setting"] == setting for r in rows)
        tables[setting] = tb = XMM.table(rows, "det")
        assert tb.pair_ids == tuple(want) and len(tb.images) == 2
        m = json.loads((leaf / "metrics.json").read_text(encoding="utf-8"))
        assert m["complete"] and m["det"]["n_pairs"] == 3 and m["seg"] is None and m["tiny"] is True and m["setting"] == setting
        assert all(r["tokens"]["generated"] <= RM.TINY_MAX_NEW_TOKENS and r["image_grid_thw"] == [1, 8, 10] for r in rows)
        s = m["summary"]
        assert s["n_calls"] == n_calls and s["n_timing_missing"] == 0
        assert set(s) >= {"n_parse_failures", "n_truncated", "stop_reason", "generated_tokens", "call_wall_s", "n_unasked_category"}
        cfg = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["model"]["kind"] == "tiny" and cfg["model"]["max_new_tokens_cap"] == RM.TINY_MAX_NEW_TOKENS
        assert cfg["setting"] == setting and cfg["query_mode"] == RR.SETTINGS[setting]["query_mode"] and cfg["data"]["n_calls"] == n_calls
        if setting == RR.SETTING_JOINT:
            assert m["smoke_line"]["pass"] is False and m["smoke_line"]["threshold"] == RR.SMOKE_THRESHOLD
            assert rows[0]["prompt"] == P.detect_prompt(["red disk", "blue square"]) and rows[0]["call"] == {"index": 0, "n_pairs": 2, "first": True}
            assert rows[1]["call"] == {"index": 0, "n_pairs": 2, "first": False} and rows[2]["call"]["index"] == 1
        else:
            assert m["smoke_line"] is None and rows[1]["prompt"] == P.detect_prompt(["blue square"])
    XMM.paired_bootstrap(tables[RR.SETTING_JOINT], tables[RR.SETTING_PER_CATEGORY], "f1_micro", n_resamples=50)  # 同一批对，可以配对
    # 续跑：什么都不重跑，逐字节相同；不加 --resume 就拒绝（码 2，报错落盘）
    leaf = RR.leaf_dir(out, "xm_tune", RR.SETTING_JOINT)
    before = (leaf / "predictions.jsonl").read_bytes()
    run(["run", "--tiny", "--split", "xm_tune", "--pairs-json", pj, "--out", out, "--batch-size", "2", "--resume"], tmp_path)
    assert (leaf / "predictions.jsonl").read_bytes() == before
    p = run(["run", "--tiny", "--split", "xm_tune", "--pairs-json", pj, "--out", out, "--batch-size", "2"], tmp_path, ok=False)
    assert p.returncode == 2 and list((out / "rex-omni" / "errors").glob("error_*.json"))
    p = run(["run", "--tiny", "--split", "xm_tune", "--pairs-json", pj, "--out", tmp_path / "0033_rexomni", "--setting", "rexomni_det",
             "rexomni_det"], tmp_path, ok=False)
    assert p.returncode == 2 and "重复" in p.stderr


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    from tests._xm_fakes import build_fake_root

    d = tmp_path_factory.mktemp("rexroot")
    build_fake_root(d / "root")
    sp = d / "xm_splits.json"
    sp.write_text(XD.dump_config(XD.build_splits_doc(d / "root", tiny=True)), encoding="utf-8")
    return d / "root", sp


def test_cli_tiny_xm_final_with_gate(fake_root, tmp_path):
    root, sp = fake_root
    good = _gate_json(tmp_path / "g.json")
    out = tmp_path / "0031_rexomni_final"
    run(["run", "--tiny", "--split", "xm_final", "--xm-gate", good, "--data-root", root, "--splits-file", sp, "--out", out], tmp_path)
    gp = XG.check_xm_gate(good, checkpoint=None)["pass"]
    pairs = XD.load_pairs("xm_final", root, xm_gate=gp, splits_path=sp)
    want = [q.id for q in pairs]
    for setting in RR.SETTINGS:
        leaf = RR.leaf_dir(out, "xm_final", setting)
        rows = jsonl(leaf / "predictions.jsonl")
        assert [r["id"] for r in rows] == want
        cfg = yaml.safe_load((leaf / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["xm_gate"]["sha256"] == hashlib.sha256(good.read_bytes()).hexdigest() and cfg["xm_gate"]["checkpoint_compared"] is False
        assert str(sp) in cfg["data"]["files_sha256"] and cfg["data"]["n_calls"] == len(RR.call_units(pairs, cfg["query_mode"]))
        assert json.loads((leaf / "metrics.json").read_text(encoding="utf-8"))["smoke_line"] is None  # 冒烟线只看 xm_tune
    joint = jsonl(RR.leaf_dir(out, "xm_final", RR.SETTING_JOINT) / "predictions.jsonl")
    for r in joint:  # 每张图问的就是它在本划分里的全部正类别，category_id 升序
        mine = sorted((q for q in pairs if RR.image_key(q) == r["image_key"]), key=lambda q: q.category_id)
        assert r["asked_categories"] == [q.category for q in mine]
    assert len(jsonl(out / "rex-omni" / XG.UNLOCK_LOG)) == 1
    p = run(["run", "--tiny", "--split", "xm_final", "--xm-gate", _gate_json(tmp_path / "g_r.json", rehearsal=True), "--data-root", root,
             "--splits-file", sp, "--out", tmp_path / "0032_rexomni_final"], tmp_path, ok=False)
    assert p.returncode == 2 and "彩排" in p.stderr and not (tmp_path / "0032_rexomni_final").exists()


def test_xm_code_unchanged():
    """X-M、X-2 的代码与配置与 git HEAD 逐字节相同（任务书：不改 X-M、X-2 的任何代码、配置与结果）。"""
    paths = ["src/xm", "scripts/run_xm.py", "scripts/gate_xm.py", "scripts/xm", "configs/xm_splits.json", "configs/xm_train.json",
             "configs/xm_train_exclusions.json", "src/explore", "scripts/run_x2.py", "configs/x2_run.json", "configs/x2_train.json"]
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=REPO, capture_output=True)
    if r.returncode not in (0, 1):
        pytest.skip("没有 git")
    assert r.returncode == 0, "X-M 或 X-2 的代码、配置被改了"
    assert shutil.which("git")
