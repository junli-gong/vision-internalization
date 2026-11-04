"""Rex-Omni 官方推理的约定（DEV-0083）。只 import 标准库：提示、图像预处理、生成参数、输出解析与坐标换算。

出处（逐条见 CITATIONS，tests/test_rexomni.py 在给了官方仓库的签出时逐字核对）：
- 官方仓库 github.com/IDEA-Research/Rex-Omni @ 6508981c（master，2026-02-22），IDEA License 1.0（非商业研究）：
  - evaluation/inference_text_prompt.py：论文 COCO / LVIS 读数用的推理脚本（vLLM 后端）——系统提示、提问模板、min/max pixels、
    生成参数与停止串、输出解析 parse_prediction / _process_box_text；
  - evaluation/scrpts/eval_coco.sh、eval_lvis.sh：两个数据集实际传的参数（max_new_tokens 4096、min/max pixels、提问）；
  - evaluation/utils/convert_coco_lvis_to_standard_format.py：解析结果 → 计分输入（类名的归一化、只认 4 个数的框、positive_only、
    常数分数 0.2，即按输出顺序）；
  - rex_omni/wrapper.py：transformers 后端（Qwen2_5_VLForConditionalGeneration、bf16、AutoProcessor(min/max pixels, use_fast=False)、
    左侧补齐、生成参数、解码不跳过特殊 token）；rex_omni/tasks.py 的检测模板；rex_omni/parser.py 的 框 → 网格（往返测试用）。
- 模型文件 huggingface.co/IDEA-Research/Rex-Omni @ 0e5693d2（权重清单 configs/rexomni_weights.json）：preprocessor_config.json 的
  min_pixels 12544 = 16·28·28、max_pixels 2007040 = 2560·28·28；chat_template.json；tokenizer_config.json 的 1,022 个新增 token
  （坐标 <0>…<999> 占 id 150643–151642，见 src/rexomni/model.py）。

**两个设置**（src/rexomni/runner.SETTINGS；QUERY_MODES）：
- joint（rexomni_det_joint，论文里外部参照的标题读数，Rex-Omni 自己的协议）：每张图调用一次，提示里用 ", " 列出这张图在本划分里的全部
  正类别（inference_text_prompt.py:599；官方评测数据 Rex-Omni-EvalData 的 COCO.jsonl：每行的 categories 就是该图 gt 的类别），
  按类名把框分给各个（图, 类别）对（select_joint）；
- per_category（rexomni_det，诊断，与 X-M 同协议）：每个（图, 类别）对调用一次，提示里只有这一个类别（select_boxes）。

**与官方的不同**（运行身份里逐条写明，见 deviations()）：
1. joint 的类别顺序用 category_id 升序，不依赖官方评测数据文件。官方 COCO.jsonl 的顺序不是升序：xm_tune 的 300 张图都在其中，
   239 张多类别图里只有 67 张是 category_id 升序（DEV-0083 核对）。per_category 不涉及顺序，但不是官方协议。
2. 用 transformers 后端（官方 rex_omni/wrapper.py 的 transformers 分支），不是论文读数用的 vLLM；vLLM 那一套的停止串与
   max_model_len 照搬（停止串 "<0><0><0><0>" 用 transformers 的 StopStringCriteria，生成上限取 min(4096, 4096 − 提示长度)）。
3. 注意力用 sdpa（服务器环境没有装 flash-attn；官方 wrapper 缺省 flash_attention_2）。
4. 分词器由官方的慢分词器文件重建（见 src/rexomni/model.py）：与 transformers 4.51.3 的慢分词器只在含有被坐标 token 顶掉的 1,000 个
   罕见 BPE token 的文本上不同，提示与类名里没有这些 token。
5. 不在所问类别里的类名下的框丢掉并计数（官方是记到那个类别上、再按 positive_only 过滤；joint 问的就是该图本划分里的全部正类别，
   lvis_sub 是每图至多 3 类的上限之后的那几类）。
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

UPSTREAM_URL = "https://github.com/IDEA-Research/Rex-Omni"
UPSTREAM_COMMIT = "6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e"  # master 的 HEAD，2026-02-22T02:59:06Z（PR #159）
UPSTREAM_LICENSE = "IDEA License 1.0（限非商业研究；权重另受 Qwen Research License 约束）"
PAPER = "Jiang et al., Detect Anything via Next Point Prediction（arXiv 2510.12798）"
WEIGHTS_REPO = "IDEA-Research/Rex-Omni"
WEIGHTS_REVISION = "0e5693d24657f6c0e091008dd6809bb1bd28988c"  # HF 上 main 的 sha（lastModified 2025-10-16T12:03:30Z）

# ---------------------------------------------------------------------------
# 提示
# ---------------------------------------------------------------------------

# inference_text_prompt.py:12 DEFAULT_PROMPT；:81 inference() 里 system_prompt = DEFAULT_PROMPT（不管传什么都用它）；
# wrapper.py:29 的缺省同此。注意没有句号（chat_template.json 只在没有 system 消息时才补 "You are a helpful assistant."）。
SYSTEM_PROMPT = "You are a helpful assistant"
# inference_text_prompt.py:58（--question 的缺省，"for box prompt detection"）与 eval_coco.sh:46、eval_lvis.sh:46；
# rex_omni/tasks.py:43 的 TaskType.DETECTION 同一句（{categories} 代替 [OBJ]）。
QUESTION_TEMPLATE = "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format."
CATEGORY_JOIN = ", "  # inference_text_prompt.py:599 question = args.question.replace("[OBJ]", ", ".join(categories))


def detect_prompt(categories: Sequence[str]) -> str:
    """官方的检测提问（inference_text_prompt.py:599）。joint 给该图本划分里的全部正类别（category_id 升序），per_category 只给一个。"""
    cats = [str(c) for c in categories]
    if not cats or any(not c.strip() for c in cats):
        raise ValueError(f"类别不能为空：{categories!r}")
    return QUESTION_TEMPLATE.replace("[OBJ]", CATEGORY_JOIN.join(cats))


def messages(categories: Sequence[str], image: Any = None) -> list[dict[str, Any]]:
    """processor.apply_chat_template 的输入（inference_text_prompt.py:82–96；wrapper.py:459–473 的 transformers 分支同结构）。"""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": detect_prompt(categories)}]}]


# ---------------------------------------------------------------------------
# 图像预处理（Qwen2.5-VL 的 smart_resize：边长取 28 的倍数，像素数落在 [min_pixels, max_pixels]）
# ---------------------------------------------------------------------------

MIN_PIXELS = 16 * 28 * 28  # inference_text_prompt.py:46；eval_coco.sh:17；wrapper.py:30；preprocessor_config.json 的 12544
MAX_PIXELS = 2560 * 28 * 28  # inference_text_prompt.py:47；eval_coco.sh:18；wrapper.py:31；preprocessor_config.json 的 2007040
PATCH_SIZE = 14  # preprocessor_config.json 的 patch_size（config.json vision_config.patch_size）
MERGE_SIZE = 2  # preprocessor_config.json 的 merge_size
RESIZE_FACTOR = PATCH_SIZE * MERGE_SIZE  # 28
MAX_ASPECT_RATIO = 200
IMAGE_MODE = "RGB"  # inference_text_prompt.py:592 Image.open(full_image_path).convert("RGB")；w, h = image.size（:593）是原图尺寸


def smart_resize(height: int, width: int, factor: int = RESIZE_FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """照 transformers 5.17.0 models/qwen2_vl/image_processing_pil_qwen2_vl.py 的 smart_resize（Apache-2.0）逐行写；
    qwen_vl_utils 0.0.14 的同名函数（官方 wrapper 用它算 resized_height/width）是同一组公式。运行时用它核对处理器给的
    image_grid_thw（src/rexomni/model.py 的入口断言）。返回 (h', w')。"""
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def expected_grid(width: int, height: int) -> tuple[int, int, int]:
    """一张图的 image_grid_thw（t, h/14, w/14）；视觉 token 数 = h·w / 14² / 2²。"""
    h, w = smart_resize(int(height), int(width))
    return (1, h // PATCH_SIZE, w // PATCH_SIZE)


def n_visual_tokens(grid: Sequence[int]) -> int:
    t, gh, gw = (int(v) for v in grid)
    return t * gh * gw // (MERGE_SIZE * MERGE_SIZE)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 4096  # inference_text_prompt.py:45；eval_coco.sh:16 MAX_TOKENS=4096；wrapper.py:32
MAX_MODEL_LEN = 4096  # inference_text_prompt.py:529（vLLM 的上下文上限 = 提示 + 生成）
REPETITION_PENALTY = 1.05  # inference_text_prompt.py:546；wrapper.py:36
STOP_STRINGS: tuple[str, ...] = ("<0><0><0><0>",)  # inference_text_prompt.py:539（vLLM 的 stop；输出不含停止串本身）
EOS_TEXT = "<|im_end|>"  # inference_text_prompt.py:537 的另一条停止串；也是 config.json / generation_config.json 的 eos 151645
# 官方的采样参数（inference_text_prompt.py:545–548；wrapper.py:33–35）：temperature 0 → 贪心（wrapper.py:837 do_sample =
# temperature > 0），top_p / top_k 在贪心下不起作用，这里不传。
OFFICIAL_SAMPLING: Mapping[str, Any] = {"temperature": 0.0, "top_p": 0.8, "top_k": 1, "repetition_penalty": REPETITION_PENALTY}
SKIP_SPECIAL_TOKENS = False  # inference_text_prompt.py:549；wrapper.py:37、:855（坐标 token 是特殊 token，跳过就没了）
CLEAN_UP_TOKENIZATION_SPACES = False  # wrapper.py:856
PADDING_SIDE = "left"  # wrapper.py:174（批量生成）
TORCH_DTYPE = "bfloat16"  # wrapper.py:146
ATTN_IMPLEMENTATION = "sdpa"  # 官方缺省 flash_attention_2（wrapper.py:148），服务器没有 flash-attn（deviations 第 3 条）


def max_new_tokens_for(prompt_len: int) -> int:
    """生成上限：max_new_tokens 4096，且提示 + 生成不超过 vLLM 的 max_model_len 4096（deviations 第 2 条）。"""
    return max(1, min(MAX_NEW_TOKENS, MAX_MODEL_LEN - int(prompt_len)))


def generation_identity() -> dict[str, Any]:
    return {"backend": "transformers generate（官方 rex_omni/wrapper.py 的 transformers 分支）", "do_sample": False, "num_beams": 1,
            "repetition_penalty": REPETITION_PENALTY, "max_new_tokens": MAX_NEW_TOKENS, "max_model_len": MAX_MODEL_LEN,
            "max_new_tokens_rule": "min(max_new_tokens, max_model_len − 提示长度)；一批取批内最长的提示", "stop_strings": list(STOP_STRINGS),
            "eos": EOS_TEXT, "pad_token_id": "eos（wrapper.py:838）", "skip_special_tokens": SKIP_SPECIAL_TOKENS,
            "clean_up_tokenization_spaces": CLEAN_UP_TOKENIZATION_SPACES, "padding_side": PADDING_SIDE, "official_sampling": dict(OFFICIAL_SAMPLING),
            "torch_dtype": TORCH_DTYPE, "attn_implementation": ATTN_IMPLEMENTATION}


def truncate_at_stop(text: str) -> tuple[str, str | None]:
    """vLLM 的停止串语义：在第一个停止串处截断，输出不含停止串（include_stop_str_in_output 缺省 False）。返回 (截断后的文本, 命中的停止串)。"""
    hits = [(text.find(s), s) for s in STOP_STRINGS if s in text]
    if not hits:
        return text, None
    pos, s = min(hits)
    return text[:pos], s


# ---------------------------------------------------------------------------
# 输出解析（inference_text_prompt.py:130–340 的 parse_prediction / _process_box_text，按网格整数写；换算见 bins_to_xyxy）
# ---------------------------------------------------------------------------

BIN_DIVISOR = 999.0  # inference_text_prompt.py:239 x0_norm = x0_bin / 999.0；:245 x0 = x0_norm * w（网格 0–999 → 原图像素）
BIN_MAX = 999
OBJ_START, OBJ_END, BOX_START, BOX_END = "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>"
GROUP_RE = re.compile(r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>")  # :161
INCOMPLETE_RE = re.compile(r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)$", re.DOTALL)  # :174–175
COORD_RE = re.compile(r"<(\d+)>")  # :209


def _box_text_bins(box_text: str, *, is_incomplete: bool) -> list[list[int]]:
    """_process_box_text（:189–340）的控制流，坐标停在网格整数：
    - 按 "," 切成若干段（:204），每段取全部 <数字>（:209–210）；
    - 段内从头起：还剩至少 4 个数（:229）且 x1 > x0、y1 > y0（:237）→ 一个框，前进 4；否则把剩下的数全当一个坐标串（:253–267）；
    - 剩下不到 4 个数：截断的输出（is_incomplete）取偶数个（:274–302），否则全取（:303–317）；
    - 非空的坐标串都留下（:320–325）。是不是框（恰好 4 个数）由 convert_coco_lvis_to_standard_format.py:102 判（select_boxes）。"""
    if is_incomplete:  # :195–201
        box_text = box_text.strip()
        if box_text.endswith(","):
            box_text = box_text[:-1]
    out: list[list[int]] = []
    for seq in (s.strip() for s in box_text.split(",") if s.strip()):
        m = [int(x) for x in COORD_RE.findall(seq)]
        i = 0
        while i < len(m):
            j = i
            if j + 3 < len(m):
                x0, y0, x1, y1 = m[j:j + 4]
                if x1 > x0 and y1 > y0:
                    coords, i = [x0, y0, x1, y1], i + 4
                else:
                    coords, i = m[j:], len(m)
            else:
                rem = len(m) - j
                if is_incomplete and rem >= 2:
                    n = (rem // 2) * 2  # :277
                    coords, i = m[j:j + n], j + n
                    if not (len(coords) == 2 or (len(coords) >= 4 and len(coords) % 2 == 0)):  # :292–302
                        coords = []
                else:
                    coords, i = m[j:], len(m)
            if coords:
                out.append(coords)
    return out


def parse_prediction(text: str) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    """parse_prediction（:130–186）：{类名（strip 后，原样大小写）: [坐标串（网格整数）, …]}，同名的组按出现顺序接起来（:336–338 的
    result[category].extend）。另返回计数：object_ref 的个数、完整的组数、是否走了"截断"那条路。"""
    text = text.split(EOS_TEXT)[0]  # :158
    matches = GROUP_RE.findall(text)  # :162
    result: dict[str, list[list[int]]] = {}
    for cat, box_text in matches:  # :165–166
        result.setdefault(cat.strip(), []).extend(_box_text_bins(box_text, is_incomplete=False))
    n_ref = text.count(OBJ_START)  # :170
    incomplete = False
    if n_ref == 1 and not matches:  # :172
        mm = INCOMPLETE_RE.search(text)
        if mm:
            cat, box_text = mm.groups()
            result.setdefault(cat.strip(), []).extend(_box_text_bins(box_text, is_incomplete=True))
            incomplete = True
    return result, {"n_object_ref": n_ref, "n_groups": len(matches) + int(incomplete), "incomplete": incomplete}


def category_key(name: str) -> str:
    """类名的归一化（convert_coco_lvis_to_standard_format.py:13 对数据集的类名、:96 对模型输出的类名，同一个写法）。"""
    return str(name).lower().replace("_", " ")


def bins_to_xyxy(bins: Sequence[int], width: float, height: float) -> list[float]:
    """网格整数（x0, y0, x1, y1）→ 原图像素（:239–250：(b / 999.0) · w 或 · h）。不截到图内（官方不截；网格 > 999 的另记数）。"""
    x0, y0, x1, y1 = (int(b) for b in bins)
    return [(x0 / BIN_DIVISOR) * width, (y0 / BIN_DIVISOR) * height, (x1 / BIN_DIVISOR) * width, (y1 / BIN_DIVISOR) * height]


def xyxy_to_bins(box: Sequence[float], width: float, height: float) -> list[int]:
    """原图像素 → 网格整数（rex_omni/parser.py:269–298 的 convert_boxes_to_normalized_bins：先截到 [0, 1]，int(· 999) 再截到
    [0, 999]）。只给往返测试与假模型用（官方的视觉提示任务用它，检测不用）。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    out = []
    for v, s in ((x0, width), (y0, height), (x1, width), (y1, height)):
        n = max(0.0, min(1.0, v / s))
        out.append(max(0, min(BIN_MAX, int(n * BIN_MAX))))
    return out


def format_boxes(category: str, bins: Sequence[Sequence[int]]) -> str:
    """官方输出的格式（rex_omni/parser.py:39 的示例；inference_text_prompt.py:136）：
    <|object_ref_start|>类名<|object_ref_end|><|box_start|><x0><y0><x1><y1>, <…><|box_end|><|im_end|>。只给测试与假模型用。"""
    body = ", ".join("".join(f"<{int(v)}>" for v in b) for b in bins)
    return f"{OBJ_START}{category}{OBJ_END}{BOX_START}{body}{BOX_END}{EOS_TEXT}"


def format_groups(groups: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> str:
    """多个类别的官方输出（inference_text_prompt.py:136 的示例：组与组之间 ", "）。只给测试与假模型用。"""
    body = ", ".join(format_boxes(c, b)[: -len(EOS_TEXT)] for c, b in groups)
    return body + EOS_TEXT


def select_boxes(extracted: Mapping[str, Sequence[Sequence[int]]], category: str, width: float, height: float, *,
                 max_dets: int) -> tuple[list[list[float]], list[list[int]], dict[str, int]]:
    """一个对的预测框（convert_coco_lvis_to_standard_format.py:93–116 的那一步，只看本对的类别）：
    - 按解析结果的键序（:95）与组内顺序取，类名归一化后等于本对的类别（:96–99）；别的类名的坐标串丢掉并记数（官方是记到那个类别上、
      再按 positive_only 过滤；每次只问一个类别时，这些框不属于本对）；
    - 只认恰好 4 个数的坐标串（:101–104 x1, y1, x2, y2 = coord），其余（点、多边形）丢掉并记数；
    - 不去重、不改顺序（官方没有 NMS，分数都是常数 0.2，:113）；最多保留 max_dets 个（XM-3a 的 100，COCOeval 的 maxDets）。
    返回 (原图像素的框, 网格整数, 计数)。"""
    want = category_key(category)
    boxes: list[list[float]] = []
    bins: list[list[int]] = []
    st = {"n_coord_lists": 0, "n_other_category": 0, "n_not_box": 0, "n_boxes_parsed": 0, "n_capped": 0, "n_out_of_grid": 0,
          "n_inverted": 0}
    for phrase, lists in extracted.items():
        same = category_key(phrase) == want
        for c in lists:
            st["n_coord_lists"] += 1
            if not same:
                st["n_other_category"] += 1
                continue
            if len(c) != 4:
                st["n_not_box"] += 1
                continue
            st["n_boxes_parsed"] += 1
            if len(boxes) >= max_dets:
                st["n_capped"] += 1
                continue
            st["n_out_of_grid"] += int(any(int(v) > BIN_MAX for v in c))
            st["n_inverted"] += int(not (c[2] > c[0] and c[3] > c[1]))
            bins.append([int(v) for v in c])
            boxes.append(bins_to_xyxy(c, width, height))
    return boxes, bins, st


def select_joint(extracted: Mapping[str, Sequence[Sequence[int]]], categories: Sequence[str], width: float, height: float, *,
                 max_dets: int) -> tuple[dict[str, tuple[list[list[float]], list[list[int]], dict[str, int]]], int]:
    """joint：一次输出按类名分给所问的各个类别（convert_coco_lvis_to_standard_format.py:93–116：类名归一化后对上数据集的类别、
    positive_only）。每个所问类别内同 select_boxes（键序与组内顺序、只认 4 个数、最多 max_dets 个）；类名不在所问类别里的坐标串丢掉，
    返回它们的个数。返回 ({类别: (框, 网格, 计数)}, 未问类名的坐标串数)。所问类别归一化后不能重名。"""
    keys = [category_key(c) for c in categories]
    if len(set(keys)) != len(keys):
        raise ValueError(f"所问类别归一化后重名：{list(categories)}")
    by: dict[str, dict[str, Sequence[Sequence[int]]]] = {k: {} for k in keys}
    n_unasked = 0
    for phrase, lists in extracted.items():
        k = category_key(phrase)
        if k in by:
            by[k][phrase] = lists
        else:
            n_unasked += len(lists)
    out = {c: select_boxes(by[k], c, width, height, max_dets=max_dets) for c, k in zip(categories, keys)}
    return out, n_unasked


# ---------------------------------------------------------------------------
# 身份与出处
# ---------------------------------------------------------------------------

QUERY_JOINT, QUERY_PER_CATEGORY = "joint", "per_category"
QUERY_MODES: tuple[str, ...] = (QUERY_JOINT, QUERY_PER_CATEGORY)
# DEV-0083 用官方 Rex-Omni-EvalData 的 _annotations/box_eval/COCO.jsonl 核对（2026-11-04）：xm_tune 的 300 张图全在其中、类别集合合计
# 923 个（与 xm_tune 的 923 对相同），239 张多类别图里 67 张的顺序是 category_id 升序、172 张不是（例：632 号图 bed, potted plant, book,
# chair）。管理员裁定 joint 用 category_id 升序，记为一处偏离。
JOINT_ORDER = "category_id 升序（官方 COCO.jsonl 的顺序不是升序：xm_tune 239 张多类别图里 67 张是升序、172 张不是）"
COMMON_DEVIATIONS: tuple[str, ...] = (
    "transformers 后端（官方 wrapper 的 transformers 分支），不是论文读数用的 vLLM；vLLM 的停止串 <0><0><0><0> 与 max_model_len 4096 照搬",
    "注意力 sdpa（官方缺省 flash_attention_2；服务器没有 flash-attn）",
    "分词器由官方慢分词器文件重建（src/rexomni/model.build_tokenizer）：transformers 5 直接读会把 1,022 个新增 token 的 id 排错；"
    "重建后与 tokenizer_config.json 逐个相同，与慢分词器只在含被顶掉的 1,000 个罕见 BPE token 的文本上不同",
    "不在所问类别里的类名下的框丢掉并计数（官方是记到那个类别上再按 positive_only 过滤）",
)
MODE_DEVIATIONS: Mapping[str, tuple[str, ...]] = {
    QUERY_JOINT: (f"提示里的类别顺序：{JOINT_ORDER}；类别 = 该图在本划分里的全部正类别（lvis_sub 是每图至多 3 类的上限之后的那几类）",),
    QUERY_PER_CATEGORY: ("每个（图像, 类别）调用一次，提示里只有这一个类别（与 X-M 同协议，诊断）。官方 COCO/LVIS 评测是每图一次、提示里用 ', ' "
                         "连起该图全部正类别（inference_text_prompt.py:599；Rex-Omni-EvalData 的 COCO.jsonl）：本读数不是官方协议的复现",),
}


def deviations(query_mode: str) -> list[str]:
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode 必须是 {QUERY_MODES} 之一，收到 {query_mode!r}")
    return [*MODE_DEVIATIONS[query_mode], *COMMON_DEVIATIONS]


def identity(query_mode: str) -> dict[str, Any]:
    """进运行身份的协议一节。"""
    return {"upstream": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "license": UPSTREAM_LICENSE, "paper": PAPER,
            "weights": {"repo": WEIGHTS_REPO, "revision": WEIGHTS_REVISION}, "query_mode": query_mode,
            "category_order": JOINT_ORDER if query_mode == QUERY_JOINT else None,
            "system_prompt": SYSTEM_PROMPT, "question_template": QUESTION_TEMPLATE, "category_join": CATEGORY_JOIN,
            "image": {"mode": IMAGE_MODE, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "factor": RESIZE_FACTOR,
                      "resize": "smart_resize（Qwen2.5-VL；PIL 后端的图像处理器，bicubic）"},
            "generation": generation_identity(),
            "parsing": {"source": "evaluation/inference_text_prompt.py parse_prediction / _process_box_text",
                        "bins": f"<0>…<{BIN_MAX}>，像素 = 网格 / {BIN_DIVISOR} × 原图宽或高，不截到图内",
                        "boxes": "只认恰好 4 个数的坐标串（convert_coco_lvis_to_standard_format.py:102），类名 lower() 且 '_' → ' ' 后比较",
                        "order": "输出顺序，不去重、不加 NMS（官方没有；分数常数 0.2）"},
            "deviations": deviations(query_mode)}


# (文件, 行号, 该行逐字含有的片段, 项)。行号是 commit 6508981c 的；给了官方仓库的签出（环境变量 REXOMNI_UPSTREAM_DIR）时
# tests/test_rexomni.py 逐条核对。
CITATIONS: tuple[tuple[str, int, str, str], ...] = (
    ("evaluation/inference_text_prompt.py", 12, 'DEFAULT_PROMPT = "You are a helpful assistant"', "系统提示"),
    ("evaluation/inference_text_prompt.py", 81, "system_prompt = DEFAULT_PROMPT", "inference() 总用 DEFAULT_PROMPT"),
    ("evaluation/inference_text_prompt.py", 45, 'parser.add_argument("--max_new_tokens", type=int, default=4096)', "max_new_tokens"),
    ("evaluation/inference_text_prompt.py", 46, 'parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)', "min_pixels"),
    ("evaluation/inference_text_prompt.py", 47, 'parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)', "max_pixels"),
    ("evaluation/inference_text_prompt.py", 58, 'default="Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",',
     "检测的提问模板"),
    ("evaluation/inference_text_prompt.py", 599, 'question = args.question.replace("[OBJ]", ", ".join(categories))', "类别用 ', ' 连起"),
    ("evaluation/inference_text_prompt.py", 592, 'image = Image.open(full_image_path).convert("RGB")', "读图：RGB"),
    ("evaluation/inference_text_prompt.py", 593, "w, h = image.size", "换算用原图尺寸"),
    ("evaluation/inference_text_prompt.py", 120, "outputs = model.generate([llm_inputs], sampling_params=sampling_params)", "每次一个请求"),
    ("evaluation/inference_text_prompt.py", 527, 'tokenizer_mode="slow",', "慢分词器"),
    ("evaluation/inference_text_prompt.py", 529, "max_model_len=4096,", "上下文上限"),
    ("evaluation/inference_text_prompt.py", 537, '"<|im_end|>",  # Standard end token', "停止串 <|im_end|>"),
    ("evaluation/inference_text_prompt.py", 539, '"<0><0><0><0>",  # Repeated zero coordinates', "停止串 <0><0><0><0>"),
    ("evaluation/inference_text_prompt.py", 545, "top_p=0.8,", "top_p"),
    ("evaluation/inference_text_prompt.py", 546, "repetition_penalty=1.05,  # Increased to reduce repetition", "repetition_penalty"),
    ("evaluation/inference_text_prompt.py", 547, "top_k=1,", "top_k"),
    ("evaluation/inference_text_prompt.py", 548, "temperature=0.0,", "temperature 0"),
    ("evaluation/inference_text_prompt.py", 549, "skip_special_tokens=False,", "不跳过特殊 token"),
    ("evaluation/inference_text_prompt.py", 158, 'text = text.split("<|im_end|>")[0]', "解析：去掉 <|im_end|> 之后"),
    ("evaluation/inference_text_prompt.py", 161,
     r'pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"', "解析：完整的组"),
    ("evaluation/inference_text_prompt.py", 172, "if object_ref_count == 1 and not matches:", "解析：只有一个组且没收尾"),
    ("evaluation/inference_text_prompt.py", 174,
     r'incomplete_pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)$"', "解析：截断的组"),
    ("evaluation/inference_text_prompt.py", 175, "incomplete_match = re.search(incomplete_pattern, text, re.DOTALL)", "截断的组：DOTALL"),
    ("evaluation/inference_text_prompt.py", 204, 'polygon_sequences = [seq.strip() for seq in box_text.split(",") if seq.strip()]', "按逗号分段"),
    ("evaluation/inference_text_prompt.py", 209, r'box_pattern = r"<(\d+)>"', "坐标 token"),
    ("evaluation/inference_text_prompt.py", 229, "if j + 3 < len(box_matches):", "还剩至少 4 个数"),
    ("evaluation/inference_text_prompt.py", 237, "if x1_bin > x0_bin and y1_bin > y0_bin:", "合法的框"),
    ("evaluation/inference_text_prompt.py", 239, "x0_norm = x0_bin / 999.0", "网格 / 999"),
    ("evaluation/inference_text_prompt.py", 245, "x0 = x0_norm * w", "× 原图宽"),
    ("evaluation/inference_text_prompt.py", 277, "valid_coords_count = (remaining_coords // 2) * 2", "截断时取偶数个"),
    ("evaluation/inference_text_prompt.py", 338, "result[category].extend(boxes)", "同名的组接起来"),
    ("evaluation/scrpts/eval_coco.sh", 16, "MAX_TOKENS=4096", "COCO：max_new_tokens"),
    ("evaluation/scrpts/eval_coco.sh", 17, "MIN_PIXELS=$((16 * 28 * 28))", "COCO：min_pixels"),
    ("evaluation/scrpts/eval_coco.sh", 18, "MAX_PIXELS=$((2560 * 28 * 28))", "COCO：max_pixels"),
    ("evaluation/scrpts/eval_coco.sh", 46, '--question "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format."',
     "COCO：提问"),
    ("evaluation/scrpts/eval_coco.sh", 57, "--positive_only", "COCO：只认正类别"),
    ("evaluation/scrpts/eval_lvis.sh", 16, "MAX_TOKENS=4096", "LVIS：max_new_tokens"),
    ("evaluation/scrpts/eval_lvis.sh", 46, '--question "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format."',
     "LVIS：提问"),
    ("evaluation/utils/convert_coco_lvis_to_standard_format.py", 13, 'cat["name"].lower().replace("_", " "): cat["id"]', "数据集类名的归一化"),
    ("evaluation/utils/convert_coco_lvis_to_standard_format.py", 96, 'cate_key = str(cate_name).lower().replace("_", " ")', "输出类名的归一化"),
    ("evaluation/utils/convert_coco_lvis_to_standard_format.py", 102, "x1, y1, x2, y2 = coord", "只认 4 个数"),
    ("evaluation/utils/convert_coco_lvis_to_standard_format.py", 107, "if args.positive_only and image_id in image_id_to_cat_ids:", "positive_only"),
    ("evaluation/utils/convert_coco_lvis_to_standard_format.py", 113, '"conf": 0.2,', "常数分数（按输出顺序）"),
    ("rex_omni/wrapper.py", 29, 'system_prompt: str = "You are a helpful assistant",', "wrapper：系统提示"),
    ("rex_omni/wrapper.py", 30, "min_pixels: int = 16 * 28 * 28,", "wrapper：min_pixels"),
    ("rex_omni/wrapper.py", 31, "max_pixels: int = 2560 * 28 * 28,", "wrapper：max_pixels"),
    ("rex_omni/wrapper.py", 32, "max_tokens: int = 4096,", "wrapper：max_tokens"),
    ("rex_omni/wrapper.py", 36, "repetition_penalty: float = 1.05,", "wrapper：repetition_penalty"),
    ("rex_omni/wrapper.py", 141, "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration", "transformers 的类"),
    ("rex_omni/wrapper.py", 146, 'torch_dtype=kwargs.get("torch_dtype", torch.bfloat16),', "bf16"),
    ("rex_omni/wrapper.py", 148, '"attn_implementation", "flash_attention_2"', "官方缺省 flash_attention_2"),
    ("rex_omni/wrapper.py", 170, "use_fast=False,", "慢处理器（PIL 后端、慢分词器）"),
    ("rex_omni/wrapper.py", 174, 'self.processor.tokenizer.padding_side = "left"', "左侧补齐"),
    ("rex_omni/wrapper.py", 837, '"do_sample": self.temperature > 0,  # Enable sampling if temperature > 0', "温度 0 → 贪心"),
    ("rex_omni/wrapper.py", 838, '"pad_token_id": self.processor.tokenizer.eos_token_id,', "pad = eos"),
    ("rex_omni/wrapper.py", 855, "skip_special_tokens=self.skip_special_tokens,", "解码不跳过特殊 token"),
    ("rex_omni/wrapper.py", 856, "clean_up_tokenization_spaces=False,", "解码不清理空格"),
    ("rex_omni/tasks.py", 43, 'prompt_template="Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format.",',
     "wrapper 的检测模板"),
    ("rex_omni/parser.py", 278, "x0_norm = max(0.0, min(1.0, x0 / ori_width))", "框 → 网格：先截到 [0, 1]"),
    ("rex_omni/parser.py", 284, "x0_bin = max(0, min(999, int(x0_norm * 999)))", "框 → 网格：int(· 999)"),
)


__all__ = [
    "BIN_DIVISOR", "BIN_MAX", "CITATIONS", "COMMON_DEVIATIONS", "EOS_TEXT", "JOINT_ORDER", "MAX_MODEL_LEN", "MAX_NEW_TOKENS", "MAX_PIXELS", "MIN_PIXELS",
    "QUERY_JOINT", "QUERY_MODES", "QUERY_PER_CATEGORY", "QUESTION_TEMPLATE", "REPETITION_PENALTY", "STOP_STRINGS", "SYSTEM_PROMPT", "UPSTREAM_COMMIT", "UPSTREAM_URL",
    "WEIGHTS_REPO", "WEIGHTS_REVISION", "bins_to_xyxy", "category_key", "detect_prompt", "deviations", "expected_grid", "format_boxes", "format_groups", "identity",
    "max_new_tokens_for", "messages", "n_visual_tokens", "parse_prediction", "select_boxes", "select_joint", "smart_resize", "truncate_at_stop",
    "xyxy_to_bins",
]
