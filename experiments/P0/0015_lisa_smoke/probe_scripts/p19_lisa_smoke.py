#!/usr/bin/env python
"""RUN-0015 §3：LISA-7B-v1 冒烟。按 LISA 官方 chat.py（commit 见 smoke.json）的推理流程逐步照做，改成非交互：
bf16 加载 → 一张图、一条提示 → model.evaluate(max_new_tokens=512) → 存 mask 与叠加图（与 chat.py 同样的写法）。
与 chat.py 唯一的不同：LISA-7B-v1 的 config 有 train_mask_decoder=True，LISAForCausalLM 因此忽略 vision_tower 参数、改用
config.vision_tower（Hub id "openai/clip-vit-large-patch14"）；内存盘上的 CLIP 是按固定 revision 下载的快照、没有 refs/main，
离线时 Hub id 解析不到。这里先读 config，把 vision_tower / mm_vision_tower 改成该快照的本地路径（同一份权重），再 from_pretrained。
LISA 源码不改。用 LISA 的独立 venv 运行，cwd = LISA 仓库根目录。

可选的第 6 个参数 --tokenizer-special-str（RUN-0015 唯一一次修法，默认不开）：
本 venv 没装 tokenizers（transformers 4.31 要求 tokenizers<0.14，Python 3.12 没有它的二进制包，本机也没有 Rust 无法编译）。
transformers 4.31 在没装 tokenizers 时用自带的 AddedToken 替身（dataclass，没有 __str__），LISA tokenizer_config.json 里
{"__type": "AddedToken", "content": "<s>"} 形式的 bos/eos/unk 于是被 str() 成 "AddedToken(content='<s>', ...)"，作为新词加到
32004/32005（"<unk>" 也因此被当新词加到 32006），超出 embed_tokens（32004 行）。开了这个参数就在加载分词器之前，给
transformers 4.31 的 AddedToken 替身补上 __str__ = content（tokenizers.AddedToken 的 str 就是 content），等于官方环境（装了
tokenizers）的行为；运行期打补丁，不改 LISA 或 transformers 的源文件。（先试过在 from_pretrained 里显式传字符串的 bos/eos/unk，
在 CPU 上核对无效：special_tokens_map.json 的字典仍然覆盖，没有上 GPU。）
usage: python p19_lisa_smoke.py <lisa_snapshot> <clip_snapshot> <image> <prompt> <out_dir> [--tokenizer-special-str]
"""
import json, os, sys, time, traceback

LISA_SNAP, CLIP_SNAP, IMAGE, PROMPT, OUT = sys.argv[1:6]
TOK_FIX = "--tokenizer-special-str" in sys.argv[6:]
os.makedirs(OUT, exist_ok=True)
res = {"argv": sys.argv[1:], "ok": False, "stage": "import"}


def dump():
    json.dump(res, open(os.path.join(OUT, "smoke.json"), "w"), indent=1, ensure_ascii=False, default=str)


try:
    sys.path.insert(0, os.getcwd())
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoTokenizer, CLIPImageProcessor

    from model.LISA import LISAForCausalLM
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from model.llava.model.language_model.llava_llama import LlavaConfig
    from model.segment_anything.utils.transforms import ResizeLongestSide
    from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

    res["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                       "numpy": np.__version__, "cv2": cv2.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}

    def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                   pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), img_size=1024):
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        return F.pad(x, (0, img_size - w, 0, img_size - h))

    # chat.py 默认参数：precision=bf16, image_size=1024, model_max_length=512, use_mm_start_end=True, conv_type=llava_v1
    res["stage"] = "load"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if TOK_FIX:
        import transformers.tokenization_utils_base as _tub
        if not transformers.utils.is_tokenizers_available():
            _tub.AddedToken.__str__ = lambda self: self.content
    tokenizer = AutoTokenizer.from_pretrained(LISA_SNAP, cache_dir=None, model_max_length=512, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    res["tokenizer"] = {"tokenizer_special_str_fix": TOK_FIX, "len": len(tokenizer),
                        **{k: [str(getattr(tokenizer, k)), getattr(tokenizer, k + "_id")] for k in ("bos_token", "eos_token", "unk_token", "pad_token")}}
    torch_dtype = torch.bfloat16
    config = LlavaConfig.from_pretrained(LISA_SNAP)
    res["config_vision_tower_original"] = {"vision_tower": config.vision_tower, "mm_vision_tower": config.mm_vision_tower}
    config.vision_tower = config.mm_vision_tower = CLIP_SNAP
    model = LISAForCausalLM.from_pretrained(LISA_SNAP, config=config, low_cpu_mem_usage=True, vision_tower=CLIP_SNAP,
                                            seg_token_idx=seg_token_idx, torch_dtype=torch_dtype)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model = model.bfloat16().cuda()
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=0)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(1024)
    model.eval()
    torch.cuda.synchronize()
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res["peak_mem_after_load_bytes"] = torch.cuda.max_memory_allocated()
    res["seg_token_idx"] = seg_token_idx
    res["n_params"] = sum(p.numel() for p in model.parameters())

    res["stage"] = "inference"
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    res["full_prompt"] = prompt
    image_np = cv2.imread(IMAGE)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda().bfloat16()
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().bfloat16()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()
    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                                max_new_tokens=512, tokenizer=tokenizer)
    torch.cuda.synchronize()
    res["inference_s"] = round(time.perf_counter() - t1, 2)
    res["peak_mem_inference_bytes"] = torch.cuda.max_memory_allocated()
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    res["text_output"] = tokenizer.decode(output_ids, skip_special_tokens=False).replace("\n", "").replace("  ", " ")
    res["n_output_tokens"] = int(output_ids.numel())
    res["image_size_hw"] = list(original_size_list[0])
    res["masks"] = []
    stem = os.path.splitext(os.path.basename(IMAGE))[0]
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            res["masks"].append({"i": i, "empty": True}); continue
        m = pred_mask.detach().float().cpu().numpy()[0] > 0
        ys, xs = np.nonzero(m)
        info = {"i": i, "shape_hw": list(m.shape), "n_pixels": int(m.sum()), "frac": float(m.mean())}
        if len(xs):
            info["bbox_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        cv2.imwrite(os.path.join(OUT, f"{stem}_mask_{i}.png"), (m * 255).astype(np.uint8))
        save_img = image_np.copy()
        save_img[m] = (image_np * 0.5 + m[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[m]
        cv2.imwrite(os.path.join(OUT, f"{stem}_masked_img_{i}.jpg"), cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
        res["masks"].append(info)
    res["ok"] = True
    res["stage"] = "done"
except Exception:  # noqa: BLE001
    res["traceback"] = traceback.format_exc()
    print(res["traceback"], file=sys.stderr)
finally:
    dump()
    print(json.dumps({k: v for k, v in res.items() if k not in ("traceback", "full_prompt")}, ensure_ascii=False, default=str))
sys.exit(0 if res["ok"] else 1)
