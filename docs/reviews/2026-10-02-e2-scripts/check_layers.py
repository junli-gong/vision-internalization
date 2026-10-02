"""独立核对 1/2/4：层号、图像区间、输入顺序（小型随机模型，CPU fp32；只读）。"""
import contextlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from src.adapters.factory import make_adapter  # noqa: E402

FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")


def img(w, h, seed=0):
    return Image.fromarray(np.random.default_rng(seed).uniform(0, 255, size=(h, w, 3)).astype(np.uint8))


def scope(ad):
    return ad.gdn_scope() if hasattr(ad, "gdn_scope") else contextlib.nullcontext()


def native_hidden(ad, inp):
    """HF 原生路径（input_ids + pixel_values，不走适配器的 inputs_embeds）。"""
    t = inp.tensors
    kw = dict(input_ids=t["input_ids"], attention_mask=t["attention_mask"], pixel_values=t["pixel_values"],
              output_hidden_states=True, use_cache=False, return_dict=True)
    if "image_grid_thw" in t:
        kw.update(image_grid_thw=t["image_grid_thw"], mm_token_type_ids=t["mm_token_type_ids"])
    if "image_position_ids" in t:
        kw.update(image_position_ids=t["image_position_ids"], mm_token_type_ids=t["mm_token_type_ids"])
    if "internvl" in ad.model_key:
        cfg = ad.model.config
        kw.update(vision_feature_layer=cfg.vision_feature_layer,
                  vision_feature_select_strategy=cfg.vision_feature_select_strategy)
    with torch.no_grad(), scope(ad):
        return ad.model(**kw).hidden_states


for key in FAMILIES:
    ad = make_adapter(key, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)
    lm = ad.model.model.language_model
    N = ad.num_layers()
    tc = ad.model.config.text_config
    print(f"\n=== {ad.model_key}  num_layers={N}  layer_types={getattr(tc, 'layer_types', None)}  "
          f"sliding_window={getattr(tc, 'sliding_window', None)}  kv_shared={getattr(tc, 'num_kv_shared_layers', None)}  "
          f"ple_dim={getattr(tc, 'hidden_size_per_layer_input', None)}  bidir={getattr(tc, 'use_bidirectional_attention', 'n/a')}  "
          f"model.training={ad.model.training}")
    for order in ("image_first", "instruction_first"):
        inp = ad.build_inputs([img(640, 480)], "Describe this image.", order)
        ids = inp.tensors["input_ids"][0].tolist()
        info = inp.images[0]
        s, e = info.span.start, info.span.end
        # 2. 区间：前后标记、区间内全是图像 token、全序列图像 token 数 = len(span)
        tok = ad.tokenizer
        conv = lambda i: tok.convert_ids_to_tokens(int(i))
        img_ids = set(ids[s:e])
        n_img_tok_total = sum(1 for x in ids if x in img_ids)
        print(f"  [{order}] span=[{s},{e}) len={e - s} grid={info.grid_hw} n_pixels={info.n_pixels} "
              f"before={conv(ids[s - 1])!r} after={conv(ids[e])!r} inside={[conv(x) for x in img_ids]} "
              f"n_img_tok_in_seq={n_img_tok_total} seq_len={len(ids)} instr_span={inp.instruction_span}")
        # 1. 层号：钩子取每层输入 / 输出与最终 norm 的输出
        cap_in, cap_out, cap_norm = {}, {}, {}
        hooks = []
        def mk(i):
            def hook(m, a, kw_, o):
                cap_in[i] = (a[0] if a else kw_["hidden_states"]).detach().clone()
                cap_out[i] = (o[0] if isinstance(o, tuple) else o).detach().clone()
                return None
            return hook
        def nhook(m, a, o):
            cap_norm[0] = o.detach().clone()
            return None
        for i, layer in enumerate(lm.layers):
            hooks.append(layer.register_forward_hook(mk(i), with_kwargs=True))
        hooks.append(lm.norm.register_forward_hook(nhook))
        layers = list(range(N + 1))
        with torch.no_grad():
            hs = ad.hidden_states(inp, layers)
            vf = ad.vision_features(inp)
        for h in hooks:
            h.remove()
        idx = torch.arange(s, e)
        ok0 = torch.equal(hs.image[0], cap_in[0][0, idx]) and torch.equal(hs.image[0], vf)
        mids = [torch.equal(hs.image[i], cap_out[i - 1][0, idx]) for i in range(1, N)]
        okN_norm = torch.equal(hs.image[N], cap_norm[0][0, idx])
        okN_raw = torch.equal(hs.image[N], cap_out[N - 1][0, idx])
        # 最后一项确实是 norm(第 N 层原始输出)
        with torch.no_grad():
            renorm = lm.norm(cap_out[N - 1])[0, idx]
        print(f"    L0 == layer[0] input == vision_features: {ok0}; L1..L{N - 1} == layer[i-1] output: {all(mids)}; "
              f"L{N} == final norm output: {okN_norm}; L{N} == raw layer[{N - 1}] output: {okN_raw}; "
              f"L{N} == norm(raw): {torch.equal(hs.image[N], renorm)}")
        # 与 HF 原生路径（input_ids + pixel_values）逐层比较
        nat = native_hidden(ad, inp)
        diffs = [float((hs.image[l] - nat[l][0, idx]).abs().max()) for l in layers]
        print(f"    native tuple len={len(nat)} (want {N + 1}); max|adapter-native| per layer = {['%.1e' % d for d in diffs]}")
        # 全序列也比较（检查 per-layer input、mask 等对非图像位置也一致）
        full = ad.sequence_hidden_states(inp, layers)
        fd = max(float((full[l] - nat[l][0]).abs().max()) for l in layers)
        print(f"    full-sequence max|adapter-native| over all layers = {fd:.1e}")

    # 4. 顺序：图在前换提示 → 图像位置不变；指令在前换指令 → 第 1 层起变化
    base = img(640, 480, seed=3)
    layers = list(range(N + 1))
    with torch.no_grad():
        a = ad.hidden_states(ad.build_inputs([base], "Describe this image.", "image_first"), layers)
        b = ad.hidden_states(ad.build_inputs([base], "Describe the photo.", "image_first"), layers)
        c = ad.hidden_states(ad.build_inputs([base], "Segment every object and estimate depth for all pixels please.", "image_first"), layers)
        ia = ad.build_inputs([base], "Find the cat.", "instruction_first")
        ib = ad.build_inputs([base], "Find the dog.", "instruction_first")
        x = ad.hidden_states(ia, layers)
        y = ad.hidden_states(ib, layers)
    print(f"  image_first same-len prompt: bitwise equal all layers = {all(torch.equal(a.image[l], b.image[l]) for l in layers)}")
    print(f"  image_first longer prompt: max diff per layer = {['%.1e' % float((a.image[l] - c.image[l]).abs().max()) for l in layers]}")
    print(f"  instruction_first spans equal={ia.images[0].span == ib.images[0].span}; "
          f"L0 equal={torch.equal(x.image[0], y.image[0])}; changed layers={[l for l in layers if not torch.equal(x.image[l], y.image[l])]}")
