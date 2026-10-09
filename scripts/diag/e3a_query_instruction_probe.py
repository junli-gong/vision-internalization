#!/usr/bin/env python
"""DEV-0041 §4 诊断（只读，不改任何实验代码）：E3a 的查询 token 在 LLM 里能不能"看到"指令？

全部走 E3a 的正式路径：inject_query_tokens → build_inputs(order = instruction_first) → features.llm_inputs（= training_forward +
HiddenRequest(regions=("query",))，层号 0..num_layers()），查询嵌入取 train.init_query_embeds（小模型）或检查点里训练好的。比较：
  - 同一张图、两句同样 token 数、只差内容的指令（A vs C：之后所有位置的 RoPE 位置都相同，纯粹是指令内容的影响）；
  - 同一张图、两句长度不同的指令（A vs B：另带图像与查询 token 的位置平移）；
  - 同一张图、指令 A vs 空串（= 评测的 no_instruction 泄漏检查，指令取配置 eval.leak_checks.no_instruction.instruction）；
  - 同一句指令 A、两张不同的图（对照：图像的影响）；
  - 同一条输入前向两次（确定性）；单独前向 vs 与另外两条左填充成一批（批处理一致性）。
每层报告 rel = ||a − b||_F / ||a||_F、max|a − b|，以及逐 token LayerNorm 之后（LayerMix 实际看到的）的 rel_ln；表里标出每个层号
对应的 decoder 层类型（Qwen3.5：linear_attention = Gated DeltaNet，full_attention = 因果全注意力）。
读出：小模型用随机初始化的 tiny Readout（均匀混合），检查点模式用训练好的读出；报告 refseg logit 的相对差与 logit > 0 的 mask 之间的 IoU。
另外两项（--no-attn 关掉）：
  - 注意力：把注意力临时切到 eager，钩住 LLM 里每个全注意力模块，报告查询行的注意力质量落在 指令 / 图像 / 查询 / 其余 token 上的
    比例（对头与查询取平均；另给"均匀因果注意力"下指令应得的比例作参照，以及指令质量最大的那个头）；
  - Gated DeltaNet 的衰减（只有 Qwen3.5）：钩住每个 GDN 层的 in_proj_a，按 HF 的公式 g = −exp(A_log)·softplus(a + dt_bias) 算从指令末尾到
    第一个查询之间 Σg（状态里指令那部分最多还剩 exp(Σg)；不计 delta rule 的擦除，是上界），报告各头的 log10 最大值与中位数。

运行：
  小模型（开发机，CPU、fp32，约 1 分钟）：
    HF_HUB_OFFLINE=1 python scripts/diag/e3a_query_instruction_probe.py [--model qwen3.5-0.8b gemma-4-e2b-it internvl3.5-2b-hf] \
        [--init-std config|hf] [--position-mode grid2d|seq1d]
  真检查点（服务器；<ckpt> 是 E3a 的 .../ckpt/final，含 model/ 与 readout/；图与两句指代句最好取 RefCOCO val 里同一张图上的两个物体）：
    python scripts/diag/e3a_query_instruction_probe.py --checkpoint <ckpt> --device cuda --dtype bf16 \
        --image <同一张图> <另一张图> --expr-a "<物体 1 的指代句>" --expr-b "<物体 2 的指代句>"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.adapters import registry  # noqa: E402
from src.adapters.factory import make_adapter  # noqa: E402
from src.adapters.image_policy import parse_policy  # noqa: E402
from src.readout.config import load_config, readout_spec  # noqa: E402
from src.readout.features import llm_inputs  # noqa: E402
from src.readout.model import Readout  # noqa: E402
from src.readout.train import init_query_embeds, layers_for  # noqa: E402

CFG = load_config()
DEFAULT_MODELS = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# 搭建
# ---------------------------------------------------------------------------


def setup_tiny(key: str, init_std: str, mode_override: str | None):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    kw = {"init_std": float(t["init_std"])} if init_std == "config" else {}
    torch.manual_seed(0)
    ad = make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]),
                      tiny=True, **kw)
    mode = mode_override or CFG.position_mode(fam)
    ad.inject_query_tokens(CFG.query_n, mode)
    qe = init_query_embeds(ad, CFG.query_n, float(CFG.doc["query"]["init_noise_ratio"]), seed=0)
    ad.eval_mode()
    torch.manual_seed(0)
    spec = readout_spec(CFG.doc, variant="full", hidden_size=ad.hidden_size(), n_layers=ad.num_layers(), tiny=True)
    ro = Readout(spec).eval()
    label = f"{key} (tiny random, init_std={init_std}, readout = random tiny Readout)"
    return ad, qe, layers_for(ad), ro, label


def setup_ckpt(path: str, device: str, dtype: str):
    from src.readout.evaluate import EvalModel

    m = EvalModel.from_checkpoint(path, CFG, device=device, dtype=DTYPES[dtype])
    if m.variant == "c2":
        raise SystemExit("c2 没有查询 token，这个诊断不适用")
    label = f"checkpoint {path} (variant {m.variant}, {dtype} on {device}, readout = trained)"
    return m.ad, m.query_embeds, m.layers, m.readout, label


def synthetic_images() -> list[np.ndarray]:
    """两张 96 × 128 的合成图：背景 + 三块彩色矩形（布局、颜色不同）。"""
    rng = np.random.default_rng(0)
    out = []
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        img = np.full((96, 128, 3), 90, np.uint8) + rng.integers(0, 10, (96, 128, 3), dtype=np.uint8)
        for _ in range(3):
            y, x = int(r.integers(0, 60)), int(r.integers(0, 90))
            img[y: y + 30, x: x + 36] = r.integers(0, 255, 3, dtype=np.uint8)
        out.append(img)
    return out


def load_images(paths: list[str] | None) -> list[np.ndarray]:
    if not paths:
        return synthetic_images()
    from src.e3.data import load_rgb

    imgs = [load_rgb(p) for p in paths]
    if len(imgs) == 1:
        imgs.append(np.ascontiguousarray(imgs[0][::-1, ::-1]))  # 只给一张时，第二张取它旋转 180°
    return imgs[:2]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / a.norm().clamp_min(1e-30))


def ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6)


def layer_types(ad) -> list[str] | None:
    cfg = ad.model.config
    tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    lt = getattr(tc, "layer_types", None)
    return list(lt) if lt is not None else None


def cols(ad, span) -> torch.Tensor:
    """单行（无填充）提示行里的列号：ModelInputs 的下标 + 家族的偏移（_user_index_shift，与 training_forward 相同）。"""
    off, start = ad._user_index_shift(CFG.enable_thinking)
    return torch.tensor([i + off if i >= start else i for i in range(span.start, span.end)], dtype=torch.long)


def layout(ad, inp) -> str:
    ids = inp.tensors["input_ids"][0].tolist()
    ins, q, img = inp.instruction_span, inp.query_span, inp.images[0].span
    row = ad._prompt_row(inp, CFG.enable_thinking)  # 与 training_forward 同一个函数
    off, _ = ad._user_index_shift(CFG.enable_thinking)
    tok = ad.tokenizer

    def dec(a, b):
        s = tok.decode(ids[a:b]) if b > a else ""
        return repr(s if len(s) < 90 else s[:40] + " ... " + s[-40:])

    return "\n".join([
        f"    user turn = {len(ids)} tokens; prompt row (+ family prefix shift {off}, + assistant header) = {len(row.ids)}",
        f"    [0,{ins.start}) {dec(0, ins.start)}",
        f"    [{ins.start},{ins.end}) INSTRUCTION {dec(ins.start, ins.end)}",
        f"    [{ins.end},{img.start}) {dec(ins.end, img.start)}",
        f"    [{img.start},{img.end}) IMAGE x{len(img)} (grid {inp.images[0].grid_hw})",
        f"    [{img.end},{q.start}) {dec(img.end, q.start)}",
        f"    [{q.start},{q.end}) QUERY x{len(q)} ({inp.query_position_mode})",
        f"    [{q.end},{len(ids)}) {dec(q.end, len(ids))}",
        f"    + assistant header {tok.decode(list(row.ids[len(ids) + off:]))!r}",
    ])


def lm_attention_modules(model) -> list[tuple[str, torch.nn.Module]]:
    out = []
    for n, m in model.named_modules():
        if type(m).__name__.endswith("Attention") and hasattr(m, "q_proj") and hasattr(m, "k_proj") \
                and "visual" not in n and "vision" not in n:
            out.append((n, m))
    return out


def attention_report(ad, inp, qe) -> None:
    """查询行的注意力质量落在各段上的比例（eager 注意力，临时切换，结束后切回 sdpa）。"""
    model = ad.model.get_base_model() if hasattr(ad.model, "get_base_model") else ad.model
    mods = lm_attention_modules(model)
    if not mods:
        print("  [attn] 没有找到 LLM 的全注意力模块")
        return
    caps: dict[str, torch.Tensor | None] = {}
    hooks = [m.register_forward_hook(lambda _m, _a, out, n=n: caps.__setitem__(
        n, None if not isinstance(out, tuple) or out[1] is None else out[1].detach()[0].float().cpu())) for n, m in mods]
    model.set_attn_implementation("eager")
    try:
        with torch.no_grad():
            llm_inputs(ad, [inp], qe, layers=[ad.num_layers()], enable_thinking=CFG.enable_thinking)
    finally:
        for h in hooks:
            h.remove()
        model.set_attn_implementation("sdpa")
    qc, ic, gc = cols(ad, inp.query_span), cols(ad, inp.instruction_span), cols(ad, inp.images[0].span)
    if len(ic) == 0:
        return
    print(f"  [attn] query-row attention mass (mean over heads x {len(qc)} query rows); "
          f"'uniform' = share the instruction would get under uniform causal attention")
    print(f"  [attn] {'module':<46} {'instr':>8} {'uniform':>8} {'image':>8} {'query':>8} {'other':>8} {'instr max-head':>15}")
    for n, w in caps.items():
        if w is None:
            print(f"  [attn] {n:<46} (no weights returned)")
            continue
        rows = w[:, qc]  # [heads, Q, L]
        tot = rows.sum(-1, keepdim=True)
        mi = rows[..., ic].sum(-1) / tot[..., 0]
        mg = rows[..., gc].sum(-1) / tot[..., 0]
        mq = rows[..., qc].sum(-1) / tot[..., 0]
        uni = float(np.mean([len(ic) / (int(c) + 1) for c in qc]))
        per_head = mi.mean(-1)
        print(f"  [attn] {n:<46} {float(mi.mean()):8.4f} {uni:8.4f} {float(mg.mean()):8.4f} {float(mq.mean()):8.4f} "
              f"{float(1 - mi.mean() - mg.mean() - mq.mean()):8.4f} {float(per_head.max()):8.4f} (h{int(per_head.argmax())})")


def gdn_report(ad, inp, qe) -> None:
    """Gated DeltaNet：从指令末尾到第一个查询之间的 Σg（各头），log10 的最大值与中位数（状态里指令那部分最多还剩 10^这个数）。"""
    model = ad.model.get_base_model() if hasattr(ad.model, "get_base_model") else ad.model
    mods = [(n, m) for n, m in model.named_modules() if hasattr(m, "A_log") and hasattr(m, "in_proj_a")
            and hasattr(m, "dt_bias")]
    if not mods:
        return
    caps: dict[str, torch.Tensor] = {}
    hooks = [m.in_proj_a.register_forward_hook(lambda _m, _a, out, n=n: caps.__setitem__(n, out.detach()[0].float()))
             for n, m in mods]
    try:
        with torch.no_grad():
            llm_inputs(ad, [inp], qe, layers=[ad.num_layers()], enable_thinking=CFG.enable_thinking)
    finally:
        for h in hooks:
            h.remove()
    ic, qc = cols(ad, inp.instruction_span), cols(ad, inp.query_span)
    lo, hi = int(ic[-1]) + 1, int(qc[0])  # 指令最后一个 token 之后、到第一个查询（含）
    lo2 = int(qc[-1])
    print(f"  [gdn] decay from end of instruction to first query ({hi - lo + 1} steps) / to last query ({lo2 - lo + 1} steps): "
          "log10 exp(sum g) per head (upper bound on surviving instruction state)")
    for n, m in mods:
        a = caps[n]
        with torch.no_grad():
            g = -m.A_log.detach().float().exp().to(a.device) * F.softplus(a + m.dt_bias.detach().float().to(a.device))  # [L, heads]
        s1 = g[lo: hi + 1].sum(0) / math.log(10)
        s2 = g[lo: lo2 + 1].sum(0) / math.log(10)
        print(f"  [gdn] {n:<46} first query: max {float(s1.max()):9.2f} median {float(s1.median()):9.2f} | "
              f"last query: max {float(s2.max()):9.2f} median {float(s2.median()):9.2f}  ({g.shape[1]} heads)")


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------


def analyze(ad, qe, layers, ro, label, imgs, expr_a, expr_b, expr_c, *, attn: bool) -> None:
    order = CFG.order("refseg")
    ia, ib, ic = (CFG.instruction("refseg", e) for e in (expr_a, expr_b, expr_c))
    ni = str(CFG.doc["eval"]["leak_checks"]["no_instruction"]["instruction"])
    print(f"\n=== {label} ===")
    print(f"  family {ad.family}; hidden {ad.hidden_size()}; layers {layers[0]}..{layers[-1]}; queries {len(qe)}; "
          f"order {order}; enable_thinking {CFG.enable_thinking}")
    lt = layer_types(ad)
    cases = {"A_img1": (imgs[0], ia), "C_img1": (imgs[0], ic), "B_img1": (imgs[0], ib), "empty_img1": (imgs[0], ni),
             "A_img2": (imgs[1], ia)}
    inputs = {k: ad.build_inputs([im], ins, order) for k, (im, ins) in cases.items()}
    la, lc = len(inputs["A_img1"].instruction_span), len(inputs["C_img1"].instruction_span)
    print(f"  instructions: A {ia!r} ({la} tok), C {ic!r} ({lc} tok{'' if la == lc else ' -- NOT same length'}), "
          f"B {ib!r} ({len(inputs['B_img1'].instruction_span)} tok), empty {ni!r}")
    print(f"  layout A_img1 (instruction {ia!r}):")
    print(layout(ad, inputs["A_img1"]))
    print(f"  layout empty_img1 (instruction {ni!r}):")
    print(layout(ad, inputs["empty_img1"]))

    with torch.no_grad():
        single = {k: llm_inputs(ad, [v], qe, layers=layers, enable_thinking=CFG.enable_thinking) for k, v in inputs.items()}
        again = llm_inputs(ad, [inputs["A_img1"]], qe, layers=layers, enable_thinking=CFG.enable_thinking)
        batched = llm_inputs(ad, [inputs["A_img1"], inputs["empty_img1"], inputs["B_img1"]], qe, layers=layers,
                             enable_thinking=CFG.enable_thinking)
    h = {k: v.query[:, 0] for k, v in single.items()}  # [S, Q, H]
    print(f"  prompt lengths: { {k: v.prompt_lengths[0] for k, v in single.items()} }; batched {batched.prompt_lengths}")
    pairs = [("A vs C (same len)", "A_img1", "C_img1"), ("A vs B", "A_img1", "B_img1"),
             ("A vs empty (=no_instr)", "A_img1", "empty_img1"), ("img1 vs img2 (instr A)", "A_img1", "A_img2")]
    print("  query hidden states, per layer (rel = ||a-b||_F/||a||_F over the 256 x H block; rel_ln = same after per-token "
          "LayerNorm, as LayerMix sees it). 'A vs C' isolates instruction content (same token count, so all later positions "
          "are identical); A vs B / A vs empty also shift the RoPE positions of image + query tokens")
    print(f"  {'layer':>5} {'type':<17}| " + " | ".join(f"{p[0]:^22}" for p in pairs))
    print(f"  {'':>5} {'':<17}| " + " | ".join(f"{'rel':>10} {'rel_ln':>10}" for _ in pairs))
    for s, l in enumerate(layers):
        typ = "embed(+vision)" if l == 0 else (lt[l - 1] if lt else "decoder")
        if l == ad.num_layers():
            typ += "+norm"
        cells = []
        for _, a, b in pairs:
            x, y = h[a][s], h[b][s]
            cells.append(f"{rel(x, y):10.3e} {rel(ln(x), ln(y)):10.3e}")
        print(f"  {l:5d} {typ[:17]:<17}| " + " | ".join(cells))
    det = max(float((again.query[s, 0].float() - h["A_img1"][s].float()).abs().max()) for s in range(len(layers)))
    bat = max(rel(h["A_img1"][s], batched.query[s, 0]) for s in range(len(layers)))
    bat_e = max(rel(h["empty_img1"][s], batched.query[s, 1]) for s in range(len(layers)))
    print(f"  same input twice: max|d| = {det:.3e}; batched (left-padded) vs single, max rel over layers: "
          f"A {bat:.3e}, empty {bat_e:.3e}")
    last = ln(h["A_img1"][-1])
    cos = F.normalize(last, dim=-1) @ F.normalize(last, dim=-1).t()
    nq = last.shape[0]
    print(f"  last layer: mean pairwise cosine among the {nq} query states (after LN) = "
          f"{float((cos.sum() - cos.diag().sum()) / (nq * (nq - 1))):.4f}")

    dev = next(ro.parameters()).device
    with torch.no_grad():
        out = {k: ro.task_output("refseg", ro(v.query.to(dev)).embed)[0, 0].float().cpu() for k, v in single.items()}
        mixed = {k: ro.mix(v.query.to(dev))[0].float().cpu() for k, v in single.items()}

    def miou(a, b):
        ma, mb = out[a] > 0, out[b] > 0
        u = int((ma | mb).sum())
        return float((ma & mb).sum()) / u if u else 1.0

    print("  readout LayerMix output (decoder input):")
    for name, a, b in pairs:
        print(f"    {name:<24} rel {rel(mixed[a], mixed[b]):.3e}")
    print("  readout refseg logits [128x128]:")
    for name, a, b in pairs:
        print(f"    {name:<24} rel {rel(out[a], out[b]):.3e}   mask IoU {miou(a, b):.3f}")
    print("  foreground fraction (logit > 0): " + ", ".join(f"{k} {float((v > 0).float().mean()):.3f}" for k, v in out.items()))
    if attn:
        attention_report(ad, inputs["A_img1"], qe)
        gdn_report(ad, inputs["A_img1"], qe)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", nargs="+", default=list(DEFAULT_MODELS), help="小模型模式：登记键（各自建一个小型随机模型）")
    ap.add_argument("--init-std", choices=("config", "hf"), default="config",
                    help="小模型：config = tests 用的 tiny.init_std 重新初始化（二维以上参数）；hf = HF 默认初始化")
    ap.add_argument("--position-mode", choices=("grid2d", "seq1d"), default=None, help="小模型：覆盖配置的家族位置模式")
    ap.add_argument("--checkpoint", default=None, help="E3a 检查点（.../ckpt/final，含 model/ 与 readout/）；给了就不跑小模型")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    ap.add_argument("--image", nargs="+", default=None, help="一到两张图（第一张用于 A/B/空串，第二张作对照）；不给用合成图")
    ap.add_argument("--expr-a", default="the person on the left")
    ap.add_argument("--expr-b", default="the dog")
    ap.add_argument("--expr-c", default="the person on the right", help="与 --expr-a 同 token 数的另一句（只差内容）")
    ap.add_argument("--no-attn", action="store_true", help="不做注意力质量与 GDN 衰减的报告")
    a = ap.parse_args(argv)
    imgs = load_images(a.image)
    exprs = (a.expr_a, a.expr_b, a.expr_c)
    if a.checkpoint:
        analyze(*setup_ckpt(a.checkpoint, a.device, a.dtype), imgs, *exprs, attn=not a.no_attn)
        return 0
    for key in a.model:
        analyze(*setup_tiny(key, a.init_std, a.position_mode), imgs, *exprs, attn=not a.no_attn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
