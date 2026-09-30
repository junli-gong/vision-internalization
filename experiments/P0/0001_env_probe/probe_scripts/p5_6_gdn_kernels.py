"""RUN-0001 §5.6.1: Gated DeltaNet kernel numerics on B200.
Same random inputs through:
  fla_chunk        fla.ops.gated_delta_rule.chunk_gated_delta_rule        (Triton)
  fla_fused_rec    fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule (Triton)
  fla_naive_rec    fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule (torch)
  fla_naive_chunk  fla.ops.gated_delta_rule.naive.naive_chunk_gated_delta_rule     (torch)
  hf_torch_chunk   transformers qwen3_5 torch_chunk_gated_delta_rule.__wrapped__   (torch)
  hf_torch_rec     transformers qwen3_5 torch_recurrent_gated_delta_rule.__wrapped__ (torch)
Two configs: (A) q,k pre-L2-normalised outside, all impls with in-kernel l2norm off (naive impls have no flag);
             (B) in-kernel l2norm on, for the impls that have the flag (as the model calls them).
Also a float64 recurrent reference (fla_naive_rec in fp64) as ground truth.
Shapes follow Qwen3.5-0.8B GDN: H=16 heads, K=V=128; T=300 (5 chunks of 64, last partial). B=2.
"""
import os, json, sys, torch, itertools
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule, naive_chunk_gated_delta_rule
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
import fla

OUT = os.path.join(os.environ['PROJ'], 'logs/0001/p5')
os.makedirs(OUT, exist_ok=True)
hf_chunk = mq.torch_chunk_gated_delta_rule.__wrapped__
hf_rec = mq.torch_recurrent_gated_delta_rule.__wrapped__
R = {'fla_version': fla.__version__, 'module_level_impl': {
    'torch_chunk_gated_delta_rule -> ': None, 'torch_recurrent_gated_delta_rule -> ': None}}
# which implementation does transformers' wrapper resolve to?
import inspect
for name in ['torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule', 'causal_conv1d_fn', 'causal_conv1d_update']:
    w = getattr(mq, name)
    cl = inspect.getclosurevars(w)
    impl = cl.nonlocals.get('implementation')
    R['module_level_impl'][name] = f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__qualname__', '?')}"
R['module_level_impl'] = {k: v for k, v in R['module_level_impl'].items() if v is not None}

dev = 'cuda'
B, T, H, K, V = 2, 300, 16, 128, 128


def l2n(x):
    return x / torch.sqrt((x.float() * x.float()).sum(-1, keepdim=True) + 1e-6).to(x.dtype)


def make(dtype, seed=0):
    g_ = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(B, T, H, K, device=dev, generator=g_)
    k = torch.randn(B, T, H, K, device=dev, generator=g_)
    v = torch.randn(B, T, H, V, device=dev, generator=g_)
    beta = torch.rand(B, T, H, device=dev, generator=g_).sigmoid()
    g = F.logsigmoid(torch.randn(B, T, H, device=dev, generator=g_))  # log decay <= 0
    return {k_: t.to(dtype) if k_ not in ('g',) else t.float() for k_, t in dict(q=q, k=k, v=v, beta=beta, g=g).items()}


def rec64(q, k, v, beta, g):
    """float64 token-by-token gated delta rule (same math as fla naive_recurrent), ground truth."""
    q, k, v, beta, g = [x.transpose(1, 2).double() for x in (q, k, v, beta, g)]
    Bq, Hq, Tq, Kq = k.shape
    S = torch.zeros(Bq, Hq, Kq, v.shape[-1], dtype=torch.float64, device=q.device)
    o = torch.zeros_like(v)
    q = q * Kq ** -0.5
    for t in range(Tq):
        S = S * g[:, :, t].exp()[..., None, None]
        vt = (v[:, :, t] - (S * k[:, :, t, :, None]).sum(-2)) * beta[:, :, t, None]
        S = S + k[:, :, t, :, None] * vt[:, :, None, :]
        o[:, :, t] = torch.einsum('bhk,bhkv->bhv', q[:, :, t], S)
    return o.transpose(1, 2), S


def run_all(inp, prenorm):
    q, k, v, beta, g = inp['q'], inp['k'], inp['v'], inp['beta'], inp['g']
    outs = {}
    if prenorm:
        qn, kn = l2n(q), l2n(k)
        outs['fla_chunk'] = chunk_gated_delta_rule(qn, kn, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False, output_final_state=True)
        outs['fla_fused_rec'] = fused_recurrent_gated_delta_rule(qn, kn, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False, output_final_state=True)
        outs['fla_naive_rec'] = naive_recurrent_gated_delta_rule(q=qn, k=kn, v=v, beta=beta, g=g, output_final_state=True)
        outs['fla_naive_chunk'] = naive_chunk_gated_delta_rule(q=qn, k=kn, v=v, g=g, beta=beta, output_final_state=True)
        outs['hf_torch_chunk'] = hf_chunk(qn, kn, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False, output_final_state=True)
        outs['hf_torch_rec'] = hf_rec(qn, kn, v, g=g, beta=beta, use_qk_l2norm_in_kernel=False, output_final_state=True)
        # fp64 ground truth
        ref = rec64(qn, kn, v, beta, g)
    else:
        outs['fla_chunk'] = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, output_final_state=True)
        outs['fla_fused_rec'] = fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, output_final_state=True)
        outs['hf_torch_chunk'] = hf_chunk(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, output_final_state=True)
        outs['hf_torch_rec'] = hf_rec(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True, output_final_state=True)
        ref = None
    # naive fla uses float32 internally even for double? check: they call .to(torch.float32) -> not a true fp64 ref
    return outs, ref


def cmp(a, b):
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return {'max_abs': d.max().item(), 'mean_abs': d.mean().item(),
            'max_rel': (d / (b.abs() + 1e-6)).max().item(),
            'rel_fro': (d.norm() / b.norm()).item()}


res = {}
for dtype, prenorm in itertools.product([torch.float32, torch.bfloat16], [True, False]):
    key = f"{str(dtype).split('.')[-1]}|{'prenorm_l2' if prenorm else 'inkernel_l2'}"
    inp = make(dtype)
    outs, ref = run_all(inp, prenorm)
    names = list(outs)
    pair = {}
    for a, b in itertools.combinations(names, 2):
        pair[f'{a} vs {b}'] = {'o': cmp(outs[a][0], outs[b][0]), 'final_state': cmp(outs[a][1], outs[b][1])}
    if ref is not None:
        for a in names:
            pair[f'{a} vs fp64_ref'] = {'o': cmp(outs[a][0], ref[0]), 'final_state': cmp(outs[a][1], ref[1])}
    res[key] = {'out_dtypes': {n: str(outs[n][0].dtype) for n in names}, 'pairs': pair}
    print(key)
    for p, s in pair.items():
        print(f"  {p:55s} o.max_abs={s['o']['max_abs']:.3e} o.rel_fro={s['o']['rel_fro']:.3e}"
              + (f" S.max_abs={s['final_state']['max_abs']:.3e}" if 'final_state' in s else ''))
R['shapes'] = dict(B=B, T=T, H=H, K=K, V=V, chunk=64)
R['results'] = res
# the fla naive functions cast to float32 internally; check whether the fp64 "ref" was really fp64
R['note'] = 'fp64_ref = local float64 token-by-token recurrence on the same (bf16/fp32-valued) inputs; fla naive_* cast to fp32 internally'
json.dump(R, open(os.path.join(OUT, 'gdn_kernels.json'), 'w'), indent=1)
print(json.dumps(R['module_level_impl'], indent=1))
