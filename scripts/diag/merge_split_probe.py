#!/usr/bin/env python
"""DEV-0045 诊断：合并核对（scripts/check_training_interfaces.py 的 restricted.across_merge）的病态检查点在本机分家多少；
确定分家的对照必须判不过。只用 CPU、fp32，开发机（Apple M4）约 1 分钟。输出 JSON，可用 --compare 与开发机的输出逐项对照。

背景（RUN-0044）：tests/test_check_training_interfaces.py::test_grid_checkpoint_split_across_merge_fails_restricted 在 HPC（x86_64）上
4 次都失败。当时病态检查点的 across_merge 读数是 max_abs_logit 8.57，但 allowed 集合上的 argmax 0 处不同，于是判据（CPU fp32 要求
逐个相同）判通过。同一条测试在开发机（Apple M4，Accelerate）和 B200（同为 x86_64）上都通过。

构造（tests 与本脚本共用，只在这里定义）：
- split_lora：病态的 LoRA 分解。r = 2，lora_alpha = 2，scaling = 1。每个目标模块取 A = [u; u + eps·n]，B = [v, −v]·scale，其中 u、n、v
  ~ N(0, 1)，由 CPU 生成器依次取。
  - 真实的 ΔW = B·A = −scale·eps·v·n'ᵀ，其中 n' = ((u + eps·n) − u) / eps（fp32 存下来的差），scale·eps = 10。
  - ΔW 并不小：小模型上它是 ‖W‖_F 的约 200 倍（读数 lora_path.dW_over_W）。
  - 病在分解：两条路径都把 ΔW 算成两个约 scale 量级、相对只差 eps 的数之差。fp32 有 24 位有效数字，相减消去约 log2(1/eps) 位。
  - 合并：fp32 的 B·A 每个元素舍入一次。不合并：每次前向 A·x 的两个分量各自舍入，再乘 B 相减；累加顺序因 BLAS 与 CPU 指令集而异。
  - fp64 下两条路径一致到 1e-4 以内（读数 fp64.merged64_vs_unmerged64），所以分家全部来自 fp32 舍入。
  - 下面括号里是开发机的读数（merge_split_probe.dev.json）。
  - SPLIT_RUN0044（eps 1e-6、scale 1e7）：RUN-0044 失败的那条测试的构造，取数顺序与表达式逐字相同（tests 钉住）。
    - 消去约 20 位、剩约 4 位：不合并一侧 A·x 两分量之差的相对误差约 12%，合并一侧 ΔW 的约 2%（读数 lora_path）。
    - 分家只落在尾部：3171 个位置里 31 处 argmax 不同。
    - 不合并一侧换一种求和就是 5、26、15 处，A·x 用 fp64 累加时是 0 处（读数 unmerged_variants）。所以位置数取决于累加顺序。
  - SPLIT（eps 1e-7、scale 1e8）：修过的测试构造。
    - 消去约 23 位、不到 1 位：两条路径的相对误差约 106% 与 20%。
    - 3171 个位置里 2714 处 argmax 不同；换求和是 1712–2604 处。
    - 即使 A·x 用 fp64 累加再舍入到 fp32（每个分量都是正确舍入，最准的 fp32 实现），也有 1157 处。
- conditioned_lora：同一个 ΔW 的良态分解。r = 1，A = [n]，B = [−scale·eps·v]，取数顺序同 split_lora，没有相消。
  - 正确合并时两侧一致到约 1e-5（max_abs_logit），作阴性对照；
  - 说明分家来自分解的病态，而不是 ΔW 大。
- merge_leaving_out(pred)：合并时漏掉一部分模块。做法是把这些模块的 lora_B 置零，再走原来的 merge_lora。这是对实现错误的模拟，
  与舍入无关。
- CONTROL = conditioned_lora 加上漏掉 mlp.down_proj 的合并：
  - 合并后的模型内部自洽，by_thinking 照样通过，只有 across_merge 查得出；
  - 几乎每个位置都不同，差远大于舍入；
  - 核对在任何平台上都必须判不过。
- grid_checkpoint：带 E3b 词表的 Qwen 小检查点（E3b --tiny 的图像口径），网格 token 的行取互不相同的随机值。

情形（cases）：
- split_run0044 / split：读数同核对的 across_merge，样本、批、目标都相同。
  - 合并一侧用 load_trained(merge=True)，在本机上受限解码（generate_restricted，与核对同一组参数），再做教师强制；
  - 不合并一侧用 load_trained(merge=False)，以前者的序列为目标做教师强制；
  - 两侧都用 checkpoint.compare_merge_sides 在 allowed 集合上比较。另报：
    - 每个位置 |ΔL| 的分位数、合并一侧前两名之差（margin）；
    - fp64 参照（真值）；
    - lora_path：A·x 的相消位数与两条路径的相对误差，ΔW 与 W 之比；
    - unmerged_variants：不合并一侧换成别的同样合法的 fp32 求和（A·x 的输入维随机置换 N 次；A·x 用 fp64 累加再舍入到 fp32），
      看 argmax 不同的位置数。
- conditioned：阴性对照。目标借用 split 情形的序列（教师强制对任何目标都成立，省一次解码）。
- control：真跑核对（check_training_interfaces.main --tiny --checks restricted --n CONTROL_N），合并时漏掉 mlp.down_proj，必须判不过
  （退出码 1，失败只在 across_merge）。

运行（只用 CPU）：
  开发机：HF_HUB_OFFLINE=1 python scripts/diag/merge_split_probe.py --out <json> --compare scripts/diag/merge_split_probe.dev.json
  HPC（hpcmgt 上，或节点上经 srun --overlap）：
          HF_HUB_OFFLINE=1 $VI/env/bin/python scripts/diag/merge_split_probe.py --out <json> \\
              --compare scripts/diag/merge_split_probe.dev.json
退出码：
- 0：SPLIT 在本机分家（across_merge 至少 1 处 argmax 不同，即测试的前提成立），且 CONTROL 判不过；
- 1：上面有一项不成立（见输出的 verdict）；
- 2：出错（已有的结果与报错先写进 --out）。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src import runlog  # noqa: E402
from src.adapters import checkpoint as CK  # noqa: E402
from src.adapters import registry  # noqa: E402
from src.adapters.factory import make_adapter  # noqa: E402

FORMAT = "visint-merge-split-probe/1"
CTI_PATH = REPO / "scripts/check_training_interfaces.py"
DEV_REFERENCE = Path(__file__).with_name("merge_split_probe.dev.json")  # 开发机的输出（--compare 拿它逐项对照）
MODEL = "qwen3.5-0.8b"  # 小模型；tests 的网格检查点同一个
SPLIT_EPS, SPLIT_SCALE = 1e-7, 1e8  # 修过的测试构造（tests 钉住）
SPLIT_RUN0044_EPS, SPLIT_RUN0044_SCALE = 1e-6, 1e7  # RUN-0044 失败的那条测试的构造
CONTROL_LEFT_OUT = "mlp.down_proj"  # CONTROL：合并时漏掉的模块（名字后缀）
CONTROL_N = 1  # CONTROL 真跑核对的样本数（确定分家，1 条 = 1057 个位置足够；省时间）
N_PERMUTATIONS = 3  # unmerged_variants 里 A·x 输入维随机置换的次数
LORA_KINDS = ("random", "split", "split_run0044", "conditioned")


# ---------------------------------------------------------------------------
# 构造（tests 共用）
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_cti() -> Any:
    """scripts/check_training_interfaces.py 按文件路径载入（同 tests 的做法；它不在 sys.modules 的包里）。"""
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location("_cti_for_merge_split_probe", CTI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def split_lora(ad: Any, g: torch.Generator, *, eps: float = SPLIT_EPS, scale: float = SPLIT_SCALE) -> None:
    """注入病态分解的 LoRA（规则见模块说明）。取数顺序：逐个目标模块（lora_target_modules() 的顺序）取 u、n、v。
    eps = SPLIT_RUN0044_EPS、scale = SPLIT_RUN0044_SCALE 时与 RUN-0044 的测试构造逐位相同。"""
    ad.inject_lora(r=2, lora_alpha=2, init_lora_weights=False)
    mods = dict(ad.model.named_modules())
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
            u = torch.randn(a_w.shape[1], generator=g)
            a_w.copy_(torch.stack([u, u + eps * torch.randn(a_w.shape[1], generator=g)]))
            v = torch.randn(b_w.shape[0], generator=g)
            b_w.copy_(torch.stack([v, -v], dim=1) * scale)


def conditioned_lora(ad: Any, g: torch.Generator, *, dw_scale: float = SPLIT_EPS * SPLIT_SCALE) -> None:
    """同一个 ΔW = −dw_scale·v·nᵀ 的良态分解（r = 1，lora_alpha = 1：A = [n]，B = [−dw_scale·v]）；取数顺序同 split_lora（u 取了不用），
    所以 ΔW 与 split_lora 的只差 n 与 n' 的舍入。没有相消：合并与不合并一致到舍入（阴性对照）。"""
    ad.inject_lora(r=1, lora_alpha=1, init_lora_weights=False)
    mods = dict(ad.model.named_modules())
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
            torch.randn(a_w.shape[1], generator=g)  # u：与 split_lora 对齐取数顺序
            n = torch.randn(a_w.shape[1], generator=g)
            v = torch.randn(b_w.shape[0], generator=g)
            a_w.copy_(n[None])
            b_w.copy_((-dw_scale) * v[:, None])


def grid_checkpoint(root: str | os.PathLike, *, lora: str, query: bool = False) -> tuple[Path, Any]:
    """带 E3b 词表的 Qwen 小检查点（E3b --tiny 的图像口径）→ (检查点目录, GridTokens)。网格 token 的行取互不相同的随机值，受限解码
    写满 1057 个 token。lora：random = randomize_lora 的随机 LoRA（r 8、lora_alpha 16、核对的 LORA_REL_SCALE、种子 3）；
    split / split_run0044 = split_lora 的两种构造；conditioned = conditioned_lora。query：再注入 4 个查询 token（grid2d）与查询嵌入。"""
    from src.adapters.image_policy import parse_policy
    from src.e3.config import load_config
    from src.e3.tokens import add_vocab

    if lora not in LORA_KINDS:
        raise ValueError(f"lora 必须是 {LORA_KINDS} 之一，收到 {lora!r}")
    cti = load_cti()
    cfg = load_config()
    pol = parse_policy(cfg.doc["tiny"]["image_policies"][registry.get_model(MODEL).family])
    ad = make_adapter(MODEL, device="cpu", dtype=torch.float32, image_policy=pol, tiny=True, init_std=cti.TINY_INIT_STD)
    tok = add_vocab(ad, cfg.vocab)
    if query:
        ad.inject_query_tokens(4, "grid2d")
    g = torch.Generator().manual_seed(0)
    if lora == "random":
        ad.inject_lora(r=8, lora_alpha=16, init_lora_weights=False)
        cti.randomize_lora(ad, rel_scale=cti.LORA_REL_SCALE, seed=3)
    elif lora == "split":
        split_lora(ad, g)
    elif lora == "split_run0044":
        split_lora(ad, g, eps=SPLIT_RUN0044_EPS, scale=SPLIT_RUN0044_SCALE)
    else:
        conditioned_lora(ad, g)
    ids = torch.tensor(sorted(tok.grid_ids))
    with torch.no_grad():
        for w in ad.trainable_token_rows(list(cfg.vocab.names)):
            w[ids] = torch.randn(len(ids), w.shape[1], generator=g)
    qe = torch.randn(4, ad.hidden_size(), generator=g) * 0.05 if query else None
    return ad.save_trained(root, query_embeds=qe), tok


def is_control_left_out(name: str) -> bool:
    return name.endswith(CONTROL_LEFT_OUT)


@contextlib.contextmanager
def merge_leaving_out(pred: Callable[[str], bool]) -> Iterator[list[str]]:
    """模拟"合并漏模块"的实现错误：期间 CheckpointSupport.merge_lora 先把 pred(模块名) 为真的目标模块的 lora_B 置零（它们的增量不进
    合并后的权重），再走原来的合并。产出被漏掉的模块名（每次合并追加一遍）。退出时恢复原来的 merge_lora。"""
    orig = CK.CheckpointSupport.merge_lora
    left_out: list[str] = []

    def merge(self: Any) -> None:
        mods = dict(self.model.named_modules())
        with torch.no_grad():
            for name in self.lora_target_modules():
                if pred(name):
                    mods[name].lora_B["default"].weight.zero_()
                    left_out.append(name)
        orig(self)

    CK.CheckpointSupport.merge_lora = merge
    try:
        yield left_out
    finally:
        CK.CheckpointSupport.merge_lora = orig


# ---------------------------------------------------------------------------
# 指纹与平台
# ---------------------------------------------------------------------------


def _tensor_sha(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def checkpoint_fingerprint(ck: Path) -> dict[str, Any]:
    """检查点内容的指纹（与文件的写法无关：adapter_config.json 里 target_modules 的顺序随进程变，manifest 的 sha256 不能跨机器比）。"""
    from safetensors.torch import load_file

    man = CK.read_manifest(ck)
    return {"base_fingerprint": man["base_fingerprint"],
            "lora_sha256": _tensor_sha(load_file(str(ck / "lora/adapter_model.safetensors"))),
            "token_rows_sha256": _tensor_sha(load_file(str(ck / "token_rows.safetensors")))}


def runs_sha256(runs: Sequence[tuple[Sequence[Any], bool, Sequence[str]]]) -> str:
    """输入（每批的 ModelInputs 张量、思考开关、样本 id）的指纹：两台机器的输入逐位相同时，差别只来自计算的舍入。"""
    h = hashlib.sha256()
    for inputs, thinking, ids in runs:
        h.update(json.dumps([bool(thinking), list(ids)]).encode())
        for inp in inputs:
            h.update(_tensor_sha({k: v for k, v in inp.tensors.items() if torch.is_tensor(v)}).encode())
    return h.hexdigest()


def _cpu_model() -> str | None:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return None


def platform_block() -> dict[str, Any]:
    cfg = torch.__config__.show()
    blas = next((p.split("=", 1)[1] for p in cfg.replace(",", " ").split() if p.startswith("BLAS_INFO=")), None)
    return {"machine": platform.machine(), "system": platform.system(), "platform": platform.platform(),
            "cpu_model": _cpu_model(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(), "torch_blas": blas,
            "mkl": torch.backends.mkl.is_available(), "mkldnn": torch.backends.mkldnn.is_available(),
            "torch_threads": torch.get_num_threads(), "fla_installed": importlib.util.find_spec("fla") is not None,
            "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None}


# ---------------------------------------------------------------------------
# 两侧与比较（与核对的 across_merge 同一组函数、同样的样本、批与目标）
# ---------------------------------------------------------------------------


class Setup:
    """核对的输入：命令行（--tiny --checkpoint ... --checks restricted）、样本、批、网格设置。"""

    def __init__(self, ck: Path, n: int | None = None) -> None:
        from src.e1.prompts import load_prompt_config
        from src.e1.runner import batch_plan

        cti = self.cti = load_cti()
        argv = ["--tiny", "--model", MODEL, "--checkpoint", str(ck), "--checks", "restricted", "--out", "unused.json"]
        self.a = cti.parse_args(argv + ([] if n is None else ["--n", str(n)]))
        prompts = load_prompt_config(self.a.prompts)
        bs, _ = cti.resolve_batch_size(self.a, prompts)
        self.samples, _, self.data = cti.load_dev(cti.data_paths(self.a), prompts, registry.get_model(MODEL).family, self.a.n)
        self.ids = [s.id for s in self.samples]
        self.plan = batch_plan([None] * len(self.samples), bs)
        self.g: Any = None

    def runs(self, ad: Any) -> list[tuple[list[Any], bool, list[str]]]:
        return self.cti.make_grid_runs(ad, self.g, self.samples, self.plan, self.ids)


def merged_side(s: Setup) -> tuple[Any, CK.MergeSide, str]:
    """合并一侧：load_trained(merge=True) 上受限解码（与核对的 restricted_consistency 同一组参数），再以解码序列为目标做教师强制。
    返回 (适配器, MergeSide, 输入指纹)。"""
    ad, _ = s.cti.load_checkpoint(s.a, merge=True)
    s.g = s.cti.grid_setup(s.a, ad)
    runs = s.runs(ad)
    targets = []
    for inputs, thinking, _ in runs:
        res = ad.generate_restricted(inputs, s.g.allowed, max_new_tokens=s.g.max_new_tokens, stop_ids=s.g.stop_ids,
                                     enable_thinking=thinking)
        targets.append([list(r.token_ids) for r in res])
    side = CK.merge_side(ad, s.cti._side_runs(runs), max_new_tokens=s.g.max_new_tokens, targets=targets, generate=False)
    return ad, side, runs_sha256(runs)


def _lora_layers(ad: Any) -> list[tuple[str, Any]]:
    return [(n, m) for n, m in ad.model.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})]


def other_side(s: Setup, targets: Sequence[Any], *, merge: bool = False, fp64: bool = False,
               hook: Callable[[Any], Any] | None = None) -> tuple[Any, CK.MergeSide]:
    """另一侧：load_trained(merge=False)（fp64 时整模型升到 fp64；merge 时随后合并），可选地先挂 hook，再以 targets 做教师强制。"""
    ad, _ = s.cti.load_checkpoint(s.a, merge=merge and not fp64)
    if fp64:  # 参照：整模型 fp64（诊断专用；适配器的 dtype 一并改，否则它的 dtype 核对会报错）
        ad.model.double()
        ad._dtype = torch.float64
        if merge:
            ad.merge_lora()
    if hook is not None:
        hook(ad)
    side = CK.merge_side(ad, s.cti._side_runs(s.runs(ad)), max_new_tokens=s.g.max_new_tokens, targets=targets, generate=False)
    return ad, side


def compare(ad_ref: Any, s: Setup, a: CK.MergeSide, b: CK.MergeSide) -> dict[str, Any]:
    """checkpoint.compare_merge_sides（allowed 集合）按 run 汇总：与核对 across_merge 的同名读数相同。"""
    reps = CK.compare_merge_sides(ad_ref, a, b, allowed_ids=s.g.allowed, chunk_size=s.cti.TF_CHUNK)
    tfs = [r["teacher_forcing"] for r in reps]
    n = sum(t["n_tokens"] for t in tfs)
    dis = sum(len(t["disagreements"]) for t in tfs)
    return {"n_tokens": n, "n_disagreements": dis, "argmax_agreement": 1.0 - dis / n if n else 1.0,
            "max_abs_logit": max(t["max_abs_logit"] for t in tfs), "max_abs_hidden": max(t["max_abs_hidden"] for t in tfs)}


def _q(x: torch.Tensor, qs: Sequence[float]) -> dict[str, float]:
    v = x.double().flatten().numpy()
    return {f"p{int(round(q * 100)):02d}" if 0 < q < 1 else ("min" if q == 0 else "max"): float(np.quantile(v, q)) for q in qs}


def logit_stats(ad_ref: Any, s: Setup, a: CK.MergeSide, b: CK.MergeSide) -> dict[str, Any]:
    """每个位置两侧 allowed logits 之差的最大绝对值的分位数，与 a（合并一侧）前两名之差的分位数。"""
    logits_fn, _, _ = CK._logits_and_columns(ad_ref, s.g.allowed)
    with torch.no_grad():
        la = torch.cat([logits_fn(h) for h in a.hidden])
        lb = torch.cat([logits_fn(h) for h in b.hidden])
    top = la.topk(2, dim=-1).values
    return {"abs_diff_per_position": _q((la - lb).abs().amax(-1), (0.5, 0.9, 0.99, 1.0)),
            "merged_margin": _q(top[:, 0] - top[:, 1], (0.0, 0.01, 0.5))}


def lora_path_stats(s: Setup, targets: Sequence[Any]) -> dict[str, Any]:
    """不合并一侧各 LoRA 模块的前向（钩子记下 lora_A 的输入与输出、lora_B 的输出），在 fp64 里与真值比较：
    - Ax_rel_diff：|(A·x)₀ − (A·x)₁| / |(A·x)₀| 的中位数（相消的程度；log2 的倒数 = 消去的位数，记在 bits_cancelled）；
    - Ax_diff_rel_err：fp32 算出的 (A·x)₀ − (A·x)₁ 与真值之差 / 真值（中位数之比）；
    - lora_out_rel_err：lora_B 的 fp32 输出与真值 ΔW·x 之差 / 真值；
    - merged_dW_rel_err：合并路径 fp32 的 B·A·scaling 与 fp64 真值之差 / 真值（逐元素，中位数之比）；
    - dW_over_W：‖ΔW‖_F / ‖W‖_F。
    各模块的中位数再取所有模块的中位数与最大值。"""
    rec: dict[str, dict[str, list[torch.Tensor]]] = {}

    def recorder(r: dict[str, list[torch.Tensor]], keys: tuple[str, ...]) -> Callable[..., None]:
        def fn(mod: Any, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:  # 只记录，不改输出
            for k, t in zip(keys, (inp[0], out) if len(keys) == 2 else (out,)):
                r[k].append(t.detach())
        return fn

    def hook(ad: Any) -> None:
        for name, m in _lora_layers(ad):
            r = rec.setdefault(name, {"x": [], "a": [], "y": []})
            m.lora_A["default"].register_forward_hook(recorder(r, ("x", "a")))
            m.lora_B["default"].register_forward_hook(recorder(r, ("y",)))

    ad, _ = other_side(s, targets, hook=hook)
    per: dict[str, list[float]] = {k: [] for k in ("Ax_rel_diff", "Ax_diff_rel_err", "lora_out_rel_err", "merged_dW_rel_err",
                                                   "dW_over_W")}
    with torch.no_grad():
        for name, m in _lora_layers(ad):
            A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
            sc = float(m.scaling["default"])
            dw64 = (B.double() @ A.double()) * sc
            dw32 = (B @ A) * sc  # 与 peft 的 get_delta_weight 同一个算法（fp32）
            W = m.base_layer.weight.double()
            per["dW_over_W"].append(float(dw64.norm() / W.norm()))
            per["merged_dW_rel_err"].append(float((dw32.double() - dw64).abs().median() / dw64.abs().median()))
            r = rec[name]
            x = torch.cat([t.reshape(-1, t.shape[-1]) for t in r["x"]]).double()
            a32 = torch.cat([t.reshape(-1, t.shape[-1]) for t in r["a"]]).double()
            y32 = torch.cat([t.reshape(-1, t.shape[-1]) for t in r["y"]]).double()
            a64 = x @ A.double().T
            if a64.shape[1] >= 2:
                d64, d32 = a64[:, 0] - a64[:, 1], a32[:, 0] - a32[:, 1]
                per["Ax_rel_diff"].append(float(d64.abs().median() / a64[:, 0].abs().median()))
                per["Ax_diff_rel_err"].append(float((d32 - d64).abs().median() / d64.abs().median()))
            y64 = a64 @ B.double().T
            per["lora_out_rel_err"].append(float((y32 - y64).abs().median() / y64.abs().median()))
    out = {k: ({"median": float(np.median(v)), "max": float(np.max(v))} if v else None) for k, v in per.items()}
    if out["Ax_rel_diff"]:
        out["bits_cancelled"] = {"median": -math.log2(out["Ax_rel_diff"]["median"]), "fp32_significand_bits": 24}
    out["n_modules"] = len(per["dW_over_W"])
    return out


def _hook_lora_A(ad: Any, fn: Callable[[Any, torch.Tensor, Any], torch.Tensor]) -> None:
    for i, (_, m) in enumerate(_lora_layers(ad)):
        m.lora_A["default"].register_forward_hook(lambda mod, inp, out, i=i: fn(mod, inp[0], i))


def permuted_A(seed: int) -> Callable[[Any], None]:
    """不合并一侧 A·x 的输入维按固定种子随机置换后再乘（数学上相同，fp32 的累加顺序不同：另一种同样合法的 BLAS 实现）。"""
    def hook(ad: Any) -> None:
        g = torch.Generator().manual_seed(seed)
        perms = [torch.randperm(m.lora_A["default"].weight.shape[1], generator=g) for _, m in _lora_layers(ad)]
        _hook_lora_A(ad, lambda mod, x, i: F.linear(x[..., perms[i]], mod.weight[:, perms[i]]))
    return hook


def fp64_accumulated_A(ad: Any) -> None:
    """不合并一侧 A·x 用 fp64 累加、再舍入到 fp32（每个分量是正确舍入的结果：最准的 fp32 实现）。"""
    _hook_lora_A(ad, lambda mod, x, i: (x.double() @ mod.weight.double().T).to(x.dtype))


# ---------------------------------------------------------------------------
# 情形
# ---------------------------------------------------------------------------


def split_case(root: Path, lora: str, n_perm: int, eps: float, scale: float) -> tuple[dict[str, Any], Setup, CK.MergeSide]:
    t0 = time.perf_counter()
    ck, _ = grid_checkpoint(root / lora, lora=lora)
    s = Setup(ck)
    ad_m, merged, in_sha = merged_side(s)
    targets = merged.targets
    t_decode = time.perf_counter() - t0
    ad0, unmerged = other_side(s, targets)
    rep: dict[str, Any] = {
        "construction": {"lora": lora, "eps": eps, "scale": scale, "r": 2, "lora_alpha": 2, **checkpoint_fingerprint(ck)},
        "inputs_sha256": in_sha,
        "targets": {"sha256": hashlib.sha256(json.dumps(targets).encode()).hexdigest(),
                    "lengths": [len(r) for t in targets for r in t]},
        "across_merge": compare(ad0, s, merged, unmerged),
    }
    rep.update(logit_stats(ad0, s, merged, unmerged))
    _, truth = other_side(s, targets, merge=True, fp64=True)
    _, unmerged64 = other_side(s, targets, fp64=True)
    rep["fp64"] = {"merged64_vs_unmerged64": compare(ad0, s, truth, unmerged64),
                   "merged32_vs_truth": compare(ad0, s, merged, truth),
                   "unmerged32_vs_truth": compare(ad0, s, unmerged, truth)}
    rep["lora_path"] = lora_path_stats(s, targets)
    var = {f"perm{k}": compare(ad0, s, merged, other_side(s, targets, hook=permuted_A(k))[1]) for k in range(n_perm)}
    var["fp64_accumulated_A"] = compare(ad0, s, merged, other_side(s, targets, hook=fp64_accumulated_A)[1])
    rep["unmerged_variants"] = {k: {"n_disagreements": v["n_disagreements"], "max_abs_logit": v["max_abs_logit"]}
                                for k, v in var.items()}
    rep["timing_s"] = {"decode": t_decode, "total": time.perf_counter() - t0}
    return rep, s, merged


def conditioned_case(root: Path, borrow: Setup, targets: Sequence[Any]) -> dict[str, Any]:
    """阴性对照：conditioned_lora，正确合并；目标借用 split 情形的序列（同样的样本与批）。"""
    t0 = time.perf_counter()
    ck, _ = grid_checkpoint(root / "conditioned", lora="conditioned")
    s = Setup(ck)
    s.g = borrow.g
    ad_m, merged = other_side(s, targets, merge=True)
    ad0, unmerged = other_side(s, targets)
    rep = {"construction": {"lora": "conditioned", "dw_scale": SPLIT_EPS * SPLIT_SCALE, "r": 1, "lora_alpha": 1,
                            **checkpoint_fingerprint(ck)},
           "targets": "split 情形的受限解码序列", "across_merge": compare(ad0, s, merged, unmerged)}
    rep.update(logit_stats(ad0, s, merged, unmerged))
    rep["timing_s"] = {"total": time.perf_counter() - t0}
    return rep


def control_case(root: Path) -> dict[str, Any]:
    """确定分家：conditioned_lora + 合并漏掉 CONTROL_LEFT_OUT，真跑核对（restricted，CONTROL_N 条样本）。"""
    t0 = time.perf_counter()
    cti = load_cti()
    ck, _ = grid_checkpoint(root / "control", lora="conditioned")
    out = root / "control_check.json"
    with merge_leaving_out(is_control_left_out) as left:
        code = cti.main(["--tiny", "--model", MODEL, "--checkpoint", str(ck), "--checks", "restricted", "--n", str(CONTROL_N),
                         "--out", str(out)])
    doc = json.loads(out.read_text())
    r = doc["checks"].get("restricted") or {}
    bt = (r.get("by_thinking") or {}).get("thinking_off") or {}
    am = (((r.get("across_merge") or {}).get("by_thinking") or {}).get("thinking_off") or {}).get("teacher_forcing") or {}
    fails = doc.get("failures", {}).get("restricted", [])
    return {"construction": {"lora": "conditioned", "merge": f"漏掉 *{CONTROL_LEFT_OUT}", "n_left_out": len(set(left)),
                             **checkpoint_fingerprint(ck)},
            "check": {"exit_code": code, "pass": doc.get("pass"), "error": doc.get("error"), "failures": fails,
                      "failures_only_across_merge": bool(fails) and all(m.startswith("restricted/across_merge/") for m in fails),
                      "by_thinking": {"teacher_forcing_agreement": (bt.get("teacher_forcing") or {}).get("agreement"),
                                      "decode_exact_rows": (bt.get("decode_vs_generate_batch") or {}).get("exact_rows")},
                      "across_merge": {k: am.get(k) for k in ("n_tokens", "n_disagreements", "argmax_agreement", "max_abs_logit",
                                                              "max_abs_hidden")},
                      "across_merge_gap": am.get("gap")},
            "timing_s": {"total": time.perf_counter() - t0}}


def verdict(cases: dict[str, Any]) -> dict[str, Any]:
    sp = cases["split"]["across_merge"]
    c = cases["control"]["check"]
    bt = c["by_thinking"]
    out = {"split_splits_here": sp["n_disagreements"] > 0,
           "control_refused": (c["exit_code"] == 1 and c["pass"] is False and c["failures_only_across_merge"]
                               and bt["teacher_forcing_agreement"] == 1.0 and bt["decode_exact_rows"] == 1.0)}
    out["ok"] = out["split_splits_here"] and out["control_refused"]
    return out


# ---------------------------------------------------------------------------
# 输出与对照
# ---------------------------------------------------------------------------

SUMMARY_KEYS = [("across_merge", "n_tokens"), ("across_merge", "n_disagreements"), ("across_merge", "max_abs_logit"),
                ("across_merge", "max_abs_hidden"), ("abs_diff_per_position", "p50"), ("abs_diff_per_position", "p99"),
                ("merged_margin", "min"), ("fp64", "merged64_vs_unmerged64", "max_abs_logit"),
                ("fp64", "merged32_vs_truth", "n_disagreements"), ("fp64", "unmerged32_vs_truth", "n_disagreements"),
                ("lora_path", "bits_cancelled", "median"), ("lora_path", "Ax_diff_rel_err", "median"),
                ("lora_path", "merged_dW_rel_err", "median"), ("lora_path", "dW_over_W", "median")]
IDENTITY_KEYS = [("construction", "base_fingerprint"), ("construction", "lora_sha256"), ("construction", "token_rows_sha256"),
                 ("inputs_sha256",), ("targets", "sha256")]
CONTROL_KEYS = [("check", "exit_code"), ("check", "pass"), ("check", "failures_only_across_merge"),
                ("check", "by_thinking", "teacher_forcing_agreement"), ("check", "across_merge", "n_tokens"),
                ("check", "across_merge", "n_disagreements"), ("check", "across_merge_gap", "median"),
                ("check", "across_merge", "max_abs_logit")]


def _get(d: Any, path: Sequence[str]) -> Any:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, str) and len(v) == 64:
        return v[:12]
    return str(v)


def summary_lines(doc: dict[str, Any], ref: dict[str, Any] | None) -> list[str]:
    cols = ["本机"] + (["对照"] if ref else [])
    lines = [f"平台：{doc['platform']['machine']} {doc['platform']['cpu_model']}，torch {doc['platform']['torch']}"
             f"（{doc['platform']['torch_blas']}，{doc['platform']['torch_cpu_capability']}，{doc['platform']['torch_threads']} 线程）"]
    if ref:
        lines.append(f"对照：{ref['platform']['machine']} {ref['platform']['cpu_model']}，torch {ref['platform']['torch']}"
                     f"（{ref['platform']['torch_blas']}，{ref['platform']['torch_cpu_capability']}）")
    lines.append("读数 | " + " | ".join(cols))
    for case, keys in (("split_run0044", IDENTITY_KEYS + SUMMARY_KEYS), ("split", IDENTITY_KEYS + SUMMARY_KEYS),
                       ("conditioned", IDENTITY_KEYS[:3] + SUMMARY_KEYS[:7]), ("control", CONTROL_KEYS)):
        c = doc["cases"].get(case)
        if c is None:
            continue
        for path in keys:
            row = [_fmt(_get(c, path))] + ([_fmt(_get(ref["cases"].get(case), path))] if ref else [])
            same = "" if not ref else ("  =" if row[0] == row[1] else "  ≠")
            lines.append(f"{case}.{'.'.join(path)} | " + " | ".join(row) + same)
        if case.startswith("split"):
            for k, v in c["unmerged_variants"].items():
                rv = _get(ref, ("cases", case, "unmerged_variants", k)) if ref else None
                lines.append(f"{case}.unmerged_variants.{k} | {v['n_disagreements']} ({_fmt(v['max_abs_logit'])})"
                             + (f" | {rv['n_disagreements']} ({_fmt(rv['max_abs_logit'])})" if rv else ""))
    lines.append(f"verdict：{json.dumps(doc['verdict'], ensure_ascii=False)}")
    return lines


def write_json(path: Path, doc: dict[str, Any]) -> None:
    doc["written_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="输出 JSON 的路径")
    ap.add_argument("--compare", help=f"对照的输出 JSON（例如开发机的 {DEV_REFERENCE.relative_to(REPO)}）：逐项并排打印")
    ap.add_argument("--threads", type=int, help="torch.set_num_threads（默认不改）")
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS, help=f"A·x 输入维置换的次数（默认 {N_PERMUTATIONS}）")
    a = ap.parse_args(argv)
    if a.threads is not None:
        torch.set_num_threads(a.threads)
    ref = json.loads(Path(a.compare).read_text()) if a.compare else None
    out = Path(a.out)
    doc: dict[str, Any] = {"format": FORMAT, "argv": list(sys.argv[1:] if argv is None else argv), "written_utc": None,
                           "code": runlog.git_info(REPO), "versions": runlog.package_versions(), "platform": platform_block(),
                           "constants": {"SPLIT_EPS": SPLIT_EPS, "SPLIT_SCALE": SPLIT_SCALE,
                                         "SPLIT_RUN0044_EPS": SPLIT_RUN0044_EPS, "SPLIT_RUN0044_SCALE": SPLIT_RUN0044_SCALE,
                                         "CONTROL_LEFT_OUT": CONTROL_LEFT_OUT, "CONTROL_N": CONTROL_N,
                                         "permutations": a.permutations},
                           "cases": {}, "verdict": None, "error": None}
    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="merge_split_probe_") as tmp:
            root = Path(tmp)
            doc["cases"]["split_run0044"], _, _ = split_case(root, "split_run0044", a.permutations, SPLIT_RUN0044_EPS,
                                                             SPLIT_RUN0044_SCALE)
            doc["cases"]["split"], s, merged = split_case(root, "split", a.permutations, SPLIT_EPS, SPLIT_SCALE)
            doc["data"] = {k: s.data[k] for k in ("n", "sample_ids", "sample_ids_sha256")}
            doc["batches"] = [list(p) for p in s.plan]
            doc["cases"]["conditioned"] = conditioned_case(root, s, merged.targets)
            doc["cases"]["control"] = control_case(root)
        doc["verdict"] = verdict(doc["cases"])
    except Exception as e:  # noqa: BLE001 — 证据先落盘
        doc["error"] = {"repr": repr(e), "traceback": traceback.format_exc()[-6000:]}
        doc["timing_s"] = time.perf_counter() - t0
        write_json(out, doc)
        print(f"失败：{e!r}（已写出 {out}）", file=sys.stderr)
        return 2
    doc["timing_s"] = time.perf_counter() - t0
    write_json(out, doc)
    print("\n".join(summary_lines(doc, ref)))
    print(f"用时 {doc['timing_s']:.1f} s -> {out}")
    return 0 if doc["verdict"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
