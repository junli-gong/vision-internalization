#!/usr/bin/env python
"""LoRA 缩放的派生检查点（DEV-0041 §3；D83：负责人要 0.25、0.5、0.75 三档的折中曲线，每档量 T 文本、网格指代分割、结构强制下的
网格深度）。

    python scripts/scale_lora_ckpt.py --checkpoint <训练叶目录>/ckpt/final/model --scale 0.5 --out <新目录>/ckpt/final/model

做法：
- 新检查点 = 原检查点逐文件拷贝，只有 lora/adapter_model.safetensors 换成"lora_B 乘 s"的版本（fp32 里乘、再存回原 dtype）。合并后的增量
  ΔW = s · (B·A) · (lora_alpha / r)，即 LoRA 的增量整体乘 s；lora_A、网格 token 行（token_rows.safetensors）、token 表、LoRA 配置、
  查询嵌入与 extra 原样保留。s = 1 时张量逐位不变；s = 0 时 B = 0，合并后与 base 相同（网格 token 行仍是训练后的值）。
- 只接受 lora_A / lora_B 两类张量（DoRA 的 magnitude、LoRA 化的嵌入、bias 训练都拒绝：它们不随 B 线性缩放）；原检查点必须有 LoRA，
  而且本身不是派生的（不叠加缩放）。
- manifest 按 src/adapters/checkpoint 的规则重写：逐文件 sha256 与字节数（file_sha256、_listed_files），同样的 json 写法、先写 .tmp 再
  替换；另加 derived = {from: 原 manifest.json 的 sha256, lora_scale: s, rule, n_scaled_tensors, lora_dtypes}。写完用 read_manifest
  核对一遍。scripts/eval_e3b.py 把 derived 写进运行身份的 checkpoint；run_e1 --checkpoint 的运行身份记 manifest 的 sha256（与原检查点
  不同）。两者的合并核对照常做。
- 叶目录的信息：--checkpoint 与 --out 都要是 <叶>/ckpt/final/model 的布局；<原叶>/config.yaml 拷到 <新叶>/config.yaml（必须有），
  train_summary.json 有就一并拷（布局同 scripts/server/train_queue.sh 的 finalize），需要叶目录信息的读取方（E5 的 _leaf_config）照样能用。
  另写 <新叶>/derived.json：来源的 manifest sha256、缩放、生成时间（路径与机器有关，只在这里记）。

退出码：0 = 写好；2 = 出错（用法、目录不合规、张量不合规；不留半成品：写在 <out>.part 里，成功后才改名）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.adapters import checkpoint as C  # noqa: E402

LEAF_LAYOUT = ("ckpt", "final", "model")
LEAF_FILES_REQUIRED = ("config.yaml",)
LEAF_FILES_OPTIONAL = ("train_summary.json",)
SCALE_RULE = ("lora_B × s（fp32 里乘、存回原 dtype）：合并后的增量 ΔW = s·(B·A)·(lora_alpha / r)；lora_A、网格 token 行与其余文件原样保留"
              "（DEV-0041 §3）")


class ScaleError(RuntimeError):
    """原检查点或输出目录不合规。"""


def leaf_of(model_dir: Path, what: str) -> Path:
    """<叶>/ckpt/final/model → <叶>；布局不对就报错。"""
    parts = model_dir.parts[-3:]
    if tuple(parts) != LEAF_LAYOUT:
        raise ScaleError(f"{what} 必须是 <叶目录>/{'/'.join(LEAF_LAYOUT)} 的布局，收到 {model_dir}")
    return model_dir.parents[2]


def check_lora(src: Path, man: dict) -> list[str]:
    """原检查点的 LoRA 必须能按 B 线性缩放；返回 lora_B 张量的键。"""
    if man.get("lora") is None or not (src / C.LORA_WEIGHTS).is_file():
        raise ScaleError(f"{src} 没有 LoRA（manifest.lora = {man.get('lora')}），没有可缩放的增量")
    if man.get("derived") is not None:
        raise ScaleError(f"{src} 本身是派生的检查点（{man['derived']}），不叠加缩放：请从原检查点派生")
    cfg = json.loads((src / C.LORA_CONFIG).read_text())
    if cfg.get("use_dora"):
        raise ScaleError("DoRA 的 magnitude 不随 B 线性缩放，拒绝")
    if str(cfg.get("bias", "none")) != "none":
        raise ScaleError(f"LoRA 训练了 bias（{cfg.get('bias')}），不随 B 线性缩放，拒绝")
    keys = sorted(C._load_tensors(src / C.LORA_WEIGHTS))
    other = [k for k in keys if ".lora_A." not in k and ".lora_B." not in k]
    if other:
        raise ScaleError(f"LoRA 权重里有 lora_A / lora_B 以外的张量（{other[:3]}），不随 B 线性缩放，拒绝")
    b = [k for k in keys if ".lora_B." in k]
    if not b or len(b) * 2 != len(keys):
        raise ScaleError(f"lora_A 与 lora_B 的张量数对不上（共 {len(keys)}，B {len(b)}）")
    return b


def scale_lora(src: Path, out: Path, scale: float) -> dict:
    """写派生检查点（规则见模块说明），返回新 manifest。"""
    if not math.isfinite(scale) or scale < 0:
        raise ScaleError(f"--scale 必须是有限的非负数，收到 {scale}")
    src, out = Path(src), Path(out)
    src_leaf, out_leaf = leaf_of(src, "--checkpoint"), leaf_of(out, "--out")
    man = C.read_manifest(src)  # 逐文件核对 sha256
    b_keys = check_lora(src, man)
    for f in LEAF_FILES_REQUIRED:
        if not (src_leaf / f).is_file():
            raise ScaleError(f"原叶目录 {src_leaf} 没有 {f}（布局同 train_queue.sh 的 finalize）")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ScaleError(f"{out} 已存在且不是空目录")
    for f in LEAF_FILES_REQUIRED + LEAF_FILES_OPTIONAL + ("derived.json",):
        if (out_leaf / f).exists():
            raise ScaleError(f"{out_leaf / f} 已存在，不覆盖")
    part = out.with_name(out.name + ".part")
    if part.exists():
        shutil.rmtree(part)
    shutil.copytree(src, part)
    (part / C.MANIFEST).unlink()
    state = C._load_tensors(src / C.LORA_WEIGHTS)
    dtypes = sorted({str(v.dtype).removeprefix("torch.") for v in state.values()})
    new = {k: ((v.to(torch.float32) * float(scale)).to(v.dtype) if k in set(b_keys) else v) for k, v in state.items()}
    (part / C.LORA_WEIGHTS).unlink()
    C._save_tensors(new, part / C.LORA_WEIGHTS)
    src_sha = C.file_sha256(src / C.MANIFEST)
    manifest = dict(man)
    manifest["files"] = {rel: {"sha256": C.file_sha256(part / rel), "bytes": (part / rel).stat().st_size}
                         for rel in sorted(C._listed_files(part) - {C.MANIFEST})}
    manifest["derived"] = {"from": src_sha, "lora_scale": float(scale), "rule": SCALE_RULE, "n_scaled_tensors": len(b_keys),
                           "lora_dtypes": dtypes}
    tmp = part / (C.MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True))
    tmp.replace(part / C.MANIFEST)
    C.read_manifest(part)  # 与 load_trained 同一个核对
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.rmdir()
    part.replace(out)
    for f in LEAF_FILES_REQUIRED + LEAF_FILES_OPTIONAL:
        if (src_leaf / f).is_file():
            shutil.copy2(src_leaf / f, out_leaf / f)
    (out_leaf / "derived.json").write_text(json.dumps(
        {"from_manifest_sha256": src_sha, "from_checkpoint": str(src), "lora_scale": float(scale), "rule": SCALE_RULE,
         "manifest_sha256": C.file_sha256(out / C.MANIFEST), "leaf_files_copied": [f for f in LEAF_FILES_REQUIRED + LEAF_FILES_OPTIONAL
                                                                                     if (src_leaf / f).is_file()],
         "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, ensure_ascii=False, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="原检查点：<训练叶目录>/ckpt/final/model")
    p.add_argument("--scale", type=float, required=True, help="LoRA 增量的倍数 s（≥ 0；如 0.25、0.5、0.75）")
    p.add_argument("--out", required=True, help="新检查点：<新叶目录>/ckpt/final/model（不存在或为空）")
    a = p.parse_args(argv)
    try:
        man = scale_lora(Path(a.checkpoint), Path(a.out), a.scale)
    except Exception as e:  # noqa: BLE001 — 用法与目录问题都以 2 退出
        print(f"失败：{e!r}", file=sys.stderr)
        return 2
    print(f"派生检查点 {a.out}：lora_scale {a.scale}，from {man['derived']['from'][:16]}…，"
          f"manifest sha256 {C.file_sha256(Path(a.out) / C.MANIFEST)[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
