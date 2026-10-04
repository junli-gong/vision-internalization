"""评测前的合并核对（简报 §12"评测前必须合并，并在开发子集上验证合并前后的输出一致"；DEV-0030 §1、§2）。

就是 scripts/check_training_interfaces.py --checkpoint 的同一套函数（按文件路径载入那个脚本，调它的 main）：E1 grid16 开发子集按 sha256
取 n 条，两种思考开关，factory.load_trained(merge=False) 对比 merge=True，判据 v1（D50-3；CPU fp32 时要求逐个相同）。
- 设置（n、检查项）读 configs/e3b_train.json 的 eval.merge_check（--tiny 时 n 取 eval.tiny.merge_check_n）。n 两种用途共用；检查项按
  用途（PURPOSES）分开：grid_eval = E3b 的网格评测（scripts/eval_e3b.py，经 evaluate.resolve_eval），merge 加 restricted（带网格 token
  的检查点用它自己的网格 token 与 E3b 单轮评测的设置，见那个脚本的模块说明；DEV-0032）；t_text = 训练后模型的 T 文本评测
  （scripts/run_e1.py --checkpoint，不传 purpose，取默认），只有 merge（T 文本评测用 base 的文本 token）。理由与用时写在配置的
  eval.merge_check.checks_basis。
- 输出写进调用方给的 JSON（那个脚本的格式）。同一个检查点（manifest 的 sha256）、同一组设置（n、检查项、判据版本、tiny、dtype）已经
  通过的 JSON 直接复用，不重跑；否则重跑并覆盖。
- 不通过（退出码 1）或出错（退出码 2）都抛 MergeCheckFailed：调用方据此拒绝评测。
- E3b 配置（DEV-0035，D67-4(a)）：调用方给的 e3b_config（scripts/eval_e3b.py 的 --config）原样透传给那个脚本的 --e3b-config，带网格
  token 的检查点做 restricted 时读的就是评测用的同一份配置（词表、allowed_scope、网格指令、思考开关）；不给时那个脚本取默认的
  configs/e3b_train.json。复用已通过的 JSON 时另核对：JSON 里有 restricted_inputs 的，它记下的配置 sha256 必须就是这一份。
- 没有 LoRA 的检查点（DEV-0035：C1′，冻结 LLM、只训练网格 token 的行；manifest 的 lora 为 null）：merge 一项跳过（NEEDS_LORA）——
  load_trained(merge=False) 与 merge=True 是同一组权重，没有可合并的东西。restricted 照做：by_thinking 核对评测用的模型（新训练的
  网格 token 行）上受限解码、generate_batch 与教师强制三者一致，这一项与 LoRA 无关，含义不变；across_merge 比较的两侧是同一个检查点
  的两次加载（都没有 LoRA），退化为"加载两次、受限 logits 相同"的复现核对，判据照旧（v1 的 merge 一节；CPU fp32 时逐个相同）。
  跳过的项记在输出 JSON 的 skipped_checks 与摘要里。跳过之后一项都不剩（T 文本评测只有 merge）时不调那个脚本，只写一份说明跳过的
  JSON（format = SKIPPED_FORMAT，pass = true）：T 文本评测的模型就是 base 加上训练过的网格 token 行，没有合并这一步。
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CTI_PATH = REPO_ROOT / "scripts" / "check_training_interfaces.py"
_CTI: Any = None


class MergeCheckFailed(RuntimeError):
    """评测前的合并核对没有通过（或出错）：拒绝评测。"""

    def __init__(self, message: str, summary: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.summary = dict(summary)


def load_cti() -> Any:
    """按路径载入 scripts/check_training_interfaces.py（scripts 不是包）；只载入一次。"""
    global _CTI
    if _CTI is None:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location("_visint_check_training_interfaces", CTI_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _CTI = mod
    return _CTI


PURPOSES: tuple[str, ...] = ("grid_eval", "t_text")  # eval.merge_check.checks 的键（模块说明）
DEFAULT_PURPOSE = "t_text"  # scripts/run_e1.py --checkpoint 调 settings_from_config 时不传 purpose
NEEDS_LORA: tuple[str, ...] = ("merge",)  # 没有 LoRA 的检查点（C1′）跳过的检查项（模块说明）
SKIPPED_FORMAT = "visint-merge-check-skipped/1"


def settings_from_config(doc: Mapping[str, Any], *, tiny: bool, n: int | None = None,
                         purpose: str = DEFAULT_PURPOSE) -> dict[str, Any]:
    """configs/e3b_train.json 的 eval.merge_check（--tiny 时 n 取 eval.tiny.merge_check_n；n 可由命令行覆盖）；检查项取
    eval.merge_check.checks[purpose]（PURPOSES；不传时是 T 文本评测的 t_text）。"""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose 必须是 {PURPOSES} 之一，收到 {purpose!r}")
    e = doc["eval"]
    nn = n if n is not None else (int(e["tiny"]["merge_check_n"]) if tiny else int(e["merge_check"]["n"]))
    if int(nn) < 1:
        raise ValueError(f"合并核对的 n 必须 ≥ 1，收到 {nn}")
    by = e["merge_check"]["checks"]
    if not isinstance(by, Mapping) or set(by) != set(PURPOSES):
        raise ValueError(f"eval.merge_check.checks 必须按用途 {PURPOSES} 分开给，收到 {by!r}")
    checks = [str(c) for c in by[purpose]]
    known = load_cti().CHECKS
    if not checks or len(set(checks)) != len(checks) or set(checks) - set(known):
        raise ValueError(f"eval.merge_check.checks.{purpose} 必须是 {known} 里不重复的非空子集，收到 {checks}")
    return {"n": int(nn), "checks": checks}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(doc: Mapping[str, Any], out_json: Path, *, reused: bool) -> dict[str, Any]:
    ck = doc.get("checkpoint") or {}
    out = {"file": str(out_json), "file_sha256": _sha256(out_json), "pass": bool(doc.get("pass")),
           "criteria_version": doc.get("criteria_version"), "n": (doc.get("data") or {}).get("n_requested"),
           "n_used": (doc.get("data") or {}).get("n"), "checks": sorted(doc.get("checks") or {}),
           "manifest_sha256": ck.get("manifest_sha256"), "reused": reused, "failures": doc.get("failures"),
           "error": (doc.get("error") or {}).get("repr") if doc.get("error") else None}
    if doc.get("skipped_checks"):  # 没有 LoRA 的检查点（模块说明）；有 LoRA 的检查点摘要不变
        out["skipped_checks"] = list(doc["skipped_checks"])
    return out


def _reusable(doc: Mapping[str, Any], *, manifest_sha: str, n: int, checks: Sequence[str], tiny: bool, dtype: str,
              criteria_version: str, e3b_config_sha: str | None = None) -> bool:
    try:
        ok = (doc.get("pass") is True and not doc.get("error")
              and doc["checkpoint"]["manifest_sha256"] == manifest_sha
              and int(doc["data"]["n_requested"]) == int(n)
              and sorted(doc["checks"]) == sorted(checks)
              and doc["criteria_version"] == criteria_version
              and bool(doc["model"]["tiny"]) == bool(tiny) and doc["model"]["dtype"] == dtype)
        if ok and e3b_config_sha is not None and doc.get("restricted_inputs"):
            ok = doc["restricted_inputs"]["e3b_config"]["sha256"] == e3b_config_sha
        return bool(ok)
    except (KeyError, TypeError):
        return False


def _skipped_document(ck: Path, manifest: Mapping[str, Any], manifest_sha: str, *, model_key: str, tiny: bool, dtype: str,
                      n: int, skipped: Sequence[str], criteria_version: str) -> dict[str, Any]:
    """没有 LoRA、检查项全部跳过时写的说明（模块说明）。"""
    return {"format": SKIPPED_FORMAT, "pass": True, "criteria_version": criteria_version, "checks": {},
            "skipped_checks": list(skipped),
            "skip_reason": "检查点没有 LoRA（manifest 的 lora 为 null，如 C1′）：load_trained(merge=False) 与 merge=True 是同一组权重，"
                           "没有可合并的东西；T 文本评测的模型就是 base 加上训练过的网格 token 行",
            "checkpoint": {"dir": str(ck), "manifest_sha256": manifest_sha, "lora": manifest.get("lora"),
                           "build_spec": manifest.get("build_spec"),
                           "trainable_token_rows": manifest.get("trainable_token_rows")},
            "model": {"key": model_key, "tiny": bool(tiny), "dtype": dtype}, "data": {"n_requested": int(n), "n": 0},
            "failures": {}, "error": None}


def run_merge_check(checkpoint: str | Path, *, model_key: str, tiny: bool, n: int, checks: Sequence[str], out_json: str | Path,
                    device: str, dtype: str, data_root: str | Path | None = None, dev_subsets: str | Path | None = None,
                    e3b_config: str | Path | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """跑（或复用）合并核对；通过返回摘要，否则抛 MergeCheckFailed（规则见模块说明）。e3b_config：透传给那个脚本的
    --e3b-config（None = 那个脚本的默认 configs/e3b_train.json）。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    cti = load_cti()
    ck = Path(checkpoint)
    out = Path(out_json)
    manifest_sha = _sha256(ck / "manifest.json")
    manifest = json.loads((ck / "manifest.json").read_text(encoding="utf-8"))
    skipped: list[str] = []
    if manifest.get("lora") is None:  # 没有 LoRA（C1′）：merge 一项跳过（模块说明）
        skipped = [c for c in checks if c in NEEDS_LORA]
        checks = [c for c in checks if c not in NEEDS_LORA]
        if skipped:
            log(f"[合并核对] 检查点没有 LoRA：跳过 {skipped}" + (f"，照做 {list(checks)}" if checks else ""))
    if not checks:
        doc = _skipped_document(ck, manifest, manifest_sha, model_key=model_key, tiny=tiny, dtype=dtype, n=n, skipped=skipped,
                                criteria_version=cti.CRITERIA_VERSION)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return _summary(doc, out, reused=False)
    cfg_sha = _sha256(Path(e3b_config)) if e3b_config is not None else None
    if out.exists():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if _reusable(old, manifest_sha=manifest_sha, n=n, checks=checks, tiny=tiny, dtype=dtype,
                     criteria_version=cti.CRITERIA_VERSION, e3b_config_sha=cfg_sha):
            log(f"[合并核对] 复用已通过的 {out}")
            return _summary(old, out, reused=True)
    argv = ["--model", model_key, "--checks", *checks, "--n", str(int(n)), "--checkpoint", str(ck), "--out", str(out),
            "--device", str(device), "--dtype", str(dtype)]
    if tiny:
        argv.append("--tiny")
    if data_root is not None:
        argv += ["--data-root", str(data_root)]
    if dev_subsets is not None:
        argv += ["--dev-subsets", str(dev_subsets)]
    if e3b_config is not None:
        argv += ["--e3b-config", str(e3b_config)]
    log(f"[合并核对] check_training_interfaces {' '.join(argv)}")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = cti.main(argv)
    except SystemExit as e:  # argparse 的用法错误
        raise MergeCheckFailed(f"合并核对的参数不合法（退出码 {e.code}）：{argv}", {"argv": argv}) from None
    for line in buf.getvalue().splitlines():
        log(f"[合并核对] {line}")
    if not out.exists():
        raise MergeCheckFailed(f"合并核对没有写出 {out}（退出码 {code}）", {"argv": argv, "code": code})
    doc = json.loads(out.read_text(encoding="utf-8"))
    if skipped:  # 跳过的项记进输出 JSON（模块说明）
        doc["skipped_checks"] = list(skipped)
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summ = _summary(doc, out, reused=False)
    summ["exit_code"] = code
    if code != 0 or not doc.get("pass"):
        raise MergeCheckFailed(f"评测前的合并核对没有通过（退出码 {code}）：{summ['failures'] or summ['error']}；拒绝评测（{out}）",
                               summ)
    return summ


__all__ = ["CTI_PATH", "DEFAULT_PURPOSE", "MergeCheckFailed", "NEEDS_LORA", "PURPOSES", "SKIPPED_FORMAT", "load_cti",
           "run_merge_check", "settings_from_config"]
