"""X-M 的密封守卫（DEV-0080；指令第二、五节；D165）。运行器（scripts/run_xm.py）在读任何数据、加载任何模型之前调用；不合格抛 XMGateError，
命令行按用法错误退出（码 2），什么都不写。

薄的一层：门名、gate 文件的核对与常量都用 src/xm/data（DEV-0079）的；这里另加"不是密封划分却给了 --gate / --xm-gate 就拒绝"、基座设置
（没有检查点）的 gate 核对、冻结提示的要求与放行记录。

- xm_decision（判定集，密封）：必须给 --gate，且恰是门名 G-XM（src/xm/data.require_unsealed）；放行记录（门名、依据、时间）追加进
  <out>/<模型>/UNSEAL_LOG，门名与依据进运行身份（不含时间，续跑身份不变）。
- xm_final、lvis_sub（最终报告集、LVIS 子集）与 reasonseg_val 上的 X-M 检查点：必须给 --xm-gate <gate_xm.json>（DEV-0081 的判定脚本写）。
  src/xm/data.check_xm_gate 按约定的键读：verdict 恰是 XM_GATE_PASS、rehearsal 恰是 false、checks 是非空对象且每一项恰是 true、
  candidate.manifest_sha256 等于本次 --checkpoint 的 manifest.json 的 sha256（基座零样本的设置没有检查点，这一条不比，身份里记
  checkpoint_compared = false）。认过的 XMGatePass 再交给 load_pairs(xm_gate = …)。
  gate_xm.json 的路径与 sha256 进运行身份，放行记录追加进 <out>/<模型>/UNLOCK_LOG。
- reasonseg_val 上的 X-2 检查点（--system x2，指令第六节 D1–D3 的描述性运行）不要 gate。
- 判定集、最终报告集、LVIS 子集只用冻结的提示（src/xm/prompts.FROZEN；--tiny 不查）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.xm import data as XD
from src.xm import prompts as XMP

GATE = XD.GATE  # "G-XM"
SEALED: tuple[str, ...] = XD.SEALED_SPLITS  # ("xm_decision",)
GATED: tuple[str, ...] = XD.FINAL_GATED  # ("xm_final", "lvis_sub")
REASONSEG = XD.REASONSEG_VAL
SEAL_DECISION = XD.SEAL_DECISION
GATE_PASS = XD.XM_GATE_PASS  # "通过"（唯一来源 src/xm/data.XM_GATE_PASS，DEV-0079）
GATE_KEYS: tuple[str, ...] = XD.XM_GATE_KEYS
UNSEAL_LOG = "unseal_xm_decision.jsonl"
UNLOCK_LOG = "xm_gate_unlock.jsonl"


class XMGateError(ValueError):
    """密封或 gate 不合格（src/xm/data 的 SealedSplitError、XMGateError 换成这一个，命令行按用法错误退出）。"""


def is_sealed(split: str) -> bool:
    return XD.is_sealed(split)


def needs_xm_gate(split: str, system: str) -> bool:
    """这个划分 × 系统要不要 --xm-gate（system：xm、base、x2）。"""
    return split in GATED or (split == REASONSEG and system == "xm")


def needs_frozen_prompts(split: str) -> bool:
    return split in SEALED or split in GATED


def require_unsealed(split: str, gate: str | None) -> dict[str, Any] | None:
    """密封守卫（src/xm/data.require_unsealed）；另外：不是密封划分却给了 --gate 也拒绝。返回放行记录 {gate, decision, unsealed_utc} 或 None。"""
    if not is_sealed(split):
        if gate is not None:
            raise XMGateError(f"--gate 只配密封的 {list(SEALED)}（--split {split} 不是）")
        return None
    try:
        return XD.require_unsealed(split, gate)
    except XD.SealedSplitError as e:
        raise XMGateError(str(e)) from e


def seal_identity(rec: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return XD.seal_identity(rec)


def _base_gate_pass(path: Path) -> XD.XMGatePass:
    """基座零样本的设置（没有检查点）：src/xm/data.check_xm_gate 的同样几条，只是不比 manifest（记成 gate 里候选的 sha256）。"""
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError) as e:
        raise XMGateError(f"gate 文件 {path} 读不了：{e}") from e
    if not isinstance(doc, dict) or any(k not in doc for k in GATE_KEYS):
        raise XMGateError(f"gate 文件 {path} 不是 G-XM 的判定输出（要有键 {list(GATE_KEYS)}）")
    if doc["rehearsal"] is not False:
        raise XMGateError(f"gate 文件 {path} 是彩排（rehearsal = {doc['rehearsal']!r}），不作数")
    if doc["verdict"] != GATE_PASS:
        raise XMGateError(f"gate 文件 {path} 的结论不是“{GATE_PASS}”（verdict = {doc['verdict']!r}）")
    checks = doc["checks"]
    if not isinstance(checks, dict) or not checks or any(v is not True for v in checks.values()):
        bad = sorted(k for k, v in checks.items() if v is not True) if isinstance(checks, dict) else checks
        raise XMGateError(f"gate 文件 {path} 的判定点核对没有全部为真：{bad or '空'}")
    have = (doc["candidate"] if isinstance(doc["candidate"], dict) else {}).get("manifest_sha256")
    if not isinstance(have, str) or not have:
        raise XMGateError(f"gate 文件 {path} 判定的检查点（manifest sha256 {have!r}）不是本次的：没有候选的 manifest sha256")
    return XD.XMGatePass(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), checkpoint_manifest_sha256=have, verdict=str(doc["verdict"]))


def check_xm_gate(path: str | Path | None, *, checkpoint: str | Path | None) -> dict[str, Any]:
    """gate_xm.json 的核对：有检查点时就是 src/xm/data.check_xm_gate（verdict、rehearsal、checks、candidate.manifest_sha256 = 本次检查点的）；
    基座零样本的设置（checkpoint = None）不比 manifest。返回 {"pass"（XMGatePass，交给 load_pairs）, "identity"（进运行身份）, "summary"}。"""
    if path is None:
        raise XMGateError("这个划分只在 G-XM 判定通过之后运行：要给 --xm-gate <gate_xm.json>（DEV-0081 的判定脚本写）")
    p = Path(path)
    if checkpoint is not None:
        try:
            gp = XD.check_xm_gate(p, checkpoint=checkpoint)
        except XD.XMGateError as e:
            raise XMGateError(str(e)) from e
    else:
        gp = _base_gate_pass(p)
    doc = json.loads(p.read_bytes())
    ident = {"path": gp.path, "sha256": gp.sha256, "verdict": gp.verdict, "candidate_manifest_sha256": gp.checkpoint_manifest_sha256,
             "checkpoint_compared": checkpoint is not None}
    return {"pass": gp, "identity": ident, "summary": {k: doc.get(k) for k in ("verdict", "rehearsal", "checks")}}


def require_frozen_prompts(split: str, *, tiny: bool) -> None:
    if needs_frozen_prompts(split) and not tiny and XMP.FROZEN is not True:
        raise XMGateError(f"--split {split} 只用冻结的提示：src/xm/prompts.FROZEN 不是 True（版本 {XMP.PROMPT_VERSION}）")


def append_log(out: str | Path, model_key: str, name: str, rec: Mapping[str, Any]) -> Path:
    p = Path(out) / model_key / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")
    return p


__all__ = ["GATE", "GATED", "GATE_KEYS", "GATE_PASS", "REASONSEG", "SEALED", "UNLOCK_LOG", "UNSEAL_LOG", "XMGateError", "append_log",
           "check_xm_gate", "is_sealed", "needs_frozen_prompts", "needs_xm_gate", "require_frozen_prompts", "require_unsealed", "seal_identity"]
