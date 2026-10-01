"""CountBench 的人工问句（DEV-0015 任务书 §4；D32-3）：google-research/big_vision 仓库里 CountBenchQA 的问句文件。

来源：big_vision（Apache-2.0）的 big_vision/datasets/countbenchqa/data/countbench_paired_questions.json，按固定 commit
从 raw.githubusercontent.com 取（commit 是这个文件在 GitHub 上唯一的一次提交 "CountBenchQA release (#119)"）。
文件是 540 条 {"question": …} 的 JSON 列表；第 k 条对应 CountBench 的第 k 行（0 起，与 countbench.jsonl 的 row 同一个编号）：
big_vision 的 countbenchqa.py 在 _generate_examples_hf 里按行号把问句并进 HF 打包版（df['question'] = df_question）。
问句怎么进提示由 E1 框架决定（configs/e1_prompts.json 的 countbenchqa_questions；src/e1/tasks.load_count_questions）。

与官方的核对：文件 sha256 与字节数钉在 SPEC（开发机 2026-10-01 下载核对）；结构（恰好 n_questions 条、每条只有非空的
question 字符串）；CountBench 已落地时，行数与 countbench.jsonl 相同。

落地后的布局（<root>/countbenchqa_questions/）：countbench_paired_questions.json（原样字节）、MANIFEST.json。
文件只有 31 KB，--limit 不起作用（总是整份落地）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.land import net
from src.data.land.common import LandContext, budget_gate, register

SOURCE = "countbenchqa_questions"
FILE_NAME = "countbench_paired_questions.json"
PROJECT_TERMS = "D28-9：限非商业研究，不再分发任何数据或由数据生成的图像"

SPEC: dict[str, Any] = {
    "repo": "google-research/big_vision",
    "commit": "ec104fd760a1272796e12e5fcde42f6bc3f78f9b",
    "path": "big_vision/datasets/countbenchqa/data/countbench_paired_questions.json",
    "url": "https://raw.githubusercontent.com/google-research/big_vision/ec104fd760a1272796e12e5fcde42f6bc3f78f9b/"
           "big_vision/datasets/countbenchqa/data/countbench_paired_questions.json",
    "sha256": "d3d94521899ff05bcd3facc8712e3cb4c1f9fb5cfa17ce35b14310a882090046",
    "bytes": 31_595,
    "n_questions": 540,
    "estimate_bytes": 31_595 + 100_000,
    "license": {
        "text": "Apache-2.0（big_vision 仓库根目录的 LICENSE；GitHub API 的 license.spdx_id = Apache-2.0）",
        "url": "https://github.com/google-research/big_vision/blob/ec104fd760a1272796e12e5fcde42f6bc3f78f9b/LICENSE",
        "project_terms": PROJECT_TERMS,
    },
}


def check_questions(doc: Any, n_expected: int) -> list[str]:
    """问句文件的结构问题（空列表 = 合格）：顶层是列表、恰好 n_expected 条、每条恰好一个非空字符串键 question。"""
    if not isinstance(doc, list):
        return [f"顶层应为列表，收到 {type(doc).__name__}"]
    bad = []
    if len(doc) != n_expected:
        bad.append(f"应有 {n_expected} 条，收到 {len(doc)} 条")
    for k, e in enumerate(doc):
        if not (isinstance(e, dict) and set(e) == {"question"} and isinstance(e["question"], str) and e["question"].strip()):
            bad.append(f"第 {k} 条不是 {{\"question\": 非空字符串}}：{e!r}"[:200])
    return bad


def land(ctx: LandContext, spec: dict[str, Any] | None = None) -> None:
    spec = {**SPEC, **(spec or {})}
    src_dir = ctx.source_dir(SOURCE)
    m = ctx.manifest(SOURCE)
    m.data["license"] = spec["license"]
    budget_gate(ctx, m, spec["estimate_bytes"], {"file": spec["bytes"], "margin": "100 kB"})
    dest = src_dir / FILE_NAME
    res = net.download(spec["url"], dest, expected_sha256=spec["sha256"], expected_size=spec["bytes"], retries=ctx.retries,
                       backoff=ctx.backoff, log=ctx.log)
    m.verify("问句文件的 sha256 与字节数 = SPEC 钉住的值", res.sha256 == spec["sha256"] and res.bytes == spec["bytes"],
             got={"sha256": res.sha256, "bytes": res.bytes}, expected={"sha256": spec["sha256"], "bytes": spec["bytes"]})
    doc = json.loads(dest.read_text(encoding="utf-8"))
    problems = check_questions(doc, spec["n_questions"])
    m.verify(f"结构：{spec['n_questions']} 条 {{\"question\": 非空字符串}}", not problems, problems_head=problems[:10])
    cb = ctx.source_dir("countbench") / "countbench.jsonl"
    if cb.exists():
        n_rows = sum(1 for line in cb.read_text(encoding="utf-8").splitlines() if line.strip())
        m.verify("条数 = 已落地的 countbench.jsonl 的行数（第 k 条对应第 k 行）", n_rows == len(doc), n_rows=n_rows,
                 n_questions=len(doc))
    else:
        m.note("countbench 还没有落地，没有核对条数与 countbench.jsonl 的行数；E1 读问句时会再核对")
    register(m, src_dir, dest, kind="metadata", url=spec["url"])
    m.data["upstream"].append({
        "repo": spec["repo"], "commit": spec["commit"], "path": spec["path"], "url": spec["url"], "sha256": res.sha256,
        "bytes": res.bytes, "skipped": res.skipped,
        "official_check": "sha256 与字节数钉在 src/data/land/countbenchqa.py 的 SPEC；结构与条数核对",
    })
    m.data["selection"] = {"rule": "整份文件原样落地；第 k 条问句对应 CountBench 第 k 行（big_vision countbenchqa.py 按行号对齐）"}
    m.data["stats"] = {"n_questions": len(doc) if isinstance(doc, list) else None}
    m.data["dedupe_facts"] = {"known": "只有文本问句，没有图像；图像是已落地的 countbench（nielsr/countbench 打包版）"}
    m.df("完成")
    m.write()


def questions_path(root: str | Path) -> Path:
    return Path(root) / SOURCE / FILE_NAME


__all__ = ["FILE_NAME", "SOURCE", "SPEC", "check_questions", "land", "questions_path"]
