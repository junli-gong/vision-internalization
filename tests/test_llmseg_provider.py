"""DEV-0066：LLMSeg 的本地提供方（src/llmseg/provider.py）。

- 阶段判定：每个阶段的标记句恰好只出现在它自己的官方模板里；渲染出的官方提示判到正确的阶段，"- id=" 行数 = 实例数 / 焦点框数。
- 输出上限：用真分词器（Qwen3.5 家族的 tokenizer）量最长合理回复（官方提示的上限：64 点轮廓、6 个焦点框、12 点焦点轮廓；缩进 2、一位小数），
  不超过 answer_cap；上限的公式钉住。
- 坐标（管理员补充第 1 条）：我们传的模型名落在官方"0–1000、x 在前"的约定上；回复里的 0–1000 坐标经官方 normalize_boxes /
  rescale_spline_groups 换回像素，与我们适配器的换算（qwen35.native_box_to_xyxy、geometry.norm1000_to_abs）逐值相同；提示里的框
  （官方 to_model_box）与 abs_to_norm1000 一致。coord_audit 只统计（> 1000 的个数），不改官方的换算。
- 合批：小型随机 Qwen（CPU fp32）上，一组跨阶段、跨图、不同上限的请求，锁步合批（slots 5、batch 3）与逐条（slots 1、batch 1，走 generate）
  的 token 逐个相同；合批确实发生（子批行数 > 1）。批的组成与线程快慢无关（工作函数里随机睡眠，跑两遍组成相同）。
- 不联网：install 之后官方的三个闭源调用都抛 ExternalCallForbidden（官方 _complete 的重试接不住）；没有 key、也没装本地提供方时，官方
  query 在任何联网之前就报"缺 key"；完整的管线跑一遍，socket 一次都没碰。
- 基础设施错误（生成报错）中止运行（LocalProviderError），不被官方管线当成"这条没分割出来"。
"""

from __future__ import annotations

import json
import random
import socket
import threading
import time

import numpy as np
import pytest
import torch
from PIL import Image

from src.adapters.factory import make_adapter
from src.adapters.qwen35 import native_box_to_xyxy
from src.e1.prompts import PROMPTS_PATH, load_prompt_config
from src.executors.geometry import abs_to_norm1000, norm1000_to_abs
from src.llmseg import provider as PV
from src.llmseg import vendor as V
from tests._llmseg_fakes import GridAdapter, grid_reply

QWEN = "qwen3.5-0.8b"
KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(scope="module")
def mods():
    return V.load()


@pytest.fixture(scope="module")
def tiny():
    return make_adapter(QWEN, device="cpu", dtype=torch.float32, tiny=True, init_std=0.05)


@pytest.fixture(scope="module")
def decoding():
    return load_prompt_config(PROMPTS_PATH).decoding()


def _inst(n: int, W: int = 640, H: int = 480) -> list[dict]:
    return [{"id": i, "label": "man", "box": [10.0 + 100 * i, 20.5, 90.7 + 100 * i, 400.2],
             "morphology": {"shape": "upright figure", "aspect": "tall", "convexity": "convex", "parts": ["head"]}} for i in range(n)]


def render(mods, stage: str, n: int, model: str = QWEN, W: int = 640, H: int = 480, path: str = "/w/x/image.jpg") -> str:
    """官方模板按官方 pipeline 的同一套填法渲染（n = 实例数，3b 是焦点框数）。"""
    llm, P, pl = mods.llm, mods.prompts, mods.pipeline
    insts = _inst(n, W, H)
    common = dict(coord_space=llm.coord_space_instructions(model, W, H), box_field=llm.box_field_name(model),
                  box_order=llm.box_order_literal(model))
    if stage == PV.STAGE_DETECT:
        return P.PROMPT_DETECT.format(image_path=path, **common, box_format=llm.box_format_instructions(model), query="the man",
                                      topology_schema=P.topology_schema(False), form_note=P.form_note(False))
    if stage == PV.STAGE_SEGMENT:
        return P.PROMPT_SEGMENT.format(image_path=path, **common, label="man", n_instances=n,
                                       instances_block=pl.format_instances(insts, model, (W, H), True), point_order_note="",
                                       loops_note=P.loops_note(False), color_scope_note=P.color_scope_note(n),
                                       output_budget_note=P.output_budget_note(n, "instances"), degree_guidance=P.segment_degree_guidance(),
                                       mask_color_options=P.mask_color_options())
    if stage == PV.STAGE_FOCUS:
        return P.PROMPT_FOCUS.format(overlay_path=path, label="man", **common,
                                     instances_block=pl.format_instances(insts, model, (W, H), False),
                                     focus_boxes_per_instance=P.FOCUS_BOXES_PER_INSTANCE, max_focus_boxes=P.FOCUS_BOXES_PER_INSTANCE * n,
                                     focus_spread_note=P.focus_spread_note(n), mask_color="cyan")
    if stage == PV.STAGE_FOCUS_SPLINES:
        lines = [f'- id={i} instance_id=0 type="positive" box={llm.format_model_box([10, 20, 60, 90], model, W, H)}  shape: bump'
                 for i in range(n)]
        return P.PROMPT_FOCUS_SPLINE.format(overlay_path=path, coord_space=common["coord_space"], label="man", n_boxes=n,
                                            focus_boxes_block="\n".join(lines), box_order=common["box_order"], point_order_note="",
                                            mask_color="cyan", degree=3, output_budget_note=P.output_budget_note(n, "focus boxes"),
                                            degree_guidance=P.focus_degree_guidance())
    if stage == PV.STAGE_BOX_ADJUST:
        return P.PROMPT_BOX_ADJUST.format(overlay_path=path, label="man", coord_space=common["coord_space"],
                                          instances_block=pl.format_instances(insts, model, (W, H), False, include_caps=True),
                                          mask_color="cyan")
    raise ValueError(stage)


# ---------------------------------------------------------------------------
# 阶段与上限
# ---------------------------------------------------------------------------


def test_stage_markers_are_unique_to_their_templates(mods):
    for stage, marker in PV.STAGE_MARKERS.items():
        owners = [s for s, t in PV.STAGE_TEMPLATES.items() if marker in getattr(mods.prompts, t)]
        assert owners == [stage], (stage, owners)
    assert set(PV.STAGES) == set(PV.STAGE_MARKERS) == set(PV.ANSWER_CAP_RULE) == set(PV.STAGE_TEMPLATES)
    for stage in PV.STAGES:
        for n in (1, 3):
            p = render(mods, stage, n)
            assert PV.stage_of(p) == stage
            assert PV.n_units(p) == (0 if stage == PV.STAGE_DETECT else n), (stage, n)
    with pytest.raises(PV.LocalProviderError):
        PV.stage_of("no marker here")


def test_answer_cap_formula():
    assert PV.ANSWER_CAP_RULE == {"detect": (2048, 0), "segment": (256, 1792), "box_adjust": (256, 128), "focus_boxes": (256, 640),
                                  "focus_splines": (256, 384)}
    one = "- id=0 x\n"
    assert PV.answer_cap("detect", one * 5) == 2048
    assert PV.answer_cap("segment", one) == 2048 and PV.answer_cap("segment", one * 3) == 256 + 3 * 1792
    assert PV.answer_cap("focus_boxes", one * 2) == 256 + 2 * 640
    assert PV.answer_cap("focus_splines", one * 6) == 256 + 6 * 384
    assert PV.answer_cap("segment", one * 200) == PV.CAP_CEIL == 16384
    assert PV.answer_cap("segment", one, cap_limit=48) == 48


def _max_legit(mods, stage: str, n: int, rng) -> str:
    """官方提示允许的最长合理回复（缩进 2、一位小数的坐标、带说明字段）。"""
    P = mods.prompts

    def pts(k):
        return [[round(float(x), 1), round(float(y), 1)] for x, y in rng.uniform(100, 999, (k, 2))]

    box = [123.4, 234.5, 876.5, 987.6]
    if stage == PV.STAGE_DETECT:  # 规则按 10 个检测给上限
        det = {"id": 0, "label": "man in the red shirt", "box": box, "confidence": 0.95,
               "morphology": {"shape": "upright human figure standing with both arms at the sides", "aspect": "tall",
                              "convexity": "slightly-concave", "parts": ["head", "left arm", "right arm", "legs", "feet"]}}
        return json.dumps({"detections": [dict(det, id=i) for i in range(10)]}, indent=2)
    if stage == PV.STAGE_SEGMENT:
        return json.dumps({"mask_color": "spring_green", "mask_color_reason": "contrasts with the red shirt and the grey wall",
                           "degree": 3, "degree_reason": "organic rounded silhouette",
                           "instances": [{"id": i, "splines": [{"role": "outer", "degree": 3, "closed": True,
                                                                "control_points": pts(P.SEGMENT_CTRL_POINTS_MAX)}]} for i in range(n)]},
                          indent=2)
    if stage == PV.STAGE_FOCUS:
        fb = {"type": "negative", "instance_id": 0, "shape": "thin curved sliver of background along the left arm, simple edge",
              "box": box}
        return json.dumps({"focus_boxes": [fb] * (P.FOCUS_BOXES_PER_INSTANCE * n)}, indent=2)
    if stage == PV.STAGE_FOCUS_SPLINES:
        return json.dumps({"refinements": [{"id": i, "splines": [{"degree": 3, "closed": True,
                                                                  "control_points": pts(P.FOCUS_CTRL_POINTS_MAX)}]} for i in range(n)]},
                          indent=2)
    return json.dumps({"instances": [{"id": i, "x1": -123, "x2": 456, "y1": 0, "y2": -78} for i in range(n)]}, indent=2)


def test_answer_caps_cover_longest_reasonable_replies(mods, tiny):
    tok = tiny.tokenizer
    rng = np.random.default_rng(0)
    for stage in PV.STAGES:
        for n in (1, 2, 3, 6):
            if stage == PV.STAGE_DETECT and n > 1:
                continue
            reply = _max_legit(mods, stage, n, rng)
            k = len(tok.encode(reply, add_special_tokens=False))
            cap = PV.answer_cap(stage, render(mods, stage, n))
            assert k <= cap, (stage, n, k, cap)
            assert k >= cap * 0.35 or stage == PV.STAGE_BOX_ADJUST, (stage, n, k, cap)  # 上限不是随便放大的


# ---------------------------------------------------------------------------
# 坐标
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"])
def test_model_names_use_official_qwen_grid_convention(mods, key):
    c = PV.check_coordinate_convention(mods.llm, key)
    assert c["grid"] == "0-1000"
    assert "NORMALIZED 0-1000" in mods.llm.coord_space_instructions(key, 640, 400)  # 提示要求模型答 0–1000
    for bad in ("tiny(qwen3.5-0.8b)", "claude-sonnet-5-5", "gemini-3.7-flash"):
        with pytest.raises(PV.LocalProviderError):
            PV.check_coordinate_convention(mods.llm, bad)


def test_official_conversion_equals_adapter_conversion(mods):
    llm = mods.llm
    W, H = 640, 400
    quads = [[672, 289, 999, 1000], [0, 0, 1, 1], [12.5, 333.3, 500.0, 777.7]]
    dets = [{"box": list(q), "label": "x"} for q in quads] + [{"bbox_2d": [100, 200, 300, 400]}]  # Qwen 原生键也收
    llm.normalize_boxes(dets, QWEN, W, H)
    for d, q in zip(dets, quads + [[100, 200, 300, 400]]):
        assert d["box"] == [float(v) for v in native_box_to_xyxy(q, width=W, height=H)], (d, q)
    assert dets[0]["box"][:2] == pytest.approx([430.08, 115.6], abs=1e-9)  # 管理员给的例子：[672, 289]/1000 × 640×400
    pts = [[672.0, 289.0], [1000.0, 0.0], [333.3, 999.9]]
    groups = [[{"control_points": [list(p) for p in pts]}]]
    llm.rescale_spline_groups(groups, QWEN, W, H)
    want = norm1000_to_abs(np.asarray(pts), width=W, height=H).tolist()
    assert groups[0][0]["control_points"] == want
    # 提示里的框：像素 → 0–1000（官方 to_model_box 与我们的 abs_to_norm1000 同一个换算）
    px = [100.0, 50.0, 600.0, 399.0]
    np.testing.assert_allclose(llm.to_model_box(px, QWEN, W, H), abs_to_norm1000(px, width=W, height=H), rtol=0, atol=1e-9)
    # 判据：有值 > 1000 时官方当像素、不换算（隐患），coord_audit 把它数出来
    dets = [{"box": [100, 200, 1200, 300]}]
    llm.normalize_boxes(dets, QWEN, W, H)
    assert dets[0]["box"] == [100.0, 200.0, 1200.0, 300.0]


def test_coord_audit_counts_values_above_1000(mods):
    llm = mods.llm
    a = PV.coord_audit(llm, json.dumps({"detections": [{"box": [100, 200, 1200, 300], "confidence": 2000}]}), PV.STAGE_DETECT)
    assert a == {"json_ok": True, "parsed": True, "n_values": 4, "n_gt_1000": 1, "max_abs": 1200.0}
    seg = json.dumps({"degree": 3, "instances": [{"id": 0, "splines": [{"control_points": [[1, 2], [3, 1001], [5, 6, 7]]}]}]})
    assert PV.coord_audit(llm, "```json\n" + seg + "\n```", PV.STAGE_SEGMENT)["n_values"] == 6
    adj = json.dumps({"instances": [{"id": 0, "x1": 5, "x2": -1500, "y1": 0, "y2": 0}]})
    assert PV.coord_audit(llm, adj, PV.STAGE_BOX_ADJUST)["n_gt_1000"] == 1
    assert PV.coord_audit(llm, adj, PV.STAGE_FOCUS)["n_values"] == 0  # 增量只在 box_adjust 里算坐标
    cut = '{"refinements": [{"id": 0, "splines": [{"control_points": [[1, 2], [3, 4]'  # 截断：官方的结构修复能读
    a = PV.coord_audit(llm, cut, PV.STAGE_FOCUS_SPLINES)
    assert a["json_ok"] is False and a["parsed"] is True and a["n_values"] == 4
    assert PV.coord_audit(llm, "no json at all", PV.STAGE_DETECT)["parsed"] is False


# ---------------------------------------------------------------------------
# 合批与逐条
# ---------------------------------------------------------------------------


def _requests(mods, imgs):
    """每条"管线"依次交 2–3 个请求（不同阶段、不同图、不同上限）。"""
    plan = []
    for i in range(6):
        seq = [(PV.STAGE_DETECT, render(mods, PV.STAGE_DETECT, 1, path=f"/w/{i}/image.jpg"), 6 + i % 3),
               (PV.STAGE_SEGMENT, render(mods, PV.STAGE_SEGMENT, 1 + i % 2), 10),
               (PV.STAGE_FOCUS, render(mods, PV.STAGE_FOCUS, 1), 8 + (i % 2) * 4)][: 2 + i % 2]
        plan.append([(st, pr, cap, imgs[(i + k) % len(imgs)]) for k, (st, pr, cap) in enumerate(seq)])
    return plan


def _drive(sched, plan, sleep=None):
    def work(i, seq):
        out = []
        for k, (st, pr, cap, img) in enumerate(seq):
            if sleep:
                time.sleep(sleep())
            res = sched.submit(PV.Request(item_index=i, seq=k + 1, stage=st, cap=cap, prompt=pr, image=img))
            out.append((tuple(res.token_ids), res.text, res.stopped, res.n_prompt_tokens))
        return out
    return sched.run(plan, work)


def test_batched_equals_sequential_token_by_token(mods, tiny, decoding):
    rng = np.random.default_rng(1)
    imgs = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for h, w in ((36, 48), (64, 40), (50, 50))]
    plan = _requests(mods, imgs)
    seq_batches, bat_batches = [], []
    seq = PV.LockstepScheduler(PV.AdapterBackend(tiny, PV.GenerationConfig(decoding=decoding, batch_size=1)), slots=1,
                               on_batch=seq_batches.append)
    bat = PV.LockstepScheduler(PV.AdapterBackend(tiny, PV.GenerationConfig(decoding=decoding, batch_size=3, max_batch_tokens=None)),
                               slots=5, on_batch=bat_batches.append)
    a = _drive(seq, plan)
    b = _drive(bat, plan)
    assert a == b
    assert all(x["rows"] == 1 for x in seq_batches)
    assert max(x["rows"] for x in bat_batches) == 3 and len(bat_batches) < len(seq_batches)
    assert {x["stage"] for x in bat_batches} == {PV.STAGE_DETECT, PV.STAGE_SEGMENT, PV.STAGE_FOCUS}
    assert len({tuple(r[0][0]) for r in a}) > 1  # 输出随输入变（不是对常数输出的空比较）
    for x in bat_batches:  # 同一个子批里上限相同（每行的 max_new_tokens 就是它自己的上限）
        assert x["max_generated"] <= x["cap"]


def test_batch_composition_does_not_depend_on_thread_timing(mods):
    img = Image.new("RGB", (40, 30))
    plan = [[(PV.STAGES[(i + k) % 2 * 3], f"{PV.STAGE_MARKERS[PV.STAGES[(i + k) % 2 * 3]]} item {i} call {k}", 50, img)
             for k in range(1 + i % 3)] for i in range(9)]

    def once(seed):
        r = random.Random(seed)
        lock = threading.Lock()
        comps = []
        ad = GridAdapter(reply=lambda p: "{}")
        sched = PV.LockstepScheduler(PV.AdapterBackend(ad, PV.GenerationConfig(decoding={"do_sample": False}, batch_size=2)),
                                     slots=4, on_batch=lambda x: comps.append((x["round"], x["stage"], x["rows"])))

        def sleep():
            with lock:
                return r.uniform(0, 0.01)
        _drive(sched, plan, sleep=sleep)
        return comps

    assert once(0) == once(1) == once(2)


# ---------------------------------------------------------------------------
# 不联网；基础设施错误
# ---------------------------------------------------------------------------


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def deny(*a, **k):
        attempts.append(a)
        raise AssertionError(f"尝试联网：{a!r}")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return attempts


def test_official_query_without_key_fails_before_network(mods, no_network, tmp_path):
    llm = mods.llm
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    for model in ("qwen3.5-9b", "claude-sonnet-5-5", "gemini-3.7-flash", "gpt-5.6-terra"):
        with pytest.raises(RuntimeError, match="needs"):
            llm.query("x", img, model, "test")
    assert no_network == []


def test_install_blocks_closed_api_calls_and_restores(mods, no_network, monkeypatch):
    llm = mods.llm
    before = {n: getattr(llm, n) for n in ("_complete", *PV.PATCHED_PROVIDER_CALLS)}
    inst = PV.install(mods, lambda *a: "{}")
    try:
        for n in PV.PATCHED_PROVIDER_CALLS:
            with pytest.raises(ExternalCallForbiddenType):
                getattr(llm, n)("m", "p", None, 1.0)
        assert llm._provider_override == PV.LOCAL_PROVIDER and llm._timeout == PV.CALL_TIMEOUT_S
        # 即使设了 key，官方的 _complete（若被调到）也到不了网络：守卫是 BaseException，它的 except Exception 重试接不住
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        with pytest.raises(ExternalCallForbiddenType):
            before["_complete"]("anthropic", "claude-sonnet-5-5", "p", None, 1.0)
    finally:
        inst.restore()
    assert {n: getattr(llm, n) for n in before} == before
    assert no_network == []


ExternalCallForbiddenType = PV.ExternalCallForbidden


def test_full_pipeline_runs_offline_through_local_provider(mods, no_network, tmp_path):
    ad = GridAdapter()
    sched = PV.LockstepScheduler(PV.AdapterBackend(ad, PV.GenerationConfig(decoding={"do_sample": False}, batch_size=4)), slots=2)
    prov = PV.LocalProvider(sched, llm=mods.llm, model_name="qwen3.5-9b")
    router = PV.LogRouter()
    inst = PV.install(mods, prov, router)
    works = []
    try:
        def work(i, _):
            w = tmp_path / f"item{i}"
            w.mkdir()
            img = w / "image.png"
            Image.fromarray(np.full((200, 300, 3), 90, np.uint8)).save(img)
            prov.register(w, i)
            buf = []
            with router.capture(buf):
                rc = mods.pipeline.run(img, "the thing", "qwen3.5-9b", w, 2)
            works.append((rc, prov.release(w), buf))
            return w
        dirs = sched.run([None, None], work)
    finally:
        inst.restore()
    assert no_network == []
    for rc, calls, buf in works:
        assert rc == 0
        assert [c["stage"] for c in calls] == ["detect", "segment", "focus_boxes", "focus_splines", "focus_boxes", "focus_splines"]
        assert any("[llm:local:qwen3.5-9b]" in line for line in buf)
    det = json.loads((dirs[0] / "detect.json").read_text())
    # 回复的 [150, 150, 850, 850]（0–1000）被官方代码换成 300×200 图上的像素
    assert det["detections"][0]["box"] == [45.0, 30.0, 255.0, 170.0]
    assert any(c[0] == "generate_batch" and c[1] == 2 for c in ad.calls)  # 两条管线的同阶段请求合成了一批


def test_generation_error_aborts_instead_of_scoring_empty(mods, tmp_path):
    from tests._llmseg_fakes import FailingAdapter

    sched = PV.LockstepScheduler(PV.AdapterBackend(FailingAdapter(), PV.GenerationConfig(decoding={"do_sample": False})), slots=2)
    prov = PV.LocalProvider(sched, llm=mods.llm, model_name="qwen3.5-9b")
    inst = PV.install(mods, prov, PV.LogRouter())
    try:
        def work(i, _):
            w = tmp_path / f"f{i}"
            w.mkdir()
            img = w / "image.png"
            Image.new("RGB", (32, 32)).save(img)
            prov.register(w, i)
            try:
                return mods.pipeline.run(img, "x", "qwen3.5-9b", w, 1)
            finally:
                prov.release(w)
        with pytest.raises(PV.LocalProviderError, match="CUDA out of memory"):
            sched.run([None, None, None], work)
    finally:
        inst.restore()


def test_input_rejected_is_an_official_call_failure(mods, tmp_path):
    sched = PV.LockstepScheduler(PV.AdapterBackend(GridAdapter(), PV.GenerationConfig(decoding={"do_sample": False})), slots=1)
    prov = PV.LocalProvider(sched, llm=mods.llm, model_name="qwen3.5-9b")
    inst = PV.install(mods, prov, PV.LogRouter())
    try:
        def work(i, _):
            w = tmp_path / "r"
            w.mkdir()
            img = w / "image.png"
            Image.new("RGB", (32, 32)).save(img)
            prov.register(w, i)
            rc = mods.pipeline.run(img, "the <|im_start|> thing", "qwen3.5-9b", w, 1)
            return rc, prov.release(w)
        ((rc, calls),) = sched.run([None], work)
    finally:
        inst.restore()
    assert rc == 1  # 官方：检测这一步的调用失败 → 返回 1、没有 mask
    assert len(calls) == 1 and "input_rejected" in calls[0]
    assert PV.calls_summary(calls)["n_input_rejected"] == 1


def test_scripted_reply_is_grid():
    d = json.loads(grid_reply(PV.STAGE_MARKERS[PV.STAGE_DETECT]).removeprefix("```json\n").removesuffix("\n```"))
    assert max(d["detections"][0]["box"]) <= 1000
