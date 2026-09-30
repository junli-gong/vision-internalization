"""IFEval 官方评分代码（src/health/ifeval）与包装层（src/health/scoring.ifeval_follow）的测试（DEV-0008）。

1. 放入的四个文件与上游逐字相同：去掉 "DEV-0008 vendoring note" 注释段、把 IMPORT_EDITS 换回原文后核对上游 sha256；LICENSE 相同。
2. 官方仓库自带的样例输入输出与期望值：上游的 instructions_test.py 与 instructions_util_test.py 原样放在
   tests/fixtures/health/ifeval_upstream/（后缀 .txt，免得 pytest 直接收集；sha256 钉住），用一个最小的 absl.testing 替身
   （named_parameters 展开、assertLen）加载，每个测试方法作为一条 pytest 用例跑，对照官方写的期望值。
   其中要用 punkt_tab 的（句子计数、word_tokenize）在找不到协议钉住的 punkt_tab 时跳过并给出原因
   （设 NLTK_DATA 指向含 tokenizers/punkt_tab 的目录即可跑；本测试不下载、不写 ~/nltk_data）。
3. ifeval_follow：strict 与 loose 的差别、空回答、多条指令、依赖 kwargs 的指令、langdetect 种子。
4. punkt_tab 的来源核对。
"""

from __future__ import annotations

import copy
import hashlib
import sys
import types
import unittest
from pathlib import Path

import pytest

from src.health import ifeval as V
from src.health import protocol as P
from src.health import scoring as S
from src.health.items import load_items

REPO_ROOT = Path(__file__).resolve().parents[1]
IFEVAL_DIR = REPO_ROOT / "src/health/ifeval"
UPSTREAM_TESTS = REPO_ROOT / "tests/fixtures/health/ifeval_upstream"
UPSTREAM_TEST_SHA256 = {
    "instructions_test.py.txt": "e677c190c393162f7733e89ca3b47ae98b6d9e816d800681d6e14ad918128de5",
    "instructions_util_test.py.txt": "e8e17d434e074c6e2ad997334f4d743b3cafe0abf15cca46d4cf97b94d67ccfe",
}
SYNTHETIC_NLTK_DATA = REPO_ROOT / "tests/fixtures/health/nltk_data"
# 需要 punkt_tab 的上游用例（NumberOfSentences、count_sentences、CapitalWordFrequencyChecker 的 word_tokenize）
PUNKT_CASES = ("test_number_sentences", "test_count_sentences", "test_capital_word_frequency")


def _punkt_status() -> tuple[bool, str]:
    try:
        prov = S.punkt_provenance()
    except S.PunktDataError:
        return False, "找不到 punkt_tab：设 NLTK_DATA 指向含 tokenizers/punkt_tab 的目录（python -m nltk.downloader -d <目录> punkt_tab）"
    if not prov["matches_pinned"]:
        return False, f"{prov['path']} 的 punkt_tab 不是协议钉住的版本"
    return True, ""


PUNKT_OK, PUNKT_REASON = _punkt_status()


# ---------------------------------------------------------------------------
# 1. 与上游逐字相同
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(V.UPSTREAM_SHA256))
def test_vendored_file_is_upstream_plus_import_edits(name):
    lines = (IFEVAL_DIR / name).read_text(encoding="utf-8").splitlines(keepends=True)
    begin = [i for i, ln in enumerate(lines) if ln.rstrip("\n") == V.NOTE_BEGIN]
    end = [i for i, ln in enumerate(lines) if ln.rstrip("\n") == V.NOTE_END]
    assert len(begin) == len(end) == 1 and begin[0] < end[0]
    assert lines[0] == "# coding=utf-8\n" and lines[1] == "# Copyright 2026 The Google Research Authors.\n"
    assert lines[begin[0] - 1] == "# limitations under the License.\n"  # 注释段紧跟在 Apache-2.0 声明之后
    note = "".join(lines[begin[0] : end[0] + 1])
    assert V.UPSTREAM_COMMIT in note and V.UPSTREAM_SHA256[name] in note
    body = lines[: begin[0]] + lines[end[0] + 1 :]
    for upstream, ours in V.IMPORT_EDITS[name]:
        assert f"`{upstream}` → `{ours}`" in note  # 每处改动都写在文件头
        hits = [i for i, ln in enumerate(body) if ln.rstrip("\n") == ours]
        assert len(hits) == 1, (name, ours)
        body[hits[0]] = upstream + "\n"
    restored = "".join(body)
    assert hashlib.sha256(restored.encode("utf-8")).hexdigest() == V.UPSTREAM_SHA256[name]
    assert not any(ln.startswith(("from absl", "import absl")) for ln in lines)  # 不依赖 absl-py
    assert "instruction_following_eval import" not in "".join(lines[end[0] + 1 :])


def test_license_is_upstream_apache2():
    raw = (IFEVAL_DIR / "LICENSE").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V.LICENSE_SHA256
    assert raw.lstrip().startswith(b"Apache License") and b"Version 2.0, January 2004" in raw


# ---------------------------------------------------------------------------
# 2. 上游自带的样例（原样），对照官方期望值
# ---------------------------------------------------------------------------


def _absl_shim() -> dict[str, types.ModuleType]:
    """只够加载上游两个测试文件的 absl.testing 替身：TestCase（加 assertLen）、named_parameters（按字典展开成方法）。"""

    class TestCase(unittest.TestCase):
        def assertLen(self, container, expected_len, msg=None):  # noqa: N802 — absltest 的名字
            self.assertEqual(len(container), expected_len, msg)

    def named_parameters(*cases):
        if len(cases) == 1 and isinstance(cases[0], list):
            cases = tuple(cases[0])

        def deco(fn):
            fn._shim_cases = cases
            return fn

        return deco

    class ParameterizedTestCase(TestCase):
        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)
            for name, fn in list(vars(cls).items()):
                cases = getattr(fn, "_shim_cases", None)
                if cases is None:
                    continue
                delattr(cls, name)
                for k, case in enumerate(cases):
                    kwargs = {a: b for a, b in case.items() if a != "testcase_name"}
                    setattr(cls, f"{name}_{k}", lambda self, _fn=fn, _kw=kwargs: _fn(self, **_kw))

    absl, testing = types.ModuleType("absl"), types.ModuleType("absl.testing")
    absltest, parameterized = types.ModuleType("absl.testing.absltest"), types.ModuleType("absl.testing.parameterized")
    absltest.TestCase, absltest.main = TestCase, lambda: None
    parameterized.TestCase, parameterized.named_parameters = ParameterizedTestCase, named_parameters
    absl.testing, testing.absltest, testing.parameterized = testing, absltest, parameterized
    return {"absl": absl, "absl.testing": testing, "absl.testing.absltest": absltest,
            "absl.testing.parameterized": parameterized}


def _load_upstream_cases() -> list[tuple[type, str]]:
    """原样执行上游测试文件（替身 absl；instruction_following_eval 指向 src.health.ifeval），返回 (测试类, 方法名)。"""
    import src.health.ifeval as pkg
    from src.health.ifeval import instructions, instructions_util

    fake = {**_absl_shim(), "instruction_following_eval": pkg,
            "instruction_following_eval.instructions": instructions,
            "instruction_following_eval.instructions_util": instructions_util}
    saved = {k: sys.modules.get(k) for k in fake}
    sys.modules.update(fake)
    cases = []
    try:
        for fname in sorted(UPSTREAM_TEST_SHA256):
            path = UPSTREAM_TESTS / fname
            module = types.ModuleType("_ifeval_upstream_" + fname.split(".")[0])
            module.__file__ = str(path)
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)  # noqa: S102
            for obj in vars(module).values():
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj.__module__ == module.__name__:
                    cases += [(obj, n) for n in sorted(dir(obj)) if n.startswith("test")]
    finally:  # 替身只在加载期间可见，不留在 sys.modules 里
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return cases


UPSTREAM_CASES = _load_upstream_cases()


def test_upstream_test_files_are_verbatim_and_loaded():
    for fname, digest in UPSTREAM_TEST_SHA256.items():
        assert hashlib.sha256((UPSTREAM_TESTS / fname).read_bytes()).hexdigest() == digest
    # instructions_test 41 个方法（展开参数后）+ instructions_util_test 7 个
    assert len(UPSTREAM_CASES) == 48
    assert sum(n.startswith(PUNKT_CASES) for _, n in UPSTREAM_CASES) == 9
    # 替身与别名只在加载期间可见：加载后 sys.modules 里没有它们（真装了 absl-py 时那是有 __file__ 的真模块）
    assert "instruction_following_eval" not in sys.modules
    for k in ("absl", "absl.testing", "absl.testing.absltest", "absl.testing.parameterized"):
        assert k not in sys.modules or getattr(sys.modules[k], "__file__", None), k


@pytest.mark.parametrize("cls, name", UPSTREAM_CASES, ids=[f"{c.__name__}.{n}" for c, n in UPSTREAM_CASES])
def test_upstream_case_matches_official_expectation(cls, name):
    if name.startswith(PUNKT_CASES) and not PUNKT_OK:
        pytest.skip(PUNKT_REASON)
    from langdetect import DetectorFactory

    DetectorFactory.seed = P.LANGDETECT_SEED  # 与健康检查的评分设置相同
    result = unittest.TestResult()
    cls(name).run(result)
    assert result.testsRun == 1
    assert result.wasSuccessful(), [(str(t), tb[-1500:]) for t, tb in result.failures + result.errors]


# ---------------------------------------------------------------------------
# 3. 包装层
# ---------------------------------------------------------------------------


def _item(ids, kwargs, prompt="Write something."):
    return {"id": 1, "prompt": prompt, "instruction_id_list": list(ids), "kwargs": list(kwargs)}


def test_strict_and_loose_differ_as_official():
    item = _item(["punctuation:no_comma"], [{}])
    r = S.ifeval_follow(item, "Sure, here it is:\nno commas in this line")
    assert r["strict"] == {"follow_all": False, "follow_list": [False]}
    assert r["loose"] == {"follow_all": True, "follow_list": [True]}  # loose 去掉首行后遵循


def test_empty_response_follows_nothing():
    item = _item(["punctuation:no_comma", "detectable_format:title"], [{}, {}])
    for text in ("", "   \n"):
        r = S.ifeval_follow(item, text)
        assert r["strict"]["follow_list"] == r["loose"]["follow_list"] == [False, False]


def test_prompt_level_needs_all_instructions():
    item = _item(["punctuation:no_comma", "detectable_format:title"], [{}, {}])
    assert S.ifeval_follow(item, "<<Rain>>\nDrops fall on the roof")["strict"] == {"follow_all": True,
                                                                                     "follow_list": [True, True]}
    assert S.ifeval_follow(item, "<<Rain>>\nDrops, then more drops")["strict"] == {"follow_all": False,
                                                                                    "follow_list": [False, True]}


def test_kwargs_are_passed_through():
    item = _item(["combination:repeat_prompt", "keywords:letter_frequency"],
                 [{"prompt_to_repeat": "Write a haiku about rain."}, {"letter": "z", "let_frequency": 2, "let_relation": "at least"}])
    r = S.ifeval_follow(item, "Write a haiku about rain. Zigzag rain, fizzing puddles")
    assert r["strict"]["follow_list"] == [True, True]
    r = S.ifeval_follow(item, "Haiku: rain falls on the zinc roof")
    assert r["strict"]["follow_list"] == [False, False]
    assert item["kwargs"][0] == {"prompt_to_repeat": "Write a haiku about rain."}  # 调用方的 kwargs 不被改动


def test_langdetect_seed_is_fixed():
    from langdetect import DetectorFactory

    DetectorFactory.seed = None
    item = _item(["language:response_language"], [{"language": "en"}])
    first = [S.ifeval_follow(item, "ok si")["strict"]["follow_all"] for _ in range(5)]
    assert DetectorFactory.seed == P.LANGDETECT_SEED == 0
    assert len(set(first)) == 1  # 固定种子后同一回答的判定不变


def test_all_item_instructions_are_supported():
    from src.health.ifeval import instructions_registry

    items, _ = load_items()
    seen = set()
    for x in items["items"]["ifeval"]:
        for iid, kw in zip(x["instruction_id_list"], x["kwargs"]):
            assert iid in instructions_registry.INSTRUCTION_DICT, iid
            instructions_registry.INSTRUCTION_DICT[iid](iid).build_description(**copy.deepcopy(kw))
            seen.add(iid)
    assert len(seen) == 23


# ---------------------------------------------------------------------------
# 4. punkt_tab 的来源
# ---------------------------------------------------------------------------


def test_punkt_missing_is_reported(tmp_path, monkeypatch):
    import nltk

    monkeypatch.setattr(nltk.data, "path", [str(tmp_path)])
    with pytest.raises(S.PunktDataError, match="NLTK_DATA"):
        S.punkt_provenance()
    with pytest.raises(S.PunktDataError):
        S.require_punkt(pinned=False)


def test_punkt_hash_is_checked(monkeypatch):
    import nltk

    monkeypatch.setattr(nltk.data, "path", [str(SYNTHETIC_NLTK_DATA)])
    prov = S.punkt_provenance()
    assert "punkt_tab" in prov["path"] and str(SYNTHETIC_NLTK_DATA) in prov["path"]
    assert prov["files_sha256"] == {k: hashlib.sha256(b"").hexdigest() for k in P.PUNKT_TAB_ENGLISH_SHA256}
    assert prov["matches_pinned"] is False
    with pytest.raises(S.PunktDataError, match="哈希"):
        S.require_punkt(pinned=True)
    assert S.require_punkt(pinned=False) == prov


@pytest.mark.skipif(not PUNKT_OK, reason=PUNKT_REASON)
def test_real_punkt_matches_pin():
    prov = S.require_punkt(pinned=True)
    assert prov["matches_pinned"] is True
