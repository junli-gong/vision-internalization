"""受限字母读数的字母表（D36-6，D36-4 / N14 → (c)；configs/e1_prompts.json 的 letter_readout；DEV-0017 任务书 §1）。

选择题与点对任务在 eval、dev 上只生成 1 个 token，allowed_token_ids = 本题选项字母的全部写法（配置的 spellings：不带空格
"A" 与带前导空格 " A"）。greedy 下生成的 token 就是这些 token 里 logit 最大的一个，等价于"每个字母取两种写法中较大的
logit，再在字母之间取最大"。解析不经过文本：生成的 token id 按字母表映射回字母。

字母表只用分词器的通用接口 encode(text, add_special_tokens=False) 与 decode 构造，与模型家族无关；分词器取自适配器的
tokenizer 属性（三个适配器都有）。规则：
- 每种写法 encode 后恰好 1 个 token 才收；不是单 token 的写法跳过并记在 skipped 里；
- 同一个字母的两种写法撞到同一个 id 时只记一次；
- 某个字母一种写法都收不到，或两个字母撞到同一个 id：报错（LetterTableError），不静默退化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


class LetterTableError(ValueError):
    """字母表构造不出来：某个字母没有单 token 的写法，或两个字母撞到同一个 token。"""


@dataclass(frozen=True)
class LetterTable:
    """letters：表里的字母（排序）；entries：字母 → ((写法, token id, decode 回来的文本), …)；skipped：不是单 token 的写法。"""

    letters: tuple[str, ...]
    entries: Mapping[str, tuple[tuple[str, int, str], ...]]
    skipped: tuple[tuple[str, str, tuple[int, ...]], ...] = ()
    _by_id: Mapping[int, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_id: dict[int, str] = {}
        for lt in self.letters:
            if not self.entries.get(lt):
                raise LetterTableError(f"字母 {lt!r} 没有单 token 的写法")
            for spelling, tid, _ in self.entries[lt]:
                if by_id.get(tid, lt) != lt:
                    raise LetterTableError(f"token {tid} 同时是字母 {by_id[tid]!r} 与 {lt!r} 的写法（{spelling!r}）")
                by_id[tid] = lt
        object.__setattr__(self, "_by_id", by_id)

    def allowed_ids(self, letters: Iterable[str]) -> list[int]:
        """这些字母的全部写法的 token id（升序、不重复），即 generate 的 allowed_token_ids。"""
        out: set[int] = set()
        for lt in letters:
            if lt not in self.entries:
                raise KeyError(f"字母表里没有 {lt!r}；表里有 {self.letters}")
            out.update(tid for _, tid, _ in self.entries[lt])
        if not out:
            raise LetterTableError("字母集合为空")
        return sorted(out)

    def letter_of(self, token_id: int) -> str | None:
        return self._by_id.get(int(token_id))

    def spelling_of(self, token_id: int) -> str | None:
        lt = self.letter_of(token_id)
        if lt is None:
            return None
        return next(sp for sp, tid, _ in self.entries[lt] if tid == int(token_id))

    def as_dict(self) -> dict[str, Any]:
        """写进 config.yaml 的形式（运行身份的一部分：换了分词器，续跑就会拒绝）。"""
        return {"letters": list(self.letters),
                "entries": {lt: [{"spelling": sp, "token_id": tid, "decoded": dec} for sp, tid, dec in self.entries[lt]]
                            for lt in self.letters},
                "skipped": [{"letter": lt, "spelling": sp, "token_ids": list(ids)} for lt, sp, ids in self.skipped]}


def build_letter_table(tokenizer: Any, letters: Iterable[str], spellings: Sequence[str]) -> LetterTable:
    """按配置的写法（如 ("{letter}", " {letter}")）给每个字母找单 token 的 id。规则见模块说明。"""
    letters = sorted(set(letters))
    if not letters:
        raise LetterTableError("没有字母")
    entries: dict[str, tuple[tuple[str, int, str], ...]] = {}
    skipped: list[tuple[str, str, tuple[int, ...]]] = []
    for lt in letters:
        if not (isinstance(lt, str) and len(lt) == 1 and lt.isalpha()):
            raise LetterTableError(f"字母必须是单个字母字符，收到 {lt!r}")
        got: list[tuple[str, int, str]] = []
        for tpl in spellings:
            sp = tpl.format(letter=lt)
            ids = tuple(int(i) for i in tokenizer.encode(sp, add_special_tokens=False))
            if len(ids) != 1:
                skipped.append((lt, sp, ids))
                continue
            if any(tid == ids[0] for _, tid, _ in got):
                continue
            got.append((sp, ids[0], tokenizer.decode([ids[0]])))
        entries[lt] = tuple(got)
    return LetterTable(letters=tuple(letters), entries=entries, skipped=tuple(skipped))


__all__ = ["LetterTable", "LetterTableError", "build_letter_table"]
