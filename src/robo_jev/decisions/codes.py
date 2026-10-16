"""The answer-code pool of the many-option format (Task v0.8a) — the one place the package keeps it.

**The pool** (:data:`CODE_POOL`, 255 codes) is the Hub runner's ``CODES[:255]`` (``robo_jev/hf/robo_jev_decide.py``): the capital letters
A–Z (the codes every earlier version trained), then the first 229 two-letter codes of the runner's list — the two-letter codes AA … ZZ in
lexicographic order that are one token both alone (``" AB"``) and as the last token of ``"Answer: AB"`` with the pinned tokenizer, less the
runner's stoplist of words and abbreviations — AA, AB, AE, … MO. It is hard-coded so that the codes never depend on the tokenizer at hand;
:func:`verify_pool_codes` checks every code a prompt shows (a test checks all 255 with the pinned tokenizer and that the pool equals the
runner's ``CODES[:255]``).

**How the prompt uses it** (:mod:`robo_jev.decisions.prompt`):

* a question shows its options with the codes in pool order, A, B, … — for up to 26 options exactly as before (the letters through
  :func:`robo_jev.model.zero_shot.verify_codes`, byte-identical prompts), past 26 the two-letter codes (:func:`sequential_codes`; only a
  record that opts in to more than 26 options can have them — :data:`robo_jev.decisions.records.OPTION_RANGE_MARK_255`);
* or with **assigned codes** — a record's ``meta.codes[question]`` lists one pool code per shown position, in display order (position j of
  every listing shows ``codes[j]``); the training input of v0.8 draws them at random from the pool for a share of its choice questions, so
  that every pool code is trained as an answer (:func:`assigned_codes`, :func:`codes_problems`).

The readout reads the LM-head rows of the shown codes' token ids, whatever the codes are.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "CODE_POOL",
    "LETTERS",
    "POOL_SIZE",
    "assigned_codes",
    "codes_problems",
    "pool_index",
    "sequential_codes",
    "verify_pool_codes",
]

#: The 26 capital letters — the codes of every earlier version (a prompt with ≤ 26 options and no assigned codes shows exactly these).
LETTERS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#: The 255 answer codes (module docstring): A–Z, then the first 229 two-letter codes of the Hub runner's list.
CODE_POOL: tuple[str, ...] = LETTERS + tuple("""
    AA AB AE AF AG AJ AK AL AO AP AQ AR AU AV AZ
    BA BB BD BF BG BH BI BJ BK BL BM BN BO BP BR BT BU BV BW BX
    CA CB CC CE CF CG CH CI CJ CK CL CN CP CR CS CT CU CV CW CX CY CZ
    DA DD DE DF DG DH DI DK DL DM DN DP DS DT DU DV DW DX DY
    EA EB EC ED EE EF EI EJ EK EL EM EN EO EP EQ ER ES ET EW EX EZ
    FA FB FC FD FE FF FG FH FI FK FL FM FN FO FP FR FS FW FX FY
    GA GC GD GE GF GG GH GI GK GL GM GN GP GR GS GT GU GV GW GX
    HB HC HD HF HG HH HK HL HN HS HT HU HV HW HY
    IA IB IC IG IH II IK IL IM IO IU IV IW IX
    JA JB JC JD JE JJ JK JL JM JO JP JS JT JU JV JW
    KA KC KD KE KF KH KI KK KL KO KP KR KS KT KU KV KY
    LC LD LE LF LG LH LI LJ LL LM LN LO LP LR LS LT LU LV LW LX LY LZ
    MC MF MH MI MJ MK MN MO
""".split())
POOL_SIZE = len(CODE_POOL)
_INDEX = {code: i for i, code in enumerate(CODE_POOL)}
assert POOL_SIZE == 255 and len(_INDEX) == 255


def pool_index(code: str) -> int:
    """The code's place in :data:`CODE_POOL` (``KeyError`` for a code outside it)."""
    return _INDEX[str(code)]


def sequential_codes(count: int) -> list[str]:
    """The first `count` pool codes — what a question with `count` options shows without assigned codes."""
    if not 1 <= int(count) <= POOL_SIZE:
        raise ValueError(f"{count} options — the code pool has {POOL_SIZE} codes")
    return list(CODE_POOL[: int(count)])


def codes_problems(codes: Any, count: int) -> list[str]:
    """Every way a list of assigned codes fails for a question with `count` options (empty = fine): a list of `count` distinct codes, each
    in :data:`CODE_POOL`."""
    if not isinstance(codes, (list, tuple)):
        return [f"assigned codes must be a list (got {type(codes).__name__})"]
    problems = []
    if len(codes) != int(count):
        problems.append(f"{len(codes)} assigned codes for {count} options")
    outside = [c for c in codes if not isinstance(c, str) or c not in _INDEX]
    if outside:
        problems.append(f"codes outside the pool: {outside[:5]!r}")
    if len(set(map(str, codes))) != len(codes):
        problems.append("assigned codes repeat")
    return problems


def assigned_codes(record: Mapping[str, Any]) -> dict[str, list[str]]:
    """The record's assigned codes (``meta.codes``: question name → codes in display order) — empty when it has none. Only the shape is
    read here; :func:`robo_jev.decisions.records.validate_record` checks the codes against the questions."""
    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    codes = meta.get("codes")
    if not isinstance(codes, Mapping):
        return {}
    return {str(name): [str(c) for c in value] for name, value in codes.items() if isinstance(value, (list, tuple))}


def verify_pool_codes(tokenizer: Any, codes: Sequence[str]) -> list[int]:
    """The token ids of `codes` — each must be a pool code that is one token alone (``" AB"``) and the last token of ``"Answer: AB"``
    (the answer boundary); otherwise ``ValueError``."""
    ids = []
    for code in codes:
        if code not in _INDEX:
            raise ValueError(f"code {code!r} is not in the code pool")
        alone = tokenizer.encode(" " + code, add_special_tokens=False).ids
        boundary = tokenizer.encode("Answer: " + code, add_special_tokens=False).ids
        if len(alone) != 1 or boundary[-1] != alone[0]:
            raise ValueError(f"code {code!r} is not one token at the answer boundary (alone={alone}, boundary tail={boundary[-3:]})")
        ids.append(int(alone[0]))
    return ids
