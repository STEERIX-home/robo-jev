"""교란 시험 세트 T6: 변형 12가지가 정답을 바꾸지 않는다(좌우 거울상은 대응으로) (Task v0b 지시서 §4 T6·§8).

기준 몇 개를 만들어(testsets._t6와 같은 길) 변형마다: 같은 것을 묻는 질문(정답 조리법의 정체 — 거울상은 선택지·사건 이름을 뒤집어)의 정답이
**정확히** 같다(choice는 정식 선택지로 맞춰서). 거울상은 사후 세계를 공통 난수로 뒤집으므로 다르면 굴리기가 좌우 대칭이 아니라는 뜻이다.
변형 자체도 확인한다: 순서 뒤집기·섞기는 순서만, 숫자 표기는 소수의 뒤 0만, 서식 잡음은 공백·글머리표만, 사실 순서는 줄의 순서만 바꾼다.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.holdout import load_holdout
from robo_jev.simgen.perturb import VARIANTS, mirror_canon, strip_trailing_zeros
from robo_jev.simgen.testsets import _t6

N_BASES = 27


@pytest.fixture(scope="module")
def groups():
    records = list(_t6(default_config(), N_BASES, "test", load_holdout(), [], stride=1))      # 묶음 아홉을 세 번씩
    out: dict[str, dict] = {}
    for r in records:
        t6 = r["meta"]["t6"]
        base_id = r["id"] if t6["variant"] == "base" else t6["base"]
        out.setdefault(base_id, {})[t6["variant"]] = r
    return out


def _canon_probs(record: dict, name: str, mirrored: bool = False) -> dict:
    """choice 정답을 정식 값으로(행동·다시 볼까: 정식 선택지, 상태: 값 목록, 규칙: 갈래)."""
    label = record["labels"][name]
    meta = record["meta"]
    if name in meta["options"]:
        mapping = meta["options"][name]
        out = {}
        for key, p in label["probabilities"].items():
            value = mapping[key]
            if isinstance(value, list):
                value = tuple(value)
            elif mirrored:
                value = mirror_canon(value)
            out[value] = p
        return out
    classes = meta["basis"][name]["classes"]
    return {classes[key]: p for key, p in label["probabilities"].items()}


def _same_label(base: dict, b_name: str, variant: dict, v_name: str, mirrored: bool) -> None:
    lb, lv = base["labels"][b_name], variant["labels"][v_name]
    assert lb["type"] == lv["type"]
    if lb["type"] == "noul":
        assert lv["p_yes"] == pytest.approx(lb["p_yes"], abs=1e-9), (variant["id"], v_name)
    elif lb["type"] == "score":
        assert lv["value"] == pytest.approx(lb["value"], abs=1e-9), (variant["id"], v_name)
    else:
        pb, pv = _canon_probs(base, b_name), _canon_probs(variant, v_name, mirrored)
        assert set(pb) == set(pv), (variant["id"], v_name)
        for value, p in pb.items():
            assert pv[value] == pytest.approx(p, abs=1e-9), (variant["id"], v_name, value)


def test_every_base_has_all_twelve_variants(groups):
    assert len(groups) == N_BASES
    for base_id, members in groups.items():
        assert set(members) == {"base", *VARIANTS}, base_id
        assert all(m["meta"]["scene"] != "p1_head_on" for m in members.values())
    assert len({m["base"]["meta"]["group"] for m in groups.values()}) == 9


def test_labels_are_invariant_under_every_variant(groups):
    """매핑된 질문의 정답이 정확히 같다. 행동 질문과 핵심 질문은 언제나 매핑된다(거울상 포함)."""
    unmapped = Counter()
    for members in groups.values():
        base = members["base"]
        for variant, record in members.items():
            if variant == "base":
                continue
            mapping = record["meta"]["t6"]["map"]
            for v_name, b_name in mapping.items():
                if b_name is None:
                    unmapped[variant] += 1
                    continue
                _same_label(base, b_name, record, v_name, variant == "mirror")
            assert mapping[record["meta"]["recommendation"]["question"]] == base["meta"]["recommendation"]["question"], (variant, record["id"])
            assert mapping[record["meta"]["key_question"]] == base["meta"]["key_question"], (variant, record["id"])
    # 한 가지 더 정하기만 새 질문(기준에 없는 것)을 하나씩 더한다; 나머지는 거의 다 매핑된다
    assert unmapped["one_more_decision"] <= N_BASES
    assert sum(v for k, v in unmapped.items() if k not in ("one_more_decision", "mirror")) == 0
    assert unmapped["mirror"] <= N_BASES


def test_one_more_decision_adds_exactly_one_question(groups):
    for members in groups.values():
        base, extra = members["base"], members["one_more_decision"]
        assert set(base["request"]["questions"]) < set(extra["request"]["questions"]) or len(extra["request"]["questions"]) == len(base["request"]["questions"])
        assert len(extra["request"]["questions"]) - len(base["request"]["questions"]) in (0, 1)


def test_reordering_changes_only_the_order(groups):
    for members in groups.values():
        base = members["base"]
        for variant in ("order_reversal", "shuffle"):
            record = members[variant]
            assert record["request"]["state"] == base["request"]["state"]
            bq, vq = base["request"]["questions"], record["request"]["questions"]
            assert set(bq) == set(vq) and all(bq[n] == vq[n] for n in bq)       # 같은 질문(딕셔너리 비교는 순서를 보지 않는다)
            if variant == "order_reversal":
                assert list(vq) == list(bq)[::-1]
                for n, q in bq.items():
                    if isinstance(q.get("criteria"), dict):
                        assert list(vq[n]["criteria"]) == list(q["criteria"])[::-1]
            assert record["labels"] == base["labels"]


def test_number_notation_only_drops_trailing_zeros(groups):
    changed = 0
    for members in groups.values():
        base, record = members["base"], members["number_notation"]
        assert record["request"]["state"] == strip_trailing_zeros(base["request"]["state"])
        changed += record["request"]["state"] != base["request"]["state"]
    assert changed >= N_BASES // 2


def _words(text: str) -> list[str]:
    return re.sub(r"[•*–\-\s]+", " ", text).split()


def test_format_noise_keeps_every_word_and_number(groups):
    for members in groups.values():
        base, record = members["base"], members["format_noise"]
        assert _words(record["request"]["state"]) == _words(base["request"]["state"])


def test_fact_order_keeps_the_same_facts(groups):
    for members in groups.values():
        base, record = members["base"], members["fact_order"]
        if base["meta"]["style"] in ("facts", "planner", "log"):
            assert Counter(base["request"]["state"].split("\n")) == Counter(record["request"]["state"].split("\n"))


def test_units_change_the_text_but_not_the_numbers_behind_it(groups):
    for members in groups.values():
        base, record = members["base"], members["units"]
        assert record["meta"]["units"] != base["meta"]["units"]
        assert record["request"] != base["request"]       # 상태 글(거리·속도·시간) 또는 선택지 설명(시간)의 단위 — 안착 측정(mm·N·°)만 있는 글은 그대로
        assert record["meta"]["holdout_vars"] == base["meta"]["holdout_vars"]


def test_paraphrase_variants_change_the_wording(groups):
    differs = Counter()
    for members in groups.values():
        base = members["base"]
        differs["state"] += members["state_paraphrase"]["request"]["state"] != base["request"]["state"]
        bq, iq = base["request"]["questions"], members["instruction_paraphrase"]["request"]["questions"]
        differs["instructions"] += any(bq[n]["instructions"] != iq[n]["instructions"] for n in bq if n in iq)
        oq = members["option_paraphrase"]["request"]["questions"]
        differs["options"] += any(bq[n].get("criteria") != oq[n].get("criteria") for n in bq if n in oq and isinstance(bq[n].get("criteria"), dict))
        kq = members["key_change"]["request"]["questions"]
        differs["keys"] += members["key_change"]["meta"]["key_style"] != base["meta"]["key_style"]
        assert set(kq) == set(bq)
    assert differs["state"] == N_BASES and differs["keys"] == N_BASES
    assert differs["instructions"] >= N_BASES - 1 and differs["options"] >= N_BASES // 2


def test_the_mirror_flips_every_side(groups):
    for members in groups.values():
        base, record = members["base"], members["mirror"]
        bm, mm = base["meta"], record["meta"]
        if bm["sides"] is not None:
            assert mm["sides"] == bm["sides"][::-1]
        if bm.get("corner"):
            assert {bm["corner"]["side"], mm["corner"]["side"]} == {"left", "right"}
        if bm["recommendation"]["canonical"]:
            assert mm["recommendation"]["canonical"] == mirror_canon(bm["recommendation"]["canonical"])
