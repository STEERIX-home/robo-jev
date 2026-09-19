"""비로봇 단일 요청 레코드 생성기 (docs/04 §2·§3).

    python -m robo_jev.data.generate --config configs/data/pilot.yaml \\
        --count 2000 --seed 17 --output artifacts/datasets/d1/single

`DIR/records.jsonl`과 `DIR/manifest.json`을 쓴다. 같은 `--count`·`--seed`·설정이면
언제나 **같은 바이트**가 나온다 (manifest에 시각을 넣지 않는 이유다).

레코드 한 건을 만드는 순서는 이렇다.

1. 분야와 장면 계열(`origin_group`)을 정하고 **생성 전에** split을 배정한다 (docs/04 §5).
2. 장면(사실)을 만들고, 그 장면에서 질문 명세를 뽑는다.
3. 명세를 표현으로 그린다. 라벨은 (사실, 명세)의 순수 함수다 (:mod:`robo_jev.data.domains`).
4. **마지막에** 후보의 순서와 id를 섞는다. 정답 위치 편향은 여기 한 곳에서만 막는다.
5. 파생본(번역·후보 재배열)은 부모의 origin group을 쓰므로 split을 그대로 승계한다.

teacher 모델은 쓰지 않는다. 모든 라벨은 규칙·solver가 낸 검증 가능한 결정적 라벨이다.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robo_jev.contracts import SCHEMA_SINGLE_REQUEST
from robo_jev.data.domains import DOMAINS, NONE_ID, QuestionSpec, Rendered, begin_phrasing, concepts_for, take_phrasing
from robo_jev.data.split import CONCEPT_TAG, TEMPLATE_TAG, SplitPolicy, ood_split

__all__ = [
    "DEFAULT_CONFIG",
    "GENERATOR_VERSION",
    "PILOT_CONFIG",
    "contrast_counts",
    "deletion_outcome",
    "generate_records",
    "load_config",
    "main",
    "semantic_answers",
    "write_dataset",
]

GENERATOR_VERSION = "gen-single-v0.1.0"
MANIFEST_VERSION = "manifest-v0"

#: 함께 배포하는 설정 파일. `load_config(PILOT_CONFIG) == DEFAULT_CONFIG`이어야 한다.
PILOT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "data" / "pilot.yaml"

#: 시작값은 docs/04 §2에서 온다. 설정 파일로 덮어쓸 수 있다.
DEFAULT_CONFIG: dict[str, Any] = {
    "version": "pilot-v0.3",
    # 비로봇 상태의 분야 비중. docs/04 §2는 dom/workflow/rules를 15/15/10으로 두고
    # 나머지를 색·위치·영역 문제가 채운다 — pilot에서는 넷을 고르게 둔다.
    "domains": {"spatial": 25, "dom": 25, "workflow": 25, "rules": 25},
    "question_types": {"choice": 60, "boolean": 25, "ordinal": 15},
    "questions_per_state": {1: 10, 4: 20, 8: 50, 16: 20},
    "languages": {"ko": 70, "en": 30},
    # 파생본 비율(%). 번역본과 후보 재배열본은 부모의 group·split을 승계한다. `contrast`는 사실 하나만 바꿔 질문 하나의
    # 라벨을 뒤집는 대조 sibling(docs/04 §3; 삭제 검사를 지난 쌍만 남긴다)이며 기본 레코드마다 하나(100 %)다.
    "derivations": {"paraphrase": 20, "reorder": 10, "contrast": 100},
    # 봉인 문구 변형(`split.holdout_templates`)의 추첨 몫(%): 그 변형이 든 표에서 봉인 변형을 이 몫만큼만 뽑는다(나머지는 다른
    # 변형이 고르게). 봉인 변형을 고르게 뽑으면 계열의 13~19 %가 템플릿 holdout에 걸려 OOD가 목표(≈10~15 %)를 넘는다 —
    # 봉인 id(한국어)는 그대로 두고 비중으로 맞춘다 (docs/04 §5).
    "sealed_phrasing_share": 10,
    "split": {
        "weights": {"train": 70, "dev": 10, "calibration": 10, "test": 10},
        "holdout_groups": [],
        "holdout_prefixes": [],
        "holdout_domains": [],
        # 봉인 holdout (docs/04 §5 표, 계약 v0.3): 문구 템플릿 변형 계열과 분야마다 개념 하나. 레코드의 provenance
        # (`phrasing`·`concepts`)와 맞대며, 한 계열의 레코드 중 하나라도 걸리면 계열 전체가 OOD(ood_dev/ood_test)다.
        "holdout_templates": ["spatial.distance.ko#1", "dom.needs_input.ko#1", "workflow.load.ko#1", "rules.conflict.ko#1"],
        "holdout_concepts": ["spatial:goal-zone:zoneC", "dom:reveal", "workflow:resource_offline", "rules:escort-policy"],
    },
}


# --------------------------------------------------------------------------
# 설정
# --------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정을 읽어 기본값 위에 얹는다."""
    import yaml  # 설정 파일을 읽을 때만 필요하다

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 설정은 매핑이어야 한다")
    return _merge(DEFAULT_CONFIG, raw)


#: 안쪽 키를 하나씩 덮어쓰는 절. 나머지 절(비중 표)은 통째로 갈아 끼운다 —
#: `domains: {rules: 100}`은 "rules만 쓴다"는 뜻이지 "rules를 더한다"가 아니다.
_SECTION_KEYS = ("split",)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    # 기본값을 깊은 복사한다. 얕게 복사하면 돌려준 설정의 안쪽 dict·list가 모듈 전역
    # `DEFAULT_CONFIG`와 같은 객체라, 부르는 쪽이 고치면 전역이 따라 바뀐다.
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        section = key in _SECTION_KEYS and isinstance(value, Mapping)
        if section and isinstance(merged.get(key), Mapping):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _resolve(config: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = _merge(DEFAULT_CONFIG, config or {})
    for section in ("domains", "question_types", "questions_per_state", "languages"):
        weights = resolved[section]
        if not weights or any(weight <= 0 for weight in weights.values()):
            raise ValueError(f"{section}: 비중은 양수여야 한다 (받은 값: {weights})")
    share = resolved.get("sealed_phrasing_share")
    if share is not None and not 0 < float(share) < 100:
        raise ValueError(f"sealed_phrasing_share: 0과 100 사이의 퍼센트여야 한다 (받은 값: {share})")
    unknown = sorted(set(resolved["domains"]) - set(DOMAINS))
    if unknown:
        raise ValueError(f"domains: 없는 분야다: {unknown} (가능: {sorted(DOMAINS)})")
    return resolved


def config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------
# 추첨 도우미
# --------------------------------------------------------------------------


def _weighted(rng: random.Random, weights: Mapping[Any, int]) -> Any:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]


def _next_domain(weights: Mapping[str, int], produced: Counter) -> str:
    """목표 비중에 가장 못 미친 분야. 난수를 쓰지 않아 비중이 흔들리지 않는다."""
    order = list(weights)
    return min(order, key=lambda name: (produced[name] / weights[name], order.index(name)))


def _select_specs(
    pool: Sequence[QuestionSpec],
    count: int,
    mix: Mapping[str, int],
    rng: random.Random,
) -> list[QuestionSpec]:
    """질문 타입 비중에 맞춰 명세를 뽑는다. 한 타입이 동나면 남은 타입에서 채운다."""
    remaining: dict[str, list[QuestionSpec]] = {
        question_type: [spec for spec in pool if spec.type == question_type] for question_type in mix
    }
    for specs in remaining.values():
        rng.shuffle(specs)
    chosen: list[QuestionSpec] = []
    for _ in range(count):
        question_type = _weighted(rng, mix)
        if not remaining[question_type]:
            question_type = max(remaining, key=lambda name: (len(remaining[name]), name))
        if not remaining[question_type]:
            break  # 장면이 줄 수 있는 질문을 다 썼다
        chosen.append(remaining[question_type].pop())
    return chosen


# --------------------------------------------------------------------------
# 레코드 조립
# --------------------------------------------------------------------------


def _shuffle_candidates(record: dict, rng: random.Random) -> None:
    """후보의 순서와 id를 섞는다 (docs/04 §3: 정답 위치 편향 제거).

    `ordinal`은 계약상 value 오름차순이어야 하므로 순서를 건드리지 않는다. `choice`는
    순서와 id를 모두 바꾸고, 그 질문의 라벨도 새 id로 다시 적는다.
    """
    labels = {label["question_id"]: label for label in record["labels"]}
    for question in record["request"]["questions"]:
        if question["type"] == "ordinal":
            continue
        criteria = question["criteria"]
        rng.shuffle(criteria)
        if question["type"] != "choice":
            continue
        numbers = rng.sample(range(10, 100), k=len(criteria))
        renamed = {}
        for criterion, number in zip(criteria, numbers):
            renamed[criterion["id"]] = f"c{number}"
            criterion["id"] = f"c{number}"
        label = labels.get(question["id"])
        if label is None:
            continue
        if "candidate_ids" in label:
            label["candidate_ids"] = [renamed[value] for value in label["candidate_ids"]]
        if isinstance(label.get("answer"), str):
            label["answer"] = renamed[label["answer"]]


def _render_record(
    *,
    domain: Any,
    scene: dict,
    specs: Sequence[QuestionSpec],
    group: str,
    split: str,
    request_id: str,
    language: str,
    seed: int,
    family_index: int,
    wording_seed: str,
    shuffle_seed: str,
    config: Mapping[str, Any],
    derived_from: str | None = None,
    derivation: str | None = None,
    contrast: dict | None = None,
    contrast_evidence: dict | None = None,
) -> dict:
    wording_rng = random.Random(wording_seed)
    # 봉인 변형은 드물게 뽑는다(`sealed_phrasing_share`) — 계열이 통째로 OOD로 가는 몫을 비중으로 맞춘다 (docs/04 §5).
    begin_phrasing(sealed=config["split"].get("holdout_templates") or (), share=config.get("sealed_phrasing_share"))
    state = domain.state(scene, wording_rng, language)
    rendered: list[Rendered] = [
        domain.render(scene, spec, wording_rng, language) for spec in specs
    ]
    phrasing = take_phrasing()
    concepts = sorted({concept for spec in specs for concept in concepts_for(domain.name, scene, spec)})

    questions = [item.question for item in rendered]
    labels = [item.label for item in rendered if item.label is not None]
    masked = {
        item.question["id"]: item.mask_reason for item in rendered if item.label is None
    }
    variants = sorted({tag for item in rendered for tag in item.variants})
    if derivation:
        variants = sorted(set(variants) | {derivation})

    record = {
        "schema_version": SCHEMA_SINGLE_REQUEST,
        "origin_group": group,
        "split": split,
        "request": {"request_id": request_id, "state": state, "questions": questions},
        "labels": labels,
        "provenance": {
            "generator": GENERATOR_VERSION,
            "config_version": config["version"],
            "domain": domain.name,
            "template": scene["template"],
            "origin_group": group,
            "family": family_index,
            "seed": seed,
            "language": language,
            "variants": variants,
            # 봉인 holdout의 근거 (docs/04 §5): 이 레코드가 쓴 문구 템플릿 변형과 다루는 개념. 생성기가 계열 단위로
            # `split.holdout_templates`·`holdout_concepts`와 맞대고, 걸린 이유는 `holdout`에 적는다 (아니면 빈 목록).
            "phrasing": phrasing,
            "concepts": concepts,
            "holdout": [],
            "rules": sorted({label["source"] for label in labels}),
            **({"derived_from": derived_from, "derivation": derivation} if derived_from else {}),
            # 대조 쌍 (docs/04 §3): sibling에는 `{role: sibling, sibling_id: 기본 레코드, focus_field, flipped_question, deletion}`,
            # 기본 레코드에는 `{role: base, sibling_id: sibling}`(없으면 `sibling_id: None`과 이유). 생성 뒤 채운다.
            "contrast": copy.deepcopy(contrast),
        },
        "evidence": {
            "rule_trace": [item.trace for item in rendered],
            "masked": masked,
            # 규칙이 읽은 사실 원본. 계열 안의 레코드가 같은 객체를 나눠 갖지 않도록 복사한다.
            "scene": copy.deepcopy(scene),
            **({"contrast": copy.deepcopy(contrast_evidence)} if contrast_evidence else {}),
        },
        "usage": {"questions_used": [question["id"] for question in questions], "commands": []},
    }
    _shuffle_candidates(record, random.Random(shuffle_seed))
    return record


def _build_family(
    *,
    domain_name: str,
    family_index: int,
    seed: int,
    config: Mapping[str, Any],
    policy: SplitPolicy,
) -> list[dict]:
    """장면 계열 하나 — 기본 레코드와 그 파생본들."""
    domain = DOMAINS[domain_name]
    family_rng = random.Random(f"{seed}:{domain_name}:{family_index}")
    # 장면 종류의 비중(`template_weights`)이 있는 분야는 그 비중으로 — 봉인 개념의 근원이 되는 장면을 드물게 둔다 (docs/04 §5).
    weights = getattr(domain, "template_weights", None)
    template = _weighted(family_rng, dict(zip(domain.templates, weights))) if weights else family_rng.choice(domain.templates)
    group = f"{domain_name}/{template}/{family_index:04d}"
    split = policy.assign(group)  # 생성 전 배정. 파생본은 이 값을 그대로 쓴다.

    scene = domain.make_scene(family_rng, template)
    question_count = _weighted(family_rng, config["questions_per_state"])
    specs = _select_specs(domain.pool(scene), question_count, config["question_types"], family_rng)
    language = _weighted(family_rng, config["languages"])

    base_id = f"{domain_name}-{family_index:04d}-0"
    shared = {
        "domain": domain,
        "scene": scene,
        "specs": specs,
        "group": group,
        "split": split,
        "seed": seed,
        "family_index": family_index,
        "config": config,
    }
    records = [
        _render_record(
            request_id=base_id,
            language=language,
            wording_seed=f"{seed}:{group}:w0",
            shuffle_seed=f"{seed}:{group}:s0",
            **shared,
        )
    ]

    derivations = config["derivations"]
    if family_rng.random() < derivations["paraphrase"] / 100:
        # 번역본: 같은 사실·같은 명세를 다른 언어로 다시 그린다. 라벨은 다시 계산된다.
        records.append(
            _render_record(
                request_id=f"{domain_name}-{family_index:04d}-1",
                language="en" if language == "ko" else "ko",
                wording_seed=f"{seed}:{group}:w1",
                shuffle_seed=f"{seed}:{group}:s1",
                derived_from=base_id,
                derivation="paraphrase",
                **shared,
            )
        )
    if family_rng.random() < derivations["reorder"] / 100:
        # 재배열본: 문장은 그대로, 후보 순서와 id만 바뀐다.
        records.append(
            _render_record(
                request_id=f"{domain_name}-{family_index:04d}-2",
                language=language,
                wording_seed=f"{seed}:{group}:w0",
                shuffle_seed=f"{seed}:{group}:s2",
                derived_from=base_id,
                derivation="reorder",
                **shared,
            )
        )
    if family_rng.random() < derivations.get("contrast", 0) / 100:
        # 대조 sibling: 사실 하나만 바꿔 질문 하나의 라벨을 뒤집는다 (docs/04 §3, analysis-nimble §3-2). 같은 문구·같은 후보
        # 배열(같은 wording·shuffle seed)이라 쌍의 차이는 그 사실뿐이다. 삭제 검사(사실을 지우면 라벨이 마스크·"해당 없음")를
        # 지나는 쌍만 남기고, 못 찾으면 기본 레코드에 이유를 적는다.
        sibling, reason = _contrast_sibling(
            records[0], request_id=f"{domain_name}-{family_index:04d}-3", wording_seed=f"{seed}:{group}:w0",
            shuffle_seed=f"{seed}:{group}:s0", policy=policy, **shared,
        )
        if sibling is not None:
            records.append(sibling)
            records[0]["provenance"]["contrast"] = {
                "role": "base",
                "sibling_id": sibling["request"]["request_id"],
                "focus_field": sibling["provenance"]["contrast"]["focus_field"],
                "flipped_question": sibling["provenance"]["contrast"]["flipped_question"],
            }
        else:
            records[0]["provenance"]["contrast"] = {"role": "base", "sibling_id": None, "missing": reason}
    # 템플릿 변형·개념 holdout은 렌더링 뒤에야 안다. 계열의 레코드 중 하나라도 걸리면 계열 전체가 OOD다 — 한 group이
    # 두 split에 걸치지 않는다(docs/04 §5). group·prefix·domain holdout은 이미 `split`에 반영돼 있다.
    reasons = policy.holdout_reasons(group, _holdout_tags(records))
    if reasons:
        for record in records:
            record["split"] = ood_split(group)
            record["provenance"]["holdout"] = list(reasons)
    return records


# --------------------------------------------------------------------------
# 대조 sibling (docs/04 §3, analysis-nimble §3-2)
# --------------------------------------------------------------------------


def semantic_answers(record: dict) -> dict[str, Any]:
    """질문 id → 표현·후보 배열과 무관한 정답 값. choice는 후보의 `ref`(없으면 "해당 없음"), boolean은 참/거짓, ordinal은
    수준 id. 라벨이 없는(마스크) 질문은 빠진다."""
    labels = {label["question_id"]: label for label in record["labels"]}
    out: dict[str, Any] = {}
    for question in record["request"]["questions"]:
        label = labels.get(question["id"])
        if label is None:
            continue
        if question["type"] == "boolean":
            out[question["id"]] = bool(label["answer"])
            continue
        keys = {criterion["id"]: criterion.get("ref", "<none>") for criterion in question["criteria"]}
        chosen = label.get("candidate_ids") or [label["answer"]]
        out[question["id"]] = tuple(sorted(keys[candidate] for candidate in chosen))
    return out


def deletion_outcome(domain: Any, deleted: dict, spec: QuestionSpec, language: str) -> str | None:
    """초점 사실을 지운 장면에서 그 질문을 다시 그리면 라벨이 어떻게 되는가: `"masked"`(근거 없음), `"none_candidate"`("해당
    없음·정보 부족"이 정답), 아니면 `None`(라벨이 남는다 — 삭제 검사 실패). 문구 난수는 라벨과 무관하다."""
    rendered = domain.render(deleted, spec, random.Random(f"deletion:{spec.id}"), language)
    take_phrasing()  # 삭제 검사의 문구 변형은 레코드의 것이 아니다
    if rendered.label is None:
        return "masked"
    label = rendered.label
    chosen = label.get("candidate_ids") or ([label["answer"]] if isinstance(label.get("answer"), str) else [])
    if chosen == [NONE_ID]:
        return "none_candidate"
    return None


def _contrast_sibling(
    base: dict,
    *,
    domain: Any,
    scene: dict,
    specs: Sequence[QuestionSpec],
    group: str,
    split: str,
    request_id: str,
    seed: int,
    family_index: int,
    wording_seed: str,
    shuffle_seed: str,
    config: Mapping[str, Any],
    policy: SplitPolicy,
) -> tuple[dict | None, str]:
    """기본 레코드의 대조 sibling과, 없으면 그 이유(`no_contrast`·`no_flip`·`deletion_failed`·`sealed_concept`).

    분야의 :meth:`contrasts`가 낸 후보를 차례로 그려, 겨냥 질문의 라벨이 (마스크가 아닌 두 라벨 사이에서) 실제로 뒤집히고
    :func:`deletion_outcome`을 지나는 첫 쌍을 택한다. 같은 wording·shuffle seed로 그리므로 sibling은 문구·후보 배열이 기본
    레코드와 같고 사실 하나만 다르다. 바뀐 사실이 계열의 봉인 여부를 바꾸는 sibling(기본 레코드에 없는 봉인 개념을 다루게
    되는 것)은 만들지 않는다 — 대조 쌍은 계열의 split을 따르지, 계열을 OOD로 끌고 가지 않는다(docs/04 §5의 몫을 지킨다).
    """
    language = base["provenance"]["language"]
    base_id = base["request"]["request_id"]
    base_answers = semantic_answers(base)
    by_id = {spec.id: spec for spec in specs}
    candidates = domain.contrasts(scene, list(specs))
    if not candidates:
        return None, "no_contrast"
    base_reasons = policy.holdout_reasons(group, _holdout_tags([base]))
    deletion_failed = sealed = False
    for contrast in candidates:
        sibling = _render_record(
            domain=domain, scene=contrast.flipped, specs=specs, group=group, split=split, request_id=request_id,
            language=language, seed=seed, family_index=family_index, wording_seed=wording_seed, shuffle_seed=shuffle_seed,
            config=config, derived_from=base_id, derivation="contrast",
        )
        if policy.holdout_reasons(group, _holdout_tags([base, sibling])) != base_reasons:
            sealed = True
            continue
        answers = semantic_answers(sibling)
        flipped = [qid for qid in by_id if qid in base_answers and qid in answers and base_answers[qid] != answers[qid]]
        for question_id in contrast.question_ids:
            if question_id not in flipped:
                continue
            outcome = deletion_outcome(domain, contrast.deleted, by_id[question_id], language)
            if outcome is None:
                deletion_failed = True
                continue
            spec = by_id[question_id]
            provenance = {
                "role": "sibling",
                "sibling_id": base_id,
                "focus_field": contrast.focus_field,
                "flipped_question": question_id,
                "flipped_questions": flipped,
                "deletion": {"field": contrast.focus_field, "outcome": outcome},
            }
            evidence = {
                "spec": {"id": spec.id, "type": spec.type, "kind": spec.kind, "params": copy.deepcopy(spec.params)},
                "base_answer": _json_answer(base_answers[question_id]),
                "sibling_answer": _json_answer(answers[question_id]),
            }
            sibling["provenance"]["contrast"] = provenance
            sibling["evidence"]["contrast"] = evidence
            return sibling, "paired"
    if deletion_failed:
        return None, "deletion_failed"
    return None, "sealed_concept" if sealed else "no_flip"


def _holdout_tags(records: Sequence[dict]) -> list[str]:
    """레코드들의 holdout 태그 (문구 템플릿 변형·개념) — `_build_family`의 계열 판정과 같은 문자열."""
    return sorted(
        {TEMPLATE_TAG + item for record in records for item in record["provenance"]["phrasing"]}
        | {CONCEPT_TAG + item for record in records for item in record["provenance"]["concepts"]}
    )


def _json_answer(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _question_kind(question_id: str) -> str:
    """`q_target_3` → `q_target` (번호 접미만 뗀다); 로봇의 `q_main`은 그대로."""
    head, _, tail = question_id.rpartition("_")
    return head if head and tail.isdigit() else question_id


def contrast_counts(records: Sequence[dict]) -> dict:
    """대조 쌍의 집계: 쌍 수, split·분야별 쌍 수, 뒤집힌 질문 종류, 삭제 검사 결과, sibling이 없는 기본 레코드의 이유."""
    pairs = 0
    by_split: Counter = Counter()
    by_domain: Counter = Counter()
    by_kind: Counter = Counter()
    deletion: Counter = Counter()
    missing: Counter = Counter()
    for record in records:
        provenance = record.get("provenance") or {}
        contrast = provenance.get("contrast")
        if not contrast:
            continue
        if contrast.get("role") == "sibling":
            pairs += 1
            by_split[record.get("split")] += 1
            by_domain[provenance.get("domain")] += 1
            by_kind[_question_kind(str(contrast.get("flipped_question", "")))] += 1
            deletion[(contrast.get("deletion") or {}).get("outcome")] += 1
        elif contrast.get("sibling_id") is None:
            missing[contrast.get("missing") or "unknown"] += 1
    bases = sum(1 for record in records if (record.get("provenance") or {}).get("derived_from") is None)
    return {
        "pairs": pairs,
        "base_records": bases,
        "pair_share_of_bases": round(pairs / bases, 4) if bases else 0.0,
        "by_split": dict(sorted(by_split.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "by_question_kind": dict(sorted(by_kind.items())),
        "deletion_outcomes": dict(sorted(deletion.items())),
        "missing": dict(sorted(missing.items())),
    }


def generate_records(
    count: int, seed: int, config: Mapping[str, Any] | None = None
) -> list[dict]:
    """비로봇 단일 요청 레코드 `count`건. 같은 인자는 언제나 같은 결과를 낸다."""
    if count < 0:
        raise ValueError(f"count는 0 이상이어야 한다 (받은 값: {count})")
    resolved = _resolve(config)
    policy = SplitPolicy.from_config(resolved["split"])
    weights = resolved["domains"]

    produced: Counter = Counter({name: 0 for name in weights})
    families: Counter = Counter({name: 0 for name in weights})
    records: list[dict] = []
    while len(records) < count:
        domain_name = _next_domain(weights, produced)
        family_index = families[domain_name]
        families[domain_name] += 1
        for record in _build_family(
            domain_name=domain_name,
            family_index=family_index,
            seed=seed,
            config=resolved,
            policy=policy,
        ):
            records.append(record)
            produced[domain_name] += 1
            if len(records) == count:
                break
    return records


# --------------------------------------------------------------------------
# 파일 쓰기
# --------------------------------------------------------------------------


def _jsonl(records: Sequence[dict]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n"
        for record in records
    )


def write_dataset(
    records: Sequence[dict], output: Path, *, seed: int, config: Mapping[str, Any]
) -> dict:
    """`records.jsonl`과 `manifest.json`을 쓰고 manifest를 돌려준다."""
    output.mkdir(parents=True, exist_ok=True)
    payload = _jsonl(records).encode("utf-8")
    (output / "records.jsonl").write_bytes(payload)

    manifest = {
        "version": MANIFEST_VERSION,
        "generator": GENERATOR_VERSION,
        "config_version": config["version"],
        "config_sha256": config_digest(config),
        "config": config,
        "seed": seed,
        "count": len(records),
        "files": {
            "records.jsonl": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "records": len(records),
            }
        },
        "counts": _counts(records),
    }
    (output / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")
    )
    return manifest


def _counts(records: Sequence[dict]) -> dict:
    groups = {record["origin_group"] for record in records}
    return {
        "states": len(records),
        "origin_groups": len(groups),
        "questions": sum(len(record["request"]["questions"]) for record in records),
        "labels": sum(len(record["labels"]) for record in records),
        "domains": dict(
            sorted(Counter(record["provenance"]["domain"] for record in records).items())
        ),
        "splits": dict(sorted(Counter(record["split"] for record in records).items())),
        "languages": dict(
            sorted(Counter(record["provenance"]["language"] for record in records).items())
        ),
        "variants": dict(
            sorted(
                Counter(
                    tag for record in records for tag in record["provenance"]["variants"]
                ).items()
            )
        ),
        "contrast": contrast_counts(records),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m robo_jev.data.generate",
        description="비로봇 단일 요청 레코드를 만든다 (docs/04 §2·§3).",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML 설정 (없으면 기본값)")
    parser.add_argument("--count", type=int, required=True, help="만들 상태(레코드) 수")
    parser.add_argument("--seed", type=int, required=True, help="재현용 씨앗")
    parser.add_argument("--output", type=Path, required=True, help="records.jsonl을 쓸 디렉터리")
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else DEFAULT_CONFIG
    records = generate_records(args.count, args.seed, config=config)
    manifest = write_dataset(records, args.output, seed=args.seed, config=config)
    counts = manifest["counts"]
    print(
        f"{counts['states']}상태 · {counts['questions']}질문 · "
        f"{counts['origin_groups']}계열 → {args.output}"
    )
    print(f"  split: {counts['splits']}")
    print(f"  분야:  {counts['domains']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
