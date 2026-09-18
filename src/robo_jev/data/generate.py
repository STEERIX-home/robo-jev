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
from robo_jev.data.domains import DOMAINS, QuestionSpec, Rendered
from robo_jev.data.split import SplitPolicy

__all__ = [
    "DEFAULT_CONFIG",
    "GENERATOR_VERSION",
    "PILOT_CONFIG",
    "generate_records",
    "load_config",
    "main",
    "write_dataset",
]

GENERATOR_VERSION = "gen-single-v0.1.0"
MANIFEST_VERSION = "manifest-v0"

#: 함께 배포하는 설정 파일. `load_config(PILOT_CONFIG) == DEFAULT_CONFIG`이어야 한다.
PILOT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "data" / "pilot.yaml"

#: 시작값은 docs/04 §2에서 온다. 설정 파일로 덮어쓸 수 있다.
DEFAULT_CONFIG: dict[str, Any] = {
    "version": "pilot-v0",
    # 비로봇 상태의 분야 비중. docs/04 §2는 dom/workflow/rules를 15/15/10으로 두고
    # 나머지를 색·위치·영역 문제가 채운다 — pilot에서는 넷을 고르게 둔다.
    "domains": {"spatial": 25, "dom": 25, "workflow": 25, "rules": 25},
    "question_types": {"choice": 60, "boolean": 25, "ordinal": 15},
    "questions_per_state": {1: 10, 4: 20, 8: 50, 16: 20},
    "languages": {"ko": 70, "en": 30},
    # 파생본 비율(%). 번역본과 후보 재배열본은 부모의 group·split을 승계한다.
    "derivations": {"paraphrase": 20, "reorder": 10},
    "split": {
        "weights": {"train": 70, "dev": 10, "calibration": 10, "test": 10},
        "holdout_groups": [],
        "holdout_prefixes": [],
        "holdout_domains": [],
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
    merged = dict(base)
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
) -> dict:
    wording_rng = random.Random(wording_seed)
    state = domain.state(scene, wording_rng, language)
    rendered: list[Rendered] = [
        domain.render(scene, spec, wording_rng, language) for spec in specs
    ]

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
            "rules": sorted({label["source"] for label in labels}),
            **({"derived_from": derived_from, "derivation": derivation} if derived_from else {}),
        },
        "evidence": {
            "rule_trace": [item.trace for item in rendered],
            "masked": masked,
            # 규칙이 읽은 사실 원본. 계열 안의 레코드가 같은 객체를 나눠 갖지 않도록 복사한다.
            "scene": copy.deepcopy(scene),
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
    template = family_rng.choice(domain.templates)
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
    return records


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
