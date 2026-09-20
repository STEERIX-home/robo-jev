"""검수 표본 — D1 질문의 **층화 표본**(≥ 500, 핵심 층마다 ≥ 30, 100+ 이중 검수)과 검수 규약을 내보낸다 (docs/04 §6).

    uv run python scripts/export_review_sample.py --robot artifacts/datasets/d1-robot/d1-rollout-labels/manifest.json \\
        --single artifacts/datasets/d1/single/manifest.json --out artifacts/reports/d1-review-sample [--per-stratum 30] [--seed 17]

층(핵심 층, 층마다 `per_stratum` 이상):

* 로봇 스트림 — 질문 묶음 × 틱 난이도: `q_main` / 게이트(`q_done·q_instr·q_observe·q_retry·q_stop`) / 부가(`q_gripper·q_path·q_speed·
  q_force`) × `steady` / `event`(사건·목표 변경·전환 틱) / `other`(commitment 없음·갓 시작; :func:`robo_jev.sampler.tick_class`) — 9층.
* 로봇 스트림 rollout 라벨 — 키프레임 틱의 `q_main`(`source: rollout_v0`) × 라벨 신뢰도 high / medium / low — 3층.
* 로봇 틱 대조 쌍(`contrast/records.jsonl`, 뒤집힌 질문) × 종류 forbidden / zone_boundary / instruction — 3층.
* 비로봇 — 분야(spatial·dom·workflow·rules) × 질문 타입(choice·boolean·ordinal) — 12층; 층 안에서 난이도(`base` / `hard`: missing_info·
  no_correct_candidate·stale_observation·boundary_level 변형 / `contrast`: 대조 sibling)를 기록하고 있으면 `hard`·`contrast`를 각각
  `per_stratum // 6` 이상 넣는다.
* 봉인(`ood_test`) — 위 층에서는 `ood_test`를 뽑지 않는다(봉인). 대신 `sealed` 층 하나(`per_stratum`개, 로봇·비로봇 반반)를 따로 두어
  **모델 개발에 관여하지 않는 검수자**가 본다(규약). 봉인 문항은 개발자용 `sample.jsonl`/`sample.md`에 들어가지 않고 **별도 파일**
  `sealed.jsonl`/`sealed.md`로 나간다(D1 리뷰 1 I3) — 검수자 분리를 파일 분리로 지킨다.

표본의 문맥·후보 텍스트는 소형 scorer의 예제 구성(:func:`robo_jev.baselines.tiny_scorer.build_examples` — 허용 필드만, 직렬화의 후보
순서)이고, 라벨은 따로 붙는다. 같은 틱에서는 층마다 질문 하나, 같은 에피소드·기본 레코드에서는 층마다 `per_source_cap`개까지만 뽑아
표본이 몇 에피소드에 몰리지 않게 한다. 무작위성은 `seed`의 `random.Random` 하나다. 산출물: `sample.jsonl`(공개 분할의 질문마다 한 줄),
`sample.md`(사람이 읽는 시트, 판정 칸 비움), `sealed.jsonl`/`sealed.md`(봉인 문항만, 같은 꼴), `strata.json`(층별 모집단·표본 수,
`open`/`sealed` 수와 파일), `protocol.md`(규약).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from robo_jev.baselines.tiny_scorer import Example, build_examples, load_records
from robo_jev.contracts import SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM
from robo_jev.sampler import tick_class

__all__ = ["REVIEW_SAMPLE_VERSION", "build_sample", "main", "render_sheet", "write_sample"]

REVIEW_SAMPLE_VERSION = "review-sample-v1"
_GATES = ("q_done", "q_instr", "q_observe", "q_retry", "q_stop")
_AUX = ("q_gripper", "q_path", "q_speed", "q_force")
_HARD_VARIANTS = ("missing_info", "no_correct_candidate", "stale_observation", "boundary_level")
_NON_ROBOT_DOMAINS = ("spatial", "dom", "workflow", "rules")
_NON_ROBOT_TYPES = ("choice", "boolean", "ordinal")
_OPEN_SPLITS = ("train", "dev", "calibration", "test", "ood_dev")


def _question_group(question_id: str) -> str:
    if question_id == "q_main":
        return "q_main"
    if question_id in _GATES:
        return "gate"
    if question_id in _AUX:
        return "aux"
    return "other"


def _label_view(label: dict | None) -> dict[str, Any] | None:
    if label is None:
        return None
    keep = ("kind", "answer", "candidate_ids", "unknown", "semantic_admissible", "probabilities", "successes", "failures", "label_confidence",
            "source", "rule", "rollout_reason", "mask", "conditioned_on")
    return {key: label[key] for key in keep if key in label}


def _candidate(example: Example, question_id: str) -> tuple[list[dict[str, str]], dict | None, str, str]:
    question = next(item for item in example.questions if item.question_id == question_id)
    return ([{"id": cid, "text": text} for cid, text in zip(question.candidate_ids, question.candidate_texts)], question.label, question.header, question.question_type)


def _robot_candidates(records: list[dict], examples_by_id: dict[str, list[Example]]) -> dict[tuple, list[dict[str, Any]]]:
    """로봇 스트림의 (층 → 후보 질문 목록). 층 키: ("robot", 묶음, 난이도) 또는 ("robot", "q_main:rollout", 신뢰도)."""
    pools: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("schema_version") != SCHEMA_STREAM:
            continue
        examples = examples_by_id.get(str(record.get("episode_id")), [])
        for example in examples:
            tick = record["ticks"][example.tick]
            klass = tick_class(record, example.tick)
            difficulty = "event" if klass in ("event", "goal_change") else klass
            for question in example.questions:
                if question.label is None or question.label.get("mask") is False:
                    continue
                base = {"record": record, "example": example, "question_id": question.question_id, "difficulty": difficulty, "tick_class": klass}
                pools[("robot", _question_group(question.question_id), difficulty)].append(base)
                if question.question_id == "q_main" and question.label.get("source") == "rollout_v0":
                    pools[("robot", "q_main:rollout", str(question.label.get("label_confidence")))].append({**base, "difficulty": "rollout"})
    return pools


def _contrast_candidates(records: list[dict], examples_by_id: dict[str, list[Example]]) -> dict[tuple, list[dict[str, Any]]]:
    pools: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("schema_version") != SCHEMA_SINGLE_REQUEST:
            continue
        provenance = record.get("provenance") or {}
        kind = str(provenance.get("kind") or (provenance.get("contrast") or {}).get("kind") or "?")
        flipped = str((provenance.get("contrast") or {}).get("flipped_question") or "q_main")
        for example in examples_by_id.get(str(record["request"].get("request_id")), []):
            if any(question.question_id == flipped and question.label is not None for question in example.questions):
                pools[("robot_contrast", kind, str((provenance.get("contrast") or {}).get("role", "?")))].append(
                    {"record": record, "example": example, "question_id": flipped, "difficulty": "contrast", "tick_class": None}
                )
    return pools


def _non_robot_candidates(records: list[dict], examples_by_id: dict[str, list[Example]]) -> dict[tuple, list[dict[str, Any]]]:
    pools: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        provenance = record.get("provenance") or {}
        domain = str(provenance.get("domain") or "?")
        variants = set(provenance.get("variants") or ())
        if provenance.get("derivation") == "contrast":
            difficulty = "contrast"
        elif variants & set(_HARD_VARIANTS):
            difficulty = "hard"
        else:
            difficulty = "base"
        for example in examples_by_id.get(str(record["request"].get("request_id")), []):
            for question in example.questions:
                if question.label is None or question.label.get("mask") is False:
                    continue
                pools[(f"non_robot:{domain}", question.question_type, "-")].append(
                    {"record": record, "example": example, "question_id": question.question_id, "difficulty": difficulty, "tick_class": None}
                )
    return pools


def _source_key(entry: dict[str, Any]) -> str:
    record = entry["record"]
    if record.get("schema_version") == SCHEMA_STREAM:
        return str(record.get("episode_id"))
    provenance = record.get("provenance") or {}
    return str(provenance.get("derived_from") or provenance.get("episode_id") or record["request"].get("request_id"))


def _draw(pool: list[dict[str, Any]], want: int, rng: random.Random, *, per_source_cap: int, quotas: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """층 하나에서 `want`개: 먼저 난이도 quota(있으면)를 채우고 나머지는 균등; 같은 틱에서 하나, 같은 출처에서 `per_source_cap`까지."""
    order = list(pool)
    rng.shuffle(order)
    chosen: list[dict[str, Any]] = []
    used_ticks: set[tuple] = set()
    per_source: Counter = Counter()

    def take(entry: dict[str, Any]) -> bool:
        tick_key = (id(entry["record"]), entry["example"].tick)
        source = _source_key(entry)
        if tick_key in used_ticks or per_source[source] >= per_source_cap:
            return False
        used_ticks.add(tick_key)
        per_source[source] += 1
        chosen.append(entry)
        return True

    for difficulty, quota in (quotas or {}).items():
        taken = 0
        for entry in order:
            if taken >= quota or len(chosen) >= want:
                break
            if entry["difficulty"] == difficulty and entry not in chosen and take(entry):
                taken += 1
    for entry in order:
        if len(chosen) >= want:
            break
        if entry not in chosen:
            take(entry)
    return chosen


def build_sample(
    robot_records: list[dict], single_records: list[dict], *, per_stratum: int = 30, double_review: int = 100, seed: int = 17,
    per_source_cap: int = 2, max_context: int = 2048, max_candidate: int = 96, sealed_robot: list[dict] | None = None, sealed_single: list[dict] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """층화 표본 (모듈 설명). 돌려주는 것은 (질문 행 목록, 층 표)."""
    rng = random.Random(seed)
    build = lambda recs, domain: build_examples(recs, domain=domain, max_context=max_context, max_candidate=max_candidate)  # noqa: E731

    def by_id(examples: list[Example]) -> dict[str, list[Example]]:
        out: dict[str, list[Example]] = defaultdict(list)
        for example in examples:
            out[example.record_id].append(example)
        return out

    robot_open = [record for record in robot_records if record.get("split") in _OPEN_SPLITS]
    single_open = [record for record in single_records if record.get("split") in _OPEN_SPLITS]
    robot_examples = by_id(build(robot_open, "robot"))
    single_examples = by_id(build(single_open, "non_robot"))
    pools: dict[tuple, list[dict[str, Any]]] = {}
    pools.update(_robot_candidates(robot_open, robot_examples))
    pools.update(_contrast_candidates(robot_open, robot_examples))
    pools.update(_non_robot_candidates(single_open, single_examples))

    rows: list[dict[str, Any]] = []
    strata: dict[str, Any] = {}
    quota_hard = max(1, per_stratum // 6)
    for key in sorted(pools):
        pool = pools[key]
        quotas = {"hard": quota_hard, "contrast": quota_hard} if key[0].startswith("non_robot") else None
        chosen = _draw(pool, per_stratum, rng, per_source_cap=per_source_cap, quotas=quotas)
        name = "/".join(key)
        strata[name] = {"population": len(pool), "sampled": len(chosen), "short": max(0, per_stratum - len(chosen)),
                        "difficulty": dict(Counter(entry["difficulty"] for entry in chosen)), "splits": dict(Counter(entry["record"].get("split") for entry in chosen))}
        for entry in chosen:
            rows.append(_row(entry, name, sealed=False))

    # 봉인 층: ood_test에서 로봇·비로봇 반반, q_main·choice 우선 — 개발에 관여하지 않는 검수자 몫.
    sealed_rows = []
    for records, domain, wanted_group in ((sealed_robot or [], "robot", "q_main"), (sealed_single or [], "non_robot", "choice")):
        sealed = [record for record in records if record.get("split") == "ood_test"]
        if not sealed:
            continue
        examples = by_id(build(sealed, domain))
        pool = []
        for record in sealed:
            rid = str(record.get("episode_id") or record["request"].get("request_id"))
            for example in examples.get(rid, []):
                for question in example.questions:
                    if question.label is None or question.label.get("mask") is False:
                        continue
                    if (domain == "robot" and question.question_id != wanted_group) or (domain != "robot" and question.question_type != wanted_group):
                        continue
                    pool.append({"record": record, "example": example, "question_id": question.question_id, "difficulty": "sealed", "tick_class": None})
        chosen = _draw(pool, max(1, per_stratum // 2), rng, per_source_cap=per_source_cap)
        strata[f"sealed/{domain}"] = {"population": len(pool), "sampled": len(chosen), "short": max(0, per_stratum // 2 - len(chosen)), "difficulty": {"sealed": len(chosen)}, "splits": {"ood_test": len(chosen)}}
        sealed_rows.extend(_row(entry, f"sealed/{domain}", sealed=True) for entry in chosen)
    rows.extend(sealed_rows)

    # 이중 검수: 층마다 고르게 (봉인 층 포함), 총 `double_review`개.
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[row["stratum"]].append(row)
    names = sorted(by_stratum)
    flagged = 0
    cursor = {name: 0 for name in names}
    for name in names:
        rng.shuffle(by_stratum[name])
    while flagged < min(double_review, len(rows)):
        progressed = False
        for name in names:
            if flagged >= double_review:
                break
            bucket = by_stratum[name]
            if cursor[name] < len(bucket):
                bucket[cursor[name]]["double_review"] = True
                cursor[name] += 1
                flagged += 1
                progressed = True
        if not progressed:
            break
    for index, row in enumerate(sorted(rows, key=lambda item: (item["sealed"], item["stratum"], item["source"]["record_id"], item["source"]["tick"] or -1, item["question_id"]))):
        row["sample_id"] = f"rs-{index + 1:04d}"
    rows.sort(key=lambda item: item["sample_id"])
    table = {
        "version": REVIEW_SAMPLE_VERSION, "seed": seed, "per_stratum": per_stratum, "per_source_cap": per_source_cap,
        "questions": len(rows), "open": len(rows) - len(sealed_rows), "sealed": len(sealed_rows),
        "double_review": sum(1 for row in rows if row.get("double_review")),
        "double_review_open": sum(1 for row in rows if row.get("double_review") and not row["sealed"]),
        "files": {"open": ["sample.jsonl", "sample.md"], "sealed": ["sealed.jsonl", "sealed.md"]},
        "strata": strata, "core_strata": len([name for name in strata if not name.startswith("sealed/")]),
        "short_strata": {name: entry["short"] for name, entry in strata.items() if entry["short"]},
        "by_domain": dict(Counter(row["stratum"].split("/")[0] for row in rows)),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_question_type": dict(Counter(row["question_type"] for row in rows)),
    }
    return rows, table


def _row(entry: dict[str, Any], stratum: str, *, sealed: bool) -> dict[str, Any]:
    record, example, question_id = entry["record"], entry["example"], entry["question_id"]
    candidates, label, header, question_type = _candidate(example, question_id)
    is_stream = record.get("schema_version") == SCHEMA_STREAM
    provenance = record.get("provenance") or {}
    return {
        "sample_id": None,
        "stratum": stratum,
        "difficulty": entry["difficulty"],
        "tick_class": entry["tick_class"],
        "split": str(record.get("split")),
        "sealed": sealed,
        "double_review": False,
        "source": {
            "schema": record.get("schema_version"),
            "record_id": example.record_id,
            "tick": example.tick,
            "t": record["ticks"][example.tick]["t"] if is_stream else None,
            "origin_group": record.get("origin_group"),
            "domain": provenance.get("domain") or ("robot" if is_stream else None),
            "derivation": provenance.get("derivation"),
            "contrast": (provenance.get("contrast") or {}).get("role"),
            "versions": record.get("versions", {}).get("harness") if is_stream else provenance.get("generator"),
        },
        "question_id": question_id,
        "question_type": question_type,
        "question": header,
        "context": example.context,
        "candidates": candidates,
        "label": _label_view(label),
        "verdict": {"correct": None, "severity": None, "note": ""},
    }


def render_sheet(rows: list[dict[str, Any]], table: dict[str, Any], *, sealed: bool = False) -> str:
    """사람이 읽는 시트. `sealed=True`면 봉인 시트(ood_test 문항만; 모델 개발에 관여하지 않는 검수자 몫)의 머리를 단다."""
    doubled = sum(1 for row in rows if row.get("double_review"))
    if sealed:
        head = f"# D1 봉인 검수 시트 (ood_test {len(rows)}문항, 이중 검수 {doubled}; seed {table['seed']}) — 모델 개발에 관여하지 않는 검수자만 본다"
    else:
        head = f"# D1 검수 시트 ({len(rows)}문항, 이중 검수 {doubled}; seed {table['seed']}; 봉인 {table['sealed']}문항은 sealed.md에 따로)"
    lines = [head, "",
             "판정 칸: `correct` (예/아니오), `severity` (major/minor/none), `note`. 규약은 protocol.md. 라벨은 문항 아래 `LABEL`에 있다 — 먼저 답을 정한 뒤 라벨과 대조한다.", ""]
    for row in rows:
        flags = " ".join(flag for flag, on in (("[이중]", row.get("double_review")), ("[봉인]", row.get("sealed"))) if on)
        lines.append(f"## {row['sample_id']} · {row['stratum']} · {row['difficulty']} · {row['split']} {flags}".rstrip())
        source = row["source"]
        where = f"{source['record_id']}" + (f" tick {source['tick']} (t={source['t']})" if source["tick"] is not None else "")
        lines.append(f"출처: {where} · {source['origin_group']}")
        lines.append("```")
        lines.append(row["context"].rstrip("\n"))
        lines.append("```")
        lines.append(f"**질문** {row['question']}")
        for candidate in row["candidates"]:
            lines.append(f"- {candidate['text']}")
        lines.append(f"LABEL: `{json.dumps(row['label'], ensure_ascii=False)}`")
        lines.append("판정: correct= severity= note=")
        lines.append("")
    return "\n".join(lines) + "\n"


PROTOCOL = """# D1 검수 규약 (docs/04 §6)

**목적.** D1 라벨의 오류율을 표본으로 재고(임시 통과 기준: **중대한 오라벨 2 % 미만**, Wilson 95 % 구간 상한과 함께 보고), 누출이 보이면
그 계열 전체를 재생성한다. 표본 검수만으로 오류 0 %를 주장하지 않는다.

**무엇을 보는가.** 문항마다 모델이 보는 것과 같은 문맥(허용 필드만)·질문·후보 줄이 있고, 라벨은 `LABEL`에 따로 있다. 근거(rollout 결과,
전문가 로그, 참값)는 시트에 없다 — 검수자는 문맥만으로 답을 정한 뒤 라벨과 대조한다.

**판정.**
* `correct = 예` — 라벨이 문맥에서 정당하다(허용 집합 라벨은 허용된 답이 모두 정당하고 빠진 정답이 없다; boolean·ordinal은 규칙대로).
* `correct = 아니오` + `severity`:
  * **major(중대)** — 라벨대로 행동하면 목표를 놓치거나 안전 규칙(금지 접촉·정지)을 어긴다, 허용 집합에 명백히 잘못된 후보가 있다, 정답이
    후보에 있는데 라벨이 "해당 없음"이다(또는 반대), boolean·ordinal 답이 문맥과 반대다.
  * **minor(경미)** — 허용 집합의 경계(비슷한 비용의 대안·인접 수준)가 다르게 그어졌다, 정답은 맞지만 신뢰도 표기·unknown 처리가 어색하다.
  * **none** — 문맥이 모호해 판정할 수 없다(문항을 `note`에 표시; 오류로 세지 않되 모호 비율을 따로 보고).
* `note` — 근거 한 줄. 문맥에 라벨·근거·미래 정보가 보이면(정보 경계 위반) 반드시 적는다 — 그 계열은 재생성 대상이다.

**층.** `stratum`은 `robot/<질문 묶음>/<틱 난이도>`, `robot/q_main:rollout/<신뢰도>`, `robot_contrast/<종류>/<role>`, `non_robot:<분야>/<타입>/-`,
`sealed/<분야>`. 핵심 층마다 ≥ 30이라 500을 넘는다(docs/04 §6 "핵심 층 30개를 확보하느라 넘으면 확대한다"). 오류율은 층별로도 낸다 —
특히 `q_main:rollout/low`·`robot_contrast/*`·`non_robot:*/ordinal`.

**이중 검수.** `[이중]` 표시 문항(≥ 100)은 두 사람이 독립 판정한 뒤 일치율·Cohen κ를 보고하고, 불일치는 셋째 판정으로 푼다.

**봉인.** `[봉인]` 문항은 `ood_test`에서 뽑았고 **`sealed.jsonl` / `sealed.md`에만** 있다 — `sample.*`에는 `ood_test` 문항이 없다.
봉인 파일은 **모델 개발에 관여하지 않는 검수자**만 열고, 개발자(구현자·학습을 돌리는 사람)에게 건네지 않는다. 그 오류를 생성기·규칙·
홀드아웃 수정에 쓰면 그 `ood_test` 버전을 은퇴시키고 새 봉인 세트를 만든다(docs/04 §5 은퇴 규칙). 봉인 문항의 결과는 별도 표로 보고한다.

**보고.** `sample.jsonl`(봉인 검수자는 `sealed.jsonl`)의 `verdict`를 채워 돌려준다. 집계: 전체·층별 major 비율과 Wilson 95 % 구간, minor
비율, 모호 비율, 이중 검수 일치율, 정보 경계 위반 목록(계열 id), 누출 의심 목록. 통과: major < 2 %(점추정)이고 정보 경계 위반 0.
"""


def write_sample(out: Path, rows: list[dict[str, Any]], table: dict[str, Any]) -> dict[str, Path]:
    """공개 문항은 `sample.jsonl`/`sample.md`, 봉인 문항(`sealed`·`ood_test`)은 `sealed.jsonl`/`sealed.md`에 따로 쓴다 (D1 리뷰 1 I3)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    open_rows = [row for row in rows if not row.get("sealed") and row.get("split") != "ood_test"]
    sealed_rows = [row for row in rows if row.get("sealed") or row.get("split") == "ood_test"]
    paths = {"sample": out / "sample.jsonl", "sheet": out / "sample.md", "sealed": out / "sealed.jsonl", "sealed_sheet": out / "sealed.md",
             "strata": out / "strata.json", "protocol": out / "protocol.md"}
    for path, subset in ((paths["sample"], open_rows), (paths["sealed"], sealed_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["sheet"].write_text(render_sheet(open_rows, table), encoding="utf-8")
    paths["sealed_sheet"].write_text(render_sheet(sealed_rows, table, sealed=True), encoding="utf-8")
    paths["strata"].write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    paths["protocol"].write_text(PROTOCOL, encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/export_review_sample.py", description=__doc__.splitlines()[0])
    parser.add_argument("--robot", type=Path, required=True, help="로봇 D1 manifest (rollout 라벨 후속 버전)")
    parser.add_argument("--single", type=Path, required=True, help="비로봇 D1 manifest")
    parser.add_argument("--out", type=Path, default=Path("artifacts/reports/d1-review-sample"))
    parser.add_argument("--per-stratum", type=int, default=30)
    parser.add_argument("--double-review", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    splits = (*_OPEN_SPLITS, "ood_test")
    robot = load_records(args.robot, splits=splits)
    single = load_records(args.single, splits=splits)
    rows, table = build_sample(robot, single, per_stratum=args.per_stratum, double_review=args.double_review, seed=args.seed, sealed_robot=robot, sealed_single=single)
    table["sources"] = {"robot": str(args.robot), "single": str(args.single), "robot_records": len(robot), "single_records": len(single)}
    paths = write_sample(args.out, rows, table)
    print(json.dumps({key: table[key] for key in ("questions", "open", "sealed", "double_review", "double_review_open", "core_strata", "short_strata", "by_domain", "by_split", "by_question_type", "files")}, ensure_ascii=False, indent=1))
    for name, entry in table["strata"].items():
        print(f"  {name:<40} population {entry['population']:>7} sampled {entry['sampled']:>3} {entry['difficulty']}")
    print(f"→ {paths['sheet']} (+ {paths['sealed_sheet'].name} for the sealed reviewer) ({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
