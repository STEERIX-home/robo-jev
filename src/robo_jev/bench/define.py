"""bench-v0 정의 — 세트마다 어떤 레코드를 벤치에 넣는가 (Task bench0 §1).

**정의 파일**: `configs/bench/bench-v0.yaml`(커밋)이 정본이다 — 원본 파일과 그 sha256, seed, 세트마다 표본 단위·크기, 층 열쇠,
그리고 고른 결과의 기대값(레코드 수·질문 수·id 목록의 sha256). :func:`build` 는 원본에서 표본을 다시 뽑아 기대값과 대조하고
(다르면 거절), 레코드를 원본의 **바이트 그대로** `artifacts/bench/v0/<세트>.jsonl`(커밋 안 함)에 쓴다.

**표본** (:func:`select_set`):

* 단위(``unit``) — ``record``(T1~T4·T7: 레코드 하나), ``pair``(T5: 사실 하나만 바꾼 쌍 — ``meta.pair``로 이어진 둘),
  ``t6_family``(T6: 기준 레코드 + 그 모든 변형 — ``meta.t6.base``), ``all``(G1·G2: 전부).
* **층화** — 층 = 단위의 대표 레코드(쌍은 ``role: base``, T6 묶음은 기준 레코드)의 ``meta.group``(현상 묶음)과 ``meta.language``.
  층마다의 몫은 그 층의 단위 수에 비례하고 최대 나머지법으로 반올림한다(같은 나머지는 층 이름 순).
* 층 안에서 — ``record`` 단위는 진단 도구의 쌍을 깨지 않는 표본기 :func:`robo_jev.decisions.records.select_records` 를 그대로
  쓰고(``--limit``과 같은 알고리즘), ``pair``·``t6_family``는 같은 알고리즘(묶음 순서를 seed로 섞고 앞에서 k개)을 묶음 위에서 쓴다.
  층의 seed는 ``sha256("bench-v0|<seed>|<세트>|<층>")``의 앞 8바이트다. 결과는 원본 파일 순서다.
* 쌍·묶음은 깨지지 않는다(검사가 지킨다). 같은 정의 → 같은 id(검사가 지킨다).

**시운전 표본** (:func:`select_pilot`): score 형식(단계 둘 그대로 대 11단계) 비교용 — 벤치에 들지 않은 T1 레코드에서
:func:`select_records` 로 고른다(seed는 위와 같은 식, 층 이름 ``pilot``).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from robo_jev.decisions.records import file_sha256, pair_groups, pair_of, select_records

__all__ = [
    "BENCH_SETS",
    "UNITS",
    "allocate",
    "build",
    "ids_sha256",
    "load_config",
    "question_counts",
    "read_raw",
    "select_pilot",
    "select_set",
    "stratum_of",
    "stratum_seed",
    "units_of",
]

#: 벤치의 세트 이름 (보고 순서).
BENCH_SETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "G1", "G2")
UNITS = ("record", "pair", "t6_family", "all")
#: 층 열쇠의 기본값 — 대표 레코드의 meta 값 (없으면 ``unknown``).
DEFAULT_STRATA = ("group", "language")


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config.get("sets"), dict) or not config.get("version"):
        raise ValueError(f"{path}: bench 정의에는 version과 sets가 있어야 한다")
    return config


def read_raw(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """JSONL → (원래 줄 글, 읽은 dict) 목록 — 벤치 파일을 원본의 바이트 그대로 쓰려고 줄 글을 지닌다(빈 줄은 건너뛴다)."""
    out = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append((line.rstrip("\n"), json.loads(line)))
    return out


def ids_sha256(ids: Iterable[str]) -> str:
    """id 목록(파일 순서)의 sha256 — 줄마다 id 하나, 끝에 줄바꿈."""
    return hashlib.sha256("".join(f"{i}\n" for i in ids).encode("utf-8")).hexdigest()


def stratum_seed(seed: int, set_name: str, stratum: str) -> int:
    digest = hashlib.sha256(f"bench-v0|{int(seed)}|{set_name}|{stratum}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stratum_of(record: dict[str, Any], keys: Iterable[str] = DEFAULT_STRATA) -> str:
    meta = record.get("meta") or {}
    return "/".join(str(meta.get(key) if meta.get(key) is not None else "unknown") for key in keys)


def _t6_base(record: dict[str, Any]) -> str:
    t6 = (record.get("meta") or {}).get("t6") or {}
    return str(t6.get("base") or record.get("id"))


def units_of(records: list[dict[str, Any]], unit: str) -> list[list[dict[str, Any]]]:
    """표본 단위의 묶음들 (묶음 안은 파일 순서, 묶음은 첫 레코드의 파일 순서). 첫 원소가 대표 레코드다(쌍은 ``role: base``,
    T6 묶음은 기준 레코드)."""
    if unit in ("record", "pair"):
        groups = pair_groups(records)
        out = []
        for group in groups:
            base = [r for r in group if ((r.get("meta") or {}).get("pair") or {}).get("role") == "base"]
            out.append(base[:1] + [r for r in group if not base or r is not base[0]])
        if unit == "pair":
            lonely = [g[0].get("id") for g in out if len(g) != 2 or pair_of(g[0])[0] is None]
            if lonely:
                raise ValueError(f"pair 단위인데 짝이 없거나 셋 이상인 묶음 {len(lonely)}개 (처음: {lonely[:3]})")
        return out
    if unit == "t6_family":
        families: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            families.setdefault(_t6_base(record), []).append(record)
        out = []
        for base_id, members in families.items():
            base = [r for r in members if str(r.get("id")) == base_id]
            if len(base) != 1:
                raise ValueError(f"T6 묶음 {base_id!r}에 기준 레코드가 {len(base)}개다")
            out.append(base + [r for r in members if r is not base[0]])
        return out
    raise ValueError(f"unit: {list(UNITS)} 중 하나 (받은 값: {unit!r})")


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """층마다 몫 — 크기에 비례, 최대 나머지법(같은 나머지는 층 이름 순). 합 = min(total, 전체)."""
    whole = sum(sizes.values())
    if total >= whole:
        return dict(sizes)
    exact = {name: total * size / whole for name, size in sizes.items()}
    out = {name: int(value) for name, value in exact.items()}
    left = total - sum(out.values())
    for name in sorted(exact, key=lambda n: (-(exact[n] - out[n]), n))[:left]:
        out[name] += 1
    return out


def select_set(
    records: list[dict[str, Any]], *, set_name: str, unit: str, count: int | None, seed: int, strata: Iterable[str] = DEFAULT_STRATA
) -> list[dict[str, Any]]:
    """세트 하나의 표본 (모듈 설명). `count`는 단위 수(레코드·쌍·T6 묶음)이고 ``all``이면 무시한다. 결과는 파일 순서."""
    if unit == "all":
        return list(records)
    if count is None or count <= 0:
        raise ValueError(f"{set_name}: unit {unit!r}에는 양의 count가 필요하다")
    keys = tuple(strata)
    groups = units_of(records, unit)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_stratum[stratum_of(group[0], keys)].append(index)
    quota = allocate({name: len(members) for name, members in by_stratum.items()}, int(count))
    picked: set[int] = set()
    for name in sorted(by_stratum):
        members = by_stratum[name]
        k = quota[name]
        rng_seed = stratum_seed(seed, set_name, name)
        if unit == "record":
            stratum_records = [r for g in members for r in groups[g]]
            chosen = {id(r) for r in select_records(stratum_records, k, seed=rng_seed)}
            picked |= {g for g in members if any(id(r) in chosen for r in groups[g])}
            continue
        order = list(range(len(members)))
        random.Random(rng_seed).shuffle(order)
        picked |= {members[i] for i in order[:k]}
    keep = {id(r) for g in picked for r in groups[g]}
    return [r for r in records if id(r) in keep]


def select_pilot(t1_records: list[dict[str, Any]], bench_ids: Iterable[str], *, count: int, seed: int) -> list[dict[str, Any]]:
    """시운전 표본: 벤치에 들지 않은 T1 레코드에서 :func:`select_records` 로 `count`개 (파일 순서)."""
    taken = {str(i) for i in bench_ids}
    rest = [r for r in t1_records if str(r.get("id")) not in taken]
    return select_records(rest, int(count), seed=stratum_seed(seed, "T1", "pilot"))


def question_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter: Counter = Counter()
    for record in records:
        for question in record["request"]["questions"].values():
            counter[str(question.get("type"))] += 1
    return {"total": sum(counter.values()), **dict(sorted(counter.items()))}


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def build(config: dict[str, Any], *, root: str | Path, out_dir: str | Path | None, check: bool = True) -> dict[str, Any]:
    """정의 → 세트마다 표본, (`out_dir`가 있으면) 레코드를 원본 바이트 그대로 쓰고 manifest를 돌려준다.

    `check`면 원본 sha256·고른 레코드 수·질문 수·id sha256을 정의의 기대값(``expect``)과 대조해 다르면 ``ValueError``.
    기대값이 없는 칸은 대조하지 않는다(처음 만들 때 — manifest의 값을 정의에 옮겨 적는다)."""
    root = Path(root)
    seed = int(config["seed"])
    strata = tuple(config.get("strata") or DEFAULT_STRATA)
    manifest: dict[str, Any] = {"version": config["version"], "seed": seed, "strata": list(strata), "sets": {}}
    problems: list[str] = []
    bench_ids: dict[str, list[str]] = {}
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    raw_by_set: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for set_name, spec in config["sets"].items():
        source = _resolve(root, spec["source"])
        raw = read_raw(source)
        raw_by_set[set_name] = raw
        records = [record for _, record in raw]
        chosen = select_set(records, set_name=set_name, unit=spec["unit"], count=spec.get("count"), seed=seed, strata=strata)
        ids = [str(r["id"]) for r in chosen]
        bench_ids[set_name] = ids
        entry = {
            "source": spec["source"], "source_sha256": file_sha256(source), "unit": spec["unit"], "count": spec.get("count"),
            "records": len(chosen), "units": len(units_of(chosen, spec["unit"])) if spec["unit"] != "all" else len(chosen),
            "questions": question_counts(chosen), "ids_sha256": ids_sha256(ids),
        }
        if out is not None:
            keep = {id(r) for r in chosen}
            path = out / f"{set_name}.jsonl"
            path.write_text("".join(line + "\n" for line, record in raw if id(record) in keep), encoding="utf-8")
            entry["file"] = str(path)
            entry["file_sha256"] = file_sha256(path)
        manifest["sets"][set_name] = entry
        expect = spec.get("expect") or {}
        if check:
            for key in ("source_sha256", "records", "ids_sha256"):
                if key in expect and expect[key] != entry[key]:
                    problems.append(f"{set_name}: {key} {entry[key]!r} ≠ expected {expect[key]!r}")
            if "questions" in expect and int(expect["questions"]) != entry["questions"]["total"]:
                problems.append(f"{set_name}: questions {entry['questions']['total']} ≠ expected {expect['questions']}")
    pilot = config.get("pilot")
    if pilot:
        source_set = pilot.get("from", "T1")
        records = [record for _, record in raw_by_set[source_set]]
        chosen = select_pilot(records, bench_ids.get(source_set, []), count=int(pilot["count"]), seed=seed)
        ids = [str(r["id"]) for r in chosen]
        entry = {"from": source_set, "count": int(pilot["count"]), "records": len(chosen), "questions": question_counts(chosen),
                 "ids_sha256": ids_sha256(ids), "overlap_with_bench": len(set(ids) & set(bench_ids.get(source_set, [])))}
        if out is not None:
            keep = {id(r) for r in chosen}
            path = out / f"pilot-{source_set}.jsonl"
            path.write_text("".join(line + "\n" for line, record in raw_by_set[source_set] if id(record) in keep), encoding="utf-8")
            entry["file"] = str(path)
            entry["file_sha256"] = file_sha256(path)
        manifest["pilot"] = entry
        expect = pilot.get("expect") or {}
        if check:
            for key in ("records", "ids_sha256"):
                if key in expect and expect[key] != entry[key]:
                    problems.append(f"pilot: {key} {entry[key]!r} ≠ expected {expect[key]!r}")
    if problems:
        raise ValueError("bench 정의와 다르다 — " + "; ".join(problems))
    return manifest
