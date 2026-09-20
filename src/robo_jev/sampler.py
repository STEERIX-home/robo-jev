"""학습 데이터의 읽기 전용 적재와 혼합 sampler (docs/04 §2·§7, docs/08 §8, docs/06 Task 5).

**적재.** :func:`load_items` 는 데이터 manifest(`files: {경로: {sha256, …}}`)가 가리키는 JSONL 파일을
manifest 순서·줄 순서로 읽고(sha256 대조), `split`으로 거른 뒤 레코드를 그 종류의 배치로 직렬화한다
(`judgment-v0` → ``state_first``, `stream-v0` → ``stream_l1a``). :func:`load_manifests` 는 여러 manifest
(로봇 batch + 비로봇 데이터)를 manifest 순서로 이어 붙이고 manifest마다 분야·자료 태그의 기본값을 달 수 있다. 직렬화를 **적재 시점에 고정된 순서로**
하는 이유는 공백 tokenizer가 id를 처음 본 순서로 주기 때문이다 — 재개한 프로세스도 같은 순서로
읽어 같은 토큰 id를 얻는다(실제 tokenizer는 순서와 무관하다). 레코드는 라벨·틱 종류 계산에 쓰려고
그대로 들고 있되, 모델에는 직렬화된 layout만 간다.

**두 sampler 축 (docs/04 §2, §7; docs/09).**

* 로봇/비로봇 — step마다 **둘 다** 넣는다(판정 e086c90, docs/04 §2): :meth:`MixedSampler.draw_step` 이
  로봇 단위(에피소드 하나 = TBPTT 구간열), 비로봇 단위(단일 요청을 ``nonrobot_tokens_per_unit``까지
  묶은 microbatch), 로봇, … 을 번갈아 뽑는다. 60/40은 학습 loop가 **step의 유효 loss 비중**으로 건다
  (``0.6·L_robot + 0.4·L_nonrobot``); sampler는 실현 토큰 비중을 기록만 한다. 누적 토큰 비중을 좇는
  규칙은 쓰지 않는다 — 에피소드 하나가 단일 요청 수백 개의 토큰이라 스트림이 굶는다(Task 5 실측).
  레코드의 분야는 태그(`provenance.domain`, 기본: 스트림 = robot, 단일 요청 = non_robot)로 정한다.
* 기존 자료/오류 계열/새 의미 계열 — `70/20/10`, 레코드의 provenance 태그(`provenance.material`,
  없으면 기존 자료)로 나눈다. 비어 있는 묶음은 재정규화하고(D0/D1에서는 뒤의 두 묶음이 비어 있을
  수 있다) 실현 비중을 기록한다 — 실패하지 않는다.

한 단위는 에피소드 하나(스트림 = accumulation 단위, docs/03 §5) 또는 토큰 예산까지 묶은 단일 요청들이다
(예산을 넘기는 레코드는 되돌려 다음 단위에 넣으므로 epoch 안에서 잃는 레코드가 없다). 묶음 안에서는
epoch마다 섞어 한 번씩 뽑는다. 모든 무작위성은 seed로 만든 `random.Random` 하나에서
나오고, :meth:`MixedSampler.state_dict` 가 위치(뽑은 수·묶음별 순서·cursor·epoch·실현 토큰·RNG)와 그 위치의
index가 가리키는 **레코드의 출처**(파일별 sha256·적재 순서·수, :func:`record_sources`)를 기본 자료형으로
돌려주어 재개할 수 있다 — :meth:`MixedSampler.load_state_dict` 는 출처가 다른 데이터(내용이 바뀐 사본)에
옛 위치를 싣지 않는다.

**틱 종류와 가중치 (docs/04 §2 "정상 유지 틱 하향, 이벤트·목표 변경 틱 상향").** 틱의 종류는 그 틱의
레코드·라벨과 **직전** 틱(목표 버전 비교)에서만 정한다 — 미래 틱을 보지 않는다.

* ``goal_change`` — `state.goal.version`이 직전 틱보다 올랐다(버전이 없으면 그 사이에 들어온 지시).
* ``event`` — `state.events`가 비어 있지 않거나, 그 틱의 채택 결과가 전환(`adopted.switch`)이다.
* ``steady`` — commitment가 `steady_min_held_ticks` 틱 이상 유지됐다.
* ``other`` — 나머지(commitment 없음, 갓 시작한 commitment).

가중치는 설정(`tick_weights`)이 정하며 손실에서 ``L_episode = Σ_t w_t L_t / Σ_t w_t`` 로 쓴다
(유효 라벨이 있는 틱만). 실제 유효 loss 비중은 학습 loop가 기록한다.

이 모듈은 generator·simulator·하네스를 import하지 않는다 (docs/06 §1).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robo_jev.contracts import QUESTION_SET_V0, SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM, SPLITS
from robo_jev.model.serialize import WINDOW_TICKS, serialize_request
from robo_jev.model.tokenizer import sha256_of_file

__all__ = [
    "DOMAINS",
    "MATERIALS",
    "TICK_CLASSES",
    "Item",
    "MixedSampler",
    "Unit",
    "load_items",
    "load_manifests",
    "manifest_files",
    "permute_candidates",
    "record_sources",
    "sha256_of",
    "tick_class",
    "tick_weights",
    "valid_label_ticks",
    "valid_single",
]

#: 로봇/비로봇 축 (docs/04 §2).
DOMAINS = ("robot", "non_robot")
#: 기존 자료 / 오류 계열 / 새 의미 계열 축 (docs/04 §7 `70/20/10`).
MATERIALS = ("existing", "error_family", "new_semantic_family")
#: 틱 종류 (모듈 설명 참조).
TICK_CLASSES = ("steady", "event", "goal_change", "other")

DEFAULT_MATERIAL_SHARES = {"existing": 0.7, "error_family": 0.2, "new_semantic_family": 0.1}
DEFAULT_LAYOUTS = {"single": "state_first", "stream": "stream_l1a"}
_KIND_OF_SCHEMA = {SCHEMA_SINGLE_REQUEST: "single", SCHEMA_STREAM: "stream"}
_DEFAULT_DOMAIN = {"single": "non_robot", "stream": "robot"}


# --------------------------------------------------------------------------
# 후보 순서 치환 증강 (docs/03 §3 "후보 순서 불변성은 구조로 보장하지 않는다 — 증강·평가로 다룬다"; analysis-nimble §3-3)
# --------------------------------------------------------------------------


def _permutation(rng_key: str, n: int) -> list[int]:
    """`rng_key`(레코드 id·seed·질문)로 결정되는 0..n−1의 순열 — 파일·프로세스와 무관하게 같은 레코드는 같은 순열."""
    digest = hashlib.sha256(rng_key.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(n))
    rng.shuffle(order)
    return order


def permute_candidates(record: dict, seed: int, *, questions: str = "choice") -> dict:
    """후보 순서를 레코드·seed로 정해진 순열로 바꾼 **새** 레코드 (라벨은 후보 id를 가리키므로 그대로 유효하다).

    단일 요청은 `choice` 질문의 `criteria` 순서(boolean은 true/false 고정, ordinal은 수준 순서가 뜻이라 두지 않는다),
    스트림은 틱마다 `request.candidates`의 동적 후보 목록(q_main·q_path)의 순서를 바꾼다. 같은 seed·레코드는 언제나 같은
    순열이고 seed마다 다르다. 평가는 원래 순서와 치환한 순서의 답 변화율(위치 편향)을 함께 적는다(:mod:`robo_jev.evaluate`).
    """
    out = copy.deepcopy(record)
    schema = out.get("schema_version")
    if schema == SCHEMA_SINGLE_REQUEST:
        rid = str(out["request"].get("request_id") or out.get("origin_group") or "")
        for question in out["request"]["questions"]:
            if question.get("type") != questions or len(question.get("criteria") or []) < 2:
                continue
            order = _permutation(f"{rid}|{seed}|{question['id']}", len(question["criteria"]))
            question["criteria"] = [question["criteria"][i] for i in order]
        return out
    if schema == SCHEMA_STREAM:
        eid = str(out.get("episode_id") or out.get("origin_group") or "")
        for tick in out["ticks"]:
            candidates = tick["request"].get("candidates") or {}
            for qid, entries in candidates.items():
                if len(entries) < 2:
                    continue
                order = _permutation(f"{eid}|{seed}|{tick.get('t', 0)}|{qid}", len(entries))
                candidates[qid] = [entries[i] for i in order]
        return out
    raise ValueError(f"permute_candidates: 알 수 없는 schema_version: {schema!r}")


# --------------------------------------------------------------------------
# 적재
# --------------------------------------------------------------------------


@dataclass
class Item:
    """학습 단위의 재료 하나 — 레코드와 그 직렬화."""

    index: int
    kind: str  # single | stream
    record_id: str
    split: str
    domain: str
    material: str
    record: dict = field(repr=False)
    layout: dict = field(repr=False)
    tokens: int = 0
    question_types: dict[str, str] = field(default_factory=dict, repr=False)
    source: str = ""
    manifest: str = ""  # 이 레코드를 가리킨 manifest 경로 (여러 manifest를 합쳐 학습할 때의 출처)
    file: str = ""  # manifest의 `files` 키 — 레코드가 든 JSONL 파일 (manifest 기준 상대 경로)
    file_sha256: str = ""  # 그 파일의 sha256 (manifest의 값 = 적재 때 대조한 실제 값) — sampler 위치의 레코드 정체


def _tag(record: dict, path: str) -> Any:
    node: Any = record
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def manifest_files(manifest: dict, where: str | Path = "manifest") -> dict[str, dict]:
    """manifest의 `files`를 `{경로: {sha256, …}}`로. dict(비로봇·D0·지금의 로봇 manifest)과 목록(`[{path, sha256, …}]`,
    이전 로봇 manifest)을 같은 뜻으로 읽는다 — 어느 쪽이든 경로와 sha256이 있어야 한다."""
    files = manifest.get("files")
    if isinstance(files, list):
        normalised: dict[str, dict] = {}
        for index, entry in enumerate(files):
            if not isinstance(entry, dict) or not entry.get("path"):
                raise ValueError(f"{where}: files[{index}].path가 없다")
            normalised[str(entry["path"])] = {key: value for key, value in entry.items() if key != "path"}
        files = normalised
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{where}: files 항목이 없다")
    return files


def sha256_of(path: Path) -> str:
    """파일의 sha256 (manifest 대조, checkpoint의 manifest 참조, tokenizer 파일 대조가 같은 함수를 쓴다)."""
    return sha256_of_file(path)


def load_items(
    manifest_path: str | Path,
    *,
    tokenizer: Any,
    splits: tuple[str, ...] | list[str] = ("train",),
    layouts: dict[str, str] | None = None,
    window_ticks: int = WINDOW_TICKS,
    max_state_tokens: int | None = 2048,
    max_total_tokens: int | None = 8192,
    stream_max_ticks: int | None = None,
    domain_tag: str = "provenance.domain",
    material_tag: str = "provenance.material",
    domain: str | None = None,
    material: str | None = None,
    index_offset: int = 0,
    permute_seed: int | None = None,
    files: list[str] | None = None,
) -> list[Item]:
    """manifest의 파일들을 읽어 직렬화된 :class:`Item` 목록으로 (모듈 설명 참조).

    ``permute_seed``가 있으면 레코드마다 :func:`permute_candidates` 로 후보 순서를 바꾼 뒤 직렬화한다(학습 증강; 라벨은
    id 기준이라 그대로). `Item.record`도 치환된 레코드다 — 라벨·틱 종류 계산은 순서와 무관하다. ``files``는 manifest 파일
    키의 fnmatch 패턴 목록 — 맞는 파일만 읽는다(예: 로봇 batch의 에피소드만: ``["episodes/*/streams.jsonl"]``).
    ``stream_max_ticks``는 CPU 검사용이다 — 에피소드를 앞 N틱으로 자른다(실제 학습에서는 `None`).
    ``domain``·``material``은 **이 manifest의** 기본 태그다(학습 설정 `dataset_manifests[].domain`): 레코드에
    태그(`domain_tag`·`material_tag`)가 없을 때 종류별 기본값(스트림 = robot, 단일 = non_robot; existing) 대신
    쓴다 — 레코드 자체의 태그가 있으면 그것이 이긴다. ``index_offset``은 여러 manifest를 이어 붙일 때의 첫
    `Item.index`다(:func:`load_manifests`).

    **레코드는 여기서 계약 검증을 지난다.** :func:`~robo_jev.model.serialize.serialize_request` 가 먼저
    :func:`robo_jev.contracts.validate_record` 를 부르므로(입력 영역의 비입력 키·라벨 구조·후보 참조를
    거절), 잘못된 레코드가 학습 loop에 닿기 전의 **유일한** 관문이 이 적재다 — 학습 loop는 layout과
    라벨을 그대로 믿는다. manifest의 sha256 대조와 JSONL 읽기는 :mod:`robo_jev.data` 의 것
    (`data/generate.py`의 `write_dataset`·`data/robot_episodes.py`의 `build_manifest`가 쓰는 manifest,
    `data/validate.py`의 `_read_jsonl`)과 **일부러 겹친다**: 학습 코드는 generator·QA를 import하지 않는다는
    경계(docs/06 §1) 때문에 읽기 쪽을 여기 다시 둔다. 두 쪽의 manifest 형식(`files: {경로: {sha256, …}}`,
    :func:`manifest_files` 가 이전 로봇 manifest의 목록 꼴도 받는다)이 바뀌면 같이 고친다.
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"데이터 manifest가 없다: {manifest_file}")
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"splits: {list(SPLITS)} 중에서 골라야 한다 (받은 값: {split!r})")
    if stream_max_ticks is not None and int(stream_max_ticks) < 1:
        raise ValueError(f"stream_max_ticks: 1 이상이거나 None이어야 한다 (받은 값: {stream_max_ticks})")
    if domain is not None and domain not in DOMAINS:
        raise ValueError(f"{manifest_file}: domain은 {list(DOMAINS)} 중 하나여야 한다 (받은 값: {domain!r})")
    if material is not None and material not in MATERIALS:
        raise ValueError(f"{manifest_file}: material은 {list(MATERIALS)} 중 하나여야 한다 (받은 값: {material!r})")
    layouts = {**DEFAULT_LAYOUTS, **(layouts or {})}
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    files_all = manifest_files(manifest, manifest_file)

    items: list[Item] = []
    if files is not None:
        from fnmatch import fnmatch

        patterns = list(files)
        selected = {name: entry for name, entry in files_all.items() if any(fnmatch(name, pattern) for pattern in patterns)}
        if not selected:
            raise ValueError(f"{manifest_file}: files 패턴 {patterns}에 맞는 파일이 없다 (있는 것: {list(files_all)[:5]}…)")
        files_all = selected
    for name, entry in files_all.items():
        path = manifest_file.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"{manifest_file}: 파일이 없다: {path}")
        expected = (entry or {}).get("sha256")
        if not expected:
            raise ValueError(f"{manifest_file}: files[{name}].sha256이 없다")
        actual = sha256_of(path)
        if actual != expected:
            raise ValueError(f"{path}: sha256이 manifest와 다르다 ({actual[:12]}… != {expected[:12]}…)")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            where = f"{name}:{line_number + 1}"
            kind = _KIND_OF_SCHEMA.get(record.get("schema_version"))
            if kind is None:
                raise ValueError(f"{where}: 알 수 없는 schema_version: {record.get('schema_version')!r}")
            split = record.get("split")
            if split not in SPLITS:
                raise ValueError(f"{where}: split이 없거나 알 수 없다: {split!r}")
            if split not in splits:
                continue
            record_domain = _tag(record, domain_tag)
            if record_domain is None:
                record_domain = domain if domain is not None else _DEFAULT_DOMAIN[kind]
            elif record_domain not in DOMAINS:
                raise ValueError(f"{where}: {domain_tag}는 {list(DOMAINS)} 중 하나여야 한다 (받은 값: {record_domain!r})")
            record_material = _tag(record, material_tag)
            if record_material is None:
                record_material = material if material is not None else MATERIALS[0]
            elif record_material not in MATERIALS:
                raise ValueError(f"{where}: {material_tag}는 {list(MATERIALS)} 중 하나여야 한다 (받은 값: {record_material!r})")
            if permute_seed is not None:
                record = permute_candidates(record, int(permute_seed))
            if kind == "single":
                layout = serialize_request(
                    record, tokenizer, layout=layouts["single"],
                    max_state_tokens=max_state_tokens, max_total_tokens=max_total_tokens,
                )  # fmt: skip
                record_id = str(record["request"].get("request_id") or where)
                question_types = {q["id"]: q["type"] for q in record["request"]["questions"]}
            else:
                if stream_max_ticks is not None:
                    record["ticks"] = record["ticks"][: int(stream_max_ticks)]
                layout = serialize_request(record, tokenizer, layout=layouts["stream"], window_ticks=window_ticks)
                record_id = str(record.get("episode_id") or where)
                question_types = {qid: spec["type"] for qid, spec in QUESTION_SET_V0.items()}
            items.append(
                Item(
                    index=int(index_offset) + len(items), kind=kind, record_id=record_id, split=split,
                    domain=record_domain, material=record_material, record=record, layout=layout,
                    tokens=len(layout["tokens"]), question_types=question_types, source=where,
                    manifest=str(manifest_path), file=name, file_sha256=actual,
                )  # fmt: skip
            )
    return items


def load_manifests(manifests: list[dict[str, Any]], **kwargs: Any) -> list[Item]:
    """여러 manifest(`[{"path", "domain", "material"}, …]`, 학습 설정 `dataset_manifests`)의 레코드를 manifest 순서로
    이어 붙인 :class:`Item` 목록. `Item.index`는 전체에서 이어지고(sampler의 열쇠), 출처는 `Item.manifest`다.
    `kwargs`는 :func:`load_items` 의 공통 인자(tokenizer·splits·layouts·…)다."""
    items: list[Item] = []
    for entry in manifests:
        items.extend(
            load_items(
                entry["path"], domain=entry.get("domain"), material=entry.get("material"), files=entry.get("files"), index_offset=len(items),
                **kwargs,
            )
        )
    return items


# --------------------------------------------------------------------------
# 틱 종류·가중치·유효 라벨
# --------------------------------------------------------------------------


def _goal_version(tick: dict) -> int | None:
    goal = (tick.get("request") or {}).get("state", {}).get("goal")
    if isinstance(goal, dict) and goal.get("version") is not None:
        return int(goal["version"])
    return None


def tick_class(record: dict, index: int, *, steady_min_held_ticks: int = 3) -> str:
    """틱 `index`의 종류 — 그 틱의 레코드·라벨과 직전 틱에서만 정한다 (모듈 설명 참조)."""
    ticks = record["ticks"]
    tick = ticks[index]
    request = tick["request"]
    state = request.get("state") or {}
    if index > 0:
        version, previous = _goal_version(tick), _goal_version(ticks[index - 1])
        if version is not None and previous is not None:
            if version > previous:
                return "goal_change"
        else:  # 버전이 없으면 직전 틱과 이 틱 사이에 들어온 지시로 본다 (직렬화의 fallback과 같은 규칙)
            low, high = int(ticks[index - 1].get("sim_ms", 0)), int(tick.get("sim_ms", 0))
            for instruction in (record.get("prefix") or {}).get("instructions", [])[1:]:
                if low < int(instruction.get("t_ms", 0)) <= high:
                    return "goal_change"
    adopted = tick.get("adopted") or {}
    if state.get("events") or adopted.get("switch") is True:
        return "event"
    commitment = request.get("commitment")
    if isinstance(commitment, dict) and int(commitment.get("held_ticks", 0)) >= int(steady_min_held_ticks):
        return "steady"
    return "other"


def tick_weights(record: dict, *, weights: dict[str, float], steady_min_held_ticks: int = 3) -> list[float]:
    """틱마다 종류의 가중치. `weights`는 네 종류를 모두 0 이상으로 주어야 한다."""
    missing = [name for name in TICK_CLASSES if name not in weights]
    if missing:
        raise ValueError(f"tick_weights: 종류마다 가중치가 필요하다 — 없는 것: {missing}")
    for name in TICK_CLASSES:
        if float(weights[name]) < 0:
            raise ValueError(f"tick_weights.{name}: 0 이상이어야 한다 (받은 값: {weights[name]})")
    return [
        float(weights[tick_class(record, index, steady_min_held_ticks=steady_min_held_ticks)])
        for index in range(len(record["ticks"]))
    ]


def _label_contributes(label: dict) -> bool:
    """:func:`robo_jev.loss.label_loss`가 `None`을 내는 조건(mask=false, 근거 없는 사건)과 :func:`robo_jev.loss.judgment_loss`
    가 상태를 세지 않는 조건(weight 0)의 거울."""
    if label.get("mask", True) is False:
        return False
    if label.get("kind") == "event" and int(label.get("successes", 0)) + int(label.get("failures", 0)) == 0:
        return False
    if float(label.get("weight", 1.0)) <= 0:
        return False
    return True


def valid_label_ticks(record: dict) -> list[bool]:
    """틱마다 기여하는 라벨이 하나라도 있는지 — 모델 없이 라벨만으로 (손실 정규화 분모에 쓴다)."""
    return [any(_label_contributes(label) for label in tick.get("labels", [])) for tick in record["ticks"]]


def valid_single(record: dict) -> bool:
    """단일 요청 레코드에 기여하는 라벨이 있는지 — 같은 규칙 (비로봇 상태 수에 쓴다)."""
    return any(_label_contributes(label) for label in record.get("labels", []))


# --------------------------------------------------------------------------
# 혼합 sampler
# --------------------------------------------------------------------------


@dataclass
class Unit:
    """accumulation 단위 하나: 에피소드 하나 또는 단일 요청 `microbatch`개."""

    index: int  # 뽑힌 순서 (sampler 위치)
    kind: str  # single | stream
    domain: str
    items: list[int]
    materials: list[str]
    tokens: int


def _bucket(domain: str, material: str) -> str:
    return f"{domain}/{material}"


def record_sources(items: list[Item]) -> list[dict[str, Any]]:
    """sampler 위치가 가리키는 레코드들의 출처 — index 순서로 같은 파일(이름·sha256)의 연속 구간마다
    ``{file, sha256, first_index, items}``. 같은 출처 목록이면 같은 index가 같은 레코드를 가리킨다(적재는 manifest
    순서·줄 순서로 결정적이고 파일 내용은 sha256으로 고정된다)."""
    sources: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda entry: entry.index):
        last = sources[-1] if sources else None
        if last is not None and (last["file"], last["sha256"]) == (item.file, item.file_sha256):
            last["items"] += 1
        else:
            sources.append({"file": item.file, "sha256": item.file_sha256, "first_index": item.index, "items": 1})
    return sources


def _describe_source_differences(saved: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for position in range(max(len(saved), len(current))):
        before = saved[position] if position < len(saved) else None
        after = current[position] if position < len(current) else None
        if before == after:
            continue
        if before is None:
            out.append(f"[{position}] {after['file']}: 저장 위치에 없던 파일 ({after['items']}개)")
        elif after is None:
            out.append(f"[{position}] {before['file']}: 지금 없는 파일 ({before['items']}개)")
        elif before["sha256"] != after["sha256"] and before["file"] == after["file"]:
            out.append(f"[{position}] {before['file']}: sha256 {str(before['sha256'])[:12]}… → {str(after['sha256'])[:12]}…")
        else:
            out.append(f"[{position}] {before['file']} ({before['items']}개, index {before['first_index']}부터) → {after['file']} ({after['items']}개, index {after['first_index']}부터)")
    return out


class MixedSampler:
    """두 축으로 단위를 뽑는 결정적·재개 가능한 sampler (모듈 설명 참조).

    * :meth:`draw_step` — step의 단위들: 로봇, 비로봇, 로봇, … 번갈아(한쪽이 없으면 있는 쪽만).
    * :meth:`draw` — 분야 하나의 단위: 스트림 분야면 에피소드 하나, 단일 요청 분야면
      ``nonrobot_tokens_per_unit``까지 묶은 microbatch(예산을 넘기는 레코드는 되돌려 다음 단위에).
    """

    def __init__(
        self,
        items: list[Item],
        *,
        material_shares: dict[str, float] | None = None,
        seed: int = 0,
        nonrobot_tokens_per_unit: int = 8192,
    ) -> None:
        if not items:
            raise ValueError("items: 뽑을 레코드가 하나도 없다")
        if int(nonrobot_tokens_per_unit) < 1:
            raise ValueError(f"nonrobot_tokens_per_unit: 1 이상이어야 한다 (받은 값: {nonrobot_tokens_per_unit})")
        shares = dict(DEFAULT_MATERIAL_SHARES if material_shares is None else material_shares)
        unknown = [name for name in shares if name not in MATERIALS]
        if unknown:
            raise ValueError(f"material_shares: 알 수 없는 묶음 {unknown} (허용: {list(MATERIALS)})")
        for name in MATERIALS:
            if float(shares.get(name, 0.0)) < 0:
                raise ValueError(f"material_shares.{name}: 0 이상이어야 한다")
        if abs(sum(float(shares.get(name, 0.0)) for name in MATERIALS) - 1.0) > 1e-6:
            raise ValueError(f"material_shares: 합이 1이어야 한다 (받은 값: {shares})")
        self.items = {item.index: item for item in items}  # 단위는 item.index로 가리킨다 (목록 위치가 아니라)
        if len(self.items) != len(items):
            raise ValueError("items: index가 중복된 레코드가 있다")
        self.material_shares = {name: float(shares.get(name, 0.0)) for name in MATERIALS}
        self.tokens_per_unit = int(nonrobot_tokens_per_unit)
        self.seed = int(seed)

        self.buckets: dict[str, list[int]] = {}
        kinds: dict[str, set[str]] = {}
        for item in items:
            self.buckets.setdefault(_bucket(item.domain, item.material), []).append(item.index)
            kinds.setdefault(item.domain, set()).add(item.kind)
        for domain, seen in kinds.items():
            if len(seen) > 1:
                raise ValueError(
                    f"domain {domain!r}에 단일 요청과 스트림이 섞여 있다 — 한 분야 태그 안의 레코드는 한 종류여야 "
                    "한 단위(에피소드 하나 / 단일 요청 microbatch)가 균일하다"
                )
        self._kind_of_domain = {domain: next(iter(seen)) for domain, seen in kinds.items()}
        self.domains: tuple[str, ...] = tuple(domain for domain in DOMAINS if domain in self._kind_of_domain)
        self.sources = record_sources(items)

        self.rng = random.Random(self.seed)
        self._drawn = 0
        self._tokens = {domain: 0 for domain in DOMAINS}
        self._counts = {bucket: 0 for bucket in self._ordered_buckets()}
        self._cursors = {bucket: {"order": [], "cursor": 0, "epoch": 0} for bucket in self._ordered_buckets()}

    # -- 묶음 --

    def _ordered_buckets(self) -> list[str]:
        return [_bucket(d, m) for d in DOMAINS for m in MATERIALS if _bucket(d, m) in self.buckets]

    def effective_material_shares(self, domain: str) -> dict[str, float]:
        """비어 있지 않은 묶음 위에서 재정규화한 70/20/10 (모두 0이면 균등)."""
        present = [name for name in MATERIALS if _bucket(domain, name) in self.buckets]
        raw = {name: self.material_shares[name] for name in present}
        total = sum(raw.values())
        if total <= 0 and present:
            raw = {name: 1.0 for name in present}
            total = float(len(present))
        return {name: (raw[name] / total if name in raw else 0.0) for name in MATERIALS}

    def _next_from(self, bucket: str) -> int:
        cursor = self._cursors[bucket]
        if cursor["cursor"] >= len(cursor["order"]):
            order = list(self.buckets[bucket])
            self.rng.shuffle(order)
            cursor["order"], cursor["cursor"], cursor["epoch"] = order, 0, cursor["epoch"] + 1
        index = cursor["order"][cursor["cursor"]]
        cursor["cursor"] += 1
        return index

    def _unread(self, bucket: str) -> None:
        """방금 뽑은 레코드를 되돌린다 — 다음 단위의 그 묶음에서 먼저 나온다."""
        self._cursors[bucket]["cursor"] -= 1

    def _choose_material(self, domain: str) -> str:
        shares = self.effective_material_shares(domain)
        present = [name for name in MATERIALS if _bucket(domain, name) in self.buckets]
        draw = self.rng.random()
        cumulative = 0.0
        for name in present:
            cumulative += shares[name]
            if draw < cumulative:
                return name
        return present[-1]

    def _draw_one(self, domain: str) -> tuple[int, str]:
        material = self._choose_material(domain)
        return self._next_from(_bucket(domain, material)), material

    # -- 공개 API --

    def draw(self, domain: str) -> Unit:
        """분야 `domain`의 다음 단위: 에피소드 하나(스트림) 또는 토큰 예산까지 묶은 단일 요청들."""
        if domain not in self._kind_of_domain:
            raise ValueError(f"domain: {list(self.domains)}에 없는 분야다 (받은 값: {domain!r})")
        kind = self._kind_of_domain[domain]
        indices: list[int] = []
        materials: list[str] = []
        tokens = 0
        while True:
            index, material = self._draw_one(domain)
            size = self.items[index].tokens
            if indices and tokens + size > self.tokens_per_unit:
                self._unread(_bucket(domain, material))  # 예산을 넘긴다 — 다음 단위로
                break
            indices.append(index)
            materials.append(material)
            tokens += size
            self._counts[_bucket(domain, material)] += 1
            if kind == "stream" or tokens >= self.tokens_per_unit:
                break
        self._tokens[domain] += tokens
        unit = Unit(index=self._drawn, kind=kind, domain=domain, items=indices, materials=materials, tokens=tokens)
        self._drawn += 1
        return unit

    def draw_step(self, units: int) -> list[Unit]:
        """한 step의 단위들: 로봇, 비로봇, 로봇, … 번갈아. 한 분야가 없으면 있는 분야만."""
        if int(units) < 1:
            raise ValueError(f"units: 1 이상이어야 한다 (받은 값: {units})")
        order = [domain for domain in DOMAINS if domain in self._kind_of_domain]
        return [self.draw(order[position % len(order)]) for position in range(int(units))]

    def realized(self) -> dict[str, Any]:
        """실현 비중: 토큰(로봇/비로봇), 단위·레코드 수(분야·묶음), epoch, 재정규화한 목표."""
        total_tokens = sum(self._tokens.values())
        total_units = sum(self._counts.values())
        by_domain = {d: sum(n for b, n in self._counts.items() if b.startswith(d + "/")) for d in DOMAINS}
        by_material = {m: sum(n for b, n in self._counts.items() if b.endswith("/" + m)) for m in MATERIALS}
        return {
            "drawn": self._drawn,
            "tokens": dict(self._tokens),
            "token_share": {
                d: (self._tokens[d] / total_tokens if total_tokens else 0.0) for d in DOMAINS
            },
            "units": dict(self._counts),
            "unit_share": {
                "domain": {d: (n / total_units if total_units else 0.0) for d, n in by_domain.items()},
                "material": {m: (n / total_units if total_units else 0.0) for m, n in by_material.items()},
            },
            "epochs": {bucket: cursor["epoch"] for bucket, cursor in self._cursors.items()},
            "effective_material_shares": {d: self.effective_material_shares(d) for d in self.domains},
        }

    def state_dict(self) -> dict[str, Any]:
        """재개용 위치 — 기본 자료형만. ``sources``는 위치의 index가 가리키는 레코드의 출처(:func:`record_sources`)다."""
        version, internal, gauss_next = self.rng.getstate()
        return {
            "drawn": self._drawn,
            "tokens": dict(self._tokens),
            "counts": dict(self._counts),
            "cursors": {
                bucket: {"order": list(cursor["order"]), "cursor": cursor["cursor"], "epoch": cursor["epoch"]}
                for bucket, cursor in self._cursors.items()
            },
            "rng": [int(version), [int(v) for v in internal], gauss_next],
            "sources": [dict(source) for source in self.sources],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """저장 위치를 싣는다. 위치의 index가 가리키는 레코드가 지금의 것과 다르면(파일별 sha256·적재 순서·수) 거절한다 —
        묶음·index가 같아도 내용이 바뀐 데이터(정답 하나를 고친 사본 등)에 옛 위치를 이어 붙이지 않는다."""
        for key in ("drawn", "tokens", "counts", "cursors", "rng", "sources"):
            if key not in state:
                raise ValueError(f"sampler.{key}: 없다")
        saved_sources = [dict(source) for source in state["sources"]]
        if saved_sources != self.sources:
            differences = _describe_source_differences(saved_sources, self.sources)
            raise ValueError(
                f"sampler.sources: 저장 위치의 레코드 파일과 지금 적재한 파일이 다르다: {'; '.join(differences)} — "
                "같은 내용의 데이터로 재개해야 한다 (데이터를 바꾸는 학습은 새 run으로 시작한다)"
            )
        if set(state["cursors"]) != set(self._cursors):
            raise ValueError(
                f"sampler.cursors: 묶음이 다르다 (저장: {sorted(state['cursors'])}, 지금: {sorted(self._cursors)}) — "
                "같은 데이터·split·태그로 재개해야 한다"
            )
        self._drawn = int(state["drawn"])
        self._tokens = {domain: int(state["tokens"].get(domain, 0)) for domain in DOMAINS}
        self._counts = {bucket: int(state["counts"].get(bucket, 0)) for bucket in self._cursors}
        for bucket, cursor in state["cursors"].items():
            order = [int(index) for index in cursor["order"]]
            for index in order:
                if index not in self.buckets[bucket]:
                    raise ValueError(f"sampler.cursors[{bucket}]: 레코드 {index}가 지금 묶음에 없다")
            self._cursors[bucket] = {"order": order, "cursor": int(cursor["cursor"]), "epoch": int(cursor["epoch"])}
        version, internal, gauss_next = state["rng"]
        self.rng.setstate((int(version), tuple(int(v) for v in internal), gauss_next))
