"""학습 데이터의 읽기 전용 적재와 혼합 sampler (docs/04 §2·§7, docs/08 §8, docs/06 Task 5).

**적재.** :func:`load_items` 는 데이터 manifest(`files: {이름: {sha256, …}}`)가 가리키는 JSONL 파일을
manifest 순서·줄 순서로 읽고(sha256 대조), `split`으로 거른 뒤 레코드를 그 종류의 배치로 직렬화한다
(`judgment-v0` → ``state_first``, `stream-v0` → ``stream_l1a``). 직렬화를 **적재 시점에 고정된 순서로**
하는 이유는 공백 tokenizer가 id를 처음 본 순서로 주기 때문이다 — 재개한 프로세스도 같은 순서로
읽어 같은 토큰 id를 얻는다(실제 tokenizer는 순서와 무관하다). 레코드는 라벨·틱 종류 계산에 쓰려고
그대로 들고 있되, 모델에는 직렬화된 layout만 간다.

**두 sampler 축 (docs/04 §2, §7; docs/09).**

* 로봇/비로봇 — **토큰**으로 관리한다(시작 60/40). 단위를 뽑을 때마다 지금까지의 실현 로봇 토큰
  비중이 목표보다 낮으면 로봇, 아니면 비로봇을 고른다(단위 하나의 크기 안에서 목표를 따른다).
  레코드의 분야는 태그(`provenance.domain`, 기본: 스트림 = robot, 단일 요청 = non_robot)로 정한다.
* 기존 자료/오류 계열/새 의미 계열 — `70/20/10`, 레코드의 provenance 태그(`provenance.material`,
  없으면 기존 자료)로 나눈다. 비어 있는 묶음은 재정규화하고(D0/D1에서는 뒤의 두 묶음이 비어 있을
  수 있다) 실현 비중을 기록한다 — 실패하지 않는다.

한 단위는 에피소드 하나(스트림 = accumulation 단위, docs/03 §5) 또는 단일 요청 ``microbatch`` 개다.
묶음 안에서는 epoch마다 섞어 한 번씩 뽑는다. 모든 무작위성은 seed로 만든 `random.Random` 하나에서
나오고, :meth:`MixedSampler.state_dict` 가 위치(뽑은 수·묶음별 순서·cursor·epoch·실현 토큰·RNG)를
기본 자료형으로 돌려주어 재개할 수 있다.

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

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robo_jev.contracts import QUESTION_SET_V0, SCHEMA_SINGLE_REQUEST, SCHEMA_STREAM, SPLITS
from robo_jev.model.serialize import WINDOW_TICKS, serialize_request

__all__ = [
    "DOMAINS",
    "MATERIALS",
    "TICK_CLASSES",
    "Item",
    "MixedSampler",
    "Unit",
    "load_items",
    "tick_class",
    "tick_weights",
    "valid_label_ticks",
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


def _tag(record: dict, path: str) -> Any:
    node: Any = record
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
) -> list[Item]:
    """manifest의 파일들을 읽어 직렬화된 :class:`Item` 목록으로 (모듈 설명 참조).

    ``stream_max_ticks``는 CPU 검사용이다 — 에피소드를 앞 N틱으로 자른다(실제 학습에서는 `None`).
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"데이터 manifest가 없다: {manifest_file}")
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"splits: {list(SPLITS)} 중에서 골라야 한다 (받은 값: {split!r})")
    if stream_max_ticks is not None and int(stream_max_ticks) < 1:
        raise ValueError(f"stream_max_ticks: 1 이상이거나 None이어야 한다 (받은 값: {stream_max_ticks})")
    layouts = {**DEFAULT_LAYOUTS, **(layouts or {})}
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{manifest_file}: files 항목이 없다")

    items: list[Item] = []
    for name, entry in files.items():
        path = manifest_file.parent / name
        if not path.is_file():
            raise FileNotFoundError(f"{manifest_file}: 파일이 없다: {path}")
        expected = (entry or {}).get("sha256")
        if not expected:
            raise ValueError(f"{manifest_file}: files[{name}].sha256이 없다")
        actual = _sha256(path)
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
            domain = _tag(record, domain_tag)
            if domain is None:
                domain = _DEFAULT_DOMAIN[kind]
            elif domain not in DOMAINS:
                raise ValueError(f"{where}: {domain_tag}는 {list(DOMAINS)} 중 하나여야 한다 (받은 값: {domain!r})")
            material = _tag(record, material_tag)
            if material is None:
                material = MATERIALS[0]
            elif material not in MATERIALS:
                raise ValueError(f"{where}: {material_tag}는 {list(MATERIALS)} 중 하나여야 한다 (받은 값: {material!r})")
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
                    index=len(items), kind=kind, record_id=record_id, split=split, domain=domain,
                    material=material, record=record, layout=layout, tokens=len(layout["tokens"]),
                    question_types=question_types, source=where,
                )  # fmt: skip
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
    """:func:`robo_jev.loss.label_loss`가 `None`을 내는 조건의 거울 (mask=false, 근거 없는 사건)."""
    if label.get("mask", True) is False:
        return False
    if label.get("kind") == "event" and int(label.get("successes", 0)) + int(label.get("failures", 0)) == 0:
        return False
    return True


def valid_label_ticks(record: dict) -> list[bool]:
    """틱마다 기여하는 라벨이 하나라도 있는지 — 모델 없이 라벨만으로 (구간 정규화 분모에 쓴다)."""
    return [any(_label_contributes(label) for label in tick.get("labels", [])) for tick in record["ticks"]]


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


class MixedSampler:
    """두 축으로 단위를 뽑는 결정적·재개 가능한 sampler (모듈 설명 참조)."""

    def __init__(
        self,
        items: list[Item],
        *,
        robot_token_share: float = 0.6,
        material_shares: dict[str, float] | None = None,
        seed: int = 0,
        microbatch: int = 1,
    ) -> None:
        if not items:
            raise ValueError("items: 뽑을 레코드가 하나도 없다")
        if not 0.0 <= float(robot_token_share) <= 1.0:
            raise ValueError(f"robot_token_share: [0, 1] 안이어야 한다 (받은 값: {robot_token_share})")
        if int(microbatch) < 1:
            raise ValueError(f"microbatch: 1 이상이어야 한다 (받은 값: {microbatch})")
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
        self.robot_token_share = float(robot_token_share)
        self.material_shares = {name: float(shares.get(name, 0.0)) for name in MATERIALS}
        self.microbatch = int(microbatch)
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

        self.rng = random.Random(self.seed)
        self._drawn = 0
        self._tokens = {domain: 0 for domain in DOMAINS}
        self._counts = {bucket: 0 for bucket in self._ordered_buckets()}
        self._cursors = {bucket: {"order": [], "cursor": 0, "epoch": 0} for bucket in self._ordered_buckets()}

    # -- 묶음 --

    def _ordered_buckets(self) -> list[str]:
        return [_bucket(d, m) for d in DOMAINS for m in MATERIALS if _bucket(d, m) in self.buckets]

    def _has(self, domain: str) -> bool:
        return any(bucket.startswith(domain + "/") for bucket in self.buckets)

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

    def _choose_domain(self) -> str:
        total = sum(self._tokens.values())
        if total > 0:
            want_robot = self._tokens["robot"] < self.robot_token_share * total
        else:
            want_robot = self.robot_token_share > 0.0
        domain = "robot" if want_robot else "non_robot"
        if not self._has(domain):
            domain = "non_robot" if domain == "robot" else "robot"
        return domain

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

    # -- 공개 API --

    def draw(self) -> Unit:
        """다음 accumulation 단위."""
        domain = self._choose_domain()
        kind = self._kind_of_domain[domain]
        count = 1 if kind == "stream" else self.microbatch
        indices: list[int] = []
        materials: list[str] = []
        for _ in range(count):
            material = self._choose_material(domain)
            bucket = _bucket(domain, material)
            indices.append(self._next_from(bucket))
            materials.append(material)
            self._counts[bucket] += 1
        tokens = sum(self.items[index].tokens for index in indices)
        self._tokens[domain] += tokens
        unit = Unit(index=self._drawn, kind=kind, domain=domain, items=indices, materials=materials, tokens=tokens)
        self._drawn += 1
        return unit

    def realized(self) -> dict[str, Any]:
        """실현 비중: 토큰(로봇/비로봇), 단위 수(분야·묶음), epoch, 재정규화한 목표."""
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
            "effective_material_shares": {
                d: self.effective_material_shares(d) for d in DOMAINS if self._has(d)
            },
        }

    def state_dict(self) -> dict[str, Any]:
        """재개용 위치 — 기본 자료형만."""
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
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("drawn", "tokens", "counts", "cursors", "rng"):
            if key not in state:
                raise ValueError(f"sampler.{key}: 없다")
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
