"""증분 스트림 직렬화 — 틱마다 **새 토큰만** 만든다 (Task R4 A1; docs/08 §3.1의 "변화분은 입력의 정의다").

:func:`robo_jev.model.serialize.serialize_request` 는 레코드 **전체**를 받아 한 번에 토큰을 만든다 — 학습·평가는 녹화된
에피소드를 갖고 있으니 그것으로 충분했다. 루프 안의 정책은 에피소드의 끝을 모른 채 틱 하나가 올 때마다 그 틱의 토큰을
붙여야 한다. 이 모듈의 :class:`IncrementalStreamSerializer` 는 같은 서식(ts0.6)을 같은 조각·같은 줄 함수·같은 변화분
상태로 **틱 단위로** 낸다. 새 서식이 아니다: 검사가 전체 직렬화와 토큰 단위로 대조한다(`tests/test_incremental_serializer.py`).

규칙은 전체 직렬화의 것을 그대로 옮긴 것이다.

* prefix = 첫 지시 + 질문 세트·표지 선언·질문 머리·정적 후보 + **첫 틱 상태의** 영역·장면 줄. 그래서 prefix는 첫 틱이 와야
  만들 수 있다(정책은 `t == 0`에서 만든다).
* 틱 = [그 틱에 놓이는 지시][t][goal][물체 소개·동적 줄(변화분)][영역·장면(바뀐 틱만)][robot][exec][사건][경유점][extra]
  [commitment][실행 이력][동적 후보] 뒤에 결정 표지. 조각은 따로 토큰화해 이어 붙인다(전체 직렬화와 같은 계산).
* 도중 지시: 알고 있는 지시(생성기가 prefix에 덧붙인 것)는 `_instruction_slots`의 규칙대로 그 버전을 처음 실은 틱 앞에
  놓고, 모르는 버전이 목표에 나타나면 생성기 :func:`robo_jev.data.episode.append_tick` 의 규칙(버전·그 틱의 `sim_ms`·목표
  텍스트)으로 지시를 **만들어** 붙인다 — 레코드가 나중에 갖게 될 prefix와 같은 줄이다.

토큰 index는 전체 스트림 기준의 **절대** index다(prefix 0부터; 결정 표지도 센다) — 정적 후보 경계는 prefix hidden을,
동적 후보 경계는 그 틱 몸통 hidden을 가리키므로 readout이 전체 직렬화의 `ticks[k]` 항목과 같은 수를 본다.

이 모듈은 하네스·시뮬레이터·생성기를 import하지 않는다 (docs/06 §1).
"""

from __future__ import annotations

import copy
from typing import Any

from robo_jev.contracts import SCHEMA_STREAM, validate_record
from robo_jev.model import serialize as _s

__all__ = ["IncrementalStreamSerializer", "STREAM_TICK_FIELDS", "STREAM_REQUEST_FIELDS", "project_stream_tick"]

#: 틱 겉봉투 가운데 모델 입력에 드는 필드 (계약 `_TICK_FIELDS`와 같다).
STREAM_TICK_FIELDS = ("t", "sim_ms", "observed_at_ms", "obs_age_ms")
#: 틱 요청 가운데 모델 입력에 드는 필드 (계약 `_STREAM_REQUEST_FIELDS`와 같다).
STREAM_REQUEST_FIELDS = ("state", "exec_history", "commitment", "candidates")

#: 계약 검사용 최소 prefix — 틱 하나를 스트림 레코드 꼴로 검사할 때만 쓴다(직렬화에는 들어가지 않는다).
_CHECK_PREFIX = {"instructions": [{"version": 1, "t_ms": 0, "text": "-"}], "question_set": "qs-v0"}


def project_stream_tick(request: dict[str, Any]) -> dict[str, Any]:
    """하네스의 틱 요청(또는 레코드의 틱) → 모델 입력 투영 하나 (:func:`robo_jev.contracts.model_input` 의 스트림 규칙을 틱 하나에).

    하네스 블록·라벨·채택·ACK·`model_output`은 어떤 경로로도 통과하지 못한다 — 허용 목록만 **깊은 복사**로 옮기고, 계약 검사
    (:func:`validate_record`)를 틱 하나짜리 레코드로 돌려 입력 영역의 비입력 키·라벨 구조를 거절한다. 원본은 그대로다.
    """
    tick = {key: copy.deepcopy(request[key]) for key in STREAM_TICK_FIELDS if key in request}
    inner = request.get("request")
    if not isinstance(inner, dict):
        raise ValueError("request: 틱 요청에는 `request`(state·exec_history·commitment·candidates)가 있어야 한다")
    tick["request"] = {key: copy.deepcopy(inner[key]) for key in STREAM_REQUEST_FIELDS if key in inner}
    validate_record({"schema_version": SCHEMA_STREAM, "prefix": copy.deepcopy(_CHECK_PREFIX), "ticks": [tick]})
    return tick


class IncrementalStreamSerializer:
    """에피소드 하나의 증분 직렬화 (모듈 설명 참조). 에피소드마다 하나를 만든다."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        question_set: str,
        instructions: list[dict[str, Any]],
        first_state: dict[str, Any],
        window_ticks: int = _s.WINDOW_TICKS,
        delta_rules: dict[str, Any] | None = None,
    ) -> None:
        if question_set not in _s.QUESTION_SETS:
            raise ValueError(f"question_set: 직렬화할 수 없는 질문 세트다: {question_set!r} (아는 것: {list(_s.QUESTION_SETS)})")
        if not instructions:
            raise ValueError("instructions: 시작 지시가 하나 이상 있어야 한다")
        if int(window_ticks) < 1:
            raise ValueError(f"window_ticks: 1 이상이어야 한다 (받은 값: {window_ticks})")
        rules = dict(_s.DELTA_RULES)
        for key, value in (delta_rules or {}).items():
            if key not in _s.DELTA_RULES:
                raise ValueError(f"delta_rules.{key}: 모르는 규칙이다 (아는 것: {list(_s.DELTA_RULES)})")
            rules[key] = value
        self.tokenizer = tokenizer
        self.question_set_id = str(question_set)
        self.question_set = _s.QUESTION_SETS[self.question_set_id]
        self.question_ids = list(self.question_set)
        self.decision_markers = _s._declared_markers(self.question_set_id, self.question_set)
        self.branch_of = {question_id: branch for branch, question_id in enumerate(self.question_ids)}
        self.window_ticks = int(window_ticks)
        self.rules = rules
        self.delta = _s._DeltaState(rules)
        #: 알고 있는 지시(전체 직렬화의 `prefix.instructions`와 같은 꼴). 목표 버전이 오르면 생성기의 규칙으로 늘어난다.
        self.instructions: list[dict[str, Any]] = [
            {"version": int(item["version"]), "t_ms": int(item.get("t_ms", 0)), "text": str(item.get("text", ""))} for item in instructions
        ]
        #: 이미 틱 앞에 놓인 지시 버전 (첫 지시는 prefix에 있다).
        self._placed: set[int] = {self.instructions[0]["version"]}
        self._index = 0
        self._last_t: int | None = None

        chunks: list[_s._Chunk] = [_s._Chunk(_s._instruction_line(self.instructions[0]), "prefix", f"instruction:{self.instructions[0]['version']}")]
        chunks.append(_s._Chunk(f"[questions {self.question_set_id}]\n", "prefix", "question_set"))
        chunks.append(_s._Chunk(_s._markers_line(self.decision_markers), "prefix", "markers"))
        for question_id, spec in self.question_set.items():
            branch = self.branch_of[question_id]
            chunks.append(_s._Chunk(_s._question_header(self.decision_markers[question_id], question_id, spec), "prefix", f"question:{question_id}", owner=branch))
            for index, criterion in enumerate(spec["criteria"]):
                chunks.append(_s._Chunk(_s.candidate_line(criterion), "prefix", f"candidate:{question_id}:{index}", candidate=index, owner=branch))
        # 영역·장면 요약은 정적 prefix다 (첫 틱의 것; 바뀐 틱에만 다시 싣는다) — 변화분 상태가 그것을 "실었다"고 기억한다.
        zone_lines = self.delta.zone_lines(first_state)
        if zone_lines:
            chunks.append(_s._Chunk("".join(zone_lines), "prefix", "zones"))
        scene_lines = self.delta.scene_lines(first_state)
        if scene_lines:
            chunks.append(_s._Chunk("".join(scene_lines), "prefix", "scene"))
        out = _s._assemble(chunks, tokenizer, stream=True)
        self.prefix_tokens: list[int] = list(out["tokens"])
        self.prefix_end = len(self.prefix_tokens)
        self.static_candidate_boundaries: dict[str, list[int]] = {}
        self.static_candidate_mapping: dict[str, list[str]] = {}
        for segment in out["segments"]:
            if segment["name"].startswith("candidate:"):
                self.static_candidate_boundaries.setdefault(self.question_ids[segment["owner"]], []).append(_s._last_index(segment))
        for question_id, spec in self.question_set.items():
            if spec["criteria"]:
                self.static_candidate_mapping[question_id] = [criterion["id"] for criterion in spec["criteria"]]
        #: 다음 틱의 첫 토큰이 받을 절대 index.
        self.cursor = self.prefix_end

    # ------------------------------------------------------------------

    @property
    def ticks_done(self) -> int:
        return self._index

    def _due_instructions(self, tick: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        """이 틱 앞에 놓을 지시 — 알고 있는 것 가운데 이 틱이 처음 싣는 버전들(`_instruction_slots` 규칙), 그리고 목표가
        알려진 것보다 높은 버전을 들면 생성기 규칙으로 **만든** 지시."""
        goal = state.get("goal")
        version = goal.get("version") if isinstance(goal, dict) else None
        due: list[dict[str, Any]] = []
        for instruction in self.instructions[1:]:
            if instruction["version"] in self._placed:
                continue
            if version is not None:
                hit = int(version) >= int(instruction["version"])
            else:
                hit = int(tick.get("sim_ms", 0)) >= int(instruction.get("t_ms", 0))
            if hit:
                due.append(instruction)
        known = max(item["version"] for item in self.instructions)
        if version is not None and int(version) > known:
            # 생성기(`append_tick`)가 레코드의 prefix에 덧붙일 바로 그 지시: 버전·이 틱의 sim_ms·목표 텍스트.
            made = {"version": int(version), "t_ms": int(goal.get("t_ms", tick.get("sim_ms", 0))), "text": str(goal.get("text", ""))}
            self.instructions.append(made)
            due.append(made)
        return due

    def tick(self, tick: dict[str, Any]) -> dict[str, Any]:
        """모델 입력 투영 하나(:func:`project_stream_tick`) → 그 틱의 토큰과 구간 표 (전체 직렬화의 `ticks[k]` 항목 + 토큰).

        돌려주는 dict: ``index``·``t``·``start``·``body_end``·``end``·``posed``·``decision_positions``·``candidate_boundaries``·
        ``candidate_mapping``(전체 직렬화와 같은 절대 index), ``tokens``·``body_tokens``·``decision_tokens``, 그리고
        ``carries_instruction``·``instructions_placed``(이 틱 앞에 놓인 지시 버전).
        """
        index = self._index
        t = int(tick["t"])
        # 틱 번호는 제어 스텝 수(0, 5, 10, …)라 +1이 아니라 **증가**만 요구한다 (계약 `_validate_tick`과 같은 규칙).
        if self._last_t is not None and t <= self._last_t:
            raise ValueError(f"t: 틱 번호가 증가하지 않는다 — 마지막 {self._last_t} 다음에 {t}를 받았다")
        request = tick["request"]
        state = request.get("state") or {}
        _s._check_envelope(state, index)
        geom_age = None
        if isinstance(state.get("t"), dict) and isinstance(state["t"].get("age_ms"), dict):
            geom_age = state["t"]["age_ms"].get("geom")

        chunks: list[_s._Chunk] = []
        due = self._due_instructions(tick, state)
        for instruction in due:
            chunks.append(_s._Chunk(_s._instruction_line(instruction), "state", f"instruction:{instruction['version']}", tick=index))
            self._placed.add(int(instruction["version"]))
        carries_instruction = bool(due)

        def add(name: str, text: str, kind: str = "state") -> None:
            if text:
                chunks.append(_s._Chunk(text, kind, name, tick=index))

        goal_period = int(self.rules["goal_text_period_ticks"])
        add("state:t", _s._tick_header_line(tick))
        with_text = goal_period > 0 and index > 0 and index % goal_period == 0 and not carries_instruction
        add("state:goal", _s._goal_line(state.get("goal"), with_text=with_text))
        intros, dynamics = self.delta.object_lines(state, index)
        object_clearance = dict(self.delta.clearance)
        add("state:objects_intro", "".join(intros))
        add("state:objects_dynamic", "".join(dynamics))
        add("state:zones", "".join(self.delta.zone_lines(state)))
        add("state:scene", "".join(self.delta.scene_lines(state)))
        add("state:robot", self.delta.robot_line(state, index) or "")
        if isinstance(state.get("exec"), dict):
            add("state:exec", _s._exec_line(state["exec"]))
        add("state:events", "".join(_s._event_line(event) for event in state.get("events") or () if isinstance(event, dict)))
        add("state:waypoints", "".join(_s._waypoint_line(item) for item in state.get("derived") or () if isinstance(item, dict) and "waypoint" in item))
        add("state:extra", "".join(line + "\n" for line in _s._extra_state_lines(state)))
        add("commitment", _s._commitment_line(request.get("commitment")))
        add("exec_history", _s._history_line(request.get("exec_history")), kind="exec")
        candidates = request["candidates"]
        for question_id in self.question_ids:
            if question_id not in candidates:
                continue
            branch = self.branch_of[question_id]
            chunks.append(_s._Chunk(f"[candidates {question_id}]\n", "question", f"candidates:{question_id}", owner=branch, tick=index))
            for position, entry in enumerate(candidates[question_id]):
                chunks.append(
                    _s._Chunk(
                        _s.stream_candidate_line(entry, geom_age_ms=geom_age, object_clearance=object_clearance),
                        "candidate", f"candidate:{question_id}:{position}", candidate=position, owner=branch, tick=index,
                    )
                )
        posed = [qid for qid, spec in self.question_set.items() if spec["criteria"] or qid in candidates]
        for question_id in posed:
            branch = self.branch_of[question_id]
            chunks.append(_s._Chunk(self.decision_markers[question_id], "decision", f"decision:{question_id}", question=branch, owner=branch, tick=index))

        out = _s._assemble(chunks, self.tokenizer, stream=True)
        tokens: list[int] = list(out["tokens"])
        segments = out["segments"]
        base = self.cursor
        body = [segment for segment in segments if segment["kind"] != "decision"]
        decisions = [segment for segment in segments if segment["kind"] == "decision"]
        start = base + body[0]["start"]
        body_end = base + body[-1]["end"]
        end = base + (decisions[-1]["end"] if decisions else body[-1]["end"])
        boundaries: dict[str, list[int]] = {}
        mapping: dict[str, list[str]] = {}
        for question_id in posed:
            if question_id in candidates:
                boundaries[question_id] = [base + _s._last_index(segment) for segment in body if segment["name"].startswith(f"candidate:{question_id}:")]
                mapping[question_id] = [entry["id"] for entry in candidates[question_id]]
            else:
                boundaries[question_id] = list(self.static_candidate_boundaries[question_id])
                mapping[question_id] = list(self.static_candidate_mapping[question_id])
        entry_out = {
            "index": index,
            "t": tick["t"],
            "start": start,
            "body_end": body_end,
            "end": end,
            "posed": posed,
            "decision_positions": {self.question_ids[segment["owner"]]: base + _s._last_index(segment) for segment in decisions},
            "candidate_boundaries": boundaries,
            "candidate_mapping": mapping,
            "tokens": tokens,
            "body_tokens": tokens[: body_end - start],
            "decision_tokens": tokens[body_end - start :],
            "carries_instruction": carries_instruction,
            "instructions_placed": [int(instruction["version"]) for instruction in due],
        }
        self.cursor = end
        self._index += 1
        self._last_t = t
        return entry_out
