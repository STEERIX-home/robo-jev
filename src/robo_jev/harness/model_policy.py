"""학습된 판단 모델을 루프의 정책 자리에 꽂는 어댑터 (Task R4 A1, docs/06 Task 6).

:func:`robo_jev.data.robot_episodes.generate_episode` 는 틱마다 ``policy.act(request, commitment, scene)`` 를 부른다.
:class:`ModelPolicy` 는 그 자리에 서서 하네스의 요청을 **학습 때와 같은 ts0.6 스트림 직렬화**로 토큰화하고
(:class:`robo_jev.model.incremental.IncrementalStreamSerializer` — 첫 틱에 prefix, 지시 변경 틱에 지시 조각, 30틱 윈도우),
실제 backbone의 스트림 상태(:class:`robo_jev.model.backbone_qwen.QwenStreamState` — 정적 prefix KV + 윈도우 버퍼)를 한 틱
전진시키며 결정 표지 열 개를 분기 배치로 돌리고(`advance_with_branches`, 서빙 지렛대 `fused`; fixture backbone은 `advance` +
`branch_step`), 학습된 pointer readout으로 열 질문의 답을 **전문가 `act`가 돌려주는 것과 같은 꼴**로 만든다 — `q_main`·
`q_path`·`q_gripper`·`q_speed`·`q_force`는 ``{후보 id: 확률}``, 게이트·정지는 ``p_true``(float). 하네스의 `compose`가 그 답을
명령으로 합성한다 — 전문가 답과 같은 경로다.

**정보 경계.** 답은 요청(`request["request"]`: 상태·실행 이력·commitment·후보)의 함수다. 하네스 블록(`request["harness"]`)·
라벨·채택·ACK는 :func:`project_stream_tick` 이 떼어 내고 계약 검사로 거절한다. `commitment`·`observation` 인자는 읽지 않는다
(전문가와 같은 규칙; 검사가 고정한다). 첫 지시는 첫 틱 요청의 `state.goal`(버전·텍스트)에서 만든다 — 레코드의 prefix가
`scene["instruction"]`에서 받는 것과 같은 세 필드(`version`·`t_ms`·`text`; v1의 `t_ms`는 0)다. 녹화된 에피소드를 재생할 때는
:meth:`ModelPolicy.begin_episode` 로 레코드의 지시 목록을 그대로 줄 수 있다(Task R4 A3).

**상태는 에피소드 안에서만 이어진다.** `t == 0`인 요청이 오면 새 에피소드다(직렬화·스트림 상태를 새로 만든다); 그 밖에는
틱이 이어져야 한다(빠지면 `ValueError`). :meth:`reset` 은 같은 것을 명시적으로 한다.

**틱 지연.** 틱마다 직렬화 시간·모델 시간(CUDA event; CPU는 벽시계)·readout 시간·`act` 전체 벽시계를 :attr:`timing` 에
남긴다 (docs/03 §7-6의 게이트를 루프 안에서 재는 재료, Task R4 A4).

이 모듈은 하네스 쪽에 있다(하네스 → 모델 방향의 import만; 모델 모듈은 하네스를 import하지 않는다, docs/06 §1).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from robo_jev.model.incremental import IncrementalStreamSerializer, project_stream_tick
from robo_jev.model.judge import typed_outputs
from robo_jev.model.serialize import QUESTION_SETS, WINDOW_TICKS
from robo_jev.model.stream import stream_state_class

__all__ = ["POLICY_VERSION", "ModelPolicy", "answers_from_typed", "batched_pointer_logits", "load_serving_judge"]

#: 어댑터 버전 — 레코드의 `provenance.policy.version`에 든다.
POLICY_VERSION = "mp0.1"


def batched_pointer_logits(judge: Any, state: Any, entry: dict[str, Any], branch_hidden: Tensor, prefix_end: int) -> dict[str, Tensor]:
    """틱의 질문별 pointer logits를 **한 matmul**로 (G0b의 `tick_readout`과 같은 계산; 값은 :meth:`Judge.pointer_logits` 와 같다).

    질문 순서 = 결정 표지 순서 = `branch_hidden`의 행이다. 정적 후보(prefix 안)의 ``h_c``는 `state.prefix_hidden`, 동적 후보는
    이 틱 몸통의 `state.hidden`이다. 돌려주는 것은 ``{qid: logits[K]}``.
    """
    rows: list[Tensor] = []
    spans: list[tuple[str, int]] = []
    start = int(entry["start"])
    for qid in entry["decision_positions"]:
        boundaries = [int(b) for b in entry["candidate_boundaries"][qid]]
        static = [b for b in boundaries if b < prefix_end]
        dynamic = [b - start for b in boundaries if b >= start]
        if static and dynamic:
            raise ValueError(f"{qid}: 후보 경계가 prefix와 틱 몸통에 섞여 있다")
        if len(static) + len(dynamic) != len(boundaries):
            raise ValueError(f"{qid}: 후보 경계가 prefix 밖이면서 이 틱 몸통 안도 아니다: {boundaries}")
        rows.append(state.prefix_hidden[static] if static else state.hidden[dynamic])
        spans.append((qid, len(boundaries)))
    h_c = torch.cat(rows)  # [ΣK, d]
    dtype = judge.readout_dtype
    scores = judge.V(h_c.to(dtype)) @ judge.U(branch_hidden.to(dtype)).T / math.sqrt(judge.rank) + judge.bias  # [ΣK, n]
    out: dict[str, Tensor] = {}
    offset = 0
    for column, (qid, count) in enumerate(spans):
        out[qid] = scores[offset : offset + count, column]
        offset += count
    return out


def answers_from_typed(typed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """:func:`typed_outputs` 의 결과 → `model_output` 꼴(전문가·규칙 판정기와 같다): choice·ordinal은 ``{id: p}``, boolean은 ``p_true``."""
    out: dict[str, Any] = {}
    for qid, entry in typed.items():
        out[qid] = float(entry["p_true"]) if entry["type"] == "boolean" else {cid: float(p) for cid, p in entry["probabilities"].items()}
    return out


class ModelPolicy:
    """루프 정책: 요청 → 열 답 (모듈 설명 참조). 에피소드 상태를 든다 — 에피소드마다 `t == 0`에서 새로 시작한다."""

    name = "ModelPolicy"

    def __init__(
        self,
        judge: Any,
        tokenizer: Any,
        *,
        question_set: str | None = None,
        window_ticks: int = WINDOW_TICKS,
        fused: bool | None = None,
        version: str = POLICY_VERSION,
        keep_timing: bool = True,
    ) -> None:
        self.judge = judge
        self.tokenizer = tokenizer
        if question_set is None:
            from robo_jev.data.episode import default_question_set  # 하네스 설정의 질문 세트 id

            question_set = default_question_set()
        if question_set not in QUESTION_SETS:
            raise ValueError(f"question_set: {list(QUESTION_SETS)} 중 하나여야 한다 (받은 값: {question_set!r})")
        self.question_set_id = str(question_set)
        self.question_set = QUESTION_SETS[self.question_set_id]
        self.window_ticks = int(window_ticks)
        self.version = str(version)
        self.keep_timing = bool(keep_timing)
        backbone = judge.backbone
        self.state_class = stream_state_class(backbone)
        has_fused = hasattr(self.state_class, "advance_with_branches")
        if fused is None:
            fused = has_fused
        if fused and not has_fused:
            raise ValueError(f"fused: {self.state_class.__name__}에는 몸통+분기 한 forward가 없다 (실제 backbone 전용)")
        self.fused = bool(fused)
        self.device = next(judge.parameters()).device
        self._cuda = self.device.type == "cuda"
        if self._cuda:
            self._events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        self.episodes = 0
        self._pending_instructions: list[dict[str, Any]] | None = None
        self.reset()

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """새 에피소드 — 직렬화·스트림 상태·틱 계수·지연 기록을 버린다 (다음 요청은 `t == 0`이어야 한다)."""
        self.serializer: IncrementalStreamSerializer | None = None
        self.state: Any = None
        self.tick = -1
        self.timing: list[dict[str, Any]] = []
        self._pending_instructions = None

    def begin_episode(self, *, instructions: list[dict[str, Any]]) -> None:
        """다음 에피소드의 지시 목록을 미리 준다 (녹화된 에피소드의 재생, Task R4 A3). 주지 않으면 첫 틱의 목표에서 만든다."""
        if not instructions:
            raise ValueError("instructions: 시작 지시가 하나 이상 있어야 한다")
        self.reset()
        self._pending_instructions = [dict(item) for item in instructions]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "question_set": self.question_set_id, "window_ticks": self.window_ticks,
            "fused": self.fused, "state_class": self.state_class.__name__, "device": str(self.device),
            "readout_dtype": str(self.judge.readout_dtype), "readout_rank": int(self.judge.rank),
        }

    # ------------------------------------------------------------------

    def _first_instructions(self, tick: dict[str, Any]) -> list[dict[str, Any]]:
        if self._pending_instructions is not None:
            pending, self._pending_instructions = self._pending_instructions, None
            return pending
        goal = (tick["request"].get("state") or {}).get("goal")
        if not isinstance(goal, dict) or goal.get("version") is None:
            raise ValueError("첫 틱의 state.goal에 version이 없다 — 시작 지시를 만들 수 없다 (begin_episode로 주거나 목표를 싣는다)")
        # 레코드의 prefix가 `scene["instruction"]`에서 받는 세 필드와 같다: v1은 t=0에 걸려 있으므로 t_ms 0.
        return [{"version": int(goal["version"]), "t_ms": int(goal.get("t_ms", 0)), "text": str(goal.get("text", ""))}]

    def act(self, request: dict[str, Any], commitment: dict[str, Any] | None = None, observation: Any = None) -> dict[str, Any]:
        """열 답 (모델 출력 꼴). `commitment`·`observation`은 읽지 않는다 (정보 경계)."""
        del commitment, observation
        started = time.perf_counter()
        t = int(request["t"])
        if t == 0:
            pending = self._pending_instructions
            self.reset()
            self._pending_instructions = pending
            self.episodes += 1
        elif self.serializer is None or t <= self.tick:
            # 틱 번호는 제어 스텝 수(0, 5, 10, …)다 — 에피소드는 0에서 시작하고 번호는 증가만 한다.
            raise ValueError(f"t: 에피소드는 0에서 시작해 틱 번호가 증가해야 한다 — 마지막 {self.tick} 다음에 {t}를 받았다")

        tick = project_stream_tick(request)
        serialize_started = time.perf_counter()
        prefix_tokens = 0
        if self.serializer is None:
            self.serializer = IncrementalStreamSerializer(
                self.tokenizer, question_set=self.question_set_id, instructions=self._first_instructions(tick),
                first_state=tick["request"].get("state") or {}, window_ticks=self.window_ticks,
            )
            prefix_tokens = self.serializer.prefix_end
        entry = self.serializer.tick(tick)
        serialize_ms = (time.perf_counter() - serialize_started) * 1e3

        with torch.no_grad():
            if self._cuda:
                self._events[0].record()
            else:
                model_started = time.perf_counter()
            if self.state is None:
                state = self.state_class.initial(self.judge.backbone, window_ticks=self.window_ticks)
                if prefix_tokens:
                    state = state.extend_prefix(self.serializer.prefix_tokens)
                self.state = state
            body, decisions = entry["body_tokens"], entry["decision_tokens"]
            if self.fused and decisions:
                self.state, branch_hidden = self.state.advance_with_branches(body, decisions)
            else:
                self.state = self.state.advance(body)
                branch_hidden = self.state.branch_step(decisions) if decisions else None
            if self._cuda:
                self._events[1].record()
            else:
                readout_started = time.perf_counter()
            logits = batched_pointer_logits(self.judge, self.state, entry, branch_hidden, self.serializer.prefix_end) if branch_hidden is not None else {}
            if self._cuda:
                self._events[2].record()
                torch.cuda.synchronize()
                model_ms = float(self._events[0].elapsed_time(self._events[2]))
                readout_ms = float(self._events[1].elapsed_time(self._events[2]))
            else:
                readout_ms = (time.perf_counter() - readout_started) * 1e3
                model_ms = (time.perf_counter() - model_started) * 1e3
            flat = {qid: z.detach().float().cpu() for qid, z in logits.items()}  # D2H 한 번, 나머지는 CPU
        candidates = {qid: list(entry["candidate_mapping"][qid]) for qid in flat}
        answers = answers_from_typed(typed_outputs(flat, candidates, self.question_set))
        for qid in self.question_set:
            # 이 틱에 묻지 않은 동적 질문(후보 없음 — 예: 경로 후보가 없는 틱)은 전문가와 같이 빈 분포다.
            answers.setdefault(qid, {})
        self.tick = t
        if self.keep_timing:
            self.timing.append({
                "t": t, "tokens_body": len(body), "tokens_decision": len(decisions), "prefix_tokens": prefix_tokens,
                "cached_tokens": int(getattr(self.state, "cached_tokens", 0)), "serialize_ms": serialize_ms,
                "model_ms": model_ms, "readout_ms": readout_ms, "act_ms": (time.perf_counter() - started) * 1e3,
            })  # fmt: skip
        return answers


# --------------------------------------------------------------------------
# 서빙 모델 적재 (GPU; Task R4 A2)
# --------------------------------------------------------------------------


def load_serving_judge(
    checkpoint: str | Path,
    *,
    model_id: str = "Qwen/Qwen3.5-2B",
    tokenizer_name: str | None = None,
    device: str = "cuda",
    compile_dense: bool = False,
    readout_rank: int | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """R3a fp32 T1 checkpoint의 **모델 가중치만**(optimizer·fp32 master 제외) bf16 backbone + fp32 readout에 싣는다 (A2).

    순서는 G0b M-b: 정적 KV 모드로 backbone을 싣고 → `Judge` → :func:`robo_jev.train.load_readout_checkpoint` (배포 계약 digest —
    직렬화·계약 소스·하네스 버전·tokenizer 파일 해시 — 가 지금 체크아웃과 다르면 **거절한다**) → 그 뒤에야 `compile`.
    readout은 fp32 그대로다(학습한 dtype; readout 시간은 ≈0.3 ms라 bf16 지렛대는 의미가 없다 — G0b S2.2).
    돌려주는 것: ``{"judge", "tokenizer", "tokenizer_name", "manifest", "compile_seconds", "load_seconds"}``.
    """
    from robo_jev.model.backbone_qwen import QwenBackbone
    from robo_jev.model.judge import Judge
    from robo_jev.model.tokenizer import available_tokenizer
    from robo_jev.train import DEFAULT_REAL_READOUT_RANK, build_tokenizer, load_readout_checkpoint, tokenizer_block

    started = time.perf_counter()
    if tokenizer_name is None:
        found = available_tokenizer()
        if found is None:
            raise FileNotFoundError("실제 tokenizer가 없다 — `uv run python scripts/fetch_tokenizer.py`")
        tokenizer_name = found[0]
    backbone = QwenBackbone.load(model_id, root=root, dtype=torch.bfloat16, device=device, kv_mode="static")
    rank = DEFAULT_REAL_READOUT_RANK if readout_rank is None else int(readout_rank)
    judge = Judge(backbone, rank=rank, readout="pointer", seed=1000, readout_dtype=torch.float32)
    manifest = load_readout_checkpoint(judge, checkpoint, tokenizer_sha256=tokenizer_block(tokenizer_name)["sha256"])
    judge.eval()
    judge.requires_grad_(False)
    compile_seconds = None
    if compile_dense:
        compile_started = time.perf_counter()
        compile_seconds = backbone.compile_dense_parts()
        compile_seconds = round(time.perf_counter() - compile_started, 1) if compile_seconds is None else compile_seconds
    return {
        "judge": judge, "tokenizer": build_tokenizer(tokenizer_name), "tokenizer_name": tokenizer_name, "manifest": manifest,
        "compile_seconds": compile_seconds, "load_seconds": round(time.perf_counter() - started, 1),
    }
