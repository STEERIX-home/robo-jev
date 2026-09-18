"""Judge — backbone + 결정 위치 pointer readout + 타입별 변환 + 참고군 R (docs/03 §3, docs/06 Task 4).

:class:`Judge` 는 직렬화된 layout(:func:`robo_jev.model.serialize.serialize_request`의 결과)을 받아
질문별 logits를 :func:`robo_jev.loss.judgment_loss`가 받는 layout으로 낸다::

    outputs = {"logits": [{qid: Tensor[K]} …상태마다], "candidates": [{qid: [id…]} …], "question_ids": […]}

logits의 k번째는 그 질문의 **요청 순서 k번째 후보**이고 id는 같은 순서의 ``candidates``
(= 직렬화의 ``candidate_mapping``, 스트림은 틱의 것)가 말한다. 명령 생성은 없다.

**pointer readout (주 경로).** ``z_ik = (U h_{d_i})ᵀ (V h_{c_ik}) / √r + b``. ``h_d`` = 질문 i의 결정
위치(예약 표지 토큰)의 hidden state, ``h_c`` = 후보 k의 **경계 토큰 = 후보 줄의 마지막 토큰(줄바꿈)**
의 hidden state(직렬화의 ``candidate_boundaries``). hidden state는 backbone의 최종 RMSNorm 뒤 출력이다.
``U``·``V``·``b``는 모든 질문·타입이 공유하고, 질문 하나의 logits만 모아 softmax한다. choice는 후보
전체, boolean은 ``true``·``false`` 두 기준, ordinal은 수준 순서의 후보가 같은 경로를 탄다
(:func:`typed_outputs`가 확률·선택·`p_true`·기댓값으로 바꾼다).

**두 실행 backend, 같은 readout.**

* ``state_first`` (L0, P0) — 질문 i마다 ``S + T_i``를 **독립 causal 경로**로 실행한다. 배치의 모든 상태의
  경로를 오른쪽 padding한 **한 배치, 한 forward**로 돌리며(causal이라 padding은 앞 토큰에 영향이 없다)
  position은 직렬화의 것(T_i는 ``len(S)``부터)이다. 경로가 독립이므로 질문 단독/묶음, 상태 단독/묶음의
  logits·loss·gradient가 같다(상태 묶음은 FP32 안에서).
* ``stream_l1a`` (L1-a) — :func:`robo_jev.model.stream.replay_layout`으로 prefix → 틱 몸통 한 번 →
  결정 위치마다 일시적 분기(fork/step). 정적 후보의 ``h_c``는 prefix hidden(윈도우 밖으로 내보내지
  않는다), 동적 후보는 그 틱 몸통의 hidden이다. ``from_scratch=True``면 같은 layout을 한 번의 forward
  (기준 mask + transient 결정)로 계산한다 — 증분 계산과 같아야 하는 기준이다.

**참고군 R (`readout="candidate_branch"`, docs/03 §3 "추가 연산 비교군").** 후보 k마다 분기를 하나 더
둔다: 같은 직렬화 계약의 **후보 줄 토큰을 결정 위치 뒤에 다시 읽고** 그 마지막 토큰의 hidden에
scalar readout ``z_ik = wᵀ h_ik + b``를 적용한다. state_first에서는 ``S + T_i + [후보 k 줄]``이 독립
경로(Q×K개), 스트림에서는 틱 공통 상태에서 fork → 결정 표지 → 후보 k 줄 토큰을 step한 분기다.
인수 검사는 "돌아가고 모양이 맞는다"까지다(docs/06 Task 4).

이 모듈은 generator·simulator·하네스를 import하지 않는다 (docs/06 §1).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from robo_jev.model.hybrid import DEFAULT_CONFIG, HybridConfig, TinyHybrid, causal_mask
from robo_jev.model.stream import LAYOUT_WINDOW, StreamState, forward_layout, replay_layout

__all__ = ["READOUTS", "Judge", "candidate_span", "typed_outputs"]

READOUTS = ("pointer", "candidate_branch")
_TRUE, _FALSE = "true", "false"


def candidate_span(layout: dict, boundary: int) -> tuple[int, int]:
    """후보 경계 토큰 ``boundary``가 끝내는 후보 줄의 토큰 구간 ``[start, end)`` (``end = boundary + 1``).

    같은 ``kind``·``question``·``candidate`` 값이 이어지는 가장 긴 구간이다 — 직렬화가 후보 줄마다
    조각을 두고 이웃 후보는 색인이 다르므로 줄 하나로 끝난다.
    """
    kind, question, candidate = layout["kind"], layout["question"], layout["candidate"]
    if candidate[boundary] < 0:
        raise ValueError(f"boundary {boundary}: 후보 토큰이 아니다 (candidate={candidate[boundary]})")
    start = boundary
    while (
        start > 0
        and kind[start - 1] == kind[boundary]
        and question[start - 1] == question[boundary]
        and candidate[start - 1] == candidate[boundary]
    ):
        start -= 1
    return start, boundary + 1


def typed_outputs(
    logits: dict[str, Tensor], candidates: dict[str, list[str]], questions: dict[str, dict]
) -> dict[str, dict[str, Any]]:
    """질문별 logits → 타입별 결과 (docs/03 §2).

    choice: 후보별 확률과 선택 id. boolean: ``true``/``false`` 확률과 ``p_true``. ordinal: 수준별 확률과,
    기준에 ``value``가 있으면 기댓값 ``Σ p_k value_k``. 모두 ``choice``(최대 확률 id)를 함께 준다.
    """
    result: dict[str, dict[str, Any]] = {}
    for qid, z in logits.items():
        if qid not in questions:
            raise ValueError(f"questions: {qid!r}의 정의가 없다")
        spec = questions[qid]
        ids = list(candidates[qid])
        kind = spec.get("type")
        if kind not in ("choice", "boolean", "ordinal"):
            raise ValueError(f"{qid}.type: choice/boolean/ordinal 중 하나여야 한다 (받은 값: {kind!r})")
        if z.dim() != 1 or z.shape[0] != len(ids):
            raise ValueError(f"{qid}: logits {tuple(z.shape)}와 candidates {len(ids)}가 다르다")
        p = torch.softmax(z.detach().float(), dim=0)
        probabilities = {cid: float(p[i]) for i, cid in enumerate(ids)}
        entry: dict[str, Any] = {"type": kind, "probabilities": probabilities, "choice": ids[int(p.argmax())]}
        if kind == "boolean":
            if set(ids) != {_TRUE, _FALSE}:
                raise ValueError(f"{qid}.candidates: boolean은 true/false 두 후보여야 한다 (받은 값: {ids})")
            entry["p_true"] = probabilities[_TRUE]
        elif kind == "ordinal":
            values = {c["id"]: c.get("value") for c in spec.get("criteria", [])}
            if ids and all(values.get(cid) is not None for cid in ids):
                entry["expected_value"] = sum(probabilities[cid] * float(values[cid]) for cid in ids)
        result[qid] = entry
    return result


class Judge(nn.Module):
    """backbone + readout (모듈 설명 참조). ``forward(batch) -> dict``.

    선택한 readout의 파라미터만 만든다 — pointer면 ``U``·``V``·``bias``, 참고군 R이면 ``w``. 그래서
    손실이 있는 forward마다 모든 파라미터가 gradient를 받고, 학습은 쓰이지 않는 파라미터를 걸러낼
    필요(`find_unused_parameters`)가 없다.
    """

    def __init__(
        self, backbone: TinyHybrid, *, rank: int, readout: str = "pointer", seed: int | None = None
    ) -> None:
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"readout: {list(READOUTS)} 중 하나여야 한다 (받은 값: {readout!r})")
        if rank < 1:
            raise ValueError(f"rank: 1 이상이어야 한다 (받은 값: {rank})")
        self.backbone = backbone
        self.rank = int(rank)
        self.readout = readout
        d = backbone.config.d_model
        if readout == "pointer":
            self.U = nn.Linear(d, rank, bias=False)
            self.V = nn.Linear(d, rank, bias=False)
            self.bias = nn.Parameter(torch.zeros(()))
            linears = (self.U, self.V)
        else:
            self.w = nn.Linear(d, 1)  # 참고군 R의 scalar readout (bias 포함)
            linears = (self.w,)
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
            with torch.no_grad():
                for linear in linears:
                    linear.weight.normal_(0.0, 1.0 / math.sqrt(d), generator=generator)
                    if linear.bias is not None:
                        linear.bias.zero_()

    @classmethod
    def from_config(
        cls,
        path: str | Path = DEFAULT_CONFIG,
        *,
        seed: int | None = None,
        readout: str | None = None,
        vocab_size: int | None = None,
    ) -> Judge:
        """설정 파일의 fixture + readout. ``vocab_size``는 검사용 작은 어휘(설정 fixture와 앞 V행이 같다)."""
        config = HybridConfig.load(path)
        if vocab_size is not None:
            config = HybridConfig(**{**config.__dict__, "vocab_size": int(vocab_size)})
        seed = config.seed if seed is None else int(seed)
        backbone = TinyHybrid(config, seed=seed)
        return cls(backbone, rank=config.readout_rank, readout=readout or config.readout_mode, seed=seed + 1000)

    # -- readout --

    def pointer_logits(self, h_d: Tensor, h_c: Tensor) -> Tensor:
        """``z_k = (U h_d)ᵀ (V h_{c_k}) / √r + b``. ``h_d [d]``, ``h_c [K, d]`` → ``[K]``."""
        if self.readout != "pointer":
            raise ValueError(f"readout: pointer readout이 아니다 ({self.readout!r})")
        return (self.V(h_c) @ self.U(h_d)) / math.sqrt(self.rank) + self.bias

    def branch_logits(self, h: Tensor) -> Tensor:
        """참고군 R: ``z_k = wᵀ h_k + b``. ``h [K, d]`` → ``[K]``."""
        if self.readout != "candidate_branch":
            raise ValueError(f"readout: 참고군 R(candidate_branch)이 아니다 ({self.readout!r})")
        return self.w(h)[:, 0]

    # -- 공개 API --

    def forward(self, batch: dict) -> dict[str, Any]:
        layout = batch.get("layout")
        if layout == "state_first":
            states = batch.get("states")
            if not isinstance(states, list) or not states:
                raise ValueError("states: state_first 배치는 직렬화된 요청의 목록이 필요하다")
            return self._state_first(states)
        if layout == "stream_l1a":
            stream = batch.get("stream")
            if not isinstance(stream, dict):
                raise ValueError("stream: stream_l1a 배치는 직렬화된 스트림 하나가 필요하다")
            return self._stream(
                stream,
                from_scratch=bool(batch.get("from_scratch", False)),
                window_ticks=batch.get("window_ticks", LAYOUT_WINDOW),
                initial=batch.get("initial"),
                state=batch.get("state"),
                start_tick=int(batch.get("start_tick", 0)),
            )
        raise ValueError(f"layout: state_first 또는 stream_l1a여야 한다 (받은 값: {layout!r})")

    # -- state_first (P0) --

    def _run_paths(self, paths: list[tuple[list[int], list[int]]]) -> Tensor:
        """경로들을 오른쪽 padding한 한 배치로 돌린다 → hidden ``[P, L, d]``."""
        length = max(len(tokens) for tokens, _ in paths)
        ids = torch.zeros(len(paths), length, dtype=torch.long)
        positions = torch.zeros(len(paths), length, dtype=torch.long)
        for row, (tokens, pos) in enumerate(paths):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            positions[row, : len(pos)] = torch.tensor(pos, dtype=torch.long)
            positions[row, len(pos) :] = pos[-1] + 1 + torch.arange(length - len(pos))
        return self.backbone(ids, positions, mask=causal_mask(length))["hidden"]

    @staticmethod
    def _question_span(layout: dict, branch: int, decision: int) -> tuple[int, int]:
        owned = [i for i, q in enumerate(layout["question"]) if q == branch]
        if not owned or owned[-1] != decision or owned[-1] - owned[0] + 1 != len(owned):
            raise ValueError(f"question {branch}: T_i가 결정 위치 {decision}로 끝나는 연속 구간이어야 한다")
        return owned[0], decision + 1

    def _state_first(self, states: list[dict]) -> dict[str, Any]:
        """모든 상태의 질문 경로를 **한 번의** backbone forward로 (경로는 독립이라 한 배치에 묶인다).

        비로봇 단위 = 토큰 예산까지 묶은 단일 요청 microbatch(docs/04 §2)가 forward 하나로 돈다. 상태를
        따로 돌린 것과 FP32 안에서 같다(padding·batch 크기에 따른 matmul 반올림 차이).
        """
        paths: list[tuple[list[int], list[int]]] = []  # 모든 상태의 경로 (행 = 전역 index)
        plan: list[tuple[dict, list[tuple[str, int, int, list[int]]], list[tuple[str, int, int]]]] = []
        for layout in states:
            if layout.get("layout") != "state_first":
                raise ValueError(f"states: layout이 state_first가 아니다 ({layout.get('layout')!r})")
            tokens, positions = layout["tokens"], layout["position"]
            S = int(layout["state_end"])
            readouts: list[tuple[str, int, int, list[int]]] = []  # (qid, row, decision local, boundary locals)
            branch_paths: list[tuple[str, int, int]] = []  # (qid, row, readout local) — 참고군 R
            for branch, qid in enumerate(layout["question_ids"]):
                decision = int(layout["decision_positions"][qid])
                t_start, t_end = self._question_span(layout, branch, decision)
                path_tokens = tokens[:S] + tokens[t_start:t_end]
                path_positions = positions[:S] + positions[t_start:t_end]
                local = lambda index, t_start=t_start: S + (index - t_start)  # noqa: E731
                boundaries = [int(b) for b in layout["candidate_boundaries"][qid]]
                if self.readout == "pointer":
                    paths.append((path_tokens, path_positions))
                    readouts.append((qid, len(paths) - 1, local(decision), [local(b) for b in boundaries]))
                else:
                    for boundary in boundaries:
                        start, end = candidate_span(layout, boundary)
                        reread = tokens[start:end]
                        extra = list(range(path_positions[-1] + 1, path_positions[-1] + 1 + len(reread)))
                        paths.append((path_tokens + reread, path_positions + extra))
                        branch_paths.append((qid, len(paths) - 1, len(path_tokens) + len(reread) - 1))
            plan.append((layout, readouts, branch_paths))
        hidden = self._run_paths(paths)

        logits_all: list[dict[str, Tensor]] = []
        candidates_all: list[dict[str, list[str]]] = []
        question_ids_all: list[list[str]] = []
        for layout, readouts, branch_paths in plan:
            logits: dict[str, Tensor] = {}
            if self.readout == "pointer":
                for qid, row, decision, boundaries in readouts:
                    logits[qid] = self.pointer_logits(hidden[row, decision], hidden[row, boundaries])
            else:
                rows: dict[str, list[Tensor]] = {}
                for qid, row, last in branch_paths:
                    rows.setdefault(qid, []).append(hidden[row, last])
                for qid in layout["question_ids"]:
                    logits[qid] = self.branch_logits(torch.stack(rows[qid]))
            logits_all.append(logits)
            candidates_all.append({qid: list(layout["candidate_mapping"][qid]) for qid in layout["question_ids"]})
            question_ids_all.append(list(layout["question_ids"]))
        return {"logits": logits_all, "candidates": candidates_all, "question_ids": question_ids_all}

    # -- stream_l1a --

    def _stream(
        self,
        layout: dict,
        *,
        from_scratch: bool,
        window_ticks: Any,
        initial: list[dict] | None,
        state: StreamState | None,
        start_tick: int,
    ) -> dict[str, Any]:
        if layout.get("layout") != "stream_l1a":
            raise ValueError(f"stream: layout이 stream_l1a가 아니다 ({layout.get('layout')!r})")
        tick_states: list[StreamState] | None = None
        final: StreamState | None = None
        if from_scratch:
            if state is not None or start_tick or initial is not None:
                raise ValueError("from_scratch: 이어 붙이기(state·start_tick·initial)와 함께 쓸 수 없다")
            if self.readout != "pointer":
                raise ValueError("from_scratch: 참고군 R은 증분 경로에서만 실행한다")
            hidden = forward_layout(layout, backbone=self.backbone, window_ticks=window_ticks)
        else:
            replay = replay_layout(
                layout, backbone=self.backbone, window_ticks=window_ticks, initial=initial, state=state,
                start_tick=start_tick,
            )  # fmt: skip
            hidden, tick_states, final = replay["hidden"], replay["tick_states"], replay["final"]

        tokens = layout["tokens"]
        logits_all: list[dict[str, Tensor]] = []
        candidates_all: list[dict[str, list[str]]] = []
        question_ids_all: list[list[str]] = []
        for tick in layout["ticks"]:
            if int(tick["index"]) < start_tick:
                continue
            logits: dict[str, Tensor] = {}
            for qid, decision in tick["decision_positions"].items():
                boundaries = [int(b) for b in tick["candidate_boundaries"][qid]]
                if self.readout == "pointer":
                    logits[qid] = self.pointer_logits(hidden[int(decision)], hidden[boundaries])
                else:
                    common = tick_states[int(tick["index"]) - start_tick]
                    outputs = []
                    for branch, boundary in zip(common.fork(len(boundaries)), boundaries):
                        branch.step(tokens[int(decision)])
                        start, end = candidate_span(layout, boundary)
                        last = None
                        for token in tokens[start:end]:
                            last = branch.step(token)
                        outputs.append(last)
                    logits[qid] = self.branch_logits(torch.stack(outputs))
            logits_all.append(logits)
            candidates_all.append({qid: list(tick["candidate_mapping"][qid]) for qid in logits})
            question_ids_all.append(list(logits))
        return {
            "logits": logits_all,
            "candidates": candidates_all,
            "question_ids": question_ids_all,
            "hidden": hidden,
            "state": final,
            "tick_states": tick_states,
        }
