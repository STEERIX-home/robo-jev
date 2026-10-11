"""학습 없는 Qwen이 판단 요청에 내는 코드 확률 읽기 (Task v0c §2.2; 기획서 docs/12 §1 기준선, §11).

**바로 답 (direct)** — :func:`robo_jev.decisions.prompt.build_prompts` 의 프롬프트(``Answer:``로 끝남)를 한 번 forward하고,
마지막 유효 위치의 hidden(최종 RMSNorm 뒤)을 **LM head의 코드 행에만** fp32로 투영해 코드 위에서 softmax한다
(:func:`robo_jev.model.zero_shot.score_prompts` 와 같은 수학 — 검사가 같은 확률을 확인한다). 우리가 쓸 형태(System 1, 한 번의
forward)와 같은 조건이다. **기본 배치는 1**(padding 없음): 실제 2B(bf16)에서 배치 구성(오른쪽 padding 길이·묶인 프롬프트)이 코드
확률을 최대 0.015–0.020 움직이고 fixture 33문항 중 3–4개의 argmax를 뒤집었다(같은 배치는 비트 단위로 재현) — 한 프롬프트의 값이
같이 묶인 다른 프롬프트에 기대지 않게 하려고 하나씩 읽는다(배포 조건 — 요청 하나 — 과도 같다). `batch`를 키우면 길이로 정렬해
오른쪽 padding으로 묶는다(빠르지만 위의 잡음이 있다; 결과는 원래 순서로 돌려준다).

**코드 질량** — 같은 자리에서 전체 어휘 softmax가 코드들에 준 몫(``code_mass``; 호출마다 LM head의 fp32 사본을 한 번 만들어
logsumexp — 2B 2.0 GB, 4B 2.5 GB). 확률은 코드 행만으로 정하고, 질량은 형식 준수(모델이 정말 글자로 답하려 하는가)를 보는 진단이다.

**생각하고 답 (reasoning)** — 같은 프롬프트에 "짧게 생각하고 답하라"를 붙인 판(``mode="reasoning"``)에서 greedy로 최대
`max_new_tokens`(기본 256) 생성한다(왼쪽 padding 배치, HF ``generate``; Qwen3.5의 DeltaNet 층은 padding 자리의 입력을 0으로
지워 상태에 섞이지 않는다 — 소형 검사가 배치 생성 == 하나씩 생성을 확인한다). 생성은 끝 토큰(``<|im_end|>``·``<|endoftext|>``,
chat은 ``</think>``도)에서 멈추고, 모델이 스스로 ``Answer:``를 쓰면 그 앞에서 자른다(:func:`finish_reasoning`). 그 뒤에
:func:`robo_jev.decisions.prompt.reasoning_close` 를 붙여 **바로 답과 같은 방식으로** 코드 확률을 읽는다. 비싸므로 일부 표본에만 쓴다.
생성은 배치로 한다(하나씩이면 너무 느리다) — 실제 2B(bf16)에서 생성 배치 1과 8의 greedy 경로가 fixture 15문항 중 11개는 64토큰까지
같았고 4개는 31·53번째 토큰에서 갈렸다(bf16 잡음). 같은 입력·같은 설정이면 비트 단위로 재현된다; 생각 글은 "그 설정에서의 한 경로"다.

GPU 규칙(GB10 통합 메모리): 진입점(``scripts/diagnose_decisions.py``)이 첫 CUDA 할당 전에 ``limit_gpu_memory(0.6)``을 걸고 적재 전에
:data:`MIN_FREE_BYTES` 만큼의 여유를 ``require_free``로 확인한다.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

import torch

from robo_jev.decisions.prompt import PROMPT_VERSION, answer_from_codes, reasoning_close, record_prompts

__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_GEN_BATCH",
    "MIN_FREE_BYTES",
    "REASONING_MAX_NEW_TOKENS",
    "diagnose_records",
    "finish_reasoning",
    "generate_reasoning",
    "score_code_prompts",
    "special_id",
]

DEFAULT_BATCH = 1
DEFAULT_GEN_BATCH = 16
REASONING_MAX_NEW_TOKENS = 256
#: 적재 전에 있어야 하는 장치(통합) 메모리 여유. 근거: bf16 가중치 2B ≈ 4.3 GiB(text 1.9 B + 시각 0.3 B 매개변수)·4B ≈ 8.7 GiB,
#: CUDA context ≈ 1 GiB, 코드 질량용 LM head fp32 사본(2B 2.0 GB·4B 2.5 GB), 채점 배치(≤ 8 × 1.5 K 토큰)의 activation 1 GiB 미만,
#: reasoning 배치 생성(16열 × ≤ 1.8 K 토큰)의 KV(2B: full-attention 6층 × 2 head × 256 × bf16 ≈ 12 KB/토큰)와 DeltaNet 상태
#: (18층 × 16 head × 128² × fp32 ≈ 19 MB/열)가 1 GiB 남짓. 적재 때 dtype 변환의 일시적 사본과 expandable segment의 조각까지 덮도록
#: 필요량의 약 3배를 요구한다 — 다른 프로세스(데스크톱 포함)가 통합 메모리를 쓰는 상자라 여유를 넉넉히 둔다.
MIN_FREE_BYTES = {"Qwen/Qwen3.5-2B": 20 * 2**30, "Qwen/Qwen3.5-4B": 30 * 2**30}
_SELF_ANSWER = re.compile(r"(?i)(?:final\s+)?answer\s*:")


def special_id(tokenizer: Any, token: str) -> int:
    """특수 토큰 하나의 id (`tokenizers.Tokenizer`의 ``token_to_id``; 없으면 오류)."""
    found = tokenizer.token_to_id(token)
    if found is None:
        raise ValueError(f"tokenizer에 {token!r}이 없다")
    return int(found)


def score_code_prompts(backbone: Any, prompts: list[dict[str, Any]], *, batch: int = DEFAULT_BATCH, code_mass: bool = True) -> list[dict[str, Any]]:
    """프롬프트(``tokens``·``code_ids``) → ``{"code_probs", "code_logits", "code_mass"}`` (입력 순서). 모듈 설명의 방식."""
    head = backbone.model.lm_head.weight
    device = backbone.device
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]["tokens"]), i))
    out: list[dict[str, Any] | None] = [None] * len(prompts)
    with torch.no_grad():
        head32 = head.float() if code_mass and prompts else None  # 전체 어휘 logsumexp용 — 호출마다 한 번
        for start in range(0, len(order), max(1, int(batch))):
            index = order[start : start + max(1, int(batch))]
            group = [prompts[i] for i in index]
            length = max(len(p["tokens"]) for p in group)
            ids = torch.zeros(len(group), length, dtype=torch.long)
            for row, p in enumerate(group):
                ids[row, : len(p["tokens"])] = torch.tensor(p["tokens"], dtype=torch.long)
            positions = torch.arange(length)[None].expand(len(group), length)
            hidden = backbone(ids.to(device), positions.to(device))["hidden"]
            last = torch.stack([hidden[row, len(p["tokens"]) - 1] for row, p in enumerate(group)]).float()
            lse = torch.logsumexp(last @ head32.T, dim=1) if head32 is not None else None
            for row, p in enumerate(group):
                rows = head[torch.tensor(p["code_ids"], device=device)].float()
                logits = rows @ last[row]
                probs = torch.softmax(logits, 0)
                mass = float(torch.exp(torch.logsumexp(logits, 0) - lse[row])) if lse is not None else None
                out[index[row]] = {"code_probs": probs.cpu().tolist(), "code_logits": logits.cpu().tolist(), "code_mass": mass}
    return out  # type: ignore[return-value]


def _stop_ids(tokenizer: Any, fmt: str) -> dict[int, str]:
    stops = {special_id(tokenizer, "<|im_end|>"): "eos", special_id(tokenizer, "<|endoftext|>"): "eos"}
    if fmt == "chat":
        stops[special_id(tokenizer, "</think>")] = "think_end"
    return stops


def generate_reasoning(
    backbone: Any, prompts: list[dict[str, Any]], tokenizer: Any, *, max_new_tokens: int = REASONING_MAX_NEW_TOKENS, batch: int = DEFAULT_GEN_BATCH
) -> list[dict[str, Any]]:
    """reasoning 프롬프트 → ``{"generated": [토큰 id…]}`` (greedy, 왼쪽 padding 배치, 입력 순서). 끝 토큰에서 열마다 멈춘다."""
    model = backbone.model
    device = backbone.device
    pad = special_id(tokenizer, "<|endoftext|>")
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]["tokens"]), i))
    out: list[dict[str, Any] | None] = [None] * len(prompts)
    with torch.no_grad():
        for start in range(0, len(order), max(1, int(batch))):
            index = order[start : start + max(1, int(batch))]
            group = [prompts[i] for i in index]
            stops = sorted(set().union(*(_stop_ids(tokenizer, p["format"]) for p in group)))
            length = max(len(p["tokens"]) for p in group)
            ids = torch.full((len(group), length), pad, dtype=torch.long)
            mask = torch.zeros(len(group), length, dtype=torch.long)
            for row, p in enumerate(group):
                ids[row, length - len(p["tokens"]) :] = torch.tensor(p["tokens"], dtype=torch.long)
                mask[row, length - len(p["tokens"]) :] = 1
            generated = model.generate(
                input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=int(max_new_tokens), do_sample=False,
                num_beams=1, eos_token_id=stops, pad_token_id=pad, use_cache=True,
            )
            for row in range(len(group)):
                out[index[row]] = {"generated": generated[row, length:].cpu().tolist()}
    return out  # type: ignore[return-value]


def finish_reasoning(prompt: dict[str, Any], generated: list[int], tokenizer: Any, *, cap: int) -> dict[str, Any]:
    """생성 토큰 → 생각 글(끝 토큰 앞, 모델이 쓴 ``Answer:`` 앞에서 자름)과 채점할 토큰열 (프롬프트 + 생각 + :func:`reasoning_close`).

    ``stop``: ``eos``(끝 토큰), ``think_end``(chat의 ``</think>``), ``cap``(상한에 닿음), ``end``(그 밖). ``generated_tokens``는 멈춘 토큰까지
    포함한 생성 길이, ``self_answer``는 모델이 스스로 답 표지를 썼는지.
    """
    stops = _stop_ids(tokenizer, prompt["format"])
    cut, stop = len(generated), ("cap" if len(generated) >= cap else "end")
    for i, token in enumerate(generated):
        if int(token) in stops:
            cut, stop = i, stops[int(token)]
            break
    text = tokenizer.decode(list(generated[:cut]), skip_special_tokens=False)
    match = _SELF_ANSWER.search(text)
    if match:
        text = text[: match.start()]
    text = text.rstrip()
    tokens = list(prompt["tokens"]) + list(tokenizer.encode(text, add_special_tokens=False).ids) + list(
        tokenizer.encode(reasoning_close(prompt["format"]), add_special_tokens=False).ids
    )
    return {
        "reasoning": text, "stop": stop, "generated_tokens": min(cut + (0 if stop in ("cap", "end") else 1), len(generated)),
        "self_answer": match is not None, "tokens": tokens, "code_ids": prompt["code_ids"],
    }


def diagnose_records(
    records: list[dict[str, Any]],
    backbone: Any,
    tokenizer: Any,
    *,
    model_id: str,
    mode: str = "direct",
    fmt: str = "plain",
    seed: int | None = 1,
    batch: int = DEFAULT_BATCH,
    gen_batch: int = DEFAULT_GEN_BATCH,
    max_new_tokens: int = REASONING_MAX_NEW_TOKENS,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """기록 → 기록마다 예측 하나 (기록 순서). 프롬프트는 기록의 ``request``만으로 만든다(:func:`record_prompts`).

    예측: ``{"id", "model", "mode", "format", "prompt_version", "shuffle_seed", "questions": {이름: {"type", "options", "values",
    "codes", "code_probs", "code_mass", "answer", "prompt_tokens", "scored_tokens", "reasoning"?}}}``.
    """
    prompts = [p for record in records for p in record_prompts(record, tokenizer, seed=seed, fmt=fmt, mode=mode)]
    started = time.perf_counter()
    if mode == "reasoning":
        generations = generate_reasoning(backbone, prompts, tokenizer, max_new_tokens=max_new_tokens, batch=gen_batch)
        if log:
            log(f"generated {len(prompts)} reasonings in {time.perf_counter() - started:.1f} s")
        finished = [finish_reasoning(p, g["generated"], tokenizer, cap=max_new_tokens) for p, g in zip(prompts, generations)]
        scoring = [{"tokens": f["tokens"], "code_ids": f["code_ids"]} for f in finished]
    else:
        finished = [None] * len(prompts)
        scoring = prompts
    scored = score_code_prompts(backbone, scoring, batch=batch)
    if log:
        log(f"scored {len(prompts)} prompts in {time.perf_counter() - started:.1f} s")
    by_id: dict[str, dict[str, Any]] = {}
    for p, s, f in zip(prompts, scored, finished):
        entry = by_id.setdefault(p["record_id"], {
            "id": p["record_id"], "model": model_id, "mode": mode, "format": fmt, "prompt_version": PROMPT_VERSION,
            "shuffle_seed": seed, "questions": {},
        })
        question = {
            "type": p["type"], "options": p["options"], "values": p["values"], "codes": p["codes"], "code_probs": s["code_probs"],
            "code_mass": s["code_mass"], "answer": answer_from_codes(p, s["code_probs"]), "prompt_tokens": len(p["tokens"]),
            "scored_tokens": len(f["tokens"]) if f else len(p["tokens"]),
        }
        if f is not None:
            question["reasoning"] = {"text": f["reasoning"], "stop": f["stop"], "generated_tokens": f["generated_tokens"], "self_answer": f["self_answer"]}
        entry["questions"][p["question"]] = question
    return [by_id[str(record.get("id") or "")] for record in records if str(record.get("id") or "") in by_id]
