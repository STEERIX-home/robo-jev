"""OpenRouter Decisions API 클라이언트 (Task bench0 §2) — 같은 판단 요청을 OpenRouter의 결정 모델에 보내고 우리 예측 형식으로 받는다.

**끝점** (문서 2026-10-12, 통제자 실측): ``POST https://openrouter.ai/api/alpha/decisions``, 머리 ``Authorization: Bearer <키>``·
``Content-Type: application/json``·``User-Agent``, 몸 ``{model, state, questions}``(+ 선택 ``provider``). 응답 ``answers[이름]`` —
noul ``{"noul": p}``, choice ``{"choice", "probabilities", "confidence"}``, score ``{"score", "probabilities", "legend", "confidence"}``
(``score`` = 기대 단계 번호 0..n−1, ``probabilities``의 열쇠는 단계 번호 글자) — 와 ``usage{input_tokens, output_tokens, cost}``,
``model``(해석된 판), ``provider``, ``id``. 모델 목록은 ``GET /api/v1/models?output_modalities=decisions``(공개, 키 없음).

**요청 변환** (:func:`api_request`) — 레코드의 ``request``를 그대로 쓰되:

* 상태 글은 문자열 그대로(``state``는 글·객체·배열 모두 받는다).
* ``choice`` — criteria 맵 그대로(저장된 순서). ``noul`` — instructions 그대로, criteria 없음.
* ``score`` — **11단계**(:func:`score_levels`): 0.0, 0.1, …, 1.0이고 양 끝 단계에 원래 기준점 글을 붙인다
  (``"0.0 (0 = no chance of contact)"`` … ``"1.0 (1 = contact is certain)"``) — 우리 모델이 p3 프롬프트에서 보는 것과 같은 기준점이다.
  API의 score는 기준 목록을 순서 있는 단계로 읽으므로 원래 기준점 둘을 그대로 보내면 "닿을 확률"이 아니라 두 단계 사이의 위치로
  읽힌다. 값 = 기대 단계 / 10(:func:`score_value`). 시운전(``levels=2``)은 원래 기준점 둘을 그대로 보내고 값 = 기대 단계.

**응답 → 예측** (:func:`to_prediction`) — 진단 도구(``scripts/diagnose_decisions.py``)의 ``.predictions.jsonl`` 꼴: 질문마다 listing
**하나**(보낸 순서 그대로 — 우리 모델의 균형 배치와 달리 한 순서로 한 번 묻는다). 그래서 ``--rescore``·``--compare``·
``convert_single_probe.py compare``가 바뀌지 않고 점수를 낸다. noul의 listing은 보이는 선택지가 없어 ``options``가 없다.

**운영**: 동시성(기본 8), 재시도(429·5xx·524·529와 연결 오류·시간 초과에 지수 backoff — ``Retry-After``를 따른다), 디스크 캐시
(:class:`ResponseCache` — (모델, 레코드 id, 요청 sha256)마다; 다시 돌리면 돈이 들지 않는다), **돈 상한**(:class:`Ledger` — 모델별·전체를
``usage.cost`` 합으로 세고, 보내기 **전에** 이 요청의 보수적 예약까지 더해 상한을 넘을지 본다; 장부는 누적이다), dry-run 견적
(:func:`estimate_request` — 돈을 쓰지 않는다).

**키** — ``~/.config/robo-jev/openrouter.env``의 ``OPENROUTER_API_KEY``만 읽는다(RunPod 키와 같은 방식: 환경 변수·인자는 보지 않고,
그룹·다른 사용자가 읽을 수 있는 파일은 거절한다). 값은 :class:`~robo_jev.launch.providers.runpod.Secret` 에 담겨 ``repr``·서식에서
``***``이고, 오류 문장은 :func:`~robo_jev.launch.providers.runpod.redact` 를 거친다 — 키가 로그·예외·캐시·장부에 나가지 않는다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import math
import re
import stat
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from robo_jev.launch.providers.runpod import MissingApiKey, Secret, UnsafeKeyFile, redact, urllib_transport

__all__ = [
    "API_URL",
    "BudgetExceeded",
    "DEFAULT_KEY_FILE",
    "DecisionsClient",
    "DecisionsError",
    "FORMAT",
    "Job",
    "KEY_NAME",
    "Ledger",
    "MODELS_URL",
    "ResponseCache",
    "api_request",
    "estimate_request",
    "fetch_models",
    "load_api_key",
    "price_of",
    "request_sha256",
    "run_requests",
    "safe_name",
    "score_levels",
    "score_value",
    "select_models",
    "to_prediction",
]

API_URL = "https://openrouter.ai/api/alpha/decisions"
MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=decisions"
DEFAULT_KEY_FILE = Path("~/.config/robo-jev/openrouter.env")
KEY_NAME = "OPENROUTER_API_KEY"
USER_AGENT = "robo-jev-bench/0.1 (task bench0; Steerix Robotics)"
#: 예측 줄의 채점 조건 칸 — 진단 도구의 ``format``·``mode``·``orders`` 자리.
FORMAT = "openrouter-decisions"
MODE = "api"
ORDERS = "stored"
#: 다시 보내는 상태 — 429(rate limit), 5xx, 524(edge 시간 초과), 529(공급자 과부하), 408.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
#: 이 상태면 그 뒤의 요청도 모두 실패한다 — 실행 전체를 멈춘다(키 없음·잔액 없음·권한 없음).
FATAL_STATUSES = frozenset({401, 402, 403})
#: 응답 없이 끝났지만 공급자가 처리했을 수 있는 시도 — 예약액을 "불확실한 지출"로 장부에 남긴다(상한이 보수적이게).
UNCERTAIN_STATUSES = frozenset({524})
DEFAULT_TIMEOUT = 60.0
DEFAULT_ATTEMPTS = 6
DEFAULT_CONCURRENCY = 8
#: 예약 = 견적(높은 쪽) × 이 값 — 실제가 예약을 넘으면 그 모델의 배수를 실제/견적 × 1.5로 올린다.
DEFAULT_RESERVE_FACTOR = 2.0
#: dry-run의 글 → 토큰 (Qwen tokenizer가 없을 때): 글자 / 4.
CHARS_PER_TOKEN = 4.0


class DecisionsError(RuntimeError):
    """API 오류 — 문장은 언제나 키를 지운 뒤의 것이다. ``status``는 HTTP 상태(전송 실패는 None)."""

    def __init__(self, message: str, *, status: int | None = None, fatal: bool = False, uncertain: bool = False):
        super().__init__(message)
        self.status = status
        self.fatal = fatal
        self.uncertain = uncertain


class BudgetExceeded(RuntimeError):
    """이 요청을 보내면 상한을 넘는다 — 보내지 않았다. ``scope``는 ``"model"``(그 모델만 멈춘다) 또는 ``"total"``(전체가 멈춘다)."""

    def __init__(self, message: str, *, scope: str):
        super().__init__(message)
        self.scope = scope


# --------------------------------------------------------------------------
# 키
# --------------------------------------------------------------------------


def _parse_key_text(text: str, path: Path) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    assigned = []
    for line in lines:
        body = line[len("export "):].strip() if line.startswith("export ") else line
        name, sep, value = body.partition("=")
        if sep and name.strip() == KEY_NAME:
            assigned.append(value.strip())
    if len(assigned) != 1:
        raise MissingApiKey(f"{path}: `{KEY_NAME}=<키>` 줄이 {len(assigned)}개다 — 하나만 둔다 (값은 찍지 않는다)")
    value = assigned[0]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    if not value or any(ch.isspace() for ch in value) or not value.isprintable() or len(value) < 8:
        raise MissingApiKey(f"{path}: 키 값의 꼴이 이상하다 — 값은 찍지 않는다")
    return value


def load_api_key(path: str | Path | None = None) -> Secret:
    """git 밖의 키 파일에서 OpenRouter 키를 읽는다. **환경 변수·인자는 보지 않는다.** 없으면 :class:`MissingApiKey`,
    그룹·다른 사용자가 읽을 수 있으면 :class:`UnsafeKeyFile`(값을 읽기 전에 거절)."""
    target = Path(path if path is not None else DEFAULT_KEY_FILE).expanduser()
    if not target.is_file():
        raise MissingApiKey(f"{target}: OpenRouter 키 파일이 없다 — `{KEY_NAME}=<키>` 한 줄로 적고 `chmod 600`한다")
    mode = target.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise UnsafeKeyFile(f"{target}: 권한이 {stat.filemode(mode)}다 — `chmod 600 {target}` 뒤 다시")
    return Secret(_parse_key_text(target.read_text(encoding="utf-8"), target))


# --------------------------------------------------------------------------
# 요청 변환
# --------------------------------------------------------------------------


def score_levels(criteria: list[str], levels: int) -> list[str]:
    """score 질문의 단계 글. ``levels=11``: 0.0 … 1.0, 양 끝에 원래 기준점 글을 괄호로 붙인다. ``levels=2``: 원래 기준점 그대로."""
    criteria = [" ".join(str(item).split()) for item in (criteria or [])]
    if levels == 2:
        if len(criteria) != 2:
            raise ValueError(f"단계 둘 그대로는 기준점이 둘인 score에만 (받은 것 {len(criteria)}개)")
        return list(criteria)
    if levels != 11:
        raise ValueError(f"score 단계는 11(기본) 또는 2(시운전 — 그대로)다 (받은 값 {levels})")
    out = [f"{i / 10:.1f}" for i in range(11)]
    if criteria:
        out[0] = f"0.0 ({criteria[0]})"
        out[-1] = f"1.0 ({criteria[-1]})"
    return out


def _api_question(question: dict[str, Any], levels: int) -> dict[str, Any]:
    kind = question.get("type")
    instructions = question.get("instructions")
    if kind == "choice":
        return {"type": "choice", "instructions": instructions, "criteria": dict(question["criteria"])}
    if kind == "noul":
        return {"type": "noul", "instructions": instructions}
    if kind == "score":
        return {"type": "score", "instructions": instructions, "criteria": score_levels(list(question.get("criteria") or []), levels)}
    raise ValueError(f"알 수 없는 질문 종류 {kind!r}")


def api_request(
    record: dict[str, Any], model: str, *, levels: int = 11, only: Iterable[str] | None = None, provider: dict[str, Any] | None = None
) -> dict[str, Any]:
    """레코드 → API 요청 몸. 모델 입력은 ``request``(상태 글·질문)뿐이다 — id·meta·정답은 보내지 않는다. `only`는 그 질문만(시운전)."""
    request = record["request"]
    wanted = set(only) if only is not None else None
    questions = {name: _api_question(q, levels) for name, q in request["questions"].items() if wanted is None or name in wanted}
    if not questions:
        raise ValueError(f"{record.get('id')}: 보낼 질문이 없다")
    payload: dict[str, Any] = {"model": model, "state": request["state"], "questions": questions}
    if provider:
        payload["provider"] = dict(provider)
    return payload


def request_sha256(payload: dict[str, Any]) -> str:
    """요청의 sha256 — 모델 이름을 뺀 몸(상태·질문·provider)의 정규 JSON. 캐시 열쇠는 (모델, 레코드 id, 이것)."""
    body = {k: v for k, v in payload.items() if k != "model"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 응답 → 예측
# --------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def score_value(answer: dict[str, Any], n_levels: int) -> tuple[float, dict[str, float] | None]:
    """score 답 → (값 ∈ [0, 1], 단계별 확률). 값 = 기대 단계 / (n − 1) — 응답의 ``score``(공식 기대 단계)를 쓰고, 없으면
    ``probabilities``(단계 번호 → 확률, 합으로 나눈다)에서 다시 계산한다."""
    probabilities = answer.get("probabilities")
    bins = None
    if isinstance(probabilities, dict) and probabilities:
        parsed = {int(k): _number(v) for k, v in probabilities.items() if str(k).lstrip("-").isdigit()}
        if all(v is not None and v >= 0 for v in parsed.values()) and parsed and all(0 <= k < n_levels for k in parsed):
            bins = {k: float(v) for k, v in parsed.items()}
    index = _number(answer.get("score"))
    if index is None:
        if not bins or sum(bins.values()) <= 0:
            raise ValueError("score 답에 score도 쓸 수 있는 probabilities도 없다")
        total = sum(bins.values())
        index = sum(k * v for k, v in bins.items()) / total
    value = min(max(index / (n_levels - 1), 0.0), 1.0)
    return value, ({str(k): v for k, v in sorted(bins.items())} if bins else None)


def _answer(question: dict[str, Any], raw: Any, levels: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """질문 하나의 API 답 → (우리 답, listing의 보이는 순서 칸). 꼴이 틀리면 ``ValueError``."""
    if not isinstance(raw, dict):
        raise ValueError("답이 객체가 아니다")
    kind = question["type"]
    if raw.get("type") not in (None, kind):
        raise ValueError(f"답의 종류 {raw.get('type')!r} ≠ 질문의 {kind!r}")
    if kind == "noul":
        p = _number(raw.get("noul"))
        if p is None or not 0.0 <= p <= 1.0:
            raise ValueError(f"noul 값 {raw.get('noul')!r}")
        return {"p_yes": p}, {"options": None, "values": None}
    if kind == "choice":
        keys = list(question["criteria"])
        probabilities = raw.get("probabilities")
        if isinstance(probabilities, dict) and probabilities:
            unknown = sorted(set(probabilities) - set(keys))
            if unknown:
                raise ValueError(f"choice 확률에 모르는 선택지 {unknown[:3]}")
            values = {k: _number(probabilities.get(k, 0.0)) for k in keys}
            if any(v is None or v < 0 for v in values.values()):
                raise ValueError("choice 확률이 수가 아니다")
            total = sum(values.values())
            if total <= 0:
                raise ValueError("choice 확률의 합이 0이다")
            answer = {"probabilities": {k: float(v) for k, v in values.items()}}
            if abs(total - 1.0) > 1e-6:  # 합이 1이 아니면(반올림 등) 나눠 분포로 — 원래 합은 남긴다
                answer = {"probabilities": {k: float(v) / total for k, v in values.items()}, "raw_sum": total}
        elif raw.get("choice") in keys:
            answer = {"probabilities": {k: (1.0 if k == raw["choice"] else 0.0) for k in keys}, "one_hot": True}
        else:
            raise ValueError(f"choice 답에 확률도 아는 선택지도 없다 ({raw.get('choice')!r})")
        return answer, {"options": keys, "values": None}
    n = len(question["criteria"])
    value, bins = score_value(raw, n)
    names = [f"{i / (n - 1):.1f}" for i in range(n)] if n == 11 else [str(i) for i in range(n)]
    answer = {"value": value, "bins": {names[int(k)]: p for k, p in bins.items()} if bins else None}
    return answer, {"options": names, "values": [i / (n - 1) for i in range(n)]}


def to_prediction(
    record: dict[str, Any], payload: dict[str, Any], response: dict[str, Any], *, model: str, levels: int, cached: bool = False,
    request_hash: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """API 응답 → 진단 도구의 예측 줄 하나와 문제 목록. 답이 없거나 꼴이 틀린 질문은 빼고(채점에서 ``missing_prediction``) 문제로 적는다."""
    answers = response.get("answers") if isinstance(response, dict) else None
    problems: list[str] = []
    if not isinstance(answers, dict):
        return {}, ["response has no answers object"]
    questions: dict[str, Any] = {}
    for name, question in payload["questions"].items():
        if name not in answers:
            problems.append(f"{name}: no answer")
            continue
        try:
            answer, shown = _answer(question, answers[name], levels)
        except (ValueError, TypeError, KeyError) as error:
            problems.append(f"{name}: {error}")
            continue
        listing = {**shown, "codes": None, "code_probs": None, "code_mass": None, "answer": answer, "raw": answers[name]}
        questions[name] = {"type": question["type"], "n_listings": 1, "answer": answer, "code_mass": None, "listings": [listing]}
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    prediction = {
        "id": str(record["id"]), "model": model, "mode": MODE, "format": FORMAT, "prompt_version": f"{FORMAT}/score{levels}",
        "orders": ORDERS, "shuffle_seed": None,
        "api": {"model": response.get("model"), "provider": response.get("provider"), "id": response.get("id"), "usage": usage,
                "request_sha256": request_hash, "cached": bool(cached)},
        "questions": questions,
    }
    return prediction, problems


# --------------------------------------------------------------------------
# 모델 목록·가격
# --------------------------------------------------------------------------


def fetch_models(*, transport: Callable[..., Any] | None = None, url: str = MODELS_URL, timeout: float = 30.0) -> dict[str, Any]:
    """결정 모델 목록 (공개 GET — 키를 보내지 않는다)."""
    status, _, body = (transport or urllib_transport)("GET", url, {"User-Agent": USER_AGENT, "Accept": "application/json"}, None, timeout)
    if status != 200:
        raise DecisionsError(f"model list GET returned {status}", status=status)
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise DecisionsError("model list has no data array")
    return data


def price_of(entry: dict[str, Any]) -> dict[str, float]:
    """목록 항목 → USD/토큰 가격 (prompt·completion; 없으면 0)."""
    pricing = entry.get("pricing") or {}
    return {key: float(pricing.get(key) or 0.0) for key in ("prompt", "completion")}


def select_models(catalog: dict[str, Any], wanted: Iterable[str] | None, *, exclude_suffixes: Iterable[str] = (":free",)) -> list[dict[str, Any]]:
    """목록에서 고른 모델들 (`wanted`의 순서; None이면 목록 전부). ``:free`` 복제본은 뺀다. 목록에 없는 모델은 ``ValueError``."""
    suffixes = tuple(exclude_suffixes)
    entries = {str(e.get("id")): e for e in catalog.get("data") or []}
    if wanted is None:
        return [e for mid, e in entries.items() if not mid.endswith(suffixes)]
    out, missing = [], []
    for mid in wanted:
        if mid.endswith(suffixes):
            raise ValueError(f"{mid}: :free 복제본은 쓰지 않는다 (같은 모델, 엄격한 rate limit)")
        if mid not in entries:
            missing.append(mid)
            continue
        out.append(entries[mid])
    if missing:
        raise ValueError(f"모델 목록에 없다: {missing}")
    return out


def safe_name(model: str) -> str:
    """모델 id → 파일 이름 (``/``→``__``, ``~``→``tilde-``, ``:``→``_``)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model.replace("~", "tilde-").replace("/", "__"))


# --------------------------------------------------------------------------
# dry-run 견적
# --------------------------------------------------------------------------


def _token_counter(tokenizer: Any | None) -> Callable[[str], int]:
    if tokenizer is None:
        return lambda text: int(math.ceil(len(text) / CHARS_PER_TOKEN))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def estimate_request(payload: dict[str, Any], *, tokenizer: Any | None = None, overhead: dict[str, float] | None = None) -> dict[str, Any]:
    """요청 하나의 입력 토큰 견적 — 공급자의 내부 프롬프트는 모르므로 두 끝을 낸다.

    * ``low`` — 상태를 한 번 센다: 상태 + Σ 질문(지시·기준의 글) + 질문마다 덧붙는 몫(``overhead["per_question_low"]``).
    * ``high`` — 질문마다 상태를 다시 센다(질문이 서로의 답을 보지 않고 따로 답한다 — 문서): 질문 수 × 상태 + Σ 질문 + 질문마다 몫(``per_question_high``);
      낮은 끝보다 작으면(질문이 하나일 때) 낮은 끝 — 상한 예약에 쓰는 보수적인 끝이다.
    질문의 글은 API 몸의 그 질문 JSON(``ensure_ascii=False``) 그대로 센다. 덧붙는 몫의 기본값은 문서 예(입력 476토큰)에 맞춘 값이다."""
    count = _token_counter(tokenizer)
    overhead = overhead or {}
    state = payload["state"] if isinstance(payload["state"], str) else json.dumps(payload["state"], ensure_ascii=False)
    s = count(state)
    qs = [count(json.dumps(q, ensure_ascii=False)) for q in payload["questions"].values()]
    n = len(qs)
    low = s + sum(qs) + n * float(overhead.get("per_question_low", 0.0))
    high = max(low, n * s + sum(qs) + n * float(overhead.get("per_question_high", 0.0)))  # 높은 끝은 언제나 보수적인 쪽 (질문 하나면 두 맞춤이 엇갈린다)
    return {"questions": n, "state_tokens": s, "question_tokens": sum(qs), "low": int(round(low)), "high": int(round(high)), "chars": len(state)}


# --------------------------------------------------------------------------
# 캐시
# --------------------------------------------------------------------------


class ResponseCache:
    """(모델, 레코드 id, 요청 sha256)마다 응답 하나 — ``<root>/<모델>/<레코드 id>.<sha 앞 16>.json``. 성공(200, 답 있음)만 둔다."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, model: str, record_id: str, sha: str) -> Path:
        return self.root / safe_name(model) / f"{safe_name(record_id)}.{sha[:16]}.json"

    def get(self, model: str, record_id: str, sha: str) -> dict[str, Any] | None:
        path = self.path(model, record_id, sha)
        if not path.is_file():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("request_sha256") != sha or entry.get("model") != model or entry.get("record_id") != record_id:
            return None
        return entry

    def put(self, model: str, record_id: str, sha: str, response: dict[str, Any], *, meta: dict[str, Any]) -> None:
        path = self.path(model, record_id, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"model": model, "record_id": record_id, "request_sha256": sha, "response": response, **meta,
                 "stored": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(path)


# --------------------------------------------------------------------------
# 장부와 상한
# --------------------------------------------------------------------------


@dataclass
class _Account:
    spent_usd: float = 0.0
    uncertain_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_from_usage: int = 0
    cost_computed: int = 0

    def total(self) -> float:
        return self.spent_usd + self.uncertain_usd


@dataclass
class Ledger:
    """돈 장부 (누적 — 이 도구로 쓴 모든 돈). 상한은 실행 인자로만 받고(기본값 없음) **누적 합**에 건다: 보내기 전에
    ``지출 + 불확실 + 진행 중 예약 + 이 요청의 예약 ≤ 상한``(모델별·전체)이 아니면 :class:`BudgetExceeded` — 보내지 않는다.

    예약 = 견적(높은 쪽) × 배수(기본 :data:`DEFAULT_RESERVE_FACTOR`; 실제 비용이 예약을 넘은 적이 있으면 그 모델의 배수를
    실제/견적 × 1.5로 올린다). 응답이 오면 예약을 실제(``usage.cost``, 없으면 토큰 × 목록 가격)로 바꾼다. 시간 초과·524처럼 처리됐을지
    모르는 시도는 예약액을 ``uncertain_usd``로 남긴다. 그래서 남은 초과 위험은 진행 중인 요청이 저마다 예약을 넘는 몫뿐이다."""

    path: Path | None
    max_total_usd: float
    max_model_usd: float
    reserve_factor: float = DEFAULT_RESERVE_FACTOR
    models: dict[str, _Account] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    _reserved: dict[str, float] = field(default_factory=dict)
    _factor: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        for name, value in (("max_total_usd", self.max_total_usd), ("max_model_usd", self.max_model_usd)):
            if value is None or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name}: 0 이상의 수를 반드시 준다 (기본값 없음)")
        if self.path is not None and Path(self.path).is_file():
            data = json.loads(Path(self.path).read_text(encoding="utf-8"))
            self.models = {name: _Account(**{k: v for k, v in acc.items() if k in _Account.__dataclass_fields__}) for name, acc in (data.get("models") or {}).items()}
            self.runs = list(data.get("runs") or [])

    def spent(self, model: str | None = None) -> float:
        if model is not None:
            return self.models.get(model, _Account()).total()
        return sum(acc.total() for acc in self.models.values())

    def reserve(self, model: str, estimate_usd: float) -> float:
        """이 요청의 예약액을 잡는다 — 상한을 넘으면 :class:`BudgetExceeded` (아무것도 잡지 않는다)."""
        with self._lock:
            amount = max(0.0, float(estimate_usd)) * max(self.reserve_factor, self._factor.get(model, 0.0))
            model_after = self.spent(model) + self._reserved.get(model, 0.0) + amount
            total_after = self.spent() + sum(self._reserved.values()) + amount
            if total_after > self.max_total_usd + 1e-12:
                raise BudgetExceeded(f"total ${total_after:.6f} would exceed the cap ${self.max_total_usd:.4f}", scope="total")
            if model_after > self.max_model_usd + 1e-12:
                raise BudgetExceeded(f"{model}: ${model_after:.6f} would exceed the per-model cap ${self.max_model_usd:.4f}", scope="model")
            self._reserved[model] = self._reserved.get(model, 0.0) + amount
            return amount

    def settle(self, model: str, reserved: float, estimate_usd: float, *, cost: float | None, usage: dict[str, Any] | None,
               price: dict[str, float] | None, uncertain: bool = False) -> float:
        """예약을 실제로 바꾼다. `uncertain`이면(응답 없이 끝났지만 처리됐을 수 있다) 예약액을 불확실한 지출로 남긴다. 실제 비용을 돌려준다."""
        with self._lock:
            self._reserved[model] = max(0.0, self._reserved.get(model, 0.0) - reserved)
            account = self.models.setdefault(model, _Account())
            if uncertain:
                account.uncertain_usd += reserved
                return reserved
            usage = usage or {}
            if cost is None:
                price = price or {}
                cost = float(usage.get("input_tokens") or 0) * price.get("prompt", 0.0) + float(usage.get("output_tokens") or 0) * price.get("completion", 0.0)
                account.cost_computed += 1
            else:
                account.cost_from_usage += 1
            account.spent_usd += float(cost)
            account.calls += 1
            account.input_tokens += int(usage.get("input_tokens") or 0)
            account.output_tokens += int(usage.get("output_tokens") or 0)
            if estimate_usd > 0 and cost > reserved:
                self._factor[model] = max(self._factor.get(model, 0.0), cost / estimate_usd * 1.5)
            return float(cost)

    def release(self, model: str, reserved: float) -> None:
        """보내지 못한(또는 처리되지 않았다고 확실한 — 429 등) 시도의 예약을 푼다."""
        with self._lock:
            self._reserved[model] = max(0.0, self._reserved.get(model, 0.0) - reserved)

    def add_uncertain(self, model: str, amount: float) -> None:
        """성공한 요청의 앞선 시도 중 처리됐을지 모르는 것(시간 초과·524)의 몫 — 예약은 건드리지 않는다."""
        with self._lock:
            self.models.setdefault(model, _Account()).uncertain_usd += max(0.0, float(amount))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            models = {name: {**acc.__dict__, "total_usd": acc.total()} for name, acc in sorted(self.models.items())}
            return {
                "kind": "openrouter-ledger-v1", "currency": "USD", "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "note": "cumulative spend through scripts/bench.py; caps are checked against these totals before every request",
                "totals": {"spent_usd": sum(m["spent_usd"] for m in models.values()), "uncertain_usd": sum(m["uncertain_usd"] for m in models.values()),
                           "calls": sum(m["calls"] for m in models.values()), "input_tokens": sum(m["input_tokens"] for m in models.values()),
                           "output_tokens": sum(m["output_tokens"] for m in models.values())},
                "models": models, "runs": list(self.runs),
            }

    def save(self) -> None:
        if self.path is None:
            return
        data = self.snapshot()
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f".{threading.get_ident()}.tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        temp.replace(path)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _error_text(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            return f"{error.get('code')}: {error.get('message')}"
    except (ValueError, UnicodeDecodeError):
        pass
    return body[:300].decode("utf-8", "replace")


class DecisionsClient:
    """``POST /api/alpha/decisions`` 하나를 재시도와 함께 보낸다. 오류 문장은 키를 지운다(:func:`redact`)."""

    def __init__(
        self, key: Secret, *, url: str = API_URL, transport: Callable[..., Any] | None = None, timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_ATTEMPTS, backoff_base: float = 1.0, backoff_max: float = 60.0, sleep: Callable[[float], None] = time.sleep,
    ):
        if not isinstance(key, Secret):
            raise TypeError("key는 Secret이어야 한다 (load_api_key)")
        self._key = key
        self.url = url
        self.transport = transport or urllib_transport
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self.sleep = sleep

    def __repr__(self) -> str:
        return f"DecisionsClient(url={self.url!r}, key=***)"

    def _delay(self, attempt: int, headers: dict[str, str]) -> float:
        delay = min(self.backoff_max, self.backoff_base * 2 ** (attempt - 1))
        retry_after = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
        if retry_after is not None:
            try:
                delay = min(self.backoff_max, max(delay, float(retry_after)))
            except ValueError:
                pass
        return delay

    def post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """→ (응답 JSON, 시도 정보 ``{"attempts", "statuses", "uncertain_attempts", "seconds"}``). 실패는 :class:`DecisionsError`."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key.reveal()}", "Content-Type": "application/json", "User-Agent": USER_AGENT,
                   "Accept": "application/json"}
        statuses: list[Any] = []
        uncertain = 0
        started = time.perf_counter()
        last = "no attempt"
        for attempt in range(1, self.max_attempts + 1):
            try:
                status, response_headers, data = self.transport("POST", self.url, headers, body, self.timeout)
            except (OSError, http.client.HTTPException, urllib.error.URLError) as error:
                statuses.append("timeout" if isinstance(error, TimeoutError) or "timed out" in str(error) else "transport")
                uncertain += 1  # 보냈지만 답을 못 받았다 — 공급자가 처리했을 수 있다
                last = redact(f"{type(error).__name__}: {error}", self._key)
                if attempt < self.max_attempts:
                    self.sleep(self._delay(attempt, {}))
                continue
            statuses.append(int(status))
            if status == 200:
                try:
                    parsed = json.loads(data.decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as error:  # 처리(청구)됐을 수 있다 — 예약액을 불확실한 지출로 남긴다
                    raise DecisionsError(redact(f"200 with an unreadable body: {error}", self._key), status=200, uncertain=True) from None
                if not isinstance(parsed, dict):
                    raise DecisionsError("200 with a non-object body", status=200, uncertain=True)
                return parsed, {"attempts": attempt, "statuses": statuses, "uncertain_attempts": uncertain,
                                "seconds": round(time.perf_counter() - started, 3)}
            if status in UNCERTAIN_STATUSES:
                uncertain += 1
            last = redact(f"HTTP {status}: {_error_text(data)}", self._key)
            if status in RETRY_STATUSES and attempt < self.max_attempts:
                self.sleep(self._delay(attempt, response_headers or {}))
                continue
            if status in RETRY_STATUSES:
                break
            raise DecisionsError(last, status=int(status), fatal=status in FATAL_STATUSES, uncertain=uncertain > 0)
        raise DecisionsError(f"gave up after {len(statuses)} attempts ({statuses}); last: {last}", status=None, uncertain=uncertain > 0)


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------


@dataclass
class Job:
    """보낼 요청 하나 — 레코드, API 몸, 요청 sha256, 견적(USD, 높은 쪽), 단계 수."""

    record: dict[str, Any]
    payload: dict[str, Any]
    sha: str
    estimate_usd: float
    levels: int = 11


def run_requests(
    model: str, jobs: list[Job], *, client: DecisionsClient | None, cache: ResponseCache, ledger: Ledger, price: dict[str, float],
    concurrency: int = DEFAULT_CONCURRENCY, calls_log: Path | None = None, log: Callable[[str], None] | None = None,
    stop: threading.Event | None = None,
) -> dict[str, Any]:
    """한 모델의 요청들 → ``{"predictions": {레코드 id: 예측}, "errors": {id: 문장}, "skipped": {id: 이유}, "stats": {...}}``.

    캐시에 있으면 보내지 않는다. 없으면 장부에 예약 → 보내기 → 정산 → 캐시. 상한에 닿으면 그 요청과 그 뒤를 보내지 않고
    ``skipped``에 ``budget``으로 남긴다(`stop`이 서면 전체가 멈춘다 — 전체 상한·치명 오류). `client`가 None이면 캐시만 읽는다."""
    stop = stop or threading.Event()
    model_stop = threading.Event()
    lock = threading.Lock()
    out: dict[str, Any] = {"predictions": {}, "errors": {}, "skipped": {}, "problems": {},
                           "stats": {"cached": 0, "sent": 0, "cost_usd": 0.0, "fatal": None, "budget": None}}

    def note(path: Path | None, entry: dict[str, Any]) -> None:
        if path is None:
            return
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def finish(job: Job, response: dict[str, Any], *, cached: bool) -> None:
        rid = str(job.record["id"])
        prediction, problems = to_prediction(job.record, job.payload, response, model=model, levels=job.levels, cached=cached, request_hash=job.sha)
        with lock:
            if prediction.get("questions"):
                out["predictions"][rid] = prediction
            if problems:
                out["problems"][rid] = problems

    def one(job: Job) -> None:
        rid = str(job.record["id"])
        hit = cache.get(model, rid, job.sha)
        if hit is not None:
            with lock:
                out["stats"]["cached"] += 1
            finish(job, hit["response"], cached=True)
            return
        if stop.is_set() or model_stop.is_set():
            with lock:
                out["skipped"][rid] = "stopped"
            return
        if client is None:
            with lock:
                out["skipped"][rid] = "not cached (cache-only run)"
            return
        try:
            reserved = ledger.reserve(model, job.estimate_usd)
        except BudgetExceeded as error:
            model_stop.set()
            if error.scope == "total":
                stop.set()
            with lock:
                out["skipped"][rid] = "budget"
                out["stats"]["budget"] = str(error)
            return
        try:
            response, info = client.post(job.payload)
        except DecisionsError as error:
            if error.uncertain:
                ledger.settle(model, reserved, job.estimate_usd, cost=None, usage=None, price=price, uncertain=True)
            else:
                ledger.release(model, reserved)
            ledger.save()
            note(calls_log, {"time": time.time(), "model": model, "record_id": rid, "request_sha256": job.sha, "error": str(error),
                             "status": error.status, "uncertain": error.uncertain})
            with lock:
                out["errors"][rid] = str(error)
                if error.fatal:
                    out["stats"]["fatal"] = str(error)
            if error.fatal:
                stop.set()
            if error.status == 404:
                model_stop.set()
            return
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        cost = _number(usage.get("cost"))
        actual = ledger.settle(model, reserved, job.estimate_usd, cost=cost, usage=usage, price=price)
        if info["uncertain_attempts"]:  # 앞선 시도(시간 초과·524)가 처리됐을 수 있다 — 시도마다 예약액을 불확실로 남긴다
            ledger.add_uncertain(model, reserved * int(info["uncertain_attempts"]))
        ledger.save()
        meta = {"resolved_model": response.get("model"), "provider": response.get("provider"), "generation_id": response.get("id"),
                "usage": usage, "cost_usd": actual, "attempts": info["attempts"], "seconds": info["seconds"]}
        if isinstance(response.get("answers"), dict):  # 답이 없는 200(오류 몸)은 캐시하지 않는다 — 다시 돌리면 다시 묻는다
            cache.put(model, rid, job.sha, response, meta=meta)
        note(calls_log, {"time": time.time(), "model": model, "record_id": rid, "request_sha256": job.sha, **meta})
        with lock:
            out["stats"]["sent"] += 1
            out["stats"]["cost_usd"] += actual
        finish(job, response, cached=False)

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        list(pool.map(one, jobs))
    if log is not None:
        stats = out["stats"]
        log(f"{model}: {len(out['predictions'])}/{len(jobs)} answered ({stats['cached']} cached, {stats['sent']} sent, "
            f"${stats['cost_usd']:.5f}); {len(out['errors'])} errors, {len(out['skipped'])} skipped")
    return out
