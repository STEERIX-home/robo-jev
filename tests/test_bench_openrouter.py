"""OpenRouter Decisions 클라이언트 (Task bench0 §2·§5) — 로컬 가짜 HTTP 서버로: 세 답 종류 읽기, 429 backoff·재시도·시간 초과,
캐시 적중 시 호출 0, 상한에서 (넘기 전에) 멈춤, 키가 로그·예외·장부·캐시에 나오지 않음, 11단계 score 변환과 값 읽기.

유료 호출은 없다 — 모든 요청은 127.0.0.1의 가짜 서버로 간다."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from helpers import FIXTURES

from robo_jev.bench.openrouter import (
    BudgetExceeded,
    DecisionsClient,
    DecisionsError,
    Job,
    Ledger,
    ResponseCache,
    api_request,
    estimate_request,
    load_api_key,
    request_sha256,
    run_requests,
    score_levels,
    score_value,
    select_models,
    to_prediction,
)
from robo_jev.launch.providers.runpod import Secret, UnsafeKeyFile

REQUESTS = FIXTURES / "decisions" / "requests.jsonl"
KEY = "sk-or-v1-TESTKEY-0123456789abcdef"
#: 통제자가 무료 모델로 한 번 받은 실제 응답 (2026-10-12) — 세 답 종류의 꼴.
REAL_SAMPLE = {
    "model": "inception/mercury-decide-20260930",
    "answers": {
        "falls": {"type": "noul", "noul": 0.9840936056349263},
        "action": {"type": "choice", "choice": "move_cup", "probabilities": {"move_cup": 0.997817283879195, "ignore": 0.0021827161208050378},
                   "confidence": 0.99563456775839},
        "risk": {"type": "score", "score": 1.8956344258618827, "legend": {"0": "harmless", "1": "minor mess", "2": "burn risk"},
                 "probabilities": {"0": 0.010061371401536874, "1": 0.0842428313350436, "2": 0.9056957972634195}, "confidence": 0.8434516387928239},
    },
    "usage": {"input_tokens": 96, "output_tokens": 3, "cost": 0},
    "id": "gen-dec-test",
    "provider": "Inception",
}


def _records() -> list[dict]:
    return [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def auto_answer(body: dict) -> dict:
    """요청의 질문마다 정해진 답 (noul 0.7, choice 첫 키 0.6, score 단계 3) + 토큰 수 × 1e-7의 비용."""
    answers = {}
    for name, q in body["questions"].items():
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.7}
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            probs = {k: (0.6 if i == 0 else 0.4 / (len(keys) - 1)) for i, k in enumerate(keys)}
            answers[name] = {"type": "choice", "choice": keys[0], "probabilities": probs, "confidence": 0.5}
        else:
            n = len(q["criteria"])
            probs = {str(i): (1.0 if i == min(3, n - 1) else 0.0) for i in range(n)}
            answers[name] = {"type": "score", "score": float(min(3, n - 1)), "probabilities": probs, "legend": {str(i): c for i, c in enumerate(q["criteria"])}}
    tokens = len(json.dumps(body)) // 4
    return {"model": body["model"] + "-20261012", "provider": "Fake", "id": "gen-dec-fake", "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": len(answers), "cost": tokens * 1e-7}}


class FakeServer:
    """가짜 Decisions 서버. `script`는 요청마다 하나씩 꺼내는 (상태, 몸|None, 머리, 지연 초) — 비면 200 + :func:`auto_answer`."""

    def __init__(self, script=None, cost=None):
        self.script = list(script or [])
        self.cost = cost
        self.requests = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # 조용히
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                with outer.lock:
                    outer.requests.append({"headers": dict(self.headers.items()), "body": body})
                    action = outer.script.pop(0) if outer.script else (200, None, {}, 0.0)
                status, payload, headers, delay = action
                if delay:
                    time.sleep(delay)
                if payload is None:
                    payload = auto_answer(body)
                    if outer.cost is not None:
                        payload["usage"]["cost"] = outer.cost
                elif callable(payload):
                    payload = payload(self.headers, body)
                data = json.dumps(payload).encode("utf-8")
                try:
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):  # 시간 초과로 끊긴 연결의 쓰기 오류는 조용히
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/alpha/decisions"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _client(url: str, **kwargs) -> DecisionsClient:
    kwargs.setdefault("sleep", lambda s: None)
    kwargs.setdefault("timeout", 5.0)
    return DecisionsClient(Secret(KEY), url=url, **kwargs)


def _jobs(records, model="vendor/model", levels=11, estimate=1e-4):
    jobs = []
    for record in records:
        payload = api_request(record, model, levels=levels)
        jobs.append(Job(record=record, payload=payload, sha=request_sha256(payload), estimate_usd=estimate, levels=levels))
    return jobs


# --------------------------------------------------------------------------
# 변환과 읽기
# --------------------------------------------------------------------------


def test_eleven_level_score_rubric_carries_the_original_anchors_at_both_ends():
    levels = score_levels(["0 = no chance of contact", "1 = contact is certain"], 11)
    assert levels == ["0.0 (0 = no chance of contact)", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0 (1 = contact is certain)"]
    assert score_levels(["0 = a", "1 = b"], 2) == ["0 = a", "1 = b"]
    with pytest.raises(ValueError):
        score_levels(["0 = a", "1 = b"], 5)


def test_the_request_is_the_records_request_with_score_as_eleven_levels():
    record = _records()[0]
    payload = api_request(record, "typesafe/jev-1.13")
    assert set(payload) == {"model", "state", "questions"} and payload["state"] == record["request"]["state"]
    for name, q in record["request"]["questions"].items():
        sent = payload["questions"][name]
        assert sent["instructions"] == q["instructions"] and sent["type"] == q["type"]
        if q["type"] == "choice":
            assert sent["criteria"] == q["criteria"] and list(sent["criteria"]) == list(q["criteria"])  # 저장된 순서 그대로
        elif q["type"] == "noul":
            assert "criteria" not in sent
        else:
            assert len(sent["criteria"]) == 11 and sent["criteria"][0].startswith("0.0 (") and q["criteria"][0] in sent["criteria"][0]
    assert "meta" not in json.dumps(payload) or "meta" in record["request"]["state"]  # 정답·meta는 보내지 않는다
    assert request_sha256(payload) == request_sha256({**payload, "model": "other/model"})  # 모델 이름은 sha에 들지 않는다
    assert request_sha256(api_request(record, "m", levels=2)) != request_sha256(payload)


def test_score_value_is_the_expected_level_over_ten_or_recomputed_from_probabilities():
    value, bins = score_value({"score": 7.2, "probabilities": {"7": 0.8, "8": 0.2}}, 11)
    assert value == pytest.approx(0.72) and bins == {"7": 0.8, "8": 0.2}
    value, _ = score_value({"probabilities": {"0": 0.5, "10": 0.5}}, 11)
    assert value == pytest.approx(0.5)
    value, _ = score_value({"score": 1.8956344258618827}, 3)
    assert value == pytest.approx(0.9478172129309414)
    with pytest.raises(ValueError):
        score_value({"type": "score"}, 11)


def test_the_three_answer_kinds_of_a_real_response_become_one_listing_each():
    record = {"id": "r1", "request": {"state": "A cup of hot coffee sits on the edge of a table.", "questions": {
        "falls": {"type": "noul", "instructions": "Will the cup fall?"},
        "action": {"type": "choice", "instructions": "What next?", "criteria": {"move_cup": "Move it.", "ignore": "Leave it."}},
        "risk": {"type": "score", "instructions": "How bad?", "criteria": ["harmless", "minor mess", "burn risk"]}}}}
    payload = {"model": "inception/mercury-decide:free", "state": record["request"]["state"],
               "questions": {"falls": record["request"]["questions"]["falls"], "action": record["request"]["questions"]["action"],
                             "risk": record["request"]["questions"]["risk"]}}
    prediction, problems = to_prediction(record, payload, REAL_SAMPLE, model="inception/mercury-decide:free", levels=3, request_hash="abc")
    assert problems == []
    q = prediction["questions"]
    assert q["falls"]["answer"] == {"p_yes": 0.9840936056349263} and q["falls"]["listings"][0]["options"] is None
    assert q["action"]["answer"]["probabilities"] == REAL_SAMPLE["answers"]["action"]["probabilities"]
    assert q["action"]["listings"][0]["options"] == ["move_cup", "ignore"]
    assert q["risk"]["answer"]["value"] == pytest.approx(1.8956344258618827 / 2)
    assert q["risk"]["listings"][0]["values"] == [0.0, 0.5, 1.0]
    assert all(v["n_listings"] == 1 and len(v["listings"]) == 1 for v in q.values())
    assert prediction["api"]["model"] == "inception/mercury-decide-20260930" and prediction["api"]["provider"] == "Inception"
    assert prediction["api"]["usage"] == {"input_tokens": 96, "output_tokens": 3, "cost": 0}
    assert prediction["format"] == "openrouter-decisions" and prediction["orders"] == "stored"


def test_choice_probabilities_that_do_not_sum_to_one_are_renormalized():
    record = _records()[3]
    payload = api_request(record, "m")
    response = auto_answer(payload)
    choice = next(n for n, q in payload["questions"].items() if q["type"] == "choice")
    keys = list(payload["questions"][choice]["criteria"])
    response["answers"][choice]["probabilities"] = {k: (0.5 if i == 0 else 0.0) for i, k in enumerate(keys)} | {keys[-1]: 0.49}
    prediction, problems = to_prediction(record, payload, response, model="m", levels=11)
    answer = prediction["questions"][choice]["answer"]
    assert not problems and sum(answer["probabilities"].values()) == pytest.approx(1.0) and answer["raw_sum"] == pytest.approx(0.99)


def test_a_200_without_answers_is_not_cached(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    with FakeServer([(200, {"error": {"code": 502, "message": "Provider returned error"}}, {}, 0)]) as server:
        out = run_requests("m", _jobs(_records()[:1]), client=_client(server.url), cache=cache, ledger=Ledger(None, max_total_usd=1.0, max_model_usd=1.0),
                           price={}, concurrency=1)
    assert out["problems"] and not out["predictions"] and not list((tmp_path / "cache").rglob("*.json"))


def test_a_choice_without_probabilities_is_one_hot_and_bad_answers_are_reported():
    record = _records()[3]
    payload = api_request(record, "m")
    response = auto_answer(payload)
    choice = next(n for n, q in payload["questions"].items() if q["type"] == "choice")
    noul = next(n for n, q in payload["questions"].items() if q["type"] == "noul")
    key = list(payload["questions"][choice]["criteria"])[-1]
    response["answers"][choice] = {"type": "choice", "choice": key}
    response["answers"][noul] = {"type": "noul", "noul": 1.7}
    prediction, problems = to_prediction(record, payload, response, model="m", levels=11)
    assert prediction["questions"][choice]["answer"]["probabilities"][key] == 1.0
    assert noul not in prediction["questions"] and any(p.startswith(noul) for p in problems)


# --------------------------------------------------------------------------
# HTTP: 재시도·시간 초과·치명 오류·키
# --------------------------------------------------------------------------


def test_429_is_retried_with_exponential_backoff_and_retry_after():
    slept = []
    with FakeServer([(429, {"error": {"code": 429, "message": "Rate limit exceeded"}}, {"Retry-After": "3"}, 0),
                     (503, {"error": {"code": 503, "message": "busy"}}, {}, 0)]) as server:
        client = _client(server.url, sleep=slept.append, backoff_base=0.5)
        response, info = client.post(api_request(_records()[0], "vendor/model"))
    assert info["attempts"] == 3 and info["statuses"] == [429, 503, 200] and info["uncertain_attempts"] == 0
    assert slept == [3.0, 1.0]  # Retry-After(3 s)가 backoff(0.5 s)보다 길면 그것을, 그다음은 0.5 × 2
    assert set(response["answers"]) == set(_records()[0]["request"]["questions"])
    headers = server.requests[0]["headers"]
    assert headers["Authorization"] == f"Bearer {KEY}" and headers["User-Agent"].startswith("robo-jev-bench") and headers["Content-Type"] == "application/json"


def test_a_timeout_is_retried_and_counted_as_possibly_billed(tmp_path):
    with FakeServer([(200, None, {}, 1.0)]) as server:
        client = _client(server.url, timeout=0.3)
        ledger = Ledger(tmp_path / "ledger.json", max_total_usd=1.0, max_model_usd=1.0)
        cache = ResponseCache(tmp_path / "cache")
        jobs = _jobs(_records()[:1], estimate=1e-3)
        out = run_requests("vendor/model", jobs, client=client, cache=cache, ledger=ledger, price={"prompt": 1e-7, "completion": 0.0}, concurrency=1)
    assert len(out["predictions"]) == 1 and not out["errors"]
    account = ledger.snapshot()["models"]["vendor/model"]
    assert account["calls"] == 1 and account["uncertain_usd"] == pytest.approx(2e-3)  # 시간 초과한 시도 하나 × 예약(견적 × 2)


def test_retries_give_up_and_fatal_statuses_stop_the_run(tmp_path):
    with FakeServer([(502, {"error": {"code": 502, "message": "Provider returned error"}}, {}, 0)] * 3) as server:
        with pytest.raises(DecisionsError, match="gave up after 3 attempts"):
            _client(server.url, max_attempts=3).post(api_request(_records()[0], "m"))
    with FakeServer([(402, {"error": {"code": 402, "message": "Insufficient credits"}}, {}, 0)]) as server:
        ledger = Ledger(None, max_total_usd=1.0, max_model_usd=1.0)
        out = run_requests("m", _jobs(_records()), client=_client(server.url), cache=ResponseCache(tmp_path / "c"), ledger=ledger,
                           price={}, concurrency=1)
        assert out["stats"]["fatal"] and "402" in out["stats"]["fatal"]
        assert len(server.requests) == 1 and len(out["skipped"]) == len(_records()) - 1  # 첫 치명 오류 뒤로는 보내지 않는다


def test_the_key_never_appears_in_errors_logs_ledger_or_cache(tmp_path):
    def echo(headers, body):  # 머리를 그대로 되돌려 주는 (나쁜) 서버
        return {"error": {"code": 400, "message": f"bad request; you sent {headers.get('Authorization')}"}}

    logs = []
    with FakeServer([(400, echo, {}, 0)]) as server:
        client = _client(server.url)
        with pytest.raises(DecisionsError) as caught:
            client.post(api_request(_records()[0], "m"))
        assert KEY not in str(caught.value) and "***" in str(caught.value)
        server.script = [(400, echo, {}, 0)]
        ledger = Ledger(tmp_path / "ledger.json", max_total_usd=1.0, max_model_usd=1.0)
        out = run_requests("m", _jobs(_records()[:3]), client=client, cache=ResponseCache(tmp_path / "cache"), ledger=ledger, price={},
                           concurrency=1, calls_log=tmp_path / "calls.jsonl", log=logs.append)
    assert len(out["errors"]) == 1 and len(out["predictions"]) == 2
    assert KEY not in repr(client) and KEY not in str(client.__dict__)
    texts = [json.dumps(out, default=str), "\n".join(logs)]
    texts += [p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file()]
    assert all(KEY not in text for text in texts)


def test_load_api_key_reads_only_a_private_file(tmp_path):
    path = tmp_path / "openrouter.env"
    path.write_text(f"# OpenRouter\nexport OPENROUTER_API_KEY='{KEY}'\n", encoding="utf-8")
    os.chmod(path, 0o644)
    with pytest.raises(UnsafeKeyFile):
        load_api_key(path)
    os.chmod(path, 0o600)
    key = load_api_key(path)
    assert key.reveal() == KEY and repr(key) == "Secret(***)" and f"{key}" == "***"


# --------------------------------------------------------------------------
# 캐시·상한
# --------------------------------------------------------------------------


def test_a_cache_hit_makes_no_call(tmp_path):
    records = _records()
    cache = ResponseCache(tmp_path / "cache")
    with FakeServer() as server:
        client = _client(server.url)
        first = run_requests("m", _jobs(records), client=client, cache=cache, ledger=Ledger(None, max_total_usd=1.0, max_model_usd=1.0),
                             price={}, concurrency=4)
        calls = len(server.requests)
        ledger = Ledger(None, max_total_usd=0.0, max_model_usd=0.0)  # 상한 0이어도 캐시는 돈이 들지 않는다
        second = run_requests("m", _jobs(records), client=client, cache=cache, ledger=ledger, price={}, concurrency=4)
    assert calls == len(records) and len(server.requests) == calls  # 두 번째는 호출 0
    assert second["stats"]["cached"] == len(records) and second["stats"]["sent"] == 0 and ledger.spent() == 0.0
    strip = lambda p: {**p, "api": {**p["api"], "cached": None}}  # noqa: E731
    assert {k: strip(v) for k, v in first["predictions"].items()} == {k: strip(v) for k, v in second["predictions"].items()}


def test_the_cap_stops_before_it_would_be_exceeded(tmp_path):
    records = _records()
    jobs = _jobs(records, estimate=1e-3)  # 예약 = 견적 × 2 = 0.002
    ledger = Ledger(tmp_path / "ledger.json", max_total_usd=0.0065, max_model_usd=1.0)
    with FakeServer(cost=0.002) as server:  # 요청마다 실제 0.002
        out = run_requests("m", jobs, client=_client(server.url), cache=ResponseCache(tmp_path / "cache"), ledger=ledger,
                           price={"prompt": 1e-7}, concurrency=1)
    sent = len(server.requests)
    # 0.002 × 3 = 0.006 ≤ 0.0065, 넷째는 0.006 + 예약 0.002 = 0.008 > 0.0065 — 보내기 전에 멈추고 나머지도 보내지 않는다
    assert sent == 3 and out["stats"]["budget"] and list(out["skipped"].values()).count("budget") == 1
    assert list(out["skipped"].values()).count("stopped") == len(records) - 4
    assert ledger.spent() <= 0.0065 and ledger.spent() == pytest.approx(0.006)
    saved = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert saved["totals"]["calls"] == 3 and saved["models"]["m"]["cost_from_usage"] == 3
    again = Ledger(tmp_path / "ledger.json", max_total_usd=0.0065, max_model_usd=1.0)  # 장부는 누적이다
    assert again.spent() == pytest.approx(ledger.spent())
    with pytest.raises(BudgetExceeded) as caught:
        again.reserve("m", 1e-3)
    assert caught.value.scope == "total"
    per_model = Ledger(None, max_total_usd=1.0, max_model_usd=0.003)
    per_model.reserve("a", 1e-3)
    with pytest.raises(BudgetExceeded, match="per-model") as caught:
        per_model.reserve("a", 1e-3)
    assert caught.value.scope == "model"
    per_model.reserve("b", 1e-3)  # 다른 모델은 제 몫이 있다
    with pytest.raises(ValueError):
        Ledger(None, max_total_usd=None, max_model_usd=1.0)


def test_missing_usage_cost_is_computed_from_the_listed_price():
    ledger = Ledger(None, max_total_usd=1.0, max_model_usd=1.0)
    reserved = ledger.reserve("m", 1e-4)
    cost = ledger.settle("m", reserved, 1e-4, cost=None, usage={"input_tokens": 1000, "output_tokens": 5}, price={"prompt": 4.2e-8, "completion": 0.0})
    assert cost == pytest.approx(4.2e-5) and ledger.snapshot()["models"]["m"]["cost_computed"] == 1


# --------------------------------------------------------------------------
# 견적·모델 목록
# --------------------------------------------------------------------------


def test_the_estimate_counts_the_state_once_or_per_question():
    payload = api_request(_records()[0], "m")
    est = estimate_request(payload, overhead={"per_question_low": 10, "per_question_high": 5})
    n = len(payload["questions"])
    assert est["questions"] == n and est["low"] < est["high"]
    assert est["high"] - est["low"] == (n - 1) * est["state_tokens"] - 5 * n
    assert est["state_tokens"] == -(-len(payload["state"]) // 4)  # tokenizer가 없으면 글자 / 4


def test_model_selection_skips_free_copies_and_rejects_unknown_ids():
    catalog = {"data": [{"id": "a/x", "pricing": {"prompt": "0.0000001"}}, {"id": "a/x:free", "pricing": {"prompt": "0"}}, {"id": "~a/latest"}]}
    assert [m["id"] for m in select_models(catalog, None)] == ["a/x", "~a/latest"]
    assert [m["id"] for m in select_models(catalog, ["~a/latest", "a/x"])] == ["~a/latest", "a/x"]
    with pytest.raises(ValueError, match="free"):
        select_models(catalog, ["a/x:free"])
    with pytest.raises(ValueError, match="b/y"):
        select_models(catalog, ["b/y"])
