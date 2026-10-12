"""Running the narration (Task v0.3a): jobs, OpenRouter calls under a hard cap, cache, request log, checks, back-check and assembly.

The HTTP layer, retries, the response cache and the spend ledger are the bench client's (:mod:`robo_jev.bench.openrouter`):
:class:`~robo_jev.bench.openrouter.DecisionsClient` posts any JSON body to a URL (here the chat-completions endpoint),
:class:`~robo_jev.bench.openrouter.ResponseCache` keeps one response per (model, job id, request sha256) so a rerun costs nothing,
and :class:`~robo_jev.bench.openrouter.Ledger` reserves a conservative amount **before** each request and refuses it if the
cumulative total (every model, every run) would pass the cap. Before reserving, :meth:`~robo_jev.simgen.narration.SealedIndex.guard`
refuses any request that carries sealed text. Every sent request is appended to the request log with the hashes the audit needs.

**Pipeline** (:func:`run_pipeline`): per moment-voice, the rewrite call → the deterministic checks
(:func:`evaluate`) → for a rewrite that passes, an independent back-check by a second model (:data:`JUDGE_SYSTEM`) that lists the
fact differences with a severity. A rewrite is kept when the checks pass and the back-check finds no major difference and at most
one minor one (:func:`judge_accepts`).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from robo_jev.bench.openrouter import BudgetExceeded, DecisionsClient, DecisionsError, Ledger, ResponseCache
from robo_jev.simgen import narration as N
from robo_jev.simgen import narration_checks as checks

__all__ = ["JUDGE_SYSTEM", "Job", "Sender", "evaluate", "judge_accepts", "judge_job", "make_job", "parse_judge", "run_jobs", "run_pipeline"]

DEFAULT_MAX_TOKENS = 4096
#: Characters per token for the reservation estimate (conservative: Korean is denser).
CHARS_PER_TOKEN = {"en": 3.0, "ko": 1.4}
#: The back-check accepts at most this many minor differences (setting, task wording, a side remark) and no major one.
MAX_MINOR = 1


@dataclass
class Job:
    """One request: a plan entry's source record in one voice (or a back-check of one rewrite)."""

    job_id: str
    entry: dict[str, Any]
    source: dict[str, Any]
    voice: N.Voice | None
    payload: dict[str, Any]
    sha: str
    estimate_usd: float
    texts: list[str] = field(default_factory=list)


def _estimate(payload: dict[str, Any], language: str, out_chars: float, price: dict[str, float]) -> float:
    cpt = CHARS_PER_TOKEN.get(language, 2.0)
    in_tokens = sum(len(m["content"]) for m in payload["messages"]) / cpt
    out_tokens = min(float(payload.get("max_tokens", DEFAULT_MAX_TOKENS)), out_chars / cpt)
    return in_tokens * price.get("prompt", 0.0) + out_tokens * price.get("completion", 0.0)


def make_job(entry: dict[str, Any], source: dict[str, Any], voice_key: str, model: str, price: dict[str, float], *,
             max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.9) -> Job:
    language = entry["language"]
    voice = N.VOICES[language][voice_key]
    hint = N.ORDER_HINTS[int(entry["order"][voice_key])]
    messages = N.build_messages(source, voice, hint)
    seed = int(hashlib.sha256(f"{source['id']}|{voice_key}".encode()).hexdigest()[:8], 16)
    payload = N.chat_payload(model, messages, max_tokens=max_tokens, temperature=temperature, seed=seed)
    state = source["request"]["state"]
    questions = source["request"]["questions"]
    out_chars = len(state) * voice.bounds[1] + (len(json.dumps(questions, ensure_ascii=False)) if N.revoices_questions(voice) else 0)
    texts = [state, messages[0]["content"], voice.description]
    for q in questions.values():
        texts.append(q["instructions"])
        criteria = q.get("criteria")
        if isinstance(criteria, dict):
            texts += [str(v) for v in criteria.values()]
        elif isinstance(criteria, list):
            texts += [str(v) for v in criteria]
    return Job(N.narrated_id(source["id"], voice_key), entry, source, voice, payload, N.payload_sha256(payload),
               _estimate(payload, language, out_chars, price), texts)


def _content(response: dict[str, Any]) -> tuple[str, str | None]:
    choices = response.get("choices") or []
    if not choices:
        return "", None
    message = choices[0].get("message") or {}
    return str(message.get("content") or ""), choices[0].get("finish_reason")


def _usage(response: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    cost = usage.get("cost")
    cost = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
    mapped = {"input_tokens": int(usage.get("prompt_tokens") or 0), "output_tokens": int(usage.get("completion_tokens") or 0)}
    details = usage.get("completion_tokens_details") or {}
    if isinstance(details, dict) and details.get("reasoning_tokens"):
        mapped["reasoning_tokens"] = int(details["reasoning_tokens"])
    return cost, mapped


class Sender:
    """Sends jobs of one model: cache first; sealed guard; ledger reservation; send; settle; cache; log. Thread-safe."""

    def __init__(self, model: str, *, client: DecisionsClient | None, cache: ResponseCache, ledger: Ledger, price: dict[str, float],
                 index: N.SealedIndex, calls_log: Path, stop: threading.Event, lock: threading.Lock):
        self.model, self.client, self.cache, self.ledger, self.price = model, client, cache, ledger, price
        self.index, self.calls_log, self.stop, self.lock = index, calls_log, stop, lock
        self.stats: dict[str, Any] = {"cached": 0, "sent": 0, "cost_usd": 0.0, "budget": None, "fatal": None, "refused_sealed": 0}
        self.errors: dict[str, str] = {}
        self.skipped: dict[str, str] = {}

    def _note(self, entry: dict[str, Any]) -> None:
        with self.lock:
            self.calls_log.parent.mkdir(parents=True, exist_ok=True)
            with self.calls_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def send(self, job: Job) -> dict[str, Any] | None:
        """The response for a job (cached or fresh), or None (not sent: stopped, budget, sealed, error)."""
        model, ledger = self.model, self.ledger
        hit = self.cache.get(model, job.job_id, job.sha)
        if hit is not None:
            with self.lock:
                self.stats["cached"] += 1
            return hit["response"]
        if self.stop.is_set():
            with self.lock:
                self.skipped[job.job_id] = "stopped"
            return None
        if self.client is None:
            with self.lock:
                self.skipped[job.job_id] = "not cached (cache-only run)"
            return None
        try:
            self.index.guard(source=job.source, texts=job.texts)
        except N.SealedTextError as error:
            with self.lock:
                self.skipped[job.job_id] = str(error)
                self.stats["refused_sealed"] += 1
            return None
        try:
            reserved = ledger.reserve(model, job.estimate_usd)
        except BudgetExceeded as error:
            self.stop.set()
            with self.lock:
                self.skipped[job.job_id] = "budget"
                self.stats["budget"] = str(error)
            return None
        sent = {"source_request_sha256": N._request_hash(job.source), **N.sent_text_hashes(self.index, job.texts)}
        base = {"time": time.time(), "model": model, "job_id": job.job_id, "source_id": job.source["id"],
                "voice": job.voice.key if job.voice else None, "rank": job.entry.get("rank"), "request_sha256": job.sha, "sent": sent}
        try:
            response, info = self.client.post(job.payload)
        except DecisionsError as error:
            if error.uncertain:
                ledger.settle(model, reserved, job.estimate_usd, cost=None, usage=None, price=self.price, uncertain=True)
            else:
                ledger.release(model, reserved)
            with self.lock:
                ledger.save()
            self._note({**base, "error": str(error), "status": error.status, "uncertain": error.uncertain})
            with self.lock:
                self.errors[job.job_id] = str(error)
                if error.fatal:
                    self.stats["fatal"] = str(error)
            if error.fatal:
                self.stop.set()
            return None
        cost, usage = _usage(response)
        actual = ledger.settle(model, reserved, job.estimate_usd, cost=cost, usage=usage, price=self.price)
        if info["uncertain_attempts"]:
            ledger.add_uncertain(model, reserved * int(info["uncertain_attempts"]))
        with self.lock:
            ledger.save()
        content, finish = _content(response)
        meta = {"resolved_model": response.get("model"), "provider": response.get("provider"), "generation_id": response.get("id"),
                "usage": usage, "cost_usd": actual, "attempts": info["attempts"], "seconds": info["seconds"], "finish_reason": finish}
        self._note({**base, **meta})
        with self.lock:
            self.stats["sent"] += 1
            self.stats["cost_usd"] += actual
        if not content:
            with self.lock:
                self.errors[job.job_id] = f"empty reply (finish {finish})"
            return None
        self.cache.put(model, job.job_id, job.sha, response, meta=meta)
        return response


def run_jobs(
    model: str, jobs: list[Job], *, client: DecisionsClient | None, cache: ResponseCache, ledger: Ledger, price: dict[str, float],
    index: N.SealedIndex, calls_log: Path, concurrency: int = 16, log: Callable[[str], None] | None = None,
    stop: threading.Event | None = None, progress_every: int = 200,
) -> dict[str, Any]:
    """Send every job of one model that is not cached → ``{"responses": {job id: response}, "errors", "skipped", "stats"}``. A job
    whose reservation would pass the cap is not sent, and the run stops (``stats.budget``). ``client=None`` reads the cache only."""
    stop = stop or threading.Event()
    lock = threading.Lock()
    sender = Sender(model, client=client, cache=cache, ledger=ledger, price=price, index=index, calls_log=calls_log, stop=stop, lock=lock)
    responses: dict[str, Any] = {}
    done = [0]
    started = time.time()

    def one(job: Job) -> None:
        response = sender.send(job)
        with lock:
            if response is not None:
                responses[job.job_id] = response
            done[0] += 1
            if log and progress_every and done[0] % progress_every == 0:
                s = sender.stats
                log(f"{model}: {done[0]}/{len(jobs)} ({s['cached']} cached, {s['sent']} sent, ${s['cost_usd']:.4f}, "
                    f"{len(sender.errors)} errors, ledger ${ledger.spent():.4f}, {time.time() - started:.0f} s)")

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        list(pool.map(one, jobs))
    if log:
        s = sender.stats
        log(f"{model}: {len(responses)}/{len(jobs)} replies ({s['cached']} cached, {s['sent']} sent, ${s['cost_usd']:.4f}); "
            f"{len(sender.errors)} errors, {len(sender.skipped)} skipped; ledger total ${ledger.spent():.4f}")
    return {"responses": responses, "errors": sender.errors, "skipped": sender.skipped, "stats": sender.stats}


def evaluate(job: Job, response: dict[str, Any]) -> dict[str, Any]:
    """Parse and check one reply → ``{"passed", "state", "questions", "question_status", "checks", "problems", "finish_reason"}``."""
    content, finish = _content(response)
    voice = job.voice
    language = voice.language
    state, questions, problems = N.parse_reply(content, voice)
    result: dict[str, Any] = {"passed": False, "state": state, "questions": None, "question_status": {}, "checks": {},
                              "problems": problems, "finish_reason": finish}
    if state is None:
        result["checks"] = {"parse": problems}
        return result
    original = job.source["request"]["state"]
    result["checks"] = checks.check_state(original, state, language, voice.bounds, copy_limit=None if voice.json_state else checks.MAX_COPY_SHARE)
    if finish == "length":
        result["checks"]["truncated"] = ["reply cut at max_tokens"]
    result["passed"] = not any(result["checks"].values())
    status: dict[str, str] = {}
    if N.revoices_questions(voice):
        for name, q in job.source["request"]["questions"].items():
            if not isinstance(questions, dict) or name not in questions:
                status[name] = "original (missing)"
                continue
            bad = checks.check_question(q, questions[name], language)
            status[name] = "revoiced" if not bad else "original (" + "; ".join(bad[:3]) + ")"
        result["questions"] = questions
    else:
        status = {name: "original (voice keeps the person)" for name in job.source["request"]["questions"]}
    result["question_status"] = status
    return result


# --------------------------------------------------------------------------
# back-check (a second model compares the facts)
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """You check a rewrite of a robot-perception text for fact errors. ORIGINAL and REWRITE must state exactly the same facts.

Never report these (they are allowed): different wording or synonyms, a different order, layout or format (prose, JSON, table, \
bullets, dialogue), a different person (I / you / the robot / the unit), radio words, field labels, articles, repeated, merged or \
split sentences, a fact moved elsewhere in the text.

Report these:
- a fact of ORIGINAL missing from REWRITE (including the setting, the robot's identity and task, side remarks, the planner's \
recommendation and the list of possible actions);
- a fact stated differently: a number or unit; which person or object a fact belongs to; a side or direction (left/right, \
ahead/behind, toward/away, above/below, short of/past); whether something is seen now or was last seen or is hidden; what lies in \
which direction; a sensor age; a tolerance; a fill level; a lid; which sides are clear, blocked or hidden; a required stop or limit;
- a fact, prediction, cause, advice or judgement that REWRITE adds.

Severity: "major" if it could change a physical judgement about the moment (motion, positions, distances, sides, timing, sensor \
readings or ages, visibility, the load, fill, lid, surfaces, people's or vehicles' cues, constraints, the recommendation or the \
actions); "minor" otherwise (the setting or time of day, task wording, a side remark).

Return only one JSON object: {"differences": [{"severity": "major" or "minor", "what": "<short description>"}]}. Return \
{"differences": []} when the facts match. Do not list anything that matches."""


def judge_job(job_id: str, original: str, rewrite: str, model: str, price: dict[str, float], language: str, *,
              entry: dict[str, Any], source: dict[str, Any]) -> Job:
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"ORIGINAL:\n<<<\n{original}\n>>>\n\nREWRITE:\n<<<\n{rewrite}\n>>>"}]
    payload = {"model": model, "messages": messages, "max_tokens": 800, "temperature": 0.0, "response_format": {"type": "json_object"},
               "reasoning": {"enabled": False}, "usage": {"include": True}, "provider": {"data_collection": "deny"}}
    texts = [original, rewrite, JUDGE_SYSTEM]
    return Job(job_id, entry, source, None, payload, N.payload_sha256(payload), _estimate(payload, language, 600, price), texts)


#: A listed "difference" whose own text clears it ("… so no error", "no change") is not counted.
_SELF_CLEARED = re.compile(
    r"\bno (?:error|change|issue|fact change|factual change|substantive change|difference|omission|additional mismatch|mismatch)\b"
    r"|\bnot a fact (?:change|loss|difference)\b|\bnothing (?:is )?missing\b|\bnot missing\b|\b(?:fact|meaning) (?:is )?preserved\b"
    r"|변화 없|차이 없|문제 없|오류 없|변화는 없|차이는 없",
    re.IGNORECASE,
)


def parse_judge(response: dict[str, Any]) -> dict[str, Any]:
    """The back-check reply → ``{"ok": bool | None, "major": n, "minor": n, "differences": [...]}`` (``ok`` None = unreadable)."""
    content, _ = _content(response)
    try:
        data = N._json_object(content, last=True)
    except ValueError:
        return {"ok": None, "error": "judge reply is not JSON", "major": 0, "minor": 0, "differences": []}
    items = data.get("differences") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {"ok": None, "error": "judge reply has no differences list", "major": 0, "minor": 0, "differences": []}
    diffs = []
    for item in items:
        text = str(item.get("what", "") if isinstance(item, dict) else item)
        if _SELF_CLEARED.search(text):  # the judge says itself that this is not an error
            continue
        if isinstance(item, dict):
            severity = "minor" if str(item.get("severity", "major")).lower().startswith("minor") else "major"
            diffs.append({"severity": severity, "what": str(item.get("what", ""))[:300]})
        elif str(item).strip():
            diffs.append({"severity": "major", "what": str(item)[:300]})
    major = sum(d["severity"] == "major" for d in diffs)
    minor = len(diffs) - major
    return {"ok": major == 0 and minor <= MAX_MINOR, "major": major, "minor": minor, "differences": diffs}


def judge_accepts(verdict: dict[str, Any] | None) -> bool:
    return bool(verdict and verdict.get("ok"))


def judge_id(job_id: str) -> str:
    return f"judge~{job_id}"


def run_pipeline(
    rewrite_model: str, judge_model: str, jobs: list[Job], *, client: DecisionsClient | None, cache: ResponseCache, ledger: Ledger,
    prices: dict[str, dict[str, float]], index: N.SealedIndex, calls_log: Path, concurrency: int = 32,
    log: Callable[[str], None] | None = None, stop: threading.Event | None = None, progress_every: int = 500,
) -> dict[str, Any]:
    """Rewrite → checks → back-check, per job, in one pool and one ledger. → ``{"results": {job id: {"result", "verdict"}},
    "stats": {...}}``. Stops at the cap (no reservation passes it) or on a fatal error."""
    stop = stop or threading.Event()
    lock = threading.Lock()
    common = dict(client=client, cache=cache, ledger=ledger, index=index, calls_log=calls_log, stop=stop, lock=lock)
    rewriter = Sender(rewrite_model, price=prices[rewrite_model], **common)
    judge = Sender(judge_model, price=prices[judge_model], **common)
    results: dict[str, Any] = {}
    counts = {"done": 0, "replies": 0, "checks_passed": 0, "judged": 0, "accepted": 0}
    started = time.time()

    def one(job: Job) -> None:
        response = rewriter.send(job)
        result, verdict = None, None
        if response is not None:
            result = evaluate(job, response)
            if result["passed"]:
                jjob = judge_job(judge_id(job.job_id), job.source["request"]["state"], result["state"], judge_model,
                                 prices[judge_model], job.voice.language, entry=job.entry, source=job.source)
                jresponse = judge.send(jjob)
                if jresponse is not None:
                    verdict = parse_judge(jresponse)
        with lock:
            counts["done"] += 1
            if result is not None:
                counts["replies"] += 1
                counts["checks_passed"] += bool(result["passed"])
                results[job.job_id] = {"result": result, "verdict": verdict}
            if verdict is not None:
                counts["judged"] += 1
                counts["accepted"] += judge_accepts(verdict)
            if log and progress_every and counts["done"] % progress_every == 0:
                log(f"{counts['done']}/{len(jobs)}: {counts['replies']} replies, {counts['checks_passed']} pass checks, "
                    f"{counts['accepted']}/{counts['judged']} pass the back-check; rewrite ${rewriter.stats['cost_usd']:.4f}, "
                    f"judge ${judge.stats['cost_usd']:.4f}, ledger ${ledger.spent():.4f}, {time.time() - started:.0f} s")

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        list(pool.map(one, jobs))
    stats = {"counts": counts, "rewrite": rewriter.stats, "judge": judge.stats, "errors": {**rewriter.errors, **judge.errors},
             "skipped": len(rewriter.skipped) + len(judge.skipped), "budget": rewriter.stats["budget"] or judge.stats["budget"],
             "fatal": rewriter.stats["fatal"] or judge.stats["fatal"]}
    if log:
        log(f"done {counts['done']}/{len(jobs)}: {counts['checks_passed']}/{counts['replies']} pass checks, {counts['accepted']}/"
            f"{counts['judged']} pass the back-check; rewrite ${rewriter.stats['cost_usd']:.4f}, judge ${judge.stats['cost_usd']:.4f}; "
            f"ledger ${ledger.spent():.4f}; budget stop: {stats['budget']}")
    return {"results": results, "stats": stats}
