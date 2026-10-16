"""Run a published robo-jev model — Decisions-API-shaped answers from the same prompts and readout the model was trained and scored with.

robo-jev does not generate text. For every question of a request it builds one fixed prompt (prompt version v0c-p3, chat format,
``direct`` mode — :mod:`robo_jev.decisions.prompt`), runs one forward pass, and reads the probabilities of the shown answer codes
(``A``, ``B``, …) at the answer position: the last hidden state (after the final RMSNorm) projected on the LM-head rows of those codes
in fp32, then a softmax over them (:func:`robo_jev.decisions.zero_shot.score_code_prompts`). Nothing here re-implements that — this
module only loads the published weights and turns the readout into answers.

* :meth:`RoboJev.from_pretrained` — a Hub repo id or a local export folder. The merged repo (``Steerix-org/robo-jev-2b``) loads with
  ``transformers``; the adapter repo (``Steerix-org/robo-jev-2b-lora``) loads its base (``Qwen/Qwen3.5-2B`` at the pinned revision in
  ``adapter_config.json``) and applies the LoRA adapter with ``peft``. Both carry ``robo_jev.json`` (prompt version and digest, answer
  codes, tokenizer hash, training summary); loading refuses a folder whose prompt or tokenizer does not match this package.
* :meth:`RoboJev.decide` — ``(state, questions)`` in the OpenRouter Decisions API shape → ``{name: answer}``:
  choice ``{"type": "choice", "choice", "probabilities"}``, noul ``{"type": "noul", "noul": p}``, score
  ``{"type": "score", "score": value in [0, 1], "probabilities": {anchor: p}}`` (anchors ``"0.0"`` … ``"1.0"``).
* **Limits**: state ≤ 16,000 characters, 1–12 questions, 2–26 options per choice question (the answer letters A–Z). Since the v0.7
  release (Task rel07) a served request is validated under the many-option opt-in of :func:`robo_jev.decisions.records.option_range_for`
  — the records' own default stays 2–8 unless a record opts in. The Hub runner ``robo_jev_decide.py`` has the same limits, except that
  it takes up to 466 options (two-letter codes past Z, which training never showed; Task rel071); up to 26 both answer byte-identically.
* **Orders.** ``orders="single"`` (default — the deployed condition, one forward pass per question) shows each question's options in
  one seeded order. ``orders="balanced"`` averages over the counterbalanced listings (every choice option in every position once; yes/no
  and score scales both ways) — less position bias, n× the cost. The order is seeded by ``key`` (default: a hash of the state text, so
  the same request always gets the same answer) and ``seed`` (default 1, the scoring tools' default). With the same request, key, seed
  and order, the answers are byte-identical to ``scripts/diagnose_decisions.py`` (test: tests/test_infer.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robo_jev.decisions.metrics import average_answers
from robo_jev.decisions.prompt import PROMPT_VERSION, answer_from_codes, build_prompts, check_displayed_lines, fixed_text_digest
from robo_jev.decisions.records import OPTION_RANGE_MARK, SCHEMA, WIDE_OPTION_RANGE, validate_record

__all__ = [
    "CHOICE_OPTIONS",
    "DEFAULT_SEED",
    "FORMAT",
    "METADATA_FILE",
    "MODE",
    "ORDERS",
    "RoboJev",
    "default_key",
    "file_sha256",
    "to_decision",
]

#: The prompt format and mode the model was trained and scored with (training manifest ``prompt``).
FORMAT = "chat"
MODE = "direct"
#: Public order names → the prompt module's listing schemes.
ORDERS = {"single": "seeded", "balanced": "counterbalanced"}
#: The scoring tools' default option-order seed (``scripts/diagnose_decisions.py --shuffle-seed``).
DEFAULT_SEED = 1
#: The export's metadata file (prompt, answer codes, tokenizer hash, training summary).
METADATA_FILE = "robo_jev.json"
#: Choice options a served request may have (module docstring) — the records' many-option opt-in range.
CHOICE_OPTIONS = WIDE_OPTION_RANGE


def default_key(state: str) -> str:
    """The option-order key of a request without one: the first 16 hex digits of sha256(state) — the same request always gets the same
    order, and different moments get different orders (as the scored records did with their ids)."""
    return hashlib.sha256(str(state).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_decision(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Our answer (:func:`robo_jev.decisions.prompt.answer_from_codes`, averaged) → the Decisions API shape.

    choice: probabilities in the request's option order and the most likely option (the first one on a tie); noul: ``p_yes``;
    score: the expected anchor value and the probability of every anchor, ascending."""
    kind = question["type"]
    if kind == "choice":
        probabilities = {key: float(answer["probabilities"][key]) for key in question["criteria"]}
        best = max(probabilities, key=lambda key: probabilities[key])
        return {"type": "choice", "choice": best, "probabilities": probabilities}
    if kind == "noul":
        return {"type": "noul", "noul": float(answer["p_yes"])}
    return {"type": "score", "score": float(answer["value"]), "probabilities": {name: float(p) for name, p in answer["bins"].items()}}


def _resolve(repo_id_or_path: str | Path, *, revision: str | None, token: str | bool | None, cache_dir: str | Path | None) -> Path:
    local = Path(repo_id_or_path)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(repo_id_or_path), revision=revision, token=token, cache_dir=cache_dir))


def _check_metadata(metadata: dict[str, Any], folder: Path) -> None:
    prompt = metadata.get("prompt") or {}
    if prompt.get("version") != PROMPT_VERSION or prompt.get("digest") != fixed_text_digest(PROMPT_VERSION):
        raise ValueError(
            f"{folder}: the model expects prompt {prompt.get('version')}/{str(prompt.get('digest'))[:12]}, this package builds "
            f"{PROMPT_VERSION}/{fixed_text_digest(PROMPT_VERSION)[:12]} — use the robo-jev version the model was exported with"
        )
    if prompt.get("format") != FORMAT or prompt.get("mode") != MODE:
        raise ValueError(f"{folder}: prompt format/mode {prompt.get('format')}/{prompt.get('mode')} is not {FORMAT}/{MODE}")


class RoboJev:
    """A loaded robo-jev model (module docstring). Build it with :meth:`from_pretrained`, or wrap an already-loaded
    ``Qwen3_5ForCausalLM`` (merged, or with the LoRA layers injected) and a ``tokenizers.Tokenizer``."""

    def __init__(self, model: Any, tokenizer: Any, *, metadata: dict[str, Any] | None = None, name: str = "robo-jev") -> None:
        from robo_jev.model.backbone_qwen import QwenBackbone

        model.eval()
        model.requires_grad_(False)
        self.backbone = QwenBackbone(model, model_id=name)
        self.tokenizer = tokenizer
        self.metadata = dict(metadata or {})
        self.name = name

    @property
    def model(self) -> Any:
        return self.backbone.model

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str | Path,
        *,
        device: str = "cuda",
        dtype: Any = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        base_model: str | Path | None = None,
        base_revision: str | None = None,
    ) -> RoboJev:
        """Load the merged repo, or the adapter repo on its base (module docstring).

        `dtype` defaults to bfloat16 (the scored condition). `revision`/`token`/`cache_dir` apply to the robo-jev repo. For the adapter
        repo, `base_model` overrides the base (e.g. a local copy of Qwen/Qwen3.5-2B) and `base_revision` its revision; by default the base
        is ``adapter_config.json``'s ``base_model_name_or_path`` at its pinned ``revision``."""
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if dtype is None else dtype
        folder = _resolve(repo_id_or_path, revision=revision, token=token, cache_dir=cache_dir)
        meta_path = folder / METADATA_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"{folder}: no {METADATA_FILE} — not a robo-jev export")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_metadata(metadata, folder)
        tokenizer_file = folder / "tokenizer.json"
        expected = (metadata.get("tokenizer") or {}).get("sha256")
        actual = file_sha256(tokenizer_file)
        if expected != actual:
            raise ValueError(f"{tokenizer_file}: sha256 {actual[:12]}… is not the training tokenizer's {str(expected)[:12]}…")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        adapter_config = folder / "adapter_config.json"
        if adapter_config.is_file():
            from peft import PeftModel

            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_id = str(base_model) if base_model is not None else str(config["base_model_name_or_path"])
            base_rev = base_revision if base_revision is not None else (None if base_model is not None else config.get("revision"))
            base = AutoModelForCausalLM.from_pretrained(base_id, revision=base_rev, dtype=dtype, cache_dir=cache_dir)
            model = PeftModel.from_pretrained(base, str(folder)).get_base_model()
            name = f"{metadata.get('name', 'robo-jev')} (adapter on {config['base_model_name_or_path']}@{str(config.get('revision'))[:7]})"
        else:
            model = AutoModelForCausalLM.from_pretrained(str(folder), dtype=dtype)
            name = f"{metadata.get('name', 'robo-jev')} (merged)"
        model.to(device)
        return cls(model, tokenizer, metadata=metadata, name=name)

    def score_request(
        self, state: str, questions: dict[str, Any], *, orders: str = "single", key: str | None = None, seed: int | None = DEFAULT_SEED
    ) -> dict[str, Any]:
        """Score every question → ``{name: {"type", "answer"(listing average, our answer form), "listings": [{"options", "codes",
        "code_probs", "answer"}, …]}}`` in request order. The listings are the scoring tools' listings for the same key, seed and order."""
        from robo_jev.decisions.zero_shot import score_code_prompts

        if orders not in ORDERS:
            raise ValueError(f"orders: one of {sorted(ORDERS)} (got {orders!r})")
        request = {"state": state, "questions": questions}
        served = {"schema": SCHEMA, "id": "request", "request": request, "meta": {"option_range": dict(OPTION_RANGE_MARK)}}  # 2..26 options
        problems = validate_record(served, require_labels=False)
        if problems:
            raise ValueError(f"not a valid decision request: {problems[:5]}")
        prompts = build_prompts(
            request, self.tokenizer, key=default_key(state) if key is None else str(key), seed=seed, fmt=FORMAT, mode=MODE,
            orders=ORDERS[orders], version=PROMPT_VERSION,
        )
        shown = [problem for p in prompts for problem in check_displayed_lines(p)]
        if shown:
            raise ValueError(f"prompt option lines do not match the recorded options: {shown[:3]}")
        scored = score_code_prompts(self.backbone, prompts, batch=1, code_mass=False)
        out: dict[str, Any] = {}
        for p, s in zip(prompts, scored):
            entry = out.setdefault(p["question"], {"type": p["type"], "listings": []})
            entry["listings"].append(
                {"options": p["options"], "codes": p["codes"], "code_probs": s["code_probs"], "answer": answer_from_codes(p, s["code_probs"])}
            )
        for entry in out.values():
            entry["answer"] = average_answers(entry["type"], [listing["answer"] for listing in entry["listings"]])
        return out

    def decide(
        self, state: str, questions: dict[str, Any], *, orders: str = "single", key: str | None = None, seed: int | None = DEFAULT_SEED
    ) -> dict[str, dict[str, Any]]:
        """``(state, questions)`` → ``{name: answer}`` in the Decisions API shape (module docstring)."""
        scored = self.score_request(state, questions, orders=orders, key=key, seed=seed)
        return {name: to_decision(questions[name], entry["answer"]) for name, entry in scored.items()}
