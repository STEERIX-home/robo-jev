#!/usr/bin/env python
"""robo_jev_decide.py — run robo-jev with nothing but PyTorch and Hugging Face ``transformers`` (+ ``peft`` for the adapter repo).

robo-jev is a decision model for robots: it reads a short text of what a robot senses and answers decision questions with calibrated
probabilities. It does not generate text. For each question it builds one fixed prompt (prompt version v0c-p3), runs one forward pass
and reads the probabilities of the shown answer codes at the answer position. This file is a self-contained copy of the prompt
builder and the readout of the robo-jev package (``robo_jev.infer.RoboJev``), tested to give byte-identical answers (up to 26 options;
the package stops there).

Command line (answers print as JSON)::

    python robo_jev_decide.py --model Steerix-org/robo-jev-2b --request request.json
    python robo_jev_decide.py --model Steerix-org/robo-jev-2b-lora --request request.json --orders balanced

``request.json`` follows the OpenRouter Decisions API: ``{"state": "...", "questions": {"name": {"type": "noul" | "choice" | "score",
"instructions": "...", "criteria": ...}}}`` — choice criteria map option keys to descriptions (2–466 options, shown as A–Z, then the
two-letter codes AA, AB, … of ``CODES``); score criteria is a list of strings saying what 0 and 1 mean; noul (yes/no) has no criteria. A
JSONL file of such requests (``--requests``) prints one line each. Limits (``LIMITS``): state ≤ 16,000 characters, 1–12 questions, 2–466
options per choice question; a request outside them is refused whole (``ValueError``).

Python::

    from robo_jev_decide import load
    model = load("Steerix-org/robo-jev-2b")           # or a local folder; the -lora repo loads Qwen/Qwen3.5-2B + the adapter
    answers = model.decide(state, questions)           # {"name": {"type": "noul", "noul": 0.83}, ...}

Answers: choice ``{"type": "choice", "choice": key, "probabilities": {key: p}}``, noul ``{"type": "noul", "noul": p_yes}``, score
``{"type": "score", "score": value in [0, 1], "probabilities": {"0.0": p, …, "1.0": p}}``.

``orders="single"`` (default) shows each question's options in one seeded order — one forward pass per question. ``orders="balanced"``
averages over counterbalanced orders (each choice option in each position once; yes/no and score scales both ways): less position
bias, n× the cost. The order depends on ``key`` (default: the first 16 hex digits of sha256(state)) and ``seed`` (default 1).

Speed and exactness: the scored runs used the ``flash-linear-attention`` kernels for Qwen3.5's linear-attention layers
(``pip install flash-linear-attention``); without them ``transformers`` falls back to a slower PyTorch path whose results can differ
in the last bits. Long states and long option lists cost more time; the limits above follow the Decisions API, except that a choice
question may have up to 466 options (one answer code each).

Many options: the model was trained with up to 26 options, shown as the letters A–Z. Options past the 26th get two-letter codes (AA, AB,
…), which were not part of training, so answers with more than 26 options are less reliable. In the default single order the model puts
nearly all of its probability on the options shown with A–Z: an option shown with a two-letter code is all but never chosen, even when
it is the only fitting one. ``orders="balanced"`` shows every option with a letter in 26 of its n listings, at the cost of n forward
passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from typing import Any

__all__ = ["LIMITS", "RoboJevRunner", "build_prompts", "default_key", "load"]

PROMPT_VERSION = "v0c-p3"
SYSTEM_TEXT = (
    "You are the judgment module of a robot. You read what the robot senses right now and answer one question about "
    "the next few seconds. Judge from the physical situation: distances, speeds, braking, timing, and what the robot "
    "is carrying. Answer with the letter of exactly one option."
)
SCORE_ANCHOR_TEXT = "Pick the value closest to your estimate."
ANSWER_CUE = "Answer:"
SCORE_ANCHORS = tuple(round(i / 10, 1) for i in range(11))
NOUL_LINES = (("yes", "Yes"), ("no", "No"))
#: The answer codes, in the order the options are shown. First the capital letters A–Z, the codes the model was trained with. Then the
#: two-letter codes AA … ZZ in lexicographic order that pass the answer-code rule with the training tokenizer — one token alone (" AB")
#: and the last token of "Answer: AB" (544 of the 676) — less :data:`CODE_STOPLIST`: 440 more, 466 codes in all. Hard-coded, so the codes
#: never depend on the tokenizer at hand; :func:`_verify_codes` checks every code a prompt shows. The two-letter codes were not part of
#: training: answers with more than 26 options are less reliable.
CODES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + tuple("""
    AA AB AE AF AG AJ AK AL AO AP AQ AR AU AV AZ
    BA BB BD BF BG BH BI BJ BK BL BM BN BO BP BR BT BU BV BW BX
    CA CB CC CE CF CG CH CI CJ CK CL CN CP CR CS CT CU CV CW CX CY CZ
    DA DD DE DF DG DH DI DK DL DM DN DP DS DT DU DV DW DX DY
    EA EB EC ED EE EF EI EJ EK EL EM EN EO EP EQ ER ES ET EW EX EZ
    FA FB FC FD FE FF FG FH FI FK FL FM FN FO FP FR FS FW FX FY
    GA GC GD GE GF GG GH GI GK GL GM GN GP GR GS GT GU GV GW GX
    HB HC HD HF HG HH HK HL HN HS HT HU HV HW HY
    IA IB IC IG IH II IK IL IM IO IU IV IW IX
    JA JB JC JD JE JJ JK JL JM JO JP JS JT JU JV JW
    KA KC KD KE KF KH KI KK KL KO KP KR KS KT KU KV KY
    LC LD LE LF LG LH LI LJ LL LM LN LO LP LR LS LT LU LV LW LX LY LZ
    MC MF MH MI MJ MK MN MO MP MQ MT MU MV MW MX
    NA NB NC ND NE NF NG NH NI NJ NK NL NN NP NR NS NT NU NV NW NX NZ
    OA OB OC OD OE OG OL OM OO OP OT OU OV
    PB PD PE PF PG PI PJ PK PL PN PO PP PQ PT PU PV PW PX PY
    QB QC QE QL QQ QR QS QT QU
    RA RB RC RD RF RG RH RI RJ RK RL RM RN RO RP RR RS RT RU RV RW RX
    SA SB SC SD SE SF SG SH SI SJ SK SL SM SN SP SQ SR SS SU SV SW SX SY SZ
    TA TC TD TE TF TG TH TI TJ TK TL TM TN TP TR TS TT TU TW TX TY TZ
    UA UB UC UD UE UF UL UR UT UU UW
    VA VB VC VD VE VF VG VH VI VK VL VM VN VO VT VW VX
    WA WB WC WD WF WG WH WI WK WL WM WN WO WP WR WS WT WV WW WX
    XB XC XD XF XI XK XL XM XO XP XR XS XT XU XV XX XY
    YM YT YY
    ZA ZE ZO ZX ZZ
""".split())
#: Two-letter codes that pass the rule but are left out, so that no option label reads as a word or a term: common English two-letter
#: words and interjections (first row), and very common abbreviations, titles and units (the rest).
CODE_STOPLIST = tuple("""
    AD AH AM AN AS AT AW AX AY BE BY DO EH GO HA HE HI HM HO IF IN IS IT MA ME MY NO OF OH OK ON OR OW PA SO TO UM UP US WE YA YE YO
    AC AI BC BS CD CM CO DB DC DJ DR EG EU EV FT FU GB HP HQ HR ID IE IP IQ IR JR KB KG KM KN KW LA LB MB MD MG ML MM MR MS NM NY
    OS PC PH PM PR PS QA RE ST TB TV UI UK UN UV UX VP VR VS
""".split())
MAX_STATE_CHARS = 16_000
MAX_QUESTIONS = 12
#: Choice options per question: at least 2, at most one per answer code (466).
MIN_OPTIONS, MAX_OPTIONS = 2, len(CODES)
#: The request limits (module docstring) — a request outside them is refused whole.
LIMITS = {"state_chars": MAX_STATE_CHARS, "questions": (1, MAX_QUESTIONS), "choice_options": (MIN_OPTIONS, MAX_OPTIONS)}
ORDERS = ("single", "balanced")
DEFAULT_SEED = 1
_NAME = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_CHAT_SYSTEM = "<|im_start|>system\n"
_CHAT_USER = "<|im_start|>user\n"
_CHAT_END = "<|im_end|>\n"
_CHAT_ASSISTANT = "<|im_start|>assistant\n"
_THINK_EMPTY = "<think>\n\n</think>\n\n"


def default_key(state: str) -> str:
    """Option-order key of a request: the first 16 hex digits of sha256(state)."""
    return hashlib.sha256(str(state).encode("utf-8")).hexdigest()[:16]


def _rng(key: str, question: str, seed: int, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{key}|{question}|{seed}|{salt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _option_order(key: str, question: str, n: int, seed: int | None) -> list[int]:
    order = list(range(n))
    if seed is not None:
        _rng(key, question, int(seed), "order").shuffle(order)
    return order


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def _displayed(name: str, question: dict, *, key: str, seed: int | None) -> dict[str, Any]:
    kind = question.get("type")
    if kind == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"choice question {name!r}: criteria must map 2 or more option keys to descriptions")
        keys = list(criteria) if seed is None else sorted(criteria)
        keys = [keys[i] for i in _option_order(key, name, len(keys), seed)]
        lines = [f"{k}: {_one_line(criteria[k])}" if _one_line(criteria[k]) else str(k) for k in keys]
        return {"options": keys, "lines": lines, "values": None}
    if kind == "noul":
        pairs = [NOUL_LINES[i] for i in _option_order(key, name, len(NOUL_LINES), seed)]
        return {"options": [k for k, _ in pairs], "lines": [line for _, line in pairs], "values": None}
    if kind == "score":
        index = list(range(len(SCORE_ANCHORS)))
        if seed is not None and _rng(key, name, int(seed), "direction").random() < 0.5:
            index.reverse()
        names = [f"{v:.1f}" for v in SCORE_ANCHORS]
        return {"options": [names[i] for i in index], "lines": [names[i] for i in index], "values": [float(SCORE_ANCHORS[i]) for i in index]}
    raise ValueError(f"question {name!r}: unknown type {kind!r} (choice, noul, score)")


def _listings(name: str, question: dict, *, key: str, seed: int | None, orders: str) -> list[dict[str, Any]]:
    base = _displayed(name, question, key=key, seed=seed)
    if orders == "single":
        return [base]
    if orders != "balanced":
        raise ValueError(f"orders: one of {list(ORDERS)} (got {orders!r})")
    fields = ("options", "lines", "values")
    if question.get("type") == "choice":
        n = len(base["options"])
        return [{f: (base[f][s:] + base[f][:s]) if base[f] is not None else None for f in fields} for s in range(n)]
    return [base, {f: list(reversed(base[f])) if base[f] is not None else None for f in fields}]


def _question_block(question: dict, codes: list[str], lines: list[str]) -> str:
    out = ["Question: " + _one_line(question.get("instructions", ""))]
    if question.get("type") == "score":
        criteria = [_one_line(item) for item in (question.get("criteria") or []) if _one_line(item)]
        if criteria:
            out.append("Scale:")
            out.extend(f"- {item}" for item in criteria)
        out.append(SCORE_ANCHOR_TEXT)
    out.append("Options:")
    out.extend(f"{code}. {line}" for code, line in zip(codes, lines))
    return "\n".join(out)


def _verify_codes(tokenizer: Any, count: int) -> tuple[list[str], list[int]]:
    """The first `count` answer codes, each one token both alone (" A") and after the answer cue ("Answer: A")."""
    if count > len(CODES):
        raise ValueError(f"{count} options — more than {len(CODES)} codes")
    codes, ids = [], []
    for code in CODES[:count]:
        alone = tokenizer.encode(" " + code, add_special_tokens=False).ids
        boundary = tokenizer.encode("Answer: " + code, add_special_tokens=False).ids
        if len(alone) != 1 or boundary[-1] != alone[0]:
            raise ValueError(f"code {code!r} is not one token at the answer boundary")
        codes.append(code)
        ids.append(int(alone[0]))
    return codes, ids


def _check_request(state: Any, questions: Any) -> None:
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be a non-empty string")
    if len(state) > MAX_STATE_CHARS:
        raise ValueError(f"state has {len(state)} characters (max {MAX_STATE_CHARS:,})")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"questions must map 1..{MAX_QUESTIONS} names to questions")
    for name, question in questions.items():
        if not _NAME.match(str(name)):
            raise ValueError(f"question name {name!r} must match {_NAME.pattern}")
        if not isinstance(question, dict) or question.get("type") not in ("choice", "noul", "score"):
            raise ValueError(f"question {name!r}: type must be choice, noul or score")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError(f"question {name!r}: instructions must be a non-empty string")
        if question["type"] == "choice":
            criteria = question.get("criteria")
            if not (isinstance(criteria, dict) and MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS):
                count = len(criteria) if isinstance(criteria, dict) else type(criteria).__name__
                raise ValueError(f"question {name!r}: choice criteria must map {MIN_OPTIONS}..{MAX_OPTIONS} option keys to descriptions "
                                 f"(got {count})")
            bad = [key for key in criteria if not _NAME.match(str(key))]
            if bad:
                raise ValueError(f"question {name!r}: option keys {bad} must match {_NAME.pattern}")
        if question["type"] == "score" and not isinstance(question.get("criteria", []), list):
            raise ValueError(f"question {name!r}: score criteria must be a list of strings")


def build_prompts(state: str, questions: dict, tokenizer: Any, *, key: str | None = None, seed: int | None = DEFAULT_SEED,
                  orders: str = "single") -> list[dict[str, Any]]:
    """One prompt per question and listing: ``{"question", "type", "listing", "options", "values", "codes", "code_ids", "tokens", "text"}``."""
    _check_request(state, questions)
    key = default_key(state) if key is None else str(key)
    prefix = _CHAT_SYSTEM + SYSTEM_TEXT + _CHAT_END + _CHAT_USER + "Situation:\n" + state.strip() + "\n\n"
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False).ids)
    out = []
    for name, question in questions.items():
        for index, shown in enumerate(_listings(str(name), question, key=key, seed=seed, orders=orders)):
            codes, code_ids = _verify_codes(tokenizer, len(shown["options"]))
            suffix = _question_block(question, codes, shown["lines"]) + _CHAT_END + _CHAT_ASSISTANT + _THINK_EMPTY + ANSWER_CUE
            out.append({
                "question": str(name), "type": question["type"], "listing": index, "options": list(shown["options"]),
                "values": shown["values"], "codes": codes, "code_ids": code_ids,
                "tokens": prefix_ids + list(tokenizer.encode(suffix, add_special_tokens=False).ids), "text": prefix + suffix,
            })
    return out


def _listing_answer(prompt: dict[str, Any], probs: list[float]) -> dict[str, Any]:
    if prompt["type"] == "choice":
        return {"probabilities": dict(zip(prompt["options"], probs))}
    if prompt["type"] == "noul":
        return {"p_yes": probs[prompt["options"].index("yes")]}
    value = sum(p * v for p, v in zip(probs, prompt["values"]))
    return {"value": value, "bins": {name: p for name, p in sorted(zip(prompt["options"], probs))}}


def _average(kind: str, answers: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(answers)
    if kind == "choice":
        keys = list(answers[0]["probabilities"])
        return {"probabilities": {k: sum(float(a["probabilities"][k]) for a in answers) / n for k in keys}}
    if kind == "noul":
        return {"p_yes": sum(float(a["p_yes"]) for a in answers) / n}
    return {"value": sum(float(a["value"]) for a in answers) / n,
            "bins": {b: sum(float(a["bins"][b]) for a in answers) / n for b in answers[0]["bins"]}}


def _decision(question: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    if question["type"] == "choice":
        probabilities = {k: float(answer["probabilities"][k]) for k in question["criteria"]}
        return {"type": "choice", "choice": max(probabilities, key=lambda k: probabilities[k]), "probabilities": probabilities}
    if question["type"] == "noul":
        return {"type": "noul", "noul": float(answer["p_yes"])}
    return {"type": "score", "score": float(answer["value"]), "probabilities": {k: float(p) for k, p in answer["bins"].items()}}


class RoboJevRunner:
    """A loaded model (a ``transformers`` Qwen3.5 causal LM — merged, or with the LoRA adapter applied) and its ``tokenizers`` tokenizer."""

    def __init__(self, model: Any, tokenizer: Any, metadata: dict[str, Any] | None = None) -> None:
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = dict(metadata or {})

    def code_probs(self, prompt: dict[str, Any]) -> list[float]:
        """One forward pass; softmax over the LM-head rows of the shown codes (fp32) at the last position's hidden state."""
        import torch

        with torch.no_grad():
            device = self.model.model.embed_tokens.weight.device
            tokens = prompt["tokens"]
            ids = torch.tensor([tokens], dtype=torch.long)
            positions = torch.arange(len(tokens))[None]
            hidden = self.model.model(input_ids=ids.to(device), position_ids=positions.to(device), use_cache=False).last_hidden_state
            last = torch.stack([hidden[0, len(tokens) - 1]]).float()
            head = self.model.lm_head.weight
            logits = head[torch.tensor(prompt["code_ids"], device=device)].float() @ last[0]
            return torch.softmax(logits, 0).cpu().tolist()

    def decide(self, state: str, questions: dict, *, orders: str = "single", key: str | None = None,
               seed: int | None = DEFAULT_SEED) -> dict[str, dict[str, Any]]:
        """``(state, questions)`` → ``{name: answer}`` in the Decisions API shape (module docstring)."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for prompt in build_prompts(state, questions, self.tokenizer, key=key, seed=seed, orders=orders):
            grouped.setdefault(prompt["question"], []).append(_listing_answer(prompt, self.code_probs(prompt)))
        return {name: _decision(questions[name], _average(questions[name]["type"], answers)) for name, answers in grouped.items()}


def load(name_or_path: str, *, device: str | None = None, dtype: Any = None, revision: str | None = None, token: str | None = None,
         base_model: str | None = None) -> RoboJevRunner:
    """Load ``Steerix-org/robo-jev-2b`` (merged) or ``Steerix-org/robo-jev-2b-lora`` (adapter on Qwen/Qwen3.5-2B at its pinned
    revision; ``base_model`` overrides the base, e.g. a local copy), or a local folder of either. dtype defaults to bfloat16."""
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if dtype is None else dtype
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(name_or_path):
        folder = name_or_path
    else:
        from huggingface_hub import snapshot_download

        folder = snapshot_download(name_or_path, revision=revision, token=token)
    metadata = {}
    if os.path.isfile(os.path.join(folder, "robo_jev.json")):
        with open(os.path.join(folder, "robo_jev.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
        version = (metadata.get("prompt") or {}).get("version")
        if version != PROMPT_VERSION:
            raise ValueError(f"{folder}: the model expects prompt version {version}, this script builds {PROMPT_VERSION}")
    tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    if os.path.isfile(os.path.join(folder, "adapter_config.json")):
        from peft import PeftModel

        with open(os.path.join(folder, "adapter_config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        base = AutoModelForCausalLM.from_pretrained(
            base_model or config["base_model_name_or_path"], revision=None if base_model else config.get("revision"), dtype=dtype
        )
        model = PeftModel.from_pretrained(base, folder).get_base_model()
    else:
        model = AutoModelForCausalLM.from_pretrained(folder, dtype=dtype)
    model.to(device)
    return RoboJevRunner(model, tokenizer, metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="robo-jev: Decisions-API answers from a robo-jev model (prints JSON)")
    parser.add_argument("--model", default="Steerix-org/robo-jev-2b", help="Hub repo id or local folder (merged or -lora)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", help="a JSON file with {state, questions}")
    source.add_argument("--requests", help="a JSONL file, one {state, questions} (or {id, request: {state, questions}}) per line")
    parser.add_argument("--orders", choices=ORDERS, default="single")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="option-order seed")
    parser.add_argument("--key-field", default=None, help="(--requests) use this field of each line as the option-order key")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--base-model", default=None, help="(-lora repo) base model id or local folder instead of the pinned Qwen/Qwen3.5-2B")
    args = parser.parse_args(argv)
    import torch

    runner = load(args.model, device=args.device, dtype=getattr(torch, args.dtype), revision=args.revision, base_model=args.base_model)
    if args.request:
        with open(args.request, encoding="utf-8") as handle:
            request = json.load(handle)
        request = request.get("request", request)
        print(json.dumps(runner.decide(request["state"], request["questions"], orders=args.orders, seed=args.seed), ensure_ascii=False, indent=1))
        return 0
    with open(args.requests, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            request = item.get("request", item)
            key = str(item[args.key_field]) if args.key_field else None
            answers = runner.decide(request["state"], request["questions"], orders=args.orders, seed=args.seed, key=key)
            print(json.dumps({"id": item.get("id"), "answers": answers}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
