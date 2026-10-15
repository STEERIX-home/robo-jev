"""Training checkpoint → Hugging Face folders: the PEFT adapter and the merged model (Task hf0 §1; the v0.7 release: Task rel07).

**Adapter.** :func:`lora_config` rebuilds the training LoRA (r, alpha, dropout, targets, no bias) as a ``peft`` ``LoraConfig`` with
``task_type="CAUSAL_LM"`` and the pinned base (``Qwen/Qwen3.5-2B`` @ revision). :func:`build_peft_model` wraps the real base model with
it and :func:`load_adapter_state` copies the checkpoint's tensors in by an **explicit key map** (:func:`adapter_key_map` — training keys
are relative to the text model, ``layers.0.linear_attn.out_proj.lora_A.default.weight``; PEFT's are ``base_model.model.model.`` + that);
any missing, unexpected or misshapen key raises. :func:`save_adapter` writes it with ``save_pretrained`` (safetensors) in the
checkpoint's own dtype (bf16 — the values the scored model used) and re-reads the file to check every tensor bit for bit.

**Merged.** :func:`save_merged` runs ``merge_and_unload()`` on a float32 copy of the base (W + scale·B·A in fp32, one rounding to bf16
at the end) and saves the bf16 text model (``Qwen3_5ForCausalLM`` — the vision tower and MTP head of the base are not part of the
model and are not saved) with ``save_pretrained(safe_serialization=True)``.

**Next to the weights** (:func:`write_common_files`): the training tokenizer (``tokenizer.json`` byte for byte — sha256 checked), the
base's ``tokenizer_config.json`` and ``chat_template.jinja`` (pinned revision), ``robo_jev.json`` (:func:`build_metadata` — prompt
version and digest, answer-code scheme and token ids, score anchors, readout, request limits, the release, training summary, source
checkpoint sha256, git sha, base revision), the standalone runner ``robo_jev_decide.py``, the model card and figures.
:func:`check_no_pickle` refuses pickle files.

**Public text only.** :func:`public_paths` turns absolute paths into a local checkout (the data manifests record them) into repo-relative
ones before they go into ``robo_jev.json``, and :func:`check_public_text` scans every text file of a folder for local paths and anything
that looks like a key or token — the export, ``cards`` and the upload refuse a folder with a hit.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from robo_jev.checkpoint import load_model_checkpoint
from robo_jev.decisions.prompt import (
    ANSWER_CUE,
    NOUL_LINES,
    PROMPT_VERSION,
    SCORE_ANCHOR_TEXT,
    SCORE_ANCHORS,
    SYSTEM_TEXT,
    fixed_text_digest,
)
from robo_jev.decisions.records import MAX_QUESTIONS, MAX_STATE_CHARS
from robo_jev.infer import CHOICE_OPTIONS, DEFAULT_SEED, FORMAT, METADATA_FILE, MODE, file_sha256

__all__ = [
    "ADAPTER_REPO",
    "BASE_ID",
    "BASE_REVISION",
    "FORBIDDEN_SUFFIXES",
    "LIMITS",
    "MERGED_REPO",
    "PEFT_PREFIX",
    "RELEASE",
    "STANDALONE",
    "TAG",
    "adapter_key_map",
    "build_metadata",
    "build_peft_model",
    "CARD_FIELDS",
    "check_no_pickle",
    "check_public_text",
    "lora_config",
    "load_adapter_state",
    "public_paths",
    "read_training_checkpoint",
    "render_card",
    "save_adapter",
    "save_merged",
    "write_common_files",
]

BASE_ID = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MERGED_REPO = "Steerix-org/robo-jev-2b"
ADAPTER_REPO = "Steerix-org/robo-jev-2b-lora"
#: The release this export makes — the Hub tag, ``robo_jev.json``'s ``version`` and the model card's title.
TAG = "v0.7"
#: What ``robo_jev.json`` records about the release (Task rel07 §0; the user chose v0.7 seed 0 on 2026-10-16).
RELEASE = {
    "tag": TAG,
    "previous": "v0.1",
    "model": "v0.7, training seed 0 (run v07-s0-2b, final checkpoint)",
    "pre_declared_rule": "not passed: S REJECTED; guards O, N and F fail",
    "why_released": "clearly better than the public v0.1; released so that robojev bench requests with up to 26 options can be served",
}
#: The request limits of both runners (the package's :class:`robo_jev.infer.RoboJev` and the Hub's ``robo_jev_decide.py``).
LIMITS = {"state_chars": MAX_STATE_CHARS, "questions": [1, MAX_QUESTIONS], "choice_options": list(CHOICE_OPTIONS)}
GITHUB = "https://github.com/STEERIX-home/robo-jev"
#: PEFT parameter name = this prefix + the training key (training keys are relative to the text model, ``Qwen3_5ForCausalLM.model``).
PEFT_PREFIX = "base_model.model.model."
#: File endings that are (or may be) pickles — never in an export.
FORBIDDEN_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib", ".npy", ".npz")
STANDALONE = Path(__file__).with_name("robo_jev_decide.py")
_TRAINING_KEY = re.compile(r"^layers\.\d+\.(self_attn|linear_attn|mlp)\.[a-z_]+\.lora_[AB]\.default\.weight$")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_training_checkpoint(path: str | Path, *, base_revision: str = BASE_REVISION) -> dict[str, Any]:
    """The checkpoint (``robo-jev-checkpoint-v0`` or its model-only form) with the checks the export relies on: a ``decisions-train-v0``
    LoRA run on ``Qwen/Qwen3.5-2B`` at the pinned revision, trained with this package's prompt version and fixed-text digest."""
    state = load_model_checkpoint(path, mmap=True)
    manifest = state.get("manifest") or {}
    _need(manifest.get("kind") == "decisions-train-v0", f"{path}: not a decisions-train-v0 checkpoint ({manifest.get('kind')!r})")
    model = manifest.get("model") or {}
    _need(model.get("mode") == "lora", f"{path}: mode {model.get('mode')!r} — only LoRA runs export to an adapter")
    base = model.get("base") or {}
    _need(base.get("model_id") == BASE_ID and base.get("revision") == base_revision, f"{path}: base {base.get('model_id')}@{base.get('revision')}")
    prompt = manifest.get("prompt") or {}
    _need(
        prompt.get("version") == PROMPT_VERSION and prompt.get("digest") == fixed_text_digest(PROMPT_VERSION)
        and prompt.get("format") == FORMAT and prompt.get("mode") == MODE,
        f"{path}: prompt {prompt} does not match this package ({PROMPT_VERSION}, {FORMAT}, {MODE})",
    )
    return state


def lora_config(lora: dict[str, Any], *, base_id: str = BASE_ID, revision: str = BASE_REVISION) -> Any:
    """The training LoRA (:func:`robo_jev.train.attach_lora`) as a ``peft.LoraConfig`` for a causal LM on the pinned base."""
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM", r=int(lora["r"]), lora_alpha=float(lora["alpha"]), lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["targets"]), bias="none", base_model_name_or_path=base_id, revision=revision,
    )


def adapter_key_map(keys: list[str] | Any) -> dict[str, str]:
    """Training key → PEFT parameter name (``base_model.model.model.`` + key). Every key must look like a text-model LoRA weight."""
    out = {}
    for key in keys:
        _need(bool(_TRAINING_KEY.match(str(key))), f"unexpected checkpoint key {key!r} (expected layers.N.<block>.<proj>.lora_A|B.default.weight)")
        out[str(key)] = PEFT_PREFIX + str(key)
    return out


def build_peft_model(base: Any, lora: dict[str, Any], *, base_id: str = BASE_ID, revision: str = BASE_REVISION) -> Any:
    """Wrap a loaded base ``Qwen3_5ForCausalLM`` with the training LoRA (adapter parameters in float32, PEFT's autocast)."""
    from peft import get_peft_model

    base.name_or_path = base_id  # PEFT records the model's name_or_path as base_model_name_or_path — keep the Hub id, not a local path
    return get_peft_model(base, lora_config(lora, base_id=base_id, revision=revision))


def load_adapter_state(peft_model: Any, saved: dict[str, torch.Tensor]) -> list[str]:
    """Copy the checkpoint's LoRA tensors into `peft_model` by the explicit map; raise on any missing, unexpected or misshapen key.
    Returns the PEFT parameter names in checkpoint order."""
    mapping = adapter_key_map(list(saved))
    params = {name: p for name, p in peft_model.named_parameters() if "lora_" in name}
    missing = sorted(set(params) - set(mapping.values()))
    unexpected = sorted(set(mapping.values()) - set(params))
    _need(not missing and not unexpected, f"LoRA keys do not match: missing {len(missing)} {missing[:3]}, unexpected {len(unexpected)} {unexpected[:3]}")
    with torch.no_grad():
        for key, name in mapping.items():
            tensor, param = saved[key], params[name]
            _need(tuple(tensor.shape) == tuple(param.shape), f"{key}: shape {tuple(tensor.shape)} vs model {tuple(param.shape)}")
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
    return list(mapping.values())


def save_adapter(peft_model: Any, saved: dict[str, torch.Tensor], out_dir: str | Path) -> dict[str, Any]:
    """``save_pretrained`` the adapter in the checkpoint's dtype (safetensors), then re-read the file: the keys must be PEFT's names for
    the checkpoint keys (adapter name dropped) and every tensor must equal the checkpoint's bit for bit."""
    from safetensors import safe_open

    out_dir = Path(out_dir)
    dtypes = {t.dtype for t in saved.values()}
    _need(len(dtypes) == 1, f"mixed checkpoint dtypes {dtypes}")
    dtype = dtypes.pop()
    lora_params = [p for name, p in peft_model.named_parameters() if "lora_" in name]
    previous = lora_params[0].dtype
    for p in lora_params:
        p.data = p.data.to(dtype)
    try:
        peft_model.save_pretrained(str(out_dir), safe_serialization=True, save_embedding_layers=False)
    finally:
        for p in lora_params:
            p.data = p.data.to(previous)
    file = out_dir / "adapter_model.safetensors"
    expected = {PEFT_PREFIX + key.replace(".default.weight", ".weight"): key for key in saved}
    with safe_open(str(file), framework="pt") as handle:
        keys = set(handle.keys())
        _need(keys == set(expected), f"{file}: keys differ (missing {sorted(set(expected) - keys)[:3]}, extra {sorted(keys - set(expected))[:3]})")
        for name, key in expected.items():
            _need(torch.equal(handle.get_tensor(name), saved[key]), f"{file}: {name} differs from the checkpoint's {key}")
    return {"file": file.name, "tensors": len(expected), "dtype": str(dtype).replace("torch.", ""), "bytes": file.stat().st_size}


def save_merged(peft_model: Any, out_dir: str | Path, *, dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    """``merge_and_unload()`` (in the model's dtype — float32 for the published merge) → cast to `dtype` → ``save_pretrained``."""
    out_dir = Path(out_dir)
    merge_dtype = next(p for name, p in peft_model.named_parameters() if "lora_" not in name).dtype
    merged = peft_model.merge_and_unload()
    _need(not any("lora_" in name for name, _ in merged.named_parameters()), "merge left LoRA parameters behind")
    merged.to(dtype)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    files = sorted(p.name for p in out_dir.glob("*.safetensors"))
    return {"files": files, "merge_dtype": str(merge_dtype).replace("torch.", ""), "dtype": str(dtype).replace("torch.", ""),
            "bytes": sum((out_dir / f).stat().st_size for f in files), "class": type(merged).__name__}


def _git_sha(repo: Path) -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _code_scheme(tokenizer: Any) -> dict[str, Any]:
    from robo_jev.model.zero_shot import CODES, verify_codes

    low, high = CHOICE_OPTIONS
    codes, ids = verify_codes(tokenizer, max(high, len(SCORE_ANCHORS)))  # every code a request can show: A–Z (choice up to 26 options)
    _need(len(set(ids)) == len(ids) == len(CODES) == high, f"answer codes {codes} → ids {ids}: not 26 distinct single tokens")
    return {
        "codes": "one capital letter per shown option, A, B, C, … in the shown order",
        "token_ids": dict(zip(codes, ids)),
        "rule": "each code is one token alone (' A') and right after the answer cue ('Answer: A'); the prompt ends with the cue",
        "readout": (
            "the last position's hidden state (after the final RMSNorm) times the LM-head rows of the shown codes, in float32; "
            "softmax over the shown codes only"
        ),
        "options": {"choice": f"{low}-{high} (the request's keys, shown as '<code>. <key>: <description>', codes A-Z)", "noul": "2 (Yes, No)",
                    "score": f"{len(SCORE_ANCHORS)} anchors"},
    }


def build_metadata(
    state: dict[str, Any], *, variant: str, checkpoint_path: str | Path, checkpoint_sha256: str, tokenizer: Any, tokenizer_sha256: str,
    repo: Path, weights: dict[str, Any], data_manifests: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``robo_jev.json`` — everything the checkpoint knows that a user of the exported weights needs (module docstring)."""
    manifest = state["manifest"]
    config = state.get("config") or {}
    data = dict(manifest.get("data") or {})
    dev_ids = data.pop("dev_ids", None) or []
    data["dev_ids_sha256"] = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest() if dev_ids else None
    lora = (manifest.get("model") or {}).get("lora") or {}
    extra = public_paths(dict(data_manifests or {}))
    return {
        "format": "robo-jev-hf/1",
        "name": f"robo-jev {TAG}",
        "version": TAG,
        "variant": variant,
        "release": dict(RELEASE),
        "limits": {**LIMITS, "refused": "a request outside the limits is refused whole"},
        "repos": {"merged": MERGED_REPO, "adapter": ADAPTER_REPO, "code": GITHUB},
        "base": {
            "id": BASE_ID, "revision": BASE_REVISION, "digest": ((manifest.get("model") or {}).get("base") or {}).get("digest"),
            "dtype": ((manifest.get("model") or {}).get("base") or {}).get("dtype"), "vocab_size": ((manifest.get("model") or {}).get("base") or {}).get("vocab_size"),
            "loaded_as": "AutoModelForCausalLM → Qwen3_5ForCausalLM (the text model; the base's vision tower and MTP head are not used)",
        },
        "weights": weights,
        "lora": {**lora, "scaling": float(lora["alpha"]) / float(lora["r"]), "bias": "none",
                 "trainable_parameters": (manifest.get("model") or {}).get("trainable_parameters"),
                 "scope": "Qwen3_5ForCausalLM.model (the text model; LM head = tied embedding, frozen)"},
        "tokenizer": {
            "file": "tokenizer.json", "sha256": tokenizer_sha256, "training": manifest.get("tokenizer"),
            "note": (
                "the exact tokenizer the model was trained and scored with (Qwen/Qwen3.8-27B tokenizer.json): the same vocabulary, merges "
                "and normalisation as Qwen/Qwen3.5-2B's, plus 7 audio/TTS special tokens (ids 248070-248076) that the base's "
                "tokenizer_config.json already names"
            ),
        },
        "prompt": {
            "version": PROMPT_VERSION, "digest": fixed_text_digest(PROMPT_VERSION), "format": FORMAT, "mode": MODE,
            "system": SYSTEM_TEXT, "answer_cue": ANSWER_CUE, "score_instruction": SCORE_ANCHOR_TEXT,
            "layout": (
                "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\nSituation:\\n{state}\\n\\nQuestion: {instructions}\\n"
                "[Scale:\\n- {criteria}…\\n{score_instruction}\\n]Options:\\n{code}. {option}…<|im_end|>\\n<|im_start|>assistant\\n"
                "<think>\\n\\n</think>\\n\\nAnswer:"
            ),
            "one_prompt_per": "question × listing; the state prefix and the question are tokenized separately and concatenated",
        },
        "answer_codes": _code_scheme(tokenizer),
        "noul_options": [line for _, line in NOUL_LINES],
        "score_anchors": list(SCORE_ANCHORS),
        "score_value": "expected anchor value: sum over anchors of p(anchor) × anchor",
        "orders": {
            "single": "one listing per question: choice options in a seeded permutation of their sorted keys, yes/no and score in a seeded direction",
            "balanced": "every cyclic shift of the seeded choice listing (each option in each position once); yes/no and score both ways; answers averaged",
            "default_seed": DEFAULT_SEED,
            "default_key": "first 16 hex digits of sha256(state)",
            "rng": "random.Random(int.from_bytes(sha256(f'{key}|{question}|{seed}|{salt}')[:8], 'big')), salt 'order' (shuffle) or 'direction'",
        },
        "decisions_api": {
            "choice": {"type": "choice", "choice": "most likely option", "probabilities": "{option: p}"},
            "noul": {"type": "noul", "noul": "p(yes)"},
            "score": {"type": "score", "score": "expected value in [0, 1]", "probabilities": "{anchor: p}"},
        },
        "training": {
            "run_id": state.get("run_id"), "step": state.get("step"), "status": state.get("status"),
            "selection": (manifest.get("selection") or {}).get("evaluated"),
            "objective": manifest.get("objective"),
            "recipe": {k: config.get(k) for k in (
                "mode", "lora_dtype", "epochs", "max_steps", "batch_examples", "lr", "weight_decay", "warmup_ratio", "gradient_clip",
                "questions_per_request", "augment", "type_weights", "code_mass_weight", "seed", "dtype",
            )},
            "precision": "bf16 base and bf16 LoRA compute with float32 master weights; the exported LoRA values are the checkpoint's bf16 model tensors (the values every reported score used)",
            "data": public_paths(data),
            "data_sources": extra,
            "real_robot_data": False,
        },
        "source": {
            "checkpoint": _public_file(checkpoint_path), "checkpoint_sha256": checkpoint_sha256, "checkpoint_format": state.get("format"),
            "training_git": manifest.get("git"),
            "training_git_note": "the cloud training run synced the code without .git, so its manifest records no commit; the data files are pinned by sha256",
            "export_git": _git_sha(repo),
            "exported": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }


def write_common_files(
    out_dir: str | Path, *, metadata: dict[str, Any], tokenizer_json: str | Path, tokenizer_config: str | Path, chat_template: str | Path,
    card: str, figures: dict[str, str | Path] | None = None,
) -> list[str]:
    """Tokenizer files, ``robo_jev.json``, the standalone runner, the model card (``README.md``) and figures into `out_dir`."""
    out_dir = Path(out_dir)
    _need(file_sha256(tokenizer_json) == metadata["tokenizer"]["sha256"], f"{tokenizer_json}: not the training tokenizer")
    shutil.copyfile(tokenizer_json, out_dir / "tokenizer.json")
    shutil.copyfile(tokenizer_config, out_dir / "tokenizer_config.json")
    shutil.copyfile(chat_template, out_dir / "chat_template.jinja")
    shutil.copyfile(STANDALONE, out_dir / STANDALONE.name)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    for name, source in (figures or {}).items():
        target = out_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())


CARD_TEMPLATE = Path(__file__).with_name("cards") / "card.md"
_HF = "https://huggingface.co"
CARD_FIELDS: dict[str, dict[str, str]] = {
    "merged": {
        "TITLE": f"robo-jev 2B — {TAG}", "RELATION": "finetune", "LIBRARY": "transformers", "REPO": MERGED_REPO, "PIP_EXTRA": "", "LOAD_NOTE": "",
        "VARIANT_NOTE": (
            f"This repository holds the **merged weights**: Qwen3.5-2B with the robo-jev {TAG} LoRA adapter merged in, saved in bfloat16 as a "
            "text-only `Qwen3_5ForCausalLM` (the base's vision tower is not part of robo-jev and is left out). The adapter alone is at "
            f"[{ADAPTER_REPO}]({_HF}/{ADAPTER_REPO})."
        ),
        "SIBLING": f"The LoRA adapter for Qwen/Qwen3.5-2B (the same model, 63 MB): [{ADAPTER_REPO}]({_HF}/{ADAPTER_REPO})",
    },
    "adapter": {
        "TITLE": f"robo-jev 2B LoRA — {TAG}", "RELATION": "adapter", "LIBRARY": "peft", "REPO": ADAPTER_REPO, "PIP_EXTRA": "peft",
        "LOAD_NOTE": f" — fetches {BASE_ID} at revision {BASE_REVISION[:7]} and applies the adapter",
        "VARIANT_NOTE": (
            "This repository holds the **LoRA adapter** (r 32, α 64, 31 M parameters, bfloat16) for "
            f"[{BASE_ID}]({_HF}/{BASE_ID}) at revision `{BASE_REVISION[:7]}`. The runners load the base at that pinned revision and apply "
            f"the adapter with `peft`. The merged model is at [{MERGED_REPO}]({_HF}/{MERGED_REPO})."
        ),
        "SIBLING": f"The merged model (no `peft` needed): [{MERGED_REPO}]({_HF}/{MERGED_REPO})",
    },
}


def render_card(variant: str, *, example_output: str = "", many_options_output: str = "") -> str:
    """The model card (``README.md``) of `variant` (``merged`` | ``adapter``) from the shared template; `example_output` and
    `many_options_output` are the real answers of the card's two examples (``scripts/export_hf.py cards --equivalence``)."""
    fields = {**CARD_FIELDS[variant], "EXAMPLE_OUTPUT": example_output.strip(), "MANY_OPTIONS_OUTPUT": many_options_output.strip()}
    text = CARD_TEMPLATE.read_text(encoding="utf-8")
    for name, value in fields.items():
        text = text.replace("{{" + name + "}}", value)
    _need("{{" not in text, f"card {variant}: unfilled placeholders {re.findall(r'{{[A-Z_]+}}', text)}")
    text = re.sub(r'"transformers>=5\.17"  +#', '"transformers>=5.17"   #', text)
    return re.sub(r"\n{3,}", "\n\n", text)


def check_no_pickle(folder: str | Path) -> list[str]:
    """Raise if any file in `folder` is (or may be) a pickle; return the file list."""
    folder = Path(folder)
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    bad = [f for f in files if f.lower().endswith(FORBIDDEN_SUFFIXES)]
    _need(not bad, f"{folder}: pickle-type files {bad}")
    from safetensors import safe_open

    for f in files:  # a pickle can hide behind any name: protocol 2+ pickles start with 0x80, torch's zip archives with PK
        path = folder / f
        if f.endswith(".safetensors"):
            with safe_open(str(path), framework="pt") as handle:  # parses the header; raises on anything else
                handle.keys()
            continue
        with path.open("rb") as stream:
            head = stream.read(2)
        _need(not head.startswith(b"\x80") and head != b"PK", f"{path}: looks like a pickle or zip archive")
    return files


#: An absolute path into a local checkout of this repo — the main one or a worktree under ``.claude/worktrees/`` — up to the repo root
#: (the data manifests record the paths they were built at). Not preceded by ``:``, ``/`` or a word character, so URLs never match.
_LOCAL_CHECKOUT = re.compile(r"(?<![\w:/.])/(?:[^/\s\"'<>|]+/)*?robo-?jev/(?:\.claude/worktrees/[^/\s\"'<>|]+/)?")
#: What must never be in a published text file: local paths and anything shaped like a key or a token.
PRIVATE_TEXT = {
    "local path": re.compile(r"(?<![\w:/.])/(?:home|root|Users|tmp|mnt|srv|workspace)/[^\s\"'<>|]*|\.claude/|worktrees/|Desktop/projects"),
    "user name": re.compile(r"rodex", re.IGNORECASE),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenRouter key": re.compile(r"sk-or-[A-Za-z0-9-]{16,}"),
    "API key": re.compile(r"\bsk-[A-Za-z0-9]{20,}|\brpa_[A-Za-z0-9]{16,}|(?:HF_TOKEN|OPENROUTER_API_KEY|RUNPOD_API_KEY)\s*="),
}


def public_paths(value: Any) -> Any:
    """`value` with every absolute path into a local checkout of this repo made repo-relative (dicts and lists recursively)."""
    if isinstance(value, str):
        return _LOCAL_CHECKOUT.sub("", value)
    if isinstance(value, dict):
        return {key: public_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_paths(item) for item in value]
    return value


def _public_file(path: str | Path) -> str:
    """A file's path as ``robo_jev.json`` records it: repo-relative inside a checkout of this repo, else just the file name."""
    text = public_paths(str(path))
    return Path(text).name if Path(text).is_absolute() else text


def check_public_text(folder: str | Path) -> dict[str, int]:
    """Scan every text file of `folder` (all but ``.safetensors`` and images) for :data:`PRIVATE_TEXT`; raise on any hit (the message names
    the file, the line and the kind of hit, never the matched text). Returns ``{file: bytes scanned}``."""
    folder = Path(folder)
    hits, scanned = [], {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if path.suffix.lower() in (".safetensors", ".png", ".jpg", ".jpeg", ".gif", ".webp"):
            continue
        name = str(path.relative_to(folder))
        text = path.read_text(encoding="utf-8")
        scanned[name] = len(text.encode("utf-8"))
        for kind, pattern in PRIVATE_TEXT.items():
            for match in pattern.finditer(text):
                hits.append(f"{name}:{text.count(chr(10), 0, match.start()) + 1} ({kind})")
    _need(not hits, f"{folder}: private text in {len(hits)} place(s): {hits[:10]}")
    return scanned
