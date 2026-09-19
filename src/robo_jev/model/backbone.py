"""backbone 가중치의 보관·대조 — `scripts/fetch_backbone.py`가 `artifacts/models/<id>/`(git 제외)에 받아 둔 파일과
그 manifest를 읽는 쪽 (Task 2b G0a).

:mod:`robo_jev.model.tokenizer` 가 `tokenizer.json`에 하는 일을 가중치에 한다. 모델 코드는 `transformers`를 import하지
않는다 — safetensors header(파일 앞 8바이트 little-endian 길이 + JSON)만 읽어 tensor 이름·모양을 세고, manifest의
해시·크기와 파일을 대조한다. 실제 적재는 측정 스크립트(`scripts/measure_candidates.py`)의 runner가 한다.

**manifest.** 보관 뿌리의 ``manifest.json`` 하나가 받은 후보 전부를 ``models[id]``로 갖는다: 해석된 revision SHA,
파일별 sha256·크기, 파일 묶음의 지문(:func:`fileset_digest`), 라이선스, 받은 시각, header에서 센 파라미터 수.

**대조.** :func:`describe_backbone` 은 읽기 전에 manifest의 모든 파일이 있고 크기가 같은지, 작은 파일
(:data:`SMALL_FILES`: config.json·index·generation_config.json)의 sha256이 같은지 본다. safetensors 전부의 sha256은
``full=True``일 때만 잰다(수십 GB) — 크기 대조만으로도 다른 revision의 파일이 같은 자리에 놓인 대부분의 경우를
잡고, 지문 대조는 fetch가 받을 때 한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from robo_jev.model.tokenizer import sha256_of_file

__all__ = [
    "ARTIFACT_DIR",
    "CANDIDATES_CONFIG",
    "FETCH_SCRIPT",
    "FILE_PATTERNS",
    "MANIFEST_NAME",
    "SMALL_FILES",
    "STRUCTURE_KEYS",
    "TEXT_PREFIXES",
    "backbone_root",
    "config_structure",
    "count_params",
    "describe_backbone",
    "fileset_digest",
    "hash_files",
    "read_manifest",
    "safetensors_header",
]

#: 저장소 뿌리 기준의 가중치 보관 위치 (git 제외). `<root>/<id>/`에 파일, `<root>/manifest.json`에 후보 전부.
ARTIFACT_DIR = Path("artifacts/models")
MANIFEST_NAME = "manifest.json"
FETCH_SCRIPT = "scripts/fetch_backbone.py"
CANDIDATES_CONFIG = Path("configs/model/candidates.yaml")

#: 받는 파일. tokenizer는 `artifacts/tokenizers`의 것을 쓰므로 받지 않는다. `generation_config.json`은 없어도 된다.
FILE_PATTERNS = ("config.json", "*.safetensors", "model.safetensors.index.json", "generation_config.json")

#: 적재 때 언제나 sha256을 대조하는 작은 파일 (safetensors는 크기만; `full=True`면 전부).
SMALL_FILES = ("config.json", "model.safetensors.index.json", "generation_config.json")

#: 체크포인트 tensor 이름 가운데 native 경로(`AutoModelForCausalLM`)가 싣는 언어 모델의 접두어. 나머지는 vision
#: tower(`model.visual.`)와 MTP head(`mtp.`)다 — `params_total`은 앞의 것만, `params_checkpoint`는 전부 센다.
TEXT_PREFIXES = ("model.language_model.", "lm_head.")

#: `configs/model/candidates.yaml`의 항목과 받은 config.json이 같아야 하는 구조 값.
STRUCTURE_KEYS = (
    "model_type",
    "layer_types",
    "full_attention_interval",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_attention",
    "vocab_size",
    "tie_word_embeddings",
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def backbone_root(root: str | Path | None = None) -> Path:
    """가중치 보관 디렉터리. 현재 작업 디렉터리에 없으면 패키지 뿌리에서 찾는다."""
    if root is not None:
        return Path(root)
    if ARTIFACT_DIR.is_dir():
        return ARTIFACT_DIR
    return _PACKAGE_ROOT / ARTIFACT_DIR


def read_manifest(root: str | Path | None = None) -> dict[str, Any]:
    """보관 뿌리의 manifest. 없으면 빈 ``{"models": {}}``."""
    path = backbone_root(root) / MANIFEST_NAME
    if not path.is_file():
        return {"models": {}}
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("models"), dict):
        raise ValueError(f"{path}: manifest 형식이 아니다 (`models` dict가 있어야 한다)")
    return record


# --------------------------------------------------------------------------
# 파일 해시·지문
# --------------------------------------------------------------------------


def hash_files(target: str | Path) -> dict[str, dict[str, Any]]:
    """`target/` 바로 아래에서 :data:`FILE_PATTERNS`에 맞는 파일의 ``{이름: {sha256, bytes}}`` (이름순)."""
    base = Path(target)
    names = sorted({path.name for pattern in FILE_PATTERNS for path in base.glob(pattern) if path.is_file()})
    return {name: {"sha256": sha256_of_file(base / name), "bytes": (base / name).stat().st_size} for name in names}


def fileset_digest(files: dict[str, dict[str, Any]]) -> str:
    """파일 묶음의 지문 — 이름순 ``"<sha256>  <이름>\\n"`` 줄들의 sha256 (sha256sum 파일의 해시와 같은 꼴).

    파일이 여럿이라 tokenizer의 파일 하나 해시 대신 이것을 `--expect-sha256`과 manifest의 `digest`로 쓴다.
    """
    lines = "".join(f"{files[name]['sha256']}  {name}\n" for name in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def matches_patterns(name: str) -> bool:
    return any(fnmatch(name, pattern) for pattern in FILE_PATTERNS)


# --------------------------------------------------------------------------
# safetensors header — 적재 없이 tensor를 센다
# --------------------------------------------------------------------------


def safetensors_header(path: str | Path) -> dict[str, dict[str, Any]]:
    """safetensors 파일의 header(``{이름: {dtype, shape, data_offsets}}``; ``__metadata__``는 뺀다)."""
    with Path(path).open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length).decode("utf-8"))
    return {name: spec for name, spec in header.items() if name != "__metadata__"}


def count_params(paths: list[str | Path]) -> dict[str, Any]:
    """safetensors 파일들의 tensor 원소 수 — ``total``(전부), ``text``(:data:`TEXT_PREFIXES`), ``vision``, ``mtp``,
    ``other``, 그리고 ``tensors``(개수). 이름의 "2B"를 믿지 않고 센다."""
    counts: Counter[str] = Counter()
    tensors = 0
    seen: set[str] = set()
    for path in paths:
        for name, spec in safetensors_header(path).items():
            if name in seen:
                raise ValueError(f"{path}: tensor 이름이 다른 파일과 겹친다: {name!r}")
            seen.add(name)
            tensors += 1
            numel = math.prod(int(dim) for dim in spec["shape"]) if spec["shape"] else 1
            counts["total"] += numel
            if name.startswith(TEXT_PREFIXES):
                counts["text"] += numel
            elif name.startswith("model.visual."):
                counts["vision"] += numel
            elif name.startswith("mtp."):
                counts["mtp"] += numel
            else:
                counts["other"] += numel
    return {key: int(counts[key]) for key in ("total", "text", "vision", "mtp", "other")} | {"tensors": tensors}


# --------------------------------------------------------------------------
# config.json ↔ candidates.yaml
# --------------------------------------------------------------------------


def config_structure(config: dict[str, Any]) -> dict[str, Any]:
    """받은 `config.json`(vision을 포함한 상위 config)에서 yaml 항목의 구조 값(:data:`STRUCTURE_KEYS`)을 뽑는다.

    `tie_word_embeddings`는 text_config에 없으면 상위 값을(9B·27B의 config이 그렇다), 그것도 없으면 false를 쓴다.
    """
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    tie = text.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
    return {
        "model_type": config.get("model_type"),
        "layer_types": dict(sorted(Counter(text.get("layer_types") or ()).items(), key=lambda item: -item[1])),
        "full_attention_interval": text.get("full_attention_interval"),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "linear_attention": {
            "num_key_heads": text.get("linear_num_key_heads"),
            "num_value_heads": text.get("linear_num_value_heads"),
            "key_head_dim": text.get("linear_key_head_dim"),
            "value_head_dim": text.get("linear_value_head_dim"),
            "conv_kernel_dim": text.get("linear_conv_kernel_dim"),
        },
        "vocab_size": text.get("vocab_size"),
        "tie_word_embeddings": bool(tie),
    }


# --------------------------------------------------------------------------
# 적재 전 대조
# --------------------------------------------------------------------------


def describe_backbone(identifier: str, root: str | Path | None = None, *, full: bool = False) -> dict[str, Any]:
    """받아 둔 후보의 정체 — manifest 항목에 `verified`를 더해 돌려준다. 파일이 없거나 다르면 예외다.

    * manifest에 없는 id → `FileNotFoundError`(받는 방법을 적는다). 조용히 다른 후보로 바꾸지 않는다.
    * 파일이 없거나 크기가 다르거나 :data:`SMALL_FILES`(``full=True``면 전부)의 sha256이 다르면 `ValueError`.
    """
    base = backbone_root(root)
    manifest = read_manifest(base)
    entry = manifest["models"].get(identifier)
    if entry is None:
        raise FileNotFoundError(
            f"backbone 가중치가 없다: {identifier!r} ({base / MANIFEST_NAME}에 항목이 없다). "
            f"`uv run python {FETCH_SCRIPT} --id {identifier}`로 받는다"
        )
    target = base / identifier
    problems: list[str] = []
    for name, spec in entry["files"].items():
        path = target / name
        if not path.is_file():
            problems.append(f"{name}: 파일이 없다")
            continue
        size = path.stat().st_size
        if size != int(spec["bytes"]):
            problems.append(f"{name}: 크기가 다르다 (파일 {size}, manifest {spec['bytes']})")
            continue
        if full or name in SMALL_FILES:
            actual = sha256_of_file(path)
            if actual != str(spec["sha256"]).lower():
                problems.append(f"{name}: sha256이 다르다 (파일 {actual[:12]}…, manifest {str(spec['sha256'])[:12]}…)")
    if problems:
        raise ValueError(
            f"backbone {identifier} ({target}): manifest({base / MANIFEST_NAME})와 다르다 — "
            + "; ".join(problems)
            + f". 다른 revision의 파일이거나 손상된 파일이다. manifest의 revision·해시로 다시 받는다: "
            f"`uv run python {FETCH_SCRIPT} --from-manifest {base / MANIFEST_NAME} --id {identifier}`"
        )
    return {**entry, "path": str(target), "verified": "full" if full else "sizes+small-files"}
