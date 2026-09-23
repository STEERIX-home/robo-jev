"""run manifest — 한 run의 명세·상한·상태를 담은 파일과 그 검증 (docs/06 Task 6, docs/05 §7).

스키마는 :data:`SCHEMA_PATH`(`infra/run-manifest.schema.json`)가 정본이고 이 모듈은 그것을 **읽어서** 검사한다
— 스키마를 코드에 두 번 적지 않는다. 검사기(:func:`validate`)는 저장소가 쓰는 조각만 구현한 작은 것이다
(`type`·`const`·`enum`·`required`·`properties`·`additionalProperties`·`items`·`minItems`·`minLength`·`minimum`·
`exclusiveMinimum`·`pattern`): 의존성을 늘리지 않기 위해서다. 스키마에 그 밖의 키워드를 쓰면
:func:`validate` 가 **조용히 넘기지 않고** 거절한다(:data:`SUPPORTED_KEYWORDS`).

**상한의 산술 (A3).** 세 상한은 모두 하나의 벽시계 마감으로 환산되고 가장 이른 것이 구속한다::

    max_wall_hours                  그대로
    max_gpu_hours / gpus            GPU 수로 나눈다 (gpu_hours = 벽시계 × gpus)
    max_usd / hourly_usd            노드 시간 단가로 나눈다

:func:`budget_deadline_hours` 가 (마감, 구속한 상한의 이름)을 돌려주고, 원격 러너는 그 마감을 자기 시계로
재므로 **실행기가 죽어도** 상한이 지켜진다. 넘으면 checkpoint를 쓴 뒤 `failed(reason=budget)`이다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "STATES",
    "TERMINAL_STATES",
    "Budget",
    "budget_deadline_hours",
    "budget_spend",
    "git_revision",
    "load_manifest",
    "new_manifest",
    "note_state",
    "now",
    "plan_budget",
    "save_manifest",
    "schema",
    "sha256_of",
    "validate",
]

REPO = Path(__file__).resolve().parents[3]
#: 스키마 정본. 저장소 안에 있으므로 launcher와 원격 러너가 같은 파일을 읽는다.
SCHEMA_PATH = REPO / "infra" / "run-manifest.schema.json"
SCHEMA_VERSION = "run-manifest-v0"

#: 상태. `unknown`은 '확인하지 못했다'이지 '끝났다'가 아니다.
STATES = ("prepared", "running", "completed", "failed", "cancelled", "unknown")
#: 더 이상 움직이지 않는 상태. `unknown`은 **여기 없다** — 사람이 확인해야 한다.
TERMINAL_STATES = ("completed", "failed", "cancelled")

#: 이 검사기가 아는 키워드. 스키마가 다른 것을 쓰면 거절한다 (조용히 통과시키지 않는다).
SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema", "$id", "title", "description", "type", "const", "enum", "required", "properties",
        "additionalProperties", "items", "minItems", "minLength", "minimum", "exclusiveMinimum", "pattern",
    }
)


def now() -> str:
    """UTC ISO-8601 (초 단위). 모든 시각 필드가 이 서식이다."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: str | Path) -> str:
    """파일의 sha256 (16진). 큰 checkpoint도 읽으므로 1 MiB씩 흘려 읽는다."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 스키마
# --------------------------------------------------------------------------


def schema(path: str | Path | None = None) -> dict[str, Any]:
    """스키마 파일을 읽는다 (캐시하지 않는다 — 검사가 파일을 고쳐도 다음 호출이 본다)."""
    return json.loads(Path(path or SCHEMA_PATH).read_text(encoding="utf-8"))


def _type_ok(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    raise ValueError(f"스키마: 모르는 type {name!r}")


def _check(node: Any, spec: dict[str, Any], where: str, errors: list[str]) -> None:
    unknown = set(spec) - SUPPORTED_KEYWORDS
    if unknown:
        raise ValueError(f"스키마 {where}: 이 검사기가 모르는 키워드 {sorted(unknown)} — 검사기를 먼저 늘린다")
    if "const" in spec and node != spec["const"]:
        errors.append(f"{where}: {spec['const']!r}여야 한다 (받은 값: {node!r})")
        return
    if "enum" in spec and node not in spec["enum"]:
        errors.append(f"{where}: {spec['enum']} 중 하나여야 한다 (받은 값: {node!r})")
        return
    if "type" in spec:
        names = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        if not any(_type_ok(node, name) for name in names):
            errors.append(f"{where}: type {names} (받은 값: {type(node).__name__})")
            return
    if isinstance(node, str):
        if "minLength" in spec and len(node) < spec["minLength"]:
            errors.append(f"{where}: 길이가 {spec['minLength']} 이상이어야 한다")
        if "pattern" in spec and not re.search(spec["pattern"], node):
            errors.append(f"{where}: {spec['pattern']!r}에 맞지 않는다 (받은 값: {node!r})")
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        if "minimum" in spec and node < spec["minimum"]:
            errors.append(f"{where}: {spec['minimum']} 이상이어야 한다 (받은 값: {node})")
        if "exclusiveMinimum" in spec and node <= spec["exclusiveMinimum"]:
            errors.append(f"{where}: {spec['exclusiveMinimum']}보다 커야 한다 (받은 값: {node})")
    if isinstance(node, list):
        if "minItems" in spec and len(node) < spec["minItems"]:
            errors.append(f"{where}: 원소가 {spec['minItems']}개 이상이어야 한다")
        item_spec = spec.get("items")
        if isinstance(item_spec, dict):
            for index, item in enumerate(node):
                _check(item, item_spec, f"{where}[{index}]", errors)
    if isinstance(node, dict):
        properties = spec.get("properties") or {}
        for key in spec.get("required") or ():
            if key not in node:
                errors.append(f"{where}: 필수 키 {key!r}가 없다")
        if spec.get("additionalProperties") is False:
            extra = [key for key in node if key not in properties]
            if extra:
                errors.append(f"{where}: 모르는 키 {sorted(extra)} (허용: {sorted(properties)})")
        for key, value in node.items():
            if key in properties:
                _check(value, properties[key], f"{where}.{key}" if where else key, errors)


def validate(manifest: dict[str, Any], *, schema_path: str | Path | None = None) -> list[str]:
    """스키마 위반을 **전부** 모아 돌려준다 (빈 목록 = 통과). 예외는 스키마 자체가 이상할 때만."""
    errors: list[str] = []
    _check(manifest, schema(schema_path), "manifest", errors)
    return errors


def require_valid(manifest: dict[str, Any], *, where: str = "manifest") -> dict[str, Any]:
    errors = validate(manifest)
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"{where}: run manifest가 스키마에 맞지 않는다 ({SCHEMA_PATH.name}):\n  - {joined}")
    return manifest


# --------------------------------------------------------------------------
# 상한의 산술
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """상한 셋 + 단가. manifest의 `budget` 블록과 같은 모양이다."""

    hourly_usd: float
    gpus: int
    max_wall_hours: float | None = None
    max_gpu_hours: float | None = None
    max_usd: float | None = None

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> Budget:
        block = manifest["budget"]
        return cls(
            hourly_usd=float(block["hourly_usd"]),
            gpus=int(block["gpus"]),
            max_wall_hours=None if block.get("max_wall_hours") is None else float(block["max_wall_hours"]),
            max_gpu_hours=None if block.get("max_gpu_hours") is None else float(block["max_gpu_hours"]),
            max_usd=None if block.get("max_usd") is None else float(block["max_usd"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "hourly_usd": self.hourly_usd,
            "gpus": self.gpus,
            "max_wall_hours": self.max_wall_hours,
            "max_gpu_hours": self.max_gpu_hours,
            "max_usd": self.max_usd,
        }


def budget_spend(budget: Budget, elapsed_seconds: float) -> dict[str, float]:
    """경과 시간에서 쓴 양. `gpu_hours = 벽시계 × gpus`, `usd = 벽시계 × 노드 단가`다 —
    단가가 **노드** 단가이므로 GPU 수를 다시 곱하지 않는다 (docs/05 §5 표가 둘을 나눠 적는다)."""
    hours = max(float(elapsed_seconds), 0.0) / 3600.0
    return {
        "wall_hours": hours,
        "gpu_hours": hours * float(budget.gpus),
        "estimated_usd": hours * float(budget.hourly_usd),
    }


def budget_deadline_hours(budget: Budget) -> tuple[float | None, str | None]:
    """상한 셋을 벽시계 마감으로 환산해 (가장 이른 마감, 그 상한의 이름). 셋 다 없으면 (None, None).

    ``max_gpu_hours``는 `gpus`가 0이면(= GPU 없는 CPU run) 구속하지 않는다 — 0으로 나누지 않는다.
    ``max_usd``도 단가가 0이면 구속하지 않는다.
    """
    candidates: list[tuple[float, str]] = []
    if budget.max_wall_hours is not None:
        candidates.append((float(budget.max_wall_hours), "max_wall_hours"))
    if budget.max_gpu_hours is not None and budget.gpus > 0:
        candidates.append((float(budget.max_gpu_hours) / float(budget.gpus), "max_gpu_hours"))
    if budget.max_usd is not None and budget.hourly_usd > 0:
        candidates.append((float(budget.max_usd) / float(budget.hourly_usd), "max_usd"))
    if not candidates:
        return None, None
    hours, name = min(candidates, key=lambda pair: (pair[0], pair[1]))
    return hours, name


def plan_budget(budget: Budget, *, seconds_per_step: float | None, max_steps: int, assumption: str | None = None) -> dict[str, Any]:
    """step당 초의 **가정**에서 예상 벽시계·GPU 시간·비용. 실측이 아니면 `assumption`에 그렇게 적는다."""
    if seconds_per_step is None:
        return {"wall_hours": None, "gpu_hours": None, "usd": None, "seconds_per_step": None, "assumption": assumption}
    hours = float(seconds_per_step) * int(max_steps) / 3600.0
    return {
        "wall_hours": round(hours, 4),
        "gpu_hours": round(hours * budget.gpus, 4),
        "usd": round(hours * budget.hourly_usd, 4),
        "seconds_per_step": float(seconds_per_step),
        "assumption": assumption,
    }


# --------------------------------------------------------------------------
# 만들기·읽기·쓰기
# --------------------------------------------------------------------------


def git_revision(repo: str | Path | None = None) -> dict[str, Any]:
    """코드 리비전 — sha, 브랜치, **커밋되지 않은 변경이 있는지**와 그 파일 이름.

    git이 없거나 저장소가 아니면 `git_sha: null`, `dirty: true`다 — '모르면 깨끗하다'로 읽지 않는다.
    """
    root = Path(repo or REPO)

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - git이 없는 상자
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    if sha is None or status is None:
        return {"git_sha": sha, "dirty": True, "branch": branch, "dirty_files": []}
    files = [line[3:] for line in status.splitlines() if line.strip()]
    return {"git_sha": sha, "dirty": bool(files), "branch": branch, "dirty_files": sorted(files)[:50]}


def new_manifest(**fields: Any) -> dict[str, Any]:
    """빈 골격 + 주어진 필드. 스키마의 필수 키를 전부 가진 dict를 돌려준다 (값은 호출자가 채운다)."""
    stamp = now()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "",
        "created_at": stamp,
        "updated_at": stamp,
        "code": {"git_sha": None, "dirty": True, "branch": None, "dirty_files": []},
        "train_config": {"path": "", "sha256": "", "resolved_sha256": ""},
        "dataset_manifests": [],
        "tokenizer": {"id": "", "sha256": None},
        "contract": {"sha256": ""},
        "seed": 0,
        "max_steps": 1,
        "checkpoint": {"every": 1, "keep_steps": [], "resume_from": None},
        "budget": {"hourly_usd": 0.0, "gpus": 0, "max_wall_hours": None, "max_gpu_hours": None, "max_usd": None},
        "artifacts": [],
        "backend": {"kind": "local", "remote_dir": "", "gpus": 0},
        "state": "prepared",
        "progress": {},
        "finished_at": None,
        "exit_reason": None,
        "budget_limit": None,
        "history": [],
        "notes": [],
    }
    manifest.update(fields)
    return manifest


def note_state(manifest: dict[str, Any], state: str, *, note: str | None = None, command: str | None = None) -> dict[str, Any]:
    """상태를 옮기고 장부에 한 줄 덧붙인다. 상태 이름은 스키마의 것만."""
    if state not in STATES:
        raise ValueError(f"state: {list(STATES)} 중 하나여야 한다 (받은 값: {state!r})")
    stamp = now()
    manifest["state"] = state
    manifest["updated_at"] = stamp
    manifest.setdefault("history", []).append({"at": stamp, "state": state, "note": note, "command": command})
    if state in TERMINAL_STATES and not manifest.get("finished_at"):
        manifest["finished_at"] = stamp
    return manifest


def save_manifest(path: str | Path, manifest: dict[str, Any], *, check: bool = True) -> Path:
    """검사한 뒤 atomic하게 쓴다 (임시 파일 → fsync → rename). 반쯤 쓰인 명세를 남기지 않는다."""
    if check:
        require_valid(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=1) + "\n"
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:  # pragma: no cover - 디스크 오류
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def load_manifest(path: str | Path, *, check: bool = True) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if check:
        require_valid(manifest, where=str(path))
    return manifest
