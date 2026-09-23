"""실행기 — `prepare · launch · status · fetch · cancel · resume` (docs/06 Task 6).

여기 있는 여섯 함수가 전부이고 :mod:`scripts.launch_run` 은 그 위의 얇은 CLI다. 모든 함수는 **run manifest
하나**를 받아 고쳐 쓴다 — 명세가 run의 정체이자 장부다.

읽는 순서::

    prepare  설정·데이터·tokenizer·계약·코드 리비전을 고정하고 상한을 적는다 → 묶음(bundle)을 만든다
    launch   묶음을 올리고 원격에서 **분리 실행**으로 러너를 띄운다 (실행기가 죽어도 run은 산다)
    status   원격 state.json + 프로세스 생사 → 명세의 state·progress를 갱신한다
    fetch    원격이 계산한 sha256과 함께 산출물을 가져와 **다시 해시해 대조**한다
    cancel   멈춤을 요청하고 **확인될 때까지 기다린다**. 확인 못 하면 `unknown`이다
    resume   **가져온** checkpoint에서 잇는 자식 run을 만든다 (등록된 것만)

**cancel의 규칙** (docs/06 Task 6: "종료 API 실패는 미종료 상태로 명시"): 정지 명령이 실패했거나, 보냈는데
기한 안에 프로세스가 사라진 것을 확인하지 못했으면 상태는 `cancelled`가 아니라 **`unknown`**이고 사람에게
알린다. 돈이 계속 나가고 있을 수 있기 때문이다.

**resume의 규칙** (R2의 재개 게이트와 같은 자리): 이어 갈 checkpoint는 이 명세의 `artifacts`에 **등록되고
sha256이 대조된** 것이어야 한다. 아무 파일에서나 잇지 않는다.
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from robo_jev.launch.manifest import (
    REPO,
    Budget,
    budget_deadline_hours,
    budget_spend,
    git_revision,
    new_manifest,
    note_state,
    now,
    plan_budget,
    require_valid,
    save_manifest,
    sha256_of,
    sha256_of_text,
)
from robo_jev.launch.providers.base import Backend, Handle

__all__ = [
    "CancelNotConfirmed",
    "artifact_plan",
    "bundle_dir",
    "portable_path",
    "cancel",
    "fetch",
    "launch",
    "prepare",
    "remaining_budget",
    "resume",
    "status",
]


class CancelNotConfirmed(RuntimeError):
    """멈춤을 확인하지 못했다 — 상태는 `unknown`이고 사람이 봐야 한다."""


def portable_path(path: str | Path) -> str:
    """저장소 **안**의 경로는 저장소 기준 상대 경로로 적는다.

    명세와 묶음은 다른 상자에서 읽힌다. 거기서는 이 상자의 체크아웃 경로가 없으므로, 저장소 안을 가리키는
    경로는 상대로 적어 원격 러너가 자기 `--repo` 기준으로 풀게 한다(:func:`robo_jev.launch.runner._resolve_paths`).
    저장소 밖의 경로(영속 볼륨 위의 데이터 등)는 그대로 둔다.

    symlink를 **따라가지 않은** 절대 경로를 먼저 본다: 이 저장소의 `artifacts`는 worktree에서 본 저장소를 가리키는
    symlink라 `resolve()`만 쓰면 저장소 밖으로 나가 버린다(그러면 다른 상자에서 못 푸는 절대 경로가 명세에 남는다).
    """
    roots = (REPO, REPO.resolve())
    for candidate in (Path(os.path.abspath(path)), Path(path).resolve()):
        for root in roots:
            try:
                return str(candidate.relative_to(root))
            except ValueError:
                continue
    return str(path)


def bundle_dir(manifest_path: str | Path) -> Path:
    """명세 옆의 묶음 디렉터리. 명세 이름에서 짓는다 — 한 디렉터리에 여러 run의 명세를 두어도 섞이지 않는다."""
    path = Path(manifest_path)
    return path.parent / f"{path.stem}-bundle"


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def artifact_plan(run_id: str) -> list[dict[str, Any]]:
    """run 하나가 남기는 것 (docs/06 Task 6: checkpoint·metrics·config·report·로그).

    `remote`는 원격 run 디렉터리 기준 상대 경로다. `required_`가 거짓인 것은 없어도 회수가 실패하지 않는다
    (예: 예산에 걸려 `checkpoint_keep_steps`에 닿기 전에 끝난 run).
    """
    return [
        {"name": "checkpoint", "kind": "checkpoint", "remote": f"runs/{run_id}/checkpoint.pt", "required_": True},
        {"name": "checkpoint_keep", "kind": "checkpoint", "remote": f"runs/{run_id}/checkpoint-step*.pt", "glob": True, "required_": False},
        {"name": "metrics", "kind": "metrics", "remote": f"runs/{run_id}/metrics.json", "required_": True},
        {"name": "config", "kind": "config", "remote": f"runs/{run_id}/config.yaml", "required_": True},
        {"name": "report", "kind": "report", "remote": f"runs/{run_id}/report.json", "required_": False},
        {"name": "state", "kind": "state", "remote": "state.json", "required_": True},
        {"name": "log", "kind": "log", "remote": "logs/runner.log", "required_": True},
    ]


def prepare(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    run_id: str,
    backend: Backend,
    budget: Budget,
    manifest_path: str | Path,
    overrides: dict[str, Any] | None = None,
    price_source: str | None = None,
    seconds_per_step: float | None = None,
    assumption: str | None = None,
    notes: list[str] | None = None,
    resume_from: str | None = None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    """명세를 만들고 묶음을 정리한다. GPU도 원격도 건드리지 않는다 — 여기까지는 전부 로컬이다.

    묶음(`<manifest 디렉터리>/bundle/`)에 드는 것: 푼 학습 설정(`config.yaml`), 이 명세(`manifest.json`),
    데이터 manifest **파일**의 사본(`datasets/`), 실제 tokenizer 파일(`tokenizer/`). **데이터 레코드는 담지
    않는다** — 명세에 경로와 sha256만 적고 원격 러너가 시작 전에 대조한다.
    """
    from robo_jev.model.contract_digest import contract_digest
    from robo_jev.train import resolve_config, tokenizer_block

    config = copy.deepcopy(config)
    config["run_id"] = run_id
    if resume_from is not None:
        config["resume"] = resume_from
    resolved = resolve_config(config)

    tokenizer = tokenizer_block(str(resolved["tokenizer"]))
    contract = contract_digest(tokenizer.get("sha256") or "whitespace")

    datasets, local_datasets = [], []
    for entry in resolved["dataset_manifests"]:
        path = Path(entry["path"])
        local_datasets.append(path)
        entry["path"] = portable_path(path)  # 묶음의 설정도 이동 가능한 경로로 나간다
        datasets.append(
            {
                "path": entry["path"],
                "sha256": sha256_of(path),
                "domain": entry.get("domain"),
                "material": entry.get("material"),
                "files": entry.get("files"),
                "bundled": True,
            }
        )
    if isinstance(resolved.get("model_config"), str):
        resolved["model_config"] = portable_path(resolved["model_config"])
    resolved_text = yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False)

    manifest = new_manifest(
        run_id=run_id,
        code=git_revision(),
        train_config={
            "path": portable_path(config_path),
            "sha256": sha256_of(config_path),
            "resolved_sha256": sha256_of_text(resolved_text),
            "run_name": resolved.get("run_name"),
            "overrides": copy.deepcopy(overrides or {}),
        },
        dataset_manifests=datasets,
        tokenizer={
            "id": str(tokenizer.get("id") or tokenizer.get("name") or resolved["tokenizer"]),
            "kind": tokenizer.get("kind"),
            "sha256": tokenizer.get("sha256"),
            "revision": tokenizer.get("revision"),
            "path": tokenizer.get("path"),
        },
        contract=contract,
        seed=int(resolved["seed"]),
        max_steps=int(resolved["max_steps"]),
        checkpoint={
            "every": int(resolved["checkpoint_every"]),
            "keep_steps": list(resolved.get("checkpoint_keep_steps") or []),
            "resume_from": resume_from,
        },
        budget={
            **budget.as_dict(),
            "price_source": price_source,
            "planned": plan_budget(budget, seconds_per_step=seconds_per_step, max_steps=int(resolved["max_steps"]), assumption=assumption),
        },
        artifacts=artifact_plan(run_id),
        backend={**backend.describe(), "unit": run_id, "launcher": None, "pid": None, "provider": "existing-box", "instance_id": None},
        notes=list(notes or []),
        parent_run_id=parent_run_id,
    )
    note_state(manifest, "prepared", note=f"{Path(config_path).name} · {len(datasets)} dataset manifest · 마감 {budget_deadline_hours(budget)[0]} h")

    bundle = bundle_dir(manifest_path)
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "datasets").mkdir(parents=True, exist_ok=True)
    (bundle / "config.yaml").write_text(resolved_text, encoding="utf-8")
    for index, path in enumerate(local_datasets):
        shutil.copy2(path, bundle / "datasets" / f"{index:02d}-{path.name}")
    token_path = tokenizer.get("path")
    if token_path and Path(token_path).is_file():
        (bundle / "tokenizer").mkdir(exist_ok=True)
        shutil.copy2(token_path, bundle / "tokenizer" / Path(token_path).name)
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    save_manifest(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------
# launch
# --------------------------------------------------------------------------


def launch(manifest: dict[str, Any], backend: Backend, *, manifest_path: str | Path, force: bool = False) -> dict[str, Any]:
    """묶음을 올리고 원격에서 분리 실행으로 러너를 띄운다."""
    if manifest["state"] == "running" and not force:
        raise RuntimeError(f"{manifest['run_id']}: 이미 running이다 — 먼저 `status`로 확인하거나 `cancel`한다 (--force로 무시)")
    bundle = bundle_dir(manifest_path)
    if not (bundle / "config.yaml").is_file():
        raise FileNotFoundError(f"{bundle}: 묶음이 없다 — 먼저 `prepare`")
    # 명세는 매번 다시 담는다 (resume이 부모 명세를 고쳐 놓았을 수 있다).
    (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    unit = str(manifest["backend"].get("unit") or manifest["run_id"])
    backend.mkdir()
    backend.mkdir("logs")
    backend.reset_unit(unit)
    # 지난 시도의 상태 파일을 지운다 — 남겨 두면 새 러너가 자기 것을 쓰기 전의 `status`가
    # **옛 run의 `completed`**를 읽는다. 로그는 지우지 않는다(장부라서 이어 붙는다).
    backend.shell(f"rm -f {shlex.quote(backend.path('state.json'))}")
    backend.push(bundle, "bundle")

    argv = [*shlex.split(backend.remote_python), "-m", "robo_jev.launch.runner", "--bundle", backend.path("bundle"), "--repo", backend.remote_repo]
    handle = backend.start(argv, unit=unit, log_relpath="logs/runner.log")
    manifest["backend"].update(handle.as_dict())
    manifest["backend"]["unit"] = unit
    note_state(manifest, "running", note=f"{handle.launcher} · {unit or handle.pid}", command=shlex.join(argv))
    save_manifest(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def _remote_state(backend: Backend) -> dict[str, Any] | None:
    text = backend.read_text("state.json")
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:  # pragma: no cover - 쓰는 도중에 읽는 경우 (atomic 쓰기라 드물다)
        return None


def _progress(remote: dict[str, Any] | None, *, alive: bool) -> dict[str, Any]:
    """원격 state.json에서 진행(step·손실·경과·**예상 비용**)을 옮겨 적는다. 비용은 원격이 계산한 값이다."""
    progress: dict[str, Any] = {"observed_at": now(), "alive": alive}
    if remote:
        for key in ("step", "max_steps", "loss", "elapsed_seconds", "gpu_hours", "estimated_usd", "seconds_per_step"):
            progress[key] = remote.get(key)
        progress["updated_at"] = remote.get("updated_at")
    return progress


def status(manifest: dict[str, Any], backend: Backend, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """원격 상태와 진행을 읽어 명세를 갱신한다. 돌려주는 것은 명세다."""
    handle = Handle.from_manifest(manifest["backend"])
    remote = _remote_state(backend)
    alive = backend.alive(handle) if handle is not None else False
    manifest["progress"] = _progress(remote, alive=alive)

    if manifest["state"] in ("prepared",) and handle is None:
        note = "아직 띄우지 않았다"
        state, reason = "prepared", None
    elif remote is None:
        # 상태 파일이 아직(또는 영영) 없다.
        state, reason = ("running", None) if alive else ("failed", "process_vanished")
        note = "원격 state.json이 없다 — " + ("러너가 막 떴다" if alive else "프로세스도 없다: 시작 전에 죽었다")
    elif remote.get("state") == "running":
        state, reason = ("running", None) if alive else ("failed", "process_vanished")
        note = f"step {remote.get('step')}/{remote.get('max_steps')}" + ("" if alive else " — 상태는 running인데 프로세스가 없다 (강제 종료)")
    else:
        state = str(remote.get("state"))
        reason = remote.get("exit_reason")
        note = f"원격이 끝냈다: {state}({reason}) · step {remote.get('step')}"
        if remote.get("budget_limit"):
            manifest["budget_limit"] = remote["budget_limit"]

    if state != manifest["state"] or manifest.get("exit_reason") != reason:
        manifest["exit_reason"] = reason
        note_state(manifest, state, note=note)
    else:
        manifest["updated_at"] = now()
    if remote and remote.get("finished_at"):
        manifest["finished_at"] = remote["finished_at"]
    if manifest_path is not None:
        save_manifest(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def fetch(
    manifest: dict[str, Any],
    backend: Backend,
    *,
    dest: str | Path,
    manifest_path: str | Path | None = None,
    names: list[str] | None = None,
    strict: bool | None = None,
) -> dict[str, Any]:
    """산출물을 가져와 **원격이 잰 sha256과 로컬에서 다시 잰 sha256을 대조**한다.

    하나라도 다르면 `ValueError`다 — 반쯤 전송된 checkpoint를 성공으로 표시하지 않는다 (docs/05 §7).

    **없는 산출물의 규칙.** `completed`로 끝난 run은 전부 남겼어야 하므로 없으면 오류다. 강제 종료된 run
    (`failed`·`cancelled`·`unknown`)은 **부분적인 것이 정상**이다 — 그때는 있는 것을 가져오고 없는 것의
    이름을 명세에 적는다(`fetch`가 조용히 성공한 척하지 않는다). `strict`로 그 기본값을 덮어쓴다.
    """
    plan = [entry for entry in manifest["artifacts"] if names is None or entry["name"] in names]
    patterns = [entry["remote"] for entry in plan]
    remote_files = {item["path"]: item for item in backend.inventory(patterns)}
    if strict is None:
        strict = manifest["state"] == "completed"

    missing = []
    for entry in plan:
        pattern = entry["remote"]
        hits = [path for path in remote_files if Path(path).match(pattern) or path == pattern]
        if not hits and entry.get("required_", True):
            missing.append(pattern)
    if missing and strict:
        raise FileNotFoundError(
            f"{manifest['run_id']}: 원격에 없는 산출물 {missing} — run이 `{manifest['state']}`인데 남겼어야 할 것이 없다 "
            "(강제 종료된 run에서 부분 회수를 하려면 strict=False)"
        )

    root = Path(dest)
    relpaths = sorted(remote_files)
    if relpaths:
        backend.fetch(relpaths, root)

    fetched: list[dict[str, Any]] = []
    mismatched: list[str] = []
    for entry in plan:
        pattern = entry["remote"]
        hits = sorted(path for path in remote_files if Path(path).match(pattern) or path == pattern)
        if not hits:
            fetched.append({**entry, "bytes": None, "sha256": None, "local": None, "fetched_at": None, "sha256_match": None})
            continue
        for index, relpath in enumerate(hits):
            local = root / relpath
            local_sha = sha256_of(local)
            remote_sha = remote_files[relpath]["sha256"]
            match = local_sha == remote_sha
            if not match:
                mismatched.append(f"{relpath}: 원격 {remote_sha[:12]}… ≠ 로컬 {local_sha[:12]}…")
            fetched.append(
                {
                    "name": entry["name"] if index == 0 else f"{entry['name']}-{index}",
                    "kind": entry["kind"],
                    "remote": relpath,
                    "glob": entry.get("glob", False),
                    "required_": entry.get("required_", True),
                    "bytes": int(remote_files[relpath]["bytes"]),
                    "sha256": remote_sha,
                    "local": str(local),
                    "fetched_at": now(),
                    "sha256_match": match,
                }
            )
    if mismatched:
        raise ValueError(f"{manifest['run_id']}: 가져온 파일의 sha256이 원격과 다르다 — {mismatched}")

    kept = [entry for entry in manifest["artifacts"] if names is not None and entry["name"] not in names]
    manifest["artifacts"] = kept + fetched
    manifest["updated_at"] = now()
    note = f"fetch {len(fetched)}개 · sha256 대조 {sum(1 for e in fetched if e['sha256_match'])}/{sum(1 for e in fetched if e['sha256'] is not None)}"
    if missing:
        note += f" · 원격에 없던 것 {missing} (run 상태 {manifest['state']})"
    manifest.setdefault("history", []).append({"at": now(), "state": manifest["state"], "note": note, "command": None})
    if manifest_path is not None:
        save_manifest(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------


def cancel(
    manifest: dict[str, Any],
    backend: Backend,
    *,
    manifest_path: str | Path | None = None,
    confirm_timeout: float = 60.0,
    poll_seconds: float = 1.0,
    signal_name: str = "TERM",
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """멈춤을 요청하고 **확인될 때까지** 기다린다. 확인하지 못하면 `unknown`을 남기고 :class:`CancelNotConfirmed`."""
    handle = Handle.from_manifest(manifest["backend"])
    if handle is None:
        note_state(manifest, "cancelled", note="띄운 적이 없다 — 멈출 것이 없다")
        manifest["exit_reason"] = "cancelled"
        if manifest_path is not None:
            save_manifest(manifest_path, manifest)
        return manifest

    result = backend.stop(handle, signal=signal_name)
    if not result.ok:
        manifest["progress"] = _progress(_remote_state(backend), alive=backend.alive(handle))
        manifest["exit_reason"] = "cancel-command-failed"
        note_state(manifest, "unknown", note=f"정지 명령이 실패했다 ({result.stderr.strip()[:200]}) — 멈췄는지 알 수 없다. 사람이 콘솔에서 확인한다")
        if manifest_path is not None:
            save_manifest(manifest_path, manifest)
        raise CancelNotConfirmed(f"{manifest['run_id']}: 정지 명령이 실패했다 — 상태를 unknown으로 둔다 ({result.stderr.strip()[:200]})")

    deadline = time.monotonic() + float(confirm_timeout)
    confirmed = False
    while True:
        if not backend.alive(handle):
            confirmed = True
            break
        if time.monotonic() >= deadline:
            break
        sleep(poll_seconds)

    if not confirmed:
        manifest["progress"] = _progress(_remote_state(backend), alive=True)
        manifest["exit_reason"] = "cancel-unconfirmed"
        note_state(manifest, "unknown", note=f"정지를 {confirm_timeout} s 동안 확인하지 못했다 — 원격이 아직 돌고 있을 수 있다 (돈이 계속 나간다). 사람이 확인한다")
        if manifest_path is not None:
            save_manifest(manifest_path, manifest)
        raise CancelNotConfirmed(
            f"{manifest['run_id']}: 정지를 확인하지 못했다 ({confirm_timeout} s) — 상태를 unknown으로 둔다. "
            f"원격({manifest['backend'].get('host') or 'local'}:{manifest['backend'].get('unit') or manifest['backend'].get('pid')})을 사람이 확인한다"
        )

    remote = _remote_state(backend)
    manifest["progress"] = _progress(remote, alive=False)
    reason = (remote or {}).get("exit_reason") or "cancelled"
    state = (remote or {}).get("state") or "cancelled"
    manifest["exit_reason"] = reason
    note_state(manifest, "cancelled" if state not in ("completed", "failed") else state, note=f"정지 확인 · 원격 상태 {state}({reason})")
    if manifest_path is not None:
        save_manifest(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def remaining_budget(manifest: dict[str, Any]) -> Budget:
    """부모 run이 쓴 만큼을 뺀 나머지 상한 — 자식 run이 원래 상한을 두 번 쓰지 않게."""
    budget = Budget.from_manifest(manifest)
    spent = budget_spend(budget, float(manifest.get("progress", {}).get("elapsed_seconds") or 0.0))

    def _left(cap: float | None, used: float) -> float | None:
        if cap is None:
            return None
        return max(round(float(cap) - used, 6), 1e-6)

    return Budget(
        hourly_usd=budget.hourly_usd,
        gpus=budget.gpus,
        max_wall_hours=_left(budget.max_wall_hours, spent["wall_hours"]),
        max_gpu_hours=_left(budget.max_gpu_hours, spent["gpu_hours"]),
        max_usd=_left(budget.max_usd, spent["estimated_usd"]),
    )


def registered_checkpoint(manifest: dict[str, Any], local_path: str | Path) -> dict[str, Any]:
    """이 명세에 **등록되고 sha256이 대조된** checkpoint인지 확인한다 (R2의 재개 게이트와 같은 자리).

    등록되지 않았거나, 대조되지 않았거나, 지금 파일의 해시가 등록된 값과 다르면 거절한다.
    """
    target = Path(local_path).resolve()
    for entry in manifest.get("artifacts") or []:
        if entry.get("kind") != "checkpoint" or not entry.get("local"):
            continue
        if Path(entry["local"]).resolve() != target:
            continue
        if not entry.get("sha256_match"):
            raise ValueError(f"{local_path}: 명세에 있지만 sha256이 대조되지 않았다 — 먼저 `fetch`")
        actual = sha256_of(target)
        if actual != entry["sha256"]:
            raise ValueError(f"{local_path}: 등록된 sha256과 지금 파일이 다르다 (등록 {entry['sha256'][:12]}…, 지금 {actual[:12]}…)")
        return entry
    known = [entry.get("local") for entry in manifest.get("artifacts") or [] if entry.get("kind") == "checkpoint" and entry.get("local")]
    raise ValueError(
        f"{local_path}: 이 run의 명세에 등록된 checkpoint가 아니다 — 재개는 **가져와서 대조한** checkpoint에서만 한다 "
        f"(등록된 것: {known or '없다 — 먼저 fetch'})"
    )


def resume(
    parent: dict[str, Any],
    backend: Backend,
    *,
    checkpoint: str | Path,
    run_id: str,
    manifest_path: str | Path,
    config: dict[str, Any],
    config_path: str | Path,
    notes: list[str] | None = None,
    budget: Budget | None = None,
) -> dict[str, Any]:
    """가져온 checkpoint를 원격 `inbox/`로 올리고, 그것에서 잇는 **자식 run**의 명세와 묶음을 만든다.

    상한의 기본값은 부모가 쓴 만큼을 뺀 나머지다(:func:`remaining_budget`) — 재개가 예산을 조용히 새로
    시작하지 않는다. 사람이 예산을 **다시 승인**했으면 `budget`을 주고, 그 사실이 명세의 `notes`에 남는다.
    """
    entry = registered_checkpoint(parent, checkpoint)
    # 자식은 **자기** 원격 디렉터리를 쓴다 — 부모의 state.json·로그·산출물을 덮어쓰지 않는다.
    child_backend = backend.with_remote_dir(str(Path(backend.remote_dir).parent / run_id))
    remote_relpath = f"inbox/{Path(entry['local']).name}"
    child_backend.mkdir("inbox")
    child_backend.push_file(entry["local"], remote_relpath)
    pushed = {item["path"]: item for item in child_backend.inventory([remote_relpath])}
    if pushed.get(remote_relpath, {}).get("sha256") != entry["sha256"]:
        raise ValueError(f"{remote_relpath}: 올린 checkpoint의 원격 sha256이 다르다 (보낸 것 {entry['sha256'][:12]}…)")

    left = remaining_budget(parent)
    extra = [] if budget is None else [f"상한을 다시 승인했다: 남은 예산 {left.as_dict()} → {budget.as_dict()}"]
    manifest = prepare(
        config=config,
        config_path=config_path,
        run_id=run_id,
        backend=child_backend,
        budget=budget or left,
        manifest_path=manifest_path,
        overrides=(parent.get("train_config") or {}).get("overrides") or {},
        price_source=(parent.get("budget") or {}).get("price_source"),
        notes=[*(notes or []), f"{parent['run_id']}의 checkpoint({entry['name']}, sha256 {entry['sha256'][:12]}…)에서 이어간다", *extra],
        resume_from=child_backend.path(remote_relpath),
        parent_run_id=parent["run_id"],
    )
    require_valid(manifest)
    return manifest
