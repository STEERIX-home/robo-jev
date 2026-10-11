"""판단 학습 job — `kind: decisions` (Task cloud1).

책상 정리 줄기의 학습(`robo_jev.train`, 러너가 같은 프로세스에서 부른다)과 달리 판단 학습은 **명령줄 도구**를 차례로 돈다:
`scripts/train_decisions.py`(Task v0d)와 `scripts/diagnose_decisions.py`(v0c·v0d). 그 도구들의 CLI는 아직 움직이므로 이 모듈은 그것을
import하지 않는다 — job 정의(YAML)의 **argv 틀**(`{python}`·`{runs}`·`{reports}` 같은 자리 표시)을 원격에서 채워 돌린다. CLI가 조금 바뀌면
YAML만 고친다.

**job 정의** (`configs/cloud/*.yaml`, `kind: robojev-cloud-job-v0`)::

    job:
      kind: decisions
      model: Qwen/Qwen3.5-2B            # 가중치는 원격이 HF에서 받고 artifacts/models/manifest.json의 revision·sha256과 대조한다
      inputs: [<저장소 기준 경로>…]      # **올려 보낸다** — 도착 뒤 sha256 대조 (다른 CPU에서 다시 만들면 끝자리가 다를 수 있다)
      setup:  [{name, argv, fallback?, max_minutes}]   # 환경 맞추기 (uv sync --frozen --group backbone)
      steps:  [{name, argv, max_minutes, optional?, wall_set?, est_minutes?}]
      fetch:  [<원격 run 디렉터리 기준 glob>…]
    provider: {...}   budget: {...}   estimate: {...}

tokenizer(파일 + manifest)와 모델 manifest(`artifacts/models/manifest.json`)는 입력에 **저절로** 더해진다.

**원격 쪽**(:func:`run_job`, `python -m robo_jev.launch.runner --bundle …`가 부른다): 입력 sha256 대조 → setup → steps를 하위 프로세스로
돌린다. 마감은 run 전체에 하나다(pod라면 실행기가 정한 절대 시각 `job.deadline_epoch` — 감시가 pod를 지우기 전에 가져갈 시간을 남긴
것). 단계마다 `min(max_minutes, 남은 시간)`을 주고, 넘기면 SIGTERM → 30 s → SIGKILL(프로세스 묶음 전체). 학습 단계(`wall_set`)에는
`--set max_wall_hours=<준 시간의 90 %>`를 덧붙여 학습기가 **스스로** 체크포인트를 쓰고 멈추게 한다. 하위 프로세스의 환경에서는
`RUNPOD_API_KEY`(pod 범위 키)를 지운다 — 학습 코드는 그 키가 필요 없다.

이 모듈은 torch·yaml을 import 시점에 부르지 않는다 — 원격 러너는 venv가 생기기 **전에** 시스템 파이썬으로 뜬다.
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from robo_jev.launch.manifest import Budget, budget_deadline_hours, now, sha256_of

__all__ = [
    "AUTO_TOKENIZER_FILES",
    "JOB_FILE_KIND",
    "JOB_KIND",
    "JobError",
    "decisions_artifact_plan",
    "expand",
    "format_plan",
    "load_job",
    "plan_job",
    "resolve_job",
    "run_job",
]

JOB_FILE_KIND = "robojev-cloud-job-v0"
JOB_KIND = "decisions"
#: 원격의 venv 파이썬 (setup의 `uv sync`가 만든다).
DEFAULT_PYTHON = "{repo}/.venv/bin/python"
MODEL_MANIFEST = "artifacts/models/manifest.json"
TOKENIZER_ROOT = "artifacts/tokenizers"
AUTO_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
#: 단계를 시작하지 않을 만큼 남은 시간(초) — 시작하자마자 잘릴 단계는 띄우지 않는다.
MIN_STEP_SECONDS = 20.0
#: SIGTERM 뒤 SIGKILL까지.
KILL_GRACE_SECONDS = 30.0
STEP_KEYS = {"name", "argv", "fallback", "max_minutes", "optional", "wall_set", "est_minutes", "env"}
JOB_KEYS = {"kind", "model", "inputs", "setup", "steps", "fetch", "runtime", "code", "deadline_epoch", "deadline_limit"}
TOP_KEYS = {"kind", "name", "description", "job", "provider", "budget", "estimate", "notes"}


class JobError(ValueError):
    """job 정의가 이상하다 — 원격에 보내기 전에 거절한다."""


# --------------------------------------------------------------------------
# job 정의
# --------------------------------------------------------------------------


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise JobError(f"--set {dotted}: {key}는 dict가 아니다")
    node[keys[-1]] = value


def load_job(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """job YAML → 검사한 dict. `overrides`는 점으로 이은 키(`budget.max_usd`, `provider.gpu_type`)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        _set_dotted(data, dotted, value)
    return resolve_job(data)


def _step(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise JobError(f"{where}: dict여야 한다")
    unknown = set(raw) - STEP_KEYS
    if unknown:
        raise JobError(f"{where}: 모르는 키 {sorted(unknown)} (허용 {sorted(STEP_KEYS)})")
    name = str(raw.get("name") or "")
    if not name or not name.replace("-", "").replace("_", "").isalnum():
        raise JobError(f"{where}: name은 영숫자·-·_ 이름이어야 한다 (받은 값 {name!r}) — 로그 파일 이름이 된다")
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, (str, int, float)) for a in argv):
        raise JobError(f"{where}: argv는 비지 않은 문자열 목록이다")
    step = {
        "name": name,
        "argv": [str(a) for a in argv],
        "fallback": [str(a) for a in raw["fallback"]] if raw.get("fallback") else None,
        "max_minutes": float(raw.get("max_minutes") or 30.0),
        "optional": bool(raw.get("optional", False)),
        "wall_set": raw.get("wall_set"),
        "est_minutes": None if raw.get("est_minutes") is None else float(raw["est_minutes"]),
        "env": {str(k): str(v) for k, v in (raw.get("env") or {}).items()},
    }
    if step["max_minutes"] <= 0:
        raise JobError(f"{where}: max_minutes는 양수다")
    return step


def resolve_job(data: dict[str, Any]) -> dict[str, Any]:
    """job 파일 전체(`job`·`provider`·`budget`·`estimate`)를 검사하고 기본값을 채운 새 dict."""
    data = copy.deepcopy(data)
    if data.get("kind") != JOB_FILE_KIND:
        raise JobError(f"kind: {JOB_FILE_KIND!r}여야 한다 (받은 값 {data.get('kind')!r})")
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise JobError(f"모르는 최상위 키 {sorted(unknown)}")
    job = data.get("job") or {}
    unknown = set(job) - JOB_KEYS
    if unknown:
        raise JobError(f"job: 모르는 키 {sorted(unknown)}")
    if job.get("kind") != JOB_KIND:
        raise JobError(f"job.kind: {JOB_KIND!r}여야 한다 (받은 값 {job.get('kind')!r})")
    if not job.get("model"):
        raise JobError("job.model: 기반 모델 id가 있어야 한다 (예: Qwen/Qwen3.5-2B; tiny는 검사용)")
    inputs = job.get("inputs") or []
    if not isinstance(inputs, list) or not all(isinstance(p, str) and p for p in inputs):
        raise JobError("job.inputs: 저장소 기준 경로 목록이다")
    for path in inputs:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise JobError(f"job.inputs: {path} — 저장소 안의 상대 경로여야 한다 (원격의 같은 자리에 놓는다)")
    job["inputs"] = list(dict.fromkeys(inputs))
    job["setup"] = [_step(raw, f"job.setup[{i}]") for i, raw in enumerate(job.get("setup") or [])]
    job["steps"] = [_step(raw, f"job.steps[{i}]") for i, raw in enumerate(job.get("steps") or [])]
    if not job["steps"]:
        raise JobError("job.steps: 단계가 하나도 없다")
    names = [s["name"] for s in job["setup"] + job["steps"]]
    if len(names) != len(set(names)):
        raise JobError(f"단계 이름이 겹친다: {names}")
    job["fetch"] = [str(p) for p in (job.get("fetch") or [])]
    runtime = dict(job.get("runtime") or {})
    runtime.setdefault("python", DEFAULT_PYTHON)
    runtime["env"] = {str(k): str(v) for k, v in (runtime.get("env") or {}).items()}
    job["runtime"] = runtime
    job["code"] = dict(job.get("code") or {"ref": "HEAD"})
    data["job"] = job
    data.setdefault("provider", None)
    data["budget"] = dict(data.get("budget") or {})
    data["estimate"] = dict(data.get("estimate") or {})
    return data


def tokenizer_inputs(repo: Path) -> list[str]:
    """tokenizer 파일과 그 manifest — 원격의 `available_tokenizer()`가 같은 자리에서 찾는다."""
    manifest = repo / TOKENIZER_ROOT / "manifest.json"
    if not manifest.is_file():
        raise JobError(f"{manifest}: tokenizer manifest가 없다 — `scripts/fetch_tokenizer.py`로 먼저 받는다")
    identifier = str(json.loads(manifest.read_text(encoding="utf-8")).get("id") or "")
    out = [f"{TOKENIZER_ROOT}/manifest.json"]
    for name in AUTO_TOKENIZER_FILES:
        if (repo / TOKENIZER_ROOT / identifier / name).is_file():
            out.append(f"{TOKENIZER_ROOT}/{identifier}/{name}")
    if len(out) < 2:
        raise JobError(f"{TOKENIZER_ROOT}/{identifier}/tokenizer.json이 없다")
    return out


def all_inputs(job: dict[str, Any], repo: Path) -> list[str]:
    """올려 보낼 파일 전부: job의 입력 + tokenizer + (실제 모델이면) 모델 manifest."""
    extra = tokenizer_inputs(repo)
    if job["model"] != "tiny":
        extra.append(MODEL_MANIFEST)
    return list(dict.fromkeys([*job["inputs"], *extra]))


def hash_inputs(paths: list[str], repo: Path) -> list[dict[str, Any]]:
    out = []
    for relpath in paths:
        path = repo / relpath
        if not path.is_file():
            raise JobError(f"{relpath}: 입력 파일이 없다 ({path})")
        out.append({"path": relpath, "sha256": sha256_of(path), "bytes": path.stat().st_size})
    return out


def model_entry(model: str, repo: Path) -> dict[str, Any] | None:
    """모델 manifest의 그 모델 항목 (revision·digest·크기) — 원격의 `fetch_backbone --from-manifest`가 대조할 값."""
    if model == "tiny":
        return None
    path = repo / MODEL_MANIFEST
    if not path.is_file():
        raise JobError(f"{path}: 모델 manifest가 없다")
    entry = (json.loads(path.read_text(encoding="utf-8")).get("models") or {}).get(model)
    if not entry:
        raise JobError(f"{model}: {MODEL_MANIFEST}에 항목이 없다 — 고정된 revision·sha256이 없으면 원격이 대조할 수 없다")
    return {"id": model, "revision": entry.get("revision"), "digest": entry.get("digest"), "bytes_total": entry.get("bytes_total"),
            "files": {name: spec.get("sha256") for name, spec in (entry.get("files") or {}).items()}}  # fmt: skip


def decisions_artifact_plan(job: dict[str, Any]) -> list[dict[str, Any]]:
    """가져올 것 — job의 `fetch` 패턴 + 언제나 가져오는 장부(상태·job 보고서·로그)."""
    plan = [
        {"name": "state", "kind": "state", "remote": "state.json", "required_": True},
        {"name": "job-report", "kind": "report", "remote": "job-report.json", "required_": False},
        {"name": "logs", "kind": "log", "remote": "logs/*", "glob": True, "required_": True},
    ]
    seen = {entry["remote"] for entry in plan}
    for index, pattern in enumerate(job.get("fetch") or []):
        if pattern in seen:
            continue
        seen.add(pattern)
        kind = "checkpoint" if pattern.endswith(".pt") else "metrics" if pattern.endswith("metrics.json") else "config" if pattern.endswith(".yaml") else "report"
        plan.append({"name": f"fetch-{index:02d}", "kind": kind, "remote": pattern, "glob": any(ch in pattern for ch in "*?["), "required_": False})
    return plan


# --------------------------------------------------------------------------
# 자리 표시
# --------------------------------------------------------------------------


def expand(argv: list[str], context: dict[str, str]) -> list[str]:
    """`{python}`·`{repo}`·`{run_dir}`·`{runs}`·`{reports}`·`{model}`·`{step}`·`{run_id}`·`{uv}`을 채운다.

    `{python}`처럼 여러 낱말로 풀리는 자리(예: ``env PYTHONPATH=… python``)가 **원소 전체**면 shlex로 나눠 그 자리에 펼친다.
    """
    out: list[str] = []
    for item in argv:
        if item.startswith("{") and item.endswith("}") and item[1:-1] in context and " " in context[item[1:-1]]:
            out.extend(shlex.split(context[item[1:-1]]))
            continue
        try:
            out.append(item.format(**context))
        except (KeyError, IndexError, ValueError) as error:
            raise JobError(f"argv {item!r}: 모르는 자리 표시 ({error}) — 쓸 수 있는 것: {sorted(context)}") from None
    return out


# --------------------------------------------------------------------------
# dry-run
# --------------------------------------------------------------------------


def _minutes(step: dict[str, Any]) -> float:
    return float(step["est_minutes"]) if step.get("est_minutes") is not None else float(step["max_minutes"])


def plan_job(manifest: dict[str, Any], *, ledger: Any = None, lifetime: tuple[float | None, str | None] | None = None) -> dict[str, Any]:
    """명세만으로 할 일·예상 시간·예상 비용·올릴 것·가져올 것 (네트워크도 키도 쓰지 않는다)."""
    job = manifest["job"]
    budget = Budget.from_manifest(manifest)
    provider = manifest.get("provider") or {}
    estimate = (manifest.get("estimate") or {}) if isinstance(manifest.get("estimate"), dict) else {}
    deadline, binding = lifetime if lifetime is not None else budget_deadline_hours(budget)
    boot = float(provider.get("boot_estimate_minutes") or estimate.get("boot_minutes") or (5.0 if provider else 0.0))
    upload_bytes = sum(int(item.get("bytes") or 0) for item in job.get("uploads") or [])
    uplink = float(estimate.get("uplink_mbps") or 50.0)
    upload_minutes = upload_bytes * 8 / (uplink * 1e6) / 60.0
    fetch_minutes = float(estimate.get("fetch_minutes") or 2.0)
    rows = []
    for phase in ("setup", "steps"):
        for step in job.get(phase) or []:
            rows.append({"phase": phase, "name": step["name"], "est_minutes": round(_minutes(step), 2), "max_minutes": step["max_minutes"],
                         "optional": step["optional"], "argv": " ".join(step["argv"])})  # fmt: skip
    work_minutes = sum(row["est_minutes"] for row in rows)
    total_minutes = boot + upload_minutes + work_minutes + fetch_minutes
    hourly = float(budget.hourly_usd)
    cap_usd = None if deadline is None else round(deadline * hourly, 4)
    plan = {
        "run_id": manifest["run_id"],
        "provider": {k: provider.get(k) for k in ("name", "gpu_type", "gpu_count", "cloud", "image", "disk_gb", "volume_gb", "network_volume_id",
                                                   "data_centers", "min_cuda_version")} if provider else None,  # fmt: skip
        "price": {"hourly_usd": hourly, "price_source": manifest["budget"].get("price_source"),
                  "checked_at_launch": bool(provider), "note": "launch가 만들기 전에 카탈로그 단가와, 만든 뒤 pod의 cost와 대조한다 — 다르면 거절"},  # fmt: skip
        "caps": {"max_wall_hours": budget.max_wall_hours, "max_usd": budget.max_usd, "max_gpu_hours": budget.max_gpu_hours,
                 "lifetime_hours": deadline, "binding": binding, "run_cap_usd": cap_usd,
                 "fetch_reserve_minutes": float(provider.get("fetch_reserve_minutes") or 0.0) if provider else 0.0,
                 "delete_margin_minutes": float(provider.get("delete_margin_minutes") or 0.0) if provider else 0.0},  # fmt: skip
        "steps": rows,
        "time": {"boot_minutes": round(boot, 2), "upload_minutes": round(upload_minutes, 2), "work_minutes": round(work_minutes, 2),
                 "fetch_minutes": round(fetch_minutes, 2), "total_minutes": round(total_minutes, 2),
                 "assumption": estimate.get("assumption")},  # fmt: skip
        "cost": {"expected_usd": round(total_minutes / 60.0 * hourly, 4), "worst_case_usd": cap_usd},
        "uploads": {"code": job.get("code"), "files": job.get("uploads") or [], "bytes": upload_bytes},
        "model": job.get("model_entry") or {"id": job.get("model")},
        "fetch": [entry["remote"] for entry in decisions_artifact_plan(job)],
        "warnings": [],
    }
    if deadline is not None and total_minutes > deadline * 60.0:
        plan["warnings"].append(
            f"계획 시간 {total_minutes:.1f}분이 상한 {deadline * 60.0:.1f}분보다 길다 — run은 상한에서 멈춘다(체크포인트·로그는 남고, 남은 단계는 건너뛴다)"
        )
    if ledger is not None and cap_usd is not None:
        try:
            plan["ledger"] = ledger.check(manifest["run_id"], cap_usd, phase=manifest["budget"].get("phase"))
        except Exception as error:  # noqa: BLE001 — dry-run은 거절 이유를 보여 준다
            plan["ledger"] = {"refused": str(error)}
            plan["warnings"].append(f"장부가 이 run을 거절한다: {error}")
    return plan


def format_plan(plan: dict[str, Any]) -> str:
    """사람이 읽는 dry-run."""
    lines = [f"== dry-run {plan['run_id']} (돈을 쓰지 않는다 — 네트워크·키 없음)"]
    if plan.get("provider"):
        p = plan["provider"]
        lines.append(f"pod      {p['gpu_count']}× {p['gpu_type']} · {p['cloud']} · 이미지 {p['image']} · 디스크 {p['disk_gb']} GB"
                     f" · 볼륨 {p.get('volume_gb') or 0} GB · 지역 {p.get('data_centers') or '자동'} · CUDA ≥ {p.get('min_cuda_version') or '아무거나'}")  # fmt: skip
    price, caps = plan["price"], plan["caps"]
    lines.append(f"단가     ${price['hourly_usd']:.4f}/h (출처: {price.get('price_source') or '적지 않음'}) — {price['note']}")
    lines.append(f"상한     벽시계 {caps['max_wall_hours']} h · 달러 {caps['max_usd']} · 수명 {caps['lifetime_hours']} h ({caps['binding']}) → run 상한 ${caps['run_cap_usd']}"
                 f" · 가져갈 몫 {caps['fetch_reserve_minutes']}분 · 지울 여유 {caps['delete_margin_minutes']}분")  # fmt: skip
    if plan.get("ledger"):
        led = plan["ledger"]
        if "refused" in led:
            lines.append(f"장부     거절: {led['refused']}")
        else:
            lines.append(f"장부     단계 {led['phase']['name']} 상한 ${led['phase']['cap_usd']:.2f} · 이미 센 것 ${led['committed_usd']:.2f} · "
                         f"이 run ${led['cap_usd']:.2f} · 뒤에 남는 것 ${led['remaining_after_usd']:.2f}")  # fmt: skip
    t = plan["time"]
    lines.append(f"시간     부팅 {t['boot_minutes']} + 올리기 {t['upload_minutes']} + 일 {t['work_minutes']} + 가져오기 {t['fetch_minutes']} = {t['total_minutes']}분"
                 + (f" (가정: {t['assumption']})" if t.get("assumption") else ""))  # fmt: skip
    lines.append(f"비용     예상 ${plan['cost']['expected_usd']:.2f} · 최악(상한) ${plan['cost']['worst_case_usd']}")
    lines.append("단계")
    for row in plan["steps"]:
        flag = " (선택)" if row["optional"] else ""
        lines.append(f"  {row['phase']:<5} {row['name']:<24} 예상 {row['est_minutes']:>6.1f}분 · 상한 {row['max_minutes']:>5.1f}분{flag}")
    up = plan["uploads"]
    lines.append(f"올리기   코드 {up['code'].get('commit') or up['code'].get('ref')} ({up['code'].get('payload_bytes')} B, sha256 {str(up['code'].get('payload_sha256'))[:12]}…) "
                 f"+ 파일 {len(up['files'])}개 {up['bytes']:,} B — 도착 뒤 sha256 대조")  # fmt: skip
    for item in up["files"]:
        lines.append(f"  {item['path']}  {item['bytes']:,} B  {item['sha256'][:12]}…")
    model = plan["model"]
    lines.append(f"모델     {model.get('id')} @ {model.get('revision')} (지문 {str(model.get('digest'))[:12]}…, {model.get('bytes_total')} B) — 원격이 HF에서 받고 대조")
    lines.append("가져오기 " + " · ".join(plan["fetch"]) + " — 원격 sha256 ↔ 가져와 다시 잰 sha256")
    for warning in plan["warnings"]:
        lines.append(f"⚠ {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 원격 쪽 — 단계 돌리기
# --------------------------------------------------------------------------


def _child_env(extra: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in ("RUNPOD_API_KEY",)}
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.update(extra)
    return env


def _terminate_group(process: subprocess.Popen[Any], grace: float) -> None:
    """하위 프로세스 묶음 전체에 SIGTERM → `grace` 초 → SIGKILL."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_step(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path, timeout: float, stopping: dict[str, Any],
             grace: float = KILL_GRACE_SECONDS, poll: float = 0.5) -> tuple[int | None, str]:  # fmt: skip
    """하위 프로세스 하나(새 세션 — 잘라도 손자까지 함께). 돌려주는 것: (종료 코드, 'ok'|'failed'|'timeout'|'cancelled')."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        handle.write(f"[runner {now()}] $ {shlex.join(argv)}\n".encode())
        handle.flush()
        try:
            process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as error:
            handle.write(f"[runner {now()}] 실행 실패: {error}\n".encode())
            return None, "failed"
        stopping["process"] = process
        started = time.monotonic()
        try:
            while True:
                code = process.poll()
                if code is not None:
                    return code, ("ok" if code == 0 else "failed")
                if stopping.get("term"):
                    _terminate_group(process, grace)
                    return process.returncode, "cancelled"
                if time.monotonic() - started >= timeout:
                    handle.write(f"[runner {now()}] {timeout:.0f} s를 넘겼다 — SIGTERM, {grace:.0f} s 뒤 SIGKILL\n".encode())
                    handle.flush()
                    _terminate_group(process, grace)
                    return process.returncode, "timeout"
                time.sleep(poll)
        finally:
            stopping["process"] = None


def verify_inputs(uploads: list[dict[str, Any]], repo: Path) -> list[str]:
    problems = []
    for item in uploads:
        path = repo / item["path"]
        if not path.is_file():
            problems.append(f"{item['path']}: 원격에 없다 (올리기가 실패했다)")
            continue
        actual = sha256_of(path)
        if actual != item["sha256"]:
            problems.append(f"{item['path']}: sha256이 다르다 (명세 {item['sha256'][:12]}…, 원격 {actual[:12]}…)")
    return problems


def run_job(bundle: str | Path, *, repo: str | Path) -> int:
    """원격에서 job 하나를 끝까지 돌린다. 돌려주는 것은 종료 코드(runner의 EXIT_* 와 같다)."""
    from robo_jev.launch.runner import EXIT_BUDGET, EXIT_CANCELLED, EXIT_ERROR, EXIT_INPUTS, EXIT_OK, RunnerState, _log

    bundle, repo = Path(bundle), Path(repo)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    job = manifest["job"]
    run_dir = bundle.parent
    budget = Budget.from_manifest(manifest)
    if job.get("deadline_epoch"):
        deadline_hours = max(float(job["deadline_epoch"]) - time.time(), 0.0) / 3600.0
        deadline_limit = str(job.get("deadline_limit") or "pod-lifetime")
    else:
        deadline_hours, deadline_limit = budget_deadline_hours(budget)
    phases = [("setup", step) for step in job.get("setup") or []] + [("steps", step) for step in job["steps"]]
    state = RunnerState(run_dir / "state.json", run_id=manifest["run_id"], max_steps=len(phases), budget=budget,
                        deadline_hours=deadline_hours, deadline_limit=deadline_limit)  # fmt: skip
    (run_dir / "started").write_text(now() + "\n", encoding="utf-8")
    _log(f"job {manifest['run_id']} · 단계 {len(phases)}개 · 마감 {deadline_hours} h ({deadline_limit})")

    stopping: dict[str, Any] = {"term": False, "process": None}

    def _on_term(signum: int, _frame: Any) -> None:
        stopping["term"] = True
        _log(f"신호 {signum} — 지금 단계를 멈추고 끝낸다")

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    context = {
        "repo": str(repo), "run_dir": str(run_dir), "runs": str(run_dir / "runs"), "reports": str(run_dir / "reports"),
        "model": str(job["model"]), "run_id": str(manifest["run_id"]), "uv": str((job.get("runtime") or {}).get("uv") or "uv"),
    }  # fmt: skip
    context["python"] = expand([str(job["runtime"]["python"])], context)[0] if "{" in str(job["runtime"]["python"]) else str(job["runtime"]["python"])
    (run_dir / "runs").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    outcome = ("completed", "completed", EXIT_OK, None)

    def _report(final_state: str, reason: str) -> None:
        env_path = run_dir / "reports" / "env.json"
        payload = {
            "run_id": manifest["run_id"], "state": final_state, "exit_reason": reason, "started_at": state.data.get("started_at"),
            "finished_at": now(), "elapsed_seconds": round(state.elapsed, 3), "deadline_hours": deadline_hours, "deadline_limit": deadline_limit,
            "estimated_usd": state.spend()["estimated_usd"], "hourly_usd": budget.hourly_usd, "steps": results,
            "env": json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else None,
        }  # fmt: skip
        target = run_dir / "job-report.json"
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(temp, target)

    problems = verify_inputs(job.get("uploads") or [], repo)
    if problems:
        for line in problems:
            _log(f"입력 대조 실패: {line}")
        state.finish("failed", exit_reason="inputs", error="; ".join(problems)[:2000])
        _report("failed", "inputs")
        (run_dir / "finished").write_text(now() + "\n", encoding="utf-8")
        return EXIT_INPUTS
    _log(f"입력 {len(job.get('uploads') or [])}개 sha256 대조 통과")

    for index, (phase, step) in enumerate(phases):
        remaining = None if deadline_hours is None else deadline_hours * 3600.0 - state.elapsed
        record: dict[str, Any] = {"phase": phase, "name": step["name"], "optional": step["optional"], "status": "skipped",
                                  "started_at": None, "seconds": None, "returncode": None}  # fmt: skip
        if stopping["term"]:
            outcome = ("cancelled", "cancelled", EXIT_CANCELLED, None)
            results.append(record)
            break
        if remaining is not None and remaining < MIN_STEP_SECONDS:
            _log(f"마감까지 {remaining:.0f} s — {step['name']}부터 띄우지 않는다")
            outcome = ("failed", "budget", EXIT_BUDGET, deadline_limit)
            results.append(record)
            break
        allowed = float(step["max_minutes"]) * 60.0 if remaining is None else min(float(step["max_minutes"]) * 60.0, remaining)
        argv = expand(step["argv"], {**context, "step": step["name"]})
        if step.get("wall_set"):
            argv += ["--set", f"{step['wall_set']}={max(allowed * 0.9, 1.0) / 3600.0:.6f}"]
        env = _child_env({**job["runtime"]["env"], **step["env"]})
        log = run_dir / "logs" / f"{index:02d}-{step['name']}.log"
        state.update(step=index, current=step["name"])
        _log(f"[{index + 1}/{len(phases)}] {step['name']} 시작 (상한 {allowed:.0f} s)")
        record["started_at"] = now()
        clock = time.monotonic()
        code, status = run_step(argv, cwd=repo, env=env, log=log, timeout=allowed, stopping=stopping)
        if status == "failed" and step.get("fallback"):
            _log(f"{step['name']} 실패(코드 {code}) — 대체 명령으로 다시")
            remaining = None if deadline_hours is None else deadline_hours * 3600.0 - state.elapsed
            allowed_left = allowed - (time.monotonic() - clock) if remaining is None else min(allowed - (time.monotonic() - clock), remaining)
            if allowed_left >= MIN_STEP_SECONDS:
                code, status = run_step(expand(step["fallback"], {**context, "step": step["name"]}), cwd=repo, env=env, log=log,
                                        timeout=allowed_left, stopping=stopping)  # fmt: skip
                record["fallback_used"] = True
        record.update(status=status, returncode=code, seconds=round(time.monotonic() - clock, 3))
        results.append(record)
        state.update(step=index + 1, current=None, last_step={k: record[k] for k in ("name", "status", "seconds", "returncode")})
        _log(f"[{index + 1}/{len(phases)}] {step['name']} → {status} (코드 {code}, {record['seconds']} s)")
        if status == "cancelled":
            outcome = ("cancelled", "cancelled", EXIT_CANCELLED, None)
            break
        if status == "timeout" and deadline_hours is not None and deadline_hours * 3600.0 - state.elapsed < MIN_STEP_SECONDS:
            outcome = ("failed", "budget", EXIT_BUDGET, deadline_limit)
            break
        if status != "ok" and not step["optional"]:
            outcome = ("failed", f"step:{step['name']}:{status}", EXIT_ERROR, None)
            break
    else:
        outcome = ("completed", "completed", EXIT_OK, None)

    final_state, reason, code, limit = outcome
    for phase, step in phases[len(results):]:
        results.append({"phase": phase, "name": step["name"], "optional": step["optional"], "status": "skipped",
                        "started_at": None, "seconds": None, "returncode": None})  # fmt: skip
    fields: dict[str, Any] = {"step": sum(1 for r in results if r["status"] == "ok")}
    if limit:
        fields["budget_limit"] = limit
    state.finish(final_state, exit_reason=reason, **fields)
    _report(final_state, reason)
    (run_dir / "finished").write_text(now() + "\n", encoding="utf-8")
    _log(f"끝: {final_state}({reason}) · {state.data['estimated_usd']} USD 추정")
    return code
