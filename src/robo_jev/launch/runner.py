"""원격 러너 — 상한을 **스스로** 재는 쪽 (docs/06 Task 6, docs/05 §7).

실행기(`scripts/launch_run.py`)는 묶음을 올리고 이것을 분리 실행으로 띄운 뒤 손을 뗀다. 그 다음부터 상한을
지키는 것은 **여기**다: 실행기가 죽어도, ssh가 끊겨도, 노트북 뚜껑을 닫아도 원격의 이 프로세스가 자기 시계로
벽시계·GPU 시간·USD를 재고 넘으면 **checkpoint를 쓴 뒤** `failed(reason=budget)`으로 끝난다.

두 가지 모드가 있다::

    python -m robo_jev.launch.runner --bundle <run_dir>/bundle      # 학습한다
    python -m robo_jev.launch.runner --inventory <run_dir> --patterns 'runs/*/checkpoint.pt' …   # 목록과 해시

**상태 파일.** `<run_dir>/state.json`을 매 step 뒤에 atomic하게 다시 쓴다. 실행기의 `status`는 그 파일과
프로세스 생사(systemd/kill -0) 둘을 읽어 명세를 갱신한다 — 상태 파일이 `running`인데 프로세스가 없으면
그 run은 `failed(reason=process_vanished)`다(강제 종료된 run이 성공으로 보이지 않게).

**신호.** `SIGTERM`(`systemctl --user stop`이 보내는 것)을 받으면 다음 step 경계에서 checkpoint를 쓰고
`cancelled`로 끝난다. `SIGKILL`은 잡을 수 없다 — 그 경우 마지막 `checkpoint_every` checkpoint와 로그가 남고,
상태는 실행기가 위의 규칙으로 판정한다.

**입력 대조.** 데이터는 묶음에 담기지 않는다(경로와 sha256만). 그러므로 학습을 시작하기 **전에** 명세에 적힌
manifest들의 sha256을 원격 파일에서 다시 재고, 하나라도 다르면 `failed(reason=inputs)`로 끝낸다 — 다른 데이터
위에서 돈 run이 같은 이름으로 남지 않게.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from robo_jev.launch.manifest import (
    Budget,
    budget_deadline_hours,
    budget_spend,
    now,
    save_manifest,
    sha256_of,
)

__all__ = ["BudgetStop", "Cancelled", "RunnerState", "inventory", "main", "run_bundle", "should_stop"]

#: 종료 코드 — systemd의 결과와 함께 로그에 남는다.
EXIT_OK, EXIT_ERROR, EXIT_BUDGET, EXIT_CANCELLED, EXIT_INPUTS = 0, 1, 2, 3, 4


class BudgetStop(RuntimeError):
    """상한에 닿았다. `limit`은 구속한 상한의 이름."""

    def __init__(self, limit: str, spend: dict[str, float]) -> None:
        super().__init__(f"예산 상한 {limit}에 닿았다 ({spend})")
        self.limit = limit
        self.spend = spend


class Cancelled(RuntimeError):
    """SIGTERM을 받았다 (정지 요청)."""


class RunnerState:
    """`state.json`을 쓰는 것 하나. 항상 atomic하게(임시 파일 → rename) 쓴다."""

    def __init__(self, path: str | Path, *, run_id: str, max_steps: int, budget: Budget, deadline_hours: float | None, deadline_limit: str | None) -> None:
        self.path = Path(path)
        self.started = time.time()
        self.monotonic = time.perf_counter()
        self.data: dict[str, Any] = {
            "run_id": run_id,
            "state": "running",
            "step": 0,
            "max_steps": int(max_steps),
            "loss": None,
            "elapsed_seconds": 0.0,
            "wall_hours": 0.0,
            "gpu_hours": 0.0,
            "estimated_usd": 0.0,
            "seconds_per_step": None,
            "budget": budget.as_dict(),
            "deadline_hours": deadline_hours,
            "deadline_limit": deadline_limit,
            "started_at": now(),
            "updated_at": now(),
            "finished_at": None,
            "exit_reason": None,
            "budget_limit": None,
            "pid": os.getpid(),
            "run_dir": None,
            "checkpoint": None,
            "error": None,
        }
        self.budget = budget
        self.write()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.monotonic

    def spend(self) -> dict[str, float]:
        return budget_spend(self.budget, self.elapsed)

    def update(self, **fields: Any) -> None:
        spend = self.spend()
        self.data.update(
            elapsed_seconds=round(self.elapsed, 3),
            wall_hours=round(spend["wall_hours"], 6),
            gpu_hours=round(spend["gpu_hours"], 6),
            estimated_usd=round(spend["estimated_usd"], 6),
            updated_at=now(),
        )
        step = int(fields.get("step") or self.data.get("step") or 0)
        if step:
            self.data["seconds_per_step"] = round(self.elapsed / step, 3)
        self.data.update(fields)
        self.write()

    def finish(self, state: str, *, exit_reason: str, **fields: Any) -> None:
        self.update(state=state, exit_reason=exit_reason, finished_at=now(), **fields)

    def write(self) -> None:
        # save_manifest는 run manifest 스키마를 검사한다 — 이것은 다른 파일이므로 검사 없이 같은 atomic 쓰기만 쓴다.
        save_manifest(self.path, self.data, check=False)


def should_stop(wall_hours: float, deadline_hours: float | None, *, term: bool) -> str | None:
    """step 경계에서 멈춰야 하는가 — `"budget"` · `"cancelled"` · `None`.

    예산이 먼저다: 정지 신호와 예산이 같이 왔으면 **예산**으로 기록한다(돈 때문에 끝난 run을 취소로 적지 않는다).
    """
    if deadline_hours is not None and wall_hours >= deadline_hours:
        return "budget"
    if term:
        return "cancelled"
    return None


def _log(message: str) -> None:
    """stdout으로. systemd가 `StandardOutput=append:`로 파일에 붙이고, nohup이면 nohup이 붙인다."""
    print(f"[runner {now()}] {message}", flush=True)


# --------------------------------------------------------------------------
# 목록과 해시
# --------------------------------------------------------------------------


def inventory(root: str | Path, patterns: list[str]) -> list[dict[str, Any]]:
    """`root` 아래에서 패턴에 맞는 파일의 상대 경로·바이트·sha256. 원격에서 돌고 결과는 JSON으로 나간다."""
    base = Path(root)
    found: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            relpath = str(path.relative_to(base))
            if relpath in found:
                continue
            found[relpath] = {"path": relpath, "bytes": path.stat().st_size, "sha256": sha256_of(path), "pattern": pattern}
    return [found[key] for key in sorted(found)]


def _matches(relpath: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(relpath, pattern) for pattern in patterns)


# --------------------------------------------------------------------------
# 학습
# --------------------------------------------------------------------------


def _resolve_paths(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    """설정 안의 상대 경로를 **원격 저장소** 기준으로 푼다 (묶음에는 경로가 그대로 들어 있다)."""
    for entry in config.get("dataset_manifests") or []:
        path = Path(entry["path"])
        entry["path"] = str(path if path.is_absolute() else repo / path)
    for key in ("model_config", "tokenizer"):
        value = config.get(key)
        if isinstance(value, str) and value not in ("whitespace",) and (repo / value).exists():
            config[key] = str(repo / value)
    return config


def check_inputs(manifest: dict[str, Any], config: dict[str, Any], repo: Path) -> list[str]:
    """명세가 적은 sha256과 원격 파일의 실제 sha256을 견준다. 다른 것들의 이름을 돌려준다 (빈 목록 = 같다).

    짝은 파일 **이름**이 아니라 명세에 적힌 **경로 그대로** 짓는다 — 데이터 manifest의 이름은 흔히 둘 다
    `manifest.json`이라(로봇 하나, 비로봇 하나) 이름으로 짝지으면 서로의 해시와 견주게 된다. 그러므로 이
    함수는 묶음에 담긴 **푸는 전** 설정을 받고, 경로는 여기서 `repo` 기준으로 푼다.
    """
    problems: list[str] = []
    recorded = {str(entry["path"]): entry for entry in manifest.get("dataset_manifests") or []}
    for entry in config.get("dataset_manifests") or []:
        written = str(entry["path"])
        spec = recorded.get(written)
        if spec is None:
            problems.append(f"{written}: 명세에 없는 데이터 manifest다 (명세: {sorted(recorded)})")
            continue
        path = Path(written)
        path = path if path.is_absolute() else repo / path
        if not path.is_file():
            problems.append(f"{path}: 원격에 없다 (데이터는 묶음에 담기지 않는다 — 상자에 먼저 있어야 한다)")
            continue
        actual = sha256_of(path)
        if actual != spec["sha256"]:
            problems.append(f"{written}: sha256이 다르다 (명세 {spec['sha256'][:12]}…, 원격 {actual[:12]}…)")
    return problems


def run_bundle(bundle: str | Path, *, repo: str | Path | None = None) -> int:
    """묶음 하나를 돌린다. 돌려주는 것은 종료 코드다."""
    bundle = Path(bundle)
    repo = Path(repo or Path.cwd())
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((bundle / "config.yaml").read_text(encoding="utf-8")) or {}
    run_dir = bundle.parent
    budget = Budget.from_manifest(manifest)
    deadline_hours, deadline_limit = budget_deadline_hours(budget)

    state = RunnerState(
        run_dir / "state.json",
        run_id=manifest["run_id"],
        max_steps=int(config.get("max_steps") or manifest["max_steps"]),
        budget=budget,
        deadline_hours=deadline_hours,
        deadline_limit=deadline_limit,
    )
    _log(f"run {manifest['run_id']} · 상한 {budget.as_dict()} · 마감 {deadline_hours} h ({deadline_limit})")

    # 대조는 묶음에 담긴 **그대로의** 경로로 한다(명세와 같은 글자) — 그 다음에 원격 기준으로 푼다.
    problems = check_inputs(manifest, config, repo)
    config = _resolve_paths(dict(config), repo)
    # 산출물은 **run 디렉터리 안에** 남는다 — 회수가 원격 뿌리 하나만 보면 되게.
    config["artifacts_dir"] = str(run_dir / "runs")
    config["run_id"] = manifest["run_id"]

    if problems:
        for line in problems:
            _log(f"입력 대조 실패: {line}")
        state.finish("failed", exit_reason="inputs", error="; ".join(problems)[:2000])
        return EXIT_INPUTS

    stopping = {"term": False}

    def _on_term(signum: int, _frame: Any) -> None:
        stopping["term"] = True
        _log(f"신호 {signum} — 다음 step 경계에서 checkpoint를 쓰고 멈춘다")

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    from robo_jev.train import Trainer  # torch를 여기서 부른다 (inventory 모드는 torch 없이 돈다)

    # 상한은 학습기에도 건다 — **구간 경계**에서도 멈추므로 긴 step 하나가 마감을 크게 넘기지 않는다.
    # 학습기의 시계는 자기가 만들어진 때부터 재므로, 여기서 **이미 쓴 시간을 빼고** 남은 만큼만 준다
    # (그렇지 않으면 적재·tokenize에 쓴 시간이 예산 밖으로 새어 나간다).
    if deadline_hours is not None:
        left = max(deadline_hours - state.spend()["wall_hours"], 1e-9)
        existing = config.get("max_wall_hours")
        config["max_wall_hours"] = left if existing is None else min(float(existing), left)

    exit_code = EXIT_OK
    trainer = None
    try:
        trainer = Trainer(config, resume=config.get("resume"))
        # 설정과 (빈) 지표를 **시작하자마자** 디스크에 남긴다 — 프로세스가 강제 종료돼도
        # "이 run이 무엇으로 돌았는가"가 남는다 (docs/06 Task 6의 영속 보존).
        trainer.write_metrics()
        state.update(run_dir=str(trainer.run_dir))
        _log(f"run 디렉터리 {trainer.run_dir} · 설정·지표 자리 기록")

        def _hook(trainer_: Any, metrics: dict[str, Any]) -> None:
            state.update(step=int(trainer_.step), loss=metrics.get("loss"))
            spend = state.spend()
            verdict = should_stop(spend["wall_hours"], deadline_hours, term=stopping["term"])
            if verdict == "budget":
                _log(f"상한 {deadline_limit}: {spend} ≥ 마감 {deadline_hours} h — checkpoint를 쓰고 멈춘다")
                raise BudgetStop(str(deadline_limit), spend)
            if verdict == "cancelled":
                raise Cancelled()

        trainer.step_hook = _hook
        result = trainer.run()
        checkpoint = result["checkpoint"]
        spend = state.spend()
        # 학습기가 자기 `max_wall_hours`로 구간 경계에서 멈췄을 수도 있다 — 그것도 예산 초과다.
        if result["status"] != "completed" and deadline_hours is not None and spend["wall_hours"] >= deadline_hours * 0.999:
            _log(f"상한 {deadline_limit}(구간 경계): {spend}")
            state.finish("failed", exit_reason="budget", budget_limit=deadline_limit, step=int(result["step"]), checkpoint=checkpoint)
            exit_code = EXIT_BUDGET
        elif result["status"] != "completed":
            state.finish("failed", exit_reason="interrupted", step=int(result["step"]), checkpoint=checkpoint)
            exit_code = EXIT_ERROR
        else:
            state.finish("completed", exit_reason="completed", step=int(result["step"]), checkpoint=checkpoint,
                         loss=result["metrics"]["summary"]["last_loss"])  # fmt: skip
            _log(f"완료 · step {result['step']} · {state.data['estimated_usd']} USD 추정")
    except BudgetStop as stop:
        checkpoint = _save_after_stop(trainer)
        state.finish("failed", exit_reason="budget", budget_limit=stop.limit, checkpoint=checkpoint,
                     step=int(getattr(trainer, "step", 0)))  # fmt: skip
        exit_code = EXIT_BUDGET
    except Cancelled:
        checkpoint = _save_after_stop(trainer)
        state.finish("cancelled", exit_reason="cancelled", checkpoint=checkpoint, step=int(getattr(trainer, "step", 0)))
        _log("정지 요청으로 끝났다 (checkpoint 저장)")
        exit_code = EXIT_CANCELLED
    except BaseException as exc:  # 어떤 실패도 상태로 남긴다 — 조용히 사라지지 않게
        _log("실패:\n" + traceback.format_exc())
        state.finish("failed", exit_reason="error", error=f"{type(exc).__name__}: {exc}"[:2000], step=int(getattr(trainer, "step", 0)))
        exit_code = EXIT_ERROR
    finally:
        if trainer is not None:
            trainer.close()
    return exit_code


def _save_after_stop(trainer: Any) -> str | None:
    """멈출 때의 저장 — checkpoint와 metrics를 남긴다. 저장이 실패해도 상태 기록은 계속한다."""
    if trainer is None:
        return None
    try:
        path = trainer.save()
        trainer.write_metrics()
        _log(f"checkpoint 저장 {path}")
        return str(path)
    except BaseException as exc:  # pragma: no cover - 디스크 오류
        _log(f"checkpoint 저장 실패: {type(exc).__name__}: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robo_jev.launch.runner", description="원격 러너 — 상한을 스스로 재고 상태를 남긴다")
    parser.add_argument("--bundle", default=None, help="run 묶음 디렉터리 (<run_dir>/bundle)")
    parser.add_argument("--repo", default=None, help="원격 저장소 체크아웃 (기본: 현재 디렉터리)")
    parser.add_argument("--inventory", default=None, help="이 디렉터리 아래 파일의 상대 경로·바이트·sha256을 JSON으로")
    parser.add_argument("--patterns", nargs="*", default=["**/*"], help="--inventory의 glob 패턴")
    args = parser.parse_args(argv)
    if args.inventory:
        print(json.dumps(inventory(args.inventory, list(args.patterns)), ensure_ascii=False))
        return 0
    if not args.bundle:
        parser.error("--bundle 또는 --inventory가 필요하다")
    return run_bundle(args.bundle, repo=args.repo)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
