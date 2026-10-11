"""클라우드 단계 장부 — run마다의 상한이 **단계 상한**(첫 단계 $150, 2026-10-11 사용자 승인)을 넘지 않게 (Task cloud1).

장부는 git 밖의 파일 하나다(:data:`DEFAULT_LEDGER` = `artifacts/cloud/runpod-ledger.json`, worktree들이 같은 `artifacts`를 본다).
run 하나가 pod를 만들기 **전에** 그 run의 달러 상한(= 단가 × 마감 시간 — 원격 감시가 그 시각에 pod를 지운다)을 예약하고,
pod를 지운 것이 **확인된 뒤에** 실제로 쓴 만큼(단가 × 만든 때부터 지운 때까지, 실행기의 시계 — 조금 많게 센다)으로 닫는다.

**세는 법** (:meth:`Ledger.committed_usd`): 닫힌 run은 쓴 돈(청구 기록이 있으면 그것과 추정 중 큰 쪽), 열린 run(예약·활성·지움 미확인)은
**상한 전부**다 — 지웠는지 모르는 pod는 상한까지 썼다고 본다. 새 run은 `이미 센 것 + 이 run의 상한 ≤ 단계 상한`일 때만 예약된다
(:class:`PhaseCapExceeded`). 예약·닫기는 파일 잠금(`fcntl.flock`) 안에서 읽고-고치고-쓴다 — 두 실행기가 동시에 남은 돈을 둘 다 가져가지 않게.

단계 상한을 올리는 것은 사람의 일이다: :meth:`Ledger.set_phase_cap`이 이유와 함께 장부의 `history`에 남긴다.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from robo_jev.launch.manifest import REPO, now

__all__ = ["DEFAULT_LEDGER", "DEFAULT_PHASE", "Ledger", "LedgerError", "PhaseCapExceeded"]

DEFAULT_LEDGER = REPO / "artifacts" / "cloud" / "runpod-ledger.json"
#: 첫 클라우드 단계 — 2026-10-11 사용자 승인(통제자 전달): 누적 $150.
DEFAULT_PHASE = {"name": "cloud-phase-1", "cap_usd": 150.0, "approved": "2026-10-11 사용자 승인 (통제자 전달) — 첫 클라우드 단계 누적 상한"}
SCHEMA = "robojev-cloud-ledger-v0"
OPEN = ("reserved", "active", "delete-unconfirmed")


class LedgerError(RuntimeError):
    """장부가 이상하다 (모르는 run, 겹친 run id 등)."""


class PhaseCapExceeded(RuntimeError):
    """이 run의 상한을 더하면 단계 누적이 단계 상한을 넘는다 — 띄우지 않는다."""


class Ledger:
    """단계 장부 하나. 모든 변경은 :meth:`_transaction` 안에서 (잠금 → 읽기 → 고치기 → atomic 쓰기)."""

    def __init__(self, path: str | Path | None = None, *, phase: dict[str, Any] | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_LEDGER
        self.default_phase = dict(phase or DEFAULT_PHASE)

    # -- 파일 --

    def _empty(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "phase": dict(self.default_phase), "runs": [], "history": [{"at": now(), "note": f"장부를 열었다 — 단계 {self.default_phase}"}]}

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise LedgerError(f"{self.path}: 장부 형식이 아니다 (schema {data.get('schema')!r})")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp", delete=False)
        try:
            with handle:
                handle.write(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:  # pragma: no cover - 디스크 오류
            Path(handle.name).unlink(missing_ok=True)
            raise

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        with open(lock, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                data = self.read()
                yield data
                self._write(data)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # -- 세기 --

    @staticmethod
    def _run_cost(run: dict[str, Any]) -> float:
        if run.get("status") in OPEN:
            return float(run.get("cap_usd") or 0.0)
        if run.get("status") == "closed":
            spent = [float(v) for v in (run.get("spent_usd_estimate"), run.get("spent_usd_billed")) if v is not None]
            return max(spent) if spent else float(run.get("cap_usd") or 0.0)
        return 0.0  # released — pod를 만들지 않았다

    def committed_usd(self, data: dict[str, Any] | None = None) -> float:
        data = data if data is not None else self.read()
        return round(sum(self._run_cost(run) for run in data["runs"]), 6)

    def summary(self) -> dict[str, Any]:
        data = self.read()
        committed = self.committed_usd(data)
        cap = float(data["phase"]["cap_usd"])
        return {
            "path": str(self.path), "phase": data["phase"], "committed_usd": committed, "remaining_usd": round(cap - committed, 6),
            "open_runs": [run["run_id"] for run in data["runs"] if run.get("status") in OPEN],
            "runs": [{k: run.get(k) for k in ("run_id", "status", "pod_id", "hourly_usd", "cap_usd", "spent_usd_estimate", "spent_usd_billed")} for run in data["runs"]],
        }

    def remaining_usd(self) -> float:
        data = self.read()
        return round(float(data["phase"]["cap_usd"]) - self.committed_usd(data), 6)

    # -- 바꾸기 --

    def _find(self, data: dict[str, Any], run_id: str) -> dict[str, Any]:
        for run in data["runs"]:
            if run["run_id"] == run_id:
                return run
        raise LedgerError(f"{self.path}: run {run_id!r}이 장부에 없다")

    def check(self, run_id: str, cap_usd: float, *, phase: dict[str, Any] | None = None) -> dict[str, Any]:
        """예약하지 않고 본다 (dry-run): 이 run을 더해도 단계 상한 안인가. 넘으면 :class:`PhaseCapExceeded`."""
        data = self.read()
        return self._admit(data, run_id, cap_usd, phase=phase)

    def _admit(self, data: dict[str, Any], run_id: str, cap_usd: float, *, phase: dict[str, Any] | None) -> dict[str, Any]:
        if phase is not None and (phase.get("name") != data["phase"]["name"] or abs(float(phase["cap_usd"]) - float(data["phase"]["cap_usd"])) > 1e-9):
            raise LedgerError(
                f"명세의 단계 {phase}가 장부 {self.path}의 단계 {data['phase']}와 다르다 — 단계 상한을 바꾸려면 "
                "`launch_run.py ledger --set-phase-cap …`로 이유와 함께 장부에 적는다"
            )
        if any(run["run_id"] == run_id and run.get("status") != "released" for run in data["runs"]):
            raise LedgerError(f"run {run_id!r}은 이미 장부에 있다 — run id는 한 번만 쓴다")
        committed = self.committed_usd(data)
        cap = float(data["phase"]["cap_usd"])
        if float(cap_usd) <= 0:
            raise PhaseCapExceeded(f"{run_id}: 이 run의 달러 상한이 {cap_usd}다 — 구속하는 상한이 있어야 장부에 올린다")
        if committed + float(cap_usd) > cap + 1e-9:
            raise PhaseCapExceeded(
                f"{run_id}: 단계 '{data['phase']['name']}'의 상한 ${cap:.2f} 중 이미 ${committed:.2f}를 셌다(닫힌 run은 쓴 돈, 열린 run은 상한 전부) — "
                f"이 run의 상한 ${float(cap_usd):.2f}를 더하면 ${committed + float(cap_usd):.2f}로 넘는다. 남은 것은 ${cap - committed:.2f}다 "
                "(상한을 줄이거나, 열린 run을 `finish`/`down`으로 닫거나, 사람이 단계 상한을 다시 승인한다)"
            )
        return {"phase": data["phase"], "committed_usd": committed, "cap_usd": float(cap_usd), "after_usd": round(committed + float(cap_usd), 6),
                "remaining_after_usd": round(cap - committed - float(cap_usd), 6)}  # fmt: skip

    def reserve(self, run_id: str, *, cap_usd: float, hourly_usd: float, manifest: str | None, phase: dict[str, Any] | None = None) -> dict[str, Any]:
        """pod를 만들기 **전에** 상한을 예약한다. 넘으면 :class:`PhaseCapExceeded`이고 아무것도 쓰지 않는다."""
        with self._transaction() as data:
            admitted = self._admit(data, run_id, cap_usd, phase=phase)
            data["runs"] = [run for run in data["runs"] if not (run["run_id"] == run_id and run.get("status") == "released")]
            data["runs"].append({
                "run_id": run_id, "manifest": manifest, "provider": "runpod", "status": "reserved", "pod_id": None,
                "hourly_usd": float(hourly_usd), "cap_usd": float(cap_usd), "reserved_at": now(), "created_at": None, "deleted_at": None,
                "spent_usd_estimate": None, "spent_usd_billed": None, "notes": [],
            })  # fmt: skip
            data["history"].append({"at": now(), "note": f"예약 {run_id}: 상한 ${float(cap_usd):.2f} (누적 {admitted['after_usd']:.2f} / {data['phase']['cap_usd']})"})
        return admitted

    def activate(self, run_id: str, *, pod_id: str, created_at: str | None = None) -> None:
        with self._transaction() as data:
            run = self._find(data, run_id)
            run.update(status="active", pod_id=pod_id, created_at=created_at or now())
            data["history"].append({"at": now(), "note": f"pod {pod_id} 생성 ({run_id})"})

    def release(self, run_id: str, *, reason: str) -> None:
        """pod를 만들지 않았다 — 예약을 푼다 (단가 불일치·키 없음·만들기 거절)."""
        with self._transaction() as data:
            run = self._find(data, run_id)
            if run.get("pod_id"):
                raise LedgerError(f"{run_id}: pod {run['pod_id']}가 있었다 — 풀지 않고 close로 닫는다")
            run.update(status="released")
            run["notes"].append(f"{now()} 풀음: {reason}")
            data["history"].append({"at": now(), "note": f"예약 풂 {run_id}: {reason}"})

    def close(self, run_id: str, *, spent_usd_estimate: float, deleted_at: str | None = None, note: str | None = None) -> None:
        """pod를 지운 것이 **확인된 뒤에** 쓴 만큼으로 닫는다."""
        with self._transaction() as data:
            run = self._find(data, run_id)
            run.update(status="closed", deleted_at=deleted_at or now(), spent_usd_estimate=round(float(spent_usd_estimate), 6))
            if note:
                run["notes"].append(f"{now()} {note}")
            data["history"].append({"at": now(), "note": f"닫음 {run_id}: 추정 ${float(spent_usd_estimate):.4f}"})

    def mark_unconfirmed(self, run_id: str, *, note: str) -> None:
        """지우기를 확인하지 못했다 — 열린 채로(상한 전부를 센 채로) 둔다."""
        with self._transaction() as data:
            run = self._find(data, run_id)
            run.update(status="delete-unconfirmed")
            run["notes"].append(f"{now()} {note}")
            data["history"].append({"at": now(), "note": f"지움 미확인 {run_id}: {note}"})

    def record_billed(self, run_id: str, *, billed_usd: float, source: str) -> None:
        """청구 기록(GET /v2/billing/pods)의 실제 금액을 적는다 — 닫힌 run은 추정과 청구 중 큰 쪽으로 센다."""
        with self._transaction() as data:
            run = self._find(data, run_id)
            run["spent_usd_billed"] = round(float(billed_usd), 6)
            run["notes"].append(f"{now()} 청구 ${float(billed_usd):.4f} ({source})")

    def set_phase_cap(self, cap_usd: float, *, reason: str, name: str | None = None) -> None:
        """사람이 단계 상한을 다시 승인했다 — 이유와 함께 남긴다 (조용히 오르지 않는다)."""
        if not reason or not reason.strip():
            raise LedgerError("단계 상한을 바꾸는 이유(--reason)가 있어야 한다")
        with self._transaction() as data:
            before = dict(data["phase"])
            data["phase"]["cap_usd"] = float(cap_usd)
            if name:
                data["phase"]["name"] = name
            data["history"].append({"at": now(), "note": f"단계 상한 변경 {before} → {data['phase']} — 이유: {reason}"})
