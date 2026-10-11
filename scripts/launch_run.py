"""실행기 CLI — `prepare · launch · status · fetch · cancel · resume` (docs/06 Task 6).

한 run을 **명세된·상한 있는·감사 가능한** 것으로 만들고, 다른 상자로 보내고, 가져온다. 이 스크립트는 얇다 —
일은 :mod:`robo_jev.launch.launcher` 가 하고 여기서는 인자를 읽어 백엔드를 만들고 명세를 읽고 쓴다.

localhost 왕복(인수, Stage B)::

    uv run python scripts/launch_run.py prepare --config configs/train/tiny_cpu.yaml \
        --run-id r3b-roundtrip --manifest artifacts/scratch/r3b/roundtrip/run.json \
        --backend ssh --host 127.0.0.1 --port 2222 --user "$USER" \
        --remote-dir "$PWD/artifacts/scratch/r3b/remote/r3b-roundtrip" --remote-repo "$PWD" \
        --remote-python "$PWD/.venv/bin/python" --gpus 0 --hourly-usd 0 --max-wall-hours 0.25
    uv run python scripts/launch_run.py launch  --manifest artifacts/scratch/r3b/roundtrip/run.json
    uv run python scripts/launch_run.py status  --manifest artifacts/scratch/r3b/roundtrip/run.json
    uv run python scripts/launch_run.py fetch   --manifest artifacts/scratch/r3b/roundtrip/run.json \
        --dest artifacts/scratch/r3b/roundtrip/fetched

**자격 증명은 인자로도 받지 않는다** — 개인키는 `--identity <경로>`, known_hosts는 `--known-hosts <경로>`이고
명세에는 경로만 적힌다. 암호·토큰을 받는 자리는 없다.

**RunPod에서 판단 학습 (Task cloud1)** — job 정의(`configs/cloud/*.yaml`)에서 명세를 만들고, dry-run을 보고, `--yes`로 띄운다::

    uv run python scripts/launch_run.py prepare --job configs/cloud/decisions-v0-smoke-h100.yaml --run-id <id> --manifest artifacts/cloud/runs/<id>.json
    uv run python scripts/launch_run.py dry-run --manifest artifacts/cloud/runs/<id>.json      # 네트워크·키 없음
    uv run python scripts/launch_run.py launch  --manifest artifacts/cloud/runs/<id>.json --yes # 만들기 → 돌리기 → 가져오기 → 지우기
    uv run python scripts/launch_run.py finish  --manifest …   # 중단·--no-watch 뒤: 가져오고 지운다
    uv run python scripts/launch_run.py down    --manifest …   # 지금 지운다
    uv run python scripts/launch_run.py pods                   # 읽기 전용: 내 pod·볼륨 (남은 것이 없는지)
    uv run python scripts/launch_run.py prices --gpu "NVIDIA H100 80GB HBM3"   # 읽기 전용: 카탈로그 단가
    uv run python scripts/launch_run.py ledger                 # 단계 장부 ($150)

API 키는 `~/.config/robo-jev/runpod.env`(600)에서만 읽는다 — 인자·환경 변수로는 받지 않고 어디에도 찍지 않는다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # `uv run python scripts/…`로도 돌게
    sys.path.insert(0, str(REPO / "src"))

from robo_jev.launch.launcher import (  # noqa: E402
    CancelNotConfirmed,
    SpendNotObserved,
    UncappedRun,
    cancel,
    fetch,
    launch,
    prepare,
    prepare_job,
    resume,
    status,
)
from robo_jev.launch.manifest import Budget, budget_deadline_hours, load_manifest, now  # noqa: E402
from robo_jev.launch.providers.base import Backend  # noqa: E402
from robo_jev.launch.providers.local import LocalBackend  # noqa: E402
from robo_jev.launch.providers.ssh import DEFAULT_SSH_OPTIONS, SshBackend  # noqa: E402

__all__ = ["backend_from_manifest", "build_parser", "load_train_config", "main"]

COMMANDS = ("prepare", "launch", "status", "fetch", "cancel", "resume", "dry-run", "finish", "down", "pods", "prices", "ledger")


def load_train_config(path: str | Path, *, mode: str | None = None, dataset: str = "d1", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """학습 설정 YAML → Trainer 설정. `extends`·`modes`는 `scripts/adapt_readout.py`의 것을 **그대로** 쓴다.

    두 벌로 갈라 두면 실행기가 보내는 설정과 사람이 직접 돌린 설정이 조용히 달라질 수 있다.
    """
    spec = importlib.util.spec_from_file_location("adapt_readout", REPO / "scripts" / "adapt_readout.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["adapt_readout"] = module
    spec.loader.exec_module(module)
    if mode is not None:
        return module.load_train_config(path, mode=mode, dataset=dataset, overrides=overrides or {})
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "extends" in config or "modes" in config:
        raise SystemExit(
            f"{path}: 이 설정은 `extends`/`modes`를 쓴다 — `--mode t0|lora|t1`을 줘야 어떤 판을 보낼지 정해진다 "
            "(모드를 고르지 않고 조용히 하나를 택하지 않는다)"
        )
    manifests = []
    for entry in config.get("dataset_manifests") or []:
        entry = dict(entry) if isinstance(entry, dict) else {"path": entry}
        manifest_path = Path(entry["path"])
        entry["path"] = str(manifest_path if manifest_path.is_absolute() else REPO / manifest_path)
        manifests.append(entry)
    if manifests:
        config["dataset_manifests"] = manifests
    single = config.get("dataset_manifest")
    if single:
        path_single = Path(single)
        config["dataset_manifest"] = str(path_single if path_single.is_absolute() else REPO / path_single)
    config.update(overrides or {})
    return config


def backend_from_manifest(manifest: dict[str, Any], *, transport: Any = None) -> Backend:
    """명세의 `backend` 블록에서 백엔드를 되살린다 — 두 번째 명령이 첫 번째와 같은 상자를 본다."""
    block = manifest["backend"]
    common = {
        "remote_dir": block["remote_dir"],
        "remote_repo": block.get("remote_repo") or str(REPO),
        "remote_python": block.get("remote_python") or sys.executable,
        "gpus": int(block.get("gpus") or 0),
    }
    if block["kind"] == "local":
        return LocalBackend(**common)
    return SshBackend(
        host=block["host"],
        user=block.get("user"),
        port=block.get("port"),
        identity_path=block.get("identity_path"),
        known_hosts=block.get("known_hosts"),
        ssh_options=tuple(block.get("ssh_options") or DEFAULT_SSH_OPTIONS),
        transport=transport,
        **common,
    )


def _backend_from_args(args: argparse.Namespace) -> Backend:
    remote_dir = args.remote_dir or str(REPO / "artifacts" / "scratch" / "launch" / args.run_id)
    common = {
        "remote_dir": remote_dir,
        "remote_repo": args.remote_repo or str(REPO),
        "remote_python": args.remote_python or sys.executable,
        "gpus": int(args.gpus),
    }
    if args.backend == "local":
        return LocalBackend(**common)
    if not args.host:
        raise SystemExit("--backend ssh에는 --host가 필요하다")
    return SshBackend(
        host=args.host,
        user=args.user,
        port=args.port,
        identity_path=args.identity,
        known_hosts=args.known_hosts,
        ssh_options=tuple(args.ssh_option) if args.ssh_option else DEFAULT_SSH_OPTIONS,
        **common,
    )


def _overrides(pairs: list[str], parser: argparse.ArgumentParser) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for assignment in pairs:
        key, separator, value = assignment.partition("=")
        if not separator:
            parser.error(f"--set은 KEY=VALUE 꼴이어야 한다: {assignment!r}")
        out[key] = yaml.safe_load(value)
    return out


def _print(manifest: dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(manifest, ensure_ascii=False, indent=1))
        return
    budget = manifest["budget"]
    deadline, limit = budget_deadline_hours(Budget.from_manifest(manifest))
    progress = manifest.get("progress") or {}
    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "state": manifest["state"],
                "exit_reason": manifest.get("exit_reason"),
                "budget_limit": manifest.get("budget_limit"),
                "step": progress.get("step"),
                "max_steps": manifest["max_steps"],
                "loss": progress.get("loss"),
                "elapsed_seconds": progress.get("elapsed_seconds"),
                "estimated_usd": progress.get("estimated_usd"),
                "alive": progress.get("alive"),
                "deadline_hours": deadline,
                "binding_limit": limit,
                "hourly_usd": budget["hourly_usd"],
            },
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripts/launch_run.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="명세를 만들고 묶음을 정리한다 (원격을 건드리지 않는다)")
    prep.add_argument("--config", default=None, help="학습 설정 YAML (책상 정리 줄기의 학습)")
    prep.add_argument("--job", default=None, help="판단 학습 job 정의 (configs/cloud/*.yaml) — --config 대신")
    prep.add_argument("--code-ref", dest="code_ref", default=None, help="pod에 보낼 코드의 git ref (기본: job의 code.ref, 없으면 HEAD)")
    prep.add_argument("--allow-dirty", dest="allow_dirty", action="store_true", help="고치는 중인 파일이 있어도 (pod에는 커밋된 것만 간다)")
    prep.add_argument("--ledger", default=None, help="단계 장부 경로 (기본 artifacts/cloud/runpod-ledger.json)")
    prep.add_argument("--mode", default=None, help="`modes:`가 있는 설정의 모드 (t0|lora|t1)")
    prep.add_argument("--dataset", default="d1", help="adapt_readout의 --dataset (모드가 있을 때만)")
    prep.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="학습 설정 덮어쓰기 (YAML로 읽는다)")
    prep.add_argument("--run-id", dest="run_id", required=True)
    prep.add_argument("--manifest", required=True, help="명세를 쓸 경로 (묶음은 그 옆 bundle/)")
    prep.add_argument("--backend", choices=("ssh", "local"), default="ssh")
    prep.add_argument("--host", default=None)
    prep.add_argument("--user", default=None)
    prep.add_argument("--port", type=int, default=None)
    prep.add_argument("--identity", default=None, help="개인키 **경로** (내용은 저장하지 않는다)")
    prep.add_argument("--known-hosts", dest="known_hosts", default=None)
    prep.add_argument("--ssh-option", dest="ssh_option", action="append", default=[], help="ssh에 그대로 넘길 인자 (여러 번)")
    prep.add_argument("--remote-dir", dest="remote_dir", default=None, help="원격 run 디렉터리")
    prep.add_argument("--remote-repo", dest="remote_repo", default=None, help="원격 저장소 체크아웃")
    prep.add_argument("--remote-python", dest="remote_python", default=None, help="원격 파이썬 (미리 준비된 venv, 또는 'uv run python')")
    prep.add_argument("--gpus", type=int, default=0)
    prep.add_argument("--hourly-usd", dest="hourly_usd", type=float, default=None, help="노드 시간 단가 (USD/h). 0이면 비용 상한을 걸 수 없다 (--max-usd와 함께 주면 거절한다)")
    prep.add_argument("--max-wall-hours", dest="max_wall_hours", type=float, default=None)
    prep.add_argument("--max-gpu-hours", dest="max_gpu_hours", type=float, default=None)
    prep.add_argument("--max-usd", dest="max_usd", type=float, default=None)
    prep.add_argument(
        "--no-cap", dest="no_cap", action="store_true",
        help="구속하는 상한 없이 돌리겠다고 **이름 붙여** 말한다 (기본은 거절이다). 그 사실이 명세의 notes와 prepared 줄에 남는다",
    )  # fmt: skip
    prep.add_argument("--price-source", dest="price_source", default=None, help="단가의 출처 (docs/05 §5 표 / 콘솔)")
    prep.add_argument("--seconds-per-step", dest="seconds_per_step", type=float, default=None, help="예상 비용 계산에 쓸 step당 초")
    prep.add_argument("--assumption", default=None, help="그 예상이 어떤 가정 위의 값인지")
    prep.add_argument("--note", action="append", default=[], help="명세에 남길 사람용 주석 (여러 번)")

    run = sub.add_parser("launch", help="묶음을 올리고 원격에서 분리 실행한다")
    run.add_argument("--manifest", required=True)
    run.add_argument("--force", action="store_true", help="running이어도 다시 띄운다")

    stat = sub.add_parser("status", help="원격 상태·진행을 읽어 명세를 갱신한다")
    stat.add_argument("--manifest", required=True)
    stat.add_argument("--json", dest="as_json", action="store_true")
    stat.add_argument("--wait", action="store_true", help="끝날 때까지 기다린다")
    stat.add_argument("--timeout", type=float, default=3600.0)
    stat.add_argument("--poll", type=float, default=5.0)

    get = sub.add_parser("fetch", help="산출물을 가져와 sha256을 대조한다")
    get.add_argument("--manifest", required=True)
    get.add_argument("--dest", required=True)
    get.add_argument("--name", action="append", default=[], help="가져올 산출물 이름 (기본: 전부)")
    get.add_argument("--strict", dest="strict", action="store_true", default=None, help="없는 산출물이 있으면 실패한다 (기본: completed인 run에서만)")
    get.add_argument("--allow-missing", dest="strict", action="store_false", help="없는 산출물을 적고 넘어간다 (강제 종료된 run의 부분 회수)")

    stop = sub.add_parser("cancel", help="멈춤을 요청하고 확인한다 (확인 못 하면 unknown)")
    stop.add_argument("--manifest", required=True)
    stop.add_argument("--confirm-timeout", dest="confirm_timeout", type=float, default=60.0)
    stop.add_argument("--signal", default="TERM", help="TERM(기본) 또는 KILL")

    again = sub.add_parser("resume", help="가져온 checkpoint에서 잇는 자식 run을 만든다")
    again.add_argument("--manifest", required=True, help="부모 run의 명세")
    again.add_argument("--from", dest="checkpoint", required=True, help="가져온 checkpoint (부모 명세에 등록된 것만)")
    again.add_argument("--run-id", dest="run_id", required=True)
    again.add_argument("--out", required=True, help="자식 명세를 쓸 경로")
    again.add_argument("--config", default=None, help="학습 설정 (기본: 부모가 쓴 것)")
    again.add_argument("--mode", default=None)
    again.add_argument("--dataset", default="d1")
    again.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    again.add_argument("--note", action="append", default=[])
    again.add_argument("--launch", action="store_true", help="명세를 만든 뒤 바로 띄운다")
    # 기본은 **남은** 예산이다. 아래를 주면 사람이 예산을 다시 승인한 것이고 그 사실이 명세에 남는다.
    again.add_argument("--hourly-usd", dest="hourly_usd", type=float, default=None)
    again.add_argument("--gpus", type=int, default=None)
    again.add_argument("--max-wall-hours", dest="max_wall_hours", type=float, default=None)
    again.add_argument("--max-gpu-hours", dest="max_gpu_hours", type=float, default=None)
    again.add_argument("--max-usd", dest="max_usd", type=float, default=None)
    plan = sub.add_parser("dry-run", help="명세만으로 할 일·예상 시간·예상 비용·가져올 파일 (네트워크·키 없음)")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--ledger", default=None)
    plan.add_argument("--json", dest="as_json", action="store_true")

    run.add_argument("--yes", action="store_true", help="(pod) 예상 비용을 봤고 띄운다 — 없으면 dry-run만 찍고 끝난다")
    run.add_argument("--dest", default=None, help="(pod) 결과를 가져올 곳 (기본: 명세 옆 <이름>-fetched/)")
    run.add_argument("--no-watch", dest="watch", action="store_false", help="(pod) 러너를 띄우고 손을 뗀다 — pod 안의 감시가 마감에 지운다")
    run.add_argument("--keep-on-interrupt", dest="keep_on_interrupt", action="store_true", help="(pod) 중단해도 지우지 않는다 (감시 마감에 지워진다)")
    run.add_argument("--ledger", default=None)
    run.add_argument("--key-file", dest="key_file", default=None, help="(pod) API 키 파일 (기본 ~/.config/robo-jev/runpod.env)")

    done = sub.add_parser("finish", help="(pod) 다시 붙어 기다리고 가져온 뒤 지운다")
    done.add_argument("--manifest", required=True)
    done.add_argument("--dest", required=True)
    done.add_argument("--no-wait", dest="wait", action="store_false")
    done.add_argument("--ledger", default=None)
    done.add_argument("--key-file", dest="key_file", default=None)

    kill = sub.add_parser("down", help="(pod) 지금 지운다 — 가져오지 않는다")
    kill.add_argument("--manifest", required=True)
    kill.add_argument("--ledger", default=None)
    kill.add_argument("--key-file", dest="key_file", default=None)

    pods = sub.add_parser("pods", help="(읽기 전용) 내 pod·네트워크 볼륨 — 남은 자원이 없는지")
    pods.add_argument("--key-file", dest="key_file", default=None)
    pods.add_argument("--json", dest="as_json", action="store_true")

    price = sub.add_parser("prices", help="(읽기 전용) 카탈로그 단가·재고")
    price.add_argument("--gpu", action="append", default=[], help="GPU id (여러 번; 기본: H100 PCIe·SXM·NVL, H200)")
    price.add_argument("--key-file", dest="key_file", default=None)

    book = sub.add_parser("ledger", help="단계 장부 ($150) 보기·상한 바꾸기")
    book.add_argument("--ledger", default=None)
    book.add_argument("--set-phase-cap", dest="set_phase_cap", type=float, default=None)
    book.add_argument("--reason", default=None)
    return parser


def _ledger(path: str | None) -> Any:
    from robo_jev.launch.ledger import Ledger

    return Ledger(path)


def _runpod(args: argparse.Namespace, *, read_only: bool) -> Any:
    """키 파일에서 키를 읽어 공급자를 만든다. 키가 없거나 권한이 넓으면 이유를 말하고 끝난다 (pod를 만들지 않는다)."""
    from robo_jev.launch.providers.runpod import MissingApiKey, RunPodClient, RunPodProvider, UnsafeKeyFile, load_api_key

    try:
        key = load_api_key(getattr(args, "key_file", None))
    except (MissingApiKey, UnsafeKeyFile) as exc:
        raise SystemExit(f"[launch_run] 키 거절: {exc}") from None
    return RunPodProvider(RunPodClient(key, read_only=read_only))


def _pod_backend(view: Any, manifest: dict[str, Any]) -> Backend:
    block = manifest["backend"]
    return SshBackend(
        host=view.host, port=view.port, user=view.username or block.get("user") or "root", remote_dir=block["remote_dir"],
        remote_repo=block["remote_repo"], remote_python=block.get("remote_python") or "python3", gpus=int(block.get("gpus") or 1),
        identity_path=block.get("identity_path"), known_hosts=block.get("known_hosts"), ssh_options=tuple(block.get("ssh_options") or DEFAULT_SSH_OPTIONS),
    )  # fmt: skip


def _pod_context(args: argparse.Namespace, *, read_only: bool = False) -> Any:
    from robo_jev.launch.cloud import PodContext

    return PodContext(provider=_runpod(args, read_only=read_only), ledger=_ledger(getattr(args, "ledger", None)), backend_factory=_pod_backend)


DEFAULT_PRICE_GPUS = ("NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H200")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare" and args.job:
        from robo_jev.launch.jobs import JobError
        from robo_jev.launch.ledger import LedgerError, PhaseCapExceeded

        if args.config:
            parser.error("--config와 --job은 함께 주지 않는다")
        overrides = _overrides(args.set, parser)
        for flag, key in (("hourly_usd", "budget.hourly_usd"), ("max_wall_hours", "budget.max_wall_hours"), ("max_usd", "budget.max_usd"),
                          ("max_gpu_hours", "budget.max_gpu_hours"), ("price_source", "budget.price_source")):  # fmt: skip
            if getattr(args, flag) is not None:
                overrides[key] = getattr(args, flag)
        backend = None if args.backend == "ssh" and not args.host else _backend_from_args(args)
        try:
            manifest = prepare_job(job_path=args.job, run_id=args.run_id, manifest_path=args.manifest, backend=backend, overrides=overrides,
                                   code_ref=args.code_ref, allow_dirty=args.allow_dirty, ledger=_ledger(args.ledger) if args.ledger else None,
                                   notes=args.note)  # fmt: skip
        except (UncappedRun, JobError, PhaseCapExceeded, LedgerError) as exc:
            raise SystemExit(f"[launch_run] prepare 거절: {exc}") from None
        _print(manifest)
        return 0

    if args.command == "prepare":
        if not args.config:
            parser.error("--config(학습 설정) 또는 --job(판단 학습 job)이 필요하다")
        if args.hourly_usd is None:
            parser.error("--hourly-usd가 필요하다 (노드 시간 단가; 0이면 비용 상한을 걸 수 없다)")
        overrides = _overrides(args.set, parser)
        config = load_train_config(args.config, mode=args.mode, dataset=args.dataset, overrides=overrides)
        backend = _backend_from_args(args)
        budget = Budget(hourly_usd=args.hourly_usd, gpus=args.gpus, max_wall_hours=args.max_wall_hours,
                        max_gpu_hours=args.max_gpu_hours, max_usd=args.max_usd)  # fmt: skip
        try:
            manifest = prepare(
                config=config, config_path=args.config, run_id=args.run_id, backend=backend, budget=budget,
                manifest_path=args.manifest, overrides=overrides, price_source=args.price_source,
                seconds_per_step=args.seconds_per_step, assumption=args.assumption, notes=args.note,
                allow_no_cap=args.no_cap,
            )  # fmt: skip
        except UncappedRun as exc:
            # 돈 울타리에 걸렸다 — 명세를 만들지 않고 이유를 말하며 0이 아닌 코드로 끝난다.
            raise SystemExit(f"[launch_run] 상한 거절: {exc}") from None
        _print(manifest)
        return 0

    if args.command == "ledger":
        ledger = _ledger(args.ledger)
        if args.set_phase_cap is not None:
            ledger.set_phase_cap(args.set_phase_cap, reason=args.reason or "")
        print(json.dumps(ledger.summary(), ensure_ascii=False, indent=1))
        return 0

    if args.command == "prices":
        provider = _runpod(args, read_only=True)
        gpus = args.gpu or list(DEFAULT_PRICE_GPUS)
        rows = {}
        for gpu in provider.client.gpu_types():
            if gpu.get("id") in gpus:
                rows[gpu["id"]] = {"name": gpu.get("name"), "memory_gb": gpu.get("memory"), "secure": gpu.get("secure"), "community": gpu.get("community"),
                                   "price_per_gpu_hour": gpu.get("price")}  # fmt: skip
        for cloud in ("SECURE", "COMMUNITY"):
            for gpu in provider.client.gpu_types(availability=True, cloud=cloud, count=1):
                if gpu.get("id") in rows:
                    rows[gpu["id"]].setdefault("availability", {})[cloud] = gpu.get("availability")
                    rows[gpu["id"]].setdefault("cuda_versions", {})[cloud] = gpu.get("cudaVersions")
        print(json.dumps({"queried_at": now(), "gpus": rows, "missing": [g for g in gpus if g not in rows]}, ensure_ascii=False, indent=1))
        return 0

    if args.command == "pods":
        provider = _runpod(args, read_only=True)
        pods = provider.client.list_pods()
        rows = [{"id": p.get("id"), "name": p.get("name"), "status": p.get("status"), "cost_per_hour": p.get("cost"), "created_at": p.get("createdAt"),
                 "robojev": bool((p.get("env") or {}).get("ROBOJEV_RUN_ID")), "terminate_at": (p.get("env") or {}).get("ROBOJEV_TERMINATE_AT")} for p in pods]  # fmt: skip
        try:
            volumes = [{k: v.get(k) for k in ("id", "name", "size", "dataCenterId")} for v in provider.client.list_network_volumes()]
        except Exception as exc:  # noqa: BLE001 — 목록이 안 되면 그렇게 적는다
            volumes = [{"error": str(exc)[:300]}]
        live = [row for row in rows if row["status"] not in ("TERMINATED",)]
        print(json.dumps({"queried_at": now(), "pods": rows, "network_volumes": volumes, "live_pods": len(live), "clean": not live and not volumes},
                         ensure_ascii=False, indent=1))  # fmt: skip
        return 0

    manifest = load_manifest(args.manifest)

    if args.command == "dry-run":
        from robo_jev.launch.jobs import format_plan, plan_job

        if not manifest.get("job"):
            parser.error("dry-run은 판단 학습 job(--job으로 prepare한 명세)만 그린다")
        plan = plan_job(manifest, ledger=_ledger(args.ledger) if manifest.get("provider") else None)
        print(json.dumps(plan, ensure_ascii=False, indent=1) if args.as_json else format_plan(plan))
        return 0

    is_pod = bool(manifest.get("provider"))
    if args.command == "launch" and is_pod:
        from robo_jev.launch.cloud import LaunchNotApproved, launch_pod
        from robo_jev.launch.ledger import LedgerError, PhaseCapExceeded
        from robo_jev.launch.providers.runpod import PriceMismatch, RunPodError

        if not args.yes:
            from robo_jev.launch.jobs import format_plan, plan_job

            print(format_plan(plan_job(manifest, ledger=_ledger(args.ledger))))
            print("[launch_run] --yes가 없다 — 위의 예상 비용을 보고 승인할 때만 띄운다. 아무것도 만들지 않았다", file=sys.stderr)
            return 2
        ctx = _pod_context(args)
        try:
            manifest = launch_pod(manifest, manifest_path=args.manifest, ctx=ctx, yes=True, dest=args.dest, watch=args.watch,
                                  keep_on_interrupt=args.keep_on_interrupt)  # fmt: skip
        except LaunchNotApproved as exc:  # pragma: no cover - 위에서 걸렀다
            print(f"[launch_run] {exc}", file=sys.stderr)
            return 2
        except (PriceMismatch, PhaseCapExceeded, LedgerError) as exc:
            raise SystemExit(f"[launch_run] 띄우지 않았다: {exc}") from None
        except RunPodError as exc:
            print(f"[launch_run] RunPod 오류: {exc}", file=sys.stderr)
            _print(load_manifest(args.manifest))
            return 1
        _print(manifest)
        return 3 if manifest.get("exit_reason") == "pod-delete-unconfirmed" else (0 if manifest["state"] in ("completed", "running") else 2)

    if args.command in ("finish", "down"):
        from robo_jev.launch.cloud import down, finish

        if not is_pod:
            parser.error(f"{args.command}은 pod run(provider가 있는 명세)에만 쓴다")
        ctx = _pod_context(args)
        if args.command == "down":
            ok = down(manifest, manifest_path=args.manifest, ctx=ctx)
            _print(load_manifest(args.manifest))
            return 0 if ok else 3
        manifest = finish(manifest, manifest_path=args.manifest, ctx=ctx, dest=args.dest, wait=args.wait)
        _print(manifest)
        return 3 if manifest.get("exit_reason") == "pod-delete-unconfirmed" else 0

    backend = backend_from_manifest(manifest)

    if args.command == "launch":
        _print(launch(manifest, backend, manifest_path=args.manifest, force=args.force))
        return 0

    if args.command == "status":
        manifest = status(manifest, backend, manifest_path=args.manifest)
        if args.wait:
            import time

            deadline = time.monotonic() + float(args.timeout)
            while manifest["state"] == "running" and time.monotonic() < deadline:
                time.sleep(float(args.poll))
                manifest = status(manifest, backend, manifest_path=args.manifest)
        _print(manifest, as_json=args.as_json)
        return 0 if manifest["state"] in ("prepared", "running", "completed") else 2

    if args.command == "fetch":
        manifest = fetch(manifest, backend, dest=args.dest, manifest_path=args.manifest, names=args.name or None, strict=args.strict)
        for entry in manifest["artifacts"]:
            if entry.get("local") is None and entry.get("required_", True):
                print(f"{entry['kind']:>10} {entry['remote']}  — 원격에 없다")
            if entry.get("local"):
                print(f"{entry['kind']:>10} {entry['remote']}  {entry['bytes']} B  sha256 {str(entry['sha256'])[:12]}…  일치 {entry['sha256_match']}")
        _print(manifest)
        return 0

    if args.command == "cancel":
        try:
            manifest = cancel(manifest, backend, manifest_path=args.manifest, confirm_timeout=args.confirm_timeout, signal_name=args.signal)
        except CancelNotConfirmed as exc:
            print(f"[launch_run] ⚠ {exc}", file=sys.stderr, flush=True)
            _print(load_manifest(args.manifest))
            return 3
        _print(manifest)
        return 0

    if args.command == "resume":
        config_path = args.config or manifest["train_config"]["path"]
        overrides = _overrides(args.set, parser) or (manifest["train_config"].get("overrides") or {})
        config = load_train_config(config_path, mode=args.mode, dataset=args.dataset, overrides=overrides)
        reauthorised = None
        if any(value is not None for value in (args.hourly_usd, args.gpus, args.max_wall_hours, args.max_gpu_hours, args.max_usd)):
            parent_budget = manifest["budget"]
            reauthorised = Budget(
                hourly_usd=parent_budget["hourly_usd"] if args.hourly_usd is None else args.hourly_usd,
                gpus=parent_budget["gpus"] if args.gpus is None else args.gpus,
                max_wall_hours=args.max_wall_hours, max_gpu_hours=args.max_gpu_hours, max_usd=args.max_usd,
            )  # fmt: skip
        try:
            child = resume(manifest, backend, checkpoint=args.checkpoint, run_id=args.run_id, manifest_path=args.out,
                           config=config, config_path=config_path, notes=args.note, budget=reauthorised)  # fmt: skip
        except SpendNotObserved as exc:
            # 부모가 얼마나 썼는지 모르면 남은 예산도 모른다 — 관측을 먼저 하라고 말하고 끝난다.
            raise SystemExit(f"[launch_run] 재개 거절: {exc}") from None
        except UncappedRun as exc:
            raise SystemExit(f"[launch_run] 상한 거절: {exc}") from None
        if args.launch:
            child = launch(child, backend_from_manifest(child), manifest_path=args.out)
        _print(child)
        return 0

    parser.error(f"모르는 명령 {args.command!r}")  # pragma: no cover
    return 1


if __name__ == "__main__":
    sys.exit(main())
