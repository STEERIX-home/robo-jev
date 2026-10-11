#!/usr/bin/env python
"""Task bench0 — 우리 판단 모델과 OpenRouter 결정 모델의 비교 벤치마크 bench-v0 (정의 configs/bench/bench-v0.yaml).

    # 정의대로 레코드를 다시 뽑아 artifacts/bench/v0/에 쓴다 (원본 sha256·레코드 수·질문 수·id sha256을 정의의 expect와 대조)
    uv run python scripts/bench.py build

정의의 규칙은 :mod:`robo_jev.bench.define` 의 모듈 설명이다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "bench0-1"
DEFAULT_CONFIG = REPO / "configs" / "bench" / "bench-v0.yaml"
DEFAULT_OUT = REPO / "artifacts" / "bench" / "v0"


def _log(message: str) -> None:
    print(f"[bench {dt.datetime.now().strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temp.replace(path)


def cmd_build(args: argparse.Namespace) -> int:
    from robo_jev.bench.define import build, load_config

    config = load_config(args.config)
    manifest = build(config, root=args.root, out_dir=args.out, check=not args.no_check)
    manifest["config"] = str(args.config)
    manifest["script_version"] = SCRIPT_VERSION
    manifest["created"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(args.out / "manifest.json", manifest)
    for name, entry in manifest["sets"].items():
        _log(f"{name}: {entry['records']} records · {entry['questions']['total']} questions · ids {entry['ids_sha256'][:12]}")
    if manifest.get("pilot"):
        pilot = manifest["pilot"]
        _log(f"pilot ({pilot['from']}, not in the bench): {pilot['records']} records · {pilot['questions']['total']} questions")
    _log(f"→ {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=REPO, help="정의의 상대 경로가 기대는 뿌리 (기본: 저장소)")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="정의대로 레코드를 다시 뽑아 쓴다 (expect와 대조)")
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.add_argument("--no-check", action="store_true", help="expect와 대조하지 않는다 (정의를 처음 만들 때만)")
    build.set_defaults(func=cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
