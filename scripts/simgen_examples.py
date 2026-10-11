"""Task v0a — 사용자 검수용 예시 20개(examples-20.jsonl, examples-20.md)를 만든다 (지시서 §6).

    uv run python scripts/simgen_examples.py --out-dir artifacts/simgen/v0a

분야 10개 × 2, P1 단독·P5 단독·조합(6개 이상), 문체 고르게, numeric 쌍 하나. 칸마다 정답이 한쪽으로 다 쏠리지 않은 첫 seed를 고른다
(`robo_jev.simgen.examples.PLAN`). GPU·모델을 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.simgen.catalog import default_config  # noqa: E402
from robo_jev.simgen.examples import build_examples, dumps_jsonl, render_markdown  # noqa: E402
from robo_jev.simgen.generate import git_short_sha  # noqa: E402
from robo_jev.simgen.record import validate_record  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = default_config()
    records = build_examples(cfg, git=git_short_sha(REPO))
    errors = {r["id"]: validate_record(r) for r in records if validate_record(r)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "examples-20.jsonl").write_text(dumps_jsonl(records), encoding="utf-8")
    (args.out_dir / "examples-20.md").write_text(render_markdown(records, cfg), encoding="utf-8")
    domains = Counter(r["meta"]["domain"] for r in records)
    phen = Counter("+".join(r["meta"]["phenomena"]) for r in records)
    styles = Counter(r["meta"]["style"] for r in records)
    print(f"{len(records)} examples, validation errors {len(errors)}; domains {dict(domains)}; phenomena {dict(phen)}; styles {dict(styles)}")
    return 1 if errors or len(records) != 20 else 0


if __name__ == "__main__":
    raise SystemExit(main())
