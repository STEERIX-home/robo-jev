"""Task v0a 리뷰 1 질문 1·6 — 두 가치표(values-v0 / values-v0.1)에서 최선 행동의 분포를 나란히 잰다.

    uv run python scripts/simgen_value_compare.py --n 2000 --seed 0 --out artifacts/simgen/v0a/value-compare.json

표본 묶음과 같은 판단 순간(같은 seed 계획)마다 같은 사후 세계 K개를 한 번 굴리고, 그 순간 가능한 모든 행동의 기대 비용(기준 없음)을
두 가치표로 계산해 최선 행동을 센다 — 장면별(끼어듦 두 장면이 사용자 결정의 초점), 그리고 급제동이 최선인 비율·급제동과 단단한 제동의
기대 비용 차. 값은 사용자가 고른다; 이 스크립트는 고를 재료다. GPU·모델을 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.simgen.catalog import VALUES_PATH, VALUES_V01_PATH, load_config  # noqa: E402
from robo_jev.simgen.generate import build_moment, plan_units  # noqa: E402
from robo_jev.simgen.labels import cost_model, expected_costs  # noqa: E402
from robo_jev.simgen.posterior import sample_posterior  # noqa: E402
from robo_jev.simgen.rollout import available_options, rollout  # noqa: E402
from robo_jev.simgen.spec import stream  # noqa: E402

STEP_IN = ("p1_step_in", "p1p5_step_in_carrying")


def compare(n: int, seed: int) -> dict:
    tables = {"values-v0": load_config(values_path=VALUES_PATH), "values-v0.1": load_config(values_path=VALUES_V01_PATH)}
    cfg = tables["values-v0"]
    best: dict[str, dict[str, Counter]] = {name: defaultdict(Counter) for name in tables}
    hard_minus_firm: dict[str, list] = {name: [] for name in tables}
    by_body: dict[str, dict[str, Counter]] = {name: defaultdict(Counter) for name in tables}
    for unit in plan_units(n, seed):
        moment = build_moment(cfg, unit["seed"], unit["domain"])
        obs = moment.obs
        options = available_options(obs)
        out = rollout(cfg, obs, sample_posterior(cfg, obs, cfg.catalog["posterior_k"], stream(unit["seed"], "posterior")), options)
        scene = f"p1_crossing_{obs.variant}" if obs.scene == "p1_crossing" else obs.scene
        for name, table in tables.items():
            ec = expected_costs(out, cost_model(table.values, None, out.body, out.loaded))
            choice = options[int(np.argmin(ec))]
            best[name][scene][choice] += 1
            if obs.scene in STEP_IN:
                best[name]["step_in_both"][choice] += 1
                by_body[name][obs.body][choice] += 1
                hard_minus_firm[name].append(float(ec[out.index("stop_hard")] - ec[out.index("stop_moderate")]))

    def shares(counter: Counter) -> dict:
        total = sum(counter.values())
        return {"n": total, **{k: round(v / total, 4) for k, v in counter.most_common()}}

    return {
        "n": n,
        "seed": seed,
        "posterior_k": cfg.catalog["posterior_k"],
        "best_action_share": {name: {scene: shares(c) for scene, c in sorted(table.items())} for name, table in best.items()},
        "step_in_best_by_body": {name: {body: shares(c) for body, c in sorted(table.items())} for name, table in by_body.items()},
        "step_in_hard_minus_firm_expected_cost": {
            name: {"median": round(float(np.median(v)), 3), "share_hard_cheaper": round(float(np.mean(np.array(v) < 0)), 4)} for name, v in hard_minus_firm.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for scene in ("p1_step_in", "p1p5_step_in_carrying", "step_in_both"):
        for name in ("values-v0", "values-v0.1"):
            row = result["best_action_share"][name].get(scene, {})
            print(f"{scene:24s} {name:12s} " + ", ".join(f"{k} {v}" for k, v in row.items()))
    print(json.dumps(result["step_in_hard_minus_firm_expected_cost"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
