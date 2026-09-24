#!/usr/bin/env python
"""Task R5 A2 — 이른 그리퍼 답이 안전한지 실측 (시뮬 CPU; 데이터 생성이 아니라 검증 실험이다).

    # 정책 하나를 R4 dev 100 seed에 돌린다 (k0 = 덮어쓰기 없는 대조군 — 전문가의 R4 기록과 같아야 한다)
    uv run python scripts/gripper_early_experiment.py run --policy k1-closed --out artifacts/scratch/r5/a2/k1-closed \\
        --report artifacts/reports/r5-a2-run-k1-closed.json
    # 정책들을 seed로 짝지어 요약한다
    uv run python scripts/gripper_early_experiment.py summarise --runs artifacts/reports/r5-a2-run-*.json \\
        --out artifacts/reports/r5-a2-early-gripper.json

왜. 규칙 v2(docs/08 §7)는 전환 앞 k틱의 허용 집합을 넓혀 **이른** `closed`를 허용한다. 그것이 안전한 까닭은 실행기가
readiness 미충족이면 `gripper_wait`로 보류하기 때문이라고 **가정**돼 있다(`sim/controller.py` `_gripper_readiness`). 이 실험이
그 가정을 잰다: 전문가를 감싸 같은 seed의 전문가 기록에서 읽은 전환 t*의 k틱 앞부터 `closed`를 내는 정책
(:class:`robo_jev.data.gripper_labels.EarlyGripperPolicy`)으로 100 dev seed를 돌려 성공(`done ∧ target_inside_zone`)·실행된
닫기 시각·금지 접촉·거절·readiness 보류를 전문가(k0)와 seed로 짝지어 비교한다. 채택 규칙(브리프 A2): **성공률 차의 구간이 0을
포함하고 금지 접촉이 없을 때만** 그 k를 허용 폭으로 채택한다; 아니면 k = 0.

정책 이름: `k0`(대조군) · `k1-closed` · `k2-closed`(닫기만 일찍) · `k1-both` · `k2-both`(열기도 일찍).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SCRIPT_VERSION = "r5-a2-early-gripper-1.0"
DEFAULT_SEEDS = REPO / "artifacts/reports/r4-seeds.json"
DEFAULT_REFERENCE = REPO / "artifacts/datasets/r4-closed-loop/expert"
DEFAULT_SEEDS_CONFIG = REPO / "configs/eval/r4-closed-loop.yaml"
POLICIES = {
    "k0": (0, ("closed",)),
    "k1-closed": (1, ("closed",)), "k2-closed": (2, ("closed",)),
    "k1-both": (1, ("closed", "open")), "k2-both": (2, ("closed", "open")),
}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"→ {path}", flush=True)


# --------------------------------------------------------------------------
# 레코드에서 읽는 것
# --------------------------------------------------------------------------


def executed_close_ticks(record: dict[str, Any]) -> list[int]:
    """실행된 그리퍼(`state.exec.gripper`)가 open → closed로 바뀐 틱 색인들 (실제로 닫힌 시각)."""
    out: list[int] = []
    previous: str | None = None
    for index, tick in enumerate(record["ticks"]):
        executed = str((((tick.get("request") or {}).get("state") or {}).get("exec") or {}).get("gripper") or "")
        if previous is not None and executed == "closed" and previous != "closed":
            out.append(index)
        previous = executed or previous
    return out


def executed_open_ticks(record: dict[str, Any]) -> list[int]:
    out: list[int] = []
    previous: str | None = None
    for index, tick in enumerate(record["ticks"]):
        executed = str((((tick.get("request") or {}).get("state") or {}).get("exec") or {}).get("gripper") or "")
        if previous is not None and executed == "open" and previous == "closed":
            out.append(index)
        previous = executed or previous
    return out


def readiness_holds(record: dict[str, Any]) -> dict[str, int]:
    """ACK의 `gripper_wait`(readiness 미충족·정지 틱으로 보류된 명령)와 `gripper_event`(실행기가 받은 그리퍼 이벤트) 수."""
    waits: Counter = Counter()
    events = 0
    for tick in record["ticks"]:
        ack = tick.get("ack") or {}
        if ack.get("gripper_wait"):
            waits[str(ack["gripper_wait"])] += 1
        if ack.get("gripper_event"):
            events += 1
    return {"gripper_wait_ticks": sum(waits.values()), "gripper_wait_reasons": dict(sorted(waits.items())), "gripper_events": events}


def executed_before_expert(record: dict[str, Any]) -> dict[str, int]:
    """실행된 전환 가운데 **그 run의 전문가 자신이 아직 반대 상태를 원하던 틱의 명령**으로 일어난 것 — 실행기가 이른 명령을 보류하지 않고
    실행한 수. 닫기: 실행 그리퍼가 closed로 바뀐 틱 c의 직전 틱에서 전문가의 원하는 상태가 `open`. 열기: 그 반대. 대조군 대비 시각 차와
    달리 궤적이 앞당겨져 따라 움직인 전환은 세지 않는다 — 이 run 안의 참조로만 판정한다."""
    desired = [((meta.get("aux") or {}).get("gripper") or {}).get("desired") for meta in record["evidence"]["expert"]["ticks"]]
    previous: str | None = None
    closes = opens = early_closes = early_opens = 0
    for index, tick in enumerate(record["ticks"]):
        executed = str((((tick.get("request") or {}).get("state") or {}).get("exec") or {}).get("gripper") or "")
        if previous is not None and executed == "closed" and previous != "closed":
            closes += 1
            early_closes += int(index > 0 and desired[index - 1] == "open")
        if previous is not None and executed == "open" and previous == "closed":
            opens += 1
            early_opens += int(index > 0 and desired[index - 1] == "closed")
        previous = executed or previous
    return {"closes": closes, "closes_before_expert": early_closes, "opens": opens, "opens_before_expert": early_opens}


def forbidden_contacts_current_goal(record: dict[str, Any]) -> int:
    """그 틱의 **현재** 목표가 금지하는 물체와의 `contact_onset` 수 (R4의 `stop_vs_reflex`는 에피소드 안에서 본 금지 집합의 합집합으로 센다)."""
    count = 0
    for tick in record["ticks"]:
        state = tick["request"]["state"]
        forbidden = {str(item) for item in ((state.get("goal") or {}).get("forbidden_contact") or ())}
        count += sum(1 for event in (state.get("events") or ()) if str(event.get("kind")) == "contact_onset" and str(event.get("object")) in forbidden)
    return count


def same_trajectory(record: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """대조군(k0)이 전문가의 R4 기록과 같은 에피소드인가 — 틱 수·채택 그리퍼·주 결정·결과가 전부 같아야 한다."""
    ticks_equal = len(record["ticks"]) == len(reference["ticks"])
    adopted_equal = ticks_equal and all(
        (a.get("adopted") or {}).get("gripper") == (b.get("adopted") or {}).get("gripper") and (a.get("adopted") or {}).get("main") == (b.get("adopted") or {}).get("main")
        for a, b in zip(record["ticks"], reference["ticks"])
    )
    outcome_equal = bool(record["provenance"]["outcome"]["done"]) == bool(reference["provenance"]["outcome"]["done"])
    return {"ticks_equal": ticks_equal, "adopted_equal": adopted_equal, "outcome_equal": outcome_equal, "same": ticks_equal and adopted_equal and outcome_equal}


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from robo_jev.closed_loop import build_policy, episode_summary, load_closed_loop_config
    from robo_jev.contracts import validate_record
    from robo_jev.data.gripper_labels import EarlyGripperPolicy, reference_gripper_schedule
    from robo_jev.data.robot_episodes import build_manifest, config_paths, episode_id, generate_episode, write_episode
    from robo_jev.sim.environment import Environment

    early_ticks, directions = POLICIES[args.policy]
    config = load_closed_loop_config(args.config)
    generator = config["generator"]
    paths = config_paths(generator)
    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    block = seeds["conditions"][args.condition]
    schedule = [(entry["profile"], int(entry["seed"])) for entry in block["seeds"]]
    if args.limit:
        schedule = schedule[: int(args.limit)]
    bundle = build_policy("expert", generator=generator)
    expert = bundle["expert"]
    policy = EarlyGripperPolicy(expert, early_ticks=early_ticks, directions=directions)
    reference_dir = Path(args.reference) / args.condition
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    envs: dict[str, Any] = {}
    episodes: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for profile, seed in schedule:
            reference_path = reference_dir / "episodes" / f"{episode_id(profile, seed)}-r4-expert" / "streams.jsonl"
            reference = json.loads(reference_path.read_text(encoding="utf-8").strip())
            env = envs.get(profile)
            if env is None:
                env = envs[profile] = Environment(config_path=paths["sim_config"], profile=profile)
            policy.begin((profile, seed), reference_gripper_schedule(reference))
            record = generate_episode(profile, seed, policy=policy, expert=expert, config=generator, env=env, id_suffix=f"-r5-{args.policy}")
            validate_record(record)
            write_episode(record, out)
            summary = episode_summary(record)
            summary.update({
                "overrides": list(policy.overrides), "readiness": readiness_holds(record),
                "executed_before_expert": executed_before_expert(record), "forbidden_contacts_current_goal": forbidden_contacts_current_goal(record),
                "executed_closes": executed_close_ticks(record), "executed_opens": executed_open_ticks(record),
                "reference_closes": executed_close_ticks(reference), "reference_opens": executed_open_ticks(reference),
                "reference_schedule": reference_gripper_schedule(reference), "same_as_reference": same_trajectory(record, reference),
            })
            episodes.append(summary)
            print(f"{record['episode_id']} ticks={summary['ticks']:<3} done={summary['done']} inside={summary['done_inside']} overrides={len(policy.overrides)} "
                  f"waits={summary['readiness']['gripper_wait_ticks']} closes={summary['executed_closes']} ref={summary['reference_closes']} same={summary['same_as_reference']['same']}", flush=True)
    finally:
        for env in envs.values():
            env.close()
    wall = time.perf_counter() - started
    manifest = build_manifest(out, generator, batch_wall_s=wall)
    manifest["experiment"] = {"script": SCRIPT_VERSION, "policy": args.policy, "early_ticks": early_ticks, "directions": list(directions), "condition": args.condition}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "policy": args.policy, "early_ticks": early_ticks, "directions": list(directions),
        "expert_version": expert.version, "condition": args.condition, "seeds": str(args.seeds), "reference": str(reference_dir), "episodes_dir": str(out),
        "wall_seconds": round(wall, 1), "episodes": episodes,
    }
    _write(Path(args.report), payload)
    return 0


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------


def _pair_by_order(a: list[int], b: list[int]) -> list[int]:
    """같은 seed의 i번째 실행 닫기끼리 짝지은 틱 차 (정책 − 대조군); 짝이 없는 것은 세지 않는다."""
    return [x - y for x, y in zip(a, b)]


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(ordered), "mean": statistics.fmean(ordered), "median": ordered[len(ordered) // 2], "min": ordered[0], "max": ordered[-1],
            "share_zero": sum(1 for v in ordered if v == 0) / len(ordered), "share_negative": sum(1 for v in ordered if v < 0) / len(ordered)}


def summarise_runs(runs: dict[str, dict[str, Any]], *, control: str = "k0") -> dict[str, Any]:
    from robo_jev.closed_loop import condition_metrics, paired_success
    from robo_jev.data.robot_episodes import read_episodes

    if control not in runs:
        raise ValueError(f"대조군 run {control!r}가 없다 (있는 것: {sorted(runs)})")
    tables: dict[str, Any] = {}
    rows_by: dict[str, list[dict[str, Any]]] = {}
    for label, payload in runs.items():
        records = [record for _, record in read_episodes(Path(payload["episodes_dir"]))]
        rows = payload["episodes"]
        by_id = {record["episode_id"]: record for record in records}
        for row in rows:  # run JSON이 이 진단을 아직 들지 않았으면 레코드에서 센다
            record = by_id[row["episode_id"]]
            row.setdefault("executed_before_expert", executed_before_expert(record))
            row.setdefault("forbidden_contacts_current_goal", forbidden_contacts_current_goal(record))
        rows_by[label] = rows
        metrics = condition_metrics(records, [{k: v for k, v in row.items()} for row in rows])
        forbidden = metrics["stop_vs_reflex"]["forbidden_contact_onsets"]
        tables[label] = {
            "early_ticks": payload["early_ticks"], "directions": payload["directions"], "episodes": metrics["episodes"],
            "success": metrics["success_rate"], "success_ci": metrics["success_ci"],
            "strict_success": metrics["strict_success_rate"], "strict_success_ci": metrics["strict_success_ci"], "strict_done": metrics["strict_done"],
            "false_done": metrics["false_done"], "failure_causes": metrics["failure_causes"], "terminated": metrics["terminated"],
            "forbidden_contact_onsets": forbidden, "forbidden_contacts_current_goal": sum(row["forbidden_contacts_current_goal"] for row in rows),
            "executed_before_expert": {key: sum(row["executed_before_expert"][key] for row in rows) for key in ("closes", "closes_before_expert", "opens", "opens_before_expert")},
            "unsafe_action_rate": metrics["selective"]["unsafe_action_rate"],
            "controller": metrics["controller"], "gripper_events_vs_reference_label": metrics["gripper_events"],
            "overrides": {"ticks": sum(len(row["overrides"]) for row in rows), "episodes_with_override": sum(1 for row in rows if row["overrides"])},
            "readiness": {"gripper_wait_ticks": sum(row["readiness"]["gripper_wait_ticks"] for row in rows),
                          "gripper_events": sum(row["readiness"]["gripper_events"] for row in rows),
                          "reasons": dict(sum((Counter(row["readiness"]["gripper_wait_reasons"]) for row in rows), Counter()))},
            "executed_closes": sum(len(row["executed_closes"]) for row in rows), "executed_opens": sum(len(row["executed_opens"]) for row in rows),
            "same_as_reference": {"episodes": sum(1 for row in rows if row["same_as_reference"]["same"]), "of": len(rows)},
            "completion_ticks": metrics["completion_ticks"], "wall_seconds": payload.get("wall_seconds"),
        }
    paired: dict[str, Any] = {}
    control_rows = {row["key"]: row for row in rows_by[control]}
    for label, rows in rows_by.items():
        if label == control:
            continue
        shared = [row for row in rows if row["key"] in control_rows]
        close_shift = [d for row in shared for d in _pair_by_order(row["executed_closes"], control_rows[row["key"]]["executed_closes"])]
        open_shift = [d for row in shared for d in _pair_by_order(row["executed_opens"], control_rows[row["key"]]["executed_opens"])]
        close_count_diff = [len(row["executed_closes"]) - len(control_rows[row["key"]]["executed_closes"]) for row in shared]
        forbidden_diff = [row["forbidden_contacts_current_goal"] - control_rows[row["key"]]["forbidden_contacts_current_goal"] for row in shared]
        paired[f"{label} - {control}"] = {
            "forbidden_contacts_current_goal": {"sum_diff": sum(forbidden_diff), "episodes_more": sum(1 for d in forbidden_diff if d > 0), "episodes_fewer": sum(1 for d in forbidden_diff if d < 0), "seeds": len(shared)},
            "success": paired_success(rows, rows_by[control]), "strict_success": paired_success(rows, rows_by[control], strict=True),
            "executed_close_tick_shift": _distribution(close_shift), "executed_open_tick_shift": _distribution(open_shift),
            "executed_close_tick_shift_values": close_shift, "executed_open_tick_shift_values": open_shift,
            "executed_close_count_diff": {"sum": sum(close_count_diff), "episodes_fewer": sum(1 for d in close_count_diff if d < 0), "episodes_more": sum(1 for d in close_count_diff if d > 0)},
            "episodes_identical_to_control": sum(1 for row in shared if same_trajectory_rows(row, control_rows[row["key"]])),
        }
    verdict: dict[str, Any] = {}
    for label, block in paired.items():
        policy = label.split(" - ")[0]
        strict = block["strict_success"] or {}
        table = tables[policy]
        # 브리프 A2의 채택 규칙: `done ∧ target_inside_zone` 차의 쌍 구간이 0을 포함하고 금지 접촉이 **대조군(전문가 자신)보다 늘지 않을 때**만
        # (전문가 자신의 dev 100편에 현재 목표 기준 금지 접촉이 있으므로 "0"은 "새 것이 없다"로 읽는다 — seed별 짝지은 차). 그 옆에 진단
        # 둘을 적는다: 실행기가 이른 명령을 보류하지 않고 **그 run의 전문가가 아직 반대 상태를 원하던 틱**에 실행한 전환 수, 그리고
        # 이른 `open`이 실행된 수 — 후자는 운반 높이에서 물체를 떨어뜨리는 것이라 그 방향은 채택하지 않는다(3 seed 예비 실행에서 본
        # 것: 놓기 영역 위에서 떨어진 물체가 영역 안에 들어가 성공률은 그것을 가려내지 못했다).
        forbidden_pair = block["forbidden_contacts_current_goal"]
        before = table["executed_before_expert"]
        no_new_contacts = forbidden_pair["episodes_more"] == 0
        safe = bool(strict.get("margin_includes_zero")) and no_new_contacts and before["opens_before_expert"] == 0
        verdict[policy] = {
            "early_ticks": table["early_ticks"], "directions": table["directions"],
            "strict_margin": strict.get("margin"), "strict_margin_ci": strict.get("margin_ci"), "margin_includes_zero": strict.get("margin_includes_zero"),
            "forbidden_contacts_current_goal": table["forbidden_contacts_current_goal"], "forbidden_contacts_vs_control": forbidden_pair, "no_new_forbidden_contacts": no_new_contacts,
            "closes_before_expert": before["closes_before_expert"], "of_closes": before["closes"],
            "opens_before_expert": before["opens_before_expert"], "of_opens": before["opens"],
            "adoptable": safe,
        }
    closed_only = [v for v in verdict.values() if list(v["directions"]) == ["closed"] and v["adoptable"]]
    both = [v for v in verdict.values() if sorted(v["directions"]) == ["closed", "open"] and v["adoptable"]]
    adopted_closed = max((int(v["early_ticks"]) for v in closed_only), default=0)
    adopted_both = max((int(v["early_ticks"]) for v in both), default=0)
    return {
        "control": control, "tables": tables, "paired": paired, "verdict": verdict,
        "rule": "adopt k only if the paired strict-success (done ∧ target_inside_zone) margin vs k0 contains zero, no seed has more current-goal forbidden contacts than the control, and no early open was executed before the expert's own release; the widest passing k is adopted. closes executed before the expert's own tick-level check are reported (the controller's 15 mm readiness bounds them) but are not a criterion",
        "adopted_early_ticks": {"closed_only": adopted_closed, "both_directions": adopted_both},
    }


def same_trajectory_rows(row: dict[str, Any], control: dict[str, Any]) -> bool:
    return row["ticks"] == control["ticks"] and row["executed_closes"] == control["executed_closes"] and row["executed_opens"] == control["executed_opens"] and row["done_inside"] == control["done_inside"]


def cmd_summarise(args: argparse.Namespace) -> int:
    paths: list[Path] = []
    for pattern in args.runs:
        matched = sorted(REPO.glob(pattern)) if not Path(pattern).is_absolute() else sorted(Path("/").glob(pattern.lstrip("/")))
        paths.extend(matched or [Path(pattern)])
    runs = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[str(payload["policy"])] = {**payload, "path": str(path)}
    report = summarise_runs(runs, control=args.control)
    report.update({"script": SCRIPT_VERSION, "generated_at": _now(), "git": _git_commit(), "runs": {label: payload["path"] for label, payload in runs.items()}})
    _write(Path(args.out), report)
    print("\n| policy | k | dirs | success (done) | done ∧ inside | paired strict − k0 | forbidden contacts (current goal; seeds with more than k0) | unsafe | reject | overrides | readiness holds | executed closes | close tick shift vs k0 mean / share 0 | closes / opens executed before the expert's own decision | same as control |")
    print("| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
    for label, table in report["tables"].items():
        pair = report["paired"].get(f"{label} - {args.control}") or {}
        strict = pair.get("strict_success") or {}
        shift = pair.get("executed_close_tick_shift") or {}
        zero = " (0 inside)" if strict.get("margin_includes_zero") else ""
        margin = "—" if strict.get("margin") is None else f"{strict['margin']:+.3f} [{strict['margin_ci'][0]:+.3f}, {strict['margin_ci'][1]:+.3f}]{zero}"
        shift_text = "—" if not shift.get("n") else f"{shift['mean']:+.2f} / {shift['share_zero']:.2f}"
        directions = "+".join(table["directions"])
        verdict = report["verdict"].get(label, {})
        more = (pair.get("forbidden_contacts_current_goal") or {}).get("episodes_more", "—")
        before = table["executed_before_expert"]
        print(f"| {label} | {table['early_ticks']} | {directions} | {table['success']:.3f} | {table['strict_done']} | {margin} | {table['forbidden_contacts_current_goal']} ({more}) | {table['unsafe_action_rate']:.4f} | "
              f"{table['controller']['rejection_rate']:.4f} | {table['overrides']['ticks']} | {table['readiness']['gripper_wait_ticks']} | {table['executed_closes']} | "
              f"{shift_text} | {before['closes_before_expert']}/{before['closes']} / {before['opens_before_expert']}/{before['opens']}{' → adoptable' if verdict.get('adoptable') else ''} | {pair.get('episodes_identical_to_control', '—')} |")
    print(f"\nadopted early_ticks: {report['adopted_early_ticks']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="정책 하나를 조건의 seed 전부에 돌린다 (CPU)")
    run.add_argument("--policy", required=True, choices=sorted(POLICIES))
    run.add_argument("--condition", default="dev")
    run.add_argument("--config", default=str(DEFAULT_SEEDS_CONFIG))
    run.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    run.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="전문가의 R4 기록 (조건별 하위 디렉터리)")
    run.add_argument("--out", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=cmd_run)
    summarise = sub.add_parser("summarise", help="정책들을 seed로 짝지어 요약한다")
    summarise.add_argument("--runs", nargs="+", required=True)
    summarise.add_argument("--control", default="k0")
    summarise.add_argument("--out", required=True)
    summarise.set_defaults(func=cmd_summarise)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
