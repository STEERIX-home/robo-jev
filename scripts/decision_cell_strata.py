#!/usr/bin/env python
"""판정 칸을 **라벨이 지금 commitment인가**로 층화해 다시 세고, 그 칸의 **구성**을 함께 적는다 (P2 리뷰 1 I1, P3 A1·B1).

왜 있는가. 844틱 가운데 **595틱(70.5 %)의 정답이 그 틱 자신의 `commitment.action_ref`**이고, 표준 대조군인 상태
섞기는 `commitment` 줄을 **일부러 남긴다**(그것이 그 틱의 실행 이력이기 때문이다). 그래서 "commitment가 있으면
그것을, 없으면 `observe` 게이트 키를 답한다"는 기계적 정책이 아무것도 읽지 않고 **751/844 = 0.890**을 받는다 —
모델의 섞인 0.904와 0.014 차이다. 곧 **집계 여유는 70 %가 '하던 것 계속하기'인 모집단에서 잰 값이라 읽기 능력을
희석한다**. 읽기가 필요한 것은 나머지 249틱이고, 거기서는 그림이 갈린다.

무엇을 읽는가. 재평가 보고서의 `evaluation.splits[split][열][질문].per_record`(`configs/eval/*.yaml`의 분할이
`store_predictions`로 켠 틱별 예측)와 데이터셋의 틱 자체다. 둘을 (record_id, tick)으로 짝지어 층을 가르고, 층마다
편 단위 집계를 다시 만들어 :func:`robo_jev.evaluate.episode_bootstrap` 에 그대로 넣는다 — 여유의 구간은 전체 칸과
똑같이 **같은 편에서 짝지은** 쌍 부트스트랩이다.

무엇을 조심할 것인가. P2가 쓴 8편 칸에서는 비-commitment 층 249틱 가운데 **159틱이 한 편(`ep-E1-000235`)**이었다
(전체 칸에서도 그 편이 300/844 = 35.5 %다). 8편을 재표집하는 구간은 그 불균형을 값 자체에 담지만, "249틱"은 실제보다
균형 있게 읽힌다. 그래서 이 스크립트는 층화 표와 **같은 파일에** 모집단의 구성(`population`)을 적는다 — 편마다 틱 수·
비-commitment 틱 수·라벨 키 갈래, 그리고 가장 큰 편의 몫. P3의 새 모집단(`configs/eval/p3-decision-cell.yaml`,
`ood_dev` 24편 전부)은 그 두 몫이 11.9 % / 30.3 %다.

    uv run python scripts/decision_cell_strata.py --out artifacts/reports/p2-decision-cell-strata.json
    uv run python scripts/decision_cell_strata.py --suite configs/eval/p3-decision-cell.yaml --runs p3 \
        --out artifacts/reports/p3-decision-cell-strata.json
    uv run python scripts/decision_cell_strata.py --population-splits ood_dev,dev \
        --out artifacts/reports/p3-population.json     # A1 — 분할 **전체**의 구성 (run 없이, 데이터만)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robo_jev.evaluate import episode_bootstrap, load_eval_suite  # noqa: E402

REPORTS = REPO / "artifacts" / "reports"
DEFAULT_SUITE = REPO / "configs" / "eval" / "pilot-decision-cell.yaml"
DEFAULT_MANIFEST = REPO / "artifacts" / "datasets" / "d1-robot" / "d1-rollout-labels" / "manifest.json"
SPLIT, QUESTION = "robot/ood_dev", "q_main"

#: 표의 열 이름 → 보고서의 자리 이름. `commitment_shuffle`·`mechanical_baseline`은 P3가 더한 열이고, 없는 보고서에서는
#: 그 줄이 그냥 빠진다(옛 보고서도 이 스크립트로 그대로 다시 만들어진다).
COLUMNS = {
    "model": "model", "state_shuffle": "context_shuffle", "instruction_shuffle": "instruction_shuffle",
    "commitment_shuffle": "commitment_shuffle", "rule_judge": "rule_judge", "mechanical_baseline": "mechanical_baseline",
}
#: 대조군 열 — 여유(margin)를 세우는 열이다. 규칙 판정기·기계적 기준군은 **기준선**이지 대조군이 아니라 여유를 세우지 않는다.
CONTROL_COLUMNS = ("state_shuffle", "instruction_shuffle", "commitment_shuffle")

#: 판정 칸의 run들 (표에 넣을 이름 → 재평가 보고서). `select_backbone.DECISION_CELL_RUNS`와 같은 run들이고,
#: 여기에는 **틱별 예측이 있는** 보고서만 들어간다(없는 것은 `missing`으로 적고 빈칸으로 둔다).
STRATA_RUNS = {
    "2B T0 (200)": "p2-reeval-2b-t0.json",
    "2B LoRA (40)": "p2-reeval-2b-lora.json",
    "2B T1 bf16 (40)": "p2-reeval-2b-t1.json",
    "4B T0 (200)": "p2-reeval-4b-t0.json",
    "4B LoRA (40)": "p2-reeval-4b-lora.json",
    "2B T1 bf16 5 s (40)": "p2-reeval-2b-t1-bf16-5s.json",
    "2B T1 fp32 master (40)": "p2-reeval-2b-t1-fp32.json",
}
#: 같은 run들을 **P3의 새 모집단**(24편 전부)에서 다시 잰 보고서.
P3_RUNS = {name: report.replace("p2-reeval-", "p3-reeval-") for name, report in STRATA_RUNS.items()}
RUN_SETS = {"p2": STRATA_RUNS, "p3": P3_RUNS}


# --------------------------------------------------------------------------
# 데이터셋 쪽 — 층과 기전
# --------------------------------------------------------------------------


def split_episodes(manifest_path: Any, split: str) -> list[str]:
    """manifest가 아는 그 split의 에피소드 id 전부 (정렬). A1이 "8편이 24편을 얼마나 대표하는가"를 물을 때 쓴다."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return sorted(
        str(entry["episode_id"])
        for name, entry in (manifest.get("files") or {}).items()
        if name.startswith("episodes/") and entry.get("split") == split
    )


def cell_ticks(suite_path: Any = DEFAULT_SUITE, *, root: Any = REPO, split: str = SPLIT) -> list[dict[str, Any]]:
    """평가 집합이 고른 편들의 `q_main` 틱마다 ``{episode_id, tick, label, commitment, is_commitment, key, rule}``.

    `label`은 채점에 쓰이는 허용 집합(`candidate_ids`)이고 P2의 칸에서는 844/844가 한 개짜리다. `key`는 그 답
    후보의 키 갈래(`observe` / `hold` / `grasp` / `place` …)로, 비-commitment 층이 무엇으로 이루어졌는지를 본다.
    """
    root = Path(root)
    suite = load_eval_suite(suite_path)
    entry = next(item for item in suite["splits"] if item["name"] == split)
    base = (root / entry["manifest"]).parent
    return dataset_ticks(base, entry["records"])


def dataset_ticks(base: Any, episodes: list[str]) -> list[dict[str, Any]]:
    """데이터셋에서 곧바로 — 평가 집합을 거치지 않고 편 목록만으로 (A1의 분할 전체 보기)."""
    base = Path(base)
    out: list[dict[str, Any]] = []
    for episode_id in episodes:
        record = json.loads((base / "episodes" / episode_id / "streams.jsonl").read_text(encoding="utf-8").strip())
        for index, tick in enumerate(record["ticks"]):
            label = next((row for row in tick.get("labels", []) if row.get("question_id") == QUESTION), None)
            if label is None:
                continue
            allowed = list(label.get("candidate_ids") or [])
            candidates = tick["request"]["candidates"][QUESTION]
            keys = {candidate["id"]: candidate.get("key") for candidate in candidates}
            commitment = ((tick["request"].get("state") or {}).get("commitment") or {}).get("action_ref")
            out.append({
                "episode_id": episode_id, "tick": index, "label": allowed, "commitment": commitment,
                "is_commitment": bool(commitment is not None and commitment in allowed),
                "key": str(keys.get(allowed[0]) if allowed else None).split(":")[0],
                "rule": label.get("rule"),
                # 아무것도 읽지 않는 기계적 정책의 답 — commitment가 있으면 그것, 없으면 `observe` 게이트 키
                "mechanical": commitment if commitment is not None else next((c["id"] for c in candidates if str(c.get("key", "")).startswith("observe")), None),
            })  # fmt: skip
    return out


def mechanism(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """대조군이 답을 그대로 나르는 크기 — 0.904의 기전 (P2 리뷰 1 I1)."""
    total = len(ticks)
    with_commitment = [row for row in ticks if row["commitment"] is not None]
    is_commitment = [row for row in ticks if row["is_commitment"]]
    mechanical = sum(1 for row in ticks if row["mechanical"] in row["label"])
    others = [row for row in ticks if not row["is_commitment"]]
    return {
        "ticks": total,
        "singleton_labels": sum(1 for row in ticks if len(row["label"]) == 1),
        "ticks_with_a_commitment": len(with_commitment),
        "label_is_the_commitment": len(is_commitment),
        "share_of_all_ticks": len(is_commitment) / total if total else None,
        "share_of_ticks_that_have_one": len(is_commitment) / len(with_commitment) if with_commitment else None,
        "ticks_without_a_commitment": total - len(with_commitment),
        "label_differs_from_a_live_commitment": len(with_commitment) - len(is_commitment),
        "keep_commitment_rule": sum(1 for row in ticks if row["rule"] == "expert-e0.4/keep_commitment"),
        "mechanical_policy": {
            "policy": "answer the tick's own commitment.action_ref if there is one, otherwise the observe gate key — both are fields the state shuffle preserves",
            "correct": mechanical, "n": total, "accuracy": mechanical / total if total else None,
        },
        "non_commitment_key_families": dict(Counter(row["key"] for row in others).most_common()),
        "non_commitment_per_episode": dict(Counter(row["episode_id"] for row in others).most_common()),
        "ticks_per_episode": dict(Counter(row["episode_id"] for row in ticks).most_common()),
    }


def population_composition(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """모집단의 **구성** — 편마다 틱 수·비-commitment 틱 수·라벨 키 갈래, 그리고 가장 큰 편의 몫 (P3 A1·N2).

    이 블록이 층화 표와 같은 파일에 있어야 하는 이유: "249틱"은 실제보다 균형 있게 읽힌다. 층의 크기가 아니라
    **그 크기가 몇 편에서 왔는지**가 구간의 폭과 한 편의 영향력을 정한다."""
    episodes = sorted({row["episode_id"] for row in ticks})
    others = [row for row in ticks if not row["is_commitment"]]
    per_episode = {}
    for episode in episodes:
        mine = [row for row in ticks if row["episode_id"] == episode]
        mine_b = [row for row in mine if not row["is_commitment"]]
        per_episode[episode] = {
            "ticks": len(mine), "non_commitment_ticks": len(mine_b),
            "non_commitment_share": len(mine_b) / len(mine) if mine else None,
            "key_families": dict(Counter(row["key"] for row in mine_b).most_common()),
        }
    largest = max((entry["ticks"] for entry in per_episode.values()), default=0)
    largest_b = max((entry["non_commitment_ticks"] for entry in per_episode.values()), default=0)
    return {
        "episodes": len(episodes), "ticks": len(ticks), "non_commitment_ticks": len(others),
        "non_commitment_share": len(others) / len(ticks) if ticks else None,
        "key_families": dict(Counter(row["key"] for row in others).most_common()),
        "largest_episode": {
            "episode_id": next((e for e in episodes if per_episode[e]["ticks"] == largest), None),
            "ticks": largest, "share_of_all_ticks": largest / len(ticks) if ticks else None,
        },
        "largest_episode_of_the_non_commitment_stratum": {
            "episode_id": next((e for e in episodes if per_episode[e]["non_commitment_ticks"] == largest_b), None),
            "ticks": largest_b, "share_of_the_stratum": largest_b / len(others) if others else None,
        },
        "episodes_with_no_non_commitment_tick": [e for e in episodes if per_episode[e]["non_commitment_ticks"] == 0],
        "per_episode": per_episode,
    }


# --------------------------------------------------------------------------
# 층화
# --------------------------------------------------------------------------


def stratum_per_episode(rows: list[dict[str, Any]], keep: set[tuple[str, int]]) -> list[dict[str, Any]]:
    """틱별 예측을 ``keep``의 (record_id, tick)으로 좁혀 편 단위 집계로 되돌린다 (:func:`aggregate` 의 꼴 그대로)."""
    counts: dict[str, list[int]] = {}
    for row in rows:
        key = (str(row["record_id"]), int(row["tick"]))
        if key not in keep:
            continue
        entry = counts.setdefault(str(row["record_id"]), [0, 0, 0])
        entry[0] += 1
        if row["correct"] is not None:
            entry[1] += 1
            entry[2] += int(bool(row["correct"]))
    return [{"episode_id": name, "n": value[0], "graded": value[1], "correct": value[2]} for name, value in sorted(counts.items())]


def repeats_the_commitment(rows: list[dict[str, Any]], ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """섞인 열이 **그 틱 자신의 commitment를 그대로** 답하는 비율 — 대조군이 무엇을 하고 있는지."""
    commitments = {(row["episode_id"], row["tick"]): row["commitment"] for row in ticks}
    live = [row for row in rows if commitments.get((str(row["record_id"]), int(row["tick"]))) is not None]
    repeated = sum(1 for row in live if row["predicted"] == commitments[(str(row["record_id"]), int(row["tick"]))])
    correct_total = sum(1 for row in rows if row["correct"])
    repeated_and_correct = sum(
        1 for row in live
        if row["predicted"] == commitments[(str(row["record_id"]), int(row["tick"]))] and row["correct"]
    )  # fmt: skip
    return {
        "ticks_with_a_commitment": len(live), "predicted_that_commitment": repeated,
        "share": repeated / len(live) if live else None,
        "correct_answers": correct_total, "of_which_are_that_repetition": repeated_and_correct,
        "share_of_correct_answers": repeated_and_correct / correct_total if correct_total else None,
    }


def run_strata(table: dict[str, Any], ticks: list[dict[str, Any]], **options: Any) -> dict[str, Any]:
    """한 run의 판정 칸 표 → 층마다 열별 정확도와 대조군 대비 **쌍** 구간."""
    columns = {
        name: ((table.get(source) or {}).get(QUESTION) or {}).get("per_record")
        for name, source in COLUMNS.items()
    }
    if not columns["model"]:
        return {"available": False, "reason": f"the report has no per_record for {SPLIT} {QUESTION} (the split did not ask for store_predictions)"}
    strata = {
        "commitment": {(row["episode_id"], row["tick"]) for row in ticks if row["is_commitment"]},
        "non_commitment": {(row["episode_id"], row["tick"]) for row in ticks if not row["is_commitment"]},
        "whole_cell": {(row["episode_id"], row["tick"]) for row in ticks},
    }
    out: dict[str, Any] = {"available": True}
    for name, keep in strata.items():
        per_episode = {column: stratum_per_episode(rows, keep) for column, rows in columns.items() if rows}
        entry = episode_bootstrap(per_episode["model"], **options) or {}
        block: dict[str, Any] = {
            "n": sum(row["n"] for row in per_episode["model"]),
            "episodes": {row["episode_id"]: row["n"] for row in per_episode["model"]},
            "model": entry.get("accuracy"), "model_ci": entry.get("accuracy_ci"),
        }
        block["model_episode_balanced"] = entry.get("episode_balanced_accuracy")  # 편 균등 평균 (P3 A3)
        block["model_episode_balanced_ci"] = entry.get("episode_balanced_accuracy_ci")
        for column in [name for name in COLUMNS if name != "model"]:
            rows = per_episode.get(column)
            if not rows:
                continue
            paired = episode_bootstrap(per_episode["model"], rows, **options) or {}
            block[column] = paired.get("control_accuracy")
            block[f"{column}_episode_balanced"] = paired.get("episode_balanced_control_accuracy")
            if column in CONTROL_COLUMNS:  # 규칙 판정기·기계적 기준군은 대조군이 아니라 기준선이다 — 여유를 세우지 않는다
                block[f"{column}_margin"] = paired.get("margin")
                block[f"{column}_margin_ci"] = paired.get("margin_ci")
                block[f"{column}_margin_includes_zero"] = paired.get("margin_includes_zero")
                block[f"{column}_margin_episode_balanced"] = paired.get("episode_balanced_margin")
                block[f"{column}_margin_episode_balanced_ci"] = paired.get("episode_balanced_margin_ci")
                block[f"{column}_margin_episode_balanced_includes_zero"] = paired.get("episode_balanced_margin_includes_zero")
        out[name] = block
    if columns.get("state_shuffle"):
        out["state_shuffle_repeats_the_commitment"] = repeats_the_commitment(columns["state_shuffle"], ticks)
    return out


#: 어느 층이 **주 지표**인가 — P3 B1이 제도화한 답. 이 이름이 보고서에 그대로 들어간다.
PRIMARY_STRATUM = "non_commitment"
PRIMARY_STRATUM_NOTE = (
    "The primary metric is the non_commitment stratum: the ticks whose expert label is NOT the tick's own "
    "commitment.action_ref. The commitment stratum is always reported next to it, but its margin must never be "
    "quoted without the note that the standard control copies the answer through verbatim on those ticks."
)


def build(*, suite_path: Any = DEFAULT_SUITE, reports: Any = REPORTS, runs: dict[str, str] | None = None,
          split: str = SPLIT, task: str = "p2-decision-cell-strata", reading: str | None = None) -> dict[str, Any]:
    reports = Path(reports)
    ticks = cell_ticks(suite_path, split=split)
    out: dict[str, Any] = {
        "task": task,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "split": split, "question": QUESTION, "suite": str(suite_path),
        "primary_stratum": PRIMARY_STRATUM, "primary_stratum_note": PRIMARY_STRATUM_NOTE,
        "reading": reading or (
            "The decision cell is ~70 % 'repeat your commitment', which dilutes every aggregate margin: on 595 of its "
            "844 ticks the expert label IS the tick's own commitment.action_ref, and the state shuffle keeps that line "
            "verbatim. Read the two strata separately. On the 249 ticks that need the goal read, the properly-trained "
            "fp32 T1 beats its own goal-blind control by a wide margin while the 4B T0 loses to its own control. "
            "Both strata are 8 episodes and one of them (ep-E1-000235) is 300 of the 844 ticks and 159 of the 249, "
            "so every number here carries the same paired episode-clustered interval and the same caveat."
        ),
        "population": population_composition(ticks),
        "mechanism": mechanism(ticks),
        "runs": {},
        "missing": {},
    }
    for label, name in (runs or STRATA_RUNS).items():
        path = reports / name
        if not path.is_file():
            out["missing"][label] = {"report": name, "reason": "report not produced"}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        table = ((payload.get("evaluation") or {}).get("splits") or {}).get(split)
        if table is None:
            out["missing"][label] = {"report": name, "reason": f"the report has no {split} split"}
            continue
        entry = run_strata(table, ticks)
        if not entry.get("available"):
            out["missing"][label] = {"report": name, "reason": entry.get("reason")}
            continue
        out["runs"][label] = {"report": name, "eval_set_sha256": ((payload.get("evaluation") or {}).get("eval_set") or {}).get("sha256"), **entry}
    return out


def build_population(splits: list[str], *, manifest: Any = DEFAULT_MANIFEST, suites: dict[str, str] | None = None) -> dict[str, Any]:
    """A1 — 분할 **전체**의 구성, 그리고 지금 쓰는 평가 집합이 그것을 얼마나 대표하는지 (run 없이, 데이터만)."""
    manifest = Path(manifest)
    out: dict[str, Any] = {
        "task": "p3-population",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "question": QUESTION, "manifest": str(manifest), "splits": {},
    }
    for split in splits:
        episodes = split_episodes(manifest, split)
        ticks = dataset_ticks(manifest.parent, episodes)
        out["splits"][split] = {
            "whole_split": {**population_composition(ticks), "mechanism": mechanism(ticks)},
            "suites": {},
        }
        for name, path in (suites or {}).items():
            suite = load_eval_suite(path)
            entry = next((item for item in suite["splits"] if item.get("split") == split and item.get("domain") == "robot"), None)
            if entry is None:
                continue
            chosen = list(entry.get("records") or [])
            subset = [row for row in ticks if row["episode_id"] in set(chosen)]
            block = {**population_composition(subset), "mechanism": mechanism(subset)}
            block["episodes_of_the_split"] = len(episodes)
            block["share_of_the_split_ticks"] = len(subset) / len(ticks) if ticks else None
            block["missing_episodes"] = [e for e in episodes if e not in set(chosen)]
            out["splits"][split]["suites"][name] = block
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--split", default=SPLIT, help="평가 집합 안의 분할 이름")
    parser.add_argument("--runs", default="p2", choices=sorted(RUN_SETS), help="어느 재평가 묶음의 층화 표인가")
    parser.add_argument("--reports", default=str(REPORTS))
    parser.add_argument("--population-splits", dest="population_splits", default=None,
                        help="이것을 주면 층화 대신 **모집단 구성만** 낸다 (쉼표로 나눈 split 이름)")
    parser.add_argument("--population-suites", dest="population_suites", default=None,
                        help="모집단 보기에 견줄 평가 집합들 — `이름=경로`를 쉼표로")
    parser.add_argument("--out", default=str(REPORTS / "p2-decision-cell-strata.json"))
    args = parser.parse_args(argv)
    if args.population_splits:
        suites = dict(pair.split("=", 1) for pair in args.population_suites.split(",")) if args.population_suites else None
        payload = build_population([name.strip() for name in args.population_splits.split(",")], suites=suites)
        summary = ", ".join(f"{name} {block['whole_split']['episodes']}편 {block['whole_split']['ticks']}틱" for name, block in payload["splits"].items())
    else:
        payload = build(suite_path=args.suite, reports=args.reports, runs=RUN_SETS[args.runs], split=args.split,
                        task=f"{args.runs}-decision-cell-strata")
        summary = f"{len(payload['runs'])} runs, {len(payload['missing'])} missing"
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
