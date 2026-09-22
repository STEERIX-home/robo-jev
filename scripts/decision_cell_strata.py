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
    uv run python scripts/decision_cell_strata.py --rescope artifacts/reports/p3-reeval-*.json
        # 이미 저장된 보고서의 commitment 섞기 열에 **범위**를 적는다 (GPU 없이, 다른 값은 그대로임을 확인하고)

`reading`은 **이 파일의 수에서 만든다**(:func:`reading_text`) — 손으로 쓴 문단은 모집단이 바뀌어도 그대로 다시
찍히고, 실제로 P3의 첫 산출물이 P2의 문단(844·595·249·8편)을 그대로 실었다 (리뷰 1 C1).
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

from robo_jev.evaluate import (  # noqa: E402
    COMMITMENT_SHUFFLE_SCOPE, episode_bootstrap, load_eval_suite, split_episode_bootstrap,
)

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
#: Task R1의 run들 — **새 계기의 첫 눈금**(서식 v0.4·R1 데이터). 옛 체크포인트는 계약 digest가 달라 거절되므로
#: P1~P3의 run은 여기 없다; 그 값들은 "옛 계약·옛 모집단"으로 범위를 붙여 보존한다.
R1_RUNS = {
    "2B zero-shot": "r1-reeval-2b-zeroshot.json",
    "4B zero-shot": "r1-reeval-4b-zeroshot.json",
    "2B T0 (200)": "r1-reeval-2b-t0.json",
}
#: Task R2의 run들 — 고친 재료(`r1-robot-v0.2`) 위의 **첫 제대로 된 학습**. 로봇 manifest는 rollout 라벨판이다.
R2_RUNS = {
    "2B T1 fp32 master (233 = 1 epoch)": "r2-reeval-2b-t1-fp32-233.json",
    "2B T1 fp32 master (40)": "r2-reeval-2b-t1-fp32-40.json",
    "2B T0 (200)": "r2-reeval-2b-t0.json",
    "4B T0 (200)": "r2-reeval-4b-t0.json",
}
#: 같은 run들을 **선택에 쓰지 않는 둘째 칸**(`dev` 42편, `configs/eval/r2-dev-cell.yaml`)에서 읽은 보고서.
R2_DEV_RUNS = {name: report.replace("r2-reeval-", "r2-dev-") for name, report in R2_RUNS.items()}
RUN_SETS = {"p2": STRATA_RUNS, "p3": P3_RUNS, "r1": R1_RUNS, "r2": R2_RUNS, "r2dev": R2_DEV_RUNS}


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


def cell_records(suite_path: Any = DEFAULT_SUITE, *, split: str = SPLIT) -> list[str]:
    """평가 집합이 고른 편들을 **설정에 적힌 순서 그대로**. 기증자 회전이 이 순서에서 나온다 (:func:`donor_rotation`)."""
    suite = load_eval_suite(suite_path)
    entry = next(item for item in suite["splits"] if item["name"] == split)
    return [str(name) for name in (entry.get("records") or [])]


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
            # `record_ticks`는 그 **레코드 전체**의 틱 수다 — 기증자 길이 고정(:func:`donor_rotation`)은 라벨이 아니라
            # 편 길이가 정하므로, 라벨 없는 틱이 있어도 회전을 옳게 세려면 이 수가 필요하다.
            allowed = list(label.get("candidate_ids") or [])
            candidates = tick["request"]["candidates"][QUESTION]
            keys = {candidate["id"]: candidate.get("key") for candidate in candidates}
            commitment = ((tick["request"].get("state") or {}).get("commitment") or {}).get("action_ref")
            out.append({
                "episode_id": episode_id, "tick": index, "label": allowed, "commitment": commitment,
                "is_commitment": bool(commitment is not None and commitment in allowed),
                "key": str(keys.get(allowed[0]) if allowed else None).split(":")[0],
                "rule": label.get("rule"), "record_ticks": len(record["ticks"]),
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


def donor_rotation(ticks: list[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    """표준 대조군의 **기증자 배정**과 길이 고정이 덮는 틱 수 — 이 모집단에서, 편 길이만으로 (P3 C1b).

    :func:`robo_jev.evaluate.context_shuffle_records` 는 기증자를 **설정 목록의 다음 레코드**로 고르고,
    회전은 **편 길이로 짝짓고** 남는 차이는 기증자를 **감아 돌아** 읽는다 (Task R1 C2) — 곧 고정된 틱은 없다.
    옛 규칙은 기증자의 틱을 ``min(position, len(donor) - 1)``에서 읽어 기증자가 더 짧은 편의 나머지 틱이 전부
    기증자의 **마지막(끝난) 상태** 하나와 섞였고, 그 수가 **설정의 편 순서**에 달려 대조군 값이 draw마다 움직였다.
    이 블록은 두 수를 함께 적는다 — 지금 감아 도는 틱 수와, 옛 규칙이었다면 얼어붙었을 틱 수.
    이 수가 파일 안에 있어야 `reading`이 기증자 의존성을 자기 파일의 수로 말할 수 있다 (리뷰 1 C1·I1)."""
    length: dict[str, int] = {}
    for row in ticks:  # 라벨 없는 틱이 있어도 편 길이는 record_ticks가 안다 (없으면 본 틱의 최대 색인 + 1)
        name = row["episode_id"]
        length[name] = max(length.get(name, 0), int(row.get("record_ticks") or 0), int(row["tick"]) + 1)
    names = [name for name in order if name in length] or sorted(length)
    # 지금의 회전은 **편 길이로 짝짓는다** (`robo_jev.evaluate.donor_rotation`과 같은 규칙).
    paired = sorted(names, key=lambda name: (length[name], name))
    per_episode: dict[str, Any] = {}
    wrapped = total = old_clamped = 0
    for position, name in enumerate(paired):
        donor = paired[(position + 1) % len(paired)]
        short = max(0, length[name] - length[donor])
        per_episode[name] = {"donor": donor, "ticks": length[name], "donor_ticks": length[donor], "wrapped_ticks": short, "clamped_ticks": 0}
        wrapped, total = wrapped + short, total + length[name]
    for position, name in enumerate(names):  # 옛 규칙(설정 순서 + 클램프)이었다면 몇 틱이 얼어붙었을까
        old_clamped += max(0, length[name] - length[names[(position + 1) % len(names)]])
    return {
        "rule": (
            "the donor rotation is paired by episode length and the remaining difference WRAPS "
            "(position % len(donor)), so no tick is frozen on a donor's finished final state (Task R1 C2). "
            "The old rule took the donor at min(position, len(donor) - 1) from the config's order, which clamped "
            "the ticks counted below and made the control's value depend on that order."
        ),
        "ticks": total, "clamped_ticks": 0, "clamped_share": 0.0,
        "wrapped_ticks": wrapped, "wrapped_share": wrapped / total if total else None,
        "clamped_ticks_under_the_old_rule": old_clamped,
        "clamped_share_under_the_old_rule": old_clamped / total if total else None,
        "per_episode": per_episode,
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
    primary = [row for row in ticks if not row["is_commitment"]]
    out["primary_stratum_by_key_family"] = by_key_family(columns, primary, **options)
    if columns.get("state_shuffle"):
        out["primary_stratum_leave_one_episode_out"] = leave_one_episode_out(columns["model"], columns["state_shuffle"], primary, **options)
    return out


def leave_one_episode_out(model_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]],
                          rows: list[dict[str, Any]], **options: Any) -> dict[str, Any]:
    """편을 하나씩 빼고 주 지표의 여유를 다시 낸다 — **한 편이 판정을 만드는가**에 대한 답 (N2).

    구간은 편을 재표집하므로 불균형을 이미 담고 있지만, "한 편을 빼도 0을 제외하는가"는 그 구간이 대답하지 않는
    물음이고 이 칸에서는 그것이 실제 쟁점이었다(옛 칸은 이 층의 63.9 %가 한 편이었다)."""
    keep = {(row["episode_id"], row["tick"]) for row in rows}
    episodes = sorted({episode for episode, _ in keep})
    out: dict[str, Any] = {"episodes": len(episodes), "drops": {}}
    for episode in episodes:
        subset = {tick for tick in keep if tick[0] != episode}
        paired = episode_bootstrap(stratum_per_episode(model_rows, subset), stratum_per_episode(control_rows, subset), **options) or {}
        out["drops"][episode] = {
            "graded": paired.get("graded"), "margin": paired.get("margin"),
            "margin_ci": paired.get("margin_ci"), "margin_includes_zero": paired.get("margin_includes_zero"),
        }
    values = [(name, row) for name, row in out["drops"].items() if row.get("margin") is not None]
    if values:
        name, row = min(values, key=lambda pair: pair[1]["margin"])
        out["worst_drop"] = {"episode_id": name, **row}
        out["every_drop_excludes_zero"] = all(row["margin_includes_zero"] is False for _, row in values)
    return out


def by_key_family(columns: dict[str, list[dict[str, Any]] | None], rows: list[dict[str, Any]], **options: Any) -> dict[str, Any]:
    """주 지표 층을 **답의 키 갈래**로 더 쪼갠다 — 여유가 어느 갈래에서 나오는지 (구간은 붙이지 않는다).

    왜. 이 층은 `hold`(편 끝의 완료 꼬리)·`observe`(관측 게이트)·`grasp`(첫 틱)로 이루어져 있고 셋은 서로 다른 물음이다.
    "읽기가 필요한 층"이라는 이름 하나로 묶어 놓으면 목표를 읽어야 답이 나오는 갈래와 실행 상태만으로 풀리는 갈래가
    한 수에 섞인다. 판정은 여전히 **층 전체의 구간**이 하지만, 갈래의 여유에도 층과 **같은 쌍 부트스트랩**(편이 표본
    단위)을 붙여 둔다 — "여유가 전부 `grasp`에서 나온다"는 이 과제의 결론 하나가 그 구간에 기대고 있고, 그 구간이
    산출물 밖 스크래치 스크립트에만 있으면 `git clone` 뒤에는 재현되지 않는다 (리뷰 1 I4)."""
    keep = {(row["episode_id"], row["tick"]): row["key"] for row in rows}
    families = sorted({key for key in keep.values()})
    out: dict[str, Any] = {}
    for family in families:
        wanted = {tick for tick, value in keep.items() if value == family}
        block: dict[str, Any] = {
            "n": len(wanted),
            "episodes": len({episode for episode, _ in wanted}),
        }
        for column, predictions in columns.items():
            if not predictions:
                continue
            graded = [row for row in predictions if (str(row["record_id"]), int(row["tick"])) in wanted and row["correct"] is not None]
            block[column] = (sum(1 for row in graded if row["correct"]) / len(graded)) if graded else None
        # **대조군 열 전부**에 층과 같은 쌍 부트스트랩을 붙인다 (R2 C1). R1까지는 상태 섞기만 있었는데, R2의
        # 한 줄짜리 답은 지시 섞기의 여유이고 "그 여유가 `grasp`에서 나오는가"는 같은 꼴로 물어야 한다.
        for column in CONTROL_COLUMNS:
            if block.get("model") is None or block.get(column) is None:
                continue
            block[f"{column}_margin"] = block["model"] - block[column]
            paired = episode_bootstrap(
                stratum_per_episode(columns["model"] or [], wanted),
                stratum_per_episode(columns[column] or [], wanted), **options,
            ) or {}
            block[f"{column}_margin_ci"] = paired.get("margin_ci")
            block[f"{column}_margin_includes_zero"] = paired.get("margin_includes_zero")
            block[f"{column}_margin_episode_balanced"] = paired.get("episode_balanced_margin")
        out[family] = block
    return out


#: 어느 층이 **주 지표**인가 — P3 B1이 제도화한 답. 이 이름이 보고서에 그대로 들어간다.
PRIMARY_STRATUM = "non_commitment"
PRIMARY_STRATUM_NOTE = (
    "The primary metric is the non_commitment stratum: the ticks whose expert label is NOT the tick's own "
    "commitment.action_ref. The commitment stratum is always reported next to it, but its margin must never be "
    "quoted without the note that the standard control copies the answer through verbatim on those ticks."
)


def _share(value: float) -> str:
    """0~1의 몫을 이 프로젝트의 표기로 — 언제나 "20.8 %"다."""
    return f"{value * 100:.1f} %"


def reading_text(population: dict[str, Any], mechanism: dict[str, Any], donor: dict[str, Any] | None = None) -> str:
    """이 파일 **자신의 수**에서 "무엇을 인용하라"를 만든다 (P3 리뷰 1 C1).

    왜 손으로 쓰지 않는가. 이 필드의 일은 다음 독자에게 인용할 층을 지시하는 것이고, 손으로 쓴 문단은 모집단이
    바뀌어도 그대로 다시 찍힌다 — 실제로 P3의 산출물이 P2의 문단(844·595·249·8편)을 글자 그대로 실었고, 그 문단은
    이 과제가 **반박한** 주장("4B T0가 자기 대조군에 진다")까지 들고 있었다. 위의 `population`·`mechanism`·
    `donor_rotation`에서 문장을 만들면 모집단이 바뀌는 순간 문장이 같이 바뀐다. 판정(어느 run이 이겼는가)은 여기
    적지 않는다 — 판정은 `runs[*]`의 구간이 하고, 이 문장은 **어디를 보라**만 말한다."""
    ticks, nc = int(population["ticks"]), int(population["non_commitment_ticks"])
    commitment = int(mechanism["label_is_the_commitment"])
    families = " · ".join(f"{name} {count}" for name, count in (population.get("key_families") or {}).items())
    policy = mechanism["mechanical_policy"]
    largest, largest_b = population["largest_episode"], population["largest_episode_of_the_non_commitment_stratum"]
    out = [
        f"This cell is {population['episodes']} episodes / {ticks:,} {QUESTION} ticks.",
        f"The primary metric is the {PRIMARY_STRATUM} stratum: the {nc:,} ticks "
        f"({_share(nc / ticks)} of them) whose expert label is NOT the tick's own commitment.action_ref — key families {families}.",
        f"The other {commitment:,} ({_share(commitment / ticks)}) are 'repeat your commitment', where the standard state "
        f"shuffle keeps that line verbatim: the policy that reads nothing but the preserved fields scores "
        f"{policy['correct']:,}/{ticks:,} = {policy['accuracy']:.3f} on the whole cell.",
        f"The largest episode ({largest['episode_id']}) is {_share(largest['share_of_all_ticks'])} of the ticks and "
        f"{_share(largest_b['share_of_the_stratum'])} of the primary stratum, and the sampling unit is the episode — so "
        f"quote a margin only with its paired episode-clustered interval, and an interval that contains zero is not a finding.",
    ]
    if donor and donor.get("wrapped_share") is not None:
        out.append(
            f"The standard control no longer freezes any tick: {donor['rule']} On this population "
            f"{donor['wrapped_ticks']:,} of {donor['ticks']:,} ticks ({_share(donor['wrapped_share'])}) read a wrapped "
            f"donor tick and {donor['clamped_ticks']:,} are clamped; under the OLD rule "
            f"{donor['clamped_ticks_under_the_old_rule']:,} ({_share(donor['clamped_share_under_the_old_rule'])}) "
            f"would have been frozen on a finished state — which is why a state-shuffle value here cannot be merged "
            f"with a P1-P3 one (`donor_rotation` in this file)."
        )
    out.append(
        "Read `runs[*].non_commitment`. `whole_cell` is kept only so earlier published numbers stay comparable; "
        "it is not the result."
    )
    return " ".join(out)


def build(*, suite_path: Any = DEFAULT_SUITE, reports: Any = REPORTS, runs: dict[str, str] | None = None,
          split: str = SPLIT, task: str = "p2-decision-cell-strata", reading: str | None = None,
          records: list[str] | None = None) -> dict[str, Any]:
    reports = Path(reports)
    ticks = cell_ticks(suite_path, split=split)
    population = population_composition(ticks)
    found = mechanism(ticks)
    donor = donor_rotation(ticks, records if records is not None else cell_records(suite_path, split=split))
    out: dict[str, Any] = {
        "task": task,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "split": split, "question": QUESTION, "suite": str(suite_path),
        "primary_stratum": PRIMARY_STRATUM, "primary_stratum_note": PRIMARY_STRATUM_NOTE,
        # 읽기 문장은 **이 파일의 수에서 만든다** — 손으로 쓴 문단은 모집단이 바뀌어도 그대로 다시 찍힌다 (리뷰 1 C1)
        "reading": reading or reading_text(population, found, donor),
        "population": population,
        "donor_rotation": donor,
        "mechanism": found,
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


def rescope(path: Any) -> dict[str, Any]:
    """이미 저장된 재평가 보고서의 `episode_bootstrap`을 **GPU 없이** 지금의 규칙으로 다시 낸다 (리뷰 1 I3).

    왜 여기인가. 이 스크립트는 바로 그 보고서들의 편 단위 집계에서 구간을 다시 내는 CPU 경로다 — 같은 계산을
    :func:`robo_jev.evaluate.split_episode_bootstrap` 로 그대로 돌리면 저장된 블록이 **비트 단위로** 다시 나온다
    (부트스트랩의 seed가 고정이다). 그래서 commitment 섞기 열의 범위를 뒤늦게 적는 데 재실행이 필요 없다.

    무엇을 바꾸는가. 열 옆의 `commitment_shuffle_scope` 한 줄과, `q_main`이 아닌 질문의 commitment 섞기 여유·
    판정 표지다(:data:`robo_jev.evaluate.COMMITMENT_SHUFFLE_SCOPE`). **그 밖의 모든 값이 저장된 것과 같은지를
    확인하고**, 다르면 아무것도 쓰지 않고 멈춘다 — 이 경로가 측정을 바꿀 수는 없어야 한다."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed: dict[str, Any] = {}
    for name, table in ((payload.get("evaluation") or {}).get("splits") or {}).items():
        if "commitment_shuffle" not in table:
            continue
        stored = table.get("episode_bootstrap")
        rebuilt = split_episode_bootstrap({key: value for key, value in table.items() if key != "episode_bootstrap"})
        kept = {qid: {k: v for k, v in entry.items() if k != "commitment_shuffle"} for qid, entry in rebuilt.items()}
        was = {qid: {k: v for k, v in entry.items() if k != "commitment_shuffle"} for qid, entry in (stored or {}).items()}
        if kept != was:  # 이 경로는 측정을 바꾸지 않는다 — 다르면 멈춘다
            raise SystemExit(f"{path}: 다시 낸 구간이 저장된 것과 다르다 ({name}) — 재실행이 필요하다")
        table["commitment_shuffle_scope"] = COMMITMENT_SHUFFLE_SCOPE
        table["episode_bootstrap"] = rebuilt
        changed[name] = sum(1 for entry in rebuilt.values() if (entry.get("commitment_shuffle") or {}).get("out_of_scope"))
    if changed:
        payload["rescoped"] = {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "by": "scripts/decision_cell_strata.py --rescope",
            "what": "added commitment_shuffle_scope and withdrew the commitment-shuffle margin on the questions that "
                    "column cannot be read on (P3 review 1 I3). CPU only, no re-run.",
            "verified": "every other value of episode_bootstrap was re-derived from the stored per-episode counts and "
                        "compared to the stored one; identical.",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--split", default=SPLIT, help="평가 집합 안의 분할 이름")
    parser.add_argument("--runs", default="p2", choices=sorted(RUN_SETS), help="어느 재평가 묶음의 층화 표인가")
    parser.add_argument("--reports", default=str(REPORTS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="모집단 구성을 읽을 데이터셋 manifest")
    parser.add_argument("--population-splits", dest="population_splits", default=None,
                        help="이것을 주면 층화 대신 **모집단 구성만** 낸다 (쉼표로 나눈 split 이름)")
    parser.add_argument("--population-suites", dest="population_suites", default=None,
                        help="모집단 보기에 견줄 평가 집합들 — `이름=경로`를 쉼표로")
    parser.add_argument("--rescope", nargs="+", default=None, metavar="REPORT",
                        help="이미 저장된 재평가 보고서들의 commitment 섞기 열에 범위를 적는다 (:func:`rescope`; GPU 없음)")
    parser.add_argument("--reading", default=None,
                        help="읽기 문장을 직접 준다 — 기본은 이 파일의 `population`·`donor_rotation`·`mechanism`에서 **생성**한다")
    parser.add_argument("--out", default=str(REPORTS / "p2-decision-cell-strata.json"))
    args = parser.parse_args(argv)
    if args.rescope:
        for name in args.rescope:
            print(f"{name}: {rescope(name)}")
        return 0
    if args.population_splits:
        suites = dict(pair.split("=", 1) for pair in args.population_suites.split(",")) if args.population_suites else None
        payload = build_population([name.strip() for name in args.population_splits.split(",")], manifest=args.manifest, suites=suites)
        summary = ", ".join(f"{name} {block['whole_split']['episodes']}편 {block['whole_split']['ticks']}틱" for name, block in payload["splits"].items())
    else:
        payload = build(suite_path=args.suite, reports=args.reports, runs=RUN_SETS[args.runs], split=args.split,
                        task=f"{args.runs}-decision-cell-strata", reading=args.reading)
        summary = f"{len(payload['runs'])} runs, {len(payload['missing'])} missing"
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
