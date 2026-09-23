#!/usr/bin/env python
"""보고서의 표를 산출물에서 **만든다** — 손으로 옮겨 적지 않는다 (R2 fix round 1, 리뷰 1 M-10).

R2까지 이 표들은 `artifacts/scratch/r2/tables.py`가 찍었는데 `artifacts/`는 커밋되지 않는다 — 곧 `git clone`한
체크아웃에서는 보고서의 사건·안전·대조 쌍 표를 아무도 다시 만들 수 없었다. 입력은 전부 커밋된 것이거나
`src/robo_jev`에 있으므로(사건 지표는 평가 JSON 안에, :func:`~robo_jev.evaluate.selective_metrics_from_stored`
는 시험이 붙은 함수다) 표를 만드는 쪽만 여기로 옮긴다. 층화 표는 `scripts/decision_cell_strata.py`가 만든
JSON에서 읽는다.

**`q_stop` 열은 검열률을 언제나 함께 찍는다** (리뷰 1 C-1, docs/08 §10: "지연·오경보를 같이 적는다 …
상한을 평균에 섞으면 '느리다'와 '안 한다'가 같은 수가 된다"). 중앙 지연은 **반응한 사건만**의 중앙값이므로,
10건 중 1건만 반응한 열의 "중앙 2틱"은 검열률 없이는 거짓말이 된다.

    uv run python scripts/report_tables.py strata   artifacts/reports/r2-decision-cell-strata.json "2B T1 fp32 master (233 = 1 epoch)"
    uv run python scripts/report_tables.py loo      artifacts/reports/r2-decision-cell-strata.json "2B T1 fp32 master (233 = 1 epoch)"
    uv run python scripts/report_tables.py seeds    artifacts/reports/r3a-decision-cell-strata.json artifacts/reports/r3a-dev-cell-strata.json
    uv run python scripts/report_tables.py events   artifacts/reports/r2-reeval-2b-t1-fp32-233.json robot/ood_dev
    uv run python scripts/report_tables.py unsafe   artifacts/reports/r2-reeval-2b-t1-fp32-233.json robot/ood_dev
    uv run python scripts/report_tables.py contrast artifacts/reports/r2-contrast-2b-t1-233.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

#: 평가 표의 열 이름 → 표에 찍는 이름. 순서가 표의 줄 순서다.
COLUMN_LABEL = {"model": "**model**", "context_shuffle": "state shuffle", "instruction_shuffle": "instruction shuffle",
                "commitment_shuffle": "commitment shuffle", "rule_judge": "**rule judge**", "mechanical_baseline": "**mechanical baseline**"}
#: 층화 표(`scripts/decision_cell_strata.py`)의 대조군 이름 → 찍는 이름. 그쪽은 상태 섞기를 `state_shuffle`로
#: 부른다(평가 표는 `context_shuffle`이다) — 두 자를 한 이름으로 합치지 않고 각자의 이름을 그대로 쓴다.
CONTROL_LABEL = {"state_shuffle": "state shuffle", "instruction_shuffle": "**instruction shuffle**",
                 "commitment_shuffle": "commitment shuffle"}


def _pct(x: Any, digits: int = 2) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f} %"


def f(x: Any, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def stop_cell(stop: dict[str, Any] | None) -> str:
    """`q_stop` 한 칸 — **중앙 지연 / 검열 / 오경보**, 셋을 언제나 함께 (리뷰 1 C-1).

    중앙값은 **반응한 사건만**의 것이다. 검열(30틱 안에 반응이 없었다)을 옆에 적지 않으면 "10건 중 1건을
    잡았고 그 1건이 2틱"이 "중앙 2틱"으로 읽힌다 — 그 둘은 같은 수가 아니다.
    """
    if not stop:
        return "— / — / —"
    onsets, censored = stop.get("onsets"), stop.get("censored")
    if onsets is None or censored is None:
        share = "—"
    else:
        share = f"{censored} of {onsets} ({(censored / onsets) * 100:.1f} %)" if onsets else "0 of 0"
    return f"{f(stop.get('median_ticks'), 1)} / {share} / {_pct(stop.get('false_alarm_rate'))}"


def ci(block: dict[str, Any], column: str) -> tuple[str, str]:
    margin, interval = block.get(f"{column}_margin"), block.get(f"{column}_margin_ci")
    zero = block.get(f"{column}_margin_includes_zero")
    if margin is None:
        return "—", "—"
    text = f"{margin:+.3f}" + (f" [{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "")
    return text, ("**YES**" if zero else "no")


def strata_table(data: dict[str, Any], run: str) -> str:
    """층화 표 + 갈래 표 — `scripts/decision_cell_strata.py`가 만든 JSON의 한 run에서."""
    block = data["runs"][run]
    lines = ["| stratum | n | model | state shuffle | margin (paired 95 %) | 0? | instr. shuffle | margin | 0? | commit. shuffle | margin | 0? | rule judge | mechanical |",
             "| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: |"]
    for name, label in (("non_commitment", "**primary (label != commitment)**"), ("commitment", "commitment"), ("whole_cell", "whole cell")):
        row = block[name]
        cells = [label, str(row["n"]), f(row.get("model")), f(row.get("state_shuffle"))]
        for column in ("state_shuffle", "instruction_shuffle", "commitment_shuffle"):
            text, zero = ci(row, column)
            cells += [text, zero] if column == "state_shuffle" else [f(row.get(column)), text, zero]
        cells += [f(row.get("rule_judge")), f(row.get("mechanical_baseline"))]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| family | n | eps | model | state sh. | margin | 0? | instr. sh. | margin | 0? | rule | mech |",
              "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: |"]
    for family, row in sorted(block["primary_stratum_by_key_family"].items(), key=lambda kv: -kv[1]["n"]):
        s_text, s_zero = ci(row, "state_shuffle")
        i_text, i_zero = ci(row, "instruction_shuffle")
        lines.append("| " + " | ".join([f"`{family}`", str(row["n"]), str(row["episodes"]), f(row.get("model")), f(row.get("state_shuffle")), s_text, s_zero,
                                        f(row.get("instruction_shuffle")), i_text, i_zero, f(row.get("rule_judge")), f(row.get("mechanical_baseline"))]) + " |")
    loo = block.get("primary_stratum_leave_one_episode_out") or {}
    if loo.get("worst_drop"):
        w = loo["worst_drop"]
        lines += ["", f"leave-one-episode-out (state shuffle): every drop excludes zero = {loo.get('every_drop_excludes_zero')}; "
                      f"worst {w['episode_id']} {w['margin']:+.3f} [{w['margin_ci'][0]:+.3f}, {w['margin_ci'][1]:+.3f}]"]
    return "\n".join(lines)


def loo_table(data: dict[str, Any], run: str) -> str:
    """편 하나 빼기 — **대조군 열마다** (R2 B2a의 표; `strata_table`은 상태 섞기 한 줄만 찍는다).

    "모든 드롭이 0을 제외하는가"는 여유가 한 편에 업혀 있지 않다는 확인이고, 대조군마다 따로 물어야 한다 —
    지시 섞기의 여유가 어느 편 없이도 서는지는 상태 섞기의 같은 질문과 다른 질문이다.
    """
    block = data["runs"][run]["primary_stratum_leave_one_episode_out_by_control"]
    lines = ["| control | episodes refit | every drop excludes zero? | worst drop |",
             "| --- | ---: | :---: | --- |"]
    for column, label in CONTROL_LABEL.items():
        row = block.get(column)
        if not row:
            continue
        worst = row.get("worst_drop") or {}
        interval = worst.get("margin_ci") or []
        text = "—" if not worst else (
            f"`{worst['episode_id']}` {worst['margin']:+.4f}"
            + (f" [{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else "")
            + ("" if not worst.get("margin_includes_zero") else " **(contains 0)**")
        )
        lines.append(f"| {label} | {row.get('episodes')} | {'**yes**' if row.get('every_drop_excludes_zero') else '**NO**'} | {text} |")
    return "\n".join(lines)


def seed_table(cell: dict[str, Any], dev: dict[str, Any] | None = None) -> str:
    """run마다 **자기 구간을 한 줄에** — seed 표(Task R3a D1)와 epoch 표(D2)가 같은 모양이다.

    **세 seed의 값으로 구간을 만들지 않는다.** 이 표는 seed마다 자기 편 단위 쌍 부트스트랩 구간을 나란히
    놓을 뿐이고, "0을 제외한다"는 줄마다 따로 읽는다 — 한 줄이라도 0을 포함하면 그 줄의 값과 함께
    "아직 seed 의존적"이라고 적는다. 마지막 두 칸은 **판정 칸이 아닌** 둘째 칸(`dev`)의 같은 여유다.
    """
    lines = ["| run | primary n | model | **instruction-shuffle margin (paired 95 %)** | 0? | `grasp` margin | 0? | state-shuffle margin | 0? | LOO worst (instruction) | `dev` cell margin | 0? |",
             "| --- | ---: | ---: | ---: | :---: | ---: | :---: | ---: | :---: | --- | ---: | :---: |"]
    for name, block in cell.get("runs", {}).items():
        if not block.get("available"):
            lines.append(f"| {name} | — | — | **missing** | — | — | — | — | — | — | — | — |")
            continue
        primary = block["non_commitment"]
        grasp = (block.get("primary_stratum_by_key_family") or {}).get("grasp") or {}
        instruction, instruction_zero = ci(primary, "instruction_shuffle")
        grasp_text, grasp_zero = ci(grasp, "instruction_shuffle")
        state, state_zero = ci(primary, "state_shuffle")
        loo = ((block.get("primary_stratum_leave_one_episode_out_by_control") or {}).get("instruction_shuffle") or {})
        worst = loo.get("worst_drop") or {}
        interval = worst.get("margin_ci") or []
        loo_text = "—" if not worst else (
            f"`{worst['episode_id']}` {worst['margin']:+.4f}"
            + (f" [{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else "")
            + ("" if not worst.get("margin_includes_zero") else " **(contains 0)**")
        )
        if loo and not loo.get("every_drop_excludes_zero"):
            loo_text += " — **not every drop excludes zero**"
        second = ((dev or {}).get("runs", {}).get(name) or {})
        if second.get("available"):
            dev_text, dev_zero = ci(second["non_commitment"], "instruction_shuffle")
        else:
            dev_text, dev_zero = "—", "—"
        lines.append("| " + " | ".join([
            name, str(primary["n"]), f(primary.get("model")), f"**{instruction}**", instruction_zero,
            grasp_text, grasp_zero, state, state_zero, loo_text, dev_text, dev_zero,
        ]) + " |")
    return "\n".join(lines)


def events_table(data: dict[str, Any], split: str) -> str:
    """사건 지표 표 — 반응 지연·안정성·`q_stop`. `q_stop`은 중앙 지연 **옆에 검열**을 달고 나온다."""
    metrics = data["evaluation"]["splits"][split]["event_metrics"]
    lines = ["| column | goal change: median / immediate / censored | world event: median / censored | switch rate | round trips | hold median | stop: median / censored / false alarm |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for column, label in COLUMN_LABEL.items():
        block = metrics.get(column)
        if not block:
            continue
        g, w = block["reaction_delay"]["goal_change"], block["reaction_delay"]["world_event"]
        st = block["stability"]
        lines.append(f"| {label} | {f(g['median_ticks'], 1)} / {g['immediate_rate'] * 100:.1f} % / {g['censored_rate'] * 100:.1f} % "
                     f"| {f(w['median_ticks'], 1)} / {w['censored_rate'] * 100:.1f} % | {st['switch_rate']:.3f} | {st['round_trips']} "
                     f"| {f(st['hold_ticks_median'], 1)} | {stop_cell(block.get('stop_timing'))} |")
    model = metrics["model"]
    lines += ["", f"(events: goal_change {model['reaction_delay']['goal_change']['events']}, "
                  f"world {model['reaction_delay']['world_event']['events']}, stop onsets {model['stop_timing']['onsets']}, "
                  f"horizon {model['reaction_delay']['horizon_ticks']})"]
    return "\n".join(lines)


def unsafe_table(data: dict[str, Any], split: str, *, root: Any = REPO) -> str:
    """안전·선택적 지표 표 — 저장된 틱별 예측에서 다시 세므로 대조군 열도 같은 자로 잰다."""
    from robo_jev.evaluate import load_eval_suite, load_suite_items, selective_metrics_from_stored
    from robo_jev.model.tokenizer import available_tokenizer, load_tokenizer

    root = Path(root)
    suite = load_eval_suite(root / data["eval_config"])
    items = load_suite_items(suite, tokenizer=load_tokenizer(available_tokenizer()[0]), root=root, domain_tag="provenance.robojev_domain")
    records = [item.record for item in items[split]]
    ticks = {r["episode_id"]: len(r["ticks"]) for r in records}
    table = data["evaluation"]["splits"][split]
    lines = ["| column | n | coverage | selective acc | wrong target | **unsafe** | forbidden target | stop ignored |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for column, label in COLUMN_LABEL.items():
        block = table.get(column)
        if not block or "q_main" not in block or not block["q_main"].get("per_record"):
            continue
        rows = list(block["q_main"]["per_record"]) + list((block.get("q_stop") or {}).get("per_record") or [])
        bad = [row for row in rows if int(row["tick"]) >= ticks.get(row["record_id"], 0)]
        if bad:
            raise SystemExit(f"저장된 예측이 지금 데이터와 다른 코퍼스의 것이다 ({len(bad)}행이 틱 범위 밖) — 나란히 놓을 수 없다")
        m = selective_metrics_from_stored(rows, records)
        lines.append(f"| {label} | {m['n']} | {f(m['coverage'])} | {f(m['selective_accuracy'])} | {f(m['wrong_target_rate'])} "
                     f"| **{f(m['unsafe_action_rate'], 4)}** | {m['forbidden_target']} | {m['stop_ignored']} |")
    return "\n".join(lines)


def contrast_table(data: dict[str, Any]) -> str:
    """지시 대조 쌍 표 — 쌍 수·종류별 민감도·`false_change`(이 데이터에서는 언제나 `null`이다)."""
    lines = ["| split | pairs | instruction pairs | **instruction sensitivity** | forbidden | zone_boundary | all | both correct | false_change |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, table in data["evaluation"]["splits"].items():
        pairs = table.get("contrast_pairs")
        if not pairs:
            continue
        whole = pairs.get("_all") or {}

        def rate(kind: str, pairs: dict[str, Any] = pairs) -> str:
            row = pairs.get(kind) or {}
            return f(row.get("sensitivity")) if row else "—"

        instr = pairs.get("instruction") or {}
        lines.append(f"| `{name}` | {whole.get('pairs')} | {instr.get('pairs', 0)} | **{rate('instruction')}** | {rate('forbidden')} "
                     f"| {rate('zone_boundary')} | {f(whole.get('sensitivity'))} | {f(whole.get('both_correct'))} | {whole.get('false_change')} |")
    return "\n".join(lines)


def _load(path: Any) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("strata", "loo", "seeds", "events", "unsafe", "contrast"):
        print(__doc__, file=sys.stderr)
        return 2
    what, rest = argv[0], argv[1:]
    if what == "strata":
        print(strata_table(_load(rest[0]), rest[1]))
    elif what == "loo":
        print(loo_table(_load(rest[0]), rest[1]))
    elif what == "seeds":
        print(seed_table(_load(rest[0]), _load(rest[1]) if len(rest) > 1 else None))
    elif what == "events":
        print(events_table(_load(rest[0]), rest[1]))
    elif what == "unsafe":
        print(unsafe_table(_load(rest[0]), rest[1]))
    else:
        print(contrast_table(_load(rest[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
