"""요약 JSON → 사람이 읽는 markdown 표 (Task v0c §2.4).

숫자는 :func:`robo_jev.decisions.metrics.summarize` 의 값 그대로(소수 셋째 자리)이고, 없는 칸은 ``—``다. 미리보기 run은
:data:`PREVIEW_NOTE` 를 머리에 단다 — 생성 표본의 숫자이지 봉인된 시험 세트(T1~T7, 다음 단계 v0b)의 숫자가 아니다.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["PREVIEW_NOTE", "reasoning_stats", "render_comparison", "render_markdown"]

PREVIEW_NOTE = (
    "PREVIEW on a generated sample (v0a `sample-2000.jsonl`), not the sealed test sets T1-T7 (sealed in v0b). "
    "Read the numbers as a first look at where and how the untrained model errs, not as the baseline of record."
)

#: 묶음 표의 핵심 열 — (묶음, 지표, 열 이름).
KEY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("noul", "kl", "noul KL"),
    ("noul", "brier", "noul Brier"),
    ("score", "kl", "score KL"),
    ("score", "brier", "score Brier"),
    ("choice/action", "accuracy", "action acc"),
    ("choice/action", "regret_mean", "action regret"),
    ("choice/state", "accuracy", "state acc"),
    ("choice/rule", "accuracy", "rule acc"),
)
_BLOCK_COLUMNS = {
    "noul": (("log_loss", "log loss"), ("kl", "KL"), ("brier", "Brier"), ("ece", "ECE"), ("mae", "MAE"), ("side_agreement", "same side"), ("mean_p", "mean p"), ("mean_y", "mean label"), ("code_mass", "code mass")),
    "score": (("log_loss", "log loss"), ("kl", "KL"), ("brier", "Brier"), ("ece", "ECE"), ("mae", "MAE"), ("side_agreement", "same side"), ("mean_p", "mean value"), ("mean_y", "mean label"), ("code_mass", "code mass")),
    "choice/action": (("ce", "CE"), ("kl", "KL"), ("accuracy", "acc (min cost)"), ("regret_mean", "regret"), ("regret_positive_rate", "regret>0"), ("expected_regret_mean", "E[regret] under p"), ("code_mass", "code mass")),
    "choice/state": (("ce", "CE"), ("kl", "KL"), ("accuracy", "acc"), ("ece", "ECE (top-label)"), ("code_mass", "code mass")),
    "choice/rule": (("accuracy", "acc"), ("ce", "CE"), ("code_mass", "code mass")),
    "choice/other": (("ce", "CE"), ("kl", "KL"), ("accuracy", "acc"), ("code_mass", "code mass")),
}
_UNIFORM_COLUMNS = {
    "noul": (("log_loss", "log loss"), ("brier", "Brier")),
    "score": (("log_loss", "log loss"), ("brier", "Brier")),
    "choice/action": (("ce", "CE"), ("accuracy", "acc"), ("expected_regret_mean", "E[regret]")),
    "choice/state": (("ce", "CE"), ("accuracy", "acc")),
    "choice/rule": (("ce", "CE"), ("accuracy", "acc")),
    "choice/other": (("ce", "CE"), ("accuracy", "acc")),
}
_GROUP_ORDER = (
    ("phenomena", "By phenomenon"), ("scene", "By scene"), ("domain", "By domain"), ("body", "By body"), ("sensor_suite", "By sensor suite"),
    ("style", "By style"), ("hint", "By computed hint"), ("recommendation", "By planner recommendation (none / right / wrong)"),
    ("margin_band", "By decision-margin band (|m| tertiles per margin name)"), ("family", "By question family"),
    ("pair", "By pair membership"), ("label_position", "By position of the best option (choice only; 1 = first listed)"),
    ("key_style", "By option-key style"), ("agent", "By moving agent"), ("distractors", "By number of irrelevant facts"),
    ("distance_unit", "By distance unit"),
)


def _f(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _key_table(groups: dict[str, Any]) -> list[str]:
    header = ["group", "questions"] + [name for _, _, name in KEY_COLUMNS]
    rows = []
    for value, blocks in groups.items():
        n = (blocks.get("_all") or {}).get("n", 0)
        cells = []
        for block, metric, _ in KEY_COLUMNS:
            entry = blocks.get(block)
            cells.append(f"{_f(entry.get(metric))} ({entry['n']})" if entry and entry.get(metric) is not None else "—")
        rows.append([str(value), str(n)] + cells)
    return _table(header, rows)


def reasoning_stats(predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """reasoning run의 생성 통계: 평균 생성 토큰, 상한에 닿은 비율, 스스로 답을 쓴 비율, 멈춘 이유별 개수."""
    entries = [q["reasoning"] for p in predictions for q in p["questions"].values() if q.get("reasoning")]
    if not entries:
        return None
    n = len(entries)
    return {
        "questions": n, "generated_tokens_mean": sum(e["generated_tokens"] for e in entries) / n,
        "cap_rate": sum(1 for e in entries if e["stop"] == "cap") / n, "self_answer_rate": sum(1 for e in entries if e["self_answer"]) / n,
        "stops": dict(sorted(Counter(e["stop"] for e in entries).items())),
    }


def render_markdown(summary: dict[str, Any], *, config: dict[str, Any], title: str, note: str | None = None) -> str:
    """요약 → markdown 보고서 (모듈 설명)."""
    lines = [f"# {title}", ""]
    if note:
        lines += [f"> **{note}**", ""]
    counts = summary.get("counts", {})
    lines += _table(
        ["model", "mode", "format", "prompt", "records", "questions scored", "input"],
        [[str(config.get("model", "—")), str(config.get("mode", "—")), str(config.get("format", "—")), str(config.get("prompt_version", "—")),
          str(counts.get("records", "—")), f"{counts.get('scored', '—')} / {counts.get('questions', '—')}",
          f"`{config.get('input', '—')}` (sha256 {str(config.get('input_sha256', ''))[:12]})"]],
    )
    if config.get("selection"):
        lines += ["", f"Selection: {config['selection']}"]
    lines += ["", "## By question type", ""]
    overall = summary.get("overall", {})
    uniform = (summary.get("reference") or {}).get("uniform", {})
    for block, columns in _BLOCK_COLUMNS.items():
        entry = overall.get(block)
        if not entry:
            continue
        ref = uniform.get(block, {})
        header = ["block", "n"] + [name for _, name in columns] + [f"uniform {name}" for _, name in _UNIFORM_COLUMNS[block]]
        row = [block, str(entry["n"])] + [_f(entry.get(key)) for key, _ in columns] + [_f(ref.get(key)) for key, _ in _UNIFORM_COLUMNS[block]]
        lines += _table(header, [row]) + [""]
    all_entry = overall.get("_all", {})
    lines += [f"All scored questions: n = {all_entry.get('n', 0)}, mean KL to the posterior label = {_f(all_entry.get('kl'))}, mean code mass = {_f(all_entry.get('code_mass'))}.", ""]

    groups = summary.get("groups", {})
    for dim, heading in _GROUP_ORDER:
        if dim in groups and groups[dim]:
            lines += [f"## {heading}", ""] + _key_table(groups[dim]) + [""]

    rec = summary.get("recommendation", {})
    lines += ["## Planner recommendation", "", "Action choices whose options include the recommended option (follow = model argmax is the recommended option).", ""]
    lines += _table(
        ["recommendation", "n", "follow rate", "model p(recommended)", "label p(recommended)", "recommended is best (label)", "model regret"],
        [[name, str(rec.get(name, {}).get("n", 0)), _f(rec.get(name, {}).get("follow_rate")), _f(rec.get(name, {}).get("model_p_recommended")),
          _f(rec.get(name, {}).get("label_p_recommended")), _f(rec.get(name, {}).get("label_best_rate")), _f(rec.get(name, {}).get("regret_mean"))]
         for name in ("wrong", "right")],
    ) + [""]

    bias = summary.get("position_bias", {})
    lines += ["## Position bias", "", "Options were shuffled per record, so the best option's position is spread; a model without bias picks positions like the label does.", ""]
    choice = bias.get("choice", {})
    rows = []
    for n, entry in choice.items():
        if n == "all":
            continue
        rows.append([n, str(entry["n"]), " / ".join(str(c) for c in entry["model_argmax_positions"]), " / ".join(str(c) for c in entry["label_best_positions"]),
                     _f(entry["model_first_rate"]), _f(entry["label_first_rate"]), " / ".join(_f(v, 2) for v in entry["mean_probability_by_position"]),
                     " / ".join(_f(v, 2) for v in entry["accuracy_by_label_position"])])
    if rows:
        lines += _table(["options", "n", "model argmax at position 1..n", "label best at position 1..n", "model first", "label first", "mean p by position", "acc by best position"], rows) + [""]
    if choice.get("all"):
        a = choice["all"]
        lines += [f"All choices: model picks the first option {_f(a['model_first_rate'])} vs label {_f(a['label_first_rate'])} (1/n = {_f(a['expected_first_rate'])}); last option {_f(a['model_last_rate'])} vs {_f(a['label_last_rate'])}.", ""]
    noul, score = bias.get("noul", {}), bias.get("score", {})
    if noul:
        lines += _table(
            ["noul listing", "n", "model mean p_yes", "label mean p_yes"],
            [["Yes listed first", str(noul["yes_first"]["n"]), _f(noul["yes_first"]["model_p_yes"]), _f(noul["yes_first"]["label_p_yes"])],
             ["Yes listed second", str(noul["yes_second"]["n"]), _f(noul["yes_second"]["model_p_yes"]), _f(noul["yes_second"]["label_p_yes"])]],
        ) + ["", f"noul shift (first − second): model {_f(noul.get('shift'), signed=True)}, label {_f(noul.get('label_shift'), signed=True)}, excess {_f(noul.get('excess'), signed=True)}.", ""]
    if score:
        lines += _table(
            ["score listing", "n", "model mean value", "label mean value"],
            [["ascending (0.0-0.1 first)", str(score["ascending"]["n"]), _f(score["ascending"]["model_value"]), _f(score["ascending"]["label_value"])],
             ["descending", str(score["descending"]["n"]), _f(score["descending"]["model_value"]), _f(score["descending"]["label_value"])]],
        ) + ["", f"score shift (ascending − descending): model {_f(score.get('shift'), signed=True)}, label {_f(score.get('label_shift'), signed=True)}, excess {_f(score.get('excess'), signed=True)}.", ""]

    pairs = summary.get("pairs", {})
    lines += ["## Fact-changed pairs", ""]
    numeric = pairs.get("numeric", {})
    lines += [f"Numeric pairs (one decision variable moved across the boundary): {numeric.get('pairs', 0)} pairs, {numeric.get('questions', 0)} shared questions, "
              f"{numeric.get('moved', 0)} whose label moved; direction correct {_f(numeric.get('direction_correct_rate'))}, model did not move {_f(numeric.get('model_still_rate'))}.", ""]
    if numeric.get("by_type"):
        lines += _table(
            ["type", "compared", "label moved", "direction correct", "rate", "model still", "mean |model move|", "mean |label move|"],
            [[t, str(e["compared"]), str(e["moved"]), str(e["direction_correct"]), _f(e["direction_correct_rate"]), _f(e["model_still_rate"]), _f(e["model_move_mean"]), _f(e["label_move_mean"])]
             for t, e in numeric["by_type"].items()],
        ) + [""]
    wording = pairs.get("wording", {})
    lines += [f"Wording pairs (numbers unchanged, situation words changed): {wording.get('pairs', 0)} pairs, {wording.get('questions', 0)} shared questions; "
              f"model shift {_f(wording.get('model_shift_mean'))} (label shift {_f(wording.get('label_shift_mean'))}), decision flips {_f(wording.get('flip_rate'))}.", ""]
    if wording.get("by_type"):
        lines += _table(
            ["type", "compared", "model shift", "label shift", "flip rate"],
            [[t, str(e["compared"]), _f(e["model_shift_mean"]), _f(e["label_shift_mean"]), _f(e["flip_rate"])] for t, e in wording["by_type"].items()],
        ) + [""]

    permutation = summary.get("permutation")
    if permutation:
        lines += ["## Option-order sensitivity", "",
                  "The same records scored again with only the option order reshuffled (other shuffle seeds); an order-blind model shows 0. "
                  "Shift = total-variation distance of the key-mapped distributions (choice) or |Δ| (noul, score); flip = the verdict changed.", ""]
        lines += _table(["type", "questions", "seed pairs", "mean shift", "flip rate"],
                        [[t, str(e["questions"]), str(e["pairs"]), _f(e["shift_mean"]), _f(e["flip_rate"])] for t, e in permutation.items()]) + [""]
    effects = summary.get("order_effects")
    if effects:
        lines += ["### Paired order effects (same question, different listing)", ""]
        rows = []
        for qtype, label in (("noul", "p_yes(Yes listed first) − p_yes(Yes listed second)"), ("score", "value(ascending) − value(descending)")):
            e = effects.get(qtype) or {}
            rows.append([qtype, label, str(e.get("questions", 0)), _f(e.get("mean_shift"), signed=True), _f(e.get("positive_rate"))])
        lines += _table(["type", "effect", "questions", "mean", "share > 0"], rows) + [""]
        choice_effects = effects.get("choice") or {}
        if choice_effects:
            lines += _table(["options", "questions", "mean deviation of p by position 1..n (vs the option's own mean)"],
                            [[n, str(e["questions"]), " / ".join(_f(v, 3, signed=True) for v in e["mean_deviation_by_position"])] for n, e in choice_effects.items()]) + [""]
    averaged = summary.get("order_averaged")
    if averaged:
        lines += ["### Order-averaged reading", "",
                  f"Answers averaged over the {averaged['orders']} listings, each distinct listing weighted equally (removes first-order position bias; "
                  "exact for noul/score when both listings were seen). Compare with the primary reading to separate position bias from not knowing.", ""]
        header = ["reading", "questions"] + [name for _, _, name in KEY_COLUMNS]
        rows = []
        for name, blocks in (("primary (first seed)", summary.get("overall", {})), ("order-averaged", averaged.get("overall", {}))):
            cells = []
            for block, metric, _ in KEY_COLUMNS:
                entry = blocks.get(block)
                cells.append(f"{_f(entry.get(metric))} ({entry['n']})" if entry and entry.get(metric) is not None else "—")
            rows.append([name, str((blocks.get("_all") or {}).get("n", 0))] + cells)
        lines += _table(header, rows) + [""]
        if (averaged.get("groups") or {}).get("phenomena"):
            lines += ["Order-averaged, by phenomenon:", ""] + _key_table(averaged["groups"]["phenomena"]) + [""]
        rec = averaged.get("recommendation") or {}
        if rec:
            lines += [f"Order-averaged wrong-recommendation follow rate: {_f((rec.get('wrong') or {}).get('follow_rate'))} "
                      f"(n = {(rec.get('wrong') or {}).get('n', 0)}); right: {_f((rec.get('right') or {}).get('follow_rate'))}.", ""]

    boot = summary.get("bootstrap", {})
    if boot:
        lines += ["## Bootstrap (record-level, 95 %)", ""]
        lines += _table(["metric", "value", "95 % interval", "n", "records"],
                        [[name, _f(e["value"]), f"[{_f(e['low'])}, {_f(e['high'])}]", str(e["n"]), str(e["records"])] for name, e in boot.items()]) + [""]

    reasoning = summary.get("reasoning")
    if reasoning:
        lines += ["## Reasoning generation", ""]
        lines += _table(["questions", "mean generated tokens", "hit the cap", "wrote its own answer cue", "stops"],
                        [[str(reasoning["questions"]), _f(reasoning["generated_tokens_mean"], 1), _f(reasoning["cap_rate"]), _f(reasoning["self_answer_rate"]),
                          ", ".join(f"{k} {v}" for k, v in reasoning["stops"].items())]]) + [""]

    definitions = summary.get("definitions") or {}
    if definitions:
        lines += ["## Definitions", ""] + [f"- **{k}**: {v}" for k, v in definitions.items()] + [""]
    return "\n".join(lines)


def render_comparison(comparison: dict[str, Any], *, label_a: str, label_b: str, title: str, note: str | None = None) -> str:
    """두 run의 짝지은 비교(:func:`robo_jev.decisions.metrics.compare_runs`) → markdown 표 (차 = a − b, 레코드 bootstrap 95 %)."""
    lines = [f"# {title}", ""]
    if note:
        lines += [f"> **{note}**", ""]
    lines += [f"a = {label_a}, b = {label_b}; {comparison.get('records', 0)} records, {comparison.get('questions', 0)} questions scored by both. Difference = a − b.", ""]
    rows = []
    for name, diff in comparison.get("differences", {}).items():
        rows.append([name, _f(comparison["a"].get(name)), _f(comparison["b"].get(name)), f"{_f(diff['value'], signed=True)} [{_f(diff['low'], signed=True)}, {_f(diff['high'], signed=True)}]", str(diff["n"])])
    lines += _table(["metric", label_a, label_b, "a − b [95 %]", "n"], rows) + [""]
    return "\n".join(lines)
