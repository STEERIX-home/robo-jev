"""요약 JSON → 사람이 읽는 markdown 표 (Task v0c §2.4; fix round 1 — 두 읽기, 기준선 대비 짝지은 구간, 기준 계열별 후회, 신호·어려움 표).

숫자는 :func:`robo_jev.decisions.metrics.summarize_all` 의 값 그대로(소수 셋째 자리)이고, 없는 칸은 ``—``다. 미리보기 run은
:data:`PREVIEW_NOTE` 를 머리에 단다 — 생성 표본의 숫자이지 봉인된 시험 세트(T1~T7, 다음 단계 v0b)의 숫자가 아니다. 설명 뒤 답 run이
완료율 문턱에 못 미치면 머리에 "판정 불가"를 단다.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PREVIEW_NOTE", "render_comparison", "render_markdown"]

PREVIEW_NOTE = (
    "PREVIEW on a generated sample (v0a `sample-2000.jsonl`), not the sealed test sets T1-T7 (sealed in v0b). "
    "Read the numbers as a first look at where and how the untrained model errs, not as the baseline of record."
)
_READING_NAMES = {"single_order": "single order (deployment)", "order_averaged": "order-averaged (what it knows)"}
#: 묶음 표의 핵심 열 — (묶음, 지표, 열 이름).
KEY_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("noul", "brier", "noul Brier"),
    ("noul", "brier_vs_uniform", "noul Brier − 0.5"),
    ("score", "brier", "score Brier"),
    ("score", "brier_vs_uniform", "score Brier − 0.5"),
    ("choice/action", "accuracy", "action acc"),
    ("choice/action", "accuracy_vs_uniform", "action acc − random"),
    ("choice/action", "regret_norm_mean", "action regret/range"),
    ("choice/state", "accuracy", "state acc"),
    ("choice/rule", "accuracy", "rule acc"),
)
_BLOCK_COLUMNS = {
    "noul": (("brier", "Brier"), ("kl", "KL"), ("log_loss", "log loss"), ("ece", "ECE"), ("pearson", "r(p, label)"), ("side_agreement", "same side"), ("mean_p", "mean p"), ("mean_y", "mean label"), ("code_mass", "code mass")),
    "score": (("brier", "Brier"), ("kl", "KL"), ("log_loss", "log loss"), ("ece", "ECE"), ("pearson", "r(value, label)"), ("side_agreement", "same side"), ("mean_p", "mean value"), ("mean_y", "mean label"), ("code_mass", "code mass")),
    "choice/action": (("accuracy", "acc (min cost)"), ("ce", "CE"), ("kl", "KL"), ("regret_norm_mean", "regret/range"), ("regret_mean", "regret"), ("regret_median", "median regret"), ("regret_positive_rate", "regret>0"), ("code_mass", "code mass")),
    "choice/state": (("accuracy", "acc"), ("ce", "CE"), ("kl", "KL"), ("ece", "ECE (top-label)"), ("code_mass", "code mass")),
    "choice/rule": (("accuracy", "acc"), ("ce", "CE"), ("code_mass", "code mass")),
    "choice/other": (("accuracy", "acc"), ("ce", "CE"), ("kl", "KL"), ("code_mass", "code mass")),
}
#: 질문 종류 표에 붙이는 기준선 대비 짝지은 구간 (bootstrap 머리 지표 이름).
_VS_UNIFORM = {
    "noul": ("noul.brier", "Brier − always 0.5"), "score": ("score.brier", "Brier − always 0.5"),
    "choice/action": ("choice/action.accuracy", "acc − random pick"), "choice/state": ("choice/state.accuracy", "acc − random pick"),
    "choice/rule": ("choice/rule.accuracy", "acc − random pick"),
}
_GROUP_ORDER = (
    ("phenomena", "By phenomenon"), ("scene", "By scene"), ("domain", "By domain"), ("body", "By body"), ("sensor_suite", "By sensor suite"),
    ("style", "By style"), ("hint", "By computed hint"), ("recommendation", "By planner recommendation (all questions of those records)"),
    ("criterion", "By action criterion"), ("family", "By question family"), ("pair", "By pair membership"),
    ("label_position", "By position of the best option (choice only; 1 = first listed)"), ("key_style", "By option-key style"),
    ("agent", "By moving agent"), ("distractors", "By number of irrelevant facts"), ("distance_unit", "By distance unit"),
)
_DIFFICULTY = (
    ("margin_norm_band", "Normalized decision margin |m / σ_m| (v0a `meta.margin_norm`; near < 0.5 ≤ mid < 1.5 ≤ far)"),
    ("key_entropy_band", "Label entropy of the record's key question (bits; low < 1/3 ≤ mid < 2/3 ≤ high = near 0.5)"),
    ("label_entropy_band", "Label entropy of the question itself"),
    ("margin_band", "Legacy: raw-margin tertiles per margin name (units differ; not a difficulty axis)"),
)


def _f(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(entry: dict[str, Any] | None, digits: int = 3) -> str:
    """``{"value", "low", "high"}`` → ``+0.012 [−0.003, +0.027]``."""
    if not entry:
        return "—"
    return f"{_f(entry.get('value'), digits, True)} [{_f(entry.get('low'), digits, True)}, {_f(entry.get('high'), digits, True)}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def _key_table(groups: dict[str, Any]) -> list[str]:
    header = ["group", "questions"] + [name for _, _, name in KEY_COLUMNS]
    rows = []
    for value, blocks in groups.items():
        cells = []
        for block, metric, _ in KEY_COLUMNS:
            entry = blocks.get(block)
            signed = metric.endswith("_vs_uniform")
            cells.append(f"{_f(entry.get(metric), signed=signed)} ({entry['n']})" if entry and entry.get(metric) is not None else "—")
        rows.append([str(value), str((blocks.get("_all") or {}).get("n", 0))] + cells)
    return _table(header, rows)


def _signal_table(signal: dict[str, Any]) -> list[str]:
    """묶음 값마다: noul·score r과 Brier − 0.5 [CI], 행동 정확도 − 무작위 [CI]와 척도 없는 후회 − 무작위 [CI]."""
    header = ["group", "noul n", "noul r", "noul Brier − 0.5 [95 %]", "score n", "score r", "score Brier − 0.5 [95 %]", "action n",
              "action acc − random [95 %]", "action regret/range − random [95 %]"]
    rows = []
    for value, blocks in signal.items():
        noul, score, action = blocks.get("noul") or {}, blocks.get("score") or {}, blocks.get("choice/action") or {}
        rows.append([
            str(value), str(noul.get("n", "—")), _f(noul.get("pearson"), 2, True), _ci((noul.get("brier") or {}).get("vs_uniform")),
            str(score.get("n", "—")), _f(score.get("pearson"), 2, True), _ci((score.get("brier") or {}).get("vs_uniform")),
            str(action.get("n", "—")), _ci((action.get("accuracy") or {}).get("vs_uniform")), _ci((action.get("regret_norm") or {}).get("vs_uniform")),
        ])
    return _table(header, rows)


def _gate_banner(stats: dict[str, Any] | None) -> list[str]:
    if not stats:
        return []
    gate = stats["gate"]
    if gate["passed"]:
        return [f"> Explanation-first completion rate {_f(stats['complete_rate'])} ≥ {gate['threshold']:.2f}: the reasoning comparison can be read.", ""]
    return [f"> **INCONCLUSIVE — explanation-first completion rate {_f(stats['complete_rate'])} < {gate['threshold']:.2f}.** "
            "Too many explanations hit the token cap before answering; do not read this run's reasoning-vs-direct difference.", ""]


def _overall_section(readings: dict[str, Any]) -> list[str]:
    lines = ["## By question type (both readings)", "",
             "Baseline = always 0.5 (noul, score) or a random pick (choice); the last column is the paired record-bootstrap 95 % interval of "
             "model − baseline on the same questions.", ""]
    first = next(iter(readings.values()))
    for block, columns in _BLOCK_COLUMNS.items():
        if not any((summary.get("overall") or {}).get(block) for summary in readings.values()):
            continue
        boot_name, boot_label = _VS_UNIFORM.get(block, (None, None))
        header = ["reading", "n"] + [name for _, name in columns] + (["baseline", f"{boot_label} [95 %]"] if boot_name else [])
        rows = []
        for reading, summary in readings.items():
            entry = (summary.get("overall") or {}).get(block)
            if not entry:
                continue
            row = [_READING_NAMES.get(reading, reading), str(entry["n"])] + [_f(entry.get(key), signed=key == "pearson") for key, _ in columns]
            if boot_name:
                boot = (summary.get("bootstrap") or {}).get(boot_name) or {}
                row += [_f(boot.get("uniform")), _ci(boot.get("vs_uniform"))]
            rows.append(row)
        lines += [f"**{block}**", ""] + _table(header, rows) + [""]
    all_entry = (first.get("overall") or {}).get("_all", {})
    lines += [f"All scored questions: n = {all_entry.get('n', 0)}, mean code mass = {_f(all_entry.get('code_mass'))}.", ""]
    return lines


def _regret_section(readings: dict[str, Any]) -> list[str]:
    lines = ["## Action regret by criterion family", "",
             "Stated-criterion questions (`avoid_*`) cost violations near 1,000 units, a different scale from the default costs, so the pooled "
             "mean is dominated by them. Read the default row, the scale-free regret (regret ÷ the question's expected-cost range) and the median.", ""]
    header = ["reading", "criterion", "n", "mean", "median", "max", "top-10 share", "regret>0", "regret/range", "random pick: mean", "random pick: regret/range"]
    rows = []
    for reading, summary in readings.items():
        for criterion, e in (summary.get("regret_by_criterion") or {}).items():
            rows.append([_READING_NAMES.get(reading, reading), criterion, str(e.get("regret_n", 0)), _f(e.get("regret_mean"), 2), _f(e.get("regret_median"), 2),
                         _f(e.get("regret_max"), 1), _f(e.get("regret_top10_share")), _f(e.get("regret_positive_rate")), _f(e.get("regret_norm_mean")),
                         _f(e.get("uniform_regret_mean"), 2), _f(e.get("uniform_regret_norm_mean"))])
    lines += _table(header, rows) + [""]
    boot_rows = []
    for reading, summary in readings.items():
        boot = summary.get("bootstrap") or {}
        for name in ("choice/action.regret_mean[default]", "choice/action.regret_mean[stated]", "choice/action.regret_norm", "choice/action.regret_norm[default]", "choice/action.regret_norm[stated]"):
            if name in boot:
                boot_rows.append([_READING_NAMES.get(reading, reading), name, _f(boot[name]["value"]), f"[{_f(boot[name]['low'])}, {_f(boot[name]['high'])}]",
                                  _f(boot[name].get("uniform")), _ci(boot[name].get("vs_uniform"))])
    if boot_rows:
        lines += _table(["reading", "metric", "value", "95 %", "random pick", "model − random [95 %]"], boot_rows) + [""]
    return lines


def render_markdown(summary: dict[str, Any], *, config: dict[str, Any], title: str, note: str | None = None) -> str:
    """:func:`robo_jev.decisions.metrics.summarize_all` 의 요약 → markdown 보고서 (모듈 설명)."""
    readings = summary.get("readings") or {}
    lines = [f"# {title}", ""]
    if note:
        lines += [f"> **{note}**", ""]
    lines += _gate_banner(summary.get("reasoning"))
    first = next(iter(readings.values()), {})
    counts = first.get("counts", {})
    lines += _table(
        ["model", "mode", "format", "prompt", "orders", "records", "questions scored", "input"],
        [[str(config.get("model", "—")), str(config.get("mode", "—")), str(config.get("format", "—")), str(config.get("prompt_version", "—")),
          str(config.get("orders", "—")), str(counts.get("records", "—")), f"{counts.get('scored', '—')} / {counts.get('questions', '—')}",
          f"`{config.get('input', '—')}` (sha256 {str(config.get('input_sha256', ''))[:12]})"]],
    )
    if config.get("selection"):
        lines += ["", f"Selection: {config['selection']}"]
    listings = (summary.get("order") or {}).get("listings") or {}
    if listings:
        lines += ["", "Listings per question: " + "; ".join(
            f"{t} {_f(e.get('mean_listings'), 2)} (≥ 2 distinct: {_f(e.get('multi_listing_rate'))})" for t, e in listings.items() if t != "merge"
        ) + (f"; merged runs {listings['merge'].get('runs')}, identical listings skipped {listings['merge'].get('duplicate_listings_skipped')}" if listings.get("merge") else "")]
    lines += ["", "Two readings: **single order** = the metric of each listing, averaged over a question's listings (what a deployed System 1 "
              "gets when asked in one order); **order-averaged** = the metric of the listing-averaged answer (first-order position bias removed).", ""]
    lines += _overall_section(readings)
    lines += _regret_section(readings)

    for reading in ("single_order", "order_averaged"):
        summary_r = readings.get(reading)
        if summary_r and summary_r.get("signal", {}).get("scene"):
            lines += [f"## Signal by scene — {_READING_NAMES[reading]}", "", "r = weighted Pearson correlation of the model value with the label; "
                      "the intervals are paired record-bootstrap model − baseline within the scene.", ""] + _signal_table(summary_r["signal"]["scene"]) + [""]
    for reading in ("single_order", "order_averaged"):
        summary_r = readings.get(reading)
        if not summary_r:
            continue
        signal = summary_r.get("signal", {})
        parts = [(dim, heading) for dim, heading in _DIFFICULTY if signal.get(dim)]
        if parts:
            lines += [f"## Difficulty axes — {_READING_NAMES[reading]}", "", "Each band compared with the baseline on the same questions, so label "
                      "extremity cannot pose as a gradient (an extreme label is easy for the model and for the baseline alike).", ""]
            for dim, heading in parts:
                lines += [f"**{heading}**", ""] + _signal_table(signal[dim]) + [""]
        if signal.get("criterion"):
            lines += [f"**Action criterion — {_READING_NAMES[reading]}**", ""] + _signal_table(signal["criterion"]) + [""]

    single = readings.get("single_order") or first
    groups = single.get("groups", {})
    for dim, heading in _GROUP_ORDER:
        if dim in groups and groups[dim]:
            lines += [f"## {heading} — single order", ""] + _key_table(groups[dim]) + [""]

    lines += ["## Planner recommendation", "",
              "Scope: the recommendation's target question (`meta.recommendation.question`; otherwise action choices offering the recommended key). "
              "Follow = model argmax is the recommended option; 'random pick' = 1/n. The by-recommendation group table above covers all action "
              "questions of those records instead.", ""]
    rec_rows = []
    for reading, summary_r in readings.items():
        rec = summary_r.get("recommendation") or {}
        for name in ("wrong", "right"):
            e = rec.get(name, {})
            rec_rows.append([_READING_NAMES.get(reading, reading), name, str(e.get("n", 0)), _f(e.get("follow_rate")), _f(e.get("uniform_follow_rate")),
                             _f(e.get("model_p_recommended")), _f(e.get("label_p_recommended")), _f(e.get("label_best_rate")), _f(e.get("regret_mean"))])
    lines += _table(["reading", "recommendation", "n", "follow rate", "random pick", "model p(recommended)", "label p(recommended)",
                     "recommended is best (label)", "model regret"], rec_rows) + [""]

    bias = single.get("position_bias") or {}
    if bias:
        lines += ["## Position bias (single-order rows)", "", "With counterbalanced listings every option appears in every position equally "
                  "often, so the label's best option is spread evenly; a model without bias picks positions like the label does.", ""]
        rows = []
        for n, entry in (bias.get("choice") or {}).items():
            if n == "all":
                continue
            rows.append([n, str(entry["n"]), " / ".join(_f(v, 2) for v in entry["model_argmax_position_rates"]),
                         " / ".join(_f(v, 2) for v in entry["label_best_position_rates"]), " / ".join(_f(v, 2) for v in entry["mean_probability_by_position"]),
                         " / ".join(_f(v, 2) for v in entry["accuracy_by_label_position"])])
        if rows:
            lines += _table(["options", "questions", "model argmax at position 1..n", "label best at position 1..n", "mean p by position", "acc by best position"], rows) + [""]
        a = (bias.get("choice") or {}).get("all")
        if a:
            lines += [f"All choices: model picks the first option {_f(a['model_first_rate'])} vs label {_f(a['label_first_rate'])} (1/n = {_f(a['expected_first_rate'])}); "
                      f"last option {_f(a['model_last_rate'])} vs {_f(a['label_last_rate'])}.", ""]
    order = summary.get("order") or {}
    effects = order.get("order_effects") or {}
    if effects:
        lines += ["### Paired order effects (same question, different listings)", ""]
        rows = []
        for qtype, label in (("noul", "p_yes(Yes listed first) − p_yes(Yes listed second)"), ("score", "value(ascending) − value(descending)")):
            e = effects.get(qtype) or {}
            rows.append([qtype, label, str(e.get("questions", 0)), _f(e.get("mean_shift"), signed=True), _f(e.get("positive_rate"))])
        lines += _table(["type", "effect", "questions", "mean", "share > 0"], rows) + [""]
        if effects.get("choice"):
            lines += _table(["options", "questions", "mean deviation of p by position 1..n (vs the option's own mean)"],
                            [[n, str(e["questions"]), " / ".join(_f(v, 3, signed=True) for v in e["mean_deviation_by_position"])] for n, e in effects["choice"].items()]) + [""]
    permutation = order.get("permutation") or {}
    if permutation:
        lines += ["### Verdict changes between listings", "", "Pairs of listings of the same question whose option order differs (identical "
                  "listings are skipped and counted). Shift = total-variation distance of the key-mapped distributions (choice) or |Δ| (noul, score).", ""]
        lines += _table(["type", "questions", "distinct-listing pairs", "identical pairs skipped", "mean shift", "flip rate"],
                        [[t, str(e["questions"]), str(e["pairs"]), str(e.get("identical_pairs_skipped", 0)), _f(e["shift_mean"]), _f(e["flip_rate"])] for t, e in permutation.items()]) + [""]

    pairs = single.get("pairs", {})
    lines += ["## Fact-changed pairs (order-averaged answers)", ""]
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

    for reading, summary_r in readings.items():
        boot = summary_r.get("bootstrap") or {}
        if boot:
            lines += [f"## Bootstrap (record-level, 95 %) — {_READING_NAMES.get(reading, reading)}", ""]
            lines += _table(["metric", "value", "95 %", "n", "records", "baseline", "model − baseline [95 %]"],
                            [[name, _f(e["value"]), f"[{_f(e['low'])}, {_f(e['high'])}]", str(e["n"]), str(e["records"]), _f(e.get("uniform")), _ci(e.get("vs_uniform"))]
                             for name, e in boot.items()]) + [""]

    reasoning = summary.get("reasoning")
    if reasoning:
        lines += ["## Explanation-first generation", ""]
        lines += _table(["questions", "mean generated tokens", "completed on its own", "hit the cap", "wrote its own answer cue", "stops", "gate"],
                        [[str(reasoning["questions"]), _f(reasoning["generated_tokens_mean"], 1), _f(reasoning["complete_rate"]), _f(reasoning["cap_rate"]),
                          _f(reasoning["self_answer_rate"]), ", ".join(f"{k} {v}" for k, v in reasoning["stops"].items()),
                          f"{'passed' if reasoning['gate']['passed'] else 'INCONCLUSIVE'} (≥ {reasoning['gate']['threshold']:.2f})"]]) + [""]

    definitions = summary.get("definitions") or {}
    if definitions:
        lines += ["## Definitions", ""] + [f"- **{k}**: {v}" for k, v in definitions.items()] + [""]
    return "\n".join(lines)


def render_comparison(comparison: dict[str, Any], *, label_a: str, label_b: str, title: str, note: str | None = None) -> str:
    """두 run의 짝지은 비교(:func:`robo_jev.decisions.metrics.compare_runs`) → markdown 표 (두 읽기; 차 = a − b, 레코드 bootstrap 95 %)."""
    lines = [f"# {title}", ""]
    if note:
        lines += [f"> **{note}**", ""]
    gate = comparison.get("gate") or {}
    if gate and not gate.get("passed", True):
        failing = [(side, g) for side, g in (("a", gate.get("a")), ("b", gate.get("b"))) if g and not g.get("passed")]
        text = ", ".join(f"{side} {_f(g.get('complete_rate'))} < {g['threshold']:.2f}" for side, g in failing)
        lines += [f"> **INCONCLUSIVE — explanation-first completion rate below the gate ({text}).** Do not read the differences below "
                  "as a reasoning-vs-direct result.", ""]
    lines += [f"a = {label_a}, b = {label_b}; {comparison.get('records', 0)} records. Difference = a − b, paired record bootstrap.", ""]
    for reading, result in (comparison.get("readings") or {}).items():
        rows = []
        for name, diff in result.get("differences", {}).items():
            rows.append([name, _f(result["a"].get(name)), _f(result["b"].get(name)), _ci(diff), str(diff["n"])])
        lines += [f"## {_READING_NAMES.get(reading, reading)} — {result.get('questions', 0)} questions scored by both", ""]
        lines += _table(["metric", label_a, label_b, "a − b [95 %]", "n"], rows) + [""]
    return "\n".join(lines)

