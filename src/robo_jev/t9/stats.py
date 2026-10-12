"""T9 label statistics (Task T9 §1-4 "Label statistics") — instrument properties only, no model answers.

* **Band profile** — yes/no and score labels in low (< 1/3), middle, high (> 2/3), overall, per family and per question kind (key,
  cf, alt), and the key label's histogram in tenths.
* **Robust share** — questions whose band (yes/no, score) or argmax (choices) is the same under all three priors, per family × kind.
* **Pair moves** — the one-fact pairs (records a/b of a group): |Δ key| histogram, share ≥ 0.1, direction the same under all priors
  (up / down / flat at ±0.05) and never reversed; the in-record direction pairs (key → cf, key → alt): share with |Δ| ≥ 0.1 and
  share whose move holds (same sign, ≥ 0.1) under every prior.
* **Direction balance** — the direction question's labels (likelier / less likely / no real change) per family, and how often the
  label matches the intended class the generator aimed at.
* **Outcome and action balance** — argmax of the outcome question and of the action question per family.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from robo_jev.t9 import text as T

__all__ = ["label_statistics", "render_markdown"]

KINDS = ("key", "cf", "alt", "dir", "act", "outcome")


def _band(x: float) -> str:
    return "low" if x < 1 / 3 else ("high" if x > 2 / 3 else "middle")


def _scalar(label: dict[str, Any]) -> float:
    return float(label["p_yes"] if label["type"] == "noul" else label["value"])


def _argmax(probabilities: dict[str, float]) -> tuple[str, ...]:
    top = max(probabilities.values())
    return tuple(sorted(k for k, v in probabilities.items() if abs(v - top) <= 1e-12))


def label_statistics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({r["meta"]["family"] for r in records}, key=lambda f: list(T.NAMES).index(f))
    bands: dict[str, Counter] = defaultdict(Counter)
    key_hist: dict[str, list[int]] = defaultdict(lambda: [0] * 10)
    robust: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    direction: dict[str, Counter] = defaultdict(Counter)
    intent_match: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    outcome: dict[str, Counter] = defaultdict(Counter)
    action: dict[str, Counter] = defaultdict(Counter)
    in_record: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # pairs, |Δ| ≥ 0.1, robust move
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups_of_law = Counter()
    for r in records:
        meta, fam = r["meta"], r["meta"]["family"]
        names = T.NAMES[fam]
        kind_of = {v: k for k, v in names.items()}
        by_group[r["id"][:-1]].append(r)
        groups_of_law[meta["law_group"]] += 1
        for name, label in r["labels"].items():
            kind = kind_of[name]
            rob = meta["robustness"][name]["robust"]
            robust[(fam, kind)][0] += int(bool(rob))
            robust[(fam, kind)][1] += 1
            if label["type"] in ("noul", "score"):
                b = _band(_scalar(label))
                for scope in ("all", fam, f"kind:{kind}"):
                    bands[scope][b] += 1
                if kind == "key":
                    key_hist[fam][min(9, int(_scalar(label) * 10))] += 1
                    key_hist["all"][min(9, int(_scalar(label) * 10))] += 1
        dir_label = _argmax(r["labels"][names["dir"]]["probabilities"])[0]
        direction[fam][dir_label] += 1
        direction["all"][dir_label] += 1
        intent_match[fam][0] += int(dir_label == meta["changes"]["dir_intent"])
        intent_match[fam][1] += 1
        out_top = _argmax(r["labels"][names["outcome"]]["probabilities"])
        outcome[fam]["+".join(out_top)] += 1
        act_top = _argmax(r["labels"][names["act"]]["probabilities"])
        action[fam]["+".join(act_top)] += 1
        lbp = meta["labels_by_prior"]
        for other in (names["cf"], names["alt"]):
            base_move = _scalar(r["labels"][other]) - _scalar(r["labels"][names["key"]])
            in_record[fam][0] += 1
            if abs(base_move) >= 0.1:
                in_record[fam][1] += 1
                moves = [_scalar(lbp[p][other]) - _scalar(lbp[p][names["key"]]) for p in lbp]
                if all(m >= 0.1 for m in moves) or all(m <= -0.1 for m in moves):
                    in_record[fam][2] += 1
    pairs: dict[str, dict[str, Any]] = {}
    for fam in families + ["all"]:
        rows = [g for g in by_group.values() if len(g) == 2 and (fam == "all" or g[0]["meta"]["family"] == fam)]
        hist = [0] * 10
        moved = same = no_rev = 0
        for a, b in rows:
            pair = a["meta"]["pair"]
            delta = abs(pair["key_delta"])
            hist[min(9, int(delta * 10))] += 1
            moved += delta >= 0.1
            same += bool(pair["direction_same"])
            no_rev += bool(pair["no_reversal"])
        n = len(rows)
        pairs[fam] = {"pairs": n, "abs_key_delta_tenths": hist, "moved_ge_0.1": moved / n if n else None, "direction_same_all_priors": same / n if n else None,
                      "never_reversed": no_rev / n if n else None}
    out: dict[str, Any] = {
        "records": len(records), "groups": len(by_group), "families": families, "records_by_law_group": dict(groups_of_law),
        "band_profile": {scope: {b: c[b] / max(sum(c.values()), 1) for b in ("low", "middle", "high")} | {"n": sum(c.values())} for scope, c in bands.items()},
        "key_label_tenths": dict(key_hist),
        "robust_share": {fam: {kind: (robust[(fam, kind)][0] / robust[(fam, kind)][1]) if robust[(fam, kind)][1] else None for kind in KINDS} for fam in families},
        "robust_share_all": {kind: sum(robust[(f, kind)][0] for f in families) / max(sum(robust[(f, kind)][1] for f in families), 1) for kind in KINDS},
        "robust_share_total": sum(v[0] for v in robust.values()) / max(sum(v[1] for v in robust.values()), 1),
        "pairs": pairs,
        "in_record_moves": {fam: {"pairs": v[0], "moved_ge_0.1": v[1] / v[0] if v[0] else None, "robust_among_moved": v[2] / v[1] if v[1] else None}
                            for fam, v in in_record.items()},
        "direction_balance": {fam: dict(c) for fam, c in direction.items()},
        "direction_matches_intent": {fam: v[0] / v[1] for fam, v in intent_match.items()},
        "outcome_balance": {fam: dict(c) for fam, c in outcome.items()},
        "action_balance": {fam: dict(c) for fam, c in action.items()},
    }
    return out


def _f(x: Any, digits: int = 2) -> str:
    return "—" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# T9 — label statistics", "", f"{report['records']} records in {report['groups']} one-fact pairs; by law group: "
             + ", ".join(f"{k} {v}" for k, v in report["records_by_law_group"].items()) + ".", "",
             "## Band profile (yes/no and score labels, base prior)", "", "| scope | n | low | middle | high |", "|---|---|---|---|---|"]
    for scope, row in report["band_profile"].items():
        lines.append(f"| {scope} | {row['n']} | {_f(row['low'])} | {_f(row['middle'])} | {_f(row['high'])} |")
    lines += ["", "## Key label in tenths (base prior)", "", "| family | " + " | ".join(str(i) for i in range(10)) + " |", "|---|" + "---|" * 10]
    for fam, hist in report["key_label_tenths"].items():
        lines.append(f"| {fam} | " + " | ".join(str(v) for v in hist) + " |")
    lines += ["", "## Robust share (band or argmax the same under all three priors)", "", "| family | " + " | ".join(KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for fam, row in report["robust_share"].items():
        lines.append(f"| {fam} | " + " | ".join(_f(row[k]) for k in KINDS) + " |")
    lines.append("| all | " + " | ".join(_f(report["robust_share_all"][k]) for k in KINDS) + " |")
    lines += ["", f"All questions: {_f(report['robust_share_total'])} robust.", "", "## One-fact pairs (record a → b)", "",
              "| family | pairs | |Δ key| ≥ 0.1 | direction same under all priors | never reversed | |Δ| in tenths |", "|---|---|---|---|---|---|"]
    for fam, row in report["pairs"].items():
        lines.append(f"| {fam} | {row['pairs']} | {_f(row['moved_ge_0.1'])} | {_f(row['direction_same_all_priors'])} | {_f(row['never_reversed'])} | "
                     + " ".join(str(v) for v in row["abs_key_delta_tenths"]) + " |")
    lines += ["", "## In-record direction pairs (key → cf, key → alt)", "", "| family | pairs | |Δ| ≥ 0.1 | holds under all priors (of moved) |", "|---|---|---|---|"]
    for fam, row in report["in_record_moves"].items():
        lines.append(f"| {fam} | {row['pairs']} | {_f(row['moved_ge_0.1'])} | {_f(row['robust_among_moved'])} |")
    lines += ["", "## Balance (argmax of the base-prior label)", "", "| family | direction | matches intent | outcome | action |", "|---|---|---|---|---|"]
    for fam in report["outcome_balance"]:
        lines.append(f"| {fam} | {report['direction_balance'].get(fam)} | {_f(report['direction_matches_intent'].get(fam))} | {report['outcome_balance'][fam]} | "
                     f"{report['action_balance'][fam]} |")
    return "\n".join(lines) + "\n"
