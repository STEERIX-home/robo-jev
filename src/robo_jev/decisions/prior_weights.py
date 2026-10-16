"""H-prior loss weights (Task v0.8a, the FP arm of the PRE-DECLARED v0.8 VERDICT RULE) — every label class of a question family carries equal
total weight.

**Why.** Trained models fall back to the generator's label base rates on unfamiliar inputs ("low risk", "stop / wait / hold"). The FP arm
weights the training loss so that, within each **question family**, every **label class** carries the same total weight; the weights have
mean 1 per family (the amount of training is unchanged) and are capped. Positions and codes are **not** weighted (the format package handles
them in both arms). The trainer turns this on with ``label_prior_weights: true`` (:mod:`robo_jev.decisions.train`); off, nothing here runs.

**Family** (:func:`family_of`) = source × question name. The source (:func:`source_of`) is the record's ``meta.set`` (the MuJoCo sets and
authors), else its general source (``meta.source.set``: hellaswag, qasc, …), else its generator (``generator.version`` — the v0b engine —,
``source_generator`` or ``converter`` — the d1 mix). The question name drops trailing ``_<digits>`` groups (``s1_band_3`` → ``s1_band``,
``s1_cmp_1_2`` → ``s1_cmp``, ``action_2`` → ``action``): indices of the same question, not different questions.

**Class** (:func:`label_class`):

=================================================  ========================================================================
question                                            class
=================================================  ========================================================================
yes/no (``noul``) and score                         the label's band: ``low`` (< 1/3), ``middle`` (1/3–2/3), ``high`` (> 2/3)
level questions (choice ``s1_band*``)               the best level (the label's argmax key: low / medium / high)
action questions (choice, label kind ``action``)    the kind of the best option (:func:`action_kind`): ``hold`` / ``continue`` / ``move``
other choice questions                              the answer key (the label's argmax key; for a v0.8 re-keyed general item its
                                                    original key, ``meta.format.expanded[q].keys``)
=================================================  ========================================================================

The best option is the label's argmax (ties: the smallest key, as the v0.7 LOCAL PLAN rule).

**Action kinds** (:func:`action_kind`) — fixed here and tested on real option texts in English and Korean:

* ``hold`` — **stop / wait / hold**: the robot comes to rest or holds off acting — stop at any braking rate, stop before a point, wait or hold
  where it is, re-check before acting (look again, measure first, check how the weight sits), stop and hand over to a person;
* ``continue`` — **carry on unchanged**: keep the current motion or the stated plan with no new manoeuvre and no added step — keep going,
  the plan as written / as planned, the usual way (take the ramp as usual, set off normally, take the corner at the current speed, put it
  down as planned), go now, decide now without another check;
* ``move`` — **other move**: any other action — slow down, creep, swerve or sidestep, turn slower, start gently or quickly, secure the
  load first, detour, re-centre, place elsewhere, press, push, redo, retract, and every non-plan alternative of the MuJoCo authors.

The rule reads the record's own structure first: the engine's canonical option names (``meta.options[question][key]``,
:data:`CANONICAL_KINDS`) and the MuJoCo authors' plan (``meta.plan.options``: the ``"planned"`` option → ``continue``, an alternative →
``move`` unless its text is a stop or a wait). Without that structure it reads the option text (:func:`text_action_kind`: the key and its
description, English and Korean patterns). On every action option of v0.7's training input the text rule alone agrees with the structure
on all engine options (English and Korean) and on all MuJoCo options but the A1 plan option ``stop_on_ramp`` ("ramp A at 0.46 m/s" — the
planned descent, whose key reads like a stop).

**Weights** (:func:`class_weights`) within a family with class counts n_c (examples of the epoch plan, :func:`plan_weights`): w_c =
min(cap, λ / n_c) with λ chosen so that the family's mean weight is 1 — every uncapped class carries total weight λ, a capped one cap · n_c
(water-filling; a one-class family keeps weight 1). :func:`weight_table` reports per family: classes, counts, weights, min, max and the
effective sample size (Σw)² / Σw².
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ACTION_KINDS",
    "BAND_EDGES",
    "CANONICAL_KINDS",
    "DEFAULT_CAP",
    "PRIOR_WEIGHTS_VERSION",
    "action_kind",
    "base_name",
    "best_key",
    "class_weights",
    "family_of",
    "is_action_question",
    "label_class",
    "plan_weights",
    "source_of",
    "text_action_kind",
    "weight_table",
]

PRIOR_WEIGHTS_VERSION = "hprior-1"
#: The cap of a weight (module docstring) — fixed before any training (v0.8a report).
DEFAULT_CAP = 5.0
#: The band edges of a yes/no or score label (the same as :data:`robo_jev.bench.system1.BAND_EDGES`).
BAND_EDGES = (1.0 / 3.0, 2.0 / 3.0)
#: The three kinds of an action option (module docstring).
ACTION_KINDS = ("hold", "continue", "move")

#: The engine's canonical action names (``meta.options[question][key]``) → kind. Every canonical name of the v0b engine's catalog
#: (``robo_jev.simgen.rollout``, ``p3``, ``p4``, ``p6``; texts in ``phrases_v01.NEW_DESCRIPTIONS``) is listed — an unknown name is an error.
CANONICAL_KINDS: dict[str, str] = {
    # stop / wait / hold
    "stop_gentle": "hold", "stop_moderate": "hold", "stop_hard": "hold", "stop_before": "hold", "wait": "hold", "look_again": "hold",
    "call_human": "hold", "check_weight": "hold",
    # carry on unchanged
    "continue": "continue", "turn_100": "continue", "ramp_normal": "continue", "place_now": "continue", "start_normal": "continue",
    "go_now": "continue", "decide_now": "continue",
    # other move
    "slow": "move", "creep": "move", "sidestep_left": "move", "sidestep_right": "move", "retract": "move", "turn_80": "move", "turn_60": "move",
    "turn_45": "move", "turn_30": "move", "start_gentle": "move", "start_quick": "move", "secure_first": "move", "ramp_slow": "move",
    "detour": "move", "recentre": "move", "place_elsewhere": "move", "press_down": "move", "push_harder": "move",
    "reseat": "move",
}

_INDEX_SUFFIX = re.compile(r"(_\d+)+$")


def base_name(name: str) -> str:
    """The question name without trailing ``_<digits>`` groups (module docstring)."""
    return _INDEX_SUFFIX.sub("", str(name)) or str(name)


def source_of(record: Mapping[str, Any]) -> str:
    """The record's source (module docstring): ``set:<meta.set>``, ``general:<meta.source.set>`` or ``gen:<generator>`` (else ``unknown``)."""
    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    if isinstance(meta.get("set"), str) and meta["set"]:
        return f"set:{meta['set']}"
    source = meta.get("source")
    if isinstance(source, Mapping) and isinstance(source.get("set"), str) and source["set"]:
        return f"general:{source['set']}"
    generator = record.get("generator") if isinstance(record.get("generator"), Mapping) else {}
    for key in ("version", "source_generator", "converter"):
        if isinstance(generator.get(key), str) and generator[key]:
            return f"gen:{generator[key]}"
    return "unknown"


def family_of(record: Mapping[str, Any], name: str) -> str:
    """``<source>|<base question name>`` (module docstring)."""
    return f"{source_of(record)}|{base_name(name)}"


def _band(value: float) -> str:
    x = float(value)
    return "low" if x < BAND_EDGES[0] else "high" if x > BAND_EDGES[1] else "middle"


def best_key(probabilities: Mapping[str, Any]) -> str:
    """The label's argmax key — ties go to the smallest key."""
    return sorted(probabilities, key=lambda k: (-float(probabilities[k]), str(k)))[0]


def is_action_question(record: Mapping[str, Any], name: str) -> bool:
    """A choice question whose label kind is ``action`` (the engine's and the MuJoCo authors' action questions)."""
    question = record["request"]["questions"][name]
    label = (record.get("labels") or {}).get(name) or {}
    return question.get("type") == "choice" and label.get("kind") == "action"


# -- the text rule -----------------------------------------------------------------------------------------------------------------------

#: The plan or the usual way, said outright → continue ("brake as planned", "hold it as planned", "set off normally", "take the ramp as usual").
_EN_PLAN = re.compile(
    r"\b(as (planned|written|usual|is|it stands|you are|it is)|the plan as it stands|sticking to the plan|(carry|carrying) on (exactly )?as planned|"
    r"at the planned (spot|position)|where you planned|without changing how you drive|(?<!half )your (current|usual|normal|standard) (speed|acceleration)|"
    r"ramp as you normally|(start|starting|set off|move off|pull away|get under way|accelerate)[^.;]*\b(normally|usual|normal|standard)\b|"
    r"do not brake for the corner|proceed as planned)\b")
_KO_PLAN = re.compile(r"(계획대로|계획한 (자리|위치|곳|지점)|예정한 (자리|위치|곳|지점)|평소(대로|처럼| 하던 대로)? ?(출발|가속|속도로 (램프|경사)|램프)|평소 가속|"
                      r"평소와 같은 (속도|가속)|램프를 평소|평소처럼 경사로|평소 속도와 제동|보통 때와 같은 (속도|가속)|늘 하던 방식대로|(보통|표준) 가속도)")
#: Proceed now → continue: go now, decide now without another check.
_EN_PROCEED = re.compile(
    r"\b((cross|go) now|(begin|start) crossing|move off now|cross without (delay|waiting)|go ahead (now|and cross)|accelerate across|cross immediately|"
    r"without (checking again|another check|a further check|re-?checking|measuring again|looking again)|skip (any|the) further check|"
    r"decide (now|right away|at once|immediately)|make (the|your) (call|decision) (now|right away)|commit to a decision now|"
    r"act on (the current|what you) (information|know)|(choose|decide|act) now|using only the information you have|"
    r"on the information already in hand|with what you already know|based on what is already known)\b")
_KO_PROCEED = re.compile(r"(지금 (바로 )?건너|(지금|곧바로|바로)[^.]{0,12}(건너|건넙|출발해서)|기다리지 않고|(다시 확인하지|추가 확인|더 살펴보지|다시 재지|다시 측정하지)[^.]*(않고|없이)|"
                         r"지금 가진 정보|이미 아는 정보|지금까지 알게 된 것|곧바로 (행동|결정|정)|바로 결정|지금 (결정|판단|정합))")
#: Speed and path unchanged, said outright → continue.
_EN_UNCHANGED = re.compile(
    r"\b(without slowing|do not slow|don't slow|not slowing|leave your speed unchanged|do not change anything|change nothing|"
    r"keep travelling|continue the motion unchanged|keep (it|the tool) (going|moving) at|stay at your current speed|leave your speed at|"
    r"hold [\d.]+ ?(m/s|km/h|mm/s) all the way|all the way through the corner|for the whole corner)\b")
_KO_UNCHANGED = re.compile(r"(속도를 줄이지 않고|감속하지 않|감속 없이|줄이지 않고|늦추지 않고|속도를 유지|속도\(?[^)]*\)?를 유지|속도를 지킵|"
                           r"지금 하던 대로|아무 조정 없이|현재 속도 그대로|손을 더 대지 않고|더 손대지 않고|추가 조치 없이|그대로 (두고|받아들|인정)|"
                           r"지금 상태를 그대로)")
#: "without stopping" says the robot keeps moving — removed before the stop words are read.
_NO_STOP = re.compile(r"\b(without stopping|never stopping|not stopping|no stop)\b|멈추지 않고|서지 않고")
#: A stop that only leads into another action → move (retract the tool, secure the load first).
_EN_MOVE_FIRST = re.compile(r"\b((pull|draw|move|back) (the tool|it) (back|off)|back the tool off|retract\w*|secure (the load|it)|strap|"
                            r"securing)\b")
_KO_MOVE_FIRST = re.compile(r"(공구를 (뒤로|되돌|물러)|뒤로 (뺍|빼|물러)|후퇴|짐을 고정|고정한 뒤)")
#: Stop / wait / hold → hold.
_EN_HOLD = re.compile(
    r"\b(stop|stops|stopping|halt|standstill|come to (a )?(stop|rest|standstill)|to a (full )?stop|emergency stop|protective stop|pause|"
    r"wait|stay (put|where you are)|keep still|hold (where you are|your position|position|still)|do not move|let them (go|pass)|give way|"
    r"measure|measurement|re-?check|check (it|them)? ?again|look again|another look|second look|fresh look|re-?measur\w*|"
    r"(check|feel|see) (how|where) (the |its )?weight|check the weight|weight balance|how the weight sits|hold off|"
    r"(call|ask|get|have) (someone|a person|a human|somebody|staff)|leave the check to a human|hand (it )?over)\b")
_KO_HOLD = re.compile(r"(멈춥|멈춰|멈추|멈출|정지|(?<!다가)(?<!비켜)섭니다|서도록|선 채|기다|그 자리에|움직이지 않|먼저 보냅|양보|"
                      r"위치를 지킵|미룹|미루|다시 (잰|잽|재|측정|확인|살펴)|한 번 더 (잰|잽|재|측정|확인)|새로 (재|측정|살펴)|측정(을|한|합)|"
                      r"확인(하고|한 (뒤|다음|후)|하기에|하기부터|해 보고)|무게가 어떻게|무게 ?(배분|균형|가|를)|사람(을|이|에게)|맡깁|불러)")
#: A change of speed or path → move.
_EN_SLOWER = re.compile(r"\b(slow(ing|er|ly|s)?|half (your )?(current )?speed|halve|speed down|reduce|reducing|cut (your|the) (speed|tool)|"
                        r"ease off|easing off|veer|swerve|sidestep|move (left|right|across)|pull over|keep to the|creep|inch(ing)?|"
                        r"edge (slowly|toward)|crawl|reduced speed|hug|shift over)\b")
_KO_SLOWER = re.compile(r"(줄이|줄여|줄인|늦추|낮추|낮춘|떨어뜨리|감속|천천히|느린|비켜|비킵|방향을 틀|붙어|붙은|옮겨)")
#: Keep going → continue.
_EN_CONTINUE = re.compile(
    r"\b(keep (going|moving)|carry on|continue|go on|stay at|leave (it|the tool's motion) (as it is|in place)|accept (it|how it is)|"
    r"do not touch it again|make no further changes|take it as it is|move on to the next step|do nothing more|"
    r"keep (it|the tool) (going|moving))\b")
_KO_CONTINUE = re.compile(r"(계속|이어 갑|이어 가|넘어갑|따라갑|진행합)")
_ANONYMOUS_KEY = re.compile(r"[a-z]\d+|[a-z]{1,2}")


def text_action_kind(key: str, description: str) -> str:
    """The kind of an action option from its text alone (key and description, English or Korean) — the fallback of :func:`action_kind`
    when a record carries no structure, and its check on real texts. In order: the plan or the usual way → ``continue``; proceed now
    (decide or go without another check) → ``continue``; speed and path unchanged, said outright → ``continue``; then, with "without
    stopping" removed: a stop that leads into another action (retract, secure first) → ``move``; a stop, wait, hold, a re-check or a
    hand-over → ``hold``; a change of speed or path → ``move``; keep going → ``continue``; anything else → ``move``. Anonymised keys
    (``a1``, ``v2``, ``o5``) say nothing."""
    words = str(key).replace("_", " ").strip().lower()
    words = "" if _ANONYMOUS_KEY.fullmatch(words) else words
    text = " ".join(f"{words} . {description}".lower().split())
    if _EN_PLAN.search(text) or _KO_PLAN.search(text) or _EN_PROCEED.search(text) or _KO_PROCEED.search(text):
        return "continue"
    if _EN_UNCHANGED.search(text) or _KO_UNCHANGED.search(text):
        return "continue"
    text = _NO_STOP.sub(" ", text)
    if _EN_MOVE_FIRST.search(text) or _KO_MOVE_FIRST.search(text):
        return "move"
    if _EN_HOLD.search(text) or _KO_HOLD.search(text):
        return "hold"
    if _EN_SLOWER.search(text) or _KO_SLOWER.search(text):
        return "move"
    if _EN_CONTINUE.search(text) or _KO_CONTINUE.search(text):
        return "continue"
    return "move"


# -- the structured rule --------------------------------------------------------------------------------------------------------------


def action_kind(record: Mapping[str, Any], name: str, key: str) -> str:
    """The kind of option `key` of action question `name` (module docstring): the engine's canonical name, else the MuJoCo plan option, else
    the text rule."""
    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    canonical = ((meta.get("options") or {}).get(name) or {}).get(key) if isinstance(meta.get("options"), Mapping) else None
    if canonical is not None:
        if canonical not in CANONICAL_KINDS:
            raise ValueError(f"{record.get('id')}/{name}: unknown canonical action {canonical!r} — add it to CANONICAL_KINDS")
        return CANONICAL_KINDS[canonical]
    description = str(((record["request"]["questions"][name].get("criteria") or {}).get(key)) or "")
    plan = meta.get("plan") if isinstance(meta.get("plan"), Mapping) else {}
    options = plan.get("options") if isinstance(plan.get("options"), Mapping) else {}
    if key in options:  # a MuJoCo author's plan option carries on unchanged; an alternative is a move unless its text is a stop or a wait
        if options[key] == "planned":
            return "continue"
        return "hold" if text_action_kind(str(key), description) == "hold" else "move"
    return text_action_kind(str(key), description)


def label_class(record: Mapping[str, Any], name: str) -> str:
    """The question's label class (module docstring)."""
    question = record["request"]["questions"][name]
    label = (record.get("labels") or {}).get(name) or {}
    kind = question.get("type")
    if kind == "noul":
        return _band(label["p_yes"])
    if kind == "score":
        return _band(label["value"])
    if kind != "choice":
        raise ValueError(f"{record.get('id')}/{name}: unknown question type {kind!r}")
    best = best_key(label["probabilities"])
    if base_name(name).startswith("s1_band"):
        return f"level:{best}"
    if is_action_question(record, name):
        return f"action:{action_kind(record, name, best)}"
    meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    keys = (((meta.get("format") or {}).get("expanded") or {}).get(name) or {}).get("keys") if isinstance(meta.get("format"), Mapping) else None
    if isinstance(keys, Mapping):  # a re-keyed item (Task v0.8a): its answer key is the key it had before
        original = {str(shown): str(old) for old, shown in keys.items()}
        return f"key:{original.get(str(best), best)}"
    return f"key:{best}"


# -- the weights -----------------------------------------------------------------------------------------------------------------------


def class_weights(counts: Mapping[str, int], cap: float = DEFAULT_CAP) -> dict[str, float]:
    """Per-class weights of one family (module docstring): w_c = min(cap, λ / n_c), mean weight 1 over the family's examples."""
    cap = float(cap)
    if not cap >= 1.0:
        raise ValueError(f"the cap must be at least 1 (got {cap})")
    present = {c: int(n) for c, n in counts.items() if int(n) > 0}
    if not present:
        return {}
    total = sum(present.values())
    if len(present) == 1:
        return {c: 1.0 for c in present}
    capped: set[str] = set()
    while True:
        free = [c for c in present if c not in capped]
        lam = (total - cap * sum(present[c] for c in capped)) / len(free)
        newly = {c for c in free if lam / present[c] > cap}
        if not newly:
            break
        capped |= newly
    return {c: (cap if c in capped else lam / present[c]) for c in present}


def plan_weights(records: Sequence[Mapping[str, Any]], plan: Iterable[tuple[int, str, Any]], *, cap: float = DEFAULT_CAP
                 ) -> tuple[list[float], dict[str, Any]]:
    """The weight of every plan entry (record index, question name, …) — classes counted over the plan's own examples — and the table
    (:func:`weight_table`)."""
    entries = list(plan)
    keys: list[tuple[str, str]] = []
    cache: dict[tuple[int, str], tuple[str, str]] = {}
    for index, name, *_ in entries:
        found = cache.get((index, name))
        if found is None:
            record = records[index]
            found = cache[(index, name)] = (family_of(record, name), label_class(record, name))
        keys.append(found)
    counts: dict[str, Counter] = defaultdict(Counter)
    for family, cls in keys:
        counts[family][cls] += 1
    weights = {family: class_weights(c, cap) for family, c in counts.items()}
    return [weights[f][c] for f, c in keys], weight_table(counts, weights, cap=cap)


def weight_table(counts: Mapping[str, Mapping[str, int]], weights: Mapping[str, Mapping[str, float]], *, cap: float) -> dict[str, Any]:
    """Per family: classes with counts, shares and weights; min, max, the effective sample size (Σw)² / Σw² and its share of n; and
    totals (families, examples, capped classes)."""
    families = {}
    capped = 0
    for family in sorted(counts):
        c, w = counts[family], weights[family]
        n = sum(c.values())
        s1 = sum(c[k] * w[k] for k in c)
        s2 = sum(c[k] * w[k] ** 2 for k in c)
        ess = s1 * s1 / s2 if s2 > 0 else 0.0
        capped += sum(1 for k in c if math.isclose(w[k], cap) and len(c) > 1)
        families[family] = {"n": n, "classes": {k: {"n": c[k], "share": c[k] / n, "weight": w[k]} for k in sorted(c)},
                            "min": min(w.values()), "max": max(w.values()), "mean": s1 / n, "ess": ess, "ess_share": ess / n}
    return {"version": PRIOR_WEIGHTS_VERSION, "cap": float(cap), "families": families, "n_families": len(families),
            "examples": sum(f["n"] for f in families.values()), "capped_classes": capped}
