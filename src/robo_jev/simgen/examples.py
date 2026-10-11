"""사용자 검수용 예시 30개 (Task v0b 지시서 §7; v0a 지시서 §6의 모양): 새 현상 P3·P4·P6과 새 조합 위주, 분야 10개 × 3, 한국어 4개 이상,
현상마다 4개 이상, '다시 볼까'(정보의 가치) 질문 3개 이상, 사실 하나 바꾼 쌍 하나.

예시는 정해 둔 칸(:data:`PLAN`)마다 장면·몸·문체(·언어·움직임 갈래)를 강제하고, seed를 차례로 시도해 **정답이 한쪽으로 다 쏠리지 않은**(기본
행동 질문의 최고 확률 ≤ 0.97, 핵심 noul ∈ [0.03, 0.97], 기대 비용이 모두 같지 않음) 첫 레코드를 고른다 — 칸의 요구(틀린 권고·힌트·기준·상태·
규칙·다시 볼까 질문·보이는 뚜껑)도 그 조건에 더한다. **고르는 판단은 K = 2048로 다시 계산한 정답으로 한다**(리뷰 1 m8: K = 256 정답의
몬테카를로 잡음으로 고르면 보여 주는 정답이 그 잡음 쪽으로 치우친다). 고른 seed는 레코드의 `generator`에 남으므로 그대로 다시 만들 수 있다.
쌍은 numeric 변형이 기본 행동의 argmax를 바꾸는 seed를 고른다.

마크다운은 예시마다: 틀(분야·몸·장면·문체·표시), 모델이 받는 글(상태 + 질문) 그대로, 정답(확률·기대 비용), 참 세계(정답의 사후에서 뽑은
세계)와 실제로 일어난 것, 계산 과정 한 문단.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from robo_jev.simgen.catalog import SimgenConfig, default_config
from robo_jev.simgen.generate import assemble, build_moment
from robo_jev.simgen.record import dumps

__all__ = ["PLAN", "Slot", "build_examples", "dumps_jsonl", "render_markdown"]

_SCENE_NAMES = {
    "p1_step_in": "someone steps into the path (P1)",
    "p1_crossing": "crossing paths (P1)",
    "p1_head_on": "meeting head-on in an aisle (P1)",
    "p5_braking": "braking while carrying liquid (P5)",
    "p5_turning": "turning while carrying liquid (P5)",
    "p1p5_step_in_carrying": "someone steps in while you carry liquid (P1+P5)",
    "p3_carrying_tall": "carrying a tall load (P3)",
    "p3_ramp": "a ramp with a tall load (P3)",
    "p3_place_stack": "setting an item down on a stack (P3)",
    "p3p5_carrying_mixed": "a tall load and an open container together (P3+P5)",
    "p1p3_step_in_carrying_tall": "someone steps in while you carry a tall load (P1+P3, test-only)",
    "p4_place_check": "did the placed item seat? (P4)",
    "p4_insert": "did the insertion go home? (P4)",
    "p4p6_stale_check": "seating check with a stale camera (P4+P6)",
    "p6_stale_track": "is the person you lost sight of still there? (P6)",
    "p6_blind_corner": "a blind corner (P6)",
    "p6_sensor_conflict": "lidar and camera disagree (P6)",
    "p1p6_emerging_step_in": "someone comes out from behind cover into your path (P1+P6)",
}
SELECTION_K = 2048


@dataclass(frozen=True)
class Slot:
    domain: str
    scene: str
    body: str
    style: str
    variant: str | None = None
    agent_kind: str | None = None
    needs: tuple[str, ...] = ()      # wrong_rec / hint / criterion / state / rule / two_fix / lid_visible / voi
    pair: bool = False                # 이 칸과 다음 칸이 쌍(다음 칸은 numeric 변형)
    seed_base: int = 0
    lang: str = "en"
    maneuver: str | None = None
    notes: dict = field(default_factory=dict)


#: 30칸(29칸 + 쌍의 변형): 분야마다 셋; P3 10 · P4 6 · P5 5 · P6 14 · P1 5(조합은 두 현상에 센다); 한국어 6; '다시 볼까' 요구 7.
PLAN: tuple[Slot, ...] = (
    Slot("cafe", "p3p5_carrying_mixed", "mobile_base", "prose", maneuver="brake", needs=("criterion",), seed_base=1000),
    Slot("cafe", "p4_place_check", "arm", "facts", needs=("voi",), lang="ko", seed_base=2000),
    Slot("cafe", "p1p6_emerging_step_in", "humanoid", "planner", agent_kind="adult", seed_base=3000),
    Slot("warehouse", "p3_carrying_tall", "forklift_amr", "planner", maneuver="turn", seed_base=4000),
    Slot("warehouse", "p6_blind_corner", "mobile_base", "facts", needs=("voi",), seed_base=5000),
    Slot("warehouse", "p3_place_stack", "forklift_amr", "prose", needs=("state",), seed_base=6000),
    Slot("factory", "p4_insert", "arm", "planner", needs=("voi",), seed_base=7000),
    Slot("factory", "p4p6_stale_check", "humanoid", "prose", lang="ko", seed_base=8000),
    Slot("factory", "p3_ramp", "mobile_base", "facts", maneuver="down", seed_base=9000),
    Slot("hospital", "p6_stale_track", "mobile_base", "prose", needs=("voi",), seed_base=10000),
    Slot("hospital", "p3p5_carrying_mixed", "humanoid", "facts", maneuver="turn", lang="ko", seed_base=11000),
    Slot("hospital", "p1p6_emerging_step_in", "mobile_base", "facts", needs=("state",), seed_base=12000),
    Slot("lab", "p4p6_stale_check", "arm", "facts", needs=("voi",), seed_base=13000),
    Slot("lab", "p3_place_stack", "arm", "planner", needs=("voi",), seed_base=14000),
    Slot("lab", "p6_sensor_conflict", "mobile_base", "prose", needs=("state",), seed_base=15000),
    Slot("sidewalk", "p6_blind_corner", "humanoid", "planner", lang="ko", seed_base=16000),
    Slot("sidewalk", "p3_carrying_tall", "mobile_base", "prose", maneuver="brake", needs=("hint",), seed_base=17000),
    Slot("sidewalk", "p5_braking", "mobile_base", "facts", seed_base=18000),
    Slot("home", "p3_carrying_tall", "humanoid", "facts", maneuver="start", seed_base=19000),
    Slot("home", "p4_place_check", "humanoid", "prose", needs=("rule",), seed_base=20000),
    Slot("home", "p6_stale_track", "humanoid", "facts", lang="ko", seed_base=21000),
    Slot("retail", "p3_ramp", "forklift_amr", "planner", maneuver="up", seed_base=22000),
    Slot("retail", "p6_sensor_conflict", "humanoid", "facts", needs=("voi",), seed_base=23000),
    Slot("retail", "p1p5_step_in_carrying", "humanoid", "prose", agent_kind="cart", seed_base=24000),
    Slot("greenhouse", "p4_insert", "humanoid", "facts", seed_base=25000),
    Slot("greenhouse", "p3p5_carrying_mixed", "mobile_base", "planner", maneuver="brake", needs=("wrong_rec",), seed_base=26000),
    Slot("greenhouse", "p6_blind_corner", "mobile_base", "prose", seed_base=27000),
    Slot("transit", "p6_sensor_conflict", "mobile_base", "planner", lang="ko", seed_base=28000),
    Slot("transit", "p1p6_emerging_step_in", "mobile_base", "facts", agent_kind="adult", pair=True, seed_base=29000),
)


def _forced(slot: Slot) -> dict:
    forced = {"scene": slot.scene, "body": slot.body, "style": slot.style, "lang": slot.lang}
    if slot.variant:
        forced["variant"] = slot.variant
    if slot.agent_kind:
        forced["agent_kind"] = slot.agent_kind
    if slot.maneuver:
        forced["maneuver"] = slot.maneuver
    return forced


def _primary(record: dict) -> dict:
    return record["labels"][record["meta"]["recommendation"]["question"]]


def _informative(record: dict, needs: tuple[str, ...]) -> bool:
    """칸의 조건 — ``record``는 고르기용(K = SELECTION_K)으로 계산한 레코드다."""
    label = _primary(record)
    key = record["labels"][record["meta"]["key_question"]]["p_yes"]
    if max(label["probabilities"].values()) > 0.97 or not 0.03 <= key <= 0.97:
        return False
    costs = label["expected_cost"].values()
    if max(costs) - min(costs) < 1.0:  # 모든 선택지가 같은 비용(예: 어느 감속으로도 넘어섬)인 평평한 정답은 보여 줄 거리가 없다
        return False
    meta, kinds = record["meta"], {lab.get("kind") for lab in record["labels"].values()}
    checks = {
        "wrong_rec": meta["recommendation"]["wrong"],
        "hint": meta["hint"],
        "criterion": any(c is not None for c in meta["criteria"].values()),
        "state": "state" in kinds,
        "rule": "rule" in kinds,
        "two_fix": bool(meta["agent"]) and meta["agent"]["velocity_mode"] == "two_fix",
        "lid_visible": bool(meta["container"]) and meta["container"]["lid_view"] == "lid",
        "voi": any(b.get("kind") == "voi" for b in meta["basis"].values()),
    }
    return all(checks[n] for n in needs)


def build_examples(cfg: SimgenConfig | None = None, git: str | None = None, id_prefix: str = "v0b-ex", max_tries: int = 600) -> list[dict]:
    """PLAN의 30개(쌍은 두 레코드)를 만든다."""
    cfg = cfg or default_config()
    records: list[dict] = []
    for slot in PLAN:
        forced = _forced(slot)
        chosen = None
        for t in range(max_tries):
            seed = slot.seed_base + t
            moment = build_moment(cfg, seed, slot.domain, forced)
            if not _informative(assemble(cfg, moment, len(records), git=git, id_prefix=id_prefix, forced=forced, k=SELECTION_K), slot.needs):
                continue
            record = assemble(cfg, moment, len(records), git=git, id_prefix=id_prefix, forced=forced)
            if slot.pair:
                variant = build_moment(cfg, seed, slot.domain, forced, "numeric")
                if variant.variant != "numeric" or variant.flipped not in ("step_in", "late", "corner", "sight"):
                    continue
                selection = assemble(cfg, variant, len(records) + 1, git=git, id_prefix=id_prefix, forced=forced, k=SELECTION_K)
                a = _primary(assemble(cfg, moment, len(records), git=git, id_prefix=id_prefix, forced=forced, k=SELECTION_K))
                b = _primary(selection)
                if max(a["probabilities"], key=a["probabilities"].get) == max(b["probabilities"], key=b["probabilities"].get):
                    continue
                ids = (f"{id_prefix}-{len(records):06d}", f"{id_prefix}-{len(records) + 1:06d}")
                record = assemble(cfg, moment, len(records), git=git, id_prefix=id_prefix, forced=forced, pair={"of": ids[1], "kind": "numeric", "role": "base"})
                twin = assemble(cfg, variant, len(records) + 1, git=git, id_prefix=id_prefix, forced=forced,
                                pair={"of": ids[0], "kind": "numeric", "role": "variant", "flipped": variant.flipped})  # fmt: skip
                a, b = _primary(record), _primary(twin)
                if max(a["probabilities"], key=a["probabilities"].get) == max(b["probabilities"], key=b["probabilities"].get):
                    continue  # 보여 주는 K에서도 기본 행동이 바뀌어야 한다
                chosen = [record, twin]
            else:
                chosen = [record]
            break
        if chosen is None:
            raise RuntimeError(f"no informative example for {slot}")
        records.extend(chosen)
    return records


# ----------------------------------------------------------------------------------------------------------------
# 마크다운
# ----------------------------------------------------------------------------------------------------------------


def _fmt_p(x: float) -> str:
    return f"{x:.3f}"


def _question_block(record: dict) -> list[str]:
    lines = []
    for i, (name, q) in enumerate(record["request"]["questions"].items(), start=1):
        lines.append(f"{i}. **`{name}`** ({q['type']}) — {q['instructions']}")
        if q["type"] == "choice":
            for key, text in q["criteria"].items():
                lines.append(f"    - `{key}`: {text}")
        elif q["type"] == "score":
            lines.append(f"    - scale: {' / '.join(q['criteria'])}")
    return lines


def _label_block(record: dict) -> list[str]:
    out = []
    for name, label in record["labels"].items():
        basis = record["meta"]["basis"][name]
        if label["type"] == "choice":
            kind = label["kind"]
            extra = f", rule '{basis['criterion']}'" if kind == "action" and basis.get("criterion") else ""
            out.append(f"**`{name}`** — {kind} choice{extra}")
            out.append("")
            if kind == "action":
                canon = record["meta"]["options"][name]
                show = any(key != value for key, value in canon.items())
                out.append("| option |" + (" meaning |" if show else "") + " probability | expected cost |")
                out.append("|---|" + ("---|" if show else "") + "---:|---:|")
                for key in sorted(label["probabilities"], key=lambda k: (-label["probabilities"][k], label["expected_cost"][k])):
                    meaning = f" {canon[key]} |" if show else ""
                    out.append(f"| `{key}` |{meaning} {_fmt_p(label['probabilities'][key])} | {label['expected_cost'][key]:.2f} |")
            else:
                out.append("| option | probability |")
                out.append("|---|---:|")
                for key in sorted(label["probabilities"], key=lambda k: -label["probabilities"][k]):
                    out.append(f"| `{key}` | {_fmt_p(label['probabilities'][key])} |")
            out.append("")
        elif label["type"] == "noul":
            what = f"{basis.get('event')}" + (f" if {basis['option']}" if basis.get("option") else "")
            how = "exact posterior probability of" if name in record["meta"].get("exact_labels", []) else "frequency over the posterior worlds of"
            out.append(f"**`{name}`** — noul (P(yes) = {how} `{what}`): **{_fmt_p(label['p_yes'])}**")
            out.append("")
        else:
            what = f"{basis.get('name')}" + (f" if {basis['option']}" if basis.get("option") else "")
            out.append(f"**`{name}`** — score (mean of `{what}` over the posterior worlds): **{_fmt_p(label['value'])}**")
            out.append("")
    return out


_HIDDEN_KEYS = {
    "gap": "centre-to-centre distance along your path (m)", "lat0": "lateral position now, + = your left (m)", "u0": "lateral speed now (m/s)",
    "ra": "agent radius (m)", "stay_s": "how long it would stay (s)", "cross_speed": "speed it would cross at (m/s)",
    "back_speed": "speed it would back off at (m/s)", "roll_decel": "ball rolling deceleration (m/s²)", "dist": "distance, agent centre to your path centreline (m)",
    "ahead": "crossing line, robot centre to agent line (m)", "va": "agent speed (m/s)", "yield_standoff": "where it would stop short of your path (m)",
    "hurry_factor": "speed-up factor if hurrying", "width": "true aisle width (m)", "lat_a": "agent lateral position (m)",
    "aside_speed": "sideways speed if moving aside (m/s)", "keep_standoff": "standoff it keeps if it stops for you (m)",
    "obstacle": "obstacle distance from your centre (m)", "fill": "true fill fraction", "k": "slosh factor k (1.0–1.6)",
    "tilt_deg": "floor tilt (deg)", "lid": "lid present (1 = yes)", "intent": "intent now",
    "reacts": "would brake late for you if you block its line (1 = yes)", "react_standoff": "where it would stop if it reacts (m)",
    "height": "true load height (m)", "base": "true load base width (m)", "com": "centre-of-mass height / load height", "off": "fore-aft offset of the centre of mass (fraction of half-base, + = toward the front)",
    "mu": "friction coefficient load–deck", "c": "weight offset of the item (m, + = your left)", "eps": "placement error (m)",
    "state": "true state now", "state_then": "true state when the camera last saw it", "changed": "state changed since the camera image (1 = yes)",
    "present": "really there / still there (1 = yes)", "lam": "hazard rate (1/s)", "t_leave": "leaves after (s)", "spot": "true distance to the spot (m)",
    "t_e": "someone emerges after (s)", "emerges": "someone emerges within the horizon (1 = yes)", "u": "emerging person's speed (m/s)",
    "lateral": "corner edge to your path (m)", "late": "distance at which it would be detected for sure (m)",
    "follow": "a second person follows (1 = yes)", "t2": "the follower appears after (s)", "u2": "follower speed (m/s)",
}  # fmt: skip
_TRUTH_ORDER = (
    "gap", "dist", "ahead", "lat0", "lat_a", "u0", "va", "ra", "width", "stay_s", "cross_speed", "back_speed", "yield_standoff", "hurry_factor",
    "aside_speed", "keep_standoff", "reacts", "react_standoff", "roll_decel", "obstacle", "fill", "k", "tilt_deg", "lid",
    "height", "base", "com", "off", "mu", "c", "eps", "state", "state_then", "changed", "present", "lam", "t_leave", "spot", "t_e", "emerges", "u",
    "lateral", "late", "follow", "t2", "u2",
)  # fmt: skip


def _truth_block(record: dict) -> list[str]:
    world = record["truth"]["world"]
    v = world["values"]
    lines = ["The true world below is **one more draw from the same posterior** the labels use (not the world the observation was generated from),",
             "so over many records the realized outcomes are calibrated against the labels.", ""]  # fmt: skip
    if "intent" in v and v["intent"] is not None:
        lines.append(f"- **intent: `{v['intent']}`**")
    is_ball = world.get("agent_kind") == "ball"
    for key in _TRUTH_ORDER:
        if key not in v or (key == "roll_decel" and not is_ball):
            continue
        if is_ball and key in ("stay_s", "cross_speed", "back_speed", "yield_standoff", "hurry_factor", "aside_speed", "keep_standoff", "reacts", "react_standoff"):
            continue
        label = "centre-to-centre distance along the aisle now (m)" if key == "dist" and record["meta"]["scene"] == "p1_head_on" else _HIDDEN_KEYS[key]
        value = round(v[key], 3)
        if key in ("state", "state_then") and record["meta"].get("place"):
            from robo_jev.simgen.p4 import STATES

            sub = int(v[key])
            value = f"{STATES[record['meta']['place']['family']][(0, 1, 2, 2)[sub]]}" + ("" if sub < 2 else (" (+)" if sub == 2 else " (−)"))
        lines.append(f"- {label}: {value}")
    views = record["meta"].get("sides")
    for i, side in enumerate(("left", "right")):
        if views and views[i] in ("occluded", "blocked"):
            present = v.get(f"side_{side}", 0.0) > 0.5
            text = "visibly blocked" if views[i] == "blocked" else ("hidden from view — something IS there" if present else "hidden from view — it is actually free")
            lines.append(f"- {side} side: {text}")
    anchor = record["meta"]["margins"]
    lines.append(f"- decision margins: observation generated at {', '.join(f'{k} = {m:+.3f}' for k, m in anchor.items())} (design target); "
                 f"this true world {', '.join(f'{k} = {m:+.3f}' for k, m in world['margins'].items())}")  # fmt: skip
    lines.append("- what actually happened in the true world:")
    for name, outcome in record["truth"]["realized"].items():
        if isinstance(outcome, dict):
            costs = ", ".join(f"`{k}` {c:.1f}" for k, c in sorted(outcome["costs"].items(), key=lambda kv: kv[1]))
            lines.append(f"    - `{name}`: cheapest in hindsight `{outcome['best']}` (realized costs: {costs})")
        else:
            lines.append(f"    - `{name}`: {outcome}")
    return lines


def _flags(record: dict) -> str:
    meta = record["meta"]
    rec = meta["recommendation"]
    bits = []
    if rec["present"]:
        shown = f"`{rec['option']}`" if rec["option"] else f"`{rec['canonical']}` (not among the options of `{rec['question']}`)"
        verdict = f"WRONG — {rec['gap']:.1f} worse than the best available action" if rec["wrong"] else f"correct — within τ of the best ({rec['gap']:.2f})"
        bits.append(f"planner recommendation {shown} ({verdict})")
    if meta["hint"]:
        bits.append("computed hint in the text")
    crit = [c for c in meta["criteria"].values() if c]
    if crit:
        bits.append(f"criterion in an action question: {', '.join(crit)}")
    if meta["agent"]:
        cue = f"cue `{meta['agent']['cue']}`" if meta["agent"]["cue"] else "no cue"
        bits.append(f"agent `{meta['agent']['kind']}` ({cue}, {meta['agent']['velocity_mode']}, track age {meta['agent']['age_s']:.2f} s)")
    if meta["container"]:
        c = meta["container"]
        bits.append(f"container `{c['kind']}` ({c['content']}; fill as {c['fill_mode']}; lid view {c['lid_view']})")
    if meta["pair"]:
        p = meta["pair"]
        bits.append(f"**minimal pair ({p['kind']}, {p['role']}) with `{p['of']}`**" + (f" — moved margin `{p.get('flipped')}`" if p.get("flipped") else ""))
    bits.append(f"key style `{meta['key_style']}`, units {meta['units']['distance']} / {meta['units']['speed']} / {meta['units']['time']}")
    return "; ".join(bits)


def render_markdown(records: list[dict], cfg: SimgenConfig | None = None) -> str:
    cfg = cfg or default_config()
    costs = cfg.values["costs"]
    speed = cfg.values.get("contact_speed", {})
    mechanisms = (
        "contact costs scaled by impact speed (floor {:g}), hard-braking cost {}".format(speed.get("floor", 1.0), cfg.values.get("hard_brake_cost"))
        if speed.get("enabled") else "contact cost independent of impact speed, no cost for hard braking itself"
    )
    out = [
        "# simgen v0b — 30 examples for review (new phenomena P3 · P4 · P6, new combinations, Korean, 'look again')",
        "",
        "Each example is one judgment moment: what the model receives (**state + questions**, verbatim), the **labels** (posterior",
        "probabilities, and expected costs for action choices), a **true world** with what actually happened in it, and the computation in one",
        "paragraph (English; for Korean records the names inside it are the Korean ones). The model sees only the state and the questions.",
        "",
        "New in v0b: tall loads that can tip or slide (P3: braking, turning, pulling away, ramps, stacking), seating / insertion checks from",
        "noisy measurements (P4, exact Bayes over the states; a stale camera in P4+P6), stale tracks, blind corners and sensor conflicts (P6,",
        "exact weights for present / emerging / following), and the **'look again' question** — its expected cost is the time to look plus the",
        "expected cost of the best action after seeing the new signal (closed form over discrete signals; quasi-random integration for a",
        "re-measurement).",
        "",
        "How the labels are defined (docs/12 §5, corrected):",
        "",
        "- **noul** = probability of the event over the posterior given what the robot can see: the share of K = 256 posterior worlds, or the",
        "  exact value where it has a closed form (stop-short, room to pass, overrunning a stop target).",
        "- **score** = mean of a [0, 1] quantity over the same worlds (a probability, closeness, share of the rim headroom used, or delay / 3 s).",
        "- **action choice** = softmax(−(E[cost] − min E[cost]) / τ), τ = {:g}. The highest probability is always the lowest expected cost. Costs"
        " ({}, to be tuned by you): person {} · child/elder {} · vehicle {} · object {} · spill hot {} / chemical {} / other {} · delay {}/s; {}.".format(
            cfg.values["tau"], cfg.versions["values"], costs["contact_person"], costs["contact_vulnerable"], costs["contact_vehicle"],
            costs["contact_object"], costs["spill_hot"], costs["spill_chemical"], costs["spill_other"], costs["delay_per_s"], mechanisms),
        "- **state choice** = posterior probability of each discrete state (intent, fill range, lid — exact). **rule choice** = the stated rule applied to the number as printed.",
        "- **planner recommendation** = chosen over every action available at that moment with the default costs: a correct one is within τ of the",
        "  best, a wrong one at least 2τ worse.",
        "- Distances in the text are surface gaps (your front → their nearest side). In an aisle (head-on) a reported gap is the gap **at the time",
        "  of the report**; both of you have moved since. Hidden values never appear in the model input.",
        "- The **true world** is one more draw from the same posterior the labels use (fix round 1): the world the observation was generated from",
        "  only sets the observation, so realized outcomes are calibrated against the labels.",
        "",
        "| # | id | domain | body | scene | style | notes |",
        "|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for i, r in enumerate(records, start=1):
        m = r["meta"]
        note = []
        if m["pair"]:
            note.append(f"pair {m['pair']['role']}")
        if m["recommendation"]["wrong"]:
            note.append("wrong recommendation")
        if any(m["criteria"].values()):
            note.append("criterion")
        if m["language"] == "ko":
            note.append("Korean")
        if any(b.get("kind") == "voi" for b in m["basis"].values()):
            note.append("'look again' question")
        out.append(f"| {i} | `{r['id']}` | {m['domain']} | {m['body']} | {m['scene']} | {m['style']} | {', '.join(note)} |")
    out.append("")
    for i, r in enumerate(records, start=1):
        m = r["meta"]
        out += [
            "---",
            "",
            f"## {i}. {cfg.domain(m['domain'])['name']} — {m['body']} — {_SCENE_NAMES[m['scene']]} — {m['style']}",
            "",
            f"`{r['id']}` · seed {r['generator']['seed']} · sensor {m['sensor_suite']}" + (f" · variant {m['variant']}" if m["variant"] != "moving" else ""),
            "",
            f"{_flags(r)}",
            "",
            "### Model input — state",
            "",
            "```text",
            r["request"]["state"],
            "```",
            "",
            "### Model input — questions",
            "",
            *_question_block(r),
            "",
            "### Labels",
            "",
            *_label_block(r),
            "### True world (a posterior draw) and what happened",
            "",
            *_truth_block(r),
            "",
            "### How the labels were computed",
            "",
            r["rationale"],
            "",
        ]
    return "\n".join(out)


def dumps_jsonl(records: list[dict]) -> str:
    return "".join(dumps(r) + "\n" for r in records)
