"""사용자 검수용 예시 20개 (Task v0a 지시서 §6): 분야 10개 × 2, P1 단독·P5 단독·조합(6개 이상), 문체 고르게, 사실 하나 바꾼 쌍 하나.

예시는 정해 둔 칸(:data:`PLAN`)마다 장면·몸·문체를 강제하고, seed를 차례로 시도해 **정답이 한쪽으로 다 쏠리지 않은**(기본 행동 질문의 최고
확률 ≤ 0.97, 핵심 noul ∈ [0.03, 0.97]) 첫 레코드를 고른다 — 칸의 요구(틀린 권고·힌트·기준·상태·규칙 질문)도 그 조건에 더한다. 고른 seed는
레코드의 `generator`에 남으므로 그대로 다시 만들 수 있다. 쌍은 numeric 변형이 기본 행동의 argmax를 바꾸는 seed를 고른다.

마크다운은 예시마다: 틀(분야·몸·장면·문체·표시), 모델이 받는 글(상태 + 질문) 그대로, 정답(확률·기대 비용), 숨은 참값과 실제로 일어난 것,
계산 과정 한 문단.
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
    "p5_braking": "braking with an open container (P5)",
    "p5_turning": "turning with an open container (P5)",
    "p1p5_step_in_carrying": "someone steps in while you carry an open container (P1+P5)",
}


@dataclass(frozen=True)
class Slot:
    domain: str
    scene: str
    body: str
    style: str
    variant: str | None = None
    agent_kind: str | None = None
    needs: tuple[str, ...] = ()      # wrong_rec / hint / criterion / state / rule / two_fix / pair_flip
    pair: bool = False                # 이 칸과 다음 칸이 쌍(다음 칸은 numeric 변형)
    seed_base: int = 0
    notes: dict = field(default_factory=dict)


#: 20칸: 분야마다 둘, 조합 8 · P1 단독 7 · P5 단독 5, 문체 facts 6 · prose 7 · planner 7, 쌍 하나(19·20).
PLAN: tuple[Slot, ...] = (
    Slot("cafe", "p1p5_step_in_carrying", "mobile_base", "prose", agent_kind="adult", needs=("state",), seed_base=1000),
    Slot("cafe", "p5_braking", "arm", "planner", needs=("hint",), seed_base=2000),
    Slot("warehouse", "p1_crossing", "forklift_amr", "planner", variant="moving", agent_kind="forklift", needs=("wrong_rec",), seed_base=3000),
    Slot("warehouse", "p1p5_step_in_carrying", "forklift_amr", "facts", agent_kind="adult", needs=("rule",), seed_base=4000),
    Slot("factory", "p1p5_step_in_carrying", "arm", "planner", needs=("state",), seed_base=5000),
    Slot("factory", "p1_head_on", "mobile_base", "prose", agent_kind="cart", needs=("state",), seed_base=6000),
    Slot("hospital", "p1_head_on", "mobile_base", "facts", agent_kind="adult", needs=("hint",), seed_base=7000),
    Slot("hospital", "p1p5_step_in_carrying", "humanoid", "prose", agent_kind="elder_walker", needs=("criterion",), seed_base=8000),
    Slot("lab", "p5_braking", "arm", "planner", needs=("state",), seed_base=9000),
    Slot("lab", "p1_step_in", "arm", "facts", needs=("two_fix",), seed_base=10000),
    Slot("sidewalk", "p1_crossing", "mobile_base", "facts", variant="stopped", agent_kind="cyclist", needs=("hint",), seed_base=11000),
    Slot("sidewalk", "p1p5_step_in_carrying", "mobile_base", "planner", agent_kind="child", needs=("wrong_rec",), seed_base=12000),
    Slot("home", "p5_turning", "mobile_base", "prose", needs=("criterion",), seed_base=13000),
    Slot("home", "p1_step_in", "humanoid", "prose", agent_kind="ball", seed_base=14000),
    Slot("retail", "p1p5_step_in_carrying", "humanoid", "facts", agent_kind="cart", needs=("rule",), seed_base=15000),
    Slot("retail", "p5_braking", "forklift_amr", "prose", seed_base=16000),
    Slot("greenhouse", "p1_crossing", "mobile_base", "prose", variant="moving", agent_kind="other_robot", needs=("state",), seed_base=17000),
    Slot("greenhouse", "p5_turning", "arm", "facts", seed_base=18000),
    Slot("transit", "p1p5_step_in_carrying", "mobile_base", "planner", agent_kind="adult", pair=True, seed_base=19000),
)


def _forced(slot: Slot) -> dict:
    forced = {"scene": slot.scene, "body": slot.body, "style": slot.style}
    if slot.variant:
        forced["variant"] = slot.variant
    if slot.agent_kind:
        forced["agent_kind"] = slot.agent_kind
    return forced


def _primary(record: dict) -> dict:
    return record["labels"][record["meta"]["recommendation"]["question"]]


def _informative(record: dict, needs: tuple[str, ...]) -> bool:
    label = _primary(record)
    key = record["labels"][record["meta"]["key_question"]]["p_yes"]
    if max(label["probabilities"].values()) > 0.97 or not 0.03 <= key <= 0.97:
        return False
    meta, kinds = record["meta"], {lab.get("kind") for lab in record["labels"].values()}
    checks = {
        "wrong_rec": meta["recommendation"]["wrong"],
        "hint": meta["hint"],
        "criterion": any(c is not None for c in meta["criteria"].values()),
        "state": "state" in kinds,
        "rule": "rule" in kinds,
        "two_fix": bool(meta["agent"]) and meta["agent"]["velocity_mode"] == "two_fix",
    }
    return all(checks[n] for n in needs)


def build_examples(cfg: SimgenConfig | None = None, git: str | None = None, id_prefix: str = "v0a-ex", max_tries: int = 400) -> list[dict]:
    """PLAN의 20개(쌍은 두 레코드)를 만든다."""
    cfg = cfg or default_config()
    records: list[dict] = []
    for slot in PLAN:
        forced = _forced(slot)
        chosen = None
        for t in range(max_tries):
            seed = slot.seed_base + t
            moment = build_moment(cfg, seed, slot.domain, forced)
            record = assemble(cfg, moment, len(records), git=git, id_prefix=id_prefix, forced=forced)
            if not _informative(record, slot.needs):
                continue
            if slot.pair:
                variant = build_moment(cfg, seed, slot.domain, forced, "numeric")
                if variant.variant != "numeric" or variant.flipped != "step_in":
                    continue
                twin = assemble(cfg, variant, len(records) + 1, git=git, id_prefix=id_prefix, forced=forced)
                a, b = _primary(record), _primary(twin)
                if max(a["probabilities"], key=a["probabilities"].get) == max(b["probabilities"], key=b["probabilities"].get):
                    continue
                ids = (f"{id_prefix}-{len(records):06d}", f"{id_prefix}-{len(records) + 1:06d}")
                record = assemble(cfg, moment, len(records), git=git, id_prefix=id_prefix, forced=forced, pair={"of": ids[1], "kind": "numeric", "role": "base"})
                twin = assemble(cfg, variant, len(records) + 1, git=git, id_prefix=id_prefix, forced=forced,
                                pair={"of": ids[0], "kind": "numeric", "role": "variant", "flipped": variant.flipped})  # fmt: skip
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
            out.append(f"**`{name}`** — noul (P(yes) = frequency of `{what}` over the posterior worlds): **{_fmt_p(label['p_yes'])}**")
            out.append("")
        else:
            what = f"{basis.get('name')}" + (f" if {basis['option']}" if basis.get("option") else "")
            out.append(f"**`{name}`** — score (mean of `{what}` over the posterior worlds): **{_fmt_p(label['value'])}**")
            out.append("")
    return out


_HIDDEN_KEYS = {
    "gap": "centre-to-centre distance along your path (m)", "lat0": "lateral position now, + = your left (m)", "u0": "lateral speed now (m/s)",
    "ra": "agent radius (m)", "stay_s": "how long it would stay (s)", "cross_speed": "speed it would cross at (m/s)",
    "back_speed": "speed it would back off at (m/s)", "roll_decel": "ball rolling deceleration (m/s²)", "dist": "distance (m)",
    "ahead": "crossing line, robot centre to agent line (m)", "va": "agent speed (m/s)", "yield_standoff": "where it would stop short of your path (m)",
    "hurry_factor": "speed-up factor if hurrying", "width": "true aisle width (m)", "lat_a": "agent lateral position (m)",
    "aside_speed": "sideways speed if moving aside (m/s)", "keep_standoff": "standoff it keeps if it stops for you (m)",
    "obstacle": "obstacle distance from your centre (m)", "fill": "true fill fraction", "k": "slosh factor k (1.0–1.6)",
    "tilt_deg": "floor tilt (deg)", "lid": "lid present (1 = yes)", "intent": "intent now",
}  # fmt: skip


def _truth_block(record: dict) -> list[str]:
    world = record["truth"]["world"]
    v = world["values"]
    lines = []
    at = world["at_observation"]
    if "intent" in v and v["intent"] is not None:
        changed = at.get("intent_obs") not in (None, v["intent"])
        lines.append(f"- **intent now: `{v['intent']}`**" + (f" (it was `{at['intent_obs']}` when observed — it changed since)" if changed else " (same as when observed)"))
    is_ball = (world.get("agent_kind") == "ball")
    for key in ("gap", "dist", "ahead", "lat0", "lat_a", "u0", "va", "ra", "width", "stay_s", "cross_speed", "back_speed", "yield_standoff",
                "hurry_factor", "aside_speed", "roll_decel", "obstacle", "fill", "k", "tilt_deg", "lid"):  # fmt: skip
        if key not in v or (key == "roll_decel" and not is_ball):
            continue
        if is_ball and key in ("stay_s", "cross_speed", "back_speed", "yield_standoff", "hurry_factor", "aside_speed", "keep_standoff"):
            continue
        lines.append(f"- {_HIDDEN_KEYS[key]}: {round(v[key], 3)}")
    for i, side in enumerate(("left", "right")):
        views = at.get("sides")
        if views and views[i] in ("occluded", "blocked"):
            present = v.get(f"side_{side}", 0.0) > 0.5
            text = "visibly blocked" if views[i] == "blocked" else ("hidden from view — something IS there" if present else "hidden from view — it is actually free")
            lines.append(f"- {side} side: {text}")
    lines.append(f"- decision margins (true world): {', '.join(f'{k} = {m:+.3f}' for k, m in world['margins'].items())}")
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
        bits.append(f"planner recommendation `{rec['option']}` ({'WRONG — physically worse than the best' if rec['wrong'] else 'correct'})")
    if meta["hint"]:
        bits.append("computed hint in the text")
    crit = [c for c in meta["criteria"].values() if c]
    if crit:
        bits.append(f"criterion in an action question: {', '.join(crit)}")
    if meta["agent"]:
        bits.append(f"agent `{meta['agent']['kind']}` (cue `{meta['agent']['cue']}`, {meta['agent']['velocity_mode']}, track age {meta['agent']['age_s']:.2f} s)")
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
    out = [
        "# simgen v0a — 20 examples for review",
        "",
        "Each example is one judgment moment: what the model receives (**state + questions**, verbatim), the **labels** (posterior",
        "probabilities, and expected costs for action choices), the **hidden truth** that produced the observation, and the computation in one",
        "paragraph. The model sees only the state and the questions.",
        "",
        "How the labels are defined (docs/12 §5, corrected):",
        "",
        "- **noul** = share of the K = 256 posterior worlds (sampled from what the robot can see) in which the event happens.",
        "- **score** = mean of a [0, 1] quantity over the same worlds (a probability, closeness, share of the rim headroom used, or delay / 3 s).",
        "- **action choice** = softmax(−(E[cost] − min E[cost]) / τ), τ = {:g}. The highest probability is always the lowest expected cost. Costs"
        " (values-v0, to be tuned by you): person {} · child/elder {} · vehicle {} · object {} · spill hot {} / chemical {} / other {} · delay {}/s.".format(
            cfg.values["tau"], costs["contact_person"], costs["contact_vulnerable"], costs["contact_vehicle"], costs["contact_object"],
            costs["spill_hot"], costs["spill_chemical"], costs["spill_other"], costs["delay_per_s"]),
        "- **state choice** = posterior share of each discrete state (intent, fill range, lid). **rule choice** = the stated rule applied to the number as printed.",
        "- Distances in the text are surface gaps (your front → their nearest side). Hidden values never appear in the model input.",
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
            "### Hidden truth",
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
