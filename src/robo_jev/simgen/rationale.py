"""정답의 계산 과정 한 문단 (Task v0a 지시서 §2 `rationale`, docs/12 §11 '계산 과정을 안으로 접기').

사람이 읽는 용도이고, 나중의 '계산을 안으로 접기' 실험 재료다. 모델 입력이 아니다 — 그래서 참 세계(숨은 값)를 마지막 문장에서 말해도 된다.
문단은 (1) 장면의 핵심 계산(보고된 수치 → 여유), (2) 사후 의도·채움, (3) 기본 행동 질문의 기대 비용 분해와 소프트맥스, (4) 참 세계
(정답의 사후에서 뽑은 세계 — 리뷰 1 I4)에서 실제로 일어난 것의 순서다.
"""

from __future__ import annotations

from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import INTENTS, SCENE_FAMILY, SimgenConfig
from robo_jev.simgen.labels import cost_model, event_weights, world_costs
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.posterior import intent_posterior
from robo_jev.simgen.rollout import Outcomes
from robo_jev.simgen.world import World

__all__ = ["head_on_current_gap", "write_rationale"]


_KIND_NAMES = {
    "adult": "an adult", "child": "a child", "elder_walker": "an older person with a walker", "cyclist": "a cyclist",
    "cart": "a person pushing a cart", "forklift": "a driven forklift", "other_robot": "another robot", "ball": "a ball", "hand": "a hand",
}  # fmt: skip


def _m(x: float, arm: bool = False) -> str:
    return f"{x:.3f} m" if arm else f"{x:.2f} m"


def _p(x: float) -> str:
    return f"{x:.2f}"


def head_on_current_gap(obs: Observation) -> tuple[float, float, float]:
    """마주 옴의 글이 뜻하는 지금 간격(명목값): (지금 간격, 다가오는 속도, 상대 속도). 위치 보고는 그 시각의 간격이고(리뷰 1 C1)
    그동안 둘이 함께 다가왔다 — 지금 간격 = 마지막 보고 − (상대 속도 + 로봇 속도)·나이. 위치 두 개면 (d0 − d1)/간격 시간이 다가오는 속도다."""
    a, v0 = obs.agent, obs.robot.speed
    if a.speed is not None:
        agent_speed = a.speed.value
        closing = agent_speed + v0
    else:
        closing = (a.track[0].value - a.track[1].value) / (a.track_ages[0] - a.track_ages[1])
        agent_speed = closing - v0
    return a.track[-1].value - closing * a.age, closing, agent_speed


def _lead(cfg: SimgenConfig, obs: Observation, out: Outcomes, exact: dict) -> list[str]:
    r = obs.robot
    parts: list[str] = []
    stop_m = ph.stopping_distance(r.speed, r.latency, r.decel_moderate)
    arm = obs.body == "arm"
    firm = "the normal pause" if arm else "firm braking"
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        a = obs.agent
        p_short = exact.get("stops_short", float(out.world_events["stops_short"].mean()))
        parts.append(
            f"The reported gap is {_m(a.ahead.value, arm)} (σ_eff {a.ahead.sigma_eff:.3f} m); {firm} from {r.speed:.2f} m/s needs "
            f"{_m(r.speed * r.latency, arm)} of reaction plus {_m(r.speed**2 / (2 * r.decel_moderate), arm)} of braking = {_m(stop_m, arm)}, so "
            f"P(stop short with {firm}) = {_p(p_short)}."
        )
    elif obs.scene == "p1_crossing":
        parts.append(
            f"If {obs.agent.descriptor.ref} keeps the same pace, the share of posterior worlds in which you clear the crossing line at least 0.3 s "
            f"before {obs.agent.descriptor.ref} reaches your path is {_p(out.world_events['clears_first'].mean())}."
        )
    elif obs.scene == "p1_head_on":
        now, closing, agent_speed = head_on_current_gap(obs)
        a = obs.agent
        free = obs.env.corridor_width.value - 2 * r.radius - a.size.value
        p_pass = exact.get("passable", float(out.world_events["passable"].mean()))
        parts.append(
            f"The last gap report ({_m(a.track[-1].value)}, {a.age:.2f} s old) was taken while both of you kept moving: closing speed "
            f"{closing:.2f} m/s ({agent_speed:.2f} + your {r.speed:.2f}), so the gap now is about {_m(a.track[-1].value)} − {closing:.2f} × {a.age:.2f} = {_m(now)}. "
            f"Measured free width with both of you at the walls is about {_m(free)} against the {_m(r.passing_clearance)} needed; "
            f"P(room to pass) = {_p(p_pass)}."
        )
    elif obs.scene == "p5_braking":
        stops = ", ".join(f"{lvl} {_m(ph.stopping_distance(r.speed, r.latency, r.decel(lvl)), arm)}" for lvl in ("gentle", "moderate", "hard"))
        parts.append(f"Stopping distances from {r.speed:.2f} m/s are {stops} against a reported gap of {_m(obs.env.obstacle_distance.value, arm)}.")
    elif obs.scene == "p5_turning":
        parts.append(f"At the current {r.speed:.2f} m/s the turn of radius {obs.env.turn_radius:.2f} m means {r.speed**2 / obs.env.turn_radius:.2f} m/s² sideways.")
    family = SCENE_FAMILY[obs.scene]
    if obs.agent is not None and obs.agent.cue is not None and family is not None:
        prior = cfg.agent(obs.agent.kind)["intent_prior"][family]
        post = intent_posterior(cfg, obs.agent.kind, family, obs.agent.cue, obs.agent.age)
        prior_txt = ", ".join(f"{k} {v:.2f}" for k, v in prior.items())
        post_txt = ", ".join(f"{k} {v:.2f}" for k, v in zip(INTENTS[family], post, strict=True))
        parts.append(
            f"The intent prior for {_KIND_NAMES[obs.agent.kind]} ({prior_txt}), updated by the cue '{obs.agent.cue}' and a {obs.agent.age:.2f} s old track, "
            f"gives {post_txt}."
        )
    elif obs.agent is not None and obs.agent.group == "ball":
        parts.append("A ball has no intent; its future path is set by the hidden rolling resistance of the floor.")
    if obs.container is not None:
        c = obs.container
        lid = {"open": "no lid", "lid": "a lid", "unknown": "an unseen lid status"}[c.lid]
        parts.append(f"Fill is reported as {c.fill.value:.2f} (σ_eff {c.fill.sigma_eff:.2f}) with {lid}; slosh factor k and floor tilt are hidden.")
    return parts


def _cost_breakdown(cfg: SimgenConfig, out: Outcomes, recipe: dict, label: dict) -> str:
    model = cost_model(cfg.values, recipe.get("criterion"), out.body, out.loaded)
    costs = model["costs"]
    rows = []
    for key, canon in recipe["options"].items():
        i = out.index(canon)
        terms = []
        for event, item in out.cost_items.items():
            w = float(event_weights(out, model, event)[i].mean())
            if w > 0:
                terms.append(f"{w:.2f}×{costs[item]:g} {event}")
        delay = float(out.quantities["delay"][i].mean())
        if abs(delay) > 0.005:
            terms.append(f"{delay:.2f} s×{costs['delay_per_s']:g}")
        if out.hard_brake is not None and out.hard_brake[i] and model["hard_brake"] > 0:
            terms.append(f"{model['hard_brake']:g} hard braking")
        rows.append((label["expected_cost"][key], f"{key} = {' + '.join(terms) or '0'} = {label['expected_cost'][key]:.2f}"))
    rows.sort()
    shown = "; ".join(text for _, text in rows[:4])
    probs = sorted(label["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    criterion = f" under the question's rule '{recipe['criterion']}'" if recipe.get("criterion") else ""
    scaled = " (contact costs scaled by impact speed)" if model["speed_scaled"] else ""
    return (
        f"Expected costs over K = {out.quantities['delay'].shape[1]} worlds{criterion}{scaled}: {shown}. "
        f"Softmax of −(E − min E)/τ with τ = {cfg.values['tau']:g} gives " + ", ".join(f"{k} {v:.2f}" for k, v in probs) + "."
    )


def _truth(cfg: SimgenConfig, obs: Observation, world: World, true_out: Outcomes, recipe: dict) -> str:
    v = world.values
    bits = []
    family = SCENE_FAMILY[obs.scene]
    if obs.agent is not None and family is not None and v.get("intent", -1) >= 0:
        bits.append(f"the intent was '{INTENTS[family][int(v['intent'])]}'")
    if "fill" in v:
        lid = "with a lid" if v.get("lid", 0) > 0.5 else "open"
        bits.append(f"the fill was {v['fill']:.2f} (k = {v['k']:.2f}, {lid})")
    best = None
    if recipe.get("options"):
        per_world = world_costs(true_out, cost_model(cfg.values, recipe.get("criterion"), true_out.body, true_out.loaded))[:, 0]
        per = {key: float(per_world[true_out.index(c)]) for key, c in recipe["options"].items()}
        best = min(per, key=per.get)
    text = "In the true world (one more draw from the same posterior) " + ("; ".join(bits) if bits else "nothing was hidden beyond sensor noise")
    if best is not None:
        text += f", and the cheapest option in hindsight was {best}"
    return text + "."


def write_rationale(
    cfg: SimgenConfig, obs: Observation, world: World, out: Outcomes, true_out: Outcomes, recipe: dict, label: dict, exact: dict | None = None
) -> str:
    return " ".join(_lead(cfg, obs, out, exact or {}) + [_cost_breakdown(cfg, out, recipe, label), _truth(cfg, obs, world, true_out, recipe)])
