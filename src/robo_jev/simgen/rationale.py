"""정답의 계산 과정 한 문단 (Task v0a 지시서 §2 `rationale`, docs/12 §11 '계산 과정을 안으로 접기').

사람이 읽는 용도이고, 나중의 '계산을 안으로 접기' 실험 재료다. 모델 입력이 아니다 — 그래서 참 세계(숨은 값)를 마지막 문장에서 말해도 된다.
문단은 (1) 장면의 핵심 계산(보고된 수치 → 여유), (2) 사후 의도·채움, (3) 기본 행동 질문의 기대 비용 분해와 소프트맥스, (4) 참 세계에서
실제로 일어난 것의 순서다.
"""

from __future__ import annotations

import numpy as np

from robo_jev.simgen import physics as ph
from robo_jev.simgen.catalog import INTENTS, SCENE_FAMILY, SimgenConfig
from robo_jev.simgen.labels import costs_for
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.posterior import intent_posterior
from robo_jev.simgen.rollout import Outcomes
from robo_jev.simgen.world import World

__all__ = ["write_rationale"]


_KIND_NAMES = {
    "adult": "an adult", "child": "a child", "elder_walker": "an older person with a walker", "cyclist": "a cyclist",
    "cart": "a person pushing a cart", "forklift": "a driven forklift", "other_robot": "another robot", "ball": "a ball", "hand": "a hand",
}  # fmt: skip


def _m(x: float, arm: bool = False) -> str:
    return f"{x:.3f} m" if arm else f"{x:.2f} m"


def _p(x: float) -> str:
    return f"{x:.2f}"


def _lead(cfg: SimgenConfig, obs: Observation, out: Outcomes) -> list[str]:
    r = obs.robot
    parts: list[str] = []
    stop_m = ph.stopping_distance(r.speed, r.latency, r.decel_moderate)
    arm = obs.body == "arm"
    firm = "the normal pause" if arm else "firm braking"
    if obs.scene in ("p1_step_in", "p1p5_step_in_carrying"):
        a = obs.agent
        parts.append(
            f"The reported gap is {_m(a.ahead.value, arm)} (σ_eff {a.ahead.sigma_eff:.3f} m); {firm} from {r.speed:.2f} m/s needs "
            f"{_m(r.speed * r.latency, arm)} of reaction plus {_m(r.speed**2 / (2 * r.decel_moderate), arm)} of braking = {_m(stop_m, arm)}, so "
            f"P(stop short with {firm}) = {_p(out.world_events['stops_short'].mean())} over the posterior worlds."
        )
    elif obs.scene == "p1_crossing":
        parts.append(
            f"If {obs.agent.descriptor.ref} keeps the same pace, the share of posterior worlds in which you clear the crossing line at least 0.3 s "
            f"before {obs.agent.descriptor.ref} reaches your path is {_p(out.world_events['clears_first'].mean())}."
        )
    elif obs.scene == "p1_head_on":
        free = obs.env.corridor_width.value - 2 * r.radius - obs.agent.size.value
        parts.append(
            f"Measured free width with both of you at the walls is about {_m(free)} against the {_m(r.passing_clearance)} needed; "
            f"P(room to pass) = {_p(out.world_events['passable'].mean())}."
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
    costs = costs_for(cfg.values, recipe.get("criterion"))
    rows = []
    for key, canon in recipe["options"].items():
        i = out.index(canon)
        terms = []
        for event, item in out.cost_items.items():
            p = float(out.events[event][i].mean())
            if p > 0:
                terms.append(f"{p:.2f}×{costs[item]:g} {event}")
        delay = float(out.quantities["delay"][i].mean())
        if abs(delay) > 0.005:
            terms.append(f"{delay:.2f} s×{costs['delay_per_s']:g}")
        rows.append((label["expected_cost"][key], f"{key} = {' + '.join(terms) or '0'} = {label['expected_cost'][key]:.2f}"))
    rows.sort()
    shown = "; ".join(text for _, text in rows[:4])
    probs = sorted(label["probabilities"].items(), key=lambda kv: -kv[1])[:3]
    criterion = f" under the question's rule '{recipe['criterion']}'" if recipe.get("criterion") else ""
    return (
        f"Expected costs over K = {out.events[next(iter(out.events))].shape[1]} worlds{criterion}: {shown}. "
        f"Softmax of −(E − min E)/τ with τ = {cfg.values['tau']:g} gives " + ", ".join(f"{k} {v:.2f}" for k, v in probs) + "."
    )


def _truth(cfg: SimgenConfig, obs: Observation, world: World, true_out: Outcomes, recipe: dict) -> str:
    v = world.values
    bits = []
    family = SCENE_FAMILY[obs.scene]
    if obs.agent is not None and family is not None and v.get("intent", -1) >= 0:
        now = INTENTS[family][int(v["intent"])]
        before = world.at_observation.get("intent_obs")
        changed = before is not None and before >= 0 and before != int(v["intent"])
        bits.append(f"the hidden intent was '{now}'" + (" (it changed after the observation)" if changed else ""))
    if "fill" in v:
        lid = "with a lid" if v.get("lid", 0) > 0.5 else "open"
        bits.append(f"the true fill was {v['fill']:.2f} (k = {v['k']:.2f}, {lid})")
    best = None
    costs = costs_for(cfg.values, recipe.get("criterion"))
    if recipe.get("options"):
        per = {key: float(np.asarray(sum(costs[item] * true_out.events[e][true_out.index(c)] for e, item in true_out.cost_items.items())
                                 + costs["delay_per_s"] * true_out.quantities["delay"][true_out.index(c)])[0])
               for key, c in recipe["options"].items()}  # fmt: skip
        best = min(per, key=per.get)
    text = "In the true world " + ("; ".join(bits) if bits else "nothing was hidden beyond sensor noise")
    if best is not None:
        text += f", and the cheapest option in hindsight was {best}"
    return text + "."


def write_rationale(cfg: SimgenConfig, obs: Observation, world: World, out: Outcomes, true_out: Outcomes, recipe: dict, label: dict) -> str:
    return " ".join(_lead(cfg, obs, out) + [_cost_breakdown(cfg, out, recipe, label), _truth(cfg, obs, world, true_out, recipe)])
