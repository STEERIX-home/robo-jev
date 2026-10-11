"""생성기: 판단 순간 하나 → `simgen-request/0` 레코드 하나, 그리고 층화·쌍을 붙인 묶음 생성 (Task v0a 지시서 §2·§3·§6).

흐름(docs/12 §2): ① 틀(분야·몸·장면·상대·감각·문체) → 결정 여유 → ② 참 세계 → ③ 관측 → 질문 → 사후 세계 K개 → ④ 갈라 굴리기 →
⑤ 정답 → 계획기 권고·힌트 → 상태 글 → 참 결과(평가용)·계산 과정.

**누출 경계.** 상태 글·질문·사후 표본·굴리기·정답은 관측에서만 나온다. 참 세계는 관측을 만드는 데와 `truth`·`rationale`에만 간다.
같은 관측에 다른 참 세계를 넣어도 `request`와 `labels`가 바이트 단위로 같다(시험).

**난수.** 레코드 seed에서 이름 붙은 갈래(:func:`~robo_jev.simgen.spec.stream`)를 나눠 쓴다 — 같은 seed면 같은 레코드(바이트 단위).
여유를 맞출 수 없으면(:class:`~robo_jev.simgen.scenes.Infeasible`) 세계 갈래를 바꿔 다시 뽑고, 8번 실패하면 여유도 다시 뽑는다(그 수를
`generator.attempts`·`meta.margin_redraws`에 남긴다).

**층화와 쌍.** :func:`generate_batch` 는 단위(레코드 하나 또는 쌍 하나)에 분야를 번갈아 배정해 분야별 몫을 같게 하고, 레코드의 약 10 %를
사실 하나만 바꾼 쌍으로 만든다: ``numeric``(결정 변수 하나만 경계 너머로 — 같은 잡음, 같은 질문 갈래) / ``wording``(숫자는 그대로,
상황 옷 단어만 바꿈 — 정답이 같다).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robo_jev.simgen import GENERATOR_VERSION, SCHEMA
from robo_jev.simgen.catalog import DOMAINS, INTENTS, SCENE_FAMILY, SCENE_PHENOMENA, SimgenConfig, default_config
from robo_jev.simgen.labels import label_for, realized
from robo_jev.simgen.narrate import RenderPlan, compute_hint, render_state
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.posterior import sample_posterior
from robo_jev.simgen.questions import write_questions
from robo_jev.simgen.rationale import write_rationale
from robo_jev.simgen.record import dumps
from robo_jev.simgen.rollout import Outcomes, available_options, rollout
from robo_jev.simgen.scenes import Context, Infeasible, build_world, draw_context, draw_margins, flip_numeric, observe
from robo_jev.simgen.spec import stream
from robo_jev.simgen.world import World

__all__ = ["Moment", "PAIR_UNIT_SHARE", "assemble", "build_moment", "generate_batch", "generate_record", "git_short_sha", "plan_units"]

#: 쌍인 단위의 비율 q — 쌍에 든 레코드 비율 2q/(1 + q) ≈ 10 %.
PAIR_UNIT_SHARE = 0.0526
NUMERIC_PAIR_SHARE = 0.6
RECOMMENDATION_SHARE = 0.5      # planner가 아닌 문체에서 권고를 넣는 비율(planner는 늘)
WRONG_RECOMMENDATION_SHARE = 0.25
HINT_SHARE = 0.3
_WORLD_ATTEMPTS = 64


@dataclass
class Moment:
    """판단 순간 하나의 재료(레코드 조립 전)."""

    seed: int
    ctx: Context
    world: World
    obs: Observation
    attempts: int
    margin_redraws: int
    variant: str | None = None
    flipped: str | None = None


def git_short_sha(repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo or Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def build_moment(cfg: SimgenConfig, seed: int, domain: str, forced: dict | None = None, variant: str | None = None) -> Moment:
    """틀 → 여유 → 참 세계 → (쌍이면 변형) → 관측."""
    ctx = draw_context(cfg, stream(seed, "context"), domain, forced)
    world = None
    attempt = 0
    for attempt in range(_WORLD_ATTEMPTS):
        margins = draw_margins(cfg, ctx, stream(seed, "margin", attempt // 8))
        try:
            world = build_world(cfg, ctx, margins, stream(seed, "world", attempt))
            break
        except Infeasible:
            continue
    if world is None:
        raise RuntimeError(f"seed {seed}: no feasible world for {ctx}")
    flipped = None
    skin_variant = 0
    if variant == "numeric":
        result = flip_numeric(cfg, world, stream(seed, "pair"))
        if result is None:
            variant = "wording"
        else:
            world, flipped = result
    if variant == "wording":
        skin_variant = 1
    obs = observe(cfg, world, ctx, stream(seed, "observe"), stream(seed, "format"), stream(seed, "skin", skin_variant))
    return Moment(seed, ctx, world, obs, attempt + 1, attempt // 8, variant, flipped)


def _recommendation(cfg: SimgenConfig, seed: int, style: str, label: dict, recipe: dict) -> dict:
    rng = stream(seed, "recommend")
    draws = rng.random(3)
    present = style == "planner" or draws[0] < RECOMMENDATION_SHARE
    if not present:
        return {"present": False, "option": None, "canonical": None, "wrong": False}
    ec = label["expected_cost"]
    best = min(ec, key=ec.get)
    gap = cfg.values["wrong_recommendation_min_gap"]
    wrong_pool = sorted(k for k, v in ec.items() if v - ec[best] >= gap)
    wrong = bool(draws[1] < WRONG_RECOMMENDATION_SHARE and wrong_pool)
    key = wrong_pool[min(int(draws[2] * len(wrong_pool)), len(wrong_pool) - 1)] if wrong else max(label["probabilities"], key=label["probabilities"].get)
    return {"present": True, "option": key, "canonical": recipe["options"][key], "wrong": wrong}


def _hint(cfg: SimgenConfig, seed: int, obs: Observation) -> str | None:
    if stream(seed, "hint").random() >= HINT_SHARE:
        return None
    nominal = None
    if obs.agent is not None:
        lo, hi = cfg.agent(obs.agent.kind)["radius"]
        nominal = (lo + hi) / 2.0
    return compute_hint(obs, nominal)


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _truth_world(world: World) -> dict:
    out: dict = {"agent_kind": world.agent_kind}
    if world.container is not None:
        out["container"] = {"kind": world.container["kind"], "content": world.container["content"]}
    values = {}
    family = SCENE_FAMILY[world.scene]
    for key, value in world.values.items():
        if key == "intent" and family is not None:
            values[key] = INTENTS[family][int(value)] if value >= 0 else None
        elif key == "tilt":
            values["tilt_deg"] = _r(np.degrees(value), 3)
        else:
            values[key] = _r(value)
    out["values"] = values
    at_obs = dict(world.at_observation)
    if family is not None and "intent_obs" in at_obs:
        at_obs["intent_obs"] = INTENTS[family][at_obs["intent_obs"]] if at_obs["intent_obs"] >= 0 else None
    if "sides" in at_obs:
        at_obs["sides"] = list(at_obs["sides"])
    out["at_observation"] = at_obs
    out["margins"] = {k: _r(v) for k, v in world.margins.items()}
    if world.geometry:
        out["geometry"] = {k: _r(v) for k, v in world.geometry.items()}
    return out


def assemble(
    cfg: SimgenConfig,
    moment: Moment,
    index: int,
    *,
    git: str | None = None,
    k: int | None = None,
    pair: dict | None = None,
    id_prefix: str = GENERATOR_VERSION,
    forced: dict | None = None,
) -> dict:
    """판단 순간 하나 → 레코드(dict)."""
    seed, ctx, world, obs = moment.seed, moment.ctx, moment.world, moment.obs
    k = int(k or cfg.catalog["posterior_k"])
    options = available_options(obs)
    qset = write_questions(cfg, obs, options, stream(seed, "questions"))
    batch = sample_posterior(cfg, obs, k, stream(seed, "posterior"))
    out: Outcomes = rollout(cfg, obs, batch, options)
    labels = {q.name: label_for(out, q.recipe, cfg.values) for q in qset.questions}
    primary = next(q for q in qset.questions if q.name == qset.primary_action)
    recommendation = _recommendation(cfg, seed, ctx.style, labels[primary.name], primary.recipe)
    hint = _hint(cfg, seed, obs)
    plan = RenderPlan(ctx.style, options, recommendation["canonical"], hint)
    state = render_state(obs, plan, stream(seed, "render"))
    true_out = rollout(cfg, obs, world.as_batch(), options)
    truth = {"world": _truth_world(world), "realized": {q.name: realized(true_out, q.recipe, cfg.values) for q in qset.questions}}
    a = obs.agent
    meta = {
        "scene": ctx.scene,
        "phenomena": list(SCENE_PHENOMENA[ctx.scene]),
        "domain": ctx.domain,
        "body": ctx.body,
        "sensor_suite": ctx.sensor_suite,
        "variant": ctx.variant,
        "agent": None if a is None else {"kind": a.kind, "group": a.group, "cue": a.cue, "velocity_mode": "direct" if a.speed is not None else "two_fix", "age_s": a.age},
        "container": None if obs.container is None else {"kind": obs.container.kind, "content": obs.container.content, "fill_mode": obs.container.fill_mode, "lid_view": obs.container.lid},
        "style": ctx.style,
        "language": "en",
        "units": {"distance": obs.fmt.distance_unit, "speed": obs.fmt.speed_unit, "time": obs.fmt.time_unit},
        "hint": hint is not None,
        "recommendation": {**recommendation, "question": primary.name},
        "margins": {name: _r(value) for name, value in world.margins.items()},
        "pair": pair,
        "key_question": qset.key_question,
        "key_style": qset.key_style,
        "criteria": {q.name: q.recipe["criterion"] for q in qset.questions if q.recipe["kind"] == "action"},
        "distractors": len(obs.skin.distractors),
        "basis": {q.name: {k2: v for k2, v in q.recipe.items() if k2 not in ("options", "answer")} for q in qset.questions},
        "options": {q.name: q.recipe["options"] for q in qset.questions if q.recipe["kind"] in ("action", "state")},
        "posterior_k": k,
        "margin_redraws": moment.margin_redraws,
    }
    generator = {
        "version": GENERATOR_VERSION,
        "catalog": cfg.versions["catalog"],
        "values": cfg.versions["values"],
        "skins": cfg.versions["skins"],
        "seed": int(seed),
        "git": git,
        "variant": moment.variant,
        "forced": {"domain": ctx.domain, **(forced or {})},
        "attempts": moment.attempts,
    }
    return {
        "schema": SCHEMA,
        "id": f"{id_prefix}-{index:06d}",
        "generator": generator,
        "meta": meta,
        "request": {"state": state, "questions": {q.name: q.request_entry() for q in qset.questions}},
        "labels": labels,
        "truth": truth,
        "rationale": write_rationale(cfg, obs, world, out, true_out, primary.recipe, labels[primary.name]),
    }


def generate_record(
    seed: int,
    index: int,
    domain: str,
    *,
    cfg: SimgenConfig | None = None,
    forced: dict | None = None,
    variant: str | None = None,
    pair: dict | None = None,
    git: str | None = None,
    k: int | None = None,
) -> dict:
    cfg = cfg or default_config()
    moment = build_moment(cfg, seed, domain, forced, variant)
    return assemble(cfg, moment, index, git=git, k=k, pair=pair, forced=forced)


def unit_seed(base_seed: int, unit: int) -> int:
    """묶음 seed와 단위 번호 → 레코드 seed(32비트)."""
    return int(np.random.SeedSequence([int(base_seed), int(unit)]).generate_state(1)[0])


def plan_units(n: int, base_seed: int) -> list[dict]:
    """레코드 n개의 단위 계획: 분야는 단위마다 번갈아(층화), 쌍은 약 10 %의 레코드.

    단위 = {"seed", "domain", "indices": [i] 또는 [i, i + 1], "pair_kind": None | "numeric" | "wording"}.
    """
    rng = stream(base_seed, "plan")
    units, unit, index = [], 0, 0
    while index < n:
        draws = rng.random(2)
        is_pair = draws[0] < PAIR_UNIT_SHARE and index + 1 < n
        units.append({
            "seed": unit_seed(base_seed, unit), "domain": DOMAINS[unit % len(DOMAINS)],
            "indices": [index, index + 1] if is_pair else [index],
            "pair_kind": ("numeric" if draws[1] < NUMERIC_PAIR_SHARE else "wording") if is_pair else None,
        })  # fmt: skip
        index += 2 if is_pair else 1
        unit += 1
    return units


def generate_batch(n: int, base_seed: int = 0, *, cfg: SimgenConfig | None = None, git: str | None = None, k: int | None = None, id_prefix: str = GENERATOR_VERSION):
    """레코드 n개를 차례로 낸다(생성기). 쌍의 두 레코드는 함께 만들고 서로의 id를 `meta.pair.of`로 가리킨다."""
    cfg = cfg or default_config()
    for unit in plan_units(n, base_seed):
        seed, domain = unit["seed"], unit["domain"]
        if unit["pair_kind"] is None:
            yield assemble(cfg, build_moment(cfg, seed, domain), unit["indices"][0], git=git, k=k, id_prefix=id_prefix)
            continue
        base = build_moment(cfg, seed, domain)
        variant = build_moment(cfg, seed, domain, None, unit["pair_kind"])
        kind = variant.variant  # numeric으로 못 옮기면 wording 쌍이 된다
        i, j = unit["indices"]
        ids = (f"{id_prefix}-{i:06d}", f"{id_prefix}-{j:06d}")
        yield assemble(cfg, base, i, git=git, k=k, id_prefix=id_prefix, pair={"of": ids[1], "kind": kind, "role": "base"})
        yield assemble(cfg, variant, j, git=git, k=k, id_prefix=id_prefix, pair={"of": ids[0], "kind": kind, "role": "variant", "flipped": variant.flipped})


def write_jsonl(path: Path, records) -> dict:
    """레코드를 JSONL로 쓰고 (개수, 걸린 시간)을 돌려준다."""
    start = time.perf_counter()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
            count += 1
    return {"records": count, "seconds": time.perf_counter() - start}
