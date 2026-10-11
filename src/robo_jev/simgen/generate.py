"""생성기: 판단 순간 하나 → `simgen-request/0` 레코드 하나, 그리고 층화·쌍을 붙인 묶음 생성 (Task v0a 지시서 §2·§3·§6).

흐름(docs/12 §2): ① 틀(분야·몸·장면·상대·감각·문체) → 결정 여유 → ② 닻 세계(여유에 맞춘 세계) → ③ 관측 → 질문 → 사후 세계 K개 →
④ 갈라 굴리기 → ⑤ 정답 → 계획기 권고·힌트 → 상태 글 → 참 결과(평가용)·계산 과정.

**참 세계는 정답의 사후에서 뽑는다(리뷰 1 I4).** 결정 여유에 맞춰 푼 '닻' 세계는 관측을 만드는 데만 쓴다 — 그래서 경계 집중은 관측(보고된
명목 여유)에 걸린다. `truth.realized`를 만드는 참 세계는 그 관측의 사후(정답과 같은 속성 목록 사전)에서 따로 한 번 뽑은 세계다
(갈래 ``truth``). 그래서 `truth.realized`가 정답에 대해 보정되고, 닻을 풀 때 생긴 결합(예: 채움을 로봇의 감속으로 맞춘 것)이 참값에
남지 않는다. 닻 세계의 여유는 `meta.margins`(설계 목표)에, 참 세계의 여유는 `truth.world.margins`에 적는다.

**누출 경계.** 상태 글·질문·사후 표본·굴리기·정답은 관측에서만 나온다. 닻 세계는 관측을 만드는 데와 `meta.margins`에만 간다.
같은 관측을 내는 두 닻 세계는 `meta.margins`(여유가 숨은 값에 걸린 경우) 말고는 바이트 단위로 같은 레코드를 낸다(시험).

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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from robo_jev.simgen import GENERATOR_VERSION, ID_PREFIX, SCHEMA
from robo_jev.simgen.catalog import DOMAINS, INTENTS, SCENE_FAMILY, SCENE_PHENOMENA, SimgenConfig, default_config
from robo_jev.simgen.labels import cost_model, expected_costs, label_for, realized, uses_exact
from robo_jev.simgen.narrate import RenderPlan, compute_hint, render_state
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.posterior import exact_labels, sample_posterior
from robo_jev.simgen.questions import write_questions
from robo_jev.simgen.rationale import write_rationale
from robo_jev.simgen.record import dumps
from robo_jev.simgen.rollout import Outcomes, available_options, rollout
from robo_jev.simgen.scenes import Context, Infeasible, build_world, draw_context, draw_margins, flip_numeric, margins_of, observe
from robo_jev.simgen.spec import Batch, stream
from robo_jev.simgen.world import World

__all__ = [
    "Moment",
    "PAIR_UNIT_SHARE",
    "assemble",
    "build_moment",
    "generate_batch",
    "generate_record",
    "git_short_sha",
    "plan_units",
    "recommend",
    "truth_world",
]

#: 쌍인 단위의 비율 q — 쌍에 든 레코드 비율 2q/(1 + q) ≈ 10 %.
PAIR_UNIT_SHARE = 0.0526
NUMERIC_PAIR_SHARE = 0.6
RECOMMENDATION_SHARE = 0.5      # planner가 아닌 문체에서 권고를 넣는 비율(planner는 늘)
#: 틀린 권고가 가능할 때(그 순간 가능한 행동 가운데 최선보다 2τ 이상 나쁜 것이 있을 때) 틀리게 둘 비율 — 전체로 약 25 %(지시서 §3.8;
#: 실제 비율은 stats.json의 recommendation.wrong_share).
WRONG_RECOMMENDATION_SHARE = 0.31
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

    def look(w: World, skin: int) -> Observation:
        return observe(cfg, w, ctx, stream(seed, "observe"), stream(seed, "format"), stream(seed, "skin", skin))

    if variant == "numeric":
        # 경계 너머로 옮긴 값이 보고에서도 달라져야 쌍이다(채움을 말 구간으로 보고하면 같은 구간에 머물 수 있다) — 관측이 바뀌는 첫 후보.
        base_obs = look(world, 0)
        moved = next(((w, name) for w, name in flip_numeric(cfg, world, stream(seed, "pair")) if look(w, 0) != base_obs), None)
        if moved is None:
            variant = "wording"
        else:
            world, flipped = moved
    if variant == "wording":
        skin_variant = 1
    obs = look(world, skin_variant)
    return Moment(seed, ctx, world, obs, attempt + 1, attempt // 8, variant, flipped)


def recommend(cfg: SimgenConfig, seed: int, style: str, out: Outcomes) -> dict:
    """계획기 권고(리뷰 1 I5): 그 순간 **가능한 모든 행동**(``out.options``)을 기본 비용표(기준 없음 — 계획표 글에는 기준이 없다)로
    비교한다. 맞는 권고는 최선에서 τ 안쪽 가운데 하나, 틀린 권고는 최선보다 2τ 이상 나쁜 것 가운데 하나(그런 행동이 있을 때
    WRONG_RECOMMENDATION_SHARE의 확률로). 기대 비용 전부를 함께 돌려준다(meta에 적어 시험·통계가 다시 잰다)."""
    draws = stream(seed, "recommend").random(3)
    ec = expected_costs(out, cost_model(cfg.values, None, out.body, out.loaded))
    costs = {option: round(float(ec[i]), 4) for i, option in enumerate(out.options)}
    best = min(costs.values())
    tau, gap = float(cfg.values["tau"]), float(cfg.values["wrong_recommendation_min_gap"])
    correct_pool = [o for o in out.options if costs[o] - best <= tau]
    wrong_pool = [o for o in out.options if costs[o] - best >= gap]
    present = style == "planner" or draws[0] < RECOMMENDATION_SHARE
    if not present:
        return {"present": False, "canonical": None, "wrong": False, "gap": None, "costs": costs}
    wrong = bool(draws[1] < WRONG_RECOMMENDATION_SHARE and wrong_pool)
    pool = wrong_pool if wrong else correct_pool
    canonical = pool[min(int(draws[2] * len(pool)), len(pool) - 1)]
    return {"present": True, "canonical": canonical, "wrong": wrong, "gap": round(costs[canonical] - best, 4), "costs": costs}


def _hint(cfg: SimgenConfig, seed: int, obs: Observation) -> str | None:
    if stream(seed, "hint").random() >= HINT_SHARE:
        return None
    nominal = None
    if obs.agent is not None:
        lo, hi = cfg.agent(obs.agent.kind)["radius"]
        nominal = (lo + hi) / 2.0
    return compute_hint(obs, nominal, k_ref=cfg.catalog["margins"]["spill_k_ref"], g=cfg.catalog["g"])


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def truth_world(cfg: SimgenConfig, obs: Observation, draw: Batch) -> World:
    """사후에서 뽑은 세계 하나(크기 1의 batch)를 World로 — 여유를 다시 재고 계산 과정 문단에 쓰려고. 관측된 것(로봇·용기 규격·장면 기하)은
    관측에서 그대로 온다."""
    values = {key: float(value[0]) for key, value in draw.values.items()}
    container = None if obs.container is None else {"kind": obs.container.kind, "content": obs.container.content, "phrases": [obs.container.phrase]}
    geometry: dict[str, float] = {}
    if obs.env.turn_radius is not None:
        geometry = {"radius": obs.env.turn_radius, "angle_deg": obs.env.turn_angle_deg, "approach": obs.env.turn_approach.value}
    if obs.env.robot_lateral is not None:
        geometry = {"robot_lateral": obs.env.robot_lateral}
    agent_kind = obs.agent.kind if obs.agent is not None else None
    world = World(obs.scene, obs.domain, obs.variant, obs.robot, agent_kind, container, values, {}, geometry=geometry)
    return replace(world, margins=margins_of(cfg, world))


def _truth_record(world: World) -> dict:
    out: dict = {"source": "posterior_draw", "agent_kind": world.agent_kind}
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
    id_prefix: str = ID_PREFIX,
    forced: dict | None = None,
) -> dict:
    """판단 순간 하나 → 레코드(dict)."""
    seed, ctx, anchor, obs = moment.seed, moment.ctx, moment.world, moment.obs
    k = int(k or cfg.catalog["posterior_k"])
    options = available_options(obs)
    qset = write_questions(cfg, obs, options, stream(seed, "questions"))
    batch = sample_posterior(cfg, obs, k, stream(seed, "posterior"))
    out: Outcomes = rollout(cfg, obs, batch, options)
    exact = exact_labels(cfg, obs)
    labels = {q.name: label_for(out, q.recipe, cfg.values, exact) for q in qset.questions}
    primary = next(q for q in qset.questions if q.name == qset.primary_action)
    recommendation = recommend(cfg, seed, ctx.style, out)
    rec_key = next((key for key, canon in primary.recipe["options"].items() if canon == recommendation["canonical"]), None)
    hint = _hint(cfg, seed, obs)
    plan = RenderPlan(ctx.style, options, recommendation["canonical"], hint)
    state = render_state(obs, plan, stream(seed, "render"))
    # 참 세계 = 정답의 사후에서 한 번 더 뽑은 세계(닻 세계가 아니다 — 리뷰 1 I4)
    true_world = truth_world(cfg, obs, sample_posterior(cfg, obs, 1, stream(seed, "truth")))
    true_out = rollout(cfg, obs, true_world.as_batch(), options)
    truth = {"world": _truth_record(true_world), "realized": {q.name: realized(true_out, q.recipe, cfg.values) for q in qset.questions}}
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
        "recommendation": {**recommendation, "option": rec_key, "question": primary.name},
        "margins": {name: _r(value) for name, value in anchor.margins.items()},
        "margin_norm": {name: _r(value / cfg.catalog["margins"][name]["sigma"][ctx.body]) for name, value in anchor.margins.items()},
        "pair": pair,
        "key_question": qset.key_question,
        "key_style": qset.key_style,
        "criteria": {q.name: q.recipe["criterion"] for q in qset.questions if q.recipe["kind"] == "action"},
        "distractors": len(obs.skin.distractors),
        "basis": {q.name: {k2: v for k2, v in q.recipe.items() if k2 not in ("options", "answer")} for q in qset.questions},
        "options": {q.name: q.recipe["options"] for q in qset.questions if q.recipe["kind"] in ("action", "state")},
        "posterior_k": k,
        "exact_labels": [q.name for q in qset.questions if uses_exact(q.recipe, exact)],
        "sides": None if obs.env.sides is None else [s.status for s in obs.env.sides],
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
        "rationale": write_rationale(cfg, obs, true_world, out, true_out, primary.recipe, labels[primary.name], exact),
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


def generate_batch(n: int, base_seed: int = 0, *, cfg: SimgenConfig | None = None, git: str | None = None, k: int | None = None, id_prefix: str = ID_PREFIX):
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
