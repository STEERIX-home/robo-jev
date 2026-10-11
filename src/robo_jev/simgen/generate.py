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
from robo_jev.simgen.catalog import DOMAINS, INTENTS, SCENE_FAMILY, SCENE_GROUP, SCENE_PHENOMENA, SimgenConfig, default_config
from robo_jev.simgen.labels import cost_model, expected_costs, label_for, realized, uses_exact
from robo_jev.simgen.narrate import RenderPlan, compute_hint, render_state
from robo_jev.simgen.observation import Observation
from robo_jev.simgen.posterior import exact_labels, sample_posterior
from robo_jev.simgen.questions import Surface, write_questions
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
_WORLD_ATTEMPTS = 192


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
    split: str = "train"                 # 나눔 이름(train / T1 … T7)
    pool: str = "train"                  # 질문 문구 묶음(train / heldout — T2 전용)
    holdout: tuple = ()                  # 이 관측이 들어간 빼 둔 칸(T3 칸 이름, 빼 둔 문체·상황·묶음)
    noise: int = 0                       # 관측 잡음 흐름 번호(받아들인 시도 — 보고가 빼 둔 칸에 닿으면 다른 잡음으로도 다시 뽑는다)

    def look(self, cfg: SimgenConfig, world: World, skin: int = 0) -> Observation:
        """이 순간의 잡음·서식·옷 흐름으로 세계 하나를 다시 관측한다(누출 시험·T6 변형이 같은 잡음을 쓰게)."""
        return observe(cfg, world, self.ctx, stream(self.seed, "observe", self.noise), stream(self.seed, "format"), stream(self.seed, "skin", skin))


def git_short_sha(repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo or Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class HeldOutError(RuntimeError):
    """나눔이 만들 수 없는 칸만 나왔다(생성 시점 거절이 끝내 통과하지 못함)."""


def build_moment(cfg: SimgenConfig, seed: int, domain: str, forced: dict | None = None, variant: str | None = None, split=None, holdout=None) -> Moment:
    """틀 → 여유 → 닻 세계 → (쌍이면 변형) → 관측. **나눔(split)이 만들 수 없는 칸이면 닻 세계를 다시 뽑는다**(생성 시점 거절 — holdout:
    학습은 빼 둔 문체·상황·묶음·조합을 고를 수 없고, 변수 범위는 뽑는 범위를 좁힌 뒤 보고값으로 한 번 더 검사한다)."""
    from robo_jev.simgen.holdout import cells_hit, load_holdout, split_for, violations

    holdout = holdout or load_holdout()
    split = split or split_for("train", holdout)
    forced = dict(forced or {})
    ctx = draw_context(cfg, stream(seed, "context"), domain, forced, split, holdout)
    pool = forced.get("pool", split.pool)
    world, reasons = None, ["no feasible world"]
    attempt = 0
    noise = 0

    def look(w: World, skin: int) -> Observation:
        return observe(cfg, w, ctx, stream(seed, "observe", noise), stream(seed, "format"), stream(seed, "skin", skin))

    fallback = None
    for attempt in range(_WORLD_ATTEMPTS):
        margins = draw_margins(cfg, ctx, stream(seed, "margin", attempt // 8))
        try:
            candidate = build_world(cfg, ctx, margins, stream(seed, "world", attempt))
        except Infeasible:
            continue
        # 시도마다 관측 잡음도 새로 뽑는다(잡음 하나가 보고를 빼 둔 칸 너머로 밀면 같은 잡음으로는 끝내 통과하지 못한다)
        noise = attempt
        reasons = violations(holdout, split, look(candidate, 0), ctx.style, ctx.setting, pool)
        if reasons:
            continue
        if candidate.parts.get("inexact"):
            # 목표 여유에 닿지 못한 닻(P3 — 가장 가까운 띠): 정확한 닻을 더 찾아보고, 끝내 없을 때만 쓴다
            fallback = fallback or (candidate, attempt)
            continue
        world = candidate
        break
    if world is None and fallback is not None:
        world, attempt = fallback
        noise = attempt
    if world is None:
        raise HeldOutError(f"seed {seed}: no world for {ctx} within split {split.name}: {reasons}")
    flipped = None
    skin_variant = 0
    if variant == "numeric":
        # 경계 너머로 옮긴 값이 보고에서도 달라져야 쌍이다(채움을 말 구간으로 보고하면 같은 구간에 머물 수 있다) — 관측이 바뀌는 첫 후보이고
        # 그 관측도 나눔 안이어야 한다.
        base_obs = look(world, 0)
        moved = None
        for w, name in flip_numeric(cfg, world, stream(seed, "pair")):
            o = look(w, 0)
            if o != base_obs and not violations(holdout, split, o, ctx.style, ctx.setting, pool):
                moved = (w, name)
                break
        if moved is None:
            variant = "wording"
        else:
            world, flipped = moved
    if variant == "wording":
        # 같은 사실·다른 문구: 옷 갈래 1부터, 상태 글이 기준과 같으면(고를 문구가 적은 상황 — 글에 쓰이지 않는 문구만 달라진 때) 다음 갈래로
        plan = RenderPlan(ctx.style, available_options(look(world, 0)))

        def text(skin: int) -> str:
            return render_state(look(world, skin), plan, stream(seed, "render"))

        base_text = text(0)
        skin_variant = next((i for i in range(1, 17) if text(i) != base_text), 1)
    obs = look(world, skin_variant)
    held = tuple(cells_hit(holdout, obs))
    if ctx.style in holdout.styles:
        held += ("style:" + ctx.style,)
    if ctx.setting in holdout.heldout_settings(domain):
        held += (f"setting:{domain}[{ctx.setting}]",)
    if pool == holdout.paraphrase_pool:
        held += ("pool:" + pool,)
    from robo_jev.simgen.catalog import SCENE_GROUP

    if SCENE_GROUP[ctx.scene] in holdout.groups:
        held += ("group:" + SCENE_GROUP[ctx.scene],)
    return Moment(seed, ctx, world, obs, attempt + 1, attempt // 8, variant, flipped, split.name, pool, held, noise)


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
    return compute_hint(obs, nominal, k_ref=cfg.catalog["margins"]["spill_k_ref"], g=cfg.catalog["g"], lang=obs.skin.lang)


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
    agent_kind = obs.agent.kind if obs.agent is not None else (obs.stale.kind if obs.stale is not None else None)
    parts: dict = {}
    from robo_jev.simgen.catalog import scene_kind

    if scene_kind(obs.scene) is not None:
        from robo_jev.simgen import p3_world, p4_world, p6_world

        module = {"p3": p3_world, "p4": p4_world, "p6": p6_world}[scene_kind(obs.scene)]
        parts, extra, more_geometry = module.truth_parts(cfg, obs)
        values.update(extra)
        geometry.update(more_geometry)
    world = World(obs.scene, obs.domain, obs.variant, obs.robot, agent_kind, container, values, {}, geometry=geometry, parts=parts)
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


def _scene_meta(obs: Observation) -> dict:
    """새 장면(Task v0b)의 장부 — 모델 입력이 아니다."""
    out: dict = {}
    if obs.load is not None:
        out["load"] = {"kind": obs.load.kind, "deck": obs.load.deck}
    if obs.stack is not None:
        out["stack"] = {"kind": obs.stack.kind, "cue": obs.stack.cue}
    if obs.place is not None:
        out["place"] = {"family": obs.place.family, "kind": obs.place.kind, "class": obs.place.cls, "measured": sorted(obs.place.readings),
                        "image_age_s": max(obs.place.ages.values()) if obs.place.ages else 0.0}  # fmt: skip
    if obs.stale is not None:
        out["stale"] = {"kind": obs.stale.kind, "age_s": obs.stale.age, "sight": obs.stale.sight}
    if obs.corner is not None:
        out["corner"] = {"side": obs.corner.side}
    if obs.conflict is not None:
        out["conflict"] = {"cue": obs.conflict.cue, "lidar": obs.conflict.lidar, "camera": obs.conflict.camera}
    return out


def _holdout_vars(obs: Observation) -> dict[str, float]:
    """이 관측의 T3 칸 변수 값(글에 보인 값) — 학습 세트 전부를 훑는 겹침 검사가 레코드만으로 칸을 다시 잴 수 있게."""
    from robo_jev.simgen.holdout import load_holdout, observed_value

    out = {}
    for cell in load_holdout().ranges:
        if obs.scene in cell.scenes and cell.var not in out:
            value = observed_value(obs, cell.var)
            if value is not None:
                out[cell.var] = round(float(value), 6)
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
    surface: dict | None = None,
    split_spec: dict | None = None,
    t6: dict | None = None,
) -> dict:
    """판단 순간 하나 → 레코드(dict).

    ``t6``(교란 세트의 변형 — testsets): ``style``(서술 문체 바꾸기), ``fact_seed``(사실 순서 섞기), ``state_post``·``request_post``(글·요청의
    뒤처리), ``extra_question``(한 가지 더 정하기), ``recommendation``(기준 레코드의 권고를 그대로 — 정식 선택지·있음·틀림),
    ``posterior_source``((기준 관측, 사후 세계 변환) — 거울상의 공통 난수), ``options``(선택지 순서 — 거울상은 기준의 순서를 뒤집은 이름으로),
    ``meta``(meta.t6에 적을 것)."""
    t6 = t6 or {}
    seed, ctx, anchor, obs = moment.seed, moment.ctx, moment.world, moment.obs
    style = t6.get("style", ctx.style)
    k = int(k or cfg.catalog["posterior_k"])
    options = tuple(t6.get("options") or available_options(obs))
    surface_obj = Surface(lang=ctx.lang, pool=moment.pool, **(surface or {}))
    qset = write_questions(cfg, obs, options, stream(seed, "questions"), surface_obj)
    if t6.get("extra_question"):
        from robo_jev.simgen.questions import candidates, identity

        used = {identity(q.recipe) for q in qset.questions}
        names = {q.name for q in qset.questions}
        fresh = [c for c in candidates(cfg, obs, options, stream(seed, "t6-extra"), surface_obj, qset.key_style) if c.ident not in used]
        if fresh:
            extra = fresh[int(stream(seed, "t6-pick").integers(len(fresh)))].build()
            while extra.name in names:
                extra.name += "_x"
            qset.questions.append(extra)
    source_obs, transform = t6.get("posterior_source", (obs, None))
    batch = sample_posterior(cfg, source_obs, k, stream(seed, "posterior"))
    if transform is not None:
        batch = transform(batch)
    out: Outcomes = rollout(cfg, obs, batch, options)
    exact = exact_labels(cfg, obs)
    labels = {q.name: label_for(out, q.recipe, cfg.values, exact) for q in qset.questions}
    primary = next(q for q in qset.questions if q.name == qset.primary_action)
    recommendation = recommend(cfg, seed, style, out)
    if "recommendation" in t6:
        keep = t6["recommendation"]
        recommendation.update(present=keep["present"], canonical=keep["canonical"], wrong=keep["wrong"])
        recommendation["gap"] = None if keep["canonical"] is None else round(recommendation["costs"][keep["canonical"]] - min(recommendation["costs"].values()), 4)
    rec_key = next((key for key, canon in primary.recipe["options"].items() if canon == recommendation["canonical"]), None)
    hint = _hint(cfg, seed, obs)
    plan = RenderPlan(style, options, recommendation["canonical"], hint, t6.get("fact_seed"))
    state = render_state(obs, plan, stream(seed, "render"))
    if t6.get("state_post"):
        state = t6["state_post"](state)
    # 참 세계 = 정답의 사후에서 한 번 더 뽑은 세계(닻 세계가 아니다 — 리뷰 1 I4). 행동 결과가 확률인 장면(P4)은 결과도 뽑는다(갈래
    # truth-outcome); 다시 보기의 결정 규칙은 정답의 사후에서(policy).
    true_batch = sample_posterior(cfg, source_obs, 1, stream(seed, "truth"))
    if transform is not None:
        true_batch = transform(true_batch)
    true_world = truth_world(cfg, obs, true_batch)
    true_out = rollout(cfg, obs, true_batch, options, rng=stream(seed, "truth-outcome"))
    true_out.policy = out
    truth = {"world": _truth_record(true_world), "realized": {q.name: realized(true_out, q.recipe, cfg.values) for q in qset.questions}}
    a = obs.agent
    meta = {
        "scene": ctx.scene,
        "phenomena": list(SCENE_PHENOMENA[ctx.scene]),
        "group": SCENE_GROUP[ctx.scene],
        "domain": ctx.domain,
        "body": ctx.body,
        "sensor_suite": ctx.sensor_suite,
        "variant": ctx.variant,
        "maneuver": ctx.maneuver,
        "setting": ctx.setting,
        "split": moment.split,
        "holdout": list(moment.holdout),
        "pool": moment.pool,
        "agent": None if a is None else {"kind": a.kind, "group": a.group, "cue": a.cue, "velocity_mode": "direct" if a.speed is not None else "two_fix", "age_s": a.age},
        "container": None if obs.container is None else {"kind": obs.container.kind, "content": obs.container.content, "fill_mode": obs.container.fill_mode, "lid_view": obs.container.lid},
        **_scene_meta(obs),
        "style": style,
        "language": ctx.lang,
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
        "options": {q.name: q.recipe["options"] for q in qset.questions if q.recipe["kind"] in ("action", "state", "voi")},
        "posterior_k": k,
        "exact_labels": [q.name for q in qset.questions if uses_exact(q.recipe, exact)],
        "sides": None if obs.env.sides is None else [s.status for s in obs.env.sides],
        "margin_redraws": moment.margin_redraws,
        "margin_fallback": list(anchor.parts.get("inexact", ())),
        "holdout_vars": _holdout_vars(obs),
    }
    if t6.get("meta"):
        meta["t6"] = t6["meta"]
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
        "split": split_spec,
        "surface": surface,
    }
    return {
        "schema": SCHEMA,
        "id": f"{id_prefix}-{index:06d}",
        "generator": generator,
        "meta": meta,
        "request": (t6.get("request_post") or (lambda r: r))({"state": state, "questions": {q.name: q.request_entry() for q in qset.questions}}),
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
    split: dict | None = None,
) -> dict:
    """레코드 하나를 generator 칸의 열쇠(seed, 강제한 칸, 쌍 변형, 나눔)로 다시 만든다."""
    cfg = cfg or default_config()
    moment = build_moment(cfg, seed, domain, forced, variant, split_from(split))
    return assemble(cfg, moment, index, git=git, k=k, pair=pair, forced=forced, split_spec=split)


def split_from(spec: dict | None):
    """레코드의 generator.split(나눔 규칙의 dict) → Split(없으면 학습)."""
    from robo_jev.simgen.holdout import Split, split_for

    if not spec:
        return split_for("train")
    fields = {**spec, "styles": tuple(spec["styles"]), "groups": tuple(spec["groups"])}
    return Split(**fields)


def split_spec(split) -> dict:
    return {"name": split.name, "styles": list(split.styles), "settings": split.settings, "pool": split.pool, "groups": list(split.groups),
            "cell": split.cell, "train_ranges": split.train_ranges, "ko_share": split.ko_share}  # fmt: skip


def unit_seed(base_seed: int, unit: int) -> int:
    """묶음 seed와 단위 번호 → 레코드 seed(32비트)."""
    return int(np.random.SeedSequence([int(base_seed), int(unit)]).generate_state(1)[0])


def plan_units(n: int, base_seed: int, *, groups: tuple[str, ...] | None = None, seed_start: int | None = None, pair_share: float = PAIR_UNIT_SHARE) -> list[dict]:
    """레코드 n개의 단위 계획: 분야는 단위마다 번갈아, 현상·조합 묶음은 분야 한 바퀴마다 다음 것으로(분야 × 묶음 층화 — Task v0b §5),
    쌍은 약 10 %의 레코드. ``seed_start``가 있으면 레코드 seed = 시작 + 단위 번호(나눔마다 겹치지 않는 seed 범위 — holdout-v0b.yaml).

    단위 = {"seed", "domain", "group", "indices": [i] 또는 [i, i + 1], "pair_kind": None | "numeric" | "wording"}.
    """
    from robo_jev.simgen.catalog import TRAIN_GROUPS

    groups = tuple(groups or TRAIN_GROUPS)
    rng = stream(base_seed, "plan")
    units, unit, index = [], 0, 0
    while index < n:
        draws = rng.random(2)
        is_pair = draws[0] < pair_share and index + 1 < n
        units.append({
            "seed": (seed_start + unit) if seed_start is not None else unit_seed(base_seed, unit), "domain": DOMAINS[unit % len(DOMAINS)],
            "group": groups[(unit // len(DOMAINS)) % len(groups)], "indices": [index, index + 1] if is_pair else [index],
            "pair_kind": ("numeric" if draws[1] < NUMERIC_PAIR_SHARE else "wording") if is_pair else None,
        })  # fmt: skip
        index += 2 if is_pair else 1
        unit += 1
    return units


def generate_unit(cfg: SimgenConfig, unit: dict, *, split=None, forced: dict | None = None, git: str | None = None, k: int | None = None,
                  id_prefix: str = ID_PREFIX, surface: dict | None = None) -> list[dict]:  # fmt: skip
    """단위 하나(레코드 하나 또는 쌍 둘)."""
    from robo_jev.simgen.holdout import split_for

    split = split or split_for("train")
    forced = {"group": unit["group"], **(forced or {})}
    seed, domain = unit["seed"], unit["domain"]
    common = {"git": git, "k": k, "id_prefix": id_prefix, "forced": forced, "split_spec": split_spec(split), "surface": surface}
    if unit["pair_kind"] is None:
        return [assemble(cfg, build_moment(cfg, seed, domain, forced, None, split), unit["indices"][0], **common)]
    base = build_moment(cfg, seed, domain, forced, None, split)
    variant = build_moment(cfg, seed, domain, forced, unit["pair_kind"], split)
    kind = variant.variant  # numeric으로 못 옮기면 wording 쌍이 된다
    i, j = unit["indices"]
    ids = (f"{id_prefix}-{i:06d}", f"{id_prefix}-{j:06d}")
    return [
        assemble(cfg, base, i, pair={"of": ids[1], "kind": kind, "role": "base"}, **common),
        assemble(cfg, variant, j, pair={"of": ids[0], "kind": kind, "role": "variant", "flipped": variant.flipped}, **common),
    ]


def generate_batch(n: int, base_seed: int = 0, *, cfg: SimgenConfig | None = None, git: str | None = None, k: int | None = None, id_prefix: str = ID_PREFIX,
                   split=None, seed_start: int | None = None):  # fmt: skip
    """레코드 n개를 차례로 낸다(생성기). 쌍의 두 레코드는 함께 만들고 서로의 id를 `meta.pair.of`로 가리킨다. 기본 나눔은 학습(train)."""
    from robo_jev.simgen.holdout import split_for

    cfg = cfg or default_config()
    split = split or split_for("train")
    for unit in plan_units(n, base_seed, groups=split.groups, seed_start=seed_start):
        yield from generate_unit(cfg, unit, split=split, git=git, k=k, id_prefix=id_prefix)


def write_jsonl(path: Path, records) -> dict:
    """레코드를 JSONL로 쓰고 (개수, 걸린 시간)을 돌려준다."""
    start = time.perf_counter()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
            count += 1
    return {"records": count, "seconds": time.perf_counter() - start}
