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

**학습 데이터의 두 손잡이(Task v0.1a — 학습 나눔에서만; 시험 세트 생성 경로는 늘 기본값이고, 기본값이 아닌 손잡이를 시험 나눔에 주면
거절한다).** ``phrasing``(문구 판: ``v0b`` 기본 / ``v01`` — 학습 묶음을 넓힌다, :mod:`~robo_jev.simgen.phrasing`)과 ``fact_shuffle``(레코드마다
그 확률로 T6 '사실 순서'와 같은 장치를 켠다 — ``RenderPlan.fact_seed``; 동전과 seed는 별도 갈래 ``train-facts``). 둘 다 주 난수 갈래를
건드리지 않아 같은 seed면 순간·질문·정답이 기본값과 바이트 단위로 같고 요청 글만 다르다(시험). 기본값이 아니면 ``generator.phrasing``·
``generator.fact_shuffle``(다시 만들기 열쇠)와 ``meta.phrasing``(판과 고른 문구 [열쇠, v0b 번호, v01 번호])·``meta.fact_order``(확률, 섞었나,
seed, 글이 실제로 바뀌었나)를 적는다 — 기본값이면 적지 않는다(봉인된 시험 세트와 v0b 학습 세트가 바이트 단위로 그대로).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from robo_jev.simgen import GENERATOR_VERSION, ID_PREFIX, SCHEMA
from robo_jev.simgen.catalog import DOMAINS, INTENTS, SCENE_FAMILY, SCENE_GROUP, SCENE_PHENOMENA, SimgenConfig, default_config
from robo_jev.simgen.labels import calibrate, cost_model, expected_costs, label_for, realized, uses_exact
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
    "DEFAULT_FACT_SHUFFLE",
    "DEFAULT_PHRASING",
    "Moment",
    "PAIR_UNIT_SHARE",
    "assemble",
    "build_moment",
    "generate_batch",
    "generate_record",
    "git_short_sha",
    "pair_records",
    "plan_units",
    "rebuild_record",
    "recommend",
    "record_moment",
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
#: 학습 데이터 손잡이의 기본값(Task v0.1a) — 시험 세트는 늘 이 값
DEFAULT_PHRASING = "v0b"
DEFAULT_FACT_SHUFFLE = 0.0


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
    pair_skip: int | None = None         # 숫자 쌍의 변형: 받아들인 후보의 번호(generator.pair_skip — 레코드에서 다시 만들 수 있게, 수정 1 N2)

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


def build_moment(cfg: SimgenConfig, seed: int, domain: str, forced: dict | None = None, variant: str | None = None, split=None, holdout=None,
                 *, pair_skip: int = 0) -> Moment:  # fmt: skip
    """틀 → 여유 → 닻 세계 → (쌍이면 변형) → 관측. **나눔(split)이 만들 수 없는 칸이면 닻 세계를 다시 뽑는다**(생성 시점 거절 — holdout:
    학습은 빼 둔 문체·상황·묶음·조합을 고를 수 없고, 변수 범위는 뽑는 범위를 좁힌 뒤 보고값으로 한 번 더 검사한다).
    숫자 쌍의 변형은 받아들일 수 있는(관측이 바뀌고 나눔 안인) 후보 가운데 ``pair_skip``번째 — 쌍 단위가 핵심 정답이 0.5를 넘는 후보를 찾을
    때까지 다음 것을 본다(:func:`pair_records`); 더 없으면 단어 쌍이 된다."""
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
        skip = int(pair_skip)
        for w, name in flip_numeric(cfg, world, stream(seed, "pair")):
            o = look(w, 0)
            if o != base_obs and not violations(holdout, split, o, ctx.style, ctx.setting, pool):
                if skip == 0:
                    moved = (w, name)
                    break
                skip -= 1
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
    return Moment(seed, ctx, world, obs, attempt + 1, attempt // 8, variant, flipped, split.name, pool, held, noise,
                  int(pair_skip) if variant == "numeric" else None)


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
    """이 관측의 칸 변수 값 전부(글에 보인 값, 장면 목록과 무관하게) — 학습 세트 전부를 훑는 겹침 검사가 레코드만으로 칸을 다시 잴 수 있게."""
    from robo_jev.simgen.holdout import observed_values

    return observed_values(obs)


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
    threshold_obs: Observation | None = None,
    phrasing: str = DEFAULT_PHRASING,
    fact_shuffle: float = DEFAULT_FACT_SHUFFLE,
) -> dict:
    """판단 순간 하나 → 레코드(dict).

    ``phrasing``·``fact_shuffle``(Task v0.1a — 학습 나눔만): 문구 판(v0b / v01)과 사실 순서 섞기 확률. 기본값이 아니면 시험 나눔에서는 거절한다.

    ``threshold_obs``(숫자 쌍의 변형): 규칙 질문의 문턱을 이 관측(기준)의 값에 놓는다 — 쌍의 두 요청이 옮긴 사실 말고는 같게(수정 1 I4).

    ``t6``(교란 세트의 변형 — testsets): ``style``(서술 문체 바꾸기), ``fact_seed``(사실 순서 섞기), ``state_post``·``request_post``(글·요청의
    뒤처리), ``extra_question``(한 가지 더 정하기), ``recommendation``(기준 레코드의 권고를 그대로 — 정식 선택지·있음·틀림),
    ``posterior_source``((기준 관측, 사후 세계 변환) — 거울상의 공통 난수), ``options``(선택지 순서 — 거울상은 기준의 순서를 뒤집은 이름으로),
    ``meta``(meta.t6에 적을 것)."""
    t6 = t6 or {}
    seed, ctx, anchor, obs = moment.seed, moment.ctx, moment.world, moment.obs
    fact_shuffle = float(fact_shuffle or 0.0)
    if (phrasing != "v0b" or fact_shuffle) and (moment.split != "train" or set(t6) - {"recommendation"} or surface):
        # 시험 세트(T1~T7)와 교란 변형은 늘 v0b 문구·정해진 사실 순서(봉인된 파일을 바이트 단위로 다시 만든다); 학습 쌍의 변형(권고 유지)만 받는다
        raise ValueError(f"phrasing {phrasing!r} / fact_shuffle {fact_shuffle} are for the training split only (got split {moment.split!r})")
    if not 0.0 <= fact_shuffle <= 1.0:
        raise ValueError(f"fact_shuffle must be a probability, got {fact_shuffle}")
    style = t6.get("style", ctx.style)
    k = int(k or cfg.catalog["posterior_k"])
    options = tuple(t6.get("options") or available_options(obs))
    surface_obj = Surface(lang=ctx.lang, pool=moment.pool, **(surface or {}), phrasing=phrasing)
    rng = stream(seed, "questions")
    fact_seed, fact_order = t6.get("fact_seed"), None
    if fact_shuffle:
        # 학습의 사실 순서 섞기(T6 '사실 순서'와 같은 장치) — 동전과 seed는 별도 갈래(주 갈래 questions·render의 소비가 그대로 — 정답이 같다)
        draw = stream(seed, "train-facts")
        shuffled = bool(draw.random() < fact_shuffle)
        fact_seed = int(draw.integers(2**31)) if shuffled else None
        fact_order = {"p": fact_shuffle, "shuffled": shuffled, "seed": fact_seed}
    qset = write_questions(cfg, obs, options, rng, surface_obj, threshold_obs=threshold_obs, phrase_seed=seed)
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
    # 한 레코드의 정답은 모두 이 무게의 같은 표본에서 — 닫힌 꼴은 표본의 무게를 그 값에 맞추는 데 쓴다(수정 1 M5)
    calibration = calibrate(out, exact)
    labels = {q.name: label_for(out, q.recipe, cfg.values) for q in qset.questions}
    primary = next(q for q in qset.questions if q.name == qset.primary_action)
    recommendation = recommend(cfg, seed, style, out)
    if "recommendation" in t6:
        keep = t6["recommendation"]
        recommendation.update(present=keep["present"], canonical=keep["canonical"], wrong=keep["wrong"])
        recommendation["gap"] = None if keep["canonical"] is None else round(recommendation["costs"][keep["canonical"]] - min(recommendation["costs"].values()), 4)
        if keep.get("kept"):
            # 쌍의 변형: 기준의 권고를 그대로 보이고(사실 하나만 바뀐다), 틀림은 이 순간의 비용으로 다시 잰다(최선보다 2τ 이상 나쁘면 틀림)
            recommendation["kept"] = True
            if keep["canonical"] is not None:
                recommendation["wrong"] = bool(recommendation["gap"] >= float(cfg.values["wrong_recommendation_min_gap"]))
    rec_key = next((key for key, canon in primary.recipe["options"].items() if canon == recommendation["canonical"]), None)
    hint = _hint(cfg, seed, obs)
    plan = RenderPlan(style, options, recommendation["canonical"], hint, fact_seed)
    state = render_state(obs, plan, stream(seed, "render"))
    if fact_order is not None:
        fact_order["changed"] = fact_order["shuffled"] and state != render_state(obs, replace(plan, fact_seed=None), stream(seed, "render"))
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
        "exact_labels": [q.name for q in qset.questions if uses_exact(q.recipe, calibration["matched"])],
        "calibration": calibration,
        "sides": None if obs.env.sides is None else [s.status for s in obs.env.sides],
        "margin_redraws": moment.margin_redraws,
        "margin_fallback": list(anchor.parts.get("inexact", ())),
        "holdout_vars": _holdout_vars(obs),
    }
    if t6.get("meta"):
        meta["t6"] = t6["meta"]
    if phrasing != "v0b":
        meta["phrasing"] = {"version": phrasing, "picks": qset.picks}
    if fact_order is not None:
        meta["fact_order"] = fact_order
    generator = {
        "version": GENERATOR_VERSION,
        "catalog": cfg.versions["catalog"],
        "values": cfg.versions["values"],
        "skins": cfg.versions["skins"],
        "seed": int(seed),
        "git": git,
        "variant": moment.variant,
        "forced": {"domain": ctx.domain, **(forced or {})},
        "pair_skip": moment.pair_skip,
        "attempts": moment.attempts,
        "split": split_spec,
        "surface": surface,
    }
    if phrasing != "v0b":
        generator["phrasing"] = phrasing
    if fact_shuffle:
        generator["fact_shuffle"] = fact_shuffle
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
    pair_skip: int | None = None,
    phrasing: str = DEFAULT_PHRASING,
    fact_shuffle: float = DEFAULT_FACT_SHUFFLE,
) -> dict:
    """레코드 하나를 generator 칸의 열쇠(seed, 강제한 칸, 쌍 변형과 받아들인 후보 번호, 나눔, 문구 판·사실 순서 섞기)로 다시 만든다(혼자인
    순간 — 쌍은 :func:`rebuild_record`)."""
    cfg = cfg or default_config()
    moment = build_moment(cfg, seed, domain, forced, variant, split_from(split), pair_skip=pair_skip or 0)
    return assemble(cfg, moment, index, git=git, k=k, pair=pair, forced=forced, split_spec=split, phrasing=phrasing, fact_shuffle=fact_shuffle)


def record_moment(cfg: SimgenConfig, record: dict) -> Moment:
    """레코드의 generator 칸에서 그 판단 순간을 다시 만든다 — 숫자 쌍의 변형은 받아들인 후보 번호(``pair_skip``)까지(수정 1 N2)."""
    g = record["generator"]
    forced = {key: value for key, value in g["forced"].items() if key != "domain"}
    return build_moment(cfg, g["seed"], g["forced"]["domain"], forced or None, g["variant"], split_from(g["split"]), pair_skip=g.get("pair_skip") or 0)


def rebuild_record(cfg: SimgenConfig, record: dict) -> dict:
    """학습 세트·T1~T5·T7의 레코드를 그 자신의 칸(generator·meta.pair·id)만으로 다시 만든다. 숫자 쌍의 변형은 같은 seed의 기준 순간에서
    기준의 계획기 권고와 규칙 문턱을 다시 얻는다(:func:`pair_records`와 같은 길). T6 변형은 testsets가 다시 만든다."""
    g = record["generator"]
    prefix, index = record["id"].rsplit("-", 1)
    forced = {key: value for key, value in g["forced"].items() if key != "domain"}
    common = {"git": g["git"], "k": record["meta"]["posterior_k"], "id_prefix": prefix, "forced": forced, "split_spec": g["split"], "surface": g.get("surface"),
              "phrasing": g.get("phrasing", DEFAULT_PHRASING), "fact_shuffle": g.get("fact_shuffle", DEFAULT_FACT_SHUFFLE)}  # fmt: skip
    pair = record["meta"]["pair"]
    extra: dict = {}
    if pair and pair["role"] == "variant" and pair["kind"] == "numeric":
        base = build_moment(cfg, g["seed"], g["forced"]["domain"], forced or None, None, split_from(g["split"]))
        rec = assemble(cfg, base, 0, **common)["meta"]["recommendation"]
        extra = {"t6": {"recommendation": {"present": rec["present"], "canonical": rec["canonical"], "wrong": rec["wrong"], "kept": True}},
                 "threshold_obs": base.obs}  # fmt: skip
    out = assemble(cfg, record_moment(cfg, record), int(index), **common, **extra)
    if pair is not None:
        out["meta"]["pair"] = dict(pair)
    return out


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
                  id_prefix: str = ID_PREFIX, surface: dict | None = None, phrasing: str = DEFAULT_PHRASING,
                  fact_shuffle: float = DEFAULT_FACT_SHUFFLE) -> list[dict]:  # fmt: skip
    """단위 하나(레코드 하나 또는 쌍 둘)."""
    from robo_jev.simgen.holdout import split_for

    split = split or split_for("train")
    forced = {"group": unit["group"], **(forced or {})}
    seed, domain = unit["seed"], unit["domain"]
    common = {"git": git, "k": k, "id_prefix": id_prefix, "forced": forced, "split_spec": split_spec(split), "surface": surface, "phrasing": phrasing,
              "fact_shuffle": fact_shuffle}  # fmt: skip
    if unit["pair_kind"] is None:
        return [assemble(cfg, build_moment(cfg, seed, domain, forced, None, split), unit["indices"][0], **common)]
    base = build_moment(cfg, seed, domain, forced, None, split)
    return pair_records(cfg, base, domain, forced, unit["pair_kind"], split, unit["indices"], **common)


#: 숫자 쌍의 변형 후보를 몇 개까지 볼까(핵심 정답이 0.5를 넘는 첫 후보 — 없으면 단어 쌍)
PAIR_TRIES = 5


def pair_records(cfg: SimgenConfig, base: Moment, domain: str, moment_forced: dict, kind: str, split, indices, *, id_prefix: str = ID_PREFIX, **common) -> list[dict]:
    """쌍 단위(기준 + 변형) 두 레코드 — 학습 세트와 T5가 같이 쓴다(수정 1 I4).

    숫자 쌍: 핵심 질문이 묻는 결정 변수(:func:`~robo_jev.simgen.catalog.pair_targets`) 하나만 경계 너머로 옮기고(가능한 선택지가 그대로인
    후보만 — 재검토 1 N1 뒤로 모퉁이의 '앞에서 멈추기'가 옮긴 거리에 따라 나타나거나 사라질 수 있다), 변형은 기준의 계획기 권고를
    그대로 보이며(틀림·간격만 그 순간의 비용으로 — ``meta.recommendation.kept``), 규칙 질문의 문턱도 기준의 값에 둔다 — 그래서 두 요청은 옮긴
    값(과 그에 딸린 표시값) 말고는 같다. 핵심 질문의 정답이 0.5를 실제로 넘는 후보만 받는다(``meta.pair.target`` = 그 핵심 질문); 넘는 후보가
    없으면 단어 쌍(같은 사실·다른 상황 문구)."""
    i, j = indices
    ids = (f"{id_prefix}-{i:06d}", f"{id_prefix}-{j:06d}")
    base_rec = assemble(cfg, base, i, id_prefix=id_prefix, **common)
    key = base_rec["meta"]["key_question"]
    p0 = base_rec["labels"][key]["p_yes"]
    rec = base_rec["meta"]["recommendation"]
    keep = {"present": rec["present"], "canonical": rec["canonical"], "wrong": rec["wrong"], "kept": True}
    variant_rec, variant = None, None
    if kind == "numeric":
        for skip in range(PAIR_TRIES):
            moment = build_moment(cfg, base.seed, domain, moment_forced, "numeric", split, pair_skip=skip)
            if moment.variant != "numeric":
                break
            if available_options(moment.obs) != available_options(base.obs):
                continue   # 옮긴 값이 가능한 선택지를 바꾸면(모퉁이 앞에서 멈추기를 할 수 없게 되는 등) 사실 둘이 바뀐다 — 받지 않는다
            cand = assemble(cfg, moment, j, id_prefix=id_prefix, t6={"recommendation": keep}, threshold_obs=base.obs, **common)
            p1 = cand["labels"][key]["p_yes"] if cand["meta"]["key_question"] == key else None
            if p1 is not None and (p0 - 0.5) * (p1 - 0.5) < 0:
                variant_rec, variant = cand, moment
                break
    if variant_rec is None:
        variant = build_moment(cfg, base.seed, domain, moment_forced, "wording", split)
        variant_rec = assemble(cfg, variant, j, id_prefix=id_prefix, **common)
    kind = variant.variant
    base_rec["meta"]["pair"] = {"of": ids[1], "kind": kind, "role": "base"}
    variant_rec["meta"]["pair"] = {"of": ids[0], "kind": kind, "role": "variant", "flipped": variant.flipped, "target": key if kind == "numeric" else None}
    return [base_rec, variant_rec]


def generate_batch(n: int, base_seed: int = 0, *, cfg: SimgenConfig | None = None, git: str | None = None, k: int | None = None, id_prefix: str = ID_PREFIX,
                   split=None, seed_start: int | None = None, phrasing: str = DEFAULT_PHRASING, fact_shuffle: float = DEFAULT_FACT_SHUFFLE):  # fmt: skip
    """레코드 n개를 차례로 낸다(생성기). 쌍의 두 레코드는 함께 만들고 서로의 id를 `meta.pair.of`로 가리킨다. 기본 나눔은 학습(train)."""
    from robo_jev.simgen.holdout import split_for

    cfg = cfg or default_config()
    split = split or split_for("train")
    for unit in plan_units(n, base_seed, groups=split.groups, seed_start=seed_start):
        yield from generate_unit(cfg, unit, split=split, git=git, k=k, id_prefix=id_prefix, phrasing=phrasing, fact_shuffle=fact_shuffle)


def write_jsonl(path: Path, records) -> dict:
    """레코드를 JSONL로 쓰고 (개수, 걸린 시간)을 돌려준다."""
    start = time.perf_counter()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps(record) + "\n")
            count += 1
    return {"records": count, "seconds": time.perf_counter() - start}
