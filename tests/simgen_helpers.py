"""simgen 시험 공용 도우미: 표본 레코드(한 프로세스에서 한 번만 만든다)와 손으로 고친 관측.

2,000개 표본은 지시서 §5의 '생성한 2,000개 전부' 검사(행동 choice argmax, 분야 균형, 여유 분포, 핵심 noul 평균)가 함께 쓴다.
"""

from __future__ import annotations

import functools
from dataclasses import replace

from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.generate import build_moment, generate_batch

SAMPLE_N = 2000
SAMPLE_SEED = 0


@functools.lru_cache(maxsize=4)
def sample_records(n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> tuple[dict, ...]:
    return tuple(generate_batch(n, base_seed=seed, cfg=default_config()))


def moment(seed: int, domain: str, **forced):
    """강제한 틀의 판단 순간 하나(Moment)."""
    return build_moment(default_config(), seed, domain, forced)


def with_reading(reading, value: float):
    return replace(reading, value=value)


def numeric_view(obs):
    """관측에서 상황 옷 문구(장소·과제·묘사·단서 문장·용기·물건 문구·장애물·경사로·가림 이름·방해 문구)를 지운 것 — 수치·상태·단위만 남는다."""
    agent = obs.agent
    if agent is not None:
        agent = replace(agent, descriptor=None, cue_phrase=None)
    container = obs.container
    if container is not None:
        container = replace(container, phrase=None, ref=None)
    env = replace(obs.env, place=None, obstacle=None, ramp=None, ramp_end=None, detour=None, occluder=None)
    if env.sides is not None:
        env = replace(env, sides=tuple(replace(s, phrase=None) for s in env.sides))
    texts = {
        "load": dict(phrase=None, ref=None, deck_phrase=None), "stack": dict(phrase=None, ref=None, spot=None),
        "place": dict(item=None, target=None, stale_reason=None), "stale": dict(descriptor=None, ref=None, pronoun=None, spot=None, occluder=None),
        "corner": dict(phrase=None), "conflict": dict(phrase=None),
    }  # fmt: skip
    parts = {name: replace(getattr(obs, name), **blank) for name, blank in texts.items() if getattr(obs, name) is not None}
    return replace(obs, agent=agent, container=container, env=env, skin=None, **parts)


def skin_strings(obs) -> list[str]:
    """글에 들어가는 상황 옷 문구 전부(긴 것부터)."""
    out = [obs.skin.setting, obs.skin.body_name, obs.skin.task, obs.skin.carry_where, *obs.skin.distractors, obs.env.place, obs.env.obstacle or ""]
    if obs.agent is not None:
        out += [obs.agent.descriptor.np, obs.agent.descriptor.ref, obs.agent.cue_phrase or ""]
    if obs.container is not None:
        out.append(obs.container.phrase)
    if obs.env.sides is not None:
        out += [s.phrase or "" for s in obs.env.sides]
    out += [obs.env.ramp or "", obs.env.ramp_end or "", obs.env.detour or "", obs.env.occluder or ""]
    if obs.load is not None:
        out += [obs.load.phrase, obs.load.ref, obs.load.deck_phrase]
    if obs.stack is not None:
        out += [obs.stack.phrase, obs.stack.ref, obs.stack.spot]
    if obs.place is not None:
        out += [obs.place.item, obs.place.target, obs.place.stale_reason or ""]
    if obs.stale is not None:
        out += [obs.stale.descriptor.np, obs.stale.descriptor.ref, obs.stale.spot, obs.stale.occluder]
    if obs.corner is not None:
        out.append(obs.corner.phrase)
    if obs.conflict is not None:
        out.append(obs.conflict.phrase)
    return sorted((t for t in out if t), key=len, reverse=True)
