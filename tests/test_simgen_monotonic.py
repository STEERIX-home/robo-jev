"""단조성: 정답이 물리적으로 말이 되는가 (Task v0a 지시서 §5 '단조성', docs/12 §12).

다른 것을 고정하고(같은 관측, 같은 사후 난수) 한 값만 옮긴다:
* gap을 줄이면 `continue`의 닿음 확률이 줄지 않는다.
* 채움을 올리면 쏟아짐 확률이 줄지 않는다.
* 감속을 올리면 쏟아짐 확률이 줄지 않는다(장애물은 멀리 — 넘어섬 충격이 끼지 않게).
"""

from dataclasses import replace

import numpy as np
import pytest
from simgen_helpers import moment

from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.posterior import sample_posterior
from robo_jev.simgen.rollout import rollout
from robo_jev.simgen.spec import Reading, stream

K = 256


def _step_in_obs(gap: float):
    """가운데에 선 어른 하나가 옆으로 천천히 움직이는 끼어듦 — 관측의 gap만 바꾼다."""
    base = moment(21, "cafe", scene="p1_step_in", body="mobile_base", agent_kind="adult").obs
    agent = replace(
        base.agent,
        track=(Reading(0.0, 0.03, 0.01),), track_ages=(0.0,), speed=Reading(0.5, 0.1, 0.05), heading="left",
        cue="glance", ahead=Reading(gap, 0.03, 0.01),
    )  # fmt: skip
    return replace(base, agent=agent)


def _p(obs, event: str, option: str, seed: int = 5) -> float:
    cfg = default_config()
    batch = sample_posterior(cfg, obs, K, stream(seed, "posterior"))
    out = rollout(cfg, obs, batch, (option,))
    return float(out.events[event][0].mean())


def test_contact_probability_under_continue_does_not_drop_when_the_gap_shrinks():
    gaps = np.round(np.arange(0.2, 3.01, 0.2), 2)
    probs = [_p(_step_in_obs(float(g)), "contact", "continue") for g in gaps]
    assert all(later <= earlier for earlier, later in zip(probs, probs[1:], strict=False)), dict(zip(gaps.tolist(), probs, strict=True))
    assert probs[0] > probs[-1]


def _braking_obs(fill: float, decel_moderate: float | None = None):
    base = moment(22, "cafe", scene="p5_braking", body="mobile_base").obs
    container = replace(base.container, fill=Reading(fill, 0.10, 0.05), fill_mode="percent", lid="open")
    env = replace(base.env, obstacle_distance=Reading(6.0, 0.03, 0.01))  # 멀리 — 넘어섬 없음
    robot = base.robot if decel_moderate is None else replace(base.robot, decel_moderate=decel_moderate)
    return replace(base, container=container, env=env, robot=robot)


def test_spill_probability_does_not_drop_when_the_fill_rises():
    fills = np.round(np.arange(0.50, 1.001, 0.05), 2)
    probs = [_p(_braking_obs(float(f)), "spill", "stop_moderate") for f in fills]
    assert all(later >= earlier for earlier, later in zip(probs, probs[1:], strict=False)), dict(zip(fills.tolist(), probs, strict=True))
    assert probs[-1] > probs[0]


def test_spill_probability_does_not_drop_with_stronger_braking_levels():
    obs = _braking_obs(0.85)
    probs = [_p(obs, "spill", option) for option in ("stop_gentle", "stop_moderate", "stop_hard")]
    assert probs[0] <= probs[1] <= probs[2]
    assert probs[2] > probs[0]


@pytest.mark.parametrize("fill", [0.75, 0.9])
def test_spill_probability_does_not_drop_when_the_braking_deceleration_rises(fill):
    decels = np.round(np.arange(1.0, 3.01, 0.25), 2)
    probs = [_p(_braking_obs(fill, float(d)), "spill", "stop_moderate") for d in decels]
    assert all(later >= earlier for earlier, later in zip(probs, probs[1:], strict=False)), dict(zip(decels.tolist(), probs, strict=True))
