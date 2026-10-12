"""T8b physics sanity (Task T8 §1-2): MuJoCo rollouts — hidden centre of mass in force, no creep, analytic thresholds, monotonicity,
determinism, box-only loads."""

import math

import mujoco
import numpy as np
import pytest

from robo_jev.outside import physics as ph
from robo_jev.outside import sanity
from robo_jev.outside.moments import _com, candidate, hidden_draws, label_moment

G = ph.GRAVITY


def test_hidden_centre_of_mass_is_in_force():
    """MuJoCo ignores runtime body_ipos for 'same frame' bodies — the load compiles with an offset, so the hidden COM applies."""
    scene = ph.Scene(ph.stack_xml((0.40, 0.25, 0.20), 3.0, (0.60, 0.40, 0.50)))
    for com in ((-0.08, 0.0, 0.02), (0.05, 0.01, -0.03)):
        scene.set_hidden(0.4, com)
        mujoco.mj_forward(scene.model, scene.data)
        assert np.allclose(scene.com_offset(), com, atol=1e-12)


def test_cylinder_loads_are_refused():
    xml = ph.deck_xml("box", (0.2, 0.2, 0.4), 2.0).replace('type="box" size="0.100000 0.100000 0.200000"', 'type="cylinder" size="0.100000 0.200000"')
    with pytest.raises(ValueError):
        ph.Scene(xml)


def test_no_creep_below_the_friction_limit():
    scene = ph.Scene(ph.deck_xml("box", (0.40, 0.40, 0.15), 3.0))
    result = ph.run_deck(scene, ph.brake_profile(2.0 * 0.8 * 0.3 * G, 0.8 * 0.3 * G, post=0.5), mu=0.3, com=(0.0, 0.0, 0.0))
    assert result["outcome"] == "stay" and result["slide_m"] < 0.002


def test_deck_follows_the_planned_trajectory():
    scene = ph.Scene(ph.deck_xml("box", (0.30, 0.30, 0.30), 4.0))
    profile = ph.turn_profile(1.5, 1.0, direction=-1)
    ph.run_deck(scene, profile, mu=0.8, com=(0.0, 0.0, 0.0))
    assert abs(profile.q[-1, 2] + math.pi / 2) < 0.01  # a right 90° curve (step rounding of the ramps only)
    assert abs(scene.data.qpos[2] - profile.q[-1, 2]) < 1e-3  # the deck tracks the plan
    assert np.allclose(scene.data.qpos[0:2], profile.q[-1, 0:2], atol=2e-3)


def test_tipping_onset_matches_the_analytic_threshold():
    entry = sanity.analytic_tip(0.20, 0.60, 0.5, mode="onset")
    assert 0.97 <= entry["ratio"] <= 1.03


def test_sliding_onset_matches_mu_g():
    entry = sanity.analytic_slide(0.3)
    assert 0.97 <= entry["ratio"] <= 1.03


def test_push_tipping_height_matches_the_analytic_height():
    entry = sanity.analytic_push(0.40, 0.80, 0.5)
    assert 0.9 <= entry["ratio"] <= 1.1


def test_release_falls_exactly_when_the_centre_of_mass_passes_the_edge():
    scene = ph.Scene(ph.stack_xml((0.40, 0.25, 0.20), 3.0, (0.60, 0.40, 0.50)))
    for offset, expected in ((-0.01, "stay"), (0.01, "fall")):
        result = ph.run_stack(scene, x_centre=0.30, drop=0.0, tilt0_deg=0.0, mu=0.4, com=(offset, 0.0, 0.0), support_height=0.5, box_height=0.2)
        assert result["outcome"] == expected


def test_harder_braking_never_makes_tipping_less_likely():
    """Common draws, decelerations swept upward: no draw tips at a gentler stop and then stays up at a harder one."""
    moment = candidate("deck_brake", 2, 20261012)  # aimed at tipping
    v = moment.visible
    draws = hidden_draws(moment.hidden, 6, moment.seed ^ 0x5EED)
    scene = ph.Scene(ph.deck_xml(v["shape"], v["dims"], v["mass"]))
    previous = [False] * len(draws)
    for factor in (0.5, 0.8, 1.0, 1.4, 2.0):
        profile = ph.brake_profile(v["speed"], v["decel"] * factor)
        tipped = [ph.run_deck(scene, profile, mu=d["mu"], com=_com("deck_brake", v, d))["outcome"] == "tip" for d in draws]
        assert all(t or not p for p, t in zip(previous, tipped))
        previous = tipped
    assert any(previous)


def test_labels_are_deterministic():
    moment = candidate("push", 1, 20261012)
    assert sanity.label_hash(label_moment(moment, k=4)) == sanity.label_hash(label_moment(moment, k=4))
