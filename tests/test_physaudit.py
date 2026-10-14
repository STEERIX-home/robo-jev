"""Physics audit (2026-10-14): a hidden centre of mass written into a compiled model at run time (``body_ipos``) moves the body's stored
bounding volumes, which MuJoCo's midphase reads; the opt-in ``refresh_bvh`` keeps them with the centre of mass. Which scene builders
collide a moved body through the midphase at all (only T8b's stack), and that the default labelling is unchanged."""

import mujoco
import numpy as np
import pytest

from robo_jev.mjgen import physics as mph
from robo_jev.mjgen.scenes import SCENES
from robo_jev.mjgen.scenes import build as B
from robo_jev.mjgen.scenes import rigs
from robo_jev.outside import moments as M
from robo_jev.outside import physics as ph

DIMS, MASS, SUPPORT = (0.40, 0.25, 0.20), 3.0, (0.50, 0.35, 0.50)
MASTER_V04 = 20261040


def _compiled_with(com: tuple[float, float, float]) -> mujoco.MjModel:
    """The stack model compiled with the centre of mass in the XML (the reference a runtime write should equal)."""
    nominal = f'<inertial pos="{ph.NOMINAL_IPOS} {ph.NOMINAL_IPOS} {ph.NOMINAL_IPOS}"'
    xml = ph.stack_xml(DIMS, MASS, SUPPORT)
    assert nominal in xml
    model = mujoco.MjModel.from_xml_string(xml.replace(nominal, f'<inertial pos="{com[0]!r} {com[1]!r} {com[2]!r}"'))
    model.geom_friction[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "surface"), 0] = 0.4
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def _arrays(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    out = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:  # noqa: BLE001 - some attributes are not readable as arrays
            continue
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            out[name] = np.array(value)
    return out


def _rest_on_support(scene: ph.Scene, com, steps: int = 600) -> tuple[float, np.ndarray]:
    """Set the hidden values, place the load resting on the support's middle and step; → (lowest centre height, final qpos)."""
    scene.set_hidden(0.4, com)
    q = scene.load_qpos
    scene.data.qpos[q : q + 7] = (0.0, 0.0, SUPPORT[2] + DIMS[2] / 2 + 0.0005, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(scene.model, scene.data)
    lowest = float(scene.data.xpos[scene.load][2])
    for _ in range(steps):
        mujoco.mj_step(scene.model, scene.data)
        lowest = min(lowest, float(scene.data.xpos[scene.load][2]))
    return lowest, np.array(scene.data.qpos)


def test_a_runtime_centre_of_mass_leaves_the_bounding_volumes_behind_and_the_refresh_matches_a_compiled_model():
    com = (0.05, 0.0, 0.012)
    stale = ph.Scene(ph.stack_xml(DIMS, MASS, SUPPORT))
    before = np.array(stale.model.bvh_aabb)
    stale.set_hidden(0.4, com)
    assert np.array_equal(stale.model.bvh_aabb, before)  # mj_setConst does not rebuild them
    fixed = ph.Scene(ph.stack_xml(DIMS, MASS, SUPPORT), refresh_bvh=True)
    fixed.set_hidden(0.4, com)
    reference = _arrays(_compiled_with(com))
    mine = _arrays(fixed.model)
    assert np.abs(mine["bvh_aabb"] - reference["bvh_aabb"]).max() < 1e-12
    assert np.abs(_arrays(stale.model)["bvh_aabb"] - reference["bvh_aabb"]).max() > 0.04
    differing = [name for name in reference if name != "bvh_aabb" and not np.array_equal(mine[name], reference[name])]
    assert differing == []


def test_a_load_with_a_raised_centre_of_mass_sinks_into_the_stack_support_unless_the_bounding_volumes_are_refreshed():
    rest = SUPPORT[2] + DIMS[2] / 2
    lowest, _ = _rest_on_support(ph.Scene(ph.stack_xml(DIMS, MASS, SUPPORT)), (0.0, 0.0, 0.02))
    assert lowest < rest - 0.015  # sinks by about the centre-of-mass height before any contact
    for com in ((0.0, 0.0, 0.02), (0.06, 0.0, 0.03), (0.0, 0.0, -0.02)):
        lowest, qpos = _rest_on_support(ph.Scene(ph.stack_xml(DIMS, MASS, SUPPORT), refresh_bvh=True), com)
        assert lowest > rest - 0.002, com
        reference = _compiled_with(com)
        data = mujoco.MjData(reference)
        data.qpos[:7] = (0.0, 0.0, SUPPORT[2] + DIMS[2] / 2 + 0.0005, 1.0, 0.0, 0.0, 0.0)
        for _ in range(600):
            mujoco.mj_step(reference, data)
        assert np.allclose(qpos, data.qpos, atol=1e-9), com


def test_t8b_stack_reads_a_fully_supported_box_as_falling_unless_the_bounding_volumes_are_refreshed():
    """The stored labeller's defect at the label level (characterisation: revisit if MuJoCo starts rebuilding the volumes): a box set
    squarely on the support, nothing sticking out, with its centre of mass at the top of the stack prior (kh = 0.58 → +1.6 cm)."""
    dims, support = (0.40, 0.25, 0.20), (0.60, 0.35, 0.90)
    outcomes = {}
    for refresh in (False, True):
        scene = ph.Scene(ph.stack_xml(dims, 3.0, support), refresh_bvh=refresh)
        outcomes[refresh] = ph.run_stack(scene, x_centre=0.0, drop=0.005, tilt0_deg=0.0, mu=0.4, com=(0.0, 0.0, 0.016),
                                         support_height=support[2], box_height=dims[2])["outcome"]
    assert outcomes == {False: "fall", True: "stay"}


def test_the_refresh_follows_a_rotated_inertial_frame():
    """``refresh_bvh`` shifts the volumes in the inertial frame (``body_iquat``): checked against a compiled model for a body whose
    principal axes are rotated (full inertia) and that holds two geoms."""
    def xml(ipos: str) -> str:
        return ('<mujoco><worldbody><geom type="plane" size="2 2 0.1"/><body name="b" pos="0 0 0.5"><freejoint/>'
                f'<inertial pos="{ipos}" mass="2" fullinertia="0.03 0.02 0.04 0.004 -0.003 0.002"/>'
                '<geom type="box" size="0.1 0.05 0.04"/><geom type="box" pos="0.15 0 0.02" size="0.03 0.03 0.03"/></body></worldbody></mujoco>')

    model = mujoco.MjModel.from_xml_string(xml("0.001 0.001 0.001"))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    assert not np.allclose(model.body_iquat[body], (1, 0, 0, 0)) and int(model.body_bvhnum[body]) == 3
    reference = ph.bvh_reference(model, body)
    model.body_ipos[body] = (0.04, -0.02, 0.03)
    ph.refresh_bvh(model, body, reference)
    compiled = mujoco.MjModel.from_xml_string(xml("0.04 -0.02 0.03"))
    assert np.allclose(compiled.body_iquat[body], model.body_iquat[body])
    adr = int(model.body_bvhadr[body])
    assert np.abs(model.bvh_aabb[adr : adr + 3] - compiled.bvh_aabb[adr : adr + 3]).max() < 1e-12


def test_the_refresh_does_not_depend_on_the_draws_before_it():
    first = ph.Scene(ph.stack_xml(DIMS, MASS, SUPPORT), refresh_bvh=True)
    first.set_hidden(0.3, (0.08, 0.0, 0.03))
    first.set_hidden(0.5, (-0.02, 0.01, -0.01))
    second = ph.Scene(ph.stack_xml(DIMS, MASS, SUPPORT), refresh_bvh=True)
    second.set_hidden(0.5, (-0.02, 0.01, -0.01))
    assert np.array_equal(first.model.bvh_aabb, second.model.bvh_aabb)


def test_only_the_stack_scene_collides_its_load_through_the_midphase():
    stack = ph.Scene(ph.stack_xml(DIMS, MASS, SUPPORT))
    assert stack.midphase_pairs() == [(stack.load, 0)]  # the world body holds the floor plane and the support box
    for xml in (ph.deck_xml("box", (0.3, 0.3, 0.5), 4.0), ph.push_xml((0.4, 0.4, 0.6), 8.0, 0.3)):
        assert ph.Scene(xml).midphase_pairs() == []


def test_the_default_labels_are_unchanged_and_the_refresh_is_marked():
    deck = M.candidate("deck_brake", 0, 20261012)
    default = M.label_moment(deck, k=2)
    refreshed = M.label_moment(deck, k=2, refresh_bvh=True)
    assert "refresh_bvh" not in default and refreshed.pop("refresh_bvh") is True
    assert refreshed == default  # single-geom pairs never read the bounding volumes


def test_no_v03b_rig_collides_a_box_through_the_midphase_and_the_refresh_changes_nothing_there():
    for orientation in mph.ORIENTATIONS:
        assert mph.tilt_rig((0.3, 0.2, 0.5), 5.0, orientation).midphase_pairs() == []
    assert mph.grip_rig((0.1, 0.1, 0.15), 1.0).midphase_pairs() == []
    boxes = [((0.40, 0.30, 0.25), 6.0), ((0.35, 0.30, 0.20), 4.0), ((0.30, 0.25, 0.20), 3.0)]
    assert mph.tower_rig(boxes[:2]).midphase_pairs() == [] and mph.tower_rig(boxes).midphase_pairs() == []
    results = []
    for refresh in (False, True):
        rig = mph.Rig(mph.tower_xml(boxes), boxes=("box0", "box1", "box2"), dims={f"box{i}": d for i, (d, _) in enumerate(boxes)},
                      refresh_bvh=refresh)
        results.append(mph.run_tower(rig, centres=(0.0, 0.08, 0.17), drop=0.04, mu=0.5, coms=[(0.06, 0.0, 0.0)] * 3))
    assert results[0] == results[1]


@pytest.mark.parametrize("name", sorted(SCENES))
def test_no_v04a_scene_rig_collides_a_box_through_the_midphase(name):
    scene = SCENES[name]
    group = B.candidate(name, 0, MASTER_V04, k=2, pair_share=0.0)
    v = group.visible
    if hasattr(scene, "rig_for"):  # platform and shift scenes
        rig = scene.rig_for(v, {})
    elif name == "table":
        rig = rigs.ledge_rig(v["dims"], v["mass"])
    else:  # squeeze scenes
        rig = rigs.squeeze_rig(name, v["dims"], pad_half=scene.pad_for(v["dims"]), hand_mass=scene.hand_mass)
    assert isinstance(rig, mph.Rig) and rig.midphase_pairs() == []


def test_the_rig_refresh_is_off_by_default():
    assert mph.Rig.refresh_bvh is False
    rig = mph.tower_rig([((0.4, 0.3, 0.25), 6.0)])
    before = np.array(rig.model.bvh_aabb)
    rig.set_hidden(friction={"box0_geom": 0.5}, com={"box0": (0.05, 0.0, 0.0)})
    assert np.array_equal(rig.model.bvh_aabb, before)
