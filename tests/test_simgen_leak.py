"""누출 금지 (Task v0a 지시서 §3.1·§5 '누출', docs/12 §6).

(a) 서술기·질문 생성기·사후 표본·굴리기·정답기 모듈이 세계(World)를 참조하지 않는다.
(b) **서로 다른 두 World가 `observe()`를 거쳐 같은 관측을 낸다**(숨은 값만 다름 — 의도·움직임 매개변수·출렁임·기울기·보이지 않는 뚜껑·
    가려진 장애물·반응 여부, 그리고 보고 간격이 그대로이게 맞춘 상대 반지름) → 레코드 전체가 바이트 단위로 같다(참 세계도 관측의
    사후에서 뽑으므로 — 리뷰 1 I4). 대조: 관측되는 값 하나를 바꾸면 관측과 글이 달라진다(리뷰 1 I1 — 옛 시험은 같은 Observation
    객체를 넘겨 observe() 안의 누출을 볼 수 없었다).
(c) 숨은 전용 필드 이름(intent, slosh, true_ 접두 …)이 모델 입력 글에 나오지 않는다.
"""

import ast
from dataclasses import replace
from pathlib import Path

import pytest
from simgen_helpers import moment, sample_records

from robo_jev.simgen import generate
from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.record import dumps

SIMGEN = Path(generate.__file__).resolve().parent
OBSERVATION_SIDE = (
    "narrate.py", "narrate_ko.py", "questions.py", "phrases.py", "phrases_ko.py", "phrases_ko_data.py", "state_text.py", "rule_text.py", "korean.py",
    "posterior.py", "rollout.py", "labels.py", "textfmt.py", "observation.py", "spec.py", "p3.py", "p4.py", "p6.py", "voi.py", "holdout.py", "perturb.py",
    "phrasing.py", "phrases_v01.py", "phrases_ko_v01.py",
)  # fmt: skip

#: 세계에만 있는 값의 이름(굴리기 batch 열쇠 가운데 관측에 없는 것)과 지시서가 든 이름.
HIDDEN_NAMES = (
    "intent", "slosh", "true_", "stay_s", "cross_speed", "cross_accel", "back_speed", "back_decel", "stop_decel", "roll_decel",
    "yield_standoff", "agent_decel", "hurry_factor", "hurry_accel", "aside_speed", "keep_standoff", "side_left", "side_right",
    "intent_obs", "lid_prior", "k_ref", "posterior", "com_ratio", "t_leave", "react_standoff", "state_then", "x_height", "x_tilt", "x_depth",
    "x_force", "x_lateral", "leave_rate", "emerge_rate", "c_max", "sigma_check", "hazard_per_s", "follow_delay", "static_u", "reacts2", "standoff2",
    "distracted2", "fail_close", "distracted_prior",
)  # fmt: skip


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


@pytest.mark.parametrize("module", OBSERVATION_SIDE)
def test_observation_side_modules_do_not_import_or_name_the_world(module):
    path = SIMGEN / module
    imports = _imports(path)
    assert not any(name.startswith("robo_jev.simgen.world") or name.endswith(".World") for name in imports), imports
    assert not any(name.startswith("robo_jev.simgen.scenes") for name in imports), imports
    names = {node.id for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.Name)}
    assert "World" not in names


def _hidden_only_world(world):
    """숨은 값만 다른 세계 — 감각 모형이 읽는 값(보고되는 거리·속도·채움·폭 …, 관측 시점의 단서·보임 상태)은 그대로 둔다."""
    v = dict(world.values)
    if "intent" in v and v["intent"] >= 0:
        v["intent"] = float((int(v["intent"]) + 1) % 3)
    for key in ("stay_s", "cross_speed", "back_speed", "yield_standoff", "hurry_factor", "hurry_accel", "aside_speed", "keep_standoff",
                "agent_decel", "stop_decel", "back_decel", "cross_accel", "react_standoff"):  # fmt: skip
        if key in v:
            v[key] = v[key] * 1.37 + 0.11
    if "roll_decel" in v and v["roll_decel"] > 0:
        v["roll_decel"] *= 1.5
    if "reacts" in v:
        v["reacts"] = 1.0 - v["reacts"]
    if "k" in v:
        v.update(k=1.6 if v["k"] < 1.3 else 1.0, tilt=-v["tilt"] + 0.01)
        if world.at_observation.get("lid_view") == "unknown":
            v["lid"] = 1.0 - v["lid"]
    # 새 현상의 숨은 값(Task v0b): 무게중심 높이 비율·치우침·마찰(P3), 쌓을 물건의 무게 치우침(P3 쌓기 — 단서는 관측이라 그대로),
    # 상태·바뀜(P4 — 보고 m_는 관측), 떠남·출현·따라옴·늦은 감지·있음(P6 — 두 센서의 보고는 관측)
    for key, change in (("com", lambda x: x * 0.9), ("off", lambda x: -x), ("mu", lambda x: x + 0.07), ("lam", lambda x: x * 1.3), ("t_leave", lambda x: x + 0.7),
                        ("t_e", lambda x: x + 0.5), ("late", lambda x: x + 0.05), ("t2", lambda x: x + 0.2), ("u2", lambda x: x * 1.1)):  # fmt: skip
        if key in v:
            v[key] = change(v[key])
    for key in ("present", "emerges", "follow", "changed", "distracted", "distracted2", "reacts2"):
        if key in v:
            v[key] = 1.0 - v[key]
    # 멈춘 로봇 앞 회피 실패를 정하는 균등 난수·둘째 상대의 감속·여유(수정 1 I3)도 숨은 값이다
    for key, change in (("static_u", lambda x: (x + 0.37) % 1.0), ("static_u2", lambda x: (x + 0.37) % 1.0), ("decel2", lambda x: x * 1.2),
                        ("standoff2", lambda x: x + 0.1)):  # fmt: skip
        if key in v:
            v[key] = change(v[key])
    if world.scene == "p3_place_stack":
        v["c"] = v["c"] * 0.5 if world.parts.get("cue") else -v["c"]
    if world.scene.startswith("p4"):
        v["state"] = float((int(v["state"]) + 1) % 4)
        for key in [k for k in v if k.startswith("x_")]:
            v[key] += 0.3
    views = world.at_observation.get("sides", ("", ""))
    for i, key in enumerate(("side_left", "side_right")):
        if key in v and views[i] == "occluded":
            v[key] = 1.0 - v[key]
    # 상대의 반지름은 step_in·crossing에서 숨은 값이다 — 보고되는 면 간격이 그대로이게 중심 거리를 함께 옮긴다.
    if world.scene in ("p1_step_in", "p1p5_step_in_carrying") and "ra" in v:
        v["gap"] += 0.04
        v["ra"] += 0.04
    if world.scene == "p1_crossing":
        v["dist"] += 0.04
        v["ra"] += 0.04
    at_obs = dict(world.at_observation)
    if "intent_obs" in at_obs and at_obs["intent_obs"] >= 0:
        at_obs["intent_obs"] = (at_obs["intent_obs"] + 2) % 3  # 관측 시점의 의도도 숨은 값이다(단서는 그대로)
    return replace(world, values=v, at_observation=at_obs)


def _observe(cfg, m, world):
    return m.look(cfg, world, 0)


LEAK_CASES = [
    (11, "cafe", {"scene": "p1p5_step_in_carrying", "body": "mobile_base"}),
    (12, "warehouse", {"scene": "p1_crossing", "body": "forklift_amr"}),
    (13, "hospital", {"scene": "p1_head_on"}),
    (14, "lab", {"scene": "p5_braking", "body": "arm"}),
    (15, "home", {"scene": "p5_turning"}),
    (16, "sidewalk", {"scene": "p1_step_in", "body": "mobile_base"}),
    (17, "transit", {"scene": "p1_crossing", "body": "mobile_base", "variant": "stopped"}),
    (18, "home", {"scene": "p1_step_in", "body": "humanoid", "agent_kind": "ball"}),
    (21, "cafe", {"scene": "p3_carrying_tall"}), (22, "warehouse", {"scene": "p3_place_stack", "body": "forklift_amr"}),
    (23, "lab", {"scene": "p4_place_check", "body": "arm"}), (24, "factory", {"scene": "p4p6_stale_check"}),
    (25, "hospital", {"scene": "p6_stale_track"}), (26, "sidewalk", {"scene": "p6_blind_corner"}), (27, "warehouse", {"scene": "p6_sensor_conflict"}),
    (28, "retail", {"scene": "p1p6_emerging_step_in"}), (29, "cafe", {"scene": "p3p5_carrying_mixed"}), (30, "factory", {"scene": "p3_ramp"}),
    # 한국어 서술의 누출 검사(Task v0b §8)
    (31, "cafe", {"scene": "p1p5_step_in_carrying", "body": "mobile_base", "lang": "ko"}), (32, "home", {"scene": "p5_turning", "lang": "ko"}),
    (33, "lab", {"scene": "p4_insert", "body": "arm", "lang": "ko"}), (34, "hospital", {"scene": "p6_stale_track", "lang": "ko"}),
    (35, "warehouse", {"scene": "p3_place_stack", "body": "humanoid", "lang": "ko"}), (36, "transit", {"scene": "p6_sensor_conflict", "lang": "ko"}),
]


@pytest.mark.parametrize("knobs", [{}, {"phrasing": "v01", "fact_shuffle": 0.5}], ids=["v0b", "v01"])     # 학습 문구 판 v01·사실 순서 섞기도(Task v0.1a)
@pytest.mark.parametrize(("seed", "domain", "forced"), LEAK_CASES)
def test_two_worlds_that_differ_only_in_hidden_values_give_the_same_observation_and_record(seed, domain, forced, knobs):
    cfg = default_config()
    base = moment(seed, domain, **forced)
    other_world = _hidden_only_world(base.world)
    assert other_world.values != base.world.values
    obs_a, obs_b = _observe(cfg, base, base.world), _observe(cfg, base, other_world)
    assert obs_a == base.obs and obs_b == obs_a  # observe()를 거친 두 관측이 같다
    other = replace(base, world=other_world, obs=obs_b)
    a = generate.assemble(cfg, base, 0, **knobs)
    b = generate.assemble(cfg, other, 0, **knobs)
    # 모델 입력·정답·참 결과(사후에서 뽑음)·계산 과정·장부가 모두 같다 — 닻 세계의 설계 여유(meta.margins, margin_norm)만 숨은 값
    # (교차의 비우는 시각은 상대 반지름에 걸린다)에 따라 다를 수 있다.
    for record in (a, b):
        record["meta"].pop("margins")
        record["meta"].pop("margin_norm")
    assert dumps(a) == dumps(b)


def _observed_change(world):
    """관측되는 값 하나를 바꾼 세계(대조군)."""
    v = dict(world.values)
    key = {"p1_step_in": "gap", "p1p5_step_in_carrying": "gap", "p1_crossing": "dist", "p1_head_on": "width", "p5_braking": "obstacle", "p5_turning": "fill"}[world.scene]
    v[key] = v[key] + (-0.12 if key == "fill" else 0.6)
    return replace(world, values=v)


@pytest.mark.parametrize(("seed", "domain", "forced"), LEAK_CASES[:6])
def test_changing_an_observed_value_changes_the_observation_and_the_text(seed, domain, forced):
    cfg = default_config()
    base = moment(seed, domain, **forced)
    changed = _observe(cfg, base, _observed_change(base.world))
    assert changed != base.obs
    a = generate.assemble(cfg, base, 0)
    b = generate.assemble(cfg, replace(base, world=_observed_change(base.world), obs=changed), 0)
    assert a["request"]["state"] != b["request"]["state"]


def test_hidden_field_names_never_appear_in_the_model_input():
    for record in sample_records()[:600]:
        text = dumps(record["request"]).lower()
        found = [name for name in HIDDEN_NAMES if name in text]
        assert not found, (record["id"], found)
