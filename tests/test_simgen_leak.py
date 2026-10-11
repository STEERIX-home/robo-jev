"""누출 금지 (Task v0a 지시서 §3.1·§5 '누출', docs/12 §6).

(a) 서술기·질문 생성기·사후 표본·굴리기·정답기 모듈이 세계(World)를 참조하지 않는다.
(b) 같은 관측, 다른 세계(숨은 값만 바꿈) → 모델 입력(`request`)과 정답(`labels`)이 바이트 단위로 같다; `truth`만 달라진다.
(c) 숨은 전용 필드 이름(intent, slosh, true_ 접두 …)이 모델 입력 글에 나오지 않는다.
"""

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest
from simgen_helpers import moment, sample_records

from robo_jev.simgen import generate
from robo_jev.simgen.catalog import default_config
from robo_jev.simgen.record import dumps

SIMGEN = Path(generate.__file__).resolve().parent
OBSERVATION_SIDE = ("narrate.py", "questions.py", "phrases.py", "posterior.py", "rollout.py", "labels.py", "textfmt.py", "observation.py", "spec.py")

#: 세계에만 있는 값의 이름(굴리기 batch 열쇠 가운데 관측에 없는 것)과 지시서가 든 이름.
HIDDEN_NAMES = (
    "intent", "slosh", "true_", "stay_s", "cross_speed", "cross_accel", "back_speed", "back_decel", "stop_decel", "roll_decel",
    "yield_standoff", "agent_decel", "hurry_factor", "hurry_accel", "aside_speed", "keep_standoff", "side_left", "side_right",
    "intent_obs", "lid_prior", "k_ref", "posterior",
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


def _other_world(world):
    """숨은 값만 다른 세계: 의도·움직임 매개변수·출렁임·기울기·뚜껑·가려진 장애물·참 채움을 바꾼다(관측된 값은 그대로)."""
    changes = {}
    v = world.values
    if "intent" in v and v["intent"] >= 0:
        changes["intent"] = float((int(v["intent"]) + 1) % 3)
    for key in ("stay_s", "cross_speed", "back_speed", "yield_standoff", "hurry_factor", "aside_speed", "keep_standoff", "agent_decel", "stop_decel"):
        if key in v:
            changes[key] = v[key] * 1.37 + 0.11
    if "k" in v:
        changes.update(k=1.6 if v["k"] < 1.3 else 1.0, tilt=-v["tilt"] + 0.01, fill=max(0.05, v["fill"] - 0.2))
        if world.at_observation.get("lid_view") == "unknown":
            changes["lid"] = 1.0 - v["lid"]
    for key in ("side_left", "side_right"):
        if key in v and world.at_observation.get("sides", ("", ""))[0 if key == "side_left" else 1] == "occluded":
            changes[key] = 1.0 - v[key]
    at_obs = dict(world.at_observation)
    if "intent_obs" in at_obs:
        at_obs["intent_obs"] = -1  # 관측 시점의 의도도 숨은 값이다
    return replace(world, values={**v, **changes}, at_observation=at_obs)


@pytest.mark.parametrize(
    ("seed", "domain", "forced"),
    [
        (11, "cafe", {"scene": "p1p5_step_in_carrying", "body": "mobile_base"}),
        (12, "warehouse", {"scene": "p1_crossing", "body": "forklift_amr"}),
        (13, "hospital", {"scene": "p1_head_on"}),
        (14, "lab", {"scene": "p5_braking", "body": "arm"}),
        (15, "home", {"scene": "p5_turning"}),
        (16, "sidewalk", {"scene": "p1_step_in", "body": "mobile_base"}),
    ],
)
def test_same_observation_with_a_different_world_gives_identical_request_and_labels(seed, domain, forced):
    cfg = default_config()
    base = moment(seed, domain, **forced)
    other = replace(base, world=_other_world(base.world))
    assert other.world.values != base.world.values
    a = generate.assemble(cfg, base, 0)
    b = generate.assemble(cfg, other, 0)
    assert dumps(a["request"]) == dumps(b["request"])
    assert json.dumps(a["labels"], sort_keys=True) == json.dumps(b["labels"], sort_keys=True)
    assert a["truth"]["world"] != b["truth"]["world"]


def test_hidden_field_names_never_appear_in_the_model_input():
    for record in sample_records()[:600]:
        text = dumps(record["request"]).lower()
        found = [name for name in HIDDEN_NAMES if name in text]
        assert not found, (record["id"], found)
