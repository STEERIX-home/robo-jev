"""D0 fixture 검사.

fixture는 `tests/fixtures/build_d0.py`가 만든다. 여기서는 (1) 빌더가 결정적인지,
(2) 모든 레코드가 계약을 지키는지, (3) 계획서가 요구한 범위가 실제로 들어 있는지를
본다. 사람 검수는 별개이며 manifest의 `reviewed_by`로 관리한다.
"""

import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from robo_jev.contracts import NON_INPUT_FIELDS, QUESTION_SET_V0, model_input, validate_record

FIXTURES = Path(__file__).parent / "fixtures"
D0 = FIXTURES / "d0.jsonl"
D0_STREAMS = FIXTURES / "d0_streams.jsonl"
D0_MANIFEST = FIXTURES / "d0_manifest.json"


@functools.lru_cache(maxsize=1)
def _load_builder():
    """빌더는 패키지가 아니라 스크립트라서 경로로 불러온다."""
    spec = importlib.util.spec_from_file_location("build_d0", FIXTURES / "build_d0.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass가 자기 모듈을 찾을 수 있어야 한다
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(D0_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def singles() -> list[dict]:
    return _read_jsonl(D0)


@pytest.fixture(scope="module")
def streams() -> list[dict]:
    return _read_jsonl(D0_STREAMS)


# --------------------------------------------------------------------------
# 결정성과 manifest
# --------------------------------------------------------------------------


def test_builder_is_deterministic(tmp_path):
    """다시 만들면 세 파일 모두 같은 바이트가 나온다."""
    builder = _load_builder()
    rebuilt_manifest = builder.build_all(tmp_path)

    for name in ("d0.jsonl", "d0_streams.jsonl", "d0_manifest.json"):
        assert (tmp_path / name).read_bytes() == (FIXTURES / name).read_bytes(), name

    committed = json.loads(D0_MANIFEST.read_text(encoding="utf-8"))
    for name, info in committed["files"].items():
        assert rebuilt_manifest["files"][name]["sha256"] == info["sha256"], name


def test_manifest_hashes_match_the_committed_files(manifest):
    for name, info in manifest["files"].items():
        data = (FIXTURES / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"], name
        assert len(data) == info["bytes"], name


def test_manifest_records_review_is_pending(manifest):
    assert manifest["reviewed_by"] == []
    assert manifest["builder"] == "tests/fixtures/build_d0.py"
    assert manifest["files"]["d0.jsonl"]["records"] == 64
    assert manifest["files"]["d0_streams.jsonl"]["records"] == 4


def test_builder_question_ids_match_the_contract():
    builder = _load_builder()
    assert list(builder.QUESTION_IDS_V0) == list(QUESTION_SET_V0)


# --------------------------------------------------------------------------
# 모든 레코드가 계약을 지킨다
# --------------------------------------------------------------------------


def test_every_single_request_record_is_valid(singles):
    assert len(singles) == 64
    for index, record in enumerate(singles):
        try:
            validate_record(record)
        except ValueError as error:  # pragma: no cover - 실패할 때만 실행
            pytest.fail(f"d0.jsonl line {index}: {error}")


def test_every_stream_record_is_valid(streams):
    assert len(streams) == 4
    for index, record in enumerate(streams):
        try:
            validate_record(record)
        except ValueError as error:  # pragma: no cover - 실패할 때만 실행
            pytest.fail(f"d0_streams.jsonl line {index}: {error}")


def _all_keys(node) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            keys.add(key)
            keys |= _all_keys(value)
    elif isinstance(node, list):
        for value in node:
            keys |= _all_keys(value)
    return keys


def test_model_input_of_every_record_excludes_non_input_fields(singles, streams):
    for record in singles + streams:
        assert _all_keys(model_input(record)).isdisjoint(NON_INPUT_FIELDS)


def test_first_line_is_choice_with_valid_set(singles):
    """계획서의 검사가 0번 줄을 choice + valid_set으로 가정한다."""
    record = singles[0]
    assert record["request"]["questions"][0]["type"] == "choice"
    assert record["labels"][0]["kind"] == "valid_set"
    assert record["labels"][0]["candidate_ids"]


# --------------------------------------------------------------------------
# 단일 요청 64건의 범위
# --------------------------------------------------------------------------


def test_single_requests_cover_all_question_types(singles):
    types = {question["type"] for record in singles for question in record["request"]["questions"]}
    assert types == {"choice", "boolean", "ordinal"}


def test_single_requests_cover_all_label_kinds(singles):
    kinds = {label["kind"] for record in singles for label in record["labels"]}
    assert kinds == {"valid_set", "single", "distribution", "event"}


def test_single_requests_include_multiple_valid_answers(singles):
    multiple = [
        label
        for record in singles
        for label in record["labels"]
        if len(label.get("candidate_ids", [])) >= 2
    ]
    assert len(multiple) >= 3


def test_single_requests_include_a_not_applicable_answer(singles):
    """정보 부족·해당 없음 후보가 정답인 건이 있어야 한다."""
    builder = _load_builder()
    not_applicable = [
        record
        for record in singles
        for label in record["labels"]
        if label.get("candidate_ids") == [builder.UNKNOWN_CANDIDATE_ID]
    ]
    assert len(not_applicable) >= 2


def test_single_requests_include_missing_labels(singles):
    missing = 0
    masked_out = 0
    for record in singles:
        labeled = {label["question_id"] for label in record["labels"]}
        missing += sum(1 for q in record["request"]["questions"] if q["id"] not in labeled)
        masked_out += sum(1 for label in record["labels"] if label.get("mask") is False)
    assert missing >= 6
    assert masked_out >= 6


def test_single_requests_include_korean_and_english_instructions(singles):
    instructions = [
        question["instructions"] for record in singles for question in record["request"]["questions"]
    ]
    assert any(any("가" <= ch <= "힣" for ch in text) for text in instructions)
    assert any(text.isascii() for text in instructions)


# --------------------------------------------------------------------------
# 스트림 4건의 범위
# --------------------------------------------------------------------------


def test_streams_have_about_400_ticks(streams):
    total = sum(len(record["ticks"]) for record in streams)
    assert total == 400
    for record in streams:
        assert len(record["ticks"]) == 100
        # 10Hz: 틱 간격은 100ms다.
        assert [tick["sim_ms"] for tick in record["ticks"]] == [t * 100 for t in range(100)]


def test_stream_main_candidates_are_joint_actions(streams):
    keys = {
        candidate["key"]
        for record in streams
        for tick in record["ticks"]
        for candidate in tick["request"]["candidates"]["q_main"]
    }
    assert any(key.startswith("grasp:") and key.count(":") == 4 for key in keys)
    assert any(key.startswith("place:") for key in keys)
    assert any(key.startswith("push:") for key in keys)
    assert {"observe", "hold", "replan"} <= keys


def test_stream_keyframes_carry_admissibility_and_rollouts(streams):
    keyframe_labels = [
        label
        for record in streams
        for tick in record["ticks"]
        for label in tick["labels"]
        if label["question_id"] == "q_main" and "event_results" in label
    ]
    assert len(keyframe_labels) >= 4 * 5
    assert all("semantic_admissible" in label for label in keyframe_labels)
    assert any("unknown" in label for label in keyframe_labels)
    assert all(label["label_confidence"] == "high" for label in keyframe_labels)


def test_stream_gripper_transitions_allow_two_states(streams):
    two_state = 0
    single_answer = 0
    for record in streams:
        transitions = record["provenance"]["marks"]["gripper_transition_ticks"]
        window = {t for switch in transitions for t in (switch - 1, switch, switch + 1)}
        for tick in record["ticks"]:
            for label in tick["labels"]:
                if label["question_id"] != "q_gripper":
                    continue
                if label["kind"] == "valid_set":
                    assert set(label["candidate_ids"]) == {"open", "closed"}
                    assert tick["t"] in window
                    two_state += 1
                else:
                    assert label["kind"] == "single"
                    assert tick["t"] not in window
                    single_answer += 1
    assert two_state >= 4
    assert single_answer >= 100


def test_stream_has_stop_true_ticks(streams):
    stop_true = [
        tick
        for record in streams
        for tick in record["ticks"]
        for label in tick["labels"]
        if label["question_id"] == "q_stop" and label["kind"] == "single" and label["answer"] is True
    ]
    assert len(stop_true) >= 3


def test_stream_aux_labels_are_conditioned_on_the_commitment(streams):
    seen = 0
    for record in streams:
        for tick in record["ticks"]:
            commitment = tick["request"]["commitment"]
            for label in tick["labels"]:
                if label["question_id"] not in {"q_gripper", "q_path", "q_speed", "q_force"}:
                    continue
                assert commitment is not None
                assert label["conditioned_on"] == f"{commitment['action_ref']}/{commitment['phase']}"
                seen += 1
    assert seen >= 4 * 4 * 80


def test_stream_has_an_instruction_change_mid_episode(streams):
    changed = [record for record in streams if len(record["prefix"]["instructions"]) >= 2]
    assert changed
    for record in changed:
        versions = [instruction["version"] for instruction in record["prefix"]["instructions"]]
        assert versions == sorted(set(versions))
        assert record["prefix"]["instructions"][1]["t_ms"] > 0


def test_stream_has_a_commitment_contrast_pair(streams):
    """같은 관측, 다른 commitment, 다른 q_main 라벨인 인접 틱 쌍이 있어야 한다."""
    pairs = []
    for record in streams:
        marks = record["provenance"]["marks"]["contrast_pair"]
        if not marks:
            continue
        first, second = (record["ticks"][t] for t in marks)
        assert first["request"]["state"] == second["request"]["state"]
        assert first["request"]["commitment"] != second["request"]["commitment"]
        first_main = next(l for l in first["labels"] if l["question_id"] == "q_main")
        second_main = next(l for l in second["labels"] if l["question_id"] == "q_main")
        assert first_main["candidate_ids"] != second_main["candidate_ids"]
        pairs.append(marks)
    assert pairs


def test_stream_object_poses_stay_on_the_workspace(streams):
    """장난감 시나리오라도 물체가 작업 공간을 벗어나거나 공중에 남으면 안 된다."""
    for record in streams:
        for tick in record["ticks"]:
            for obj in tick["request"]["state"]["objects"]:
                x, y, z = obj["pose_mm"]
                assert all(isinstance(axis, int) for axis in (x, y, z)), (record["episode_id"], obj)
                assert -700 <= x <= 700 and -500 <= y <= 500, (record["episode_id"], tick["t"], obj)
                carried = tick["request"]["state"]["robot"]["holding"] == obj["id"]
                if carried:
                    assert 742 <= z <= 1000, (record["episode_id"], tick["t"], obj)
                else:
                    assert z == 742, (record["episode_id"], tick["t"], obj)


def test_stream_target_object_reaches_the_goal_zone(streams):
    """각 에피소드의 대상 물체가 실제로 움직여 목표 영역에 도달해야 한다."""
    for record in streams:
        marks = record["provenance"]["marks"]
        first_state = record["ticks"][0]["request"]["state"]
        last_state = record["ticks"][-1]["request"]["state"]
        start = next(o for o in first_state["objects"] if o["id"] == marks["target"])["pose_mm"]
        end = next(o for o in last_state["objects"] if o["id"] == marks["target"])["pose_mm"]
        travelled = sum((a - b) ** 2 for a, b in zip(start, end)) ** 0.5
        assert travelled >= 300, (record["episode_id"], start, end)

        zone = next(z for z in last_state["zones"] if z["id"] == marks["zone"])
        x_min, _, x_max, _ = zone["bounds_mm"]
        assert x_min <= end[0] <= x_max, (record["episode_id"], end, zone)


def test_stream_goal_version_follows_the_instruction_change(streams):
    for record in streams:
        versions = {tick["request"]["state"]["goal"]["version"] for tick in record["ticks"]}
        declared = {instruction["version"] for instruction in record["prefix"]["instructions"]}
        assert versions == declared, record["episode_id"]


def test_stream_has_an_adopted_switch(streams):
    switches = [
        tick
        for record in streams
        for tick in record["ticks"]
        if tick["adopted"].get("switch") is True
    ]
    assert switches


def test_stream_retry_labels_are_masked_before_any_failure(streams):
    """실패 이력이 없는 틱에는 q_retry 라벨이 없다 (loss mask)."""
    episode = next(record for record in streams if record["episode_id"] == "ep-d0-002")
    labeled_ticks = [
        tick["t"]
        for tick in episode["ticks"]
        if any(label["question_id"] == "q_retry" for label in tick["labels"])
    ]
    assert labeled_ticks
    assert min(labeled_ticks) == 40
    assert all(
        not any(label["question_id"] == "q_retry" for label in tick["labels"])
        for tick in episode["ticks"][:40]
    )
