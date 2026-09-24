"""그리퍼 라벨 규칙 v2·파생 데이터셋·틱 종류·이른 답 실험 정책 검사 (Task R5 Stage A, docs/08 §7 `q_gripper`)."""

import copy
import json

import pytest

from robo_jev.contracts import validate_record
from robo_jev.data.dagger import DAGGER_VERSION, rule_judge_policy
from robo_jev.data.gripper_labels import (
    DEFAULT_EARLY_DIRECTIONS,
    GRIPPER_LABEL_CLASSES,
    GRIPPER_LABELS_VERSION,
    EarlyGripperPolicy,
    build_dagger_dataset,
    count_gripper_classes,
    derive_gripper_v2_dataset,
    desired_from_evidence,
    early_tolerance_indices,
    gripper_tick_class,
    gripper_transitions,
    reference_gripper_schedule,
    reset_gripper_labels,
)
from robo_jev.data.robot_episodes import (
    GENERATOR_VERSION,
    _tolerate_gripper_transitions,
    _tolerate_gripper_transitions_v2,
    build_manifest,
    generate_episode,
    gripper_label_rule,
    load_generator_config,
    read_episodes,
    write_episode,
)
from robo_jev.data.validate import validate_dataset
from robo_jev.sim.expert import Expert

CONFIG = load_generator_config()


# --------------------------------------------------------------------------
# 규칙 v2 — 순수 함수
# --------------------------------------------------------------------------


def test_transitions_are_the_first_tick_whose_desired_state_differs_from_the_previous_labelled_tick():
    assert gripper_transitions(["open", "open", "closed", "closed", "open"]) == [2, 4]
    assert gripper_transitions(["open", "open", None, "closed", "closed"]) == [3]  # 라벨 없는 틱을 건너뛰어 이웃을 본다
    assert gripper_transitions(["closed", "closed"]) == [] and gripper_transitions([None, "open"]) == []
    assert gripper_transitions(["open", "closed", "open", "closed"]) == [1, 2, 3]


def test_v2_widens_only_the_k_ticks_before_a_close_transition_and_never_the_transition_itself():
    desired = ["open", "open", "open", "closed", "closed", "closed", "open", "open"]
    assert DEFAULT_EARLY_DIRECTIONS == ("closed",)  # 놓기 전환(closed→open) 앞은 넓히지 않는다 — 이른 open은 운반 높이에서 실행된다 (A2)
    assert early_tolerance_indices(desired, 0) == set()
    assert early_tolerance_indices(desired, 1) == {2}
    assert early_tolerance_indices(desired, 2) == {1, 2}
    assert early_tolerance_indices(desired, 10) == {0, 1, 2}  # 첫 전환 앞은 편 시작에서 멈춘다
    both = ("closed", "open")
    assert early_tolerance_indices(desired, 1, directions=both) == {2, 5}
    assert early_tolerance_indices(desired, 2, directions=both) == {1, 2, 4, 5}
    # 전환 틱과 그 뒤(정착)는 어떤 k·방향에서도 한 값이다
    for k in (1, 2, 10):
        assert not early_tolerance_indices(desired, k, directions=both) & {3, 6, 7}


def test_v2_stops_at_another_transition_a_gap_or_a_different_state():
    both = ("closed", "open")
    assert early_tolerance_indices(["open", "closed", "open", "closed"], 2, directions=both) == {0}  # 1·2는 전환 틱 자체
    assert early_tolerance_indices(["open", "closed", "open", "closed"], 2) == {0}
    assert early_tolerance_indices(["open", "open", None, "closed", "closed"], 2) == set()  # 라벨 없는 틱에서 멈춘다
    assert early_tolerance_indices(["open"] * 5, 3) == set()  # 전환이 없으면 아무것도 넓히지 않는다
    with pytest.raises(ValueError, match="early_ticks"):
        early_tolerance_indices(["open", "closed"], -1)
    with pytest.raises(ValueError, match="directions"):
        early_tolerance_indices(["open", "closed"], 1, directions=("shut",))


def _labelled_record(desired: list, *, executed: list | None = None, reasons: list | None = None) -> dict:
    """`q_gripper` 라벨(한 값)과 실행 그리퍼 상태를 든 최소 레코드 — 라벨이 None인 틱은 라벨을 적지 않는다."""
    ticks = []
    evidence = []
    for index, value in enumerate(desired):
        reason = (reasons or [None] * len(desired))[index] or ("at_grasp_point" if value == "closed" else "phase:approach")
        labels = [] if value is None else [
            {"question_id": "q_gripper", "kind": "valid_set", "candidate_ids": [value], "rule": f"phase-profile-v0/{reason}", "source": "expert_v0", "conditioned_on": "c1/grasp"},
        ]
        labels.append({"question_id": "q_main", "kind": "valid_set", "candidate_ids": ["c1"], "source": "expert_v0"})
        exec_gripper = (executed or ["open"] * len(desired))[index]
        ticks.append({"t": index * 5, "request": {"state": {"exec": {"gripper": exec_gripper}}, "candidates": {"q_main": [{"id": "c1", "key": "grasp:o1:top:zoneL"}]}}, "labels": labels})
        evidence.append({"t": index * 5, "aux": None if value is None else {"gripper": {"desired": value, "reason": reason}}})
    return {"episode_id": "ep-E0-1", "ticks": ticks, "evidence": {"expert": {"ticks": evidence}}}


def _gripper_ids(record: dict) -> list:
    out = []
    for tick in record["ticks"]:
        label = next((item for item in tick["labels"] if item["question_id"] == "q_gripper"), None)
        out.append(None if label is None else list(label["candidate_ids"]))
    return out


def test_the_record_level_v2_rule_marks_the_widened_ticks_and_refuses_an_already_widened_record():
    record = _labelled_record(["open", "open", "closed", "closed", "open"])
    changed = _tolerate_gripper_transitions_v2(record, 1)
    assert changed == [1]  # open→closed 전환(2) 앞만; closed→open 전환(4) 앞의 3은 그대로 한 값 `closed`
    assert _gripper_ids(record) == [["open"], ["open", "closed"], ["closed"], ["closed"], ["open"]]
    widened = next(item for item in record["ticks"][1]["labels"] if item["question_id"] == "q_gripper")
    assert widened["rule"].endswith("+early-tolerance") and widened["tolerance"] == {"version": GRIPPER_LABELS_VERSION, "early_ticks": 1, "directions": ["closed"]}
    both = _labelled_record(["open", "open", "closed", "closed", "open"])
    assert _tolerate_gripper_transitions_v2(both, 1, ("closed", "open")) == [1, 3]
    untouched = next(item for item in record["ticks"][2]["labels"] if item["question_id"] == "q_gripper")
    assert untouched["rule"] == "phase-profile-v0/at_grasp_point" and "tolerance" not in untouched
    with pytest.raises(ValueError, match="두 값"):
        _tolerate_gripper_transitions_v2(record, 1)
    assert _tolerate_gripper_transitions_v2(_labelled_record(["open", "closed", "closed"]), 0) == []


def test_the_old_rule_makes_the_transition_tick_itself_two_valued_and_v2_does_not():
    """R4 리뷰 1의 기전: 옛 규칙(±1)은 전환 틱 t*를 언제나 두 값으로 만든다 — v2에서 t*는 한 값 `closed`다."""
    old = _labelled_record(["open", "open", "closed", "closed", "closed"])
    _tolerate_gripper_transitions(old, 1)
    assert _gripper_ids(old) == [["open"], ["open", "closed"], ["open", "closed"], ["closed"], ["closed"]]  # t*−1과 t* 자체가 두 값
    new = _labelled_record(["open", "open", "closed", "closed", "closed"])
    _tolerate_gripper_transitions_v2(new, 1)
    assert _gripper_ids(new) == [["open"], ["open", "closed"], ["closed"], ["closed"], ["closed"]]


def test_reset_restores_the_single_valued_desired_state_and_the_rule_from_the_evidence():
    record = _labelled_record(["open", "open", "closed", "closed"])
    _tolerate_gripper_transitions(record, 1)
    assert desired_from_evidence(record) == ["open", "open", "closed", "closed"]
    changed = reset_gripper_labels(record)
    assert changed == [1, 2]  # 옛 규칙이 두 값으로 만든 틱 둘(t*−1과 t*)이 되돌아왔다
    assert _gripper_ids(record) == [["open"], ["open"], ["closed"], ["closed"]]
    rules = [next(item for item in tick["labels"] if item["question_id"] == "q_gripper")["rule"] for tick in record["ticks"]]
    assert rules == ["phase-profile-v0/phase:approach"] * 2 + ["phase-profile-v0/at_grasp_point"] * 2
    assert reset_gripper_labels(record) == []  # 이미 한 값이면 아무것도 바꾸지 않는다
    # 증거와 라벨의 틱 수가 다르면 거절한다
    broken = copy.deepcopy(record)
    broken["evidence"]["expert"]["ticks"].pop()
    with pytest.raises(ValueError, match="evidence"):
        reset_gripper_labels(broken)


# --------------------------------------------------------------------------
# 틱 종류 (R4 C0 표의 분류)
# --------------------------------------------------------------------------


def test_gripper_tick_classes_follow_the_label_and_the_executed_state():
    record = _labelled_record(["open", "open", "closed", "closed", "closed", None], executed=["open", "open", "open", "open", "closed", "closed"])
    _tolerate_gripper_transitions_v2(record, 1)
    classes = [gripper_tick_class(tick) for tick in record["ticks"]]
    assert classes == ["open", "window", "initiate", "initiate", "settled", None]
    assert GRIPPER_LABEL_CLASSES == ("initiate", "window", "settled", "open", "window_closed")
    old = _labelled_record(["open", "open", "closed", "closed", "closed"], executed=["open", "open", "closed", "closed", "closed"])
    _tolerate_gripper_transitions(old, 1)
    assert [gripper_tick_class(tick) for tick in old["ticks"]] == ["open", "window", "window_closed", "settled", "settled"]
    counts = count_gripper_classes([record, old])
    assert counts["classes"] == {"initiate": 2, "window": 2, "settled": 3, "open": 2, "window_closed": 1}
    assert counts["labelled_ticks"] == 10 and counts["executed_closes"] == 2 and counts["label_transitions"] == 2
    assert counts["initiate_per_executed_close"] == 1.0


# --------------------------------------------------------------------------
# 생성기의 기본 규칙과 버전
# --------------------------------------------------------------------------


def test_the_generator_picks_the_rule_from_the_expert_label_config_and_v2_is_the_default():
    assert GENERATOR_VERSION == "gen-robot-v0.3"
    assert gripper_label_rule({}) == ("v2", 0)
    assert gripper_label_rule({"gripper_label_rule": "v2", "gripper_early_ticks": 2}) == ("v2", 2)
    assert gripper_label_rule({"gripper_label_rule": "v1", "gripper_transition_tolerance_ticks": 1}) == ("v1", 1)
    with pytest.raises(ValueError, match="gripper_label_rule"):
        gripper_label_rule({"gripper_label_rule": "v3"})
    expert = Expert()
    rule, early = gripper_label_rule(expert.label_config)
    assert rule == "v2" and early >= 0  # 설정 파일의 값 — 실측(A2)이 정한 k


@pytest.fixture(scope="module")
def parent_batch(tmp_path_factory):
    """옛 규칙(v1, ±1)으로 라벨한 짧은 부모 배치 둘 — R1 v0.2의 라벨 꼴을 재현한다."""
    expert = Expert()
    expert.label_config = {**expert.label_config, "gripper_label_rule": "v1", "gripper_transition_tolerance_ticks": 1}
    out = tmp_path_factory.mktemp("parent")
    records = []
    for profile, seed in (("E0", 17), ("E0", 5)):
        record = generate_episode(profile, seed, policy=expert, expert=expert, config=CONFIG, max_ticks=40)
        write_episode(record, out)
        records.append(record)
    build_manifest(out, CONFIG, batch_wall_s=1.0)
    return {"out": out, "records": records}


def test_a_generated_episode_under_the_default_rule_keeps_the_close_transition_single_valued():
    expert = Expert()
    record = generate_episode("E0", 17, policy=expert, expert=expert, config=CONFIG, max_ticks=40)
    desired = desired_from_evidence(record)
    transitions = gripper_transitions(desired)
    assert transitions, "E0 에피소드에는 닫기 전환이 있어야 한다"
    ids = _gripper_ids(record)
    for at in transitions:
        assert ids[at] == [desired[at]]  # 전환 틱은 한 값
    _, early = gripper_label_rule(expert.label_config)
    widened = [index for index, value in enumerate(ids) if value == ["open", "closed"]]
    assert widened == sorted(early_tolerance_indices(desired, early))
    assert record["versions"]["generator"] == GENERATOR_VERSION and record["provenance"]["gripper_label_rule"] == {"rule": "v2", "early_ticks": early}


# --------------------------------------------------------------------------
# A3 — 부모에서 라벨만 바꾼 파생 데이터셋
# --------------------------------------------------------------------------


def test_the_derived_dataset_changes_only_the_gripper_labels_and_keeps_the_lineage(parent_batch, tmp_path):
    out = tmp_path / "g2"
    manifest = derive_gripper_v2_dataset(parent_batch["out"], out, early_ticks=1)
    children = {record["episode_id"]: record for _, record in read_episodes(out)}
    assert set(children) == {record["episode_id"] for record in parent_batch["records"]}
    for parent in parent_batch["records"]:
        child = children[parent["episode_id"]]
        assert child["split"] == parent["split"] and child["versions"]["labels"] == f"{parent['versions'].get('labels', 'expert')}+{GRIPPER_LABELS_VERSION}"
        assert child["provenance"]["lineage"]["parent_dataset"] == str(parent_batch["out"])
        assert child["provenance"]["lineage"]["gripper_labels"] == {"version": GRIPPER_LABELS_VERSION, "rule": "v2", "early_ticks": 1}
        assert {k: v for k, v in child["provenance"].items() if k != "lineage"} == {k: v for k, v in parent["provenance"].items() if k != "lineage"}
        desired = desired_from_evidence(parent)
        expected_widened = early_tolerance_indices(desired, 1)
        for index, (old_tick, new_tick) in enumerate(zip(parent["ticks"], child["ticks"])):
            assert {k: v for k, v in old_tick.items() if k != "labels"} == {k: v for k, v in new_tick.items() if k != "labels"}
            assert [l for l in old_tick["labels"] if l["question_id"] != "q_gripper"] == [l for l in new_tick["labels"] if l["question_id"] != "q_gripper"]
            new_label = next((l for l in new_tick["labels"] if l["question_id"] == "q_gripper"), None)
            if desired[index] is None:
                assert new_label is None
            elif index in expected_widened:
                assert new_label["candidate_ids"] == ["open", "closed"]
            else:
                assert new_label["candidate_ids"] == [desired[index]]
        validate_record(child)
    assert manifest["lineage"]["labels_version"] == f"expert+{GRIPPER_LABELS_VERSION}"  # 부모 레코드에 labels 버전이 없으면 `expert`
    assert manifest["lineage"]["gripper_labels"] == {"version": GRIPPER_LABELS_VERSION, "rule": "v2", "early_ticks": 1}
    for parent in parent_batch["records"]:
        child = children[parent["episode_id"]]
        assert {k: v for k, v in child["versions"].items() if k != "labels"} == {k: v for k, v in parent["versions"].items() if k != "labels"}  # 라벨 버전 말고는 그대로
    assert manifest["generator"] == GENERATOR_VERSION and children[parent_batch["records"][0]["episode_id"]]["versions"]["generator"] == GENERATOR_VERSION
    assert manifest["gripper_labels"]["parent"]["classes"]["initiate"] <= manifest["gripper_labels"]["derived"]["classes"]["initiate"]
    assert manifest["gripper_labels"]["derived"]["classes"]["initiate"] >= manifest["gripper_labels"]["derived"]["executed_closes"]
    assert manifest["episodes"] == 2 and (out / "manifest.json").is_file()
    assert manifest["gripper_labels"]["derived"]["sealed"] == {} and set(manifest["gripper_labels"]["derived"]["by_split"]) == {r["split"] for r in parent_batch["records"]}
    report = validate_dataset([record for _, record in read_episodes(out)])
    assert report["invalid_records"] == 0 and not report.get("errors")
    # 봉인 분할(`ood_test`)의 레코드가 부모에 있으면 그 라벨 종류 수는 적지 않는다 — 편 수만 (리뷰 1 M9)
    sealed_parent = tmp_path / "parent-with-sealed"
    for record in parent_batch["records"]:
        write_episode(record, sealed_parent)
    sealed = copy.deepcopy(parent_batch["records"][0])
    sealed["episode_id"] = sealed["episode_id"] + "-sealed"
    sealed["split"] = "ood_test"
    write_episode(sealed, sealed_parent)
    build_manifest(sealed_parent, CONFIG, batch_wall_s=1.0)
    manifest = derive_gripper_v2_dataset(sealed_parent, tmp_path / "g2-sealed", early_ticks=1)
    for side in ("parent", "derived"):
        block = manifest["gripper_labels"][side]
        assert "ood_test" not in block["by_split"] and block["sealed"] == {"ood_test": {"episodes": 1}}
        assert block["episodes"] == 2  # 전체 수도 봉인 분할을 뺀 것이다
    assert manifest["splits"].get("ood_test") == 1  # 분할 크기는 manifest에 그대로 있다 (데이터로 복사된다)


# --------------------------------------------------------------------------
# A4 — 모델 주행 기록의 DAgger 재료
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loop_batch(tmp_path_factory):
    """정책이 expert가 아닌 짧은 폐루프 기록 둘 — R4의 `closed_loop.py run`이 남기는 꼴(참조 라벨·정책 provenance)."""
    expert = Expert()
    policy = rule_judge_policy()
    out = tmp_path_factory.mktemp("loop") / "dev"
    for profile, seed in (("E0", 900101), ("E0", 900105)):  # R4 dev 목록의 E0 seed 둘 (계열 split = dev)
        record = generate_episode(profile, seed, policy=policy, expert=expert, config=CONFIG, max_ticks=30, id_suffix="-r4-x")
        assert record["split"] == "dev"
        write_episode(record, out)
    manifest = build_manifest(out, CONFIG, batch_wall_s=1.0)
    manifest["closed_loop"] = {"version": "cl0.1", "policy": {"name": "RuleJudge", "kind": "rule"}, "condition": "dev", "label": "x", "episodes": 2}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def test_the_dagger_dataset_relabels_model_driven_records_and_marks_them_as_training_material(loop_batch, tmp_path):
    out = tmp_path / "dagger-0"
    manifest = build_dagger_dataset([loop_batch], out, early_ticks=1, config=CONFIG, cycle=0)
    records = [record for _, record in read_episodes(out)]
    assert len(records) == 2 and manifest["episodes"] == 2
    for record in records:
        validate_record(record)
        assert record["split"] == "train" and record["episode_id"].endswith("-r4-x")
        assert record["versions"]["labels"] == GRIPPER_LABELS_VERSION
        dagger = record["provenance"]["dagger"]
        assert dagger["version"] == DAGGER_VERSION and dagger["cycle"] == 0 and dagger["relabel_source"] == "expert_v0"
        assert dagger["scene_split"] == "dev" and dagger["scene_condition"] == "dev" and dagger["source_dataset"] == str(loop_batch)
        assert dagger["policy"]["name"] == "RuleJudge" and dagger["gripper_labels"] == {"version": GRIPPER_LABELS_VERSION, "rule": "v2", "early_ticks": 1}
        assert record["provenance"]["material"] == "error_family"
        for tick in record["ticks"]:
            assert all(label["relabel"] is True for label in tick["labels"])
            assert "model_output" in tick and "adopted" in tick  # 실행된 것은 그대로
        desired = desired_from_evidence(record)
        ids = _gripper_ids(record)
        for at in gripper_transitions(desired):
            assert ids[at] == [desired[at]]
    assert manifest["dagger"]["version"] == DAGGER_VERSION and manifest["dagger"]["cycle"] == 0 and manifest["dagger"]["episodes"] == 2
    assert manifest["dagger"]["sources"] == [str(loop_batch)] and manifest["dagger"]["relabel_source"] == "expert_v0"
    assert manifest["splits"] == {"train": 2} and manifest["dagger"]["scene_splits"] == {"dev": 2}
    assert set(manifest["gripper_labels"]["derived"]["classes"]) == set(GRIPPER_LABEL_CLASSES)
    assert manifest["dagger"]["totals"]["ticks"] == sum(len(record["ticks"]) for record in records)
    report = validate_dataset(records)
    assert report["invalid_records"] == 0 and not report.get("errors")


def test_the_dagger_dataset_refuses_sealed_or_ood_dev_sources(loop_batch, tmp_path):
    """세 가지 검사를 각각 건드린다 (리뷰 1 M10): 경로에 봉인 이름이 든 것, 경로는 깨끗하지만 manifest의 `closed_loop.condition`이 ood_dev인 것,
    경로·조건은 깨끗하지만 레코드의 `split`이 ood_dev인 것 — 셋 다 거절이고, 거절 전에 아무것도 쓰지 않는다."""
    # (1) 경로만으로 — 디렉터리는 존재하지 않아도 된다 (읽기 전에 거절한다)
    with pytest.raises(ValueError, match="ood_test"):
        build_dagger_dataset([tmp_path / "ood_test"], tmp_path / "rejected-path", early_ticks=1, config=CONFIG)
    assert not (tmp_path / "rejected-path").exists()
    # (2) 조건만으로 — 경로 이름은 깨끗하다
    by_condition = tmp_path / "loop-cond" / "run"
    for _, record in read_episodes(loop_batch):
        write_episode(record, by_condition)
    manifest = build_manifest(by_condition, CONFIG, batch_wall_s=1.0)
    manifest["closed_loop"] = {"condition": "ood_dev", "policy": {"name": "RuleJudge"}}
    (by_condition / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ood_dev"):
        build_dagger_dataset([by_condition], tmp_path / "rejected-cond", early_ticks=1, config=CONFIG)
    assert not (tmp_path / "rejected-cond").exists()
    # (3) 레코드의 split만으로 — 경로도 조건도 깨끗하다
    by_split = tmp_path / "loop-split" / "run"
    for _, record in read_episodes(loop_batch):
        record = copy.deepcopy(record)
        record["split"] = "ood_dev"
        write_episode(record, by_split)
    manifest = build_manifest(by_split, CONFIG, batch_wall_s=1.0)
    manifest["closed_loop"] = {"condition": "dev", "policy": {"name": "RuleJudge"}}
    (by_split / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ood_dev"):
        build_dagger_dataset([by_split], tmp_path / "rejected-split", early_ticks=1, config=CONFIG)
    assert not (tmp_path / "rejected-split").exists()


# --------------------------------------------------------------------------
# A2 — 이른 답 실험 정책
# --------------------------------------------------------------------------


def test_the_reference_schedule_reads_the_transitions_of_the_experts_own_record():
    record = _labelled_record(["open", "open", "closed", "closed", "open", "open", "closed"])
    schedule = reference_gripper_schedule(record)
    assert schedule == {"closed": [2, 6], "open": [4]}


def test_the_early_policy_overrides_only_the_k_ticks_before_a_scheduled_close_while_the_expert_still_says_open():
    class Stub:
        version = "stub"
        label_config = {}
        label_source = "expert_v0"

        def __init__(self, desired):
            self.desired = desired
            self.calls = 0

        def act(self, request, commitment=None, observation=None):
            value = self.desired[self.calls]
            self.calls += 1
            spread = {"open": 0.9, "closed": 0.1} if value == "open" else {"open": 0.1, "closed": 0.9}
            return {"q_gripper": spread, "q_main": {"c1": 1.0}, "expert_meta": {"aux": {"gripper": {"desired": value, "reason": "x"}}}}

    stub = Stub(["open", "open", "open", "closed", "closed", "open"])
    policy = EarlyGripperPolicy(stub, early_ticks=2, directions=("closed",))
    policy.begin(("E0", 1), {"closed": [3], "open": [5]})
    answers = [policy.act({"t": 5 * index}, None, None) for index in range(6)]
    closed = [round(a["q_gripper"]["closed"], 1) for a in answers]
    assert closed == [0.1, 0.9, 0.9, 0.9, 0.9, 0.1]  # 틱 1·2가 일찍 `closed`; 전환 틱 3부터는 expert 자신의 답
    assert [a["expert_meta"]["aux"]["gripper"]["desired"] for a in answers] == stub.desired  # 참조(expert_meta)는 바꾸지 않는다
    assert policy.overrides == [1, 2] and policy.name == "EarlyGripperExpert" and policy.version.startswith("early-k2")
    both = EarlyGripperPolicy(Stub(["open", "open", "open", "closed", "closed", "open"]), early_ticks=1, directions=("closed", "open"))
    both.begin(("E0", 1), {"closed": [3], "open": [5]})
    answers = [both.act({"t": 5 * index}, None, None) for index in range(6)]
    assert [round(a["q_gripper"]["closed"], 1) for a in answers] == [0.1, 0.1, 0.9, 0.9, 0.1, 0.1]  # 틱 2 이른 closed, 틱 4 이른 open
    assert both.overrides == [2, 4]
    none = EarlyGripperPolicy(Stub(["open", "closed"]), early_ticks=0)
    none.begin(("E0", 1), {"closed": [1], "open": []})
    assert [round(none.act({"t": 0}, None, None)["q_gripper"]["closed"], 1), round(none.act({"t": 5}, None, None)["q_gripper"]["closed"], 1)] == [0.1, 0.9]
    assert none.overrides == []
