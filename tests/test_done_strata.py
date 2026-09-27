"""`q_done` 틱 층·거짓 done 사실 검사 (Task R6 A0·A1; docs/08 §7 `q_done` 라벨 `goal-zone-containment-v0`)."""

import copy
import math

import pytest

from robo_jev.data.done_strata import (
    DONE_STRATA,
    RELEASE_WINDOW_TICKS,
    count_done_strata,
    done_strata_by_split,
    false_done_facts,
    model_q_done,
    model_rates_by_stratum,
    old_goal_variants,
    release_events,
    release_window_counts,
    tick_done_strata,
)

ZONES = [{"id": "zoneL", "bounds_mm": [440, 150, 740, 330]}, {"id": "zoneR", "bounds_mm": [440, -330, 740, -150]}]
IN_L, IN_R, OUT = [500, 200, -80], [500, -200, -80], [300, 0, -80]


def _tick(t, *, version, target, zone, poses, holding=None, done=False, p_done=None, gate=None, extra_usage=None):
    """층이 읽는 필드만 가진 틱 — 목표(구조화)·물체 자세·영역·손·`q_done` 라벨·모델 답."""
    tick = {
        "t": t * 5,
        "request": {"state": {
            "goal": {"version": version, "target_ref": target, "target_zone": zone, "text": f"v{version}"},
            "objects": [{"id": name, "pose_mm": list(pose)} for name, pose in poses.items()],
            "zones": copy.deepcopy(ZONES), "robot": {"holding": holding},
        }},
        "labels": [{"question_id": "q_done", "kind": "single", "answer": bool(done), "source": "expert_v0", "rule": "goal-zone-containment-v0"}],
        "usage": {"gate": gate, **(extra_usage or {})},
    }
    if p_done is not None:
        tick["model_output"] = {"q_done": float(p_done)}
    return tick


def _record(ticks, *, done=False, inside=False, profile="E1", episode="ep-E1-000001"):
    first_done = next((index for index, tick in enumerate(ticks) if (tick.get("usage") or {}).get("gate") == "done"), None)
    return {
        "episode_id": episode, "split": "train", "ticks": ticks,
        "provenance": {"profile": profile, "outcome": {"done": done, "target_inside_zone": inside, "first_done_tick": first_done}},
    }


def _false_done_record():
    """v1: o1 → zoneL. 모델이 o1을 zoneL에 놓는 동안 지시가 v2(o2 → zoneR)로 바뀌고, 놓은 직후 모델의 `q_done`이 든다."""
    ticks = [
        _tick(0, version=1, target="o1", zone="zoneL", poses={"o1": OUT, "o2": OUT}, p_done=0.0),
        _tick(1, version=1, target="o1", zone="zoneL", poses={"o1": OUT, "o2": OUT}, holding="o1", p_done=0.0),
        _tick(2, version=2, target="o2", zone="zoneR", poses={"o1": IN_L, "o2": OUT}, holding="o1", p_done=0.0),
        _tick(3, version=2, target="o2", zone="zoneR", poses={"o1": IN_L, "o2": OUT}, p_done=0.7, gate="done"),  # 놓았다
        _tick(4, version=2, target="o2", zone="zoneR", poses={"o1": IN_L, "o2": OUT}, p_done=0.99, gate="done"),
        _tick(5, version=2, target="o2", zone="zoneR", poses={"o1": IN_L, "o2": OUT}, p_done=1.0, gate="done"),
    ]
    return _record(ticks, done=True, inside=False)


# --------------------------------------------------------------------------
# 층의 정의
# --------------------------------------------------------------------------


def test_the_strata_are_five_in_precedence_order_and_the_window_is_six_ticks():
    assert DONE_STRATA == ("done_true", "post_release_other", "old_goal_satisfied", "post_release_current_outside", "other_false")
    assert RELEASE_WINDOW_TICKS == 6


def test_a_release_is_the_first_tick_whose_hand_no_longer_holds_the_object():
    record = _false_done_record()
    assert release_events(record["ticks"]) == [(3, "o1")]
    # 다른 물체로 바로 바뀌어도 앞 물체를 놓은 것이다
    ticks = [_tick(i, version=1, target="o1", zone="zoneL", poses={"o1": OUT, "o2": OUT}, holding=h) for i, h in enumerate(["o1", "o2", None])]
    assert release_events(ticks) == [(1, "o1"), (2, "o2")]


def test_releasing_a_non_target_puts_the_next_six_ticks_in_post_release_other_and_then_the_old_goal_takes_over():
    record = _false_done_record()
    base = record["ticks"][-1]
    ticks = copy.deepcopy(record["ticks"]) + [copy.deepcopy(base) for _ in range(6)]
    for index, tick in enumerate(ticks):
        tick["t"] = index * 5
        tick["usage"]["gate"] = None
    strata = tick_done_strata({"ticks": ticks})
    # 틱 3(놓은 틱)부터 3+6 = 9까지가 창이다 — 놓은 물체 o1은 지금 대상(o2)이 아니다
    assert strata[3:10] == ["post_release_other"] * 7
    # 창이 끝나면 이전 지시(o1 → zoneL)가 성립해 있으므로 old_goal_satisfied
    assert strata[10:] == ["old_goal_satisfied"] * (len(ticks) - 10)
    # 놓기 전: 틱 2는 지시가 막 바뀐 틱이고 o1은 아직 손에 있다 — 이전 목표는 성립하지 않는다(손에 있다)
    assert strata[:3] == ["other_false"] * 3


def test_a_true_reference_label_is_done_true_whatever_else_holds():
    ticks = [
        _tick(0, version=1, target="o1", zone="zoneL", poses={"o1": OUT}, holding="o1"),
        _tick(1, version=1, target="o1", zone="zoneL", poses={"o1": IN_L}, done=True),  # 대상을 영역 안에 놓았다
    ]
    assert tick_done_strata({"ticks": ticks}) == ["other_false", "done_true"]


def test_releasing_the_current_target_outside_its_zone_is_its_own_stratum():
    ticks = [
        _tick(0, version=1, target="o1", zone="zoneL", poses={"o1": OUT}, holding="o1"),
        _tick(1, version=1, target="o1", zone="zoneL", poses={"o1": OUT}),  # 영역 밖에 떨어뜨렸다
        _tick(2, version=1, target="o1", zone="zoneL", poses={"o1": OUT}),
    ]
    assert tick_done_strata({"ticks": ticks}) == ["other_false", "post_release_current_outside", "post_release_current_outside"]
    # 창(6틱)이 지나면 other_false
    long = ticks + [copy.deepcopy(ticks[-1]) for _ in range(6)]
    for index, tick in enumerate(long):
        tick["t"] = index * 5
    assert tick_done_strata({"ticks": long})[1:8] == ["post_release_current_outside"] * 7 and tick_done_strata({"ticks": long})[8] == "other_false"


def test_an_old_goal_counts_whichever_earlier_instruction_it_was_and_only_while_the_object_is_out_of_the_hand():
    # v1: o1 → zoneL (o1은 처음부터 zoneL 안), v2: o2 → zoneR, v3: o3 → zoneR — 지금은 v3이고 v1의 목표가 성립해 있다
    ticks = [
        _tick(0, version=1, target="o1", zone="zoneL", poses={"o1": IN_L, "o2": OUT, "o3": OUT}, done=True),
        _tick(1, version=2, target="o2", zone="zoneR", poses={"o1": IN_L, "o2": OUT, "o3": OUT}),
        _tick(2, version=3, target="o3", zone="zoneR", poses={"o1": IN_L, "o2": OUT, "o3": OUT}),
        _tick(3, version=3, target="o3", zone="zoneR", poses={"o1": IN_L, "o2": OUT, "o3": OUT}, holding="o1"),  # 손에 들면 성립하지 않는다
    ]
    assert tick_done_strata({"ticks": ticks}) == ["done_true", "old_goal_satisfied", "old_goal_satisfied", "other_false"]


def test_the_strata_never_look_at_future_ticks():
    record = _false_done_record()
    full = tick_done_strata(record)
    for cut in range(1, len(record["ticks"]) + 1):
        prefix = {"ticks": copy.deepcopy(record["ticks"][:cut])}
        assert tick_done_strata(prefix) == full[:cut]


def test_ticks_without_a_done_label_have_no_stratum():
    record = _false_done_record()
    record["ticks"][0]["labels"] = []
    assert tick_done_strata(record)[0] is None
    assert count_done_strata([record])["labelled_ticks"] == len(record["ticks"]) - 1


# --------------------------------------------------------------------------
# 셈
# --------------------------------------------------------------------------


def test_counts_cover_every_labelled_tick_and_name_the_episodes_each_stratum_comes_from():
    record = _false_done_record()
    counts = count_done_strata([record, copy.deepcopy(record)])
    assert set(counts["strata"]) == set(DONE_STRATA)
    assert sum(counts["strata"].values()) == counts["labelled_ticks"] == 2 * len(record["ticks"])
    assert counts["strata"]["post_release_other"] == 6 and counts["episodes_with"]["post_release_other"] == 2
    assert counts["episodes"] == 2


def test_sealed_splits_are_counted_by_episodes_only():
    record = _false_done_record()
    sealed = copy.deepcopy(record)
    sealed["split"] = "ood_test"
    out = done_strata_by_split([record, sealed])
    assert set(out["by_split"]) == {"train"} and out["sealed"] == {"ood_test": {"episodes": 1}}
    assert out["labelled_ticks"] == len(record["ticks"])  # 전체는 봉인 분할을 뺀 수다


def test_the_model_answer_is_read_from_the_raw_usage_field_when_a_collection_policy_replaced_it():
    record = _false_done_record()
    tick = record["ticks"][3]
    assert model_q_done(tick) == pytest.approx(0.7)
    tick["usage"]["model_q_done"] = 0.2  # 수집 정책은 하네스에 expert의 답을 넘기고 모델의 raw 답을 따로 남긴다
    assert model_q_done(tick) == pytest.approx(0.2)
    del tick["model_output"]
    tick["usage"].pop("model_q_done")
    assert model_q_done(tick) is None


def test_model_rates_count_true_answers_by_the_harness_threshold_per_stratum():
    record = _false_done_record()
    rates = model_rates_by_stratum([record])
    assert rates["post_release_other"] == {"n": 3, "answered": 3, "model_true": 3, "rate": 1.0, "episodes": 1}
    assert rates["other_false"]["model_true"] == 0 and rates["other_false"]["n"] == 3


def test_release_window_counts_split_the_window_by_whose_object_was_released_and_by_the_reference():
    """창의 규칙 자체를 고정한다 (브리프의 g2 train 403 / 107 / 853은 산출물 `r6-a0-false-done.json`의 수다 — 이 시험은 그 수를 재현하지 않는다):
    놓은 틱(0)부터 6틱까지를 (놓은 물체가 지금 대상인가) × (참조가 참인가)로 네 칸에 나누고, 창 밖의 틱은 세지 않는다."""
    o9, o1 = {"o9": OUT}, "o1"
    specs = [
        dict(holding="o9", o1=OUT, done=False),  # 0: 아직 o9를 든다 — 창 없음
        dict(holding=None, o1=OUT, done=False),  # 1: o9를 놓았다 (지금 대상 o1이 아니다) → other_not_done
        dict(holding=None, o1=IN_L, done=True),  # 2: 창 안에서 o1이 영역 안 — 참조 참 → other_done
    ] + [dict(holding=None, o1=OUT, done=False) for _ in range(5)] + [  # 3~7: 창 안(1~6) → other_not_done
        dict(holding=o1, o1=OUT, done=False),  # 8: o9의 창 밖(7), o1을 든다 — 세지 않는다
        dict(holding=None, o1=OUT, done=False),  # 9: o1(지금 대상)을 영역 밖에 놓았다 → current_not_done
        dict(holding=None, o1=IN_L, done=True),  # 10: o1이 영역 안 → current_done
    ]
    ticks = [_tick(i, version=1, target="o1", zone="zoneL", poses={**o9, "o1": spec["o1"]}, holding=spec["holding"], done=spec["done"]) for i, spec in enumerate(specs)]
    counts = release_window_counts([_record(ticks)])
    assert counts == {"other_not_done": 6, "other_done": 1, "current_not_done": 1, "current_done": 1, "window_ticks": RELEASE_WINDOW_TICKS}


# --------------------------------------------------------------------------
# A0 — 거짓 done의 사실
# --------------------------------------------------------------------------


def _three_instruction_record(*, o1, o2, released):
    """v1: o1 → zoneL, v2: o2 → zoneR, v3: o3 → zoneR(지금). 틱 1에 `released`를 들고 틱 2에 놓는다(o1이면 zoneL 안에 놓는다)."""
    poses = {"o1": o1, "o2": o2, "o3": OUT, "o9": OUT}
    ticks = [
        _tick(0, version=1, target="o1", zone="zoneL", poses=poses),
        _tick(1, version=2, target="o2", zone="zoneR", poses=poses, holding=released),
        _tick(2, version=3, target="o3", zone="zoneR", poses=poses),
    ]
    return {"ticks": ticks}


def test_old_goal_variants_distinguish_the_immediate_previous_any_earlier_and_the_released_objects_instruction():
    """세 정의가 **갈리는** 장면마다 (A0의 16 / 21 / 20이 이 셋이다): 바로 앞 지시(v2)·어느 이전 지시·마지막으로 놓은 물체의 지시."""
    # 모두 참: 두 버전, 놓은 o1이 v1의 대상이고 v1의 영역 안
    assert old_goal_variants(_false_done_record()) == {"immediately_previous": True, "any_earlier": True, "released_objects_instruction": True, "last_released": "o1"}
    # v1만 성립(o1이 원래 zoneL 안), v2 불성립, 놓은 것은 대상이 아닌 o9 → 어느 이전 지시만 참
    only_any = old_goal_variants(_three_instruction_record(o1=IN_L, o2=OUT, released="o9"))
    assert only_any == {"immediately_previous": False, "any_earlier": True, "released_objects_instruction": False, "last_released": "o9"}
    # v1의 대상 o1을 zoneL에 놓았다, v2 불성립 → 바로 앞 지시는 거짓, 나머지 둘은 참
    released_old = old_goal_variants(_three_instruction_record(o1=IN_L, o2=OUT, released="o1"))
    assert released_old == {"immediately_previous": False, "any_earlier": True, "released_objects_instruction": True, "last_released": "o1"}
    # v2가 성립(o2가 원래 zoneR 안), 놓은 것은 o9 → 바로 앞·어느 이전은 참, 놓은 물체의 지시는 거짓
    previous_only = old_goal_variants(_three_instruction_record(o1=OUT, o2=IN_R, released="o9"))
    assert previous_only == {"immediately_previous": True, "any_earlier": True, "released_objects_instruction": False, "last_released": "o9"}
    # 아무 이전 목표도 성립하지 않는다
    assert old_goal_variants(_three_instruction_record(o1=OUT, o2=OUT, released="o9"))["any_earlier"] is False


def test_false_done_facts_describe_each_false_done_episode_and_leave_true_completions_out():
    record = _false_done_record()
    good = copy.deepcopy(record)
    good["episode_id"] = "ep-E1-000002"
    good["provenance"]["outcome"]["target_inside_zone"] = True
    facts = false_done_facts([record, good])
    assert facts["episodes"] == 2 and facts["false_done"] == 1
    row = facts["rows"][0]
    assert row["episode_id"] == "ep-E1-000001" and row["profile"] == "E1" and row["goal_version"] == 2
    assert row["held_current_target_ever"] is False and row["held_current_target_during_instruction"] is False
    assert row["target_outside_mm"] == pytest.approx(math.hypot(140.0, 150.0))  # (300, 0)에서 zoneR 사각형 [440, -330, 740, -150]까지
    assert row["reference_false_on_every_done_tick"] is True
    assert row["q_done_before_first_done"] == pytest.approx(0.0) and row["q_done_first_done"] == pytest.approx(0.7)
    assert row["q_done_done_ticks"] == [pytest.approx(0.7), pytest.approx(0.99), pytest.approx(1.0)]
    assert facts["summary"]["profiles"] == {"E1": 1} and facts["summary"]["never_held_current_target"] == 1
    assert facts["summary"]["old_goal"] == {"immediately_previous": 1, "any_earlier": 1, "released_objects_instruction": 1}


# --------------------------------------------------------------------------
# 모델의 raw `q_done`이 거짓으로 든 사건과 그 뒤 (A2·A3)
# --------------------------------------------------------------------------


def _collected_record():
    """수집 정책의 기록 꼴: 하네스는 expert의 `q_done`을 받았고(`model_output`), 모델의 raw 답은 `usage.model_q_done`에 있다.
    모델은 틱 3~6에서 거짓으로 들고(4틱 연속), 틱 8에서 한 번 더 들며, 틱 9에 지금 대상 o2를 잡고, 틱 11에 영역 안에 놓는다."""
    raw = [0.0, 0.0, 0.0, 0.8, 0.9, 0.95, 0.7, 0.1, 0.6, 0.0, 0.0, 0.9]
    ticks = []
    for index, p in enumerate(raw):
        holding = "o1" if index in (1, 2) else ("o2" if index in (9, 10) else None)
        version, target, zone = (1, "o1", "zoneL") if index < 2 else (2, "o2", "zoneR")
        poses = {"o1": IN_L if index >= 2 else OUT, "o2": IN_R if index >= 11 else OUT}
        done = index >= 11
        ticks.append(_tick(index, version=version, target=target, zone=zone, poses=poses, holding=holding, done=done, p_done=0.95 if done else 0.05,
                           gate="done" if done else None, extra_usage={"model_q_done": p}))
    return _record(ticks, done=True, inside=True, episode="ep-E1-600001")


def test_false_done_events_are_runs_of_raw_answers_over_the_gate_on_reference_false_ticks():
    from robo_jev.data.done_strata import false_done_events

    record = _collected_record()
    events = false_done_events([record])
    assert events["ticks"] == 5 and events["episodes"] == 1 and events["events"] == 2
    assert events["runs_reaching_the_tail"] == 1 and events["tail_ticks"] == 4  # 꼬리 3 + 1틱 연속이면 모델 자신의 게이트가 편을 끝냈다
    assert events["per_episode"]["ep-E1-600001"]["first_tick"] == 3


def test_recovery_counts_episodes_that_grasp_the_current_target_after_their_first_false_done():
    from robo_jev.data.done_strata import recovery_after_false_done

    record = _collected_record()
    never = copy.deepcopy(record)
    never["episode_id"] = "ep-E1-600002"
    for tick in never["ticks"][7:]:
        tick["request"]["state"]["robot"]["holding"] = None
    never["provenance"]["outcome"]["target_inside_zone"] = False
    out = recovery_after_false_done([record, never])
    assert out["episodes_with_false_done"] == 2
    assert out["grasped_current_target_after"] == 1 and out["completed_after"] == 1
    assert out["per_episode"]["ep-E1-600001"] == {"first_false_done_tick": 3, "grasped_current_target_at": 9, "completed": True}
    assert out["per_episode"]["ep-E1-600002"]["grasped_current_target_at"] is None


def test_the_count_script_never_opens_a_sealed_split_file(tmp_path):
    """봉인 분할(`ood_test`)은 읽지도 않는다 — manifest의 `files` 항목의 split으로 먼저 거르고 편 수만 센다 (docs/04 §5)."""
    import importlib.util
    import json
    import sys

    from helpers import REPO

    spec = importlib.util.spec_from_file_location("done_strata_script", REPO / "scripts" / "done_strata.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    record = _false_done_record()
    (tmp_path / "episodes" / "a").mkdir(parents=True)
    (tmp_path / "episodes" / "a" / "streams.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "episodes" / "s").mkdir(parents=True)
    (tmp_path / "episodes" / "s" / "streams.jsonl").write_text("this file must never be parsed\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"episodes/a/streams.jsonl": {"split": "train"}, "episodes/s/streams.jsonl": {"split": "ood_test"}}}), encoding="utf-8")
    records, sealed = module.load_open_records(tmp_path)
    assert [item["episode_id"] for item in records] == ["ep-E1-000001"] and sealed == {"ood_test": {"episodes": 1}}
    payload = module.build_counts({"x": tmp_path})
    assert payload["datasets"]["x"]["sealed"] == {"ood_test": {"episodes": 1}} and set(payload["datasets"]["x"]["by_split"]) == {"train"}
    # manifest가 없으면 파일을 열지 않고는 분할을 알 수 없다 — 읽고 버리는 대신 거절한다 (리뷰 1 I-1)
    bare = tmp_path / "bare"
    (bare / "episodes" / "a").mkdir(parents=True)
    (bare / "episodes" / "a" / "streams.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        module.load_open_records(bare)


# --------------------------------------------------------------------------
# B1 — 루프의 거짓 done 상태: 이전 지시의 대상을 그 지시의 영역에 막 놓은 틱 (수정 라운드 1, 리뷰 1 U-5)
# --------------------------------------------------------------------------


def test_old_target_release_ticks_mark_the_window_after_an_old_target_lands_in_its_old_zone():
    from robo_jev.data.done_strata import old_target_release_ticks, old_target_releases

    record = _false_done_record()
    assert old_target_release_ticks(record) == [False, False, False, True, True, True]  # 틱 3에 o1(v1의 대상)을 zoneL(v1의 영역)에 놓았다
    assert old_target_releases(record) == [{"tick": 3, "object": "o1", "instruction_version": 1, "ticks_since_instruction_change": 1}]
    # 놓은 물체가 이전 대상이 아니거나(o9) 이전 영역 밖이면 그 상태가 아니다
    other = _three_instruction_record(o1=IN_L, o2=OUT, released="o9")
    assert not any(old_target_release_ticks(other)) and old_target_releases(other) == []
    outside = _three_instruction_record(o1=OUT, o2=OUT, released="o1")
    assert not any(old_target_release_ticks(outside)) and old_target_releases(outside) == []


def test_the_loop_state_release_block_counts_releases_their_timing_and_a_q_done_that_follows():
    import importlib.util
    import sys

    from helpers import REPO

    spec = importlib.util.spec_from_file_location("done_strata_script_b1", REPO / "scripts" / "done_strata.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    record = _false_done_record()  # 틱 3에 옛 대상 o1을 옛 영역에 놓았고, 지금 지시(v2)는 틱 2에 시작했다; 모델 q_done은 틱 3에 0.7
    block = module._release_block([record], answers=lambda rec, index: model_q_done(rec["ticks"][index]))
    assert block == {"releases": 1, "episodes": 1, "ticks_since_instruction_change": {"median": 1.0, "min": 1, "max": 1}, "followed_by_q_done_within_3_ticks": 1}
    quiet = copy.deepcopy(record)
    for tick in quiet["ticks"]:
        tick["model_output"]["q_done"] = 0.1
    assert module._release_block([quiet], answers=lambda rec, index: model_q_done(rec["ticks"][index]))["followed_by_q_done_within_3_ticks"] == 0
    assert "followed_by_q_done_within_3_ticks" not in module._release_block([record])  # 답이 없는 칸(전문가 기록)에는 세지 않는다
