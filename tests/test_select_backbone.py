"""`scripts/select_backbone.py`의 요약 — 대조군 열이 **어느 대조군인지**를 잃지 않는지 (D1 리뷰 2 N4).

G0b의 평가 JSON(`adapt-*.json`·`zero-shot-*.json`)은 `context_shuffle_kind`가 생기기 전의 것이라 로봇 스트림 열이
**지시 텍스트** 대조군이다. 지금 실행은 같은 키에 id 재매핑 **상태** 섞기를 적는다 — 둘을 한 열로 읽으면 안 되므로
요약이 종류를 반드시 적는다.
"""

import functools
import importlib.util
import sys

from helpers import REPO

SCRIPT = REPO / "scripts" / "select_backbone.py"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("select_backbone", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _table(**extra) -> dict:
    return {
        "n_states": 3,
        "model": {"q_main": {"n": 3, "accuracy": 0.5, "nll": 1.0, "brier": 0.4, "first_position_rate": 0.1}, "_all": {"accuracy": 0.5, "nll": 1.0, "brier": 0.4}},
        "answer_change": {"rate": 0.33},
        "context_shuffle": {"_all": {"accuracy": 0.4}},
        "rule_judge": {"_all": {"accuracy": 0.7}},
        **extra,
    }


def test_eval_summary_names_the_control_column_and_labels_runs_that_predate_the_key():
    module = script()
    new = module._eval_summary({
        "robot/dev": _table(context_shuffle_kind="state", instruction_shuffle={"_all": {"accuracy": 0.45}}, instruction_shuffle_kind="instruction"),
    })["robot/dev"]
    assert new["context_shuffle_accuracy"] == 0.4 and new["context_shuffle_kind"] == "state"
    assert new["instruction_shuffle_accuracy"] == 0.45 and new["instruction_shuffle_kind"] == "instruction"

    legacy = module._eval_summary({"batch0/dev": _table()})["batch0/dev"]
    assert legacy["context_shuffle_accuracy"] == 0.4
    assert legacy["context_shuffle_kind"] == module.LEGACY_CONTEXT_SHUFFLE_KIND
    assert "instruction" in legacy["context_shuffle_kind"] and legacy["instruction_shuffle_accuracy"] is None


def test_report_carries_the_note_that_the_two_controls_are_not_one_column():
    report = script().build("선정 문단")
    assert "instruction/text control" in report["context_shuffle_kind_note"]
    assert module_kind_in(report) is True


def module_kind_in(report: dict) -> bool:
    """요약이 있는 후보라면 모든 분할이 `context_shuffle_kind`를 갖는다 (없는 실행은 이름으로 표시된다)."""
    for candidate in report["candidates"].values():
        for mode in ("t0", "lora"):
            summary = (candidate.get(mode) or {}).get("evaluation") or {}
            for split in summary.values():
                if "context_shuffle_kind" not in split:
                    return False
    return True
