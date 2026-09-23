"""`scripts/report_tables.py` — 보고서의 사건·안전·대조 표를 산출물에서 찍는 쪽 (R2 fix round 1, 리뷰 1 M-10·C-1).

두 가지를 묶는다.

1. **`q_stop` 열은 검열을 언제나 데리고 나온다** (C-1). 중앙 지연은 *반응한* 사건만의 중앙값이라, 10건 중
   1건만 잡은 열의 "중앙 2틱"은 검열률 없이 읽으면 거짓이다 — docs/08 §10이 "지연과 오경보를 같이 적는다,
   상한을 평균에 섞으면 '느리다'와 '안 한다'가 같은 수가 된다"고 못 박은 자리가 그것이다.
2. **표를 만드는 코드가 `git clone`에서 산다** (M-10). R2까지 이 표는 `artifacts/scratch/`의 스크립트가
   찍었고 `artifacts/`는 커밋되지 않는다.
"""

import functools
import importlib.util
import sys

from helpers import REPO

SCRIPT = REPO / "scripts" / "report_tables.py"


@functools.lru_cache(maxsize=1)
def script():
    spec = importlib.util.spec_from_file_location("report_tables", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _events(**stop):
    """사건 지표 한 열 — `q_stop` 말고는 표가 서기만 하면 되는 최소값."""
    return {
        "reaction_delay": {"goal_change": {"median_ticks": 0.0, "immediate_rate": 0.6, "censored_rate": 0.0, "events": 45},
                           "world_event": {"median_ticks": 0.0, "censored_rate": 0.002, "events": 513},
                           "horizon_ticks": 30},
        "stability": {"switch_rate": 0.016, "round_trips": 1, "hold_ticks_median": 4.0},
        "stop_timing": dict(stop) or None,
    }


def test_the_stop_column_always_carries_the_censoring_beside_the_median():
    """C-1 — R2 B2d의 실제 값: 정지 사건 10건 중 **1건** 반응, 9건 검열, 그 1건의 지연이 2틱이다."""
    module = script()
    cell = module.stop_cell({"onsets": 10, "reacted": 1, "censored": 9, "censored_rate": 0.9,
                             "median_ticks": 2.0, "false_alarm_rate": 0.0})
    assert cell == "2.0 / 9 of 10 (90.0 %) / 0.00 %"
    # 아무것도 발화하지 않은 열은 중앙값이 없고 검열이 전부다 (T0·40 step T1이 그랬다)
    assert module.stop_cell({"onsets": 10, "reacted": 0, "censored": 10, "median_ticks": None,
                             "false_alarm_rate": 0.0}) == "— / 10 of 10 (100.0 %) / 0.00 %"
    # 규칙 판정기: 9건 반응, 1건 검열, 오경보 1.72 %
    assert module.stop_cell({"onsets": 10, "reacted": 9, "censored": 1, "median_ticks": 0.0,
                             "false_alarm_rate": 0.01723037651563497}) == "0.0 / 1 of 10 (10.0 %) / 1.72 %"
    # 이 열이 아예 없는 기준군(기계적 기준군)은 빈칸 셋
    assert module.stop_cell(None) == "— / — / —"


def test_the_event_table_header_and_row_name_the_censoring():
    """표의 머리글과 줄이 함께 움직인다 — 머리글만 고치고 칸을 안 고치는 일이 없게."""
    module = script()
    data = {"evaluation": {"splits": {"robot/ood_dev": {"event_metrics": {
        "model": _events(onsets=10, reacted=1, censored=9, median_ticks=2.0, false_alarm_rate=0.0),
        "rule_judge": _events(onsets=10, reacted=9, censored=1, median_ticks=0.0, false_alarm_rate=0.0172),
    }}}}}
    table = module.events_table(data, "robot/ood_dev")
    header, *_ = table.splitlines()
    assert "stop: median / censored / false alarm" in header
    assert "| 2.0 / 9 of 10 (90.0 %) / 0.00 % |" in table
    assert "stop onsets 10" in table


def test_the_repo_root_is_found_from_the_script_not_from_a_hard_coded_worktree():
    """M-10 — 스크래치 판은 worktree 경로가 박혀 있었다. 커밋된 코드는 `git clone` 어디서나 서야 한다."""
    module = script()
    assert module.REPO == REPO and (module.REPO / "scripts" / "report_tables.py").is_file()
