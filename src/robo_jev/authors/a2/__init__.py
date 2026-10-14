"""Author A2 of the v0.6 author-diversity experiment — home and hospitality service, second-person supervisor briefings.

Three MuJoCo scenes, one per law: ``turntable`` (A, tip or slide under acceleration: an item on a lazy Susan the robot turns),
``tiltback`` (B, centre of mass past an edge: furniture tipped back on its rear edge and held by the robot's arm) and ``wallpress``
(C, friction-limited holding: a flat item pressed to a wall by the robot's pad). Modules: :mod:`.physics` (MJCF and rollouts),
:mod:`.catalog` (word → range priors, costs), :mod:`.scenes` (facts, hidden ranges, variants, static aim), :mod:`.text` (wording),
:mod:`.generate` (plan → label → records), :mod:`.sanity` (thresholds, sweeps, determinism). Script: ``scripts/author_a2.py``.

The in-family scorer (:mod:`robo_jev.bench.infamily`) keys the key event and the plan's comparison option by family; this author's
entries are :data:`EVENT` and :data:`PLAN_KEY` (also written into every record as ``meta.key_events`` and ``meta.plan_key``), and
:func:`registered_with_infamily` adds them for the duration of a block without editing the shared module.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

__all__ = ["EVENT", "PLAN_KEY", "SCENE_NAMES", "registered_with_infamily"]

SCENE_NAMES = ("turntable", "tiltback", "wallpress")
#: The key event of each scene (any of these outcomes).
EVENT = {"turntable": ("skids", "topples"), "tiltback": ("hangs", "over"), "wallpress": ("slips", "drops")}
#: The plan's option key in the comparison (and action) question of each scene.
PLAN_KEY = {"turntable": "turn_as_planned", "tiltback": "tip_as_planned", "wallpress": "hold_as_planned"}


@contextlib.contextmanager
def registered_with_infamily() -> Iterator[None]:
    """Temporarily add this author's scenes to :data:`robo_jev.bench.infamily.EVENT` / ``PLAN_KEY`` (restored on exit)."""
    from robo_jev.bench import infamily

    saved_event, saved_plan = dict(infamily.EVENT), dict(infamily.PLAN_KEY)
    infamily.EVENT.update(EVENT)
    infamily.PLAN_KEY.update(PLAN_KEY)
    try:
        yield
    finally:
        infamily.EVENT.clear()
        infamily.EVENT.update(saved_event)
        infamily.PLAN_KEY.clear()
        infamily.PLAN_KEY.update(saved_plan)
