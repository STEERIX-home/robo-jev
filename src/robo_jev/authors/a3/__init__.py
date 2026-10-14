"""Author A3 of the v0.6 author-diversity experiment — an independent MuJoCo-labelled generator.

Domain: laboratory, hospital and light manufacturing. Style: numbered work orders with review items. One scene per law:

* ``indexdial`` (law A, tip or slide under acceleration) — a loose part on a rotary index table (:mod:`.sim`, :mod:`.scenes`);
* ``mayostand`` (law B, centre of mass past the support edge) — an instrument set released onto a cantilevered stand tray;
* ``holddown`` (law C, friction-limited holding) — a part under a pneumatic hold-down while a cobot presses a fitting into its side.

:data:`EVENT` and :data:`PLAN_KEY` are what :mod:`robo_jev.bench.infamily` needs to read A3's comparison questions
(:func:`register_infamily` adds them to its tables).
"""

from __future__ import annotations

EVENT = {"indexdial": ("slides_off_mark", "rocks_back", "falls_over"), "mayostand": ("rocks", "tips_over"), "holddown": ("creeps", "breaks_free")}
PLAN_KEY = {"indexdial": "as_written", "mayostand": "as_written", "holddown": "as_written"}


def register_infamily() -> None:
    """Add A3's families to :data:`robo_jev.bench.infamily.EVENT` / ``PLAN_KEY`` (in-process; the module is not edited)."""
    from robo_jev.bench import infamily

    infamily.EVENT.update(EVENT)
    infamily.PLAN_KEY.update(PLAN_KEY)
