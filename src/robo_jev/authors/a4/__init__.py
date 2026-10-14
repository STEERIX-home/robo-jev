"""Author A4 of the v0.6 author-diversity experiment — the held-out test author (never trained on; sealed).

An independent MuJoCo-labelled generator: outdoor and field robots (orchards, farms, building sites, marinas, parks, footpath verges),
written as radio transcripts between a robot and its supervisor. One scene per law: ``bank`` (A, tip or slide under tilt), ``ledge``
(B, centre of mass past an edge), ``anchor`` (C, friction-limited holding). Modules: :mod:`.sims` (MJCF and rollouts), :mod:`.design`
(catalogs, priors, sampling, aiming), :mod:`.label` (groups, rollouts over the shared draws, labels, robustness), :mod:`.radio` (the
transcripts and questions), :mod:`.build` (records and selection), :mod:`.checks` (validation, sanity, overlap, prompt lengths).

``EVENT`` and ``PLAN_KEY`` give, per scene, the outcomes that make up the key event and the plan's option key — what
:func:`robo_jev.bench.infamily.comparison_items` reads from its tables (every record also carries them as ``meta.key_events`` and
``meta.plan_option``).
"""

from robo_jev.authors.a4.design import EVENT, OUTCOMES, PLAN_KEY, SCENES, SEVERE

SET_NAME = "v06-a4"
ID_PREFIX = "v06a4-"
TEST_FILE = "artifacts/v06/a4/test.jsonl"

__all__ = ["EVENT", "ID_PREFIX", "OUTCOMES", "PLAN_KEY", "SCENES", "SET_NAME", "SEVERE", "TEST_FILE"]
