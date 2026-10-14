"""Author A1 for v0.6 (author diversity): warehouse and logistics, a terse telemetry block, English, metric units.

Three MuJoCo scenes, one per law — ``mast`` (A: a mast picker AMR braking — front-wheel skid or rear-caster lift), ``dock``
(B: a pallet load set down overhanging the dock edge) and ``ramp`` (C: a tugger and its cart held on a ramp by braked wheels).

* :mod:`~robo_jev.authors.a1.sims` — the MJCF scenes and rollouts;
* :mod:`~robo_jev.authors.a1.catalog` — word → hidden-range catalogs;
* :mod:`~robo_jev.authors.a1.scenes` — stated facts, priors, actions, the aiming surrogate and the wording;
* :mod:`~robo_jev.authors.a1.labels` — labels from per-draw outcomes;
* :mod:`~robo_jev.authors.a1.build` — plans, rollouts and records; :mod:`~robo_jev.authors.a1.assemble` — selection and stats;
* :mod:`~robo_jev.authors.a1.checks` / :mod:`~robo_jev.authors.a1.sanity` — validation, overlap and physics sanity.

Driver: ``scripts/author_a1.py``; outputs under ``artifacts/v06/a1/``.
"""
