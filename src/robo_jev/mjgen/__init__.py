"""MuJoCo-labelled **training** families (Task v0.3b) — a second, independent physics source next to the closed-form engine.

* :mod:`~robo_jev.mjgen.physics` — scenes and rollouts: a load on a pitching deck (``tilt``), an item held by a two-finger
  gripper while the arm moves (``grip``), a tower of two or three boxes released by the robot (``tower``). Built on the T8
  rigid-body settings and helpers (:mod:`robo_jev.outside.physics`), with new geometry.
* :mod:`~robo_jev.mjgen.families` — catalogs (the priors behind the words), candidate scenes aimed so the key label covers [0, 1]
  evenly, counterfactual and paired values, labels = outcome frequencies over K hidden draws.
* :mod:`~robo_jev.mjgen.text` — four narration styles and the question wording, written for this task.
* :mod:`~robo_jev.mjgen.generate` — records (``simgen-request/0``), one-fact pairs, balanced selection, statistics.
* :mod:`~robo_jev.mjgen.sanity` — physics checks (monotonicity, analytic thresholds, determinism).
* :mod:`~robo_jev.mjgen.overlap` — the zero-overlap check against the sealed outside test T8b.

T8 (:mod:`robo_jev.outside`) stays an outside test: none of its four families (braking deck, turning deck, release over a
support's edge, push), catalogs or wording is used here — only its physics primitives (contact settings, profiles, tracking).
"""
