"""T9 — the second outside physical-judgment test (Task T9): MuJoCo-labelled robot moments written by a fresh author, sealed.

* :mod:`~robo_jev.t9.physics` — six small MuJoCo scenes (roll, swing, knock, conveyor, overhang, suction) and their rollouts.
* :mod:`~robo_jev.t9.priors` — word → range catalogs read by three readers (the catalog, and every range shifted down or up by a
  third of its width) and the shared hidden draws.
* :mod:`~robo_jev.t9.families` — catalogs, visible moments, aiming, one-fact changes and the textbook calculators.
* :mod:`~robo_jev.t9.text` — the robot's first-person text and the six System-1 questions.
* :mod:`~robo_jev.t9.build` — candidate groups (a moment and its one-fact partner), MuJoCo labels under the three priors, records,
  robustness and balanced selection.
* :mod:`~robo_jev.t9.sanity`, :mod:`~robo_jev.t9.overlap`, :mod:`~robo_jev.t9.stats`, :mod:`~robo_jev.t9.reference`,
  :mod:`~robo_jev.t9.score` — physics sanity, overlap with the training inputs and T8b, label statistics, the calculator and
  constant references, and the instrument-sanity scoring.

Nothing here imports the engine (:mod:`robo_jev.simgen`), the v0.3b generator (:mod:`robo_jev.mjgen`) or T8b's moments. T9 is
evaluation-only and sealed: never trained on, never sent to any API (:mod:`robo_jev.data.sealed`), and until the next gate used for
instrument sanity only.
"""
