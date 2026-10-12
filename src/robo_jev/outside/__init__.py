"""Outside tests (Task T8) — physical judgment measured on data our own simulation engine did not make.

* :mod:`~robo_jev.outside.piqa` — T8a: public PIQA validation (goal + two solutions) as decision requests.
* :mod:`~robo_jev.outside.physics` — T8b physics: rigid-body rollouts in MuJoCo (a load on a braking or turning
  robot deck, a release on top of a support, a push at a chosen height).
* :mod:`~robo_jev.outside.moments` — T8b moments: visible scene + hidden-parameter priors, labels as outcome
  frequencies over K MuJoCo rollouts, and request text written for T8 (no phrase banks from :mod:`robo_jev.simgen`).
* :mod:`~robo_jev.outside.score` — scoring tables: T8a accuracy, T8b Brier / physics skill, paired intervals.

Everything here is evaluation-only and sealed: no training configuration points at these files.
"""
