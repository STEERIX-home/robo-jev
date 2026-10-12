"""Task E1 — one pre-declared 2×2 isolation experiment: why does situation reading collapse outside the engine?

The binding design is the section "PRE-DECLARED ISOLATION EXPERIMENT E1" of ``.superpowers/sdd/progress.md`` (2026-10-12): two
hypotheses — H1 the description format, H2 the physical factors / physics source — and four paired cells:

* :mod:`~robo_jev.e1.engine` — **A**, fresh engine moments (P3 tipping/falling scenes, v0b phrasing, new seeds, not sealed) in the
  engine's format, restricted to their tipping/falling yes/no and score questions;
* :mod:`~robo_jev.e1.rewrite` — **B**, the same moments rewritten by an LLM (OpenRouter) into first-person narration by the robot,
  every number and unit kept (checked; failures dropped);
* :mod:`~robo_jev.e1.render` — **C**, T8b moments rendered deterministically (no LLM) from ``meta.visible`` in the engine's prose
  style and units; **D** is T8b itself, restricted to the same questions;
* :mod:`~robo_jev.e1.analysis` — situation reading (within-question Pearson r, pooled over question groups), paired bootstrap, and
  the pre-declared interpretation rule applied mechanically.

Evaluation only: nothing here trains, and no T8 or sealed text is ever sent to an API (C and D are scored locally).
"""
