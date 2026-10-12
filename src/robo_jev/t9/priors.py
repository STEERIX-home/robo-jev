"""T9 priors — word → range catalogs read by three readers, and the shared hidden draws (Task T9 §1-1 "Hidden parameters and priors").

**What is hidden.** Every hidden quantity of a moment comes from a *word* in its text (a surface, a material, the contents, a strap,
the pump's state): the family catalogs (:mod:`robo_jev.t9.families`) map each word to a range. Tolerances stated as numbers in the text
(a placement accuracy "to within 1 cm", a mass "about 1.2 kg" ± 8 %) are hidden draws too, but every reader takes the stated number
at face value — they are ``fixed`` ranges, the same under every prior.

**Three readers (priors).** ``base`` reads every word as the catalog says. The two alternative priors follow one rule, T9's own:

* ``low_reader`` — every word range [lo, hi] moves **down by a third of its width**: [lo − w/3, hi − w/3];
* ``high_reader`` — every word range moves **up by a third of its width**: [lo + w/3, hi + w/3];

each clipped to the quantity's physical bounds (:data:`BOUNDS`). The width is kept, so the readers disagree about the level a word
implies, not about how sure it is; stated numbers are not re-read. (v0.3b scaled friction ×0.7 / ×1.3 and widened or narrowed the
mass ranges; T9 shifts every word range by the same share of its own width.)

**Shared draws.** One scrambled Latin hypercube in the unit cube per moment group (:func:`unit_draws`, K points, one column per hidden
quantity in sorted name order), mapped through each prior's ranges (:func:`draws_for`): every prior, every variant of a moment and both
members of a one-fact pair see the same K quantile worlds, so differences between them are not sampling noise.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

__all__ = ["BOUNDS", "PRIORS", "SHIFT", "draws_for", "prior_ranges", "unit_draws"]

PRIORS = ("base", "low_reader", "high_reader")
#: Share of a word range's width by which the alternative readers move it.
SHIFT = 1.0 / 3.0
#: Physical bounds of every hidden quantity (clip limits for the alternative readers).
BOUNDS: dict[str, tuple[float, float]] = {
    "mu": (0.05, 1.5),
    "mu_roll": (0.0, 0.02),
    "k": (0.05, 1.0),
    "stretch": (0.0, 0.2),
    "slop": (-0.05, 0.05),
    "com": (0.1, 0.9),
    "hc": (0.12, 0.88),
    "cx": (-0.35, 0.35),
    "board_density": (150.0, 1500.0),
    "item_scale": (0.3, 3.0),
    "icx": (-0.4, 0.4),
    "err": (-0.2, 0.2),
    "vacuum": (2000.0, 95000.0),
    "mass_scale": (0.3, 3.0),
}


def prior_ranges(words: Mapping[str, Sequence[float]], prior: str) -> dict[str, tuple[float, float]]:
    """Word ranges of one moment as read by `prior` (module docstring)."""
    if prior not in PRIORS:
        raise ValueError(f"prior: one of {PRIORS} (got {prior!r})")
    out = {}
    for name, (lo, hi) in words.items():
        lo, hi = float(lo), float(hi)
        if prior != "base":
            shift = (hi - lo) * SHIFT * (-1.0 if prior == "low_reader" else 1.0)
            low, high = BOUNDS[name]
            lo, hi = min(max(lo + shift, low), high), min(max(hi + shift, low), high)
        out[name] = (lo, hi)
    return out


def unit_draws(names: Sequence[str], k: int, seed: int) -> np.ndarray:
    """K × len(names) points of a scrambled Latin hypercube (one column per name, in the given order)."""
    sampler = qmc.LatinHypercube(d=max(1, len(names)), scramble=True, seed=int(seed) % (2**32))
    return sampler.random(int(k))[:, : len(names)]


def draws_for(words: Mapping[str, Sequence[float]], fixed: Mapping[str, Sequence[float]], prior: str, units: np.ndarray,
              names: Sequence[str]) -> list[dict[str, float]]:
    """The K hidden draws of a moment under `prior`: column j of `units` maps uniformly onto the range of ``names[j]`` (word ranges
    as read by the prior, fixed ranges as stated)."""
    ranges: dict[str, tuple[float, float]] = {**prior_ranges(words, prior), **{n: (float(a), float(b)) for n, (a, b) in fixed.items()}}
    missing = [n for n in names if n not in ranges]
    if missing:
        raise ValueError(f"no range for hidden quantities {missing}")
    out: list[dict[str, Any]] = []
    for row in units:
        out.append({n: ranges[n][0] + float(u) * (ranges[n][1] - ranges[n][0]) for n, u in zip(names, row)})
    return out
