"""A3's word → range catalogs and the three prior settings (author A3: laboratory, hospital and light manufacturing).

Every hidden quantity is drawn from a range that a **named option** in the work order selects (at least four options per hidden
quantity, all names invented by A3). The hidden quantities, by kind:

* **friction** — ``mu`` (index dial: part base material × plate condition), ``mu_nest`` and ``mu_pad`` (hold-down clamp);
* **centre of mass / mass** — ``kh`` (height fraction) and ``ex`` (offset along the long side, a fraction of its length) on the dial;
  ``load_kg`` (instrument set mass), ``com_frac`` (its centre-of-mass offset along the tray, a fraction of the container length) and
  ``base_kg`` (the stand's base mass); ``force_factor`` (the clamp's actual / set force) is treated like a mass range;
* **placement** — ``dr`` (radial placement error on the dial) and ``dx`` (set-down error along the tray), normal with σ from the
  placement-method word.

**Priors** (:func:`prior_ranges`): ``base`` = the catalog; ``alt_a`` = friction × 0.7, centre-of-mass / mass ranges 1.5 × wider around
the same centre, placement σ × 1.5; ``alt_b`` = friction × 1.3, those ranges half as wide, placement σ × 0.75. One set of uniform
draw points u ∈ [0, 1]^D per group is shared by every prior, variant and both pair members (:func:`draw_values`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "PRIORS",
    "PRIOR_SETTINGS",
    "QUANTITY_KIND",
    "Option",
    "draw_values",
    "prior_ranges",
    "transform_range",
]

PRIORS = ("base", "alt_a", "alt_b")
PRIOR_SETTINGS = {
    "base": "A3 catalog ranges: each named material, condition, contents, build and placement word selects its own range",
    "alt_a": "friction ranges x 0.7; centre-of-mass, mass and force-factor ranges 1.5 x wider around the same centre; placement sigma x 1.5",
    "alt_b": "friction ranges x 1.3; centre-of-mass, mass and force-factor ranges half as wide around the same centre; placement sigma x 0.75",
}
#: Hidden quantity → kind (decides the prior transform).
QUANTITY_KIND = {
    "mu": "friction", "mu_nest": "friction", "mu_pad": "friction",
    "kh": "com", "ex": "com", "com_frac": "com",
    "load_kg": "mass", "base_kg": "mass", "force_factor": "mass",
    "dr": "placement", "dx": "placement",
}
#: Normal draws are clipped at this many σ (a placement error never exceeds 2.5 σ).
CLIP_SIGMA = 2.5


@dataclass(frozen=True)
class Option:
    """A named catalog option: the words shown in the work order and the range (``lo``, ``hi``) or σ it selects."""

    key: str
    words: str
    lo: float = 0.0
    hi: float = 0.0
    sigma: float = 0.0


def _opts(*rows: tuple) -> dict[str, Option]:
    out = {}
    for row in rows:
        if len(row) == 3:
            key, words, sigma = row
            out[key] = Option(key, words, sigma=float(sigma))
        else:
            key, words, lo, hi = row
            out[key] = Option(key, words, float(lo), float(hi))
    return out


# --------------------------------------------------------------------------
# Law A — index dial
# --------------------------------------------------------------------------

#: Part base material → friction on a clean, dry dial plate.
DIAL_BASE = _opts(
    ("bare_alu", "bare machined aluminium", 0.36, 0.52),
    ("cast_iron", "as-cast iron", 0.28, 0.42),
    ("powder_coat", "powder-coated steel", 0.40, 0.58),
    ("gf_nylon", "glass-filled nylon", 0.24, 0.36),
    ("rubber_feet", "rubber feet", 0.70, 0.95),
    ("hard_anod", "hard-anodised aluminium", 0.30, 0.44),
)
#: Plate condition → multiplier on the base-material range; the anti-slip mat replaces the range.
DIAL_PLATE = {
    "dry": ("clean and dry", 1.00),
    "coolant": ("coolant mist on it", 0.72),
    "oil": ("thin oil film", 0.50),
    "dust": ("grinding dust on it", 0.85),
    "mat": ("anti-slip mat fitted", None),
}
DIAL_MAT_RANGE = (0.80, 1.05)
#: Build → centre-of-mass height fraction ``kh`` and offset along the long side ``ex`` (fraction of the long side).
DIAL_BUILD = {
    "solid": ("solid through", (0.46, 0.54), (-0.03, 0.03)),
    "base_weighted": ("base-weighted", (0.28, 0.40), (-0.04, 0.04)),
    "top_loaded": ("top-heavy (actuator on top)", (0.58, 0.72), (-0.06, 0.06)),
    "thin_wall": ("thin-walled shell", (0.40, 0.52), (-0.04, 0.04)),
    "offset_insert": ("off-centre insert, side unknown", (0.42, 0.58), (-0.16, 0.16)),
}
#: Placement method → radial placement error σ (m).
DIAL_PLACEMENT = _opts(
    ("vision", "vision-checked placement", 0.002),
    ("taught", "taught-point placement", 0.005),
    ("by_eye", "hand-placed by eye", 0.012),
    ("chute", "fed in by the chute", 0.020),
)


def dial_mu_range(base: str, plate: str) -> tuple[float, float]:
    if plate == "mat":
        return DIAL_MAT_RANGE
    factor = DIAL_PLATE[plate][1]
    option = DIAL_BASE[base]
    return (round(option.lo * factor, 4), round(option.hi * factor, 4))


# --------------------------------------------------------------------------
# Law B — cantilever instrument stand
# --------------------------------------------------------------------------

#: Stand model → base mass (kg).
STAND_MODEL = _opts(
    ("light_alu", "lightweight aluminium stand", 1.6, 2.6),
    ("std_steel", "standard stainless stand", 3.5, 5.0),
    ("cast_foot", "column stand with a cast foot", 5.5, 7.5),
    ("heavy_base", "heavy-base stand", 7.0, 9.5),
    ("unlabelled", "older stand, no model plate", 2.0, 7.0),
)
#: Instrument set → container mass (kg).
STAND_SET = _opts(
    ("dressing", "suture and dressing pack", 1.5, 2.6),
    ("basic", "basic instrument tray", 3.2, 4.8),
    ("lap", "laparoscopy set with optics", 4.5, 6.5),
    ("major", "major abdominal set", 6.5, 8.8),
    ("ortho", "orthopaedic implant set", 8.5, 11.5),
    ("power", "power-tool kit with batteries", 5.0, 7.0),
)
#: Packing → centre-of-mass offset along the tray (fraction of the container length; + = toward the free end of the tray).
STAND_PACKING = _opts(
    ("even", "packed evenly", -0.03, 0.03),
    ("far_heavy", "heavy pieces at the far end", 0.06, 0.15),
    ("post_heavy", "heavy pieces at the post end", -0.15, -0.06),
    ("unknown", "packing not recorded", -0.12, 0.12),
)
#: Set-down method → error σ along the tray (m).
STAND_METHOD = _opts(
    ("vision", "vision-guided set-down", 0.004),
    ("taught", "taught set-down point", 0.010),
    ("touch_off", "tray-edge touch-off", 0.018),
    ("teleop", "teleoperated set-down", 0.030),
)


# --------------------------------------------------------------------------
# Law C — hold-down clamp
# --------------------------------------------------------------------------

#: Nest surface → part-on-nest friction.
CLAMP_NEST = _opts(
    ("steel_dry", "ground steel, dry", 0.38, 0.52),
    ("steel_oil", "ground steel with an oil film", 0.11, 0.19),
    ("anodised", "anodised aluminium", 0.30, 0.44),
    ("acetal", "acetal (POM) nest", 0.17, 0.27),
    ("urethane", "urethane-lined nest", 0.55, 0.80),
    ("coolant", "steel, wet with coolant", 0.15, 0.25),
)
#: Pad → pad-on-part friction.
CLAMP_PAD = _opts(
    ("steel_button", "polished steel button", 0.14, 0.22),
    ("brass", "brass tip", 0.24, 0.36),
    ("pu", "polyurethane pad", 0.45, 0.65),
    ("serrated", "serrated hardened tip", 0.62, 0.90),
    ("nylon", "nylon cap", 0.16, 0.26),
)
#: Clamp condition → force factor (actual / set).
CLAMP_CONDITION = _opts(
    ("serviced", "serviced this week", 0.93, 1.03),
    ("normal", "in normal service", 0.84, 0.98),
    ("slow", "slow to close", 0.66, 0.84),
    ("leak", "air-leak tag on it", 0.45, 0.68),
)


# --------------------------------------------------------------------------
# Priors and draws
# --------------------------------------------------------------------------


def transform_range(kind: str, spec: Sequence, prior: str) -> list:
    """One hidden quantity's distribution under a prior: ``["u", lo, hi]`` or ``["n", mean, sigma]`` (module docstring)."""
    if prior not in PRIORS:
        raise ValueError(f"prior: one of {PRIORS} (got {prior!r})")
    shape = spec[0]
    if prior == "base":
        return list(spec)
    if kind == "placement":
        if shape != "n":
            raise ValueError("placement quantities are normal")
        return ["n", float(spec[1]), float(spec[2]) * (1.5 if prior == "alt_a" else 0.75)]
    lo, hi = float(spec[1]), float(spec[2])
    if kind == "friction":
        factor = 0.7 if prior == "alt_a" else 1.3
        return ["u", lo * factor, hi * factor]
    if kind in ("com", "mass"):
        centre, half = (lo + hi) / 2, (hi - lo) / 2
        scale = 1.5 if prior == "alt_a" else 0.5
        return ["u", centre - half * scale, centre + half * scale]
    raise ValueError(f"unknown quantity kind {kind!r}")


def prior_ranges(base: Mapping[str, Sequence]) -> dict[str, dict[str, list]]:
    """``{prior: {quantity: spec}}`` from the base specs (every quantity's kind from :data:`QUANTITY_KIND`)."""
    return {prior: {q: transform_range(QUANTITY_KIND[q], spec, prior) for q, spec in base.items()} for prior in PRIORS}


_NORMAL = NormalDist()
_U_CLIP = (_NORMAL.cdf(-CLIP_SIGMA), _NORMAL.cdf(CLIP_SIGMA))


def draw_values(spec: Sequence, u: np.ndarray) -> np.ndarray:
    """Map shared uniform draw points `u` through one quantity's distribution (inverse CDF; normal draws clipped at ±2.5 σ)."""
    u = np.asarray(u, dtype=float)
    if spec[0] == "u":
        return float(spec[1]) + u * (float(spec[2]) - float(spec[1]))
    if spec[0] == "n":
        clipped = np.clip(u, *_U_CLIP)
        z = np.array([_NORMAL.inv_cdf(float(x)) for x in clipped.ravel()]).reshape(clipped.shape)
        return float(spec[1]) + float(spec[2]) * z
    raise ValueError(f"unknown distribution {spec[0]!r}")


def uniform_points(seed: int, k: int, names: Sequence[str]) -> dict[str, np.ndarray]:
    """K shared draw points per hidden quantity (independent uniforms; the order of `names` is fixed per scene)."""
    rng = np.random.default_rng(int(seed))
    u = rng.random((int(k), len(names)))
    return {name: u[:, i].copy() for i, name in enumerate(names)}


def finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))
