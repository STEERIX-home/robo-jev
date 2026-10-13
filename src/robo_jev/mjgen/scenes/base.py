"""v0.4a scene interface — what every new scene provides (Task v0.4a §1.1).

A :class:`Scene` is one situation of one physical law: its catalogs (the priors behind the words), the visible facts it samples, the
hidden values the words imply, a closed-form (static) outcome used only to aim the sampled numbers, its MuJoCo rollout, its action
options, its one-number and one-factor pairs, and its text (four narration styles and the question wording). The generic pipeline
(:mod:`robo_jev.mjgen.scenes.build`) turns a scene into labelled groups and ``simgen-request/0`` records exactly as v0.3b did for
tilt, grip and tower (same question kinds, label rules, priors, K and pair design).
"""

from __future__ import annotations

import json
import random
from typing import Any, Callable

__all__ = ["LAWS", "Scene", "a_or_an", "cm", "num"]

#: the three physical laws and their v0.3b scene (counted as each law's fourth scene; not regenerated here)
LAWS = {
    "tip_slide": {"what": "tip or slide under acceleration or tilt", "v03b": "tilt"},
    "com_edge": {"what": "centre of mass past an edge", "v03b": "tower"},
    "friction_hold": {"what": "friction-limited holding", "v03b": "grip"},
}


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def cm(value: float) -> str:
    out = round(value * 100, 1)
    return f"{out:.0f} cm" if abs(out - round(out)) < 1e-9 else f"{out:.1f} cm"


def a_or_an(word: str) -> str:
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


class Scene:
    """Base class of a v0.4a scene; subclasses fill in the attributes and hooks below."""

    name: str = ""
    law: str = ""
    #: raw outcome codes (the "what happens" question's order), the key event (a failure) and its severe part
    outcomes: tuple[str, ...] = ()
    event: tuple[str, ...] = ()
    severe: str = ""
    #: question kind → question name, answer key → raw outcome, the compare question's key for the planned action
    question_names: dict[str, str] = {}
    outcome_keys: dict[str, str] = {}
    planned_key: str = "as_planned"
    #: an option that is safest in most groups by construction: left out of a seeded half of the safest-action questions (v0.3b rule)
    dominant_option: str | None = None
    #: one-factor pairs: factor → {"field", "fact": "word"|"number", "how": "word"|"aim"|"choice"}; "aim" also has "span"
    factors: dict[str, dict[str, Any]] = {}
    #: far-aimed one-number pairs from the base run: (field, share); the decisive number uses its own span
    pair_fields: tuple[tuple[str, float], ...] = ()
    #: numeric pair spans (field → callable(current, visible) → (lo, hi, step))
    pair_spans: dict[str, Callable[[float, dict[str, Any]], tuple[float, float, float]]] = {}
    domains: tuple[str, ...] = ()

    # -- sampling and priors -------------------------------------------------------------------------------------------------------

    def sample(self, rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
        """Visible facts (the decisive number still unset) and the decisive spec ``{"field", "lo", "hi", "step", "increasing"}``."""
        raise NotImplementedError

    def hidden_for(self, visible: dict[str, Any]) -> dict[str, Any]:
        """Base hidden specs implied by the stated facts (kinds as :func:`robo_jev.mjgen.families.prior_ranges`)."""
        raise NotImplementedError

    def static_outcome(self, visible: dict[str, Any], draw: dict[str, float]) -> str:
        """Closed-form outcome of one hidden draw (quasi-static textbook rules) — used to aim sampled numbers, never as a label."""
        raise NotImplementedError

    def accept(self, visible: dict[str, Any]) -> bool:
        """A last check of a sampled scene (e.g. the planned motion is not too long to roll out)."""
        return True

    def options(self, group: Any, rng: random.Random) -> None:
        """Add the two action options (``group.variants["opt_…"]``, ``group.plan["options"]`` = answer key → variant name)."""
        raise NotImplementedError

    def factor_choices(self, group: Any, factor: str, current: Any) -> list[Any]:
        """Alternative values of a word or choice factor (empty when none applies to this scene)."""
        return []

    # -- physics -------------------------------------------------------------------------------------------------------------------

    def physics_key(self, visible: dict[str, Any]) -> str:
        """Everything the rollout depends on besides the hidden draw (variants with equal keys are rolled once)."""
        return json.dumps({k: visible.get(k) for k in self.physics_fields}, sort_keys=True)

    physics_fields: tuple[str, ...] = ()

    def roll(self, visible: dict[str, Any], draw: dict[str, float], rigs: dict[str, Any]) -> dict[str, Any]:
        """One MuJoCo rollout → at least ``{"outcome": <raw code>, "unstable": bool}``."""
        raise NotImplementedError

    def xml_example(self, visible: dict[str, Any]) -> str:
        raise NotImplementedError

    # -- text ----------------------------------------------------------------------------------------------------------------------

    def state_text(self, visible: dict[str, Any], style: str, rng: random.Random) -> str:
        raise NotImplementedError

    def questions(self, visible: dict[str, Any], mix: list[str], plan: dict[str, Any], person: str, rng: random.Random, cf_option: str,
                  action_keys: list[str]) -> dict[str, Any]:
        raise NotImplementedError

    # -- catalogs (manifest, overlap checks) -----------------------------------------------------------------------------------------

    def catalogs(self) -> dict[str, Any]:
        raise NotImplementedError

    def catalog_names(self) -> set[str]:
        """Lower-cased object, surface and robot names of this scene's catalogs."""
        raise NotImplementedError
