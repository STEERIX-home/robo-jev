"""Author A4 — records of the held-out test (``simgen-request/0``, the v0.3b MuJoCo record contract) and the set's selection.

A group (two records) becomes two records ``v06a4-<scene>-<index>`` and ``v06a4-<scene>-<index>p`` (set ``v06-a4``, split ``test``). The
model input is ``request`` only; ``meta`` carries the contract fields (``question_kind``, ``key_question``, ``visible``, ``hidden_prior``,
``prior_settings``, ``variants``, ``plan``, ``outcomes``, ``labels_by_prior``, ``robustness``, ``contrast_direction_*``, ``compare_option``,
``action_options``, ``static_aim``, ``pair``) plus this author's ``key_events`` / ``severe_events`` / ``plan_option`` (what the in-family
comparison reading needs: the outcomes summed for the key event and the plan's option key) and ``law``.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from robo_jev.authors.a4 import design as D
from robo_jev.authors.a4 import label as L
from robo_jev.authors.a4 import radio as R

__all__ = ["GENERATOR", "ID_PREFIX", "SET_NAME", "SPLIT", "group_records", "record_id", "select_groups"]

SET_NAME = "v06-a4"
SPLIT = "test"
ID_PREFIX = "v06a4"
GENERATOR = "v06-a4-field-radio-v1"
STYLE = "radio-transcript"
PRIOR_SETTINGS = {
    "base": "the author's catalog ranges for every hidden quantity",
    "alt_a": "friction and drag ranges x 0.7; centre-of-mass and mass ranges 1.5 x wider around the same centre; placement sigma x 1.5",
    "alt_b": "friction and drag ranges x 1.3; centre-of-mass and mass ranges half as wide around the same centre; placement sigma x 0.75",
}


def record_id(scene: str, index: int, member: int) -> str:
    return f"{ID_PREFIX}-{scene}-{int(index):06d}" + ("p" if member else "")


def group_records(group: dict[str, Any], sim: dict[str, Any], *, git: str | None) -> list[dict[str, Any]]:
    """The two records of a labelled group."""
    scene = group["scene"]
    spec = D.SCENE_SPECS[scene]
    names = R.QUESTION_NAMES[scene]
    ids = (record_id(scene, group["index"], 0), record_id(scene, group["index"], 1))
    out = []
    for member, vis in enumerate(group["records"]):
        state, questions = R.render(group, member)
        labelled = L.record_labels(group, sim, member, names)
        variants = D.resolve_variants(scene, vis, group["options"], group["cf_value"])
        contrast = labelled["contrast_direction_by_prior"]
        option_variants = {v: variants[v] for v in D.OPTIONS[scene].values() if v != "planned"}
        meta = {
            "set": SET_NAME, "split": SPLIT, "scene": scene, "group": scene, "law": D.LAW[scene], "phenomena": list(D.PHENOMENA[scene]),
            "author": D.AUTHOR, "language": "en", "domain": vis["domain"], "style": STYLE, "key_question": names["key"],
            "question_kind": {names[k]: R.QUESTION_KIND[k] for k in ("key", "severe", "score_cf", "direction", "compare", "action", "outcome")},
            "visible": dict(vis), "hidden_prior": D.prior_specs(scene, vis), "prior_settings": dict(PRIOR_SETTINGS), "variants": variants,
            "plan": {"decisive": spec.decisive, "aim_key": group["aims"]["key"] if member == 0 else group["aims"]["partner_key"],
                     "aim_cf": group["aims"]["cf"], "cf_kind": group["cf_kind"], "cf_value": group["cf_value"], "options": dict(D.OPTIONS[scene]),
                     "option_values": option_variants, "option_parameters": group["options"]},
            "outcomes": labelled["outcomes"], "labels_by_prior": labelled["labels_by_prior"], "robustness": labelled["robustness"],
            "contrast_direction_by_prior": contrast, "contrast_direction_same": len(set(contrast.values())) == 1,
            "contrast_no_reversal": not {"up", "down"} <= set(contrast.values()), "compare_option": group["compare_option"],
            "action_options": list(D.OPTIONS[scene]), "static_aim": {"key": group["aims"]["key"] if member == 0 else group["aims"]["partner_key"],
                                                                     "cf": group["aims"]["cf"], "cf_kind": group["cf_kind"]},
            "pair": L.pair_meta(group, sim, member, ids), "key_events": list(D.EVENT[scene]), "severe_events": list(D.SEVERE[scene]),
            "plan_option": D.PLAN_KEY[scene],
        }
        out.append({
            "schema": "simgen-request/0", "id": ids[member],
            "generator": {"name": GENERATOR, "family": scene, "group": int(group["index"]), "seed": int(group["group_seed"]),
                          "draws_seed": int(group["draws_seed"]), "k": D.K, "priors": list(D.PRIORS), "mujoco": mujoco.__version__, "git": git,
                          "member": "base" if member == 0 else "pair"},
            "meta": meta, "request": {"state": state, "questions": questions}, "labels": labelled["labels_by_prior"]["base"],
        })
    return out


def select_groups(candidates: list[dict[str, Any]], per_scene: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """Pick `per_scene` labelled groups of each scene for a flat key profile: candidates are binned by the base record's MuJoCo key
    label (tenths); the bins are filled round-robin in a seeded random order. Groups with an unstable rollout are never picked."""
    rng = np.random.default_rng(seed)
    out = []
    for scene in D.SCENES:
        pool = [c for c in candidates if c["scene"] == scene and not c["unstable"]]
        bins: dict[int, list[dict[str, Any]]] = {}
        for c in pool:
            bins.setdefault(min(9, int(c["key"] * 10)), []).append(c)
        for members in bins.values():
            rng.shuffle(members)
        order = sorted(bins)
        picked: list[dict[str, Any]] = []
        while len(picked) < per_scene and any(bins[b] for b in order):
            for b in order:
                if bins[b] and len(picked) < per_scene:
                    picked.append(bins[b].pop())
        out += sorted(picked, key=lambda c: c["index"])
    return out
