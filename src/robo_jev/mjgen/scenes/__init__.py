"""v0.4a — MuJoCo-labelled training scenes: the same three physical laws across many scenes (Task v0.4a).

The variable of v0.4 is scene diversity only: question kinds, label conventions (base-prior labels, three priors with v0.3b's
× 0.7 / × 1.3 rule, K = 32, flat key labels), one-factor pairs, the four templated narration styles, English and text length are
v0.3b's (:mod:`robo_jev.mjgen`). New here: nine scenes, three per law, each with its own MJCF geometry, catalogs and wording.

* :mod:`~robo_jev.mjgen.scenes.rigs` — four physics engines (platform, squeeze, ledge, shift);
* :mod:`~robo_jev.mjgen.scenes.law_a`, :mod:`~robo_jev.mjgen.scenes.law_b`, :mod:`~robo_jev.mjgen.scenes.law_c` — the scenes;
* :mod:`~robo_jev.mjgen.scenes.build` — candidates, pairs, labels, records, selection, statistics;
* :mod:`~robo_jev.mjgen.scenes.sanity` — monotonicity sweeps, analytic thresholds, determinism, one-factor sweeps.

**Internal scene holdout (H-set).** :data:`HOLDOUT_SCENES` — chosen before any labelling, two scenes of two different laws whose robots
appear in no training scene — are never written to the training file; they form ``artifacts/simgen/v04/scene-holdout.jsonl``.

Excluded scene types (T9 and T8b, the sealed outside tests): belts and conveyors; anything over a shelf edge, or a board, plank or tray
cantilevered with an item on it; suction or vacuum cups; rolling round items; anything hanging on a strap or hook; strikes, knocks or
impacts; a flat robot deck braking or turning with a free load; release over a stack edge; a bumper push. Nothing here imports the engine
(:mod:`robo_jev.simgen`), T8b's moments or T9's catalogs and text.
"""

from __future__ import annotations

from robo_jev.mjgen.scenes.base import LAWS, Scene
from robo_jev.mjgen.scenes.law_a import BinScene, ForksScene, TrayScene
from robo_jev.mjgen.scenes.law_b import CabinetScene, ReachScene, TableScene
from robo_jev.mjgen.scenes.law_c import ClampScene, PalmsScene, PlateScene

SCENES: dict[str, Scene] = {scene.name: scene for scene in (ForksScene(), BinScene(), TrayScene(), TableScene(), ReachScene(), CabinetScene(),
                                                            PlateScene(), ClampScene(), PalmsScene())}

#: the internal scene holdout: one tip-or-slide scene and one friction-holding scene, never trained on
HOLDOUT_SCENES = ("tray", "palms")
#: the new training scenes (the v0.3b scene of each law — tilt, tower, grip — is added at assembly)
TRAIN_SCENES = tuple(name for name in SCENES if name not in HOLDOUT_SCENES)

__all__ = ["HOLDOUT_SCENES", "LAWS", "SCENES", "TRAIN_SCENES", "Scene"]
