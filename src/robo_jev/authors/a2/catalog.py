"""Author A2 — word → range catalogs, the three prior settings, outcome costs and action costs.

Every hidden quantity is a uniform range picked by the words of the briefing, or a normal placement error:

* **friction** — named materials × named surface conditions (turntable: the item's base × the top × its condition; tiltback: the
  floor; wallpress: the item's back × the wall, and the robot's pad);
* **centre of mass / mass** — contents words (turntable: how full the bottle or vase is → centre-of-mass height as a share of the
  item's height; tiltback: how the piece is loaded → centre-of-mass height share, the depth offset and the added mass; wallpress: the
  build → mass per square metre of the front);
* **placement** — a normal error with a scene-specific σ (turntable: where the item really stands; tiltback: the tilt the robot really
  holds; wallpress: where the item sits under the pad).

**Prior settings** (the standard alternative-prior rule of the v0.6 plan, applied to this catalog): ``base`` = the catalog;
``alt_a`` = friction × 0.7, centre-of-mass and mass ranges 1.5 × wider around the same centre, placement σ × 1.5; ``alt_b`` = friction
× 1.3, those ranges half as wide, placement σ × 0.75.
"""

from __future__ import annotations

from typing import Any

PRIORS = ("base", "alt_a", "alt_b")
FRICTION_FACTOR = {"base": 1.0, "alt_a": 0.7, "alt_b": 1.3}
SPREAD_FACTOR = {"base": 1.0, "alt_a": 1.5, "alt_b": 0.5}
SIGMA_FACTOR = {"base": 1.0, "alt_a": 1.5, "alt_b": 0.75}
PRIOR_SETTINGS = {
    "base": "A2's catalog as written: friction, centre-of-mass and mass ranges from the briefing's words, placement errors at their catalog sigma",
    "alt_a": "slipperier and vaguer: every friction range scaled by 0.7, centre-of-mass and mass ranges 1.5 times as wide about the same "
             "middle, placement sigma times 1.5",
    "alt_b": "grippier and tighter: every friction range scaled by 1.3, centre-of-mass and mass ranges half as wide about the same middle, "
             "placement sigma times 0.75",
}
#: Hidden-quantity kinds: how a prior setting transforms them.
KINDS = ("friction", "spread", "place")


def prior_range(kind: str, lo: float, hi: float, prior: str) -> tuple[float, float]:
    """The range (or, for ``place``, the σ in `hi`) of one hidden quantity under a prior setting."""
    if kind == "friction":
        f = FRICTION_FACTOR[prior]
        return lo * f, hi * f
    if kind == "spread":
        centre, half = (lo + hi) / 2, (hi - lo) / 2 * SPREAD_FACTOR[prior]
        return centre - half, centre + half
    if kind == "place":
        return 0.0, hi * SIGMA_FACTOR[prior]
    raise ValueError(f"unknown hidden kind {kind!r}")


# --------------------------------------------------------------------------
# A — turntable (lazy Susan)
# --------------------------------------------------------------------------

#: key → name, family, glass or ceramic, base side a (cm), tangential/radial ratio, height (cm), empty mass (kg), allowed bases.
TT_ITEMS: dict[str, dict[str, Any]] = {
    "soy": {"name": "square soy-sauce bottle", "family": "bottle", "side": (5.5, 7.0), "ratio": (1.0, 1.0), "height": (19, 25),
            "shell": 0.30, "bases": ("glass", "felt", "rubber", "cork")},
    "chili": {"name": "square chili-oil bottle", "family": "bottle", "side": (5.0, 6.5), "ratio": (1.0, 1.0), "height": (17, 22),
              "shell": 0.25, "bases": ("glass", "felt", "rubber", "cork")},
    "oil": {"name": "square olive-oil bottle", "family": "bottle", "side": (5.5, 7.0), "ratio": (1.0, 1.0), "height": (27, 32),
            "shell": 0.40, "bases": ("glass", "felt", "rubber", "cork")},
    "vinegar": {"name": "tall black-vinegar bottle", "family": "bottle", "side": (5.0, 6.0), "ratio": (1.0, 1.0), "height": (27, 33),
                "shell": 0.35, "bases": ("glass", "felt", "rubber")},
    "carafe": {"name": "square glass carafe", "family": "bottle", "side": (8.5, 10.5), "ratio": (1.0, 1.0), "height": (22, 27),
               "shell": 0.55, "bases": ("glass", "felt", "cork")},
    "jug": {"name": "square juice jug", "family": "bottle", "side": (9.5, 11.5), "ratio": (1.0, 1.0), "height": (23, 28),
            "shell": 0.60, "bases": ("glass", "rubber", "cork")},
    "budvase": {"name": "slim boxy bud vase", "family": "vase", "side": (5.0, 6.5), "ratio": (1.0, 1.0), "height": (18, 24),
                "shell": 0.30, "bases": ("glaze", "bisque", "felt")},
    "tallvase": {"name": "tall boxy glass vase", "family": "vase", "side": (8.0, 10.5), "ratio": (1.0, 1.0), "height": (26, 34),
                 "shell": 0.80, "bases": ("glass", "felt", "rubber")},
    "ceramicvase": {"name": "boxy ceramic vase", "family": "vase", "side": (9.0, 11.0), "ratio": (1.0, 1.0), "height": (20, 26),
                    "shell": 0.90, "bases": ("glaze", "bisque", "cork")},
    "flowerbox": {"name": "low rectangular flower box", "family": "vase", "side": (14.0, 18.0), "ratio": (0.45, 0.6), "height": (11, 15),
                  "shell": 0.70, "bases": ("glaze", "bisque", "felt", "cork")},
}
#: Contents → (words, centre-of-mass height as a share of the height, fill share for the mass).
TT_CONTENTS: dict[str, dict[str, tuple[str, tuple[float, float], float]]] = {
    "bottle": {
        "full": ("full to the neck", (0.40, 0.48), 0.90),
        "half": ("about half full", (0.27, 0.35), 0.50),
        "dregs": ("nearly empty", (0.37, 0.46), 0.08),
        "pourer": ("topped up, with a heavy metal pourer in the neck", (0.46, 0.56), 0.92),
    },
    "vase": {
        "lilies": ("holding a big bunch of lilies", (0.55, 0.68), 0.35),
        "roses": ("with a few short roses and plenty of water", (0.30, 0.40), 0.60),
        "empty": ("empty", (0.40, 0.48), 0.0),
        "pebbles": ("with pebbles in the bottom and a couple of stems", (0.22, 0.31), 0.45),
    },
}
#: The item's base → friction range on a clean, dry lacquered-wood top.
TT_BASES = {
    "glass": ("a bare glass bottom", (0.32, 0.45)),
    "felt": ("a felt pad stuck underneath", (0.24, 0.34)),
    "rubber": ("a rubber ring on the bottom", (0.75, 1.0)),
    "glaze": ("a glazed foot", (0.26, 0.38)),
    "bisque": ("an unglazed ceramic rim", (0.48, 0.66)),
    "cork": ("a cork pad underneath", (0.52, 0.72)),
}
#: The turntable top → friction factor.
TT_TOPS = {"glass": ("glass", 0.90), "lacquer": ("lacquered wood", 1.00), "stone": ("polished stone", 0.85), "bamboo": ("bamboo", 1.10)}
#: The top's condition → friction factor.
TT_CONDITIONS = {
    "dry": ("clean and dry", 1.00),
    "damp": ("still damp from wiping", 0.82),
    "greasy": ("a bit greasy from the last sitting", 0.62),
    "sticky": ("sticky where syrup got spilled", 1.30),
}
#: Turn → (degrees, words).
TT_TURNS = {"quarter": (90.0, "a quarter turn"), "third": (120.0, "a third of a turn"), "half": (180.0, "half a turn")}
#: The same seat reached the other way round (the "go_the_other_way" alternative).
TT_OTHER_WAY = {"quarter": "threequarter", "third": "twothird", "half": "half"}
TT_LONG_TURNS = {"threequarter": (270.0, "three quarters of a turn"), "twothird": (240.0, "two thirds of a turn")}
#: Where the item really stands: normal radial error σ (m).
TT_PLACE_SIGMA = 0.015
TT_RADIUS_CM = (12, 34)
TT_DURATION = (0.4, 6.0)
TT_SLOWER = 1.8
TT_INWARD_CM = 9
TT_OUTCOMES = ("stays", "skids", "topples")
TT_KEY = ("skids", "topples")
TT_SEVERE = ("topples",)
TT_COSTS = {"stays": 0.0, "skids": 1.0, "topples": 4.0}
TT_ACTION_COSTS = {"turn_as_planned": 0.0, "turn_slower": 0.15, "move_it_inward": 0.25, "go_the_other_way": 0.0, "carry_it_round": 0.70}
TT_VENUES = {
    "restaurant": ("table 12 in the banquet room", "the big round table by the window", "table 4 in the dim sum hall",
                   "the round table in the private room"),
    "home": ("the family's dinner table", "the round kitchen table", "the dining table at the Parks' place"),
    "hotel": ("a round table in the breakfast room", "the group table in the hotel restaurant", "the round table in the suite's dining nook"),
}

# --------------------------------------------------------------------------
# B — tiltback
# --------------------------------------------------------------------------

#: key → name, depth / width / height ranges (m), centre-of-mass depth offset range (share of the depth; + = towards the front),
#: allowed builds, contents words per contents key, what gets slid under it.
TB_PIECES: dict[str, dict[str, Any]] = {
    "bookcase": {"name": "bookcase", "depth": (0.28, 0.36), "width": (0.75, 1.0), "height": (1.6, 2.0), "cx": (-0.05, 0.05),
                 "builds": ("oak", "pine", "chipboard"),
                 "contents": {"top": "top shelves full of hardbacks, bottom ones bare",
                              "even": "books spread over every shelf",
                              "bottom": "heavy albums on the bottom shelves, little up top",
                              "empty": "every shelf empty"}},
    "wardrobe": {"name": "wardrobe", "depth": (0.52, 0.62), "width": (0.9, 1.2), "height": (1.8, 2.05), "cx": (-0.05, 0.05),
                 "builds": ("oak", "pine", "chipboard"),
                 "contents": {"top": "suitcases crammed on the top shelf",
                              "even": "clothes hanging, a few boxes up top",
                              "bottom": "shoes and a toolbox in the bottom, nothing up top",
                              "empty": "cleared out"}},
    "tallboy": {"name": "tallboy", "depth": (0.40, 0.50), "width": (0.6, 0.85), "height": (1.2, 1.5), "cx": (-0.05, 0.05),
                "builds": ("oak", "pine", "chipboard"),
                "contents": {"top": "top drawers stuffed, bottom ones empty",
                             "even": "every drawer about half full",
                             "bottom": "bottom drawers full of jeans, top ones empty",
                             "empty": "all drawers empty"}},
    "display": {"name": "display cabinet", "depth": (0.36, 0.45), "width": (0.8, 1.0), "height": (1.7, 1.95), "cx": (-0.05, 0.05),
                "builds": ("oak", "pine"),
                "contents": {"top": "the good china on the upper shelves",
                             "even": "glassware on every shelf",
                             "bottom": "heavy serving dishes on the bottom shelf only",
                             "empty": "shelves cleared"}},
    "fridge": {"name": "upright drinks fridge", "depth": (0.55, 0.62), "width": (0.55, 0.62), "height": (1.4, 1.85),
               "cx": (-0.14, -0.06), "builds": ("steel",),
               "contents": {"top": "top racks full of wine, lower ones empty",
                            "even": "fully stocked with cans and bottles",
                            "bottom": "beer crates in the bottom only",
                            "empty": "switched off and empty"}},
    "linen": {"name": "linen cupboard", "depth": (0.45, 0.55), "width": (0.8, 1.0), "height": (1.8, 2.0), "cx": (-0.05, 0.05),
              "builds": ("oak", "pine", "chipboard"),
              "contents": {"top": "spare duvets packed on the top shelves",
                           "even": "towels on every shelf",
                           "bottom": "a rolled mattress topper in the bottom only",
                           "empty": "nothing in it"}},
    "shoecab": {"name": "shoe cabinet", "depth": (0.24, 0.32), "width": (0.8, 1.05), "height": (1.0, 1.3), "cx": (-0.05, 0.05),
                "builds": ("oak", "pine", "chipboard"),
                "contents": {"top": "winter boots on the top tier",
                             "even": "shoes on every tier",
                             "bottom": "heavy boots on the bottom tier only",
                             "empty": "empty"}},
}
#: Contents key → centre-of-mass height share and added mass per cubic metre of envelope (kg/m³). The fridge has its own rows (its
#: compressor sits at the bottom).
TB_CONTENTS = {"top": ((0.56, 0.68), (60.0, 110.0)), "even": ((0.46, 0.54), (50.0, 100.0)), "bottom": ((0.32, 0.42), (60.0, 110.0)),
               "empty": ((0.47, 0.53), (0.0, 0.0))}
TB_FRIDGE_CONTENTS = {"top": ((0.46, 0.56), (70.0, 120.0)), "even": ((0.40, 0.48), (90.0, 140.0)), "bottom": ((0.30, 0.38), (80.0, 130.0)),
                      "empty": ((0.36, 0.44), (0.0, 0.0))}
#: Build → (words, empty mass per cubic metre of envelope, kg/m³).
TB_BUILDS = {"oak": ("solid oak", (75.0, 95.0)), "pine": ("painted pine", (50.0, 65.0)), "chipboard": ("flat-pack chipboard", (38.0, 52.0)),
             "steel": ("steel-bodied", (105.0, 130.0))}
#: Floor → (words, friction range).
TB_FLOORS = {
    "carpet": ("short-pile carpet", (0.65, 0.90)),
    "parquet": ("polished parquet", (0.28, 0.40)),
    "vinyl": ("vinyl plank", (0.38, 0.52)),
    "tile": ("glazed tiles", (0.30, 0.45)),
    "wettile": ("freshly mopped tiles", (0.14, 0.24)),
    "rug": ("a thin rug", (0.50, 0.70)),
}
TB_LIMITS = (15.0, 20.0, 30.0, 40.0, 60.0, 90.0)
#: The tilt the robot really holds: normal error σ (degrees).
TB_PLACE_SIGMA = 1.0
TB_TILT = (2.0, 32.0)
TB_TASKS = ("get the new rug under it", "put felt pads under the front", "slip sliders under the front", "pull out the old carpet gripper",
            "shim the front level", "fish out what rolled under it")
TB_OUTCOMES = ("rests", "hangs", "over")
TB_KEY = ("hangs", "over")
TB_SEVERE = ("over",)
TB_COSTS = {"rests": 0.0, "hangs": 1.2, "over": 6.0}
TB_ACTION_COSTS = {"tip_as_planned": 0.0, "empty_top_first": 0.40, "tip_it_sideways": 0.15, "load_top_first": 0.0, "use_the_lifter": 0.60}
TB_VENUES = {
    "home": ("the living room", "the spare bedroom", "the hallway", "the kids' room"),
    "hotel": ("room 214", "the guest lounge", "the corner suite", "the staff changing room"),
    "restaurant": ("the back office", "the bar area", "the private dining room"),
}

# --------------------------------------------------------------------------
# C — wallpress
# --------------------------------------------------------------------------

#: key → name, thickness / width / height ranges (m), builds, allowed backs.
WP_ITEMS: dict[str, dict[str, Any]] = {
    "mirror": {"name": "bathroom mirror", "thick": (0.006, 0.010), "width": (0.40, 0.70), "height": (0.50, 0.90),
               "builds": ("plain", "bevel", "slimframe", "carved"), "backs": ("glassback", "felt", "cork", "rubber")},
    "print": {"name": "framed print", "thick": (0.020, 0.035), "width": (0.40, 0.80), "height": (0.50, 1.00),
              "builds": ("alu", "oakglass", "gilt", "canvas"), "backs": ("paper", "felt", "cork", "mdf", "rubber")},
    "pinboard": {"name": "pinboard", "thick": (0.015, 0.025), "width": (0.60, 0.90), "height": (0.45, 0.60),
                 "builds": ("thincork", "mdfcork", "foam", "steelboard"), "backs": ("mdf", "felt", "cork", "rubber")},
    "cabinet": {"name": "small bathroom cabinet", "thick": (0.12, 0.16), "width": (0.40, 0.60), "height": (0.50, 0.70),
                "builds": ("cabempty", "cabhalf", "cabfull", "cabmirror"), "backs": ("mdf", "felt", "rubber")},
    "headboard": {"name": "headboard panel", "thick": (0.05, 0.08), "width": (0.90, 1.60), "height": (0.50, 0.70),
                  "builds": ("upholstered", "timber", "velvet", "rattan"), "backs": ("fabric", "mdf", "felt", "rubber")},
    "clock": {"name": "square wall clock", "thick": (0.04, 0.06), "width": (0.35, 0.50), "height": (0.35, 0.50),
              "builds": ("clockwood", "clockmetal", "clockplastic", "clockstone"), "backs": ("mdf", "felt", "cork")},
}
#: Build → (words, mass per square metre of the front, kg/m²).
WP_BUILDS = {
    "plain": ("plain 4 mm glass, no frame", (9.5, 11.5)),
    "bevel": ("6 mm bevelled glass", (14.5, 17.5)),
    "slimframe": ("glass in a slim timber surround", (17.0, 22.0)),
    "carved": ("glass in a heavy carved frame", (25.0, 34.0)),
    "alu": ("a thin aluminium frame with acrylic glazing", (4.0, 7.0)),
    "oakglass": ("a chunky oak frame with real glass", (9.0, 13.0)),
    "gilt": ("an ornate gilt frame with glass", (14.0, 22.0)),
    "canvas": ("canvas on a wooden stretcher, no glass", (2.5, 4.0)),
    "thincork": ("cork on a thin backing board", (4.0, 6.0)),
    "mdfcork": ("cork on thick MDF", (9.0, 13.0)),
    "foam": ("fabric over foam board", (2.0, 3.5)),
    "steelboard": ("a magnetic steel board", (12.0, 16.0)),
    "cabempty": ("empty", (25.0, 32.0)),
    "cabhalf": ("about half full of toiletries", (35.0, 45.0)),
    "cabfull": ("packed with bottles and jars", (50.0, 65.0)),
    "cabmirror": ("empty but with heavy mirrored doors", (33.0, 41.0)),
    "upholstered": ("upholstered foam on plywood", (8.0, 12.0)),
    "timber": ("solid timber", (18.0, 26.0)),
    "velvet": ("deep-buttoned velvet on a frame", (10.0, 14.0)),
    "rattan": ("rattan on a light frame", (5.0, 8.0)),
    "clockwood": ("a wooden case", (7.0, 10.0)),
    "clockmetal": ("a brushed-metal case", (11.0, 15.0)),
    "clockplastic": ("a plastic case", (3.5, 5.5)),
    "clockstone": ("a slate face", (20.0, 28.0)),
}
#: The item's back → friction range against matt-painted plaster.
WP_BACKS = {
    "felt": ("felt bumpers", (0.38, 0.52)),
    "mdf": ("bare painted board", (0.30, 0.44)),
    "cork": ("cork pads", (0.52, 0.70)),
    "rubber": ("rubber bumpers", (0.78, 1.00)),
    "paper": ("a brown-paper dust cover", (0.25, 0.38)),
    "glassback": ("smooth painted glass", (0.20, 0.30)),
    "fabric": ("plain fabric", (0.45, 0.62)),
}
#: The wall → friction factor.
WP_WALLS = {
    "matt": ("matt-painted plaster", 1.00),
    "gloss": ("gloss-painted plaster", 0.85),
    "tile": ("glazed tiles", 0.72),
    "steamy": ("tiles still steamy from a shower", 0.55),
    "vinyl": ("vinyl wallpaper", 0.90),
    "textured": ("textured wallpaper", 1.15),
    "panel": ("varnished wood panelling", 0.88),
    "brick": ("painted brick", 1.25),
}
#: The robot's pad → (words, friction range).
WP_PADS = {
    "foam": ("the foam pad", (0.55, 0.70)),
    "silicone": ("the silicone pad", (0.80, 1.00)),
    "feltpad": ("the felt-faced pad", (0.35, 0.48)),
    "plastic": ("the bare plastic paddle", (0.20, 0.30)),
}
#: The rubber sheet of the "rubber_sheet_behind" alternative → friction range against matt plaster (times the wall factor).
WP_RUBBER_SHEET = (0.80, 1.05)
WP_REFLEX = (1.5, 2.0, 2.5, 3.0)
#: Where the item really sits under the pad: normal error σ (m) of each of the two offsets.
WP_PLACE_SIGMA = 0.02
WP_FORCE = (8.0, 700.0)
WP_OUTCOMES = ("holds", "slips", "drops")
WP_KEY = ("slips", "drops")
WP_SEVERE = ("drops",)
WP_COSTS = {"holds": 0.0, "slips": 0.8, "drops": 5.0}
WP_ACTION_COSTS = {"hold_as_planned": 0.0, "use_both_pads": 0.10, "rubber_sheet_behind": 0.20, "felt_pad_instead": 0.0, "batten_first": 0.55}
WP_VENUES = {
    "home": ("the upstairs bathroom", "the hallway", "the living room", "the main bedroom"),
    "hotel": ("room 308's bathroom", "the lobby", "the corridor by the lifts", "the honeymoon suite"),
    "restaurant": ("the dining room", "the entrance", "the staff toilets", "the private room"),
}
WP_HELPERS = ("Sam", "Priya", "Tomas", "Ana", "Jun", "the handyman", "Leila", "Marco")
