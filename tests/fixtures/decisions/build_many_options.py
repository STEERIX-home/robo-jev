#!/usr/bin/env python
"""Writes many-options.jsonl: two handwritten requests with a 21-option and a 26-option choice question (Task rel07).

They are not test data of any model run and have no labels — they check that both runners (``robo_jev.infer.RoboJev`` and the Hub's
``robo_jev_decide.py``) accept and answer choices beyond 8 options, and they are the release's many-option examples (equivalence run,
model card, ``scripts/export_hf.py verify-hub``). Each record carries the many-option opt-in (``meta.option_range``), so the diagnosis tools
read it too.

    python tests/fixtures/decisions/build_many_options.py
"""

import json
from pathlib import Path

MARK = {"version": "options-26-1", "max": 26}
GENERATOR = {"version": "fixture-rel07", "catalog": "handwritten", "values": None, "seed": None, "git": None}

PLATE_STATE = """You are a lab robot arm at a liquid-handling bench, holding a 96-well plate in your gripper 15 cm above the deck.
- The plate: standard footprint 128 x 86 mm, no lid, every well about 80 % full of water-based buffer (camera estimate).
- Your gripper holds the plate level; you move it at 0.25 m/s between deck positions.
- The scheduler asks you to set this plate down within the next 10 s so the next run can start.
- The deck positions below are as the deck camera saw them 1 s ago.
- A technician stands at the far end of the bench, 1.2 m away, not reaching toward the deck."""

PLATE_OPTIONS = {
    "nest_a1": "Deck nest A1: empty, level, plate nest with all four corner guides.",
    "nest_a2": "Deck nest A2: another plate is already in it.",
    "nest_a3": "Deck nest A3: empty and level, but one corner guide is broken off.",
    "nest_b1": "Deck nest B1: empty and level; a used pipette tip lies inside the nest.",
    "magnet": "Magnetic bead stand: empty, level, plate nest with guides.",
    "sealer": "Plate sealer input tray: empty, level, ready.",
    "reader_tray": "Plate reader tray: extended, empty, level.",
    "shaker_on": "Shaker platform: running at 600 rpm right now.",
    "shaker_off": "Shaker platform: stopped, clamps open.",
    "heater_hot": "Heating block: at 95 °C.",
    "cooler": "Cooling block at 4 °C; the next run needs the plate at room temperature.",
    "stack_top": "On top of a stack of four lidded plates, no nest.",
    "ledge": "Shelf ledge 60 mm deep (narrower than the plate).",
    "tilted_rack": "Rack slot tilted about 15 degrees (worn rails).",
    "centrifuge": "Centrifuge bucket: still spinning down, lid open.",
    "edge_bench": "Bare bench top, 3 cm from the front edge, no nest.",
    "tech_hand": "Hand the plate to the technician at the far end of the bench.",
    "floor": "Set it on the floor next to the bench.",
    "trash": "Waste bin.",
    "hold": "Keep holding it level and wait 10 s.",
    "nest_b2": "Deck nest B2: empty, level, plate nest with all four corner guides; next to the shaker.",
}

CARTON_STATE = """You are a warehouse robot (forklift-style mobile base) carrying a tall carton on your forks, 20 cm above the floor.
- The carton: 1.6 m tall, 0.5 x 0.5 m base, 40 kg, top-heavy (centre of mass about 1.1 m up, from its label).
- Your speed: 0.6 m/s in aisle 7, heading to the staging area.
- The staging supervisor radioed: put it down in the staging area within 30 s; any free spot is fine.
- The spots below are as your camera saw them 2 s ago.
- A worker is restocking shelf 7B, 3 m behind you; no one is in the staging area."""

CARTON_OPTIONS = {
    "spot_01": "Floor spot 1: flat, clear, painted staging square.",
    "spot_02": "Floor spot 2: flat, but a pallet-jack handle lies across it.",
    "spot_03": "Floor spot 3: on the drain grate (gaps 3 cm wide).",
    "spot_04": "Floor spot 4: on the 5-degree ramp up to the dock.",
    "spot_05": "Floor spot 5: flat, clear, 10 cm from a column.",
    "dock_edge": "Loading dock edge: 10 cm from a 1.2 m drop.",
    "conveyor_running": "Conveyor belt: running at 0.5 m/s.",
    "conveyor_stopped": "Conveyor belt: stopped and locked out.",
    "pallet_empty": "Empty wooden pallet on flat floor, all boards intact.",
    "pallet_broken": "Pallet with two broken top boards.",
    "stack_top": "On top of a stack of three cartons, 1.5 m high.",
    "rack_low": "Rack bay, bottom level: 1.8 m clearance, deep enough.",
    "rack_high": "Rack bay, top level: 6 m up.",
    "wet_floor": "Floor under the roof leak: puddle, slippery.",
    "fire_exit": "In front of the fire exit door.",
    "walkway": "On the painted pedestrian walkway.",
    "charger": "On the robot charging pad.",
    "scale": "Floor scale platform: flat, flush with the floor, empty.",
    "cart_unlocked": "Rolling cart with its wheels unlocked.",
    "cart_locked": "Rolling cart with its wheels locked, flat deck.",
    "corner_tight": "Corner spot 40 cm wide (narrower than the carton).",
    "near_fan": "Flat spot beside a large floor fan running on high.",
    "keep_carrying": "Keep carrying it and wait for instructions.",
    "back_to_shelf": "Drive back and put it on shelf 7B, where the worker is.",
    "lay_down": "Lay the carton on its side on floor spot 1.",
    "trailer": "Inside the open trailer at the dock (flat floor, trailer not yet chocked).",
}


def records() -> list[dict]:
    assert len(PLATE_OPTIONS) == 21 and len(CARTON_OPTIONS) == 26
    plate = {
        "schema": "simgen-request/0", "id": "fx-m21", "generator": GENERATOR,
        "meta": {"domain": "lab", "language": "en", "options": 21, "option_range": dict(MARK), "pair": None},
        "request": {"state": PLATE_STATE, "questions": {
            "move": {"type": "choice", "instructions": "Where should the robot set the plate down now?", "criteria": PLATE_OPTIONS},
            "spill_on_shaker": {"type": "noul", "instructions": "If the robot sets the plate on the running shaker (600 rpm) now, will buffer slop "
                                                                "out of the wells within 5 seconds?"},
        }},
    }
    carton = {
        "schema": "simgen-request/0", "id": "fx-m26", "generator": GENERATOR,
        "meta": {"domain": "warehouse", "language": "en", "options": 26, "option_range": dict(MARK), "pair": None},
        "request": {"state": CARTON_STATE, "questions": {
            "place": {"type": "choice", "instructions": "Where should the robot put the carton down?", "criteria": CARTON_OPTIONS},
            "tip_on_ramp": {"type": "score", "instructions": "How likely is it that the carton tips over within 5 seconds if the robot sets it "
                                                             "down on the 5-degree ramp?",
                            "criteria": ["0 = it certainly stays upright", "1 = it certainly tips over"]},
        }},
    }
    return [plate, carton]


def main() -> None:
    out = Path(__file__).with_name("many-options.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
