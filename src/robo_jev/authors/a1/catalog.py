"""Author A1 catalogs — warehouse and logistics words and the hidden ranges they stand for.

Every hidden quantity is read from a named option (at least four per quantity). Ranges are the **base** prior; the two
alternative priors follow the common rule (:func:`robo_jev.authors.a1.scenes.alt_priors`): friction × 0.7 / × 1.3, mass and
centre-of-mass ranges 1.5 × wider / half as wide around the same centre, placement σ × 1.5 / × 0.75. All names are mine.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# mast — mast picker AMR braking (law A)
# --------------------------------------------------------------------------

#: Mast picker models: wheelbase (m), track (m), chassis mass (kg), chassis centre-of-mass height (m).
MAST_UNITS = {
    "Kestrel MP-4": {"wheelbase": 0.48, "track": 0.40, "base_mass": 46.0, "base_cg_h": 0.19},
    "Strata Lift 2": {"wheelbase": 0.52, "track": 0.42, "base_mass": 54.0, "base_cg_h": 0.21},
    "Orrin PM-60": {"wheelbase": 0.58, "track": 0.46, "base_mass": 68.0, "base_cg_h": 0.18},
    "Halden TowerPick": {"wheelbase": 0.44, "track": 0.38, "base_mass": 41.0, "base_cg_h": 0.22},
    "Vantor M9": {"wheelbase": 0.62, "track": 0.50, "base_mass": 75.0, "base_cg_h": 0.17},
}
#: Tote sizes (mm, along × across × high).
MAST_TOTES = {"std": (600, 400, 320), "half": (400, 300, 250), "deep": (600, 400, 420)}
#: Tote contents → loaded tote mass range (kg, tote tare included).
MAST_CONTENTS = {
    "paper towel rolls": (3.0, 5.5),
    "phone cases": (2.5, 4.5),
    "boxed cereal": (6.0, 10.0),
    "pet food pouches": (11.0, 16.0),
    "bottled water": (16.0, 23.0),
    "canned tomatoes": (20.0, 28.0),
    "hand tools": (24.0, 33.0),
    "bagged fasteners": (30.0, 40.0),
}
#: How the tote sits on the mast tray → fore-aft placement σ of its centre (m).
MAST_SEATS = {"clamped": 0.006, "rail-guided": 0.015, "hand-placed": 0.030, "slid on, unlatched": 0.045}
#: Floor condition → tyre-floor friction range.
MAST_FLOORS = {
    "sealed concrete, dry": (0.70, 0.88),
    "sealed concrete, dusty": (0.46, 0.62),
    "epoxy, dry": (0.58, 0.74),
    "epoxy, damp after mopping": (0.30, 0.44),
    "chill-room floor, condensation": (0.22, 0.34),
    "oil-stained aisle": (0.14, 0.26),
    "anti-slip coating": (0.82, 1.00),
}
MAST_REASONS = ("pick face", "aisle end", "cross-aisle", "charging dock", "pedestrian zone", "handoff point")

# --------------------------------------------------------------------------
# dock — pallet overhanging the dock edge (law B)
# --------------------------------------------------------------------------

#: Pallet mover models (text only).
DOCK_MOVERS = ("PalletPro AM-12", "Ridgeway Mover 3", "Tessel PM2", "Corvo AutoJack", "Nordhaul PX")
#: Pallet footprints: (length along the lead direction, width) in m and the label.
DOCK_PALLETS = {"EUR 1200x800, long side leading": (1.20, 0.80), "EUR 1200x800, short side leading": (0.80, 1.20),
                "industrial 1200x1000": (1.20, 1.00), "half 800x600": (0.80, 0.60)}
#: Load layout → centre-of-mass offset toward the lead end (m, along the pallet) and centre-of-mass height (fraction of the load
#: height).
DOCK_LOADS = {
    "uniform cases, single SKU": {"cg_x": (-0.02, 0.02), "cg_z": (0.44, 0.52)},
    "mixed cases, heavy rows at lead end": {"cg_x": (0.06, 0.14), "cg_z": (0.38, 0.48)},
    "mixed cases, heavy rows at tail end": {"cg_x": (-0.14, -0.06), "cg_z": (0.38, 0.48)},
    "short layer at lead end": {"cg_x": (-0.10, -0.03), "cg_z": (0.40, 0.50)},
    "drink crates stacked at lead end": {"cg_x": (0.10, 0.20), "cg_z": (0.35, 0.45)},
    "heavy base layer, light cartons above": {"cg_x": (-0.03, 0.03), "cg_z": (0.28, 0.38)},
    "top-heavy display units": {"cg_x": (-0.03, 0.03), "cg_z": (0.58, 0.70)},
}
#: Dock surface → pallet-dock friction range.
DOCK_SURFACES = {
    "steel dock plate": (0.32, 0.45),
    "painted concrete": (0.45, 0.60),
    "rubberised dock mat": (0.72, 0.95),
    "wet concrete, rain blown in": (0.28, 0.40),
    "timber dock boards": (0.40, 0.55),
}
#: Set-down guidance → placement σ of the pallet along the lead direction (m).
DOCK_GUIDANCE = {"reflector-guided": 0.008, "lidar map": 0.018, "floor tags": 0.030, "odometry only": 0.050}

# --------------------------------------------------------------------------
# ramp — tugger and cart holding on a ramp (law C)
# --------------------------------------------------------------------------

#: Tugger models → stated mass (kg).
RAMP_TUGS = {"Bruin T-200": 240.0, "Haulmark 4": 330.0, "Tandem TX": 420.0, "Oxen Pull 9": 520.0}
#: Cart types → stated tare (kg).
RAMP_CARTS = {"roll cage": 45.0, "tub cart": 60.0, "platform trolley": 70.0, "shelf cart": 95.0}
#: Cart contents → payload mass range (kg).
RAMP_CONTENTS = {
    "nested empty crates": (30.0, 70.0),
    "returned totes": (60.0, 120.0),
    "mixed grocery totes": (150.0, 260.0),
    "bagged flour": (280.0, 380.0),
    "case-packed water": (380.0, 520.0),
    "copier paper": (420.0, 580.0),
    "brake rotors": (500.0, 680.0),
}
#: Ramp surface → tyre-ramp friction range of the tugger's braked wheels.
RAMP_SURFACES = {
    "brushed concrete, dry": (0.70, 0.88),
    "ramp with grit strips": (0.78, 0.98),
    "concrete, rain-wet": (0.45, 0.60),
    "painted ramp, dusty": (0.40, 0.54),
    "steel ramp plate, wet": (0.24, 0.36),
    "frost on ramp": (0.12, 0.22),
}
RAMP_PLACES = ("dock 4 approach", "mezzanine ramp", "yard ramp B-side", "cold-store ramp", "returns bay ramp")
