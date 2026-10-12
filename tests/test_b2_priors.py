"""Task B2 — T8b's labeller under alternative priors: the prior transform (v0.3b's rule on T8b's hidden parameters), the default that
keeps T8b's labelling unchanged, the same hidden-draw points under every prior, the scene rebuilt from a stored record, and the
``scripts/t8.py relabel`` command."""

import importlib.util
import json
import sys

import pytest
from helpers import REPO
from scipy.stats import norm

from robo_jev.outside import moments as mo

T8B = REPO / "artifacts" / "t8" / "T8b.jsonl"
MASTER = 20261012


def _t8():
    spec = importlib.util.spec_from_file_location("t8_b2test", REPO / "scripts" / "t8.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_original_setting_leaves_the_priors_and_the_labels_unchanged():
    moment = mo.candidate("stack", 0, MASTER)
    assert mo.prior_ranges(moment.hidden) == moment.hidden and mo.prior_ranges(moment.hidden, "original") == moment.hidden
    default = mo.label_moment(moment, k=4)
    assert default == mo.label_moment(moment, k=4, prior="original")
    assert "prior" not in default and "hidden" not in default  # the result T8b's labelling has always returned


def test_the_alternatives_scale_friction_spread_the_centre_of_mass_and_scale_the_release_sigmas():
    hidden = {"mu": (0.4, 0.6), "cx": (-0.22, -0.06), "kh": (0.42, 0.58), "eps": ("normal", 0.0, 0.01), "tilt0": ("normal", 0.0, 1.0)}
    a, b = mo.prior_ranges(hidden, "alt_a"), mo.prior_ranges(hidden, "alt_b")
    assert a["mu"] == pytest.approx((0.28, 0.42)) and b["mu"] == pytest.approx((0.52, 0.78))
    assert a["cx"] == pytest.approx((-0.26, -0.02)) and b["cx"] == pytest.approx((-0.18, -0.10))  # same centre −0.14
    assert a["kh"] == pytest.approx((0.38, 0.62)) and b["kh"] == pytest.approx((0.46, 0.54))
    assert a["eps"] == ("normal", 0.0, pytest.approx(0.015)) and b["eps"] == ("normal", 0.0, pytest.approx(0.0075))
    assert a["tilt0"] == ("normal", 0.0, pytest.approx(1.5)) and b["tilt0"] == ("normal", 0.0, pytest.approx(0.75))
    with pytest.raises(ValueError):
        mo.prior_ranges({**hidden, "damping": (0.1, 0.2)}, "alt_a")  # a hidden parameter without a rule is refused
    with pytest.raises(ValueError):
        mo.prior_ranges({"cx": (0.10, 0.45)}, "alt_a")  # a centre of mass outside the box is refused


def test_every_catalog_prior_stays_inside_the_box_under_both_alternatives():
    specs = [{"kh": f["kh"], "cx": f["cx"]} for f in mo.FILLS.values()] + [{"kh": s["kh"], "cx": s["cx"]} for s in mo.PUSH_SPREADS.values()]
    specs += [{"cx": s["cx"], "kh": (0.42, 0.58)} for s in mo.STACK_SPREADS.values()]
    for spec in specs:
        for setting in ("alt_a", "alt_b"):
            mo.prior_ranges(spec, setting)  # raises if a range leaves the box


def test_every_prior_rolls_the_same_hidden_draw_points():
    moment = mo.candidate("stack", 3, MASTER)
    names = sorted(moment.hidden)
    units = {}
    for setting in mo.PRIOR_SETTINGS:
        ranges = mo.prior_ranges(moment.hidden, setting)
        draws = mo.hidden_draws(ranges, 8, moment.seed ^ 0x5EED)
        rows = []
        for draw in draws:
            row = []
            for name in names:
                spec = ranges[name]
                row.append(norm.cdf((draw[name] - spec[1]) / spec[2]) if spec[0] == "normal" else (draw[name] - spec[0]) / (spec[1] - spec[0]))
            rows.append(row)
        units[setting] = rows
    for setting in ("alt_a", "alt_b"):
        assert [u for row in units[setting] for u in row] == pytest.approx([u for row in units["original"] for u in row], abs=1e-9)


def test_a_stored_record_gives_back_its_candidate():
    moment = mo.candidate("deck_turn", 2, MASTER)
    record = json.loads(json.dumps(mo.moment_record(moment, mo.label_moment(moment, k=4), number=7, mujoco_version="test")))
    rebuilt = mo.moment_from_record(record)
    assert (rebuilt.family, rebuilt.index, rebuilt.seed, rebuilt.target) == (moment.family, moment.index, moment.seed, moment.target)
    assert rebuilt.visible == moment.visible and rebuilt.variants == moment.variants
    assert rebuilt.hidden == {k: tuple(v) for k, v in moment.hidden.items()}
    assert mo.label_moment(rebuilt, k=4) == mo.label_moment(moment, k=4)


def test_the_relabel_command_reproduces_the_labels_under_the_original_prior_and_writes_only_its_own_files(tmp_path):
    t8 = _t8()
    records = []
    for family in ("deck_brake", "push"):
        moment = mo.candidate(family, 1, MASTER)
        records.append(mo.moment_record(moment, mo.label_moment(moment, k=4), number=1, mujoco_version="test"))
    source = tmp_path / "T8b.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "b2"
    assert t8.main(["relabel", "--input", str(source), "--priors", "original", "alt_a", "--workers", "1", "--out-dir", str(out)]) == 0
    rows = {p: [json.loads(line) for line in (out / f"T8b-labels-{p}.jsonl").read_text(encoding="utf-8").splitlines()] for p in ("original", "alt_a")}
    assert [r["labels"] for r in rows["original"]] == [r["labels"] for r in records]
    assert all(r["scene_matches_generator"] and r["request_matches"] and r["draws_seed_matches"] for p in rows for r in rows[p])
    assert all(r["prior"] == "alt_a" and r["hidden"]["mu"][0] == pytest.approx(0.7 * records[i]["meta"]["hidden_prior"]["mu"][0])
               for i, r in enumerate(rows["alt_a"]))
    manifest = json.loads((out / "T8b-labels.manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"original", "alt_a"} and manifest["priors"]["alt_a"]["friction"] == 0.7
    with pytest.raises(SystemExit):  # never into artifacts/t8
        t8.main(["relabel", "--input", str(source), "--priors", "original", "--workers", "1", "--out-dir", str(t8.DEFAULT_OUT)])


@pytest.mark.skipif(not T8B.is_file(), reason="needs the sealed T8b file")
def test_relabelling_a_stored_t8b_moment_under_the_original_prior_reproduces_its_labels():
    t8 = _t8()
    record = next(json.loads(line) for line in T8B.read_text(encoding="utf-8").splitlines() if '"deck_turn"' in line)
    row = t8._relabel_job((record, "original", MASTER))
    assert row["labels"] == record["labels"] and row["scene_matches_generator"] and row["request_matches"] and row["draws_seed_matches"]
