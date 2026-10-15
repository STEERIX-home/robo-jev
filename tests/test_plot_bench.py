"""bench-v0 그래프 (Task bench0 §4·§5) — 가짜 결과로 SVG가 만들어지고(두 판), 점·막대·글이 그림 안·값 범위 안에 있으며, 가짜 결과는
워터마크를 달고 docs/ 아래에는 쓰지 않는다."""

import functools
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET

import pytest
from helpers import REPO

from robo_jev.bench.figures import THEMES, bars_svg, figure_rows, hero_rows, hero_svg, readme_snippet, scatter_points, scatter_svg

NS = {"s": "http://www.w3.org/2000/svg"}
SETS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")


def _model(tag, label, role, skill, g1, g2, brier):
    sets = {s: {"readings": {"single_order": {"noul_brier": brier + 0.01 * i}}} for i, s in enumerate(SETS)}
    sets["G1"] = {"readings": {"single_order": {"accuracy": g1}}}
    sets["G2"] = {"readings": {"single_order": {"accuracy": g2, "chance_accuracy": 0.4167}}}
    return {"tag": tag, "label": label, "role": role, "sets": sets, "pooled": {"unseen": {"skill": skill}}, "price_per_m_input": None}


def _results(fake=False, extra=()):
    models = [
        _model("qwen3.5-2b", "Qwen3.5-2B (untrained)", "baseline", -0.148, 0.484, 0.739, 0.18),
        _model("robojev-v0", "robojev v0 (2B)", "ours", 0.777, 0.429, 0.650, 0.03),
        _model("robojev-v0.1-main", "robojev v0.1-main (2B)", "ours", 0.80, 0.47, 0.70, 0.028),
        _model("robojev-v0.1-mix", "robojev v0.1-mix (2B)", "ours", 0.79, 0.62, 0.72, 0.029),
        _model("a", "togethercomputer/tev1-4b-experimental", "api", 0.31, 0.81, 0.86, 0.09),
        _model("b", "openai/gpt-6-luna-decisions", "api", 0.52, 0.90, 0.93, 0.06),
        _model("c", "~typesafe/jev-latest", "api", -0.35, 0.77, 0.88, 0.21),
        *extra,
    ]
    return {"date": "2026-10-12", "fake": fake, "models": models}


def _parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    return root


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_the_scatter_draws_every_model_inside_the_plot_area(theme):
    results = _results()
    svg = scatter_svg(results, theme_name=theme)
    root = _parse(svg)
    width, height = float(root.get("width")), float(root.get("height"))
    marks = [c for c in root.iter("{http://www.w3.org/2000/svg}circle") if c.find("s:title", NS) is not None]
    assert len(marks) == len(results["models"])
    left, right, top, bottom = 74, width - 28, 132, height - 92  # scatter_svg의 그림 칸
    for circle in marks:
        assert left <= float(circle.get("cx")) <= right and top <= float(circle.get("cy")) <= bottom
    texts = {t.text for t in root.iter("{http://www.w3.org/2000/svg}text")}
    assert {m["label"] for m in results["models"]} <= texts
    assert "Physics skill on unseen sets T2–T4 →" in texts and "G2 reading accuracy →" in texts and "always 0.5" in texts
    for text in root.iter("{http://www.w3.org/2000/svg}text"):
        assert 0 <= float(text.get("x")) <= width and 0 <= float(text.get("y")) <= height
    ticks = sorted(float(t.text) for t in root.iter("{http://www.w3.org/2000/svg}text") if re.fullmatch(r"-?\d\.\d", t.text or ""))
    assert ticks[0] <= -0.35 and max(ticks) <= 1.0  # 값 범위를 덮고, skill·정확도 축은 1을 넘지 않는다
    assert svg.count("marker-end") == 3  # 학습 없는 2B → v0 → v0.1 둘
    assert "FAKE API VALUES" not in svg
    assert THEMES[theme]["ours"] in svg and THEMES[theme]["api"] in svg


def test_scatter_labels_do_not_overlap_each_other():
    from robo_jev.bench.figures import _place_labels, _text_width

    points, _ = scatter_points(_results())
    for k, p in enumerate(points):
        p["px"], p["py"] = 100 + 60 * (k % 3), 200 + 5 * k  # 일부러 몰아 둔다
    _place_labels(points, size=12, left=0, right=800, top=0, bottom=600)
    boxes = []
    for p in points:
        w = _text_width(p["label"], 12)
        x0 = p["lx"] if p["anchor"] == "start" else p["lx"] - w
        boxes.append((x0, p["ly"] - 11, x0 + w, p["ly"] + 3))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1], (points[i]["label"], points[j]["label"])


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_the_set_bars_start_at_zero_and_fit_their_columns(theme):
    results = _results()
    svg = bars_svg(results, theme_name=theme)
    root = _parse(svg)
    width = float(root.get("width"))
    bars = [p for p in root.iter("{http://www.w3.org/2000/svg}path") if p.find("s:title", NS) is not None]
    assert len(bars) == len(results["models"]) * 9
    for bar in bars:
        start = re.match(r"M([\d.]+),", bar.get("d"))
        assert start and 0 < float(start.group(1)) < width
    rows = figure_rows(results)
    assert [r["label"] for r in rows][:4] == ["robojev v0 (2B)", "robojev v0.1-main (2B)", "robojev v0.1-mix (2B)", "Qwen3.5-2B (untrained)"]
    assert [r["label"] for r in rows][4:] == ["openai/gpt-6-luna-decisions", "togethercomputer/tev1-4b-experimental", "~typesafe/jev-latest"]
    assert [r["label"] for r in figure_rows(results, top=1)][4:] == ["openai/gpt-6-luna-decisions"]
    best = [t for t in root.iter("{http://www.w3.org/2000/svg}text") if t.get("font-weight") == "600" and re.fullmatch(r"\d\.\d{3}", t.text or "")]
    assert len(best) == 9  # 열마다 가장 좋은 값 하나에만 수
    for text in root.iter("{http://www.w3.org/2000/svg}text"):
        assert 0 <= float(text.get("x")) <= width


def test_fake_results_carry_a_watermark_and_never_land_in_docs(tmp_path, capsys):
    assert all("FAKE API VALUES" in render(_results(fake=True)) for render in (scatter_svg, bars_svg, hero_svg))
    spec = importlib.util.spec_from_file_location("plot_bench", REPO / "scripts" / "plot_bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fake = tmp_path / "results-fake.json"
    fake.write_text(json.dumps(_results(fake=True)), encoding="utf-8")
    target = REPO / "docs" / "figures-should-not-exist"
    with pytest.raises(SystemExit, match="FAKE"):
        module.main(["--results", str(fake), "--out-dir", str(target)])
    assert not target.exists()
    real = tmp_path / "results.json"
    real.write_text(json.dumps(_results()), encoding="utf-8")
    assert module.main(["--results", str(real), "--out-dir", str(tmp_path / "figs")]) == 0
    written = sorted(p.name for p in (tmp_path / "figs").iterdir())
    assert written == ["bench-v0-physics-dark.svg", "bench-v0-physics-light.svg", "bench-v0-scatter-dark.svg", "bench-v0-scatter-light.svg",
                       "bench-v0-sets-dark.svg", "bench-v0-sets-light.svg"]
    assert capsys.readouterr().out.count("<picture>") == 3
    assert 'media="(prefers-color-scheme: dark)"' in readme_snippet(["x"])


def test_models_without_the_axes_values_are_listed_not_drawn():
    incomplete = {"tag": "z", "label": "vendor/partial", "role": "api", "sets": {}, "pooled": {"unseen": None}}
    points, missing = scatter_points(_results(extra=[incomplete]))
    assert "vendor/partial" in missing and all(p["label"] != "vendor/partial" for p in points)
    assert "not shown (no T2–T4 or G2 result): vendor/partial" in scatter_svg(_results(extra=[incomplete]))


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_the_hero_chart_shows_one_bar_per_model_with_only_our_chosen_version(theme):
    results = _results(extra=[_model("robojev-v0.7", "robojev v0.7 (2B)", "ours", 0.81, 0.79, 0.71, 0.027)])
    rows = hero_rows(results)
    roles = [r["role"] for r in rows]
    assert roles.count("ours") == 1 and rows[roles.index("ours")]["label"].startswith("robo-jev v0.7")  # the release
    assert roles.count("previous") == 1 and rows[roles.index("previous")]["label"] == "robo-jev v0.1 · 2B (previous release)"
    assert [r["skill"] for r in rows] == sorted((r["skill"] for r in rows), reverse=True)
    assert len(rows) == 6  # the release + the previous release + baseline + three API models (v0 and v0.1-main left out)
    assert [r["role"] for r in hero_rows(results, previous_tag=None)].count("previous") == 0
    root = _parse(hero_svg(results, theme_name=theme))
    width = float(root.get("width"))
    bars = [e for e in root.iter("{http://www.w3.org/2000/svg}rect") if e.find("s:title", NS) is not None]
    assert len(bars) == len(rows)
    for bar in bars:
        x, w = float(bar.get("x")), float(bar.get("width"))
        assert 0 <= x and x + w <= width
    solid = [b for b in bars if b.get("fill") == THEMES[theme]["ours"]]
    outlined = [b for b in bars if b.get("stroke") == THEMES[theme]["ours"]]
    assert len(solid) == 1 and solid[0].find("s:title", NS).text.startswith("robo-jev v0.7")
    assert len(outlined) == 1 and outlined[0].get("fill") == THEMES[theme]["surface"] and "previous release" in outlined[0].find("s:title", NS).text
    texts = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
    assert "Together Tev1 4B" in texts and "OpenAI GPT-6 Luna Decisions" in texts and "−0.35" in texts  # display names, true minus



def test_the_scatter_draws_the_step_to_the_release():
    results = _results(extra=[_model("robojev-v0.7", "robojev v0.7 (2B)", "ours", 0.95, 0.79, 0.82, 0.027)])  # far enough for an arrow
    svg = scatter_svg(results)
    assert svg.count("marker-end") == 4  # untrained 2B → v0 → v0.1 (two) → v0.7
    assert "robojev v0.7 (2B)" in svg
