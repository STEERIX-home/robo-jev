"""bench-v0 그래프 — 의존성 없이 SVG를 직접 쓴다 (Task bench0 §4). 글자는 영어(GitHub README).

* :func:`scatter_svg` — 그림 1: 모델 하나 = 점 하나. x = 처음 보는 물리(T2+T3+T4)의 결과 예측 skill(1 − Brier / Brier(언제나 0.5),
  noul·score 질문), y = 일반 읽기 G2 정확도. 우리 모델은 파랑(slot 1), OpenRouter 모델은 주황(slot 2), 학습 없는 2B는 속이 빈
  회색 고리(기준점). 학습 없는 2B → v0 → v0.1의 이동을 화살표로 잇는다. 점마다 이름표(겹치면 밀고 끌줄을 단다).
* :func:`bars_svg` — 그림 2: 행 = 모델, 열 = 세트(T1~T7 예/아니오 Brier — 낮을수록 좋다, G1·G2 정확도 — 높을수록 좋다). 열마다
  작은 막대(0에서 시작), 열 안의 가장 좋은 값에만 수를 단다.
* 색 — 데이터 시각화 기술(dataviz)의 기준 팔레트: 밝은 판·어두운 판을 따로 고른 값(slot 1·2, 둘 다 all-pairs 검사 통과), 글자는
  언제나 글자 색(데이터 색이 아니다), 격자는 hairline. README는 ``<picture>``로 두 판을 GitHub 테마에 맞춰 고른다(:func:`readme_snippet`).
* 가짜 결과(``fake``)로 그리면 대각선 워터마크를 단다 — 장치 시험용이지 결과가 아니다.
"""

from __future__ import annotations

import math
from typing import Any, Iterable
from xml.sax.saxutils import escape

__all__ = ["DISPLAY_NAMES", "THEMES", "bars_svg", "figure_rows", "hero_rows", "hero_svg", "readme_snippet", "scatter_points", "scatter_svg"]

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
THEMES: dict[str, dict[str, str]] = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "ours": "#2a78d6", "api": "#eb6834", "base": "#52514e", "border": "rgba(11,11,11,0.10)", "mark": "#a8a7a0"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "ours": "#3987e5", "api": "#d95926", "base": "#c3c2b7", "border": "rgba(255,255,255,0.10)", "mark": "#6b6a65"},
}
ROLE_NAMES = {"ours": "ours (Steerix, 2B)", "api": "OpenRouter decision models", "baseline": "Qwen3.5-2B untrained (reference)"}


def _t(text: Any) -> str:
    return escape(str(text), {'"': "&quot;"})


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _ticks(low: float, high: float, step: float) -> list[float]:
    start = math.ceil(low / step - 1e-9) * step
    out, value = [], start
    while value <= high + 1e-9:
        out.append(round(value, 6))
        value += step
    return out


def _text_width(text: str, size: float) -> float:
    """글 폭의 어림 (system sans, 평균 글자 폭 ≈ 0.56 em)."""
    return len(text) * size * 0.56


def _frame(width: int, height: int, theme: dict[str, str], title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="t d" font-family=\'{FONT}\'>',
        f"<title id=\"t\">{_t(title)}</title>",
        f"<desc id=\"d\">{_t(desc)}</desc>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" fill="{theme["surface"]}" stroke="{theme["border"]}"/>',
    ]


def _watermark(width: int, height: int, theme: dict[str, str]) -> str:
    return (f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" font-size="34" font-weight="700" fill="{theme["muted"]}" '
            f'opacity="0.35" transform="rotate(-18 {width / 2:.0f} {height / 2:.0f})">FAKE API VALUES — pipeline test only</text>')


# --------------------------------------------------------------------------
# 그림 1 — 산점도
# --------------------------------------------------------------------------


def scatter_points(results: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """결과 → 점들 (x = T2–T4 physics skill, y = G2 정확도, role, label) 과 그리지 못한 모델 이름."""
    points, missing = [], []
    for model in results["models"]:
        unseen = (model.get("pooled") or {}).get("unseen")
        g2 = (((model.get("sets") or {}).get("G2") or {}).get("readings") or {}).get("single_order") or {}
        if not unseen or g2.get("accuracy") is None:
            missing.append(model["label"])
            continue
        points.append({"tag": model.get("tag"), "label": model["label"], "role": model.get("role", "api"), "x": unseen["skill"], "y": g2["accuracy"],
                       "chance": g2.get("chance_accuracy")})
    return points, missing


def _nice_domain(values: list[float], *, floor: float, ceil: float, step: float, pad: float) -> tuple[float, float]:
    low = min([floor] + [v - pad for v in values])
    high = max([ceil] + [v + pad for v in values])
    return math.floor(low / step) * step, math.ceil(high / step) * step


def _place_labels(items: list[dict[str, Any]], *, size: float, left: float, right: float, top: float, bottom: float,
                  obstacles: Iterable[tuple[float, float, float, float]] = ()) -> None:
    """이름표 자리: 점의 오른쪽(가장자리에 닿으면 왼쪽)을 먼저, 다른 이름표·다른 점과 겹치면 반대쪽과 아래·위로 민다.
    ``lx, ly, anchor, leader``를 채운다(밀린 이름표는 끌줄을 단다)."""
    marks = [(i["px"] - 8, i["py"] - 8, i["px"] + 8, i["py"] + 8) for i in items]
    placed: list[tuple[float, float, float, float]] = list(obstacles)

    def free(box: tuple[float, float, float, float], own: int) -> bool:
        if box[0] < left or box[2] > right or box[1] < top or box[3] > bottom:
            return False
        others = placed + [m for k, m in enumerate(marks) if k != own]
        return all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in others)

    order = sorted(range(len(items)), key=lambda k: (items[k]["py"], items[k]["px"]))
    for k in order:
        item = items[k]
        width = _text_width(item["label"], size)
        sides = [("start", item["px"] + 9), ("end", item["px"] - 9)]
        if item["px"] + 9 + width > right:
            sides.reverse()
        best = None
        for shift in [0, 14, -14, 28, -28, 42, -42, 56, -56, 70, -70]:
            for anchor, x0 in sides:
                box_x = x0 if anchor == "start" else x0 - width
                y = item["py"] + 4 + shift
                box = (box_x, y - size + 1, box_x + width, y + 3)
                if free(box, k):
                    best = (anchor, x0, y, box)
                    break
            if best:
                break
        if best is None:
            anchor, x0 = sides[0]
            box_x = x0 if anchor == "start" else x0 - width
            best = (anchor, x0, item["py"] + 4, (box_x, item["py"] - size + 5, box_x + width, item["py"] + 7))
        anchor, x0, y, box = best
        item.update({"lx": x0, "ly": y, "anchor": anchor, "leader": abs(y - (item["py"] + 4)) > 1})
        placed.append(box)


def scatter_svg(results: dict[str, Any], *, theme_name: str = "light", width: int = 820, height: int = 560) -> str:
    """그림 1 (모듈 설명). 점이 없으면 빈 축과 안내문."""
    theme = THEMES[theme_name]
    points, missing = scatter_points(results)
    left, right, top, bottom = 74, width - 28, 132, height - 92
    xs, ys = [p["x"] for p in points], [p["y"] for p in points]
    x0, x1 = _nice_domain(xs, floor=-0.2, ceil=1.0, step=0.2, pad=0.05)
    y0, y1 = _nice_domain(ys, floor=0.5, ceil=0.9, step=0.1, pad=0.03)
    x1, y1 = min(x1, 1.0), min(y1, 1.0)  # skill과 정확도는 1을 넘지 않는다
    chance = next((p["chance"] for p in points if p.get("chance") is not None), None)
    if chance is not None:
        y0 = min(y0, math.floor((chance - 0.02) / 0.1) * 0.1)

    def sx(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (right - left)

    def sy(y: float) -> float:
        return bottom - (y - y0) / (y1 - y0) * (bottom - top)

    title = "Physical judgment vs general reading — bench-v0"
    desc = ("Scatter plot, one point per model. x: physics skill on the unseen sets T2–T4 (1 minus Brier over Brier of always 0.5, yes/no and score "
            "questions). y: accuracy on G2 (public BoolQ + MultiNLI). Up and to the right is better. "
            + "; ".join(f"{p['label']}: skill {p['x']:.3f}, G2 {p['y']:.3f}" for p in points))
    out = _frame(width, height, theme, title, desc)
    out.append(f'<text x="24" y="34" font-size="17" font-weight="600" fill="{theme["ink"]}">{_t(title)}</text>')
    out.append(f'<text x="24" y="56" font-size="12.5" fill="{theme["ink2"]}">x: physics skill on the unseen sets T2–T4 = 1 − Brier / Brier of “always 0.5” '
               f'(yes/no + score questions)</text>')
    out.append(f'<text x="24" y="74" font-size="12.5" fill="{theme["ink2"]}">y: G2 reading accuracy (public BoolQ + MultiNLI, 1,000 questions) · '
               f'up and to the right is better</text>')
    # 범례 (둘 이상의 묶음이면 언제나)
    roles = [r for r in ("ours", "api", "baseline") if any(p["role"] == r for p in points)]
    lx = 24
    for role in roles:
        if role == "baseline":
            out.append(f'<circle cx="{lx + 6}" cy="98" r="5" fill="{theme["surface"]}" stroke="{theme["base"]}" stroke-width="2"/>')
        else:
            out.append(f'<circle cx="{lx + 6}" cy="98" r="5" fill="{theme[role]}"/>')
        out.append(f'<text x="{lx + 17}" y="102" font-size="12.5" fill="{theme["ink2"]}">{_t(ROLE_NAMES[role])}</text>')
        lx += 17 + _text_width(ROLE_NAMES[role], 12.5) + 26
    # 격자·축
    for x in _ticks(x0, x1, 0.2):
        out.append(f'<line x1="{sx(x):.1f}" y1="{top}" x2="{sx(x):.1f}" y2="{bottom}" stroke="{theme["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{sx(x):.1f}" y="{bottom + 18}" text-anchor="middle" font-size="11.5" fill="{theme["muted"]}" '
                   f'style="font-variant-numeric: tabular-nums">{_f(x, 1)}</text>')
    for y in _ticks(y0, y1, 0.1):
        out.append(f'<line x1="{left}" y1="{sy(y):.1f}" x2="{right}" y2="{sy(y):.1f}" stroke="{theme["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{sy(y) + 4:.1f}" text-anchor="end" font-size="11.5" fill="{theme["muted"]}" '
                   f'style="font-variant-numeric: tabular-nums">{_f(y, 1)}</text>')
    out.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{theme["axis"]}" stroke-width="1"/>')
    if x0 < 0 < x1:
        out.append(f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{bottom}" stroke="{theme["axis"]}" stroke-width="1.5"/>')
        out.append(f'<text x="{sx(0) + 5:.1f}" y="{top + 12}" font-size="11" fill="{theme["muted"]}">always 0.5</text>')
    if chance is not None and y0 <= chance <= y1:
        out.append(f'<line x1="{left}" y1="{sy(chance):.1f}" x2="{right}" y2="{sy(chance):.1f}" stroke="{theme["axis"]}" stroke-width="1.5"/>')
        out.append(f'<text x="{right - 4}" y="{sy(chance) - 5:.1f}" text-anchor="end" font-size="11" fill="{theme["muted"]}">chance {chance:.2f}</text>')
    out.append(f'<text x="{(left + right) / 2:.0f}" y="{bottom + 38}" text-anchor="middle" font-size="12.5" fill="{theme["ink2"]}">'
               f'Physics skill on unseen sets T2–T4 →</text>')
    out.append(f'<text x="20" y="{(top + bottom) / 2:.0f}" text-anchor="middle" font-size="12.5" fill="{theme["ink2"]}" '
               f'transform="rotate(-90 20 {(top + bottom) / 2:.0f})">G2 reading accuracy →</text>')
    # 화살표: 학습 없는 2B → v0 → v0.1 (태그로 찾는다)
    by_tag = {p["tag"]: p for p in points if p.get("tag")}
    arrow_boxes: list[tuple[float, float, float, float]] = []
    marker = f"arrow-{theme_name}"
    out.append(f'<defs><marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
               f'<path d="M0,0 L10,5 L0,10 z" fill="{theme["muted"]}"/></marker></defs>')
    for a, b in (("qwen3.5-2b", "robojev-v0"), ("robojev-v0", "robojev-v0.1-main"), ("robojev-v0", "robojev-v0.1-mix")):
        if a in by_tag and b in by_tag:
            ax, ay, bx, by = sx(by_tag[a]["x"]), sy(by_tag[a]["y"]), sx(by_tag[b]["x"]), sy(by_tag[b]["y"])
            length = math.hypot(bx - ax, by - ay)
            if length > 16:
                ux, uy = (bx - ax) / length, (by - ay) / length
                out.append(f'<line x1="{ax + ux * 9:.1f}" y1="{ay + uy * 9:.1f}" x2="{bx - ux * 10:.1f}" y2="{by - uy * 10:.1f}" '
                           f'stroke="{theme["muted"]}" stroke-width="1.5" marker-end="url(#{marker})"/>')
                for step in range(int(length // 8) + 1):  # 이름표가 화살표를 덮지 않게 화살표 위의 점들을 장애물로
                    cx, cy = ax + ux * step * 8, ay + uy * step * 8
                    arrow_boxes.append((cx - 2, cy - 2, cx + 2, cy + 2))
    # 점과 이름표
    for p in points:
        p["px"], p["py"] = sx(p["x"]), sy(p["y"])
    _place_labels(points, size=12, left=left, right=right, top=top, bottom=bottom, obstacles=arrow_boxes)
    for p in points:
        tip = f"{p['label']} — physics skill T2–T4 {p['x']:.3f}, G2 accuracy {p['y']:.3f}"
        if p.get("leader"):
            out.append(f'<line x1="{p["px"]:.1f}" y1="{p["py"]:.1f}" x2="{p["lx"] + (-3 if p["anchor"] == "start" else 3):.1f}" y2="{p["ly"] - 4:.1f}" '
                       f'stroke="{theme["grid"]}" stroke-width="1"/>')
        fill, ring = (theme["surface"], theme["base"]) if p["role"] == "baseline" else (theme[p["role"]], theme["surface"])
        out.append(f'<circle cx="{p["px"]:.1f}" cy="{p["py"]:.1f}" r="5.5" fill="{fill}" stroke="{ring}" stroke-width="2"><title>{_t(tip)}</title></circle>')
        weight = ' font-weight="600"' if p["role"] == "ours" else ""
        out.append(f'<text x="{p["lx"]:.1f}" y="{p["ly"]:.1f}" text-anchor="{p["anchor"]}" font-size="12" fill="{theme["ink"] if p["role"] == "ours" else theme["ink2"]}"'
                   f'{weight}>{_t(p["label"])}</text>')
    if not points:
        out.append(f'<text x="{(left + right) / 2:.0f}" y="{(top + bottom) / 2:.0f}" text-anchor="middle" font-size="13" fill="{theme["muted"]}">no results yet</text>')
    notes = [f"bench-v0 · evaluated {results.get('date', '')} · home ground: T1–T7 come from our own physics generator",
             "OpenRouter models: one Decisions API call per request, stored option order · ours: expectation over balanced option orders"]
    if missing:
        notes.insert(0, f"not shown (no T2–T4 or G2 result): {', '.join(missing)}")
    for index, note in enumerate(reversed(notes)):
        out.append(f'<text x="24" y="{height - 16 - 15 * index}" font-size="11" fill="{theme["muted"]}">{_t(note)}</text>')
    if results.get("fake"):
        out.append(_watermark(width, height, theme))
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# 그림 2 — 세트별 막대
# --------------------------------------------------------------------------

BAR_COLUMNS: tuple[tuple[str, str, str], ...] = tuple((f"T{i}", "noul_brier", "lower") for i in range(1, 8)) + (("G1", "accuracy", "higher"), ("G2", "accuracy", "higher"))


def figure_rows(results: dict[str, Any], *, top: int | None = None) -> list[dict[str, Any]]:
    """그림 2의 행: 우리 모델(정의 순서, 학습 없는 2B는 끝) 다음 API 모델(T2–T4 physics skill 순, 위에서 `top`개). 값은 한 순서 읽기."""
    ours = [m for m in results["models"] if m.get("role") == "ours"]
    base = [m for m in results["models"] if m.get("role") == "baseline"]
    api = [m for m in results["models"] if m.get("role") == "api"]
    api.sort(key=lambda m: -((m.get("pooled") or {}).get("unseen") or {}).get("skill", -math.inf))
    if top is not None:
        api = api[:top]
    rows = []
    for model in ours + base + api:
        values = {}
        for set_name, metric, _ in BAR_COLUMNS:
            reading = (((model.get("sets") or {}).get(set_name) or {}).get("readings") or {}).get("single_order") or {}
            values[set_name] = reading.get(metric)
        rows.append({"label": model["label"], "role": model.get("role", "api"), "values": values})
    return rows


def bars_svg(results: dict[str, Any], *, theme_name: str = "light", top: int | None = None) -> str:
    """그림 2 (모듈 설명)."""
    theme = THEMES[theme_name]
    rows = figure_rows(results, top=top)
    label_w, col_w, gap_group = 262, 80, 18
    row_h, head = 22, 136
    width = 24 + label_w + col_w * len(BAR_COLUMNS) + gap_group + 20
    height = head + row_h * max(len(rows), 1) + 64
    brier_max = max([0.25] + [r["values"][c] for r in rows for c, m, _ in BAR_COLUMNS if m == "noul_brier" and r["values"][c] is not None])
    brier_max = math.ceil(brier_max / 0.1 - 1e-9) * 0.1  # 0.3 → 가운데 눈금 0.15
    title = "Per-set results — bench-v0"
    desc = "Bars per model and set. " + "; ".join(
        f"{r['label']}: " + ", ".join(f"{c} {r['values'][c]:.3f}" for c, _, _ in BAR_COLUMNS if r["values"][c] is not None) for r in rows)
    out = _frame(width, height, theme, title, desc)
    out.append(f'<text x="24" y="34" font-size="17" font-weight="600" fill="{theme["ink"]}">{_t(title)}</text>')
    out.append(f'<text x="24" y="55" font-size="12.5" fill="{theme["ink2"]}">T1–T7: yes/no Brier against the simulated outcome probability '
               f'(lower is better; thin vertical tick = always answering 0.5) · G1, G2: reading accuracy (higher is better)</text>')
    uniform = {}
    for model in results["models"]:
        for set_name, metric, _ in BAR_COLUMNS:
            reading = (((model.get("sets") or {}).get(set_name) or {}).get("readings") or {}).get("single_order") or {}
            if metric == "noul_brier" and reading.get("noul_brier_uniform") is not None:
                uniform.setdefault(set_name, reading["noul_brier_uniform"])
    roles = [r for r in ("ours", "baseline", "api") if any(row["role"] == r for row in rows)]
    lx = 24
    for role in roles:
        color = theme["mark"] if role == "baseline" else theme[role]
        out.append(f'<rect x="{lx}" y="72" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(f'<text x="{lx + 18}" y="82" font-size="12.5" fill="{theme["ink2"]}">{_t(ROLE_NAMES[role])}</text>')
        lx += 18 + _text_width(ROLE_NAMES[role], 12.5) + 26

    def col_x(index: int) -> float:
        return 24 + label_w + index * col_w + (gap_group if index >= 7 else 0)

    for index, (set_name, metric, better) in enumerate(BAR_COLUMNS):
        x = col_x(index)
        out.append(f'<text x="{x + 4}" y="{head - 12}" font-size="12.5" font-weight="600" fill="{theme["ink"]}">{set_name}</text>')
        scale_max = brier_max if metric == "noul_brier" else 1.0
        inner = col_w - 14
        out.append(f'<line x1="{x + 4}" y1="{head - 4}" x2="{x + 4}" y2="{head + row_h * len(rows)}" stroke="{theme["axis"]}" stroke-width="1"/>')
        mid = x + 4 + inner / 2
        out.append(f'<line x1="{mid:.1f}" y1="{head - 4}" x2="{mid:.1f}" y2="{head + row_h * len(rows)}" stroke="{theme["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x + 4}" y="{head + row_h * len(rows) + 15}" font-size="10.5" fill="{theme["muted"]}">0</text>')
        out.append(f'<text x="{mid:.1f}" y="{head + row_h * len(rows) + 15}" text-anchor="middle" font-size="10.5" fill="{theme["muted"]}">'
                   f'{_f(scale_max / 2, 2 if metric == "noul_brier" else 1)}</text>')
        if set_name in uniform:  # 언제나 0.5의 Brier — 이 선을 넘는 막대는 상수 답보다 나쁘다
            ux = x + 4 + min(uniform[set_name], scale_max) / scale_max * inner
            out.append(f'<line x1="{ux:.1f}" y1="{head - 4}" x2="{ux:.1f}" y2="{head + row_h * len(rows)}" stroke="{theme["ink2"]}" stroke-width="1.5">'
                       f"<title>{set_name}: always 0.5 scores {uniform[set_name]:.3f}</title></line>")
        present = [r["values"][set_name] for r in rows if r["values"][set_name] is not None]
        best_value = (min(present) if better == "lower" else max(present)) if present else None
        best = next((i for i, r in enumerate(rows) if r["values"][set_name] is not None and abs(r["values"][set_name] - best_value) < 1e-12), None)
        for r_index, row in enumerate(rows):
            value = row["values"][set_name]
            y = head + r_index * row_h + 4
            if value is None:
                out.append(f'<text x="{x + 8}" y="{y + 11}" font-size="10.5" fill="{theme["muted"]}">n/a</text>')
                continue
            length = max(1.5, min(value, scale_max) / scale_max * inner)
            color = theme["mark"] if row["role"] == "baseline" else theme[row["role"]]
            radius = min(4.0, length / 2)
            out.append(f'<path d="M{x + 4:.1f},{y:.1f} h{length - radius:.1f} a{radius:.1f},{radius:.1f} 0 0 1 {radius:.1f},{radius:.1f} v{14 - 2 * radius:.1f} '
                       f'a{radius:.1f},{radius:.1f} 0 0 1 -{radius:.1f},{radius:.1f} h-{length - radius:.1f} z" fill="{color}">'
                       f"<title>{_t(row['label'])} — {set_name} {'yes/no Brier' if metric == 'noul_brier' else 'accuracy'} {value:.3f}</title></path>")
            if r_index == best:  # 열마다 가장 좋은 값 하나에만 수 — 막대 끝 밖에 자리가 없으면 막대 안 끝에(채움의 밝기에 맞춘 글자 색)
                text = _f(value, 3)
                room = inner - length
                if room >= _text_width(text, 10.5) + 5:
                    out.append(f'<text x="{x + 4 + length + 3:.1f}" y="{y + 11}" font-size="10.5" font-weight="600" fill="{theme["ink"]}">{text}</text>')
                else:
                    out.append(f'<text x="{x + 4 + length - 4:.1f}" y="{y + 11}" text-anchor="end" font-size="10.5" font-weight="600" '
                               f'fill="{_on(color)}">{text}</text>')
    for r_index, row in enumerate(rows):
        y = head + r_index * row_h + 15
        weight = ' font-weight="600"' if row["role"] == "ours" else ""
        label = row["label"] if _text_width(row["label"], 12) < label_w - 12 else row["label"][: int((label_w - 24) / (12 * 0.56))] + "…"
        out.append(f'<text x="{24 + label_w - 10}" y="{y}" text-anchor="end" font-size="12" fill="{theme["ink"] if row["role"] == "ours" else theme["ink2"]}"'
                   f'{weight}>{_t(label)}</text>')
    out.append(f'<text x="{col_x(0) + 4}" y="{head - 32}" font-size="11.5" fill="{theme["muted"]}">yes/no Brier ↓ (lower is better) · '
               f'the number marks the best model in each column</text>')
    out.append(f'<text x="{col_x(7) + 4}" y="{head - 32}" font-size="11.5" fill="{theme["muted"]}">accuracy ↑</text>')
    note = f"bench-v0 · {results.get('date', '')} · single-order reading · T1–T7 are home ground (our generator)"
    out.append(f'<text x="24" y="{height - 16}" font-size="11" fill="{theme["muted"]}">{_t(note)}</text>')
    if results.get("fake"):
        out.append(_watermark(width, height, theme))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _on(fill: str) -> str:
    """채움 위 글자 색 — 상대 휘도로 흰 글자와 검은 글자 중 대비가 큰 쪽."""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#0b0b0b" if (luminance + 0.05) / 0.05 > 1.05 / (luminance + 0.05) else "#ffffff"


def readme_snippet(names: Iterable[str], *, folder: str = "docs/figures") -> str:
    """README에 넣을 ``<picture>`` 조각 — GitHub 테마(밝음·어두움)에 맞는 판을 고른다."""
    parts = []
    for name in names:
        parts.append(
            "<picture>\n"
            f'  <source media="(prefers-color-scheme: dark)" srcset="{folder}/{name}-dark.svg">\n'
            f'  <img alt="{name}" src="{folder}/{name}-light.svg">\n'
            "</picture>"
        )
    return "\n\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# Hero figure — physical judgment on unseen situations (one bar per model)
# --------------------------------------------------------------------------

#: Display names for the hero figure (provider ids → product names).
DISPLAY_NAMES = {
    "~typesafe/jev-latest": "TypeSafe Jev",
    "liquid/d1": "Liquid d1",
    "microsoft/microsoft-decision-1": "Microsoft-Decision-1",
    "togethercomputer/tev1-4b-experimental": "Together Tev1 4B",
    "openai/gpt-6-luna-decisions": "OpenAI GPT-6 Luna Decisions",
    "upstage/solar-decide": "Upstage Solar Decide",
}


def hero_rows(results: dict[str, Any], *, ours_tag: str = "robojev-v0.1-mix", ours_label: str = "robo-jev v0.1 · 2B (ours)",
              base_label: str = "Qwen3.5-2B untrained (our base)") -> list[dict[str, Any]]:
    """One row per model: our chosen version, every API model and the untrained base, sorted by physics skill on T2–T4."""
    rows = []
    for model in results["models"]:
        role = model.get("role", "api")
        if role == "ours" and model.get("tag") != ours_tag:
            continue
        unseen = (model.get("pooled") or {}).get("unseen") or {}
        if unseen.get("skill") is None:
            continue
        if role == "ours":
            label = ours_label
        elif role == "baseline":
            label = base_label
        else:
            label = DISPLAY_NAMES.get(str(model.get("model_id") or model["label"]), str(model["label"]))
        rows.append({"label": label, "role": role, "skill": float(unseen["skill"]), "coverage": unseen.get("coverage")})
    rows.sort(key=lambda r: -r["skill"])
    return rows


def hero_svg(results: dict[str, Any], *, theme_name: str = "light", width: int = 860, **names: Any) -> str:
    """Horizontal bars of physics skill (1 − Brier / Brier of always 0.5) on the unseen sets T2–T4, ours highlighted."""
    theme = THEMES[theme_name]
    rows = hero_rows(results, **names)
    label_w, row_h, head, foot = 250, 34, 112, 58
    left, right = 24 + label_w, width - 64
    lo, hi = -0.2, 1.0
    height = head + row_h * max(len(rows), 1) + foot

    def sx(value: float) -> float:
        return left + (min(max(value, lo), hi) - lo) / (hi - lo) * (right - left)

    title = "Physical judgment on our simulation benchmark (held-out moments)"
    desc = "Physics skill on bench-v0 T2–T4 (higher is better; 0 = always answering 0.5). " + "; ".join(f"{r['label']} {r['skill']:.2f}" for r in rows)
    out = _frame(width, height, theme, title, desc)
    out.append(f'<text x="24" y="36" font-size="19" font-weight="700" fill="{theme["ink"]}">{_t(title)}</text>')
    out.append(f'<text x="24" y="60" font-size="12.5" fill="{theme["ink2"]}">Will it tip, spill, collide, slip, or sit wrong? Skill vs. always answering 50 %, '
               f'on 1,100 held-out robot moments:</text>')
    out.append(f'<text x="24" y="77" font-size="12.5" fill="{theme["ink2"]}">new wording and settings, out-of-range values, and unseen combinations of phenomena. '
               f'Higher is better.</text>')
    top = head - 14
    bottom = head + row_h * len(rows) - 6
    for tick in _ticks(lo, hi, 0.2):
        x = sx(tick)
        strong = abs(tick) < 1e-9
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{theme["axis"] if strong else theme["grid"]}" stroke-width="{1.4 if strong else 1}"/>')
        out.append(f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="11.5" fill="{theme["muted"]}">{f"{tick:.1f}".replace("-", "−")}</text>')
    out.append(f'<text x="{sx(0) + 5:.1f}" y="{top + 2}" font-size="11" fill="{theme["muted"]}">always 50 %</text>')
    for index, row in enumerate(rows):
        y = head + index * row_h
        color = theme["mark"] if row["role"] == "baseline" else theme[row["role"]]
        x0, x1 = sx(0), sx(row["skill"])
        bar_x, bar_w = min(x0, x1), max(abs(x1 - x0), 1.5)
        weight = ' font-weight="700"' if row["role"] == "ours" else ""
        ink = theme["ink"] if row["role"] == "ours" else theme["ink2"]
        out.append(f'<text x="{left - 12}" y="{y + 15}" text-anchor="end" font-size="13"{weight} fill="{ink}">{_t(row["label"])}</text>')
        out.append(f'<rect x="{bar_x:.1f}" y="{y + 2}" width="{bar_w:.1f}" height="{row_h - 12}" rx="4" fill="{color}">'
                   f'<title>{_t(row["label"])}: {row["skill"]:.3f}</title></rect>')
        value = f"{row['skill']:.2f}".replace("-", "−")
        if row.get("coverage") is not None and row["coverage"] < 0.995:
            value += f"  ({row['coverage']:.0%} answered)"
        if row["skill"] >= 0:
            out.append(f'<text x="{x1 + 6:.1f}" y="{y + 15}" font-size="12.5"{weight} fill="{ink}">{value}</text>')
        else:
            out.append(f'<text x="{x0 + 6:.1f}" y="{y + 15}" font-size="12.5"{weight} fill="{ink}">{value}</text>')
    out.append(f'<text x="24" y="{height - 26}" font-size="11" fill="{theme["muted"]}">bench-v0 · 2026-10-12 · outcome probabilities from our physics simulator '
               f'(home ground) · OpenRouter models via the Decisions API, one call per request</text>')
    out.append(f'<text x="24" y="{height - 11}" font-size="11" fill="{theme["muted"]}">skill = 1 − Brier / Brier(always 0.5) over the yes/no and score questions '
               f'of T2–T4 · details: docs/results.md</text>')
    if results.get("fake"):
        out.append(_watermark(width, height, theme))
    out.append("</svg>")
    return "\n".join(out)
