"""빼 둔 칸과 나눔(split) — 봉인된 시험 세트의 정직성 (Task v0b 지시서 §4·§5, configs/simgen/holdout-v0b.yaml).

**정본은 설정 파일 하나**(:data:`HOLDOUT_PATH`). 학습 나눔(``train``)은 빼 둔 칸을 만들 수 없다:

* 문체(``log``), 분야마다 다섯째 상황, 빼 둔 바꿔 쓰기 묶음, 조합 P1+P3 — 생성기가 **고를 수 없게** 막는다(:class:`Split`의 허용 목록).
* 변수 범위(T3 칸) — 생성기가 뽑는 범위를 학습 쪽으로 좁히고(:func:`ranges_for`), 관측을 만든 뒤 보고값이 시험 쪽에 들어가면
  거절해 다시 뽑는다(:func:`violations`). 마지막 관문으로 레코드를 쓰기 전에 한 번 더 검사한다(generate).
* **칸의 자르기는 그 변수가 관측되는 모든 장면에 건다**(Task v0b 수정 1 — I1): 칸의 ``scenes``는 T3가 그 칸의 레코드를 뽑는 장면일 뿐,
  학습 쪽 자르기의 범위가 아니다. 로봇 속도처럼 로봇 자체의 변수가 목록 밖 장면(P3+P5·P3·P6)으로 새던 것을 막는다. 같은 변수를 쓰는
  칸들은 한 묶음이다: 같은 쪽·같은 자르기를 써야 하고 장면 목록이 겹치면 안 된다(어기면 :func:`load_holdout` 이 거절한다). 묶음의 시험 쪽
  관측은 그 장면을 목록에 가진 칸 이름으로(없으면 묶음의 첫 칸으로) 적는다 — T3 레코드는 강제한 칸 하나로 적힌다.

시험 나눔은 칸을 강제한다(T2: 문체·상황·바꿔 쓰기 축, T3: 칸 하나, T4: P1+P3). :func:`cells_hit` 는 관측이 어느 T3 칸의 시험 쪽에 있는지,
:func:`request_hash` 는 겹침 검사에 쓰는 요청 글의 해시다. 세계 모듈을 import하지 않는다(관측만 본다).
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from robo_jev.simgen.catalog import GROUPS, SCENE_GROUP
from robo_jev.simgen.observation import Observation

__all__ = [
    "HOLDOUT_PATH",
    "OBSERVED_VARS",
    "Holdout",
    "RangeCell",
    "Split",
    "cells_hit",
    "cells_of_values",
    "load_holdout",
    "observed_value",
    "observed_values",
    "ranges_for",
    "request_hash",
    "split_for",
    "violations",
]

HOLDOUT_PATH = Path(__file__).resolve().parents[3] / "configs" / "simgen" / "holdout-v0b.yaml"
TRAIN_STYLES = ("facts", "prose", "planner")
#: 칸이 쓸 수 있는 관측 변수(:func:`observed_value`) — 그 변수를 보이는 장면이면 어디서나 칸의 자르기가 걸린다
OBSERVED_VARS = ("robot_speed", "track_age", "fill_reading", "load_aspect", "ramp_angle", "gauge_ratio", "image_age", "stale_age", "corner_lateral")


@dataclass(frozen=True)
class RangeCell:
    """T3 칸 하나: 관측 변수 ``var``가 ``side`` 쪽(above: > cut, below: < cut)이면 시험 전용."""

    name: str
    phenomenon: str
    var: str
    side: str
    cut: float | dict
    scenes: tuple[str, ...]

    def cut_for(self, body: str) -> float:
        return float(self.cut[body]) if isinstance(self.cut, dict) else float(self.cut)

    def in_test(self, value: float, body: str) -> bool:
        cut = self.cut_for(body)
        return value > cut + 1e-9 if self.side == "above" else value < cut - 1e-9

    def train_bounds(self, body: str) -> tuple[float, float]:
        cut = self.cut_for(body)
        return (-1e9, cut) if self.side == "above" else (cut, 1e9)

    def test_bounds(self, body: str) -> tuple[float, float]:
        cut = self.cut_for(body)
        return (cut, 1e9) if self.side == "above" else (-1e9, cut)


@dataclass(frozen=True)
class Holdout:
    version: str
    styles: tuple[str, ...]
    settings: dict
    paraphrase_pool: str
    groups: tuple[str, ...]
    ranges: tuple[RangeCell, ...]
    seed_ranges: dict
    sha256: str

    def heldout_settings(self, domain: str) -> tuple[int, ...]:
        return tuple(self.settings.get(domain, ()))

    def cell(self, name: str) -> RangeCell:
        return next(c for c in self.ranges if c.name == name)

    def group(self, cell: RangeCell) -> tuple[RangeCell, ...]:
        """같은 변수를 쓰는 칸들(한 물리량 — 같은 자르기)."""
        return tuple(c for c in self.ranges if c.var == cell.var)

    def reported_under(self, cell: RangeCell, scene: str) -> bool:
        """시험 쪽 관측을 이 칸 이름으로 적나: 묶음에서 그 장면을 목록에 가진 칸, 없으면 묶음의 첫 칸."""
        group = self.group(cell)
        listing = [c for c in group if scene in c.scenes]
        return cell in listing if listing else cell is group[0]

    def definition(self) -> dict:
        """manifest에 적는 빼 둔 칸의 정의."""
        return {
            "version": self.version, "sha256": self.sha256, "styles": list(self.styles), "settings": {d: list(v) for d, v in self.settings.items()},
            "paraphrase_pool": self.paraphrase_pool, "groups": list(self.groups),
            "ranges": [{"name": c.name, "phenomenon": c.phenomenon, "var": c.var, "side": c.side, "cut": c.cut, "scenes": list(c.scenes)} for c in self.ranges],
            "seed_ranges": self.seed_ranges,
        }  # fmt: skip


def holdout_errors(cells: tuple[RangeCell, ...]) -> list[str]:
    """칸 정의의 규칙 위반(빈 목록이면 통과): 아는 변수, 같은 변수의 칸은 같은 쪽·같은 자르기(학습 쪽이 하나로 정해지게), 장면 목록이
    서로 겹치지 않음(시험 쪽 관측을 적을 칸이 하나로 정해지게), 이름이 겹치지 않음."""
    errors = []
    if len({c.name for c in cells}) != len(cells):
        errors.append("duplicate cell names")
    for c in cells:
        if c.var not in OBSERVED_VARS:
            errors.append(f"{c.name}: unknown variable {c.var}")
        if c.side not in ("above", "below"):
            errors.append(f"{c.name}: side must be above or below")
    by_var: dict[str, list[RangeCell]] = {}
    for c in cells:
        by_var.setdefault(c.var, []).append(c)
    for var, group in by_var.items():
        first = group[0]
        for c in group[1:]:
            if (c.side, c.cut) != (first.side, first.cut):
                errors.append(f"{var}: cells {first.name} and {c.name} cut the same variable differently")
            if set(c.scenes) & set(first.scenes):
                errors.append(f"{var}: cells {first.name} and {c.name} share T3 scenes {sorted(set(c.scenes) & set(first.scenes))}")
    return errors


@functools.lru_cache(maxsize=2)
def load_holdout(path: Path = HOLDOUT_PATH) -> Holdout:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    cells = tuple(RangeCell(c["name"], c["phenomenon"], c["var"], c["side"], c["cut"], tuple(c["scenes"])) for c in data["ranges"])
    errors = holdout_errors(cells)
    if errors:
        raise ValueError(f"{path.name}: " + "; ".join(errors))
    groups = tuple(data["groups"])
    assert set(groups) <= set(GROUPS)
    return Holdout(data["version"], tuple(data["styles"]), dict(data["settings"]), data["paraphrase_pool"], groups, cells,
                   {k: tuple(v) for k, v in data["seed_ranges"].items()}, hashlib.sha256(text.encode("utf-8")).hexdigest())  # fmt: skip


@dataclass(frozen=True)
class Split:
    """나눔 하나의 허용 규칙.

    * ``styles`` — 고를 수 있는 문체. ``settings`` — 'train'(빼 둔 상황 제외) / 'heldout'(빼 둔 상황만) / 'any'.
    * ``pool`` — 질문 문구 묶음('train' / 'heldout'). ``groups`` — 고를 수 있는 현상·조합 묶음.
    * ``cell`` — 강제한 T3 칸 이름(그 칸의 시험 쪽에서 뽑는다; 다른 칸은 자유) 또는 None. ``train_ranges`` — 참이면 모든 T3 칸을 학습
      쪽으로 좁힌다(학습·T1·T5·T6·T7 — 같은 분포).
    * ``ko_share`` — 한국어 레코드 비율.
    """

    name: str
    styles: tuple[str, ...] = TRAIN_STYLES
    settings: str = "train"
    pool: str = "train"
    groups: tuple[str, ...] = tuple(g for g in GROUPS if g != "P1+P3")
    cell: str | None = None
    train_ranges: bool = True
    ko_share: float = 0.15
    extra: dict = field(default_factory=dict)

    @property
    def is_train(self) -> bool:
        return self.name == "train"


def split_for(name: str, holdout: Holdout | None = None, **changes) -> Split:
    """이름 붙은 나눔의 기본 규칙(T2·T3·T4는 시험 세트 생성기가 레코드마다 축·칸을 정해 바꾼다)."""
    holdout = holdout or load_holdout()
    train_groups = tuple(g for g in GROUPS if g not in holdout.groups)
    base = {"train": Split("train", groups=train_groups), "T1": Split("T1", groups=train_groups), "T5": Split("T5", groups=train_groups),
            "T6": Split("T6", groups=train_groups), "T7": Split("T7", groups=train_groups),
            "T2": Split("T2", styles=TRAIN_STYLES + holdout.styles, settings="any", groups=train_groups),
            "T3": Split("T3", groups=train_groups, train_ranges=False), "T4": Split("T4", groups=holdout.groups)}[name]  # fmt: skip
    return Split(**{**base.__dict__, **changes})


# ----------------------------------------------------------------------------------------------------------------
# 관측에서 읽는 변수
# ----------------------------------------------------------------------------------------------------------------


def observed_value(obs: Observation, var: str, cfg=None) -> float | None:
    """T3 칸의 변수를 관측(글에 보이는 값)에서 읽는다 — 그 변수가 없는 관측이면 None."""
    if var == "robot_speed":
        return float(obs.robot.cruise if obs.robot.speed <= 0 else obs.robot.speed)
    if var == "track_age":
        return None if obs.agent is None else float(obs.agent.age)
    if var == "fill_reading":
        return None if obs.container is None else float(obs.container.fill.value)
    if var == "load_aspect":
        return None if obs.load is None else obs.load.aspect_shown()
    if var == "ramp_angle":
        return obs.env.ramp_angle_deg
    if var == "gauge_ratio":
        return None if obs.place is None else float(obs.place.gauge_ratio)
    if var == "image_age":
        return None if obs.place is None or not obs.place.ages else float(max(obs.place.ages.values()))
    if var == "stale_age":
        return None if obs.stale is None else float(obs.stale.age)
    if var == "corner_lateral":
        return None if obs.corner is None else float(obs.corner.lateral.value)
    raise ValueError(var)


def observed_values(obs: Observation) -> dict[str, float]:
    """관측에서 읽히는 칸 변수 전부(장면 목록과 무관하게) — 레코드의 meta.holdout_vars."""
    out = {}
    for var in OBSERVED_VARS:
        value = observed_value(obs, var)
        if value is not None:
            out[var] = round(float(value), 6)
    return out


def cells_of_values(holdout: Holdout, scene: str, body: str, values: dict[str, float]) -> list[str]:
    """관측 변수 값(장면·몸)이 시험 쪽에 들어간 칸들 — 칸의 자르기는 변수가 관측되는 모든 장면에 걸고, 묶음의 적을 칸 이름으로 적는다."""
    out = []
    for cell in holdout.ranges:
        value = values.get(cell.var)
        if value is not None and cell.in_test(value, body) and holdout.reported_under(cell, scene):
            out.append(cell.name)
    return out


def cells_hit(holdout: Holdout, obs: Observation) -> list[str]:
    """관측이 시험 쪽에 들어간 T3 칸들(장면 목록과 무관하게 — 변수가 관측되면 언제나)."""
    return cells_of_values(holdout, obs.scene, obs.body, observed_values(obs))


def ranges_for(holdout: Holdout, split: Split, scene: str, body: str) -> dict[str, tuple[float, float]]:
    """생성기가 변수를 뽑을 범위(변수 → (하한, 상한)): 학습 쪽 나눔은 모든 칸을 학습 쪽으로(장면과 무관하게 — 변수가 그 장면에 없으면
    쓰이지 않을 뿐), T3는 강제한 칸을 시험 쪽으로(그 칸의 묶음의 다른 칸은 같은 물리량이라 좁히지 않는다)."""
    out: dict[str, tuple[float, float]] = {}

    def narrow(var: str, bounds: tuple[float, float]) -> None:
        lo, hi = out.get(var, (-1e9, 1e9))
        out[var] = (max(lo, bounds[0]), min(hi, bounds[1]))

    forced = holdout.cell(split.cell) if split.cell is not None else None
    for cell in holdout.ranges:
        if forced is not None and cell.name == forced.name:
            narrow(cell.var, cell.test_bounds(body))
        elif forced is not None and cell.var == forced.var:
            continue
        elif split.train_ranges:
            narrow(cell.var, cell.train_bounds(body))
    return out


def violations(holdout: Holdout, split: Split, obs: Observation, style: str, setting: int, pool: str) -> list[str]:
    """이 나눔이 만들면 안 되는 칸(또는 강제한 칸을 못 맞춘 것)의 목록 — 비면 통과."""
    out = []
    group = SCENE_GROUP[obs.scene]
    if group not in split.groups:
        out.append(f"group {group}")
    if style not in split.styles:
        out.append(f"style {style}")
    held = setting in holdout.heldout_settings(obs.domain)
    if split.settings == "train" and held:
        out.append(f"setting {obs.domain}[{setting}]")
    if split.settings == "heldout" and not held:
        out.append(f"setting {obs.domain}[{setting}] is not held out")
    if pool != split.pool:
        out.append(f"pool {pool}")
    hit = cells_hit(holdout, obs)
    if split.train_ranges and split.cell is None and hit:
        out.append(f"range {hit}")
    if split.cell is not None and split.cell not in hit:
        out.append(f"range {split.cell} not reached")
    if split.cell is not None and split.train_ranges and [h for h in hit if h != split.cell]:
        out.append(f"range {[h for h in hit if h != split.cell]}")
    return out


def request_hash(record: dict) -> str:
    """요청(모델 입력) 글의 해시 — 상태 글과 질문 전부의 정규 JSON."""
    text = json.dumps(record["request"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
