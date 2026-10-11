"""빼 둔 칸과 나눔(split) — 봉인된 시험 세트의 정직성 (Task v0b 지시서 §4·§5, configs/simgen/holdout-v0b.yaml).

**정본은 설정 파일 하나**(:data:`HOLDOUT_PATH`). 학습 나눔(``train``)은 빼 둔 칸을 만들 수 없다:

* 문체(``log``), 분야마다 다섯째 상황, 빼 둔 바꿔 쓰기 묶음, 조합 P1+P3 — 생성기가 **고를 수 없게** 막는다(:class:`Split`의 허용 목록).
* 변수 범위(T3 칸) — 생성기가 뽑는 범위를 학습 쪽으로 좁히고(:func:`ranges_for`), 관측을 만든 뒤 보고값이 시험 쪽에 들어가면
  거절해 다시 뽑는다(:func:`violations`). 마지막 관문으로 레코드를 쓰기 전에 한 번 더 검사한다(generate).

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

__all__ = ["HOLDOUT_PATH", "Holdout", "RangeCell", "Split", "cells_hit", "load_holdout", "observed_value", "ranges_for", "request_hash", "split_for", "violations"]

HOLDOUT_PATH = Path(__file__).resolve().parents[3] / "configs" / "simgen" / "holdout-v0b.yaml"
TRAIN_STYLES = ("facts", "prose", "planner")


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

    def definition(self) -> dict:
        """manifest에 적는 빼 둔 칸의 정의."""
        return {
            "version": self.version, "sha256": self.sha256, "styles": list(self.styles), "settings": {d: list(v) for d, v in self.settings.items()},
            "paraphrase_pool": self.paraphrase_pool, "groups": list(self.groups),
            "ranges": [{"name": c.name, "phenomenon": c.phenomenon, "var": c.var, "side": c.side, "cut": c.cut, "scenes": list(c.scenes)} for c in self.ranges],
            "seed_ranges": self.seed_ranges,
        }  # fmt: skip


@functools.lru_cache(maxsize=2)
def load_holdout(path: Path = HOLDOUT_PATH) -> Holdout:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    cells = tuple(RangeCell(c["name"], c["phenomenon"], c["var"], c["side"], c["cut"], tuple(c["scenes"])) for c in data["ranges"])
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
        return None if obs.place is None else float(max(obs.place.ages.values()))
    if var == "stale_age":
        return None if obs.stale is None else float(obs.stale.age)
    if var == "corner_lateral":
        return None if obs.corner is None else float(obs.corner.lateral.value)
    raise ValueError(var)


def cells_hit(holdout: Holdout, obs: Observation) -> list[str]:
    """관측이 시험 쪽에 들어간 T3 칸들."""
    out = []
    for cell in holdout.ranges:
        if obs.scene not in cell.scenes:
            continue
        value = observed_value(obs, cell.var)
        if value is not None and cell.in_test(value, obs.body):
            out.append(cell.name)
    return out


def ranges_for(holdout: Holdout, split: Split, scene: str, body: str) -> dict[str, tuple[float, float]]:
    """생성기가 변수를 뽑을 범위(변수 → (하한, 상한)): 학습 쪽 나눔은 모든 칸을 학습 쪽으로, T3는 강제한 칸을 시험 쪽으로."""
    out: dict[str, tuple[float, float]] = {}

    def narrow(var: str, bounds: tuple[float, float]) -> None:
        lo, hi = out.get(var, (-1e9, 1e9))
        out[var] = (max(lo, bounds[0]), min(hi, bounds[1]))

    for cell in holdout.ranges:
        if scene not in cell.scenes:
            continue
        if split.cell == cell.name:
            narrow(cell.var, cell.test_bounds(body))
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
