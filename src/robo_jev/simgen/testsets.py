"""봉인된 시험 세트 T1~T7 만들기와 봉인(manifest) (Task v0b 지시서 §4, configs/simgen/holdout-v0b.yaml).

| 세트 | 정의 |
|---|---|
| T1 | 학습과 같은 분포(같은 나눔 규칙 — 빼 둔 칸 없음), 새 seed 범위, 한국어 약 15 % |
| T2 | 레코드마다 빼 둔 축 하나 이상: 문체 ``log``, 분야의 빼 둔 다섯째 상황, 빼 둔 바꿔 쓰기 묶음(축마다 0.6으로 켜고 모두 꺼지면 다시) |
| T3 | 레코드마다 T3 칸 하나의 시험 쪽(그 칸의 장면에서; 나머지 칸은 학습 쪽) — 칸마다 같은 수 |
| T4 | 빼 둔 조합 P1+P3만 |
| T5 | 사실 하나만 바꾼 쌍(숫자 쌍 약 60 %·단어 쌍), T1 분포 |
| T6 | 기준 300개(T1 분포, 마주 옴 제외 — 거울상 불변이 아니다) × 변형 12가지(perturb.VARIANTS) + 기준 자신 |
| T7 | T1 분포, K = 1024(닫힌 꼴이 있는 정답은 닫힌 꼴) — 보정은 정답에 대해 잰다 |

seed = 세트의 seed 범위 시작 + 단위 번호(학습·세트끼리 겹치지 않는다). 칸을 만들 수 없는 seed(HeldOutError)는 범위 뒤쪽 예비 seed로 바꾼다
(레코드의 generator.seed가 실제 seed다). 쌍은 T5에만 둔다(다른 세트는 레코드끼리 독립).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from robo_jev.simgen import ID_PREFIX
from robo_jev.simgen.catalog import DOMAINS, SCENE_BODIES, SCENE_GROUP, TRAIN_GROUPS, SimgenConfig
from robo_jev.simgen.generate import HeldOutError, Moment, assemble, build_moment, pair_records, plan_units, split_spec
from robo_jev.simgen.holdout import TRAIN_STYLES, Holdout, Split, load_holdout, split_for
from robo_jev.simgen.perturb import (
    MIRROR_SCENES_EXCLUDED,
    VARIANTS,
    alt_formatting,
    format_noise,
    mirror_batch,
    mirror_canon,
    mirror_observation,
    mirror_recipe,
    reorder_request,
    strip_trailing_zeros,
)
from robo_jev.simgen.questions import KEY_STYLES, identity
from robo_jev.simgen.record import dumps
from robo_jev.simgen.rollout import available_options
from robo_jev.simgen.spec import stream

__all__ = ["SET_SIZES", "build_set", "manifest_entry", "question_map", "recipe_of", "write_set"]

SET_SIZES = {"T1": 2000, "T2": 2000, "T3": 2000, "T4": 1000, "T5": 500, "T6": 300, "T7": 2000}
T7_K = 1024
_RESERVE = 50_000        # 예비 seed: 범위 시작 + 50,000부터


class _Seeds:
    """세트의 seed 범위 안에서 seed 주기(주 seed = 시작 + 단위 번호, 예비 seed = 시작 + 50,000부터)."""

    def __init__(self, holdout: Holdout, name: str, used: list[int]):
        self.start, self.end = holdout.seed_ranges[name]
        self.reserve = self.start + _RESERVE
        self.used = used

    def main(self, unit: int) -> int:
        seed = self.start + unit
        assert seed < self.start + _RESERVE <= self.end
        return seed

    def spare(self) -> int:
        seed = self.reserve
        self.reserve += 1
        assert seed <= self.end
        return seed


def _moment(cfg: SimgenConfig, seeds: _Seeds, seed: int, domain: str, forced: dict, split: Split, variant=None, tries: int = 40) -> Moment:
    """판단 순간 하나(칸을 만들 수 없으면 예비 seed로)."""
    for _ in range(tries):
        try:
            moment = build_moment(cfg, seed, domain, forced, variant, split)
            seeds.used.append(seed)
            return moment
        except HeldOutError:
            seed = seeds.spare()
    raise HeldOutError(f"no feasible seed for {domain} {forced} in {split.name}")


def _record(cfg: SimgenConfig, seeds: _Seeds, seed: int, domain: str, forced: dict, split: Split, index: int, prefix: str, git, *, k=None,
            tries: int = 40) -> dict:  # fmt: skip
    moment = _moment(cfg, seeds, seed, domain, forced, split, tries=tries)
    return assemble(cfg, moment, index, git=git, k=k, id_prefix=prefix, forced=forced, split_spec=split_spec(split))


def _domains_for(cfg: SimgenConfig, scene: str) -> list[str]:
    return [d for d in DOMAINS if any(cfg.domain(d)["bodies"].get(b, 0) > 0 for b in SCENE_BODIES[scene])]


# ----------------------------------------------------------------------------------------------------------------
# 세트별
# ----------------------------------------------------------------------------------------------------------------


def _plain(cfg: SimgenConfig, name: str, n: int, git, holdout: Holdout, used: list[int], *, groups=TRAIN_GROUPS, k=None):
    """T1·T4·T7: 같은 나눔 규칙으로 n개(분야 × 묶음 층화, 쌍 없음)."""
    split = split_for(name, holdout)
    seeds = _Seeds(holdout, name, used)
    prefix = f"{ID_PREFIX}-{name}"
    for unit in plan_units(n, 0, groups=groups, seed_start=0, pair_share=0.0):
        forced = {"group": unit["group"]}
        yield _record(cfg, seeds, seeds.main(unit["seed"]), unit["domain"], forced, split, unit["indices"][0], prefix, git, k=k)


def _t2(cfg: SimgenConfig, n: int, git, holdout: Holdout, used: list[int]):
    """T2: 레코드마다 빼 둔 축(문체·상황·바꿔 쓰기 묶음) — 축마다 0.6, 하나도 없으면 다시 뽑는다."""
    seeds = _Seeds(holdout, "T2", used)
    prefix = f"{ID_PREFIX}-T2"
    rng = stream(0, "T2-axes")
    for unit in plan_units(n, 0, groups=TRAIN_GROUPS, seed_start=0, pair_share=0.0):
        while True:
            axes = {a: bool(rng.random() < 0.6) for a in ("style", "setting", "pool")}
            if any(axes.values()):
                break
        split = split_for("T2", holdout, styles=tuple(holdout.styles) if axes["style"] else TRAIN_STYLES,
                          settings="heldout" if axes["setting"] else "train", pool=holdout.paraphrase_pool if axes["pool"] else "train")  # fmt: skip
        forced = {"group": unit["group"]}
        if axes["style"]:
            forced["style"] = holdout.styles[0]
        yield _record(cfg, seeds, seeds.main(unit["seed"]), unit["domain"], forced, split, unit["indices"][0], prefix, git)


def _t3(cfg: SimgenConfig, n: int, git, holdout: Holdout, used: list[int]):
    """T3: 칸마다 같은 수 — 레코드마다 칸 하나의 시험 쪽(그 칸의 장면을 돌아가며, 그 장면이 되는 분야를 돌아가며; 안 되면 다음 짝)."""
    seeds = _Seeds(holdout, "T3", used)
    prefix = f"{ID_PREFIX}-T3"
    cells = holdout.ranges
    turn: dict[str, int] = {}
    for i in range(n):
        cell = cells[i % len(cells)]
        split = split_for("T3", holdout, cell=cell.name, train_ranges=True)
        record = None
        seed = seeds.main(i)
        for _ in range(len(cell.scenes) * len(DOMAINS)):
            j = turn.get(cell.name, 0)
            turn[cell.name] = j + 1
            scene = cell.scenes[j % len(cell.scenes)]
            domains = _domains_for(cfg, scene)
            domain = domains[(j // len(cell.scenes)) % len(domains)]
            try:
                record = _record(cfg, seeds, seed, domain, {"group": SCENE_GROUP[scene], "scene": scene}, split, i, prefix, git, tries=6)
                break
            except HeldOutError:
                seed = seeds.spare()
        if record is None:
            raise HeldOutError(f"T3: no record for cell {cell.name}")
        yield record


def _t5(cfg: SimgenConfig, n_pairs: int, git, holdout: Holdout, used: list[int]):
    """T5: 쌍 n_pairs개(레코드 2개씩) — 숫자 쌍(결정 여유 하나가 경계를 넘는다)과 단어 쌍(같은 사실·다른 상황 문구)."""
    split = split_for("T5", holdout)
    seeds = _Seeds(holdout, "T5", used)
    prefix = f"{ID_PREFIX}-T5"
    for unit in plan_units(2 * n_pairs, 0, groups=TRAIN_GROUPS, seed_start=0, pair_share=1.0):
        forced = {"group": unit["group"]}
        base = _moment(cfg, seeds, seeds.main(unit["seed"]), unit["domain"], forced, split)
        # 학습 세트의 쌍과 같은 규칙(수정 1 I4): 핵심이 묻는 변수 하나, 기준의 권고 그대로, 핵심 정답이 0.5를 넘는 것만
        yield from pair_records(cfg, base, unit["domain"], forced, unit["pair_kind"], split, unit["indices"], git=git, id_prefix=prefix, forced=forced,
                                split_spec=split_spec(split))  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# T6 교란
# ----------------------------------------------------------------------------------------------------------------


def recipe_of(record: dict, name: str) -> dict:
    """레코드 meta에서 질문의 정답 조리법(정체를 재는 데 쓰는 부분)."""
    meta = record["meta"]
    recipe = dict(meta["basis"][name])
    if name in meta["options"]:
        recipe["options"] = meta["options"][name]
    return recipe


def question_map(base: dict, variant: dict, mirrored: bool = False) -> dict[str, str | None]:
    """변형 레코드의 질문 → 같은 것을 묻는 기준 레코드의 질문(정답 조리법의 정체로; 거울상이면 선택지·사건 이름을 뒤집어 맞춘다)."""
    index = {identity(recipe_of(base, name)): name for name in base["request"]["questions"]}
    out = {}
    for name in variant["request"]["questions"]:
        recipe = recipe_of(variant, name)
        if mirrored:
            recipe = mirror_recipe(recipe)
        out[name] = index.get(identity(recipe))
    return out


def _t6_variant(cfg: SimgenConfig, moment: Moment, base: dict, variant: str, index: int, git, split: Split) -> dict:
    seed, obs = moment.seed, moment.obs
    rec = base["meta"]["recommendation"]
    keep = {"present": rec["present"], "canonical": rec["canonical"], "wrong": rec["wrong"]}
    t6: dict = {"recommendation": keep}
    surface: dict = {}
    use = moment
    style = base["meta"]["style"]
    if variant == "state_paraphrase":
        t6["style"] = {"facts": "prose", "prose": "planner", "planner": "facts", "log": "facts"}[style]
    elif variant == "instruction_paraphrase":
        surface = {"shift": 1}
    elif variant == "option_paraphrase":
        surface = {"desc_shift": 1}
    elif variant == "order_reversal":
        t6["request_post"] = lambda r: reorder_request(r, None)
    elif variant == "shuffle":
        t6["request_post"] = lambda r: reorder_request(r, stream(seed, "t6-shuffle"))
    elif variant == "key_change":
        current = base["meta"]["key_style"]
        surface = {"key_style": KEY_STYLES[(KEY_STYLES.index(current) + 1) % len(KEY_STYLES)]}
    elif variant == "format_noise":
        t6["state_post"] = lambda text: format_noise(text, stream(seed, "t6-noise"))
    elif variant == "one_more_decision":
        t6["extra_question"] = True
    elif variant == "units":
        use = replace(moment, obs=replace(obs, fmt=alt_formatting(obs)))
    elif variant == "number_notation":
        t6["state_post"] = strip_trailing_zeros
    elif variant == "mirror":
        use = replace(moment, obs=mirror_observation(obs))
        t6["posterior_source"] = (obs, lambda b: mirror_batch(obs, b))
        t6["options"] = tuple(mirror_canon(o) for o in available_options(obs))
        surface = {"mirror": True}
        keep = dict(keep, canonical=None if keep["canonical"] is None else mirror_canon(keep["canonical"]))
        t6["recommendation"] = keep
    elif variant == "fact_order":
        t6["fact_seed"] = int(stream(seed, "t6-facts").integers(2**31))
    else:
        raise ValueError(variant)
    t6["meta"] = {"base": base["id"], "variant": variant}
    forced = {key: value for key, value in base["generator"]["forced"].items() if key != "domain"}
    record = assemble(cfg, use, index, git=git, id_prefix=f"{ID_PREFIX}-T6-{variant}", forced=forced, surface=surface or None, split_spec=split_spec(split),
                      t6=t6)  # fmt: skip
    record["meta"]["t6"]["map"] = question_map(base, record, mirrored=variant == "mirror")
    record["meta"]["t6"]["identical_to_base"] = json.dumps(record["request"], ensure_ascii=False) == json.dumps(base["request"], ensure_ascii=False)
    return record


def _t6(cfg: SimgenConfig, n_bases: int, git, holdout: Holdout, used: list[int], stride: int = len(DOMAINS)):
    """T6: 기준 n_bases개(T1 분포) × 12 변형, 기준 자신도 쓴다(meta.t6.variant = 'base'). 마주 옴은 좌우가 다른 사전이라 거울상만 뺀다
    (11 변형 — 기준의 meta.t6.not_applicable = ['mirror'], 수정 1 결정). 기준과 요청이 글자 그대로 같은 변형(옆이 없는 장면의 거울상 등)은
    meta.t6.identical_to_base = True(불변성 점수에서 뺄 수 있게 — 수정 1 M4). ``stride``: 묶음을 바꾸는 단위 수(기본: 분야 한 바퀴 —
    분야 × 묶음 층화; 시험은 1로 묶음을 고루)."""
    split = split_for("T6", holdout)
    seeds = _Seeds(holdout, "T6", used)
    made = unit = 0
    while made < n_bases:
        domain = DOMAINS[unit % len(DOMAINS)]
        forced = {"group": TRAIN_GROUPS[(unit // stride) % len(TRAIN_GROUPS)]}
        seed = seeds.main(unit)
        unit += 1
        try:
            moment = build_moment(cfg, seed, domain, forced, None, split)
        except HeldOutError:
            continue
        seeds.used.append(seed)
        skip = ("mirror",) if moment.ctx.scene in MIRROR_SCENES_EXCLUDED else ()
        base = assemble(cfg, moment, made, git=git, id_prefix=f"{ID_PREFIX}-T6-base", forced=forced, split_spec=split_spec(split),
                        t6={"meta": {"base": None, "variant": "base", "not_applicable": list(skip)}})  # fmt: skip
        yield base
        for variant in VARIANTS:
            if variant not in skip:
                yield _t6_variant(cfg, moment, base, variant, made, git, split)
        made += 1


# ----------------------------------------------------------------------------------------------------------------
# 쓰기·봉인
# ----------------------------------------------------------------------------------------------------------------

_DEFINITIONS = {
    "T1": "same distribution as training (same split rules, no held-out cell), new seed range; ~15 % Korean",
    "T2": "every record has >= 1 held-out axis: style 'log', the domain's held-out fifth setting, the held-out paraphrase pool (each axis on with p = 0.6)",
    "T3": "every record is on the test side of exactly one T3 range cell (equal records per cell; other cells on the training side)",
    "T4": "held-out combination P1+P3 only",
    "T5": "minimal pairs, T1 distribution (numeric: the decision variable the key question asks about moved across its boundary, kept only when the key "
          "label crosses 0.5, the base's planner recommendation shown unchanged and rule thresholds placed at the base's values — the two requests differ "
          "only in that value and what is derived from it; wording: same facts, different skin text)",
    "T6": "300 bases (T1 distribution) x 12 perturbations + the base; p1_head_on bases skip the mirror (not mirror-symmetric, 11 perturbations); labels "
          "invariant (mirror mapped); variants whose request is byte-identical to the base are flagged meta.t6.identical_to_base",
    "T7": f"calibration set, T1 distribution, K = {T7_K} (all labels of a record from one weighted batch, calibrated to the closed forms where available)",
}


def build_set(cfg: SimgenConfig, name: str, n: int | None = None, git: str | None = None, holdout: Holdout | None = None):
    """세트 하나: (레코드 생성기, 쓴 seed 목록 — 생성기를 다 돌면 채워진다)."""
    holdout = holdout or load_holdout()
    n = n or SET_SIZES[name]
    used: list[int] = []
    if name == "T1":
        return _plain(cfg, name, n, git, holdout, used), used
    if name == "T4":
        return _plain(cfg, name, n, git, holdout, used, groups=tuple(holdout.groups)), used
    if name == "T7":
        return _plain(cfg, name, n, git, holdout, used, k=T7_K), used
    builder = {"T2": _t2, "T3": _t3, "T5": _t5, "T6": _t6}[name]
    return builder(cfg, n, git, holdout, used), used


def write_set(path: Path, records) -> dict:
    """JSONL로 쓰고 (개수, sha256, 걸린 시간)."""
    start = time.perf_counter()
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = dumps(record) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return {"records": count, "sha256": digest.hexdigest(), "seconds": round(time.perf_counter() - start, 2)}


def manifest_entry(name: str, path: Path, written: dict, seeds: list[int], holdout: Holdout, k: int) -> dict:
    return {
        "path": path.name, "definition": _DEFINITIONS[name], "records": written["records"], "sha256": written["sha256"],
        "seed_range": list(holdout.seed_ranges[name]), "seeds_used": [min(seeds), max(seeds)] if seeds else None, "posterior_k": k,
    }  # fmt: skip


def manifest(entries: dict, holdout: Holdout, cfg: SimgenConfig, git: str | None) -> dict:
    from robo_jev.simgen import GENERATOR_VERSION

    return {
        "version": "v0b", "generator": GENERATOR_VERSION, "generator_git": git, "catalog": cfg.versions["catalog"], "values": cfg.versions["values"],
        "skins": cfg.versions["skins"], "holdout": holdout.definition(), "sets": entries,
        "sealed_note": "Test sets only: never train on these files. The training generator cannot produce any held-out cell (generation-time rejection).",
    }  # fmt: skip


def dump_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
