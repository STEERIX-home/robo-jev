"""규칙 적용 choice의 글(영어·한국어) — 질문에 적힌 규칙을 글에 보인 수에 적용한다 (Task v0a §3.6, v0b §1·§3).

자리 표시: {what} 무엇을 재나, {lo}/{hi}/{t} 문턱(단위 포함), 그 밖 질문 생성기의 값.
"""

from __future__ import annotations

__all__ = ["rule_texts"]

_EN = {
    "what": {"gap": "the gap to {ref}", "crossing": "{ref}'s last reported distance from your path", "head_on": "the last reported gap to {ref}",
             "obstacle": "the distance to {obstacle}", "conflict": "the reported distance to the object ahead", "stale": "the distance to {spot}",
             "corner": "the distance to {corner}"},
    "distance": {"instructions": "Site rule: classify {what} as 'far' if it is at least {hi}, 'near' if it is at least {lo} but less than {hi}, and 'close' otherwise.",
                 "far": "at least {hi}", "near": "at least {lo} but less than {hi}", "close": "less than {lo}"},
    "age": {"instructions": "Treat a sighting as 'stale' if it is older than {t}, otherwise 'fresh'. Which applies to {ref}?",
            "stale": "last seen more than {t} ago", "fresh": "last seen {t} ago or more recently"},
    "speed": {"instructions": "The speed limit in this zone is {t}. Under that limit, is your current speed 'over_limit' or 'within_limit'?",
              "over_limit": "faster than {t}", "within_limit": "{t} or slower"},
    "fill": {"instructions": "House rule: anything filled above {t} must be carried at reduced speed. Under this rule, is {cref} a 'reduced' or a 'normal' case?",
             "reduced": "filled above {t}", "normal": "filled to {t} or less"},
    "width": {"instructions": "Aisle rule: an aisle narrower than {t} is 'single_file'; otherwise it is 'two_way'. Which is this aisle?",
              "single_file": "narrower than {t}", "two_way": "{t} wide or more"},
    "turn": {"instructions": "Route rule: a turn with a radius under {t} counts as 'sharp', otherwise 'gentle'. How is the coming turn classed?",
             "sharp": "radius under {t}", "gentle": "radius {t} or more"},
    "aspect": {"instructions": "House rule: a load whose measured height is more than {t} times its measured base width must be secured before moving. Is {lref} a 'secure' or an 'as_is' case?",
               "secure": "height more than {t} × base width", "as_is": "height {t} × base width or less"},
    "ramp": {"instructions": "Route rule: a ramp steeper than {t} counts as 'steep'; otherwise 'normal'. How is {ramp} classed?",
             "steep": "steeper than {t}", "normal": "{t} or less"},
    "p4": {"instructions_height_mm": "Line rule: if the measured height is more than {t} away from nominal, mark {item} 'recheck'; otherwise 'accept'. Which applies?",
           "instructions_depth_mm": "Line rule: if the measured insertion depth is more than {t} away from fully home, mark {item} 'recheck'; otherwise 'accept'. Which applies?",
           "recheck": "more than {t} away from nominal", "accept": "within {t} of nominal"},
}  # fmt: skip


def rule_texts(lang: str = "en") -> dict:
    if lang == "ko":
        from robo_jev.simgen.phrases_ko import RULE_TEXT

        return RULE_TEXT
    return _EN
