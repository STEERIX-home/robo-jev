"""상태 choice의 선택지 글(영어·한국어) — 의도·손·공·채움·뚜껑·P4 상태·센서·무게 쏠림 (Task v0a §3.9, v0b §1·§3).

관측만 다루는 글이다(세계 모듈을 import하지 않는다). 자리 표시는 질문 생성기의 값({item}, {target}, {sref} …)을 쓴다.
"""

from __future__ import annotations

__all__ = ["state_texts"]

_EN = {
    "intent": {
        "step_in": {
            "stay": ["stop and stay in your lane", "stay where they are, inside your path"],
            "cross": ["keep going across your lane", "carry on across and out of your path"],
            "back": ["step back the way they came", "retreat out of your path on the side they came from"],
        },
        "crossing": {
            "cross": ["keep coming at the same pace and cross your path", "carry on across at their current pace"],
            "yield": ["stop and wait before your path", "hold back and let you go first"],
            "hurry": ["speed up to get across first", "hurry across ahead of you"],
        },
        "head_on": {
            "keep": ["keep coming down the middle", "hold their line in the centre"],
            "aside_left": ["move over to your left", "step aside toward your left"],
            "aside_right": ["move over to your right", "step aside toward your right"],
            "stop": ["stop and wait", "come to a halt and let you decide"],
        },
    },
    "hand": {
        "stay": ["stay in the tool path", "remain where it is, in the path"],
        "cross": ["sweep on across the path", "continue across and out of the path"],
        "back": ["pull back out of the path", "withdraw the way it came"],
        "yield": ["pause before reaching the path", "hold back short of the path"],
        "hurry": ["dart across quickly", "move across faster"],
    },
    "ball": {"reaches_path": "It rolls on into your path", "stops_short": "It stops before reaching your path",
             "stops_in_lane": "It comes to rest inside your lane", "rolls_clear": "It rolls on out of your lane"},
    "fill": {"under_70": "Less than 70 % full", "from_70_to_82": "Between 70 % and 82 % full", "from_82_to_92": "Between 82 % and 92 % full", "over_92": "More than 92 % full"},
    "lid": {"has_lid": "Yes, it has a lid on", "open": "No, it is open"},
    "p4": {
        "place": {"seated": "Properly seated in {target}", "perched": "Resting high on a guide or lip, not down in {target}", "tilted": "Tilted and wedged at an angle"},
        "insert": {"inserted": "Fully home in {target}", "stuck": "Stopped short, caught partway in", "misaligned": "Jammed at an angle, out of line"},
    },
    # [라이다가 '없다'고 할 때의 글, 라이다가 '있다'고 할 때의 글]
    "sensor": {"lidar_right": ["The lidar: there is nothing there", "The lidar: something really is there"],
               "camera_right": ["The camera: something really is there", "The camera: there is nothing there"]},
    "weight": {"toward_overhang": "Toward the end that will stick out past the edge of {spot}", "away_from_overhang": "Toward the end that will rest over {spot}"},
}  # fmt: skip


def state_texts(lang: str = "en") -> dict:
    if lang == "ko":
        from robo_jev.simgen.phrases_ko import STATE_TEXT

        return STATE_TEXT
    return _EN
