"""문구 묶음(영어): 선택지 설명·짧은 이름, 질문 지시문, 기준 문구, 점수 기준(0·1의 뜻) — 언어·묶음별 (Task v0a §3.9, v0b §3·§4).

모두 관측에 있는 것(로봇 자신의 사양, 상대·용기·물건을 가리키는 말, 보고된 수치)만 채운다. 세계 모듈을 import하지 않는다.

**묶음(pool).** 문구마다 ``train``(학습·T1 …)과 ``heldout``(시험 세트 T2 전용 — 학습 생성기는 쓰지 못한다, configs/simgen/holdout-v0b.yaml)
두 목록이 있다. :func:`bank` 가 언어·묶음으로 목록을 돌려준다(한국어는 :mod:`~robo_jev.simgen.phrases_ko`).

**질문 문구가 정답과 같은 것을 묻게(Task v0b — v0c 진단의 반대 상관 점검).** 정답이 세는 원인과 시간 창을 문구에 적는다:
쏟아짐·림(뚜껑)까지 차오름·빈 높이 사용은 출렁임과 **충돌**(넘어섬·사람·옆 장애물)을 함께 센다 — 충돌이 있을 수 있는 장면이면 문구가 그것을
말한다; 닿음·가까움은 **앞으로 3초**(가까움은 지금 간격도 센다); 선택지를 하는 동안 '지나간다(pass)'는 말을 쓰지 않는다(멈춤에도 쓰이므로).
사건은 일어남을 묻는다(이중 부정 없음 — '안 넘어지나' 대신 '넘어지나').

자리 표시(영어): {ref}/{Ref} 상대, {cref} 용기, {lref} 실은 키 큰 물건, {item}/{target} 놓는 물건·자리, {obstacle} 장애물, {corner} 모퉁이의
가림, {spot} 그 자리, {action} 선택지의 짧은 이름(명령형), 그 밖 숫자는 호출자가 채운다.
"""

from __future__ import annotations

from robo_jev.simgen import textfmt
from robo_jev.simgen.observation import Observation

__all__ = [
    "ARM_VERB_KEYS",
    "POOLS",
    "VERB_KEYS",
    "bank",
    "cap",
    "container_ref",
    "option_descriptions",
    "option_short",
    "speed_text",
]

POOLS = ("train", "heldout")

VERB_KEYS = {
    "continue": "keep_going", "slow": "slow_down", "stop_gentle": "ease_to_stop", "stop_moderate": "brake_firmly",
    "stop_hard": "brake_hard", "sidestep_left": "swerve_left", "sidestep_right": "swerve_right", "retract": "retract_tool",
    "go_now": "go_now", "creep": "creep_forward", "wait": "hold_position", "turn_100": "full_speed_turn",
    "turn_80": "turn_at_80pct", "turn_60": "turn_at_60pct", "turn_45": "turn_at_45pct", "turn_30": "turn_at_30pct",
    "start_gentle": "pull_away_gently", "start_normal": "pull_away_normally", "start_quick": "pull_away_quickly", "secure_first": "secure_load_first",
    "ramp_normal": "take_ramp_normally", "ramp_slow": "take_ramp_slowly", "detour": "take_detour",
    "place_now": "set_down_now", "recentre": "recentre_first", "place_elsewhere": "put_it_elsewhere", "check_weight": "check_weight_first",
    "press_down": "press_down", "push_harder": "push_harder", "reseat": "redo_it", "call_human": "call_someone", "look_again": "look_again",
}  # fmt: skip
ARM_VERB_KEYS = {"stop_gentle": "pause_smoothly", "stop_moderate": "pause", "stop_hard": "protective_stop", "continue": "keep_moving"}


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def container_ref(phrase: str) -> str:
    """'a mug of hot cocoa' → 'the mug'; 'a double espresso' → 'the double espresso'."""
    head = phrase.split(" of ")[0]
    for article in ("an ", "a "):
        if head.startswith(article):
            return "the " + head[len(article) :]
    return head if head.startswith("the ") else "the " + head


def speed_text(obs: Observation, value: float) -> str:
    resolution = 0.01 if obs.body == "arm" else 0.05
    return textfmt.speed(value, resolution, obs.fmt.speed_unit)


def _T(train: list[str], heldout: list[str]) -> dict:
    return {"train": train, "heldout": heldout}


# ----------------------------------------------------------------------------------------------------------------
# 선택지: 짧은 이름(소문자 명령형 — 계획표의 '가능한 동작', 권고 문장, 질문의 {action})
# ----------------------------------------------------------------------------------------------------------------

_SHORT = {
    "continue": "keep going at your current speed",
    "slow": "slow to about half speed and keep going",
    "stop_gentle": "ease to a gentle stop",
    "stop_moderate": "brake firmly to a full stop",
    "stop_hard": "brake as hard as you can",
    "sidestep_left": "swerve left while slowing",
    "sidestep_right": "swerve right while slowing",
    "go_now": "start crossing now",
    "creep": "creep forward slowly",
    "wait": "wait where you are",
    "start_gentle": "pull away gently",
    "start_normal": "pull away at your normal acceleration",
    "start_quick": "pull away quickly",
    "secure_first": "stop and secure the load first",
    "ramp_normal": "take the ramp at your normal speed",
    "ramp_slow": "take the ramp slowly",
    "detour": "take the detour",
    "place_now": "set it down as planned",
    "recentre": "shift it to the centre first, then set it down",
    "place_elsewhere": "put it down somewhere else instead",
    "check_weight": "lift it slightly to check the weight first",
    "press_down": "press it down",
    "push_harder": "push it in harder",
    "reseat": "take it out and do it again",
    "call_human": "call a person to check it",
    "look_again": "look again before deciding",
}
_SHORT_ARM = {
    "continue": "keep the tool moving at its current speed",
    "slow": "slow the tool to half speed",
    "stop_gentle": "pause smoothly",
    "stop_moderate": "pause at the normal deceleration",
    "stop_hard": "make a protective stop",
    "retract": "stop and retract the tool",
}
_SHORT_HEAD_ON = {"sidestep_left": "move over to the left wall and keep going slowly", "sidestep_right": "move over to the right wall and keep going slowly"}
_SHORT_P6 = {"look_again": "pause briefly and look again", "creep": "creep up to the corner", "detour": "go around it"}
_SHORT_P4 = {"look_again": "measure it again before deciding", "continue": "leave it as it is and move on"}


def option_short(canon: str, obs: Observation) -> str:
    """선택지의 짧은 이름(소문자 명령형)."""
    if canon.startswith("turn_"):
        ratio = int(canon.split("_")[1]) / 100.0
        speed = speed_text(obs, ratio * obs.robot.speed)
        return f"take the turn at {speed}" if ratio < 1.0 else f"take the turn without slowing ({speed})"
    if obs.scene.startswith(("p4",)) and canon in _SHORT_P4:     # 놓기·끼우기 확인의 '계속'은 그대로 두고 넘어가기(팔이어도)
        return _SHORT_P4[canon]
    if obs.body == "arm" and canon in _SHORT_ARM:
        return _SHORT_ARM[canon]
    if obs.scene == "p1_head_on" and canon in _SHORT_HEAD_ON:
        return _SHORT_HEAD_ON[canon]
    if obs.scene.startswith(("p6", "p1p6")) and canon in _SHORT_P6:
        return _SHORT_P6[canon]
    return _SHORT[canon]


def _numbers(obs: Observation) -> dict[str, str]:
    r = obs.robot
    out = {
        "v": speed_text(obs, r.speed),
        "v_half": speed_text(obs, r.slow_ratio * r.speed),
        "a_g": textfmt.accel(r.decel_gentle),
        "a_m": textfmt.accel(r.decel_moderate),
        "a_h": textfmt.accel(r.decel_hard),
        "off": textfmt.distance(r.sidestep_offset, 0.05, obs.fmt.distance_unit if obs.fmt.distance_unit != "mm" else "cm"),
        "v_cruise": speed_text(obs, r.cruise),
    }
    env = obs.env
    for key, attr in (("t_secure", "secure_s"), ("t_detour", "detour_s"), ("t_look", "look_s"), ("t_recentre", "recentre_s"),
                      ("t_elsewhere", "elsewhere_s"), ("t_check", "check_s")):  # fmt: skip
        value = getattr(env, attr, None)
        if value is not None:
            out[key] = textfmt.time(value, 0.5 if value >= 5 else 0.1, obs.fmt.time_unit)
    if getattr(env, "start_accels", None):
        for name, value in env.start_accels.items():
            out[f"a_{name}"] = textfmt.accel(value)
    if obs.place is not None:
        for name, value in obs.place.action_times.items():
            out[f"t_{name}"] = textfmt.time(value, 0.5 if value >= 5 else 0.1, obs.fmt.time_unit)
    if obs.stack is not None:
        for key, value in (("t_recentre", obs.stack.recentre_s), ("t_elsewhere", obs.stack.elsewhere_s), ("t_check", obs.stack.check_s)):
            out[key] = textfmt.time(value, 0.5 if value >= 5 else 0.1, obs.fmt.time_unit)
    return out


_DESCRIPTIONS = {
    "continue": _T(["Keep going at your current speed ({v}).", "Carry on as you are, without slowing.", "Do not change anything; continue along your path at {v}."],
                   ["Hold your present speed of {v} and carry straight on."]),
    "slow": _T(["Ease off to about half your current speed ({v_half}) and keep going.", "Slow down gently to {v_half} and continue.", "Cut your speed roughly in half but keep moving."],
               ["Drop to roughly {v_half} and keep rolling."]),
    "stop_gentle": _T(["Come to a gentle stop ({a_g}).", "Ease to a stop using soft braking.", "Stop smoothly, braking at only {a_g}."],
                      ["Roll to a soft stop at {a_g}."]),
    "stop_moderate": _T(["Brake firmly to a full stop ({a_m}).", "Stop using your normal braking rate ({a_m}).", "Come to a stop with firm, standard braking."],
                        ["Pull up with your usual firm braking ({a_m})."]),
    "stop_hard": _T(["Emergency stop at maximum braking ({a_h}).", "Brake as hard as you can and stop as quickly as possible.", "Stop at full braking force ({a_h})."],
                    ["Slam on the brakes at the full {a_h}."]),
    "sidestep_left": _T(["Swerve to the left by about {off} while slowing.", "Shift left out of the way, slowing slightly.", "Veer left and keep moving slowly."],
                        ["Steer about {off} to the left and slow a little."]),
    "sidestep_right": _T(["Swerve to the right by about {off} while slowing.", "Shift right out of the way, slowing slightly.", "Veer right and keep moving slowly."],
                         ["Steer about {off} to the right and slow a little."]),
    "go_now": _T(["Start crossing now.", "Go: accelerate across immediately.", "Move off now and cross."], ["Set off across right away."]),
    "creep": _T(["Edge forward slowly to show that you intend to cross.", "Creep forward at low speed.", "Inch ahead slowly without committing to the crossing."],
                ["Nudge forward at a crawl."]),
    "wait": _T(["Stay where you are and let them pass.", "Wait at the edge until they have gone by.", "Hold your position for now."], ["Remain stationary for the moment."]),
    "start_gentle": _T(["Pull away gently ({a_start_gentle}).", "Set off with a soft acceleration of {a_start_gentle}.", "Move off slowly, accelerating at only {a_start_gentle}."],
                       ["Ease into motion at {a_start_gentle}."]),
    "start_normal": _T(["Pull away at your normal acceleration ({a_start_normal}).", "Set off as usual ({a_start_normal}).", "Start moving at the standard rate of {a_start_normal}."],
                       ["Get going at your everyday acceleration, {a_start_normal}."]),
    "start_quick": _T(["Pull away quickly ({a_start_quick}).", "Set off briskly at {a_start_quick}.", "Accelerate hard to get going ({a_start_quick})."],
                      ["Spring into motion at {a_start_quick}."]),
    "secure_first": _T(["Stop and secure the load first (about {t_secure}), then go.", "Take about {t_secure} to strap or hold the load, then move.", "Secure the load before moving; that costs about {t_secure}."],
                       ["Spend roughly {t_secure} fastening the load, then continue."]),
    "ramp_normal": _T(["Take the ramp at your normal speed and braking.", "Drive the ramp as you normally would.", "Go down (or up) the ramp at your usual speed."],
                      ["Handle the ramp at your regular pace."]),
    "ramp_slow": _T(["Take the ramp slowly, with gentle speed changes.", "Crawl over the ramp at half speed.", "Go slowly on the ramp and brake gently."],
                    ["Creep along the ramp at half your speed."]),
    "detour": _T(["Take the detour instead (about {t_detour} longer).", "Avoid the ramp and go the long way, which adds about {t_detour}.", "Use the other route; it costs about {t_detour} more."],
                 ["Go round the other way, adding roughly {t_detour}."]),
    "place_now": _T(["Set it down as planned.", "Put it down where you planned to.", "Lower it onto the stack at the planned spot."], ["Place it exactly as planned."]),
    "recentre": _T(["Shift it to the centre of the stack first (about {t_recentre}), then set it down.", "Re-align it over the middle before placing (about {t_recentre}).", "Centre it on the stack first; that costs about {t_recentre}."],
                   ["Move it to the middle, then lower it (roughly {t_recentre})."]),
    "place_elsewhere": _T(["Put it down somewhere else instead (about {t_elsewhere} more).", "Skip this stack and place it elsewhere, which takes about {t_elsewhere}.", "Find another place for it (about {t_elsewhere})."],
                          ["Leave the stack alone and set it down elsewhere (roughly {t_elsewhere})."]),
    "check_weight": _T(["Lift it slightly to feel where the weight is (about {t_check}), then decide.", "Do a quick lift test of the weight balance first (about {t_check}).", "Check how the weight sits before placing (about {t_check})."],
                       ["Tilt-test the weight distribution first (roughly {t_check})."]),
    "press_down": _T(["Press it down firmly into place (about {t_press_down}).", "Push down on it to seat it (about {t_press_down}).", "Apply a firm downward press (about {t_press_down})."],
                     ["Give it a firm push downward (roughly {t_press_down})."]),
    "push_harder": _T(["Push it in harder (about {t_push_harder}).", "Apply more insertion force (about {t_push_harder}).", "Force it the rest of the way in (about {t_push_harder})."],
                      ["Press it home with more force (roughly {t_push_harder})."]),
    "reseat": _T(["Take it out and do it again (about {t_reseat}).", "Lift it out and place it again from scratch (about {t_reseat}).", "Undo it and redo the whole move (about {t_reseat})."],
                 ["Remove it and repeat the move (roughly {t_reseat})."]),
    "call_human": _T(["Stop and call a person to check it (about {t_call_human}).", "Ask a human to look at it (about {t_call_human}).", "Hand it over to a person to check (about {t_call_human})."],
                     ["Get someone to inspect it (roughly {t_call_human})."]),
    "look_again": _T(["Take another measurement first (about {t_look_again}), then decide.", "Measure it again before deciding (about {t_look_again}).", "Re-check it with your sensors (about {t_look_again}), then choose."],
                     ["Repeat the measurement first (roughly {t_look_again})."]),
}
_DESCRIPTIONS_ARM = {
    "continue": _T(["Keep the tool moving at its current speed ({v}).", "Carry on along the planned path without slowing.", "Do not change the motion; continue at {v}."],
                   ["Let the tool carry on at {v}."]),
    "slow": _T(["Slow the tool to half speed ({v_half}).", "Reduce the tool speed to about {v_half} and continue.", "Halve the speed and keep moving."], ["Bring the tool down to about {v_half}."]),
    "stop_gentle": _T(["Pause smoothly with a soft deceleration ({a_g}).", "Bring the tool to a gentle pause.", "Pause, decelerating at only {a_g}."], ["Glide to a pause at {a_g}."]),
    "stop_moderate": _T(["Pause at the normal deceleration ({a_m}).", "Stop the tool with the standard pause ramp.", "Halt the motion at the usual rate ({a_m})."], ["Use the standard pause, {a_m}."]),
    "stop_hard": _T(["Protective stop at maximum deceleration ({a_h}).", "Trigger a protective stop and halt as fast as possible.", "Stop at full deceleration ({a_h})."], ["Halt instantly with a protective stop ({a_h})."]),
    "retract": _T(["Stop and pull the tool back, away from the hand.", "Retract the tool along its path, away from the hand.", "Halt and back the tool off."], ["Withdraw the tool away from the hand."]),
}
_DESCRIPTIONS_HEAD_ON = {
    "sidestep_left": _T(["Move over to the left wall and keep going slowly.", "Hug the left side of the aisle and continue at reduced speed.", "Shift left as far as the aisle allows and keep moving slowly."],
                        ["Tuck in against the left wall and carry on slowly."]),
    "sidestep_right": _T(["Move over to the right wall and keep going slowly.", "Hug the right side of the aisle and continue at reduced speed.", "Shift right as far as the aisle allows and keep moving slowly."],
                         ["Tuck in against the right wall and carry on slowly."]),
}
_DESCRIPTIONS_P6 = {
    "look_again": _T(["Pause briefly (about {t_look}) to look again, then decide.", "Stop for about {t_look} and re-check with your sensors before choosing.", "Take about {t_look} to get a fresh look first."],
                     ["Hold for roughly {t_look} to re-scan, then pick."]),
    "creep": _T(["Creep up to the corner at a crawl.", "Inch forward to the corner slowly.", "Slow right down and edge up to the corner."], ["Approach the corner at walking-frame speed."]),
    "detour": _T(["Go around it the other way (about {t_detour} longer).", "Avoid it by taking the other route (about {t_detour} more).", "Steer around the area; that adds about {t_detour}."],
                 ["Route around it, adding roughly {t_detour}."]),
}
_DESCRIPTIONS_P4 = {
    "continue": _T(["Leave it as it is and move on to the next step.", "Accept it as placed and carry on.", "Do nothing more to it and continue with the task."],
                   ["Let it stand and proceed without further checks."]),
}
_DESCRIPTIONS_TURN = _T(["Take the corner at {speed}.", "Slow to {speed} for the bend.", "Go through the turn at {speed}."], ["Round the bend at {speed}."])
_DESCRIPTIONS_TURN_FULL = _T(["Take the corner without slowing ({speed}).", "Keep your current speed ({speed}) through the bend.", "Do not slow down for the turn ({speed})."],
                             ["Carry your full {speed} through the turn."])


def option_descriptions(canon: str, obs: Observation, pool: str = "train") -> list[str]:
    """선택지 설명의 바꿔 쓰기(학습 묶음은 셋 이상)."""
    if canon.startswith("turn_"):
        ratio = int(canon.split("_")[1]) / 100.0
        speed = speed_text(obs, ratio * obs.robot.speed)
        bank = _DESCRIPTIONS_TURN_FULL if ratio >= 1.0 else _DESCRIPTIONS_TURN
        return [t.format(speed=speed) for t in bank[pool]]
    numbers = _numbers(obs)
    if obs.scene.startswith("p4") and canon in _DESCRIPTIONS_P4:
        bank = _DESCRIPTIONS_P4[canon]
    elif obs.body == "arm" and canon in _DESCRIPTIONS_ARM:
        bank = _DESCRIPTIONS_ARM[canon]
    elif obs.scene == "p1_head_on" and canon in _DESCRIPTIONS_HEAD_ON:
        bank = _DESCRIPTIONS_HEAD_ON[canon]
    elif obs.scene.startswith(("p6", "p1p6")) and canon in _DESCRIPTIONS_P6:
        bank = _DESCRIPTIONS_P6[canon]
    else:
        bank = _DESCRIPTIONS[canon]
    return [t.format(**numbers) for t in bank[pool]]


# ----------------------------------------------------------------------------------------------------------------
# 질문 지시문·점수 기준 — 열쇠 → {train, heldout}
# ----------------------------------------------------------------------------------------------------------------

EN: dict[str, dict] = {
    # 행동
    "act.generic": _T(["What should you do now?", "Choose your next action.", "Which action do you take in this moment?", "Pick the best thing to do right now."],
                      ["What is your move at this moment?"]),
    "act.p1_step_in": _T(["{Ref} {entered} {lane}. What do you do?", "Something is in your way. Choose how to respond.", "How do you respond to {ref} in your path?"],
                         ["{Ref} is now in {lane}. Your response?"]),
    "act.p1_crossing": _T(["{Ref} is heading for your path. What do you do?", "Choose how to handle the crossing.", "How do you respond to {ref} approaching from the side?"],
                          ["{Ref} is closing in on your path from the side. Your move?"]),
    "act.p1_crossing_stopped": _T(["Do you go now or wait?", "You are waiting at the edge of the crossing. What do you do?", "Choose when to cross."], ["Cross now, or hold back?"]),
    "act.p1_head_on": _T(["You and {ref} are about to meet in this aisle. What do you do?", "How do you handle the oncoming {ref_bare}?", "Choose how to deal with meeting {ref} in the aisle."],
                         ["{Ref} is coming down the aisle toward you. Your move?"]),
    "act.p5_braking": _T(["You have to stop before {obstacle}. How do you brake?", "Choose how hard to brake.", "Pick a braking strength for this stop."], ["Which braking do you use to stop before {obstacle}?"]),
    "act.p5_turning": _T(["How fast do you take the corner?", "Choose a speed for the upcoming turn.", "Pick your speed through the bend."], ["At what speed do you take this bend?"]),
    "act.p1p5_step_in_carrying": _T(["{Ref} {entered} {lane} while you are carrying {cref}. What do you do?", "Choose how to respond, keeping both {ref} and {cref} in mind.",
                                     "How do you react, given what is ahead of you and the liquid you carry?"], ["With {cref} on board and {ref} in {lane}, what now?"]),
    "act.p3_brake": _T(["You have to stop before {obstacle} with {lref} on board. How do you brake?", "Choose how hard to brake, keeping {lref} in mind.", "Pick a braking strength for this stop."],
                       ["Which braking do you use to stop before {obstacle} without losing {lref}?"]),
    "act.p3_turn": _T(["How fast do you take the corner with {lref} on board?", "Choose a speed for the turn, keeping {lref} in mind.", "Pick your speed through the bend."],
                      ["At what speed do you take this bend with {lref} on board?"]),
    "act.p3_start": _T(["How do you pull away with {lref} on board?", "Choose how to set off.", "Pick how to start moving, keeping {lref} in mind."], ["How do you get going with {lref} on board?"]),
    "act.p3_ramp": _T(["How do you deal with {ramp} while carrying {lref}?", "Choose how to handle the ramp.", "Pick how to get past the ramp with {lref} on board."],
                      ["What do you do about {ramp} with {lref} on board?"]),
    "act.p3_place_stack": _T(["How do you put {sref} down on {spot}?", "Choose how to place {sref}.", "Pick what to do with {sref} before letting go."], ["What do you do with {sref} at {spot}?"]),
    "act.p3p5": _T(["With {lref} and {cref} on board, how do you handle this?", "Choose what to do, keeping both {lref} and {cref} in mind.", "Pick your action for the load you carry."],
                   ["{Lref} and {cref} are both on board. Your move?"]),
    "act.p4_place_check": _T(["You have just set {item} down in {target}. What do you do next?", "Choose what to do about {item} now.", "Decide how to proceed with {item}."],
                             ["{Item} is in {target}. What is your next step?"]),
    "act.p4_insert": _T(["You have just pushed {item} into {target}. What do you do next?", "Choose what to do about {item} now.", "Decide how to proceed with the insertion."],
                        ["{Item} has been pushed into {target}. What next?"]),
    "act.p6_stale_track": _T(["You can no longer see {ref}. What do you do?", "Choose how to approach the spot where {ref} was.", "How do you proceed toward {spot}?"],
                             ["{Ref} is out of sight. Your move?"]),
    "act.p6_blind_corner": _T(["You are approaching {corner} and cannot see past it. What do you do?", "Choose how to approach the blind spot.", "How do you pass {corner}?"],
                              ["{Corner} hides what is beyond it. Your move?"]),
    "act.p6_sensor_conflict": _T(["Your sensors disagree about what is ahead. What do you do?", "Choose how to handle the conflicting readings.", "How do you proceed?"],
                                 ["Lidar and camera do not agree. Your move?"]),
    "act.p1p6": _T(["{Ref} {entered} {lane} from behind {occluder}. What do you do?", "Choose how to respond, given what you can and cannot see.", "How do you respond to {ref} coming out into your path?"],
                   ["{Ref} has appeared from behind {occluder}. Your move?"]),
    # 기준 — 비용 덮어쓰기와 같은 뜻(리뷰 1 m4)
    "crit.avoid_contact": _T(["Choose the quickest option that does not touch {ref} or anything else; any other risk counts as usual.",
                              "Among the options that avoid all contact (with {ref} or with anything else), pick the best one, weighing other risks and delay as usual.",
                              "Rule for this question: touching {ref} or anything else is not acceptable; apart from that, weigh risks and delay as usual. Which option?"],
                             ["Contact of any kind is ruled out here; otherwise judge risks and delay as you normally would. Which option?"]),
    "crit.avoid_spill": _T(["Choose the fastest option that does not spill anything from {cref}; any other risk counts as usual.",
                            "Pick the quickest option that keeps {cref} from spilling, weighing any other risk as usual.",
                            "Spilling is not acceptable here; apart from that, weigh risks and delay as usual. Which option?"],
                           ["Nothing may spill from {cref}; judge everything else as usual. Which option?"]),
    "crit.time_critical": _T(["This run is urgent: each second of delay counts three times as much as usual, while the cost of any accident is unchanged. What do you do?",
                              "You are running late, so every second of delay weighs three times the usual; risks are weighed as usual. Choose an action.",
                              "Time-critical job: count each second of delay three times as heavily as usual, with accidents costed as usual. Which action?"],
                             ["Delay costs triple today, while accidents cost what they always do. Which action?"]),
    "crit.avoid_fall": _T(["Choose the quickest option that does not let anything you carry fall or topple; any other risk counts as usual.",
                           "Nothing on board may tip over or fall off; apart from that, weigh risks and delay as usual. Which option?",
                           "Pick the fastest option that keeps the load standing and on board, weighing other risks as usual."],
                          ["The load must not topple or drop; judge everything else as usual. Which option?"]),
    "crit.no_defect": _T(["Choose the quickest option that does not leave {item} badly seated or damaged; any other cost counts as usual.",
                          "Leaving {item} out of place or broken is not acceptable; apart from that, weigh time and costs as usual. Which option?",
                          "Pick the fastest option that ends with {item} properly in place and undamaged, weighing other costs as usual."],
                         ["{Item} must end up correctly in place and intact; judge the rest as usual. Which option?"]),
    # 핵심 noul
    "key.stops_short": _T(["If you brake firmly now, will you come to rest before reaching {ref}'s current position?",
                           "Is your firm-braking stopping distance, reaction included, shorter than the gap to {ref}?",
                           "Braking firmly from now on, do you stop short of where {ref} is now?"],
                          ["With firm braking starting now, do you halt before the spot where {ref} is standing?"]),
    "key.stops_short_arm": _T(["If you pause at the normal deceleration now, does the tool stop before reaching {ref}?",
                               "Is the tool's normal pause distance, reaction included, shorter than its gap to {ref}?",
                               "Pausing at the standard rate from now, does the tool come to rest short of {ref}?"],
                              ["With the standard pause starting now, does the tool halt before {ref}?"]),
    "key.clears_first": _T(["If you keep your speed and {ref} keeps {poss} current pace, will you be clear of {ref}'s line at least 0.3 s before {ref} reaches your path?",
                            "Keeping your speed, do you get across {ref}'s line with at least 0.3 s to spare, assuming {ref} does not change pace?",
                            "At constant speeds on both sides, would you clear the crossing at least 0.3 s ahead of {ref}?"],
                           ["If neither of you changes speed, are you past {ref}'s line with 0.3 s or more in hand?"]),
    "key.clears_first_stopped": _T(["If you start now and {ref} keeps {poss} current pace, will you be across {ref}'s line at least 0.3 s before {ref} reaches your path?",
                                    "Going right away, do you clear the crossing with at least 0.3 s to spare, assuming {ref} does not change pace?",
                                    "If you move off now, would you be past the crossing line at least 0.3 s ahead of {ref} (both at constant pace)?"],
                                   ["Setting off immediately, are you over {ref}'s line with 0.3 s or more in hand?"]),
    "key.passable": _T(["If you and {ref} both keep to opposite walls, is the aisle wide enough to pass with at least {clear} of spare width in total?",
                        "Could you and {ref} get past each other here with {clear} of combined side clearance, hugging opposite walls?",
                        "With both of you against opposite walls, is there at least {clear} of spare width left over?"],
                       ["Hugging opposite walls, would the two of you have {clear} or more of width to spare?"]),
    "key.stops_before_corner": _T(["If someone stepped out from behind {corner} right now, could you stop before reaching it, braking as hard as you can (reaction included)?",
                                   "Braking as hard as you can from the moment someone appears, reaction included, do you come to rest before you reach {corner}?",
                                   "Is your hardest-braking stopping distance, reaction included, shorter than the distance to {corner}?"],
                                  ["Could you halt short of {corner} with maximum braking, reaction time included, if someone appeared there now?"]),
    # 쏟아짐·뚜껑 — 출렁임과 충돌을 함께 센다
    "noul.spill_impact": _T(["If you {action}, will anything spill from {cref}, either slopping over the rim or knocked out in a collision?",
                             "Suppose you {action}. Does {cref} spill, counting both slopping over and any bump into {hazard}?",
                             "Would {cref} spill if you {action}, whether it slops over or is knocked out in a collision?"],
                            ["If you {action}, does liquid end up out of {cref}, by slopping over or by a knock?"]),
    "noul.spill_slosh": _T(["If you {action}, will anything slop over the rim of {cref}?", "Does liquid slop over the rim of {cref} if you {action}?",
                            "Suppose you {action}. Does {cref} spill?"], ["If you {action}, does any liquid escape over the edge of {cref}?"]),
    "noul.rim_impact": _T(["If you {action}, will the liquid in {cref} reach the lid, by slopping up or in a collision?",
                           "Suppose you {action}. Does the surface in {cref} hit the lid, counting any bump into {hazard}?",
                           "Would the liquid in {cref} touch the lid if you {action}, either by slopping up or in a collision?"],
                          ["If you {action}, does the liquid get up to the lid of {cref}, by slopping up or a knock?"]),
    "noul.rim_slosh": _T(["If you {action}, will the liquid in {cref} surge up against the lid?", "Does the surface in {cref} climb all the way to the lid if you {action}?",
                          "Suppose you {action}. Does the liquid in {cref} reach the lid?"], ["If you {action}, does the liquid slop up to the lid of {cref}?"]),
    "noul.contact": _T(["If you {action}, will you touch {ref} within the next 3 seconds?", "Suppose you {action}. Do you make contact with {ref} in the next 3 s?",
                        "Would you end up touching {ref} in the next 3 seconds if you {action}?"], ["If you {action}, do you bump into {ref} during the next 3 s?"]),
    "noul.contact_any": _T(["If you {action}, will you touch {ref} or anyone coming out behind {ref} within the next 3 seconds?",
                            "Suppose you {action}. Do you make contact with {ref} or a second person following, in the next 3 s?",
                            "Would you touch {ref} or someone right behind {ref} in the next 3 seconds if you {action}?"],
                           ["If you {action}, do you bump into {ref} or a follower during the next 3 s?"]),
    "noul.ball_stops_in_lane": _T(["Will {ref} come to rest inside your lane?", "Does {ref} stop rolling before it leaves your lane?", "Is {ref} going to stop in your path?"],
                                  ["Will {ref} end up lying in your lane?"]),
    "noul.in_lane_2s": _T(["Will {ref} still be in {lane} 2 seconds from now?", "Two seconds from now, is {ref} still inside {lane}?", "Is {ref} going to be in your way 2 s from now?"],
                          ["Is {ref} still inside {lane} in 2 s?"]),
    "noul.side_free": _T(["Is the space you cannot see on your {side} actually free for a swerve?", "If you swerve {side}, is there room (nothing out of sight there)?",
                          "Is the area you cannot see on your {side} empty?"], ["Is the hidden space on your {side} clear?"]),
    "noul.ball_reaches_path": _T(["Will {ref} actually reach your path before it stops rolling?", "Does {ref} roll far enough to enter your path?", "Will {ref} make it into your path?"],
                                 ["Does {ref} get as far as your path?"]),
    "noul.blocked": _T(["If you brake firmly to a stop now, will you and {ref} end up stuck facing each other?", "Stopping now, do the two of you end up blocking each other at a standstill?",
                        "If you stop here, are you and {ref} left facing each other with no one able to pass?"], ["If you stop now, do you and {ref} end up in a stand-off?"]),
    "noul.overrun": _T(["If you {action}, will you hit {obstacle} before you stop?", "Suppose you {action}. Do you reach {obstacle} before coming to rest?",
                        "Would you run into {obstacle} if you {action}?"], ["If you {action}, do you make contact with {obstacle} before stopping?"]),
    # P3
    "noul.fall": _T(["If you {action}, will {lref} topple over or slide off?", "Suppose you {action}. Does {lref} tip over or fall off?",
                     "Would {lref} fall over or off if you {action}?"], ["If you {action}, does {lref} end up toppled or dropped?"]),
    "noul.tip": _T(["If you {action}, will {lref} tip over?", "Suppose you {action}. Does {lref} topple?", "Would {lref} tip over if you {action}?"],
                   ["If you {action}, does {lref} go over?"]),
    "noul.slide": _T(["If you {action}, will {lref} slide on {deck}?", "Suppose you {action}. Does {lref} start to slip on {deck}?", "Would {lref} skid on {deck} if you {action}?"],
                     ["If you {action}, does {lref} shift on {deck}?"]),
    "noul.topple_stack": _T(["If you {action}, will {sref} topple off {spot}?", "Suppose you {action}. Does {sref} fall off the stack?", "Would {sref} tip off {spot} if you {action}?"],
                            ["If you {action}, does {sref} come off the stack?"]),
    # P4
    "noul.seated": _T(["Is {item} properly in place in {target} right now, within the stated tolerance?", "Is {item} correctly seated at this moment?",
                       "Right now, is {item} sitting correctly in {target}?"], ["At this moment, is {item} where it should be, within tolerance?"]),
    "noul.inserted": _T(["Is {item} fully home in {target} right now, within the stated tolerance?", "Is {item} completely inserted at this moment?",
                         "Right now, is {item} all the way into {target}?"], ["At this moment, is {item} pushed fully home?"]),
    "noul.ok_after": _T(["If you {action}, will {item} end up properly in place and undamaged?", "Suppose you {action}. Is {item} then correctly in place and intact?",
                         "Would {item} end up correctly in place, with nothing broken, if you {action}?"], ["If you {action}, is {item} then in place and intact?"]),
    "noul.damage": _T(["If you {action}, will {item} or {target} be damaged?", "Suppose you {action}. Does anything get broken?", "Would you damage {item} if you {action}?"],
                      ["If you {action}, does something break?"]),
    # P6
    "noul.still_there": _T(["Is {ref} still at {spot} now?", "Right now, is {ref} still standing where you last saw {obj}?", "Has {ref} stayed at {spot} since you last saw {obj}?"],
                           ["Is {ref} still there at this moment?"]),
    "noul.emerge": _T(["Will someone step out from behind {corner} into your path within the next 3 seconds?", "In the next 3 s, does anyone come out from behind {corner}?",
                       "Is someone going to emerge from behind {corner} in the next 3 seconds?"], ["Does anyone appear from behind {corner} during the next 3 s?"]),
    "noul.present": _T(["Is there really something in your path about {dist} ahead?", "Is the object one sensor reports at about {dist} actually there?",
                        "Is there a real obstacle about {dist} ahead of you?"], ["Is something truly blocking your path about {dist} ahead?"]),
    "noul.follower": _T(["Is someone else coming out right behind {ref}?", "Will a second person follow {ref} out from behind {occluder} in the next couple of seconds?",
                         "Is {ref} likely to be followed out by someone else?"], ["Does another person come out after {ref}?"]),
    "noul.contact_obstacle": _T(["If you {action}, will you hit it within the next 3 seconds?", "Suppose you {action}. Do you run into whatever is there in the next 3 s?",
                                 "Would you collide with it in the next 3 seconds if you {action}?"], ["If you {action}, do you strike it during the next 3 s?"]),
    "noul.contact_emerge": _T(["If you {action}, will you hit someone coming out from behind {corner} within the next 3 seconds?",
                               "Suppose you {action}. Do you collide with anyone emerging from behind {corner} in the next 3 s?",
                               "Would you touch someone stepping out from behind {corner} in the next 3 seconds if you {action}?"],
                              ["If you {action}, do you bump into anybody who comes out from behind {corner} during the next 3 s?"]),
    "noul.contact_stale": _T(["If you {action}, will you hit {ref} within the next 3 seconds?", "Suppose you {action}. Do you run into {ref} in the next 3 s?",
                              "Would you collide with {ref} in the next 3 seconds if you {action}?"], ["If you {action}, do you strike {ref} during the next 3 s?"]),
    # score
    "score.contact": _T(["How likely is contact with {ref} in the next 3 seconds if you {action}?", "Rate the risk of touching {ref} within the next 3 s if you {action}.",
                         "What is the chance that you touch {ref} in the next 3 seconds if you {action}?"], ["If you {action}, how probable is a bump with {ref} during the next 3 s?"]),
    "score.closeness": _T(["If you {action}, how small does the gap between you and {ref} get during the next 3 seconds (the gap right now counts too)?",
                           "Rate how close you and {ref} get at the closest moment of the next 3 s if you {action}, counting the gap right now.",
                           "If you {action}, how near are you to {ref} at the closest point within the next 3 seconds (including now)?"],
                          ["If you {action}, what is the tightest gap to {ref} over the next 3 s, counting now?"]),
    "score.headroom_impact": _T(["If you {action}, how close does the liquid in {cref} come to the rim? A collision that throws it over counts as reaching the rim.",
                                 "Rate how much of the free space above the liquid in {cref} gets used if you {action}, counting a spill from any bump as all of it.",
                                 "If you {action}, how near does the surface in {cref} climb to the rim (a collision that slops it over counts as the full way)?"],
                                ["If you {action}, how far up toward the rim of {cref} does the liquid get, a knock counting as all the way?"]),
    "score.headroom_slosh": _T(["How much of the free space above the liquid in {cref} gets used up if you {action}?", "Rate how close the liquid in {cref} comes to the rim if you {action}.",
                                "If you {action}, how near does the surface in {cref} climb to the rim?"], ["If you {action}, how far up toward the rim of {cref} does the liquid get?"]),
    "score.rim_impact": _T(["How likely is the liquid in {cref} to reach the lid if you {action}, by slopping up or in a collision?",
                            "Rate the chance that the surface in {cref} hits the lid if you {action}, counting any bump.",
                            "If you {action}, what is the chance that the liquid in {cref} touches the lid, by slopping up or in a collision?"],
                           ["If you {action}, how probable is it that the liquid reaches the lid of {cref}, slopping or knocked?"]),
    "score.rim_slosh": _T(["How likely is the liquid in {cref} to surge up to the lid if you {action}?", "Rate the chance that the surface in {cref} reaches the lid if you {action}.",
                           "If you {action}, what is the chance that the liquid in {cref} touches the lid?"], ["If you {action}, how probable is it that the liquid slops up to the lid?"]),
    "score.spill_impact": _T(["How likely is a spill from {cref} if you {action}, by slopping over or in a collision?", "Rate the chance that {cref} spills if you {action}, counting any bump.",
                              "What is the risk of spilling from {cref} if you {action}, whether by slopping over or by a knock?"],
                             ["If you {action}, how probable is it that something spills from {cref}, slopping or knocked?"]),
    "score.spill_slosh": _T(["How likely is a spill from {cref} if you {action}?", "Rate the chance that {cref} spills if you {action}.", "What is the risk of spilling from {cref} if you {action}?"],
                            ["If you {action}, how probable is it that {cref} spills?"]),
    "score.overrun": _T(["How likely are you to hit {obstacle} if you {action}?", "Rate the risk of reaching {obstacle} before you stop if you {action}.",
                         "What is the chance of running into {obstacle} if you {action}?"], ["If you {action}, how probable is it that you strike {obstacle}?"]),
    "score.fall": _T(["How likely is {lref} to topple over or slide off if you {action}?", "Rate the chance that {lref} tips or falls off if you {action}.",
                      "What is the risk that {lref} ends up toppled or dropped if you {action}?"], ["If you {action}, how probable is it that {lref} goes over or off?"]),
    "score.slide": _T(["If you {action}, how far does {lref} slide toward the edge?", "Rate how much of the distance to the edge {lref} slides if you {action}.",
                       "If you {action}, how close to the edge does {lref} slide?"], ["If you {action}, how far toward the edge does {lref} shift?"]),
    "score.topple_stack": _T(["How likely is {sref} to topple off {spot} if you {action}?", "Rate the chance that {sref} falls off the stack if you {action}.",
                              "What is the risk that {sref} tips off {spot} if you {action}?"], ["If you {action}, how probable is it that {sref} comes off the stack?"]),
    "score.unseated": _T(["If you {action}, how likely is it that {item} ends up badly in place or damaged?", "Rate the risk that {item} is left out of place or broken if you {action}.",
                          "What is the chance that {item} does not end up correctly in place and intact if you {action}?"], ["If you {action}, how probable is a bad result for {item}?"]),
    "score.damage": _T(["How likely is it that something breaks if you {action}?", "Rate the risk of damaging {item} or {target} if you {action}.",
                        "What is the chance of damage if you {action}?"], ["If you {action}, how probable is breakage?"]),
    "score.contact_stale": _T(["How likely are you to hit {ref} in the next 3 seconds if you {action}?", "Rate the risk of running into {ref} within the next 3 s if you {action}.",
                               "What is the chance of a collision with {ref} in the next 3 seconds if you {action}?"], ["If you {action}, how probable is striking {ref} during the next 3 s?"]),
    "score.contact_emerge": _T(["How likely are you to hit someone coming out from behind {corner} in the next 3 seconds if you {action}?",
                                "Rate the risk of a collision with anyone emerging from behind {corner} within the next 3 s if you {action}.",
                                "What is the chance of touching someone who steps out from behind {corner} in the next 3 seconds if you {action}?"],
                               ["If you {action}, how probable is a bump with someone from behind {corner} during the next 3 s?"]),
    "score.contact_obstacle": _T(["How likely are you to hit it in the next 3 seconds if you {action}?", "Rate the risk of running into whatever is there within the next 3 s if you {action}.",
                                  "What is the chance of a collision in the next 3 seconds if you {action}?"], ["If you {action}, how probable is striking it during the next 3 s?"]),
    "score.delay.p1": _T(["Over the next 3 seconds, how much progress do you lose if you {action}, compared with {reference}?",
                          "Rate the delay if you {action}: the progress you give up during the next 3 s relative to {reference}.",
                          "If you {action}, how far behind {reference} are you after 3 seconds, measured in seconds of travel?"],
                         ["If you {action}, how much ground do you give up in the next 3 s compared with {reference}?"]),
    "score.delay.p5_braking": _T(["If you {action}, how much time do you lose because you stop short of the spot right in front of {obstacle} and have to creep up the rest of the way?",
                                  "Rate the time lost if you {action}: stopping early means closing the remaining gap to {obstacle} again at a gentle pace.",
                                  "If you {action}, how long does it take to make up the distance you leave before {obstacle}, compared with stopping right at it?"],
                                 ["If you {action}, what time do you waste stopping early and creeping up to {obstacle}?"]),
    "score.delay.p5_turning": _T(["How much later do you come out of the turn if you {action} than if you took it without slowing?",
                                  "Rate the delay if you {action}, counting the slowdown, the slower arc and getting back up to speed, against not slowing at all.",
                                  "If you {action}, how much time does the turn cost you compared with taking it at your current speed?"],
                                 ["If you {action}, how much later do you finish the turn than at full speed?"]),
    "score.delay.time": _T(["How much time does it cost you if you {action}, compared with the quickest option?", "Rate the time lost if you {action}, against the fastest choice.",
                            "If you {action}, how much later are you done than with the quickest option?"], ["If you {action}, how much extra time does it take?"]),
    # 상태
    "state.intent": _T(["What will {ref} most likely do in the next few seconds?", "Which best describes what {ref} is about to do?", "Predict what {ref} will do next."],
                       ["What is {ref} about to do?"]),
    "state.ball": _T(["Where does {ref} end up?", "Where does {ref} come to rest?", "How does {ref}'s roll end?"], ["What happens to {ref}?"]),
    "state.fill": _T(["How full is {cref} most likely?", "Which fill range best matches {cref}?", "Estimate the fill level of {cref}."], ["Roughly how full is {cref}?"]),
    "state.lid": _T(["Does {cref} have a lid on?", "Is {cref} covered?", "Is there a lid on {cref}?"], ["Is {cref} closed with a lid?"]),
    "state.p4_place": _T(["What is the most likely state of {item}?", "Which best describes how {item} is sitting?", "How is {item} most likely sitting in {target}?"],
                         ["What condition is {item} in?"]),
    "state.p4_insert": _T(["What is the most likely state of {item}?", "Which best describes the insertion?", "How far and how straight did {item} most likely go in?"],
                          ["What condition is the insertion in?"]),
    "state.sensor": _T(["Which sensor is most likely right?", "Whose reading should you believe?", "Which of the two readings is most likely correct?"], ["Which sensor is telling the truth?"]),
    "state.weight_side": _T(["Which way is the weight of {sref} most likely shifted?", "Where is the heavy side of {sref} most likely?", "How is the weight inside {sref} most likely distributed?"],
                            ["Which side of {sref} is heavier?"]),
    # 다시 볼까(정보의 가치)
    "voi": _T(["Should you decide now, or first {look}?", "Is it worth taking the time to {look} before you act?", "Do you act on what you know now, or {look} first?"],
              ["Act now, or {look} first?"]),
    "voi.decide_now": _T(["Decide now with what you already know.", "Act on the current information.", "Make the call now without checking again."], ["Go with what you know now."]),
    "voi.look_first": _T(["First {look} (about {t}), then decide.", "Spend about {t} to {look}, then choose.", "Take about {t} to {look} before acting."], ["Use about {t} to {look} first."]),
}

#: 점수 기준(0·1의 뜻) — 학습·빼 둔 묶음이 같다(기준은 정답의 뜻을 정하는 정의라서 바꿔 쓰지 않는다).
ANCHORS = {
    "contact": ["0 = no chance of contact", "1 = contact is certain"],
    "closeness": ["0 = the gap stays at least {near} the whole time", "1 = you touch {ref}"],
    "headroom_impact": ["0 = the surface barely moves", "1 = it reaches the rim or spills (a collision counts as reaching the rim)"],
    "headroom_slosh": ["0 = the surface barely moves", "1 = it reaches the rim (or spills)"],
    "rim": ["0 = it certainly stays below the lid", "1 = it certainly reaches the lid"],
    "spill": ["0 = no chance of a spill", "1 = a spill is certain"],
    "delay": ["0 = no time lost", "1 = 3 seconds or more"],
    "delay_long": ["0 = no time lost", "1 = {long} or more"],
    "overrun": ["0 = you certainly stop short", "1 = you certainly hit it"],
    "fall": ["0 = it certainly stays upright and on board", "1 = it certainly topples or falls off"],
    "slide": ["0 = it does not move", "1 = it slides all the way to the edge (or off)"],
    "topple": ["0 = it certainly stays on the stack", "1 = it certainly topples off"],
    "unseated": ["0 = it certainly ends up in place and intact", "1 = it certainly ends up out of place or damaged"],
    "damage": ["0 = nothing can break", "1 = something certainly breaks"],
}


def bank(key: str, lang: str = "en", pool: str = "train") -> list[str]:
    """문구 묶음 하나(언어·묶음별)."""
    if lang == "ko":
        from robo_jev.simgen.phrases_ko import KO

        return KO[key][pool]
    return EN[key][pool]
