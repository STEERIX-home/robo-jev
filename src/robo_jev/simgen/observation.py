"""관측: 로봇 감각이 보고하는 것만 담는 타입 (docs/12 §6, Task v0a 지시서 §3.1·§3.5).

서술기(:mod:`~robo_jev.simgen.narrate`)·질문 생성기(:mod:`~robo_jev.simgen.questions`)·사후 표본(:mod:`~robo_jev.simgen.posterior`)은
이 타입만 받는다 — 세계(:mod:`~robo_jev.simgen.world`)를 import하지 않는다(누출 금지, 시험이 검사). 여기 있는 수치는 모두 글에 쓰이는
값(잡음 낀 보고, 반올림 끝)이고, 숨은 값(의도, 출렁임 계수, 참 채움, 가려진 쪽의 장애물 …)은 하나도 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

from robo_jev.simgen.spec import Reading, RobotSpec

__all__ = [
    "AgentObs", "ConflictObs", "ContainerObs", "CornerObs", "Descriptor", "EnvObs", "Formatting", "LoadObs", "Observation", "PlacementObs",
    "SideObs", "SkinText", "StackObs", "StaleObs",
]  # fmt: skip


@dataclass(frozen=True)
class Descriptor:
    """상대를 가리키는 말: 처음 소개(np)와 다시 가리킬 때(ref)."""

    np: str
    ref: str


@dataclass(frozen=True)
class AgentObs:
    """움직이는 상대의 관측.

    ``track``은 위치 보고(한 개 또는 두 개)이고 ``track_ages``는 각 보고의 나이(초 전)다. 어느 축의 위치인지는 장면이 정한다:
    p1_step_in = 진로 중심선에서의 옆 위치(+ = 로봇의 왼쪽), p1_crossing = 로봇 진로 가장자리에서 상대의 가까운 면까지 거리,
    p1_head_on = **그 시각의** 둘 사이 간격(면 사이; 리뷰 1 C1). 두 개면 속도를 주지 않고(``speed`` = None) 위치 두 개와 시각만 준다
    (지시서 §3.5) — head_on에서 그 차이는 다가오는 속도(상대 + 로봇)다.
    """

    kind: str
    group: str
    descriptor: Descriptor
    track: tuple[Reading, ...]
    track_ages: tuple[float, ...]
    speed: Reading | None
    heading: str                      # step_in: 'left'/'right'(옆으로 움직이는 쪽); crossing: 'from_left'/'from_right'; head_on: 'toward'
    cue: str | None                   # 단서 열쇠(공은 None)
    cue_phrase: str | None
    ahead: Reading | None = None      # step_in: 진행 방향 앞 거리; crossing: 교차점까지 앞 거리
    lateral: Reading | None = None    # head_on: 통로 중심에서 상대의 옆 위치(+ = 로봇의 왼쪽)
    size: Reading | None = None       # head_on: 상대의 폭(전체)

    @property
    def age(self) -> float:
        """가장 최근 위치 보고의 나이."""
        return self.track_ages[-1]


@dataclass(frozen=True)
class ContainerObs:
    """들고 있는 용기의 관측: 종류·내용물·규격(알려짐), 채움(거친 추정), 뚜껑 보임 상태."""

    kind: str
    content: str                      # hot / chemical / other
    phrase: str
    diameter: float
    height: float
    fill: Reading
    fill_mode: str                    # 'phrase'(말로 거칠게) | 'percent'
    lid: str                          # 'open'(없음이 보임) | 'lid'(있음이 보임) | 'unknown'(보이지 않음)
    ref: str | None = None            # 다시 가리키는 짧은 말('the mug' / '머그잔') — 없으면 문구에서 만든다(영어)


@dataclass(frozen=True)
class LoadObs:
    """받침 위의 키 큰 물건(P3): 종류·문구·잰 높이와 바닥 폭·쏠리는 쪽 가장자리까지 거리·받침 표면(말). 무게중심·마찰은 숨은 값."""

    kind: str
    phrase: str
    ref: str
    height: Reading                   # 물건 높이 H(m)
    base: Reading                     # 바닥 폭(m)
    edge_front: Reading               # 물건 → 받침 앞 가장자리(감속·내려가며 멈출 때 쏠리는 쪽)
    edge_back: Reading                # 뒤(출발·올라갈 때)
    edge_side: Reading                # 회전 바깥쪽
    deck: str                         # catalog decks의 종류(그 표면의 마찰 범위가 사전)
    deck_phrase: str

    def aspect_shown(self) -> float:
        """글에 보인 높이 / 바닥 폭(규칙 질문용)."""
        return round(self.height.value / self.base.value, 6)


@dataclass(frozen=True)
class StackObs:
    """쌓기(P3): 위에 놓을 물건의 계획 위치(아래 물건 중심 기준, + = 로봇의 왼쪽), 두 물건 길이, 놓기 정확도, 들어 올릴 때의 쏠림 단서."""

    kind: str
    phrase: str
    ref: str
    spot: str                         # 쌓을 곳(문구)
    length_top: float                 # 위 물건 길이(쌓는 축) m — 알려짐
    length_low: float                 # 아래 받침면 길이 m — 알려짐
    offset: float                     # 계획한 중심 어긋남 d(m, + = 왼쪽) — 알려짐
    sigma_place: float                # 놓기 오차 σ(글에 적는다)
    cue: str | None                   # 'left' / 'right'(들어 올릴 때 그쪽이 무거웠다) / None
    recentre_s: float
    elsewhere_s: float
    check_s: float


@dataclass(frozen=True)
class PlacementObs:
    """놓기·끼우기 확인(P4): 물건·자리, 갈래(place/insert), 눈금 클래스, 측정값(이름 → 보고), 측정 나이(오래된 카메라), 허용 오차, 행동 시간."""

    family: str                       # place / insert
    kind: str
    cls: str                          # fine / medium / coarse
    item: str
    target: str
    readings: dict                    # 'height_mm' / 'tilt_deg' / 'force_n' (place), 'depth_mm' / 'force_n' / 'lateral_n' (insert) → Reading
    ages: dict                        # 측정 이름 → 나이 s(오래된 카메라면 > 0)
    tolerance: dict                   # 글에 적는 허용 오차
    action_times: dict                # 선택지 → 걸리는 시간 s(글에 적는다)
    stale_reason: str | None = None   # 오래된 카메라의 까닭(P4+P6)
    gauge_ratio: float = 0.0          # 주 측정의 감각 잡음 / 그 클래스의 잡음 범위 상한(T3 칸 '거친 게이지')

    def primary(self) -> tuple[str, Reading]:
        name = "height_mm" if self.family == "place" else "depth_mm"
        return name, self.readings[name]


@dataclass(frozen=True)
class StaleObs:
    """오래된 추적(P6): 진로에 서 있던 것을 a초 전에 봤고 지금은 가려졌다."""

    kind: str
    descriptor: Descriptor
    ref: str
    pronoun: str                      # 'them' / 'it'
    spot: str                         # 그 자리(문구)
    age: float                        # 마지막으로 본 때 s
    ahead: Reading                    # 그 자리까지 지금 거리(면 사이, 로봇 앞면 → 그 자리의 가까운 쪽) m
    sight: float                      # 가림이 걷혀 그 자리를 다시 보게 되는 거리(그 자리까지 남은 거리, 지도에서) m
    occluder: str


@dataclass(frozen=True)
class CornerObs:
    """모퉁이(P6): 가림(문구), 쪽(left/right), 그 모퉁이까지 앞 거리, 가림 가장자리에서 진로 가장자리까지 옆 거리, 분야의 분주함(글)."""

    phrase: str
    side: str
    ahead: Reading                    # 로봇 앞면 → 상대가 나올 줄(가림 끝) m
    lateral: Reading                  # 가림 가장자리 → 진로 가장자리 m
    busy: str                         # 'busy' / 'normal' / 'quiet'(상황 글의 말 — 출현 위험률의 사전은 분야 카드)


@dataclass(frozen=True)
class ConflictObs:
    """센서 불일치(P6): 단서(유리·먼지·햇빛·검은 표면·김), 라이다·카메라의 보고(있다/없다), 보고된 거리, 방위."""

    cue: str
    phrase: str
    lidar: bool
    camera: bool
    distance: Reading
    bearing: str                      # 'ahead' / 'left' / 'right'


@dataclass(frozen=True)
class SideObs:
    """옆 공간 한쪽: clear(비어 보임) / blocked(무언가 바로 옆) / occluded(가려서 안 보임)."""

    status: str
    phrase: str | None = None


@dataclass(frozen=True)
class EnvObs:
    """장면의 환경 관측."""

    place: str
    sides: tuple[SideObs, SideObs] | None = None          # (left, right) — 비켜설 수 있는 몸의 step_in
    corridor_width: Reading | None = None                 # head_on
    robot_lateral: float | None = None                    # head_on: 통로 중심에서 로봇의 옆 위치(위치 추정, 알려짐)
    obstacle: str | None = None                           # p5_braking: 멈춰야 할 이유
    obstacle_distance: Reading | None = None
    turn_radius: float | None = None                      # p5_turning: 지도에서 읽은 회전 반경(알려짐)
    turn_angle_deg: float | None = None
    turn_approach: Reading | None = None
    turn_direction: str | None = None                     # 'left' / 'right' (물리는 같다 — 좌우 거울 교란용)
    maneuver: str | None = None                           # P3: 'brake' / 'turn' / 'start'
    start_accels: dict | None = None                      # P3 출발: 선택지 → 가속 m/s²(알려짐)
    secure_s: float | None = None                         # P3: 멈추고 고정하는 시간(글)
    ramp: str | None = None                               # P3 경사로(문구)
    ramp_angle_deg: float | None = None
    ramp_length: float | None = None
    ramp_direction: str | None = None                     # 'down' / 'up'
    ramp_end: str | None = None                           # 내려가서 멈출 곳(문구)
    detour: str | None = None                             # 돌아가는 길(문구)
    detour_s: float | None = None
    look_s: float | None = None                           # P6: 다시 보는 데 드는 시간
    occluder: str | None = None                           # P1+P6: 상대가 나온 가림


@dataclass(frozen=True)
class SkinText:
    """상황 옷 문구(분야 카드에서 고른 것)."""

    domain: str
    setting: str
    body_name: str
    task: str
    distractors: tuple[str, ...]
    carry_where: str = ""             # 용기를 어디에 실었나('on your forks', 'on the cargo platform you are towing' …)
    lang: str = "en"                  # 상황 옷 문구의 언어(en / ko — 같은 순간이 두 언어로 쓰이지는 않는다)


@dataclass(frozen=True)
class Formatting:
    """단위 표기(관측 쪽에서 정한다 — 서술기와 규칙 질문이 같은 표기를 쓴다)."""

    distance_unit: str                # 'm' | 'cm' | 'mm'
    speed_unit: str                   # 'm/s' | 'km/h' | 'mm/s'
    time_unit: str                    # 's' | 'ms'


@dataclass(frozen=True)
class Observation:
    """판단 순간 하나의 관측 전부 — 모델 입력(상태 글·질문)과 사후 정답은 이것에서만 나온다."""

    scene: str
    domain: str
    body: str
    sensor_suite: str
    variant: str                      # p1_crossing: 'moving' | 'stopped'; 그 밖 'moving'
    robot: RobotSpec
    agent: AgentObs | None
    container: ContainerObs | None
    env: EnvObs
    skin: SkinText
    fmt: Formatting
    load: LoadObs | None = None
    stack: StackObs | None = None
    place: PlacementObs | None = None
    stale: StaleObs | None = None
    corner: CornerObs | None = None
    conflict: ConflictObs | None = None
