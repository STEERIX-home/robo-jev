"""관측: 로봇 감각이 보고하는 것만 담는 타입 (docs/12 §6, Task v0a 지시서 §3.1·§3.5).

서술기(:mod:`~robo_jev.simgen.narrate`)·질문 생성기(:mod:`~robo_jev.simgen.questions`)·사후 표본(:mod:`~robo_jev.simgen.posterior`)은
이 타입만 받는다 — 세계(:mod:`~robo_jev.simgen.world`)를 import하지 않는다(누출 금지, 시험이 검사). 여기 있는 수치는 모두 글에 쓰이는
값(잡음 낀 보고, 반올림 끝)이고, 숨은 값(의도, 출렁임 계수, 참 채움, 가려진 쪽의 장애물 …)은 하나도 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

from robo_jev.simgen.spec import Reading, RobotSpec

__all__ = ["AgentObs", "ContainerObs", "Descriptor", "EnvObs", "Formatting", "Observation", "SideObs", "SkinText"]


@dataclass(frozen=True)
class Descriptor:
    """상대를 가리키는 말: 처음 소개(np)와 다시 가리킬 때(ref)."""

    np: str
    ref: str


@dataclass(frozen=True)
class AgentObs:
    """움직이는 상대의 관측.

    ``track``은 위치 보고(한 개 또는 두 개)이고 ``track_ages``는 각 보고의 나이(초 전)다. 어느 축의 위치인지는 장면이 정한다:
    p1_step_in = 진로 중심선에서의 옆 위치(+ = 로봇의 왼쪽), p1_crossing = 상대 중심에서 로봇 진로 중심선까지의 거리,
    p1_head_on = 앞쪽 거리. 두 개면 속도를 주지 않고(``speed`` = None) 위치 두 개와 시각만 준다(지시서 §3.5).
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


@dataclass(frozen=True)
class SkinText:
    """상황 옷 문구(분야 카드에서 고른 것)."""

    domain: str
    setting: str
    body_name: str
    task: str
    distractors: tuple[str, ...]


@dataclass(frozen=True)
class Formatting:
    """단위 표기(관측 쪽에서 정한다 — 서술기와 규칙 질문이 같은 표기를 쓴다)."""

    distance_unit: str                # 'm' | 'cm' | 'mm'
    speed_unit: str                   # 'm/s' | 'km/h' | 'cm/s'
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
