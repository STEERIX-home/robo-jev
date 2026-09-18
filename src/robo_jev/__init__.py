"""robojev: 로봇용 typed judgment 모델.

공개 표면은 다섯이다.

* :mod:`robo_jev.contracts`  — 입력·라벨 계약 (정본).
* :mod:`robo_jev.sim`        — E0/E1 환경과 컨트롤러 계약.
* :mod:`robo_jev.perception` — 3D 재구성 → 공통 구조화 상태.
* :mod:`robo_jev.harness`    — 스트림 요청 구성·조합 규칙과 규칙 기반 기준군.
* :mod:`robo_jev.data`       — 데이터 생성·레코드 조립·자동 QA.
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
