"""시뮬레이션 판단 데이터 엔진 (docs/12, Task v0a·v0b).

판단이 필요한 순간 하나를 만들고(장면·몸·상대·용기), 로봇 감각만 담은 글과 질문을 쓰고, 로봇이 모르는 숨은 값을 여러 번 바꿔 본
사후 세계에서 선택지마다 갈라 굴린 결과로 확률 정답을 붙인다. 레코드 형식은 `simgen-request/0`(README.md).
"""

GENERATOR_VERSION = "v0b"     # Task v0b: P3·P4·P6·조합·한국어·봉인된 시험 세트 (v0a.1 = 리뷰 1 수정)
ID_PREFIX = "v0b"             # 레코드 id 앞머리(버전과 따로 — 같은 과제의 묶음)
SCHEMA = "simgen-request/0"
