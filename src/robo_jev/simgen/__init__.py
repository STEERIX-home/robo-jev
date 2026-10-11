"""시뮬레이션 판단 데이터 엔진 (docs/12, Task v0a).

판단이 필요한 순간 하나를 만들고(장면·몸·상대·용기), 로봇 감각만 담은 글과 질문을 쓰고, 로봇이 모르는 숨은 값을 여러 번 바꿔 본
사후 세계에서 선택지마다 갈라 굴린 결과로 확률 정답을 붙인다. 레코드 형식은 `simgen-request/0`(README.md).
"""

GENERATOR_VERSION = "v0a"
SCHEMA = "simgen-request/0"
