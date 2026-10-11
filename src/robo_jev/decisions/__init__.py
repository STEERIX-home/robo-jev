"""판단 요청 진단 도구 (Task v0c) — 학습 없는 Qwen이 `simgen-request/0` 판단 요청에 내는 확률을 읽고 사후 확률 정답과 비교한다.

정본 기획은 `docs/12-sim-judgment-data-engine.md`(v0a 브랜치) §1·§5·§10·§11·§14다. 엔진(v0a)의 코드에 기대지 않고 JSONL
형식에만 기댄다.

* :mod:`~robo_jev.decisions.records` — 기록 읽기·형식 검사·쌍 단위 표본 고르기.
* :mod:`~robo_jev.decisions.prompt` — 요청 → 질문마다 프롬프트 하나(같은 상태 글 + 그 질문), 한 글자 코드, seed로 섞은 순서.
* :mod:`~robo_jev.decisions.zero_shot` — 실제 backbone에서 코드 확률 읽기(바로 답 / 생각하고 답).
* :mod:`~robo_jev.decisions.metrics` — 정답(사후 확률)과의 지표: 두 읽기(한 순서·순서 평균), 기준선 대비 짝지은 구간, 기준 계열별 후회, 어려움 축, 묶음, 쌍, 권고, 위치 편향.
* :mod:`~robo_jev.decisions.report` — 요약 JSON → 사람이 읽는 markdown 표.
"""
