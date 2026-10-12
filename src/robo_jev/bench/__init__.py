"""비교 벤치마크 bench-v0 (Task bench0) — 우리 판단 모델과 OpenRouter의 결정 모델에 같은 요청을 보내고 같은 지표로 잰다.

* :mod:`~robo_jev.bench.define` — 벤치 정의: 봉인 시험 세트(T1~T7)의 층화·쌍/묶음 보존 표본과 일반 읽기 세트(G1·G2) 전부,
  정의 파일(`configs/bench/bench-v0.yaml`)과의 대조, 시운전용 T1 레코드.
* :mod:`~robo_jev.bench.openrouter` — OpenRouter Decisions API 클라이언트: 우리 요청 → API 요청(score는 11단계), 응답 → 우리 예측
  형식(질문마다 listing 하나), 재시도·디스크 캐시·돈 상한(장부)·dry-run 견적. 키는 파일에서만 읽고 절대 찍지 않는다.
* :mod:`~robo_jev.bench.score` — 같은 진단 지표(:mod:`robo_jev.decisions.metrics`)로 세트·모델마다 점수표, T6 불변성, README 표.
* :mod:`~robo_jev.bench.figures` — 의존성 없는 SVG 그래프(산점도·세트별 막대), 밝은·어두운 판.
* :mod:`~robo_jev.bench.system1` — System-1 metrics on the existing predictions (Task s1score): risk bands, severe errors, direction,
  comparison, action choice, outcome category, with cluster-bootstrap and paired intervals (`scripts/bench.py system1`).

이 꾸러미는 학습을 하지 않는다. 유료 호출은 :mod:`~robo_jev.bench.openrouter` 의 실행 경로뿐이고 돈 상한 인자 없이는 돌지 않는다.
"""
