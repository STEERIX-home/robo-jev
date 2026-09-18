"""데이터 제작: 비로봇 단일 요청 생성기, origin group 분할, 자동 QA.

* :mod:`robo_jev.data.split`    — origin group → split (생성 **전에** 배정한다).
* :mod:`robo_jev.data.domains`  — 네 비로봇 분야의 규칙 기반 장면·질문·라벨.
* :mod:`robo_jev.data.generate` — 레코드 조립과 `python -m robo_jev.data.generate`.
* :mod:`robo_jev.data.validate` — 데이터셋 QA와 `python -m robo_jev.data.validate`.

로봇 에피소드 스트림 생성은 여기 없다 (Task 3). 다만 :func:`~robo_jev.data.split.assign_split`은
장면 계열 id만 받으므로 스트림 쪽에서도 그대로 쓴다.
"""
