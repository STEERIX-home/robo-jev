"""데이터 제작: 비로봇 단일 요청 생성기, origin group 분할, 자동 QA.

* :mod:`robo_jev.data.split`    — origin group → split (생성 **전에** 배정한다).
* :mod:`robo_jev.data.domains`  — 네 비로봇 분야의 규칙 기반 장면·질문·라벨.
* :mod:`robo_jev.data.generate` — 레코드 조립과 `python -m robo_jev.data.generate`.
* :mod:`robo_jev.data.episode`  — 로봇 에피소드 스트림 레코드의 조립과 집계.
* :mod:`robo_jev.data.validate` — 데이터셋 QA와 `python -m robo_jev.data.validate`.

에피소드를 **실행**하는 쪽(시뮬레이터·전문가·하네스)은 :mod:`robo_jev.sim`과
:mod:`robo_jev.harness`에 있다. 여기서는 그 결과를 계약에 맞는 레코드로 묶기만 한다.
:func:`~robo_jev.data.split.assign_split`은 장면 계열 id만 받으므로 두 경로가 같이 쓴다.
"""
