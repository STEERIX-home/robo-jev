"""클라우드 실행기 — 같은 run을 다른 상자에서, 돈 상한과 함께, 감사 가능하게 (docs/06 Task 6, docs/05 §6-§7).

네 조각이다::

    manifest.py           run 명세(`infra/run-manifest.schema.json`)와 상한의 산술
    launcher.py           prepare · launch · status · fetch · cancel · resume
    runner.py             **원격에서** 도는 쪽 — 상한을 스스로 재고 상태·로그를 남긴다
    providers/            백엔드(ssh · local)와 공급자 인터페이스(create/destroy/describe, 이월)

CLI는 `scripts/launch_run.py`다. 이 패키지는 torch를 import하지 않는다 — 학습을 시작할 때
:mod:`robo_jev.train` 을 그때 부른다(목록·해시 같은 원격 잡일이 torch 없는 상자에서도 돌게).
"""

from robo_jev.launch.manifest import Budget, budget_deadline_hours, budget_spend, load_manifest, save_manifest, validate

__all__ = ["Budget", "budget_deadline_hours", "budget_spend", "load_manifest", "save_manifest", "validate"]
