# HANDOFF — 다른 머신(DGX Spark)에서 개발을 이어가기

기준: `main` @ `df5e00a`(PR #1 merge) 이후 문서 커밋 포함, 이 파일이 있는 커밋. 검증 상태: `uv run pytest -q` → **885 passed, 1 xfailed**(`test_the_expert_completes_an_e1_episode[29]`, 후보 상한 — 결정 대기; tokenizer가 있어야 skip 0).

역할 분리: **학습은 클라우드**, Spark는 에이전트 개발 루프·테스트·데이터 생성·**배포급 지연 측정**(배포·실행 장비가 Spark/Jetson급). ARM64·CUDA 실행은 아직 검증하지 않았다.

## 1. 완료된 것 (docs/README.md의 구현 상태 표가 정본)

Task 1·2·3a·3b·3c·4(CPU)·5(CPU) 전부 리뷰 통과·merge. 규칙 판단기·전문가 모두 E0 폐루프 완료, E1 3 seed 완료. 첫 40 에피소드 배치와 첫 100 rollout 비용 측정 완료(0.345 s/rollout → D1 128k ≈ 12.3 CPU-h). CPU 학습 경로는 D0 + 생성한 2 에피소드 batch로만 검증.

## 2. 결정 대기 (사용자)

1. 계약 v0.3: 서식 축약 + 변화분 틱 + 후보 공간 축소(결합 키에서 `profile` 제거, push 방향, 목표 관련성 선필터) + 비키프레임 허용 집합. 근거: 틱당 토큰 실측 1,764~3,712(08 §3.4), E1 퇴화 틱 44%(README).
2. holdout 봉인(템플릿 변형 계열 + 개념 계열, 04 §5).
3. backbone 본선 2~4B(9B는 FP8 별도), 지연 게이트는 Spark에서 실측(03·05 갱신 완료).
4. 첫 실가중치 학습은 클라우드(5-CPU 뒤 Task 5 GPU 부분).

## 3. Spark에서 이어갈 순서

1. `git clone <origin> && cd robo-jev && uv sync`
2. tokenizer를 **같은 revision·해시로** 받는다(아래 §4의 값; `scripts/fetch_tokenizer.py --revision … --expect-sha256 …`, 없으면 파일 전송 뒤 해시 확인). **tokenizer 확보 뒤에** `uv run pytest -q`(885 passed, 1 xfailed 기대; skip이 남으면 tokenizer가 없는 것).
3. `.superpowers/sdd/`와 Claude 메모리 디렉터리를 §4대로 복사한다.
4. Spark 첫 측정: `scripts/measure_candidates.py`(Task 2b G0a native 경로, 아직 미구현 — 첫 태스크) 로 2~4B·9B 후보를 **대표 입력**(현재 서식 하한·상한 1.8K/3.7K, K=32, 지시 변경 틱, warm history 30틱; 계약 v0.3 뒤 ≈500)에서 BF16(+FP8 별도)으로 잰다.
5. 결정 1~2 반영 → 하네스·직렬화 v0.3 구현 → batch-0 재생성 → B2 재측정 → `--limit 1000` rollout sweep → 128k → 클라우드 학습.

## 4. git 밖에 있는 것 — 전달 목록

| 항목 | 위치 | 전달/재생성 | 검증 |
| --- | --- | --- | --- |
| tokenizer | `artifacts/tokenizers/Qwen/Qwen3.8-27B/tokenizer.json` + `manifest.json` | id `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`. 재다운로드 또는 파일 전송 | `load_tokenizer`가 manifest 해시와 대조 |
| 개발 장부·브리프·보고서·리뷰 패키지 | `.superpowers/sdd/` (≈9MB; `progress.md`가 태스크별 이월 항목의 정본) | rsync | `progress.md` 끝 항목이 "BRANCH FINISHED" 이후인지 |
| Claude 메모리 | `~/.claude/projects/-Users-user-Documents-robo-jev/memory/` | Spark의 프로젝트 경로 키(`-<경로를 -로 치환>`)로 복사 | `MEMORY.md` 한 줄 |
| 로봇 데이터 batch-0·rollout·DAgger probe | `artifacts/datasets/d1-robot/` | **전달하지 않는다** — 지문(config_digest) 변경으로 재생 시 거부됨. `uv run python scripts/generate_episodes.py --config configs/data/d1_robot.yaml --count 40 --out artifacts/datasets/d1-robot/batch-0`로 재생성 | manifest의 `versions` 집합이 하나 |
| B2 토큰 측정 | `artifacts/reports/tokens-b2.json` | 재생성: `uv run python scripts/measure_tokens.py` | 08 §3.4의 수치와 일치 |
| 리뷰 10·11의 재현 스크립트 | `artifacts/reviews/` | 필요 시 rsync(참고용) | — |

## 5. 작업 규칙

superpowers SDD 루프(브리프 파일 → 구현 에이전트 → 리뷰 패키지 → 리뷰 → 수정 → 재리뷰 → `progress.md` 한 줄). 커밋은 사용자 승인 하에, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. `.superpowers/`·`.venv/`·`artifacts/`·`data/`·`.claude/worktrees/`는 커밋 금지. docs/08이 로봇 스트림 계약의 정본, README의 구현 상태 표가 태스크 상태의 정본. 외부 리뷰는 코드로 검증한 뒤 동의를 받아 반영(07·09·10·11 반영 완료).
