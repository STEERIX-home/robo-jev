# HANDOFF — 다른 머신(DGX Spark)에서 개발을 이어가기

기준: `main` @ `df5e00a`(PR #1 merge) 이후 문서 커밋 포함, 이 파일이 있는 커밋. 검증 상태: `uv run pytest -q` → **계약 v0.3 브랜치(task-v03, 리뷰 1 수정 뒤)에서 962 passed, 0 xfailed, 0 skipped (203 s)**(E1 seed 29의 xfail은 지시 조합 예약이 고쳐 지웠다; tokenizer가 있어야 skip 0 — 보고서 `.superpowers/sdd/task-v03-contract-report.md`). Spark 인수는 2026-09-19 완료(§3-1~4).

역할 분리: **학습은 클라우드**, Spark는 에이전트 개발 루프·테스트·데이터 생성·**배포급 지연 측정**(배포·실행 장비가 Spark/Jetson급). ARM64·CUDA 실행은 2026-09-19 검증했다(GB10, driver 580.159.03, CUDA 13.0, torch 2.14.0+cu130, triton 3.8.0; fla Triton·causal-conv1d 커널 sm_121에서 활성).

## 1. 완료된 것 (docs/README.md의 구현 상태 표가 정본)

Task 1·2·3a·3b·3c·4(CPU)·5(CPU) 전부 리뷰 통과·merge. Task 2b-G0a(Spark native 지연 선별: `candidates.yaml`·`fetch_backbone.py`·`measure_candidates.py`, `artifacts/reports/backbone-screen.json`)는 2026-09-19 Spark에서 구현·실측·리뷰 통과. 규칙 판단기·전문가 모두 E0 폐루프 완료, E1 3 seed 완료. 첫 40 에피소드 배치와 첫 100 rollout 비용 측정 완료(0.345 s/rollout → D1 128k ≈ 12.3 CPU-h). CPU 학습 경로는 D0 + 생성한 2 에피소드 batch로만 검증.

## 2. 결정 (사용자)

1. **확정(2026-09-19, Task v0.3)** — 계약 v0.3: 서식 축약 + 변화분 틱 + 후보 공간 축소(결합 키에서 `profile` 제거, 영역 쪽 push, 지시 조합 예약, K≤12) + 비키프레임 비용 허용 집합(τ=0.15)·hold∉A. 실측(리뷰 1 수정 뒤): 10물체·K=12에서 틱당 p50 332 / p95 599 / 첫 틱 880(에피소드 prefix 451을 더한 첫 호출 1,331 ≈ 2B에서 80 ms), 100틱 37K(08 §3.4); E1 sweep(seed 1~24·29·43) 퇴화 틱 49.7% → 0%, 완료 20/26 → 23/26 — 그중 A6 장면 수정이 계획을 바꾼 seed 3·16·17·22를 뺀 22편은 16/22 → 19/22, 바꾼 4편은 4/4(seed 16은 옛 코드에서 틱 0 완료) → 4/4. 버전 h0.4·ts0.5·e0.3·rj0.4(실행기 c0.5 그대로), batch-0 재생성(40편, E0·E1 40/40 완료). 이월: 놓기 국면이 운반 높이에서 여는 실행기·하네스 결함(E1 seed 15·19의 원통 구름; 08 §5)은 `--limit 1000` sweep 전에 고친다.
2. **확정(2026-09-19, Task v0.3; 리뷰 1 뒤 OOD 몫 ≈10~15 %로 조정)** — holdout 봉인(04 §5 표): 분야마다 템플릿 변형 하나·개념 하나, 로봇은 zoneF 목표 계열 + 지시 변형 3번(v1#2·v2#2, group 해시로 선택) + E1 계열 하나; OOD는 `ood_dev`/`ood_test`로 반분, QA가 누출 0을 확인한다. 봉인 id(한국어)는 그대로 두고 생성 비중(봉인 문구 10 %, 봉인 개념의 근원 장면 4~15 %)으로 몫을 맞췄다: pilot 2,000상태 분야별 10.6~12.5 %, 로봇 400편 일정 14.5 %(2,000편 11.7 %). 비로봇 개념 어휘의 선택(spatial goal-zone zoneC, dom reveal, workflow resource_offline, rules escort-policy)은 보고서(`.superpowers/sdd/task-v03-contract-report.md`)에서 확인.
3. **확정(2026-09-19)** — Spark native 실측(Task 2b-G0a, 03 §"지연 예산" 표)으로 G0b 후보 = `Qwen/Qwen3.5-2B`(주) + `Qwen/Qwen3.5-4B`(5 Hz 대비). 9B는 10 Hz 트랙 제외(FP8-9B는 G0b에서 단일 요청의 profiler·CUDA graph 귀속 측정 뒤 판단), 27B 제외. 현재 서식은 전 후보가 80 ms의 ≥7×라 결정 1(계약 v0.3)이 전제. v0.3 재실측에서도 native의 자라는 캐시에서는 2B가 `fails_10hz`(upper·지시 변경은 `deadline_fail`)이고 윈도우 크기 외삽 65~68 ms·첫 5틱 평균 71~74 ms만 예산 안(`passes_10hz_window`)이다 — 10 Hz 판정은 G0b의 정적 윈도우 `stream` 경로 실측이 한다.
4. 첫 실가중치 학습은 클라우드(5-CPU 뒤 Task 5 GPU 부분).

## 3. Spark에서 이어갈 순서

1. `git clone <origin> && cd robo-jev && uv sync`
2. tokenizer를 **같은 revision·해시로** 받는다(아래 §4의 값; `uv run python scripts/fetch_tokenizer.py --from-manifest artifacts/tokenizers/manifest.json` — manifest 파일을 먼저 복사; 또는 `--revision … --expect-sha256 …`; 다른 후보로의 fallback은 `--allow-fallback`을 줄 때만). **tokenizer 확보 뒤에** `uv run pytest -q`(962 passed, 0 xfailed 기대; skip이 남으면 tokenizer가 없는 것).
3. `.superpowers/sdd/`와 Claude 메모리 디렉터리를 §4대로 복사한다.
4. **완료(2026-09-19)** Spark 첫 측정: `uv sync --group backbone`(transformers 5.17.0·flash-linear-attention 0.5.2; causal-conv1d 1.7.0은 `uv pip install --no-build-isolation causal-conv1d==1.7.0`로 소스 빌드 ≈6분) → `uv run python scripts/fetch_backbone.py --id Qwen/Qwen3.5-2B`(4B·9B도; 재현은 `--from-manifest artifacts/models/manifest.json`) → `uv run python scripts/measure_candidates.py --config configs/model/candidates.yaml --path native --ticks 70 --report artifacts/reports/backbone-screen.json`(≈81분, BF16, 대표 입력 = D0 스트림 + 합성 하한/상한/지시 변경 + 500토큰 길이 대용 + 단일 요청). 결과: 현재 서식(≈1.85K/틱)에서 p95 모델 시간 2B 928 / 4B 2,299 / 9B 2,447 / 27B 5,640 ms, 500토큰 틱에서 123 / 305 / 378 / 886 ms(윈도우 크기 캐시로 **외삽** 83 / 197 / 278 ms) → 결정 3. 계약 v0.3 직렬화가 나오면 같은 명령에 `--profiles lower,v03_target`로 재실행.
5. **완료(2026-09-19, Task v0.3)** 결정 1~2 반영 → 하네스·직렬화 v0.3 구현 → batch-0 재생성 → B2 재측정(`artifacts/reports/tokens-b2.json`, 옛 값은 `tokens-b2-v0.json`) → 2b 재실행(`uv run python scripts/measure_candidates.py --config configs/model/candidates.yaml --path native --candidates Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B --profiles lower,upper,instruction_change --ticks 70 --report artifacts/reports/backbone-screen-v03.json`; `v03_target` 대역은 은퇴). 읽는 법: 문자 그대로의 판정은 자라는 캐시(15.7K~31K)의 p95라 2B도 `fails_10hz`이고 upper·지시 변경은 초과율 0.11·0.14로 `deadline_fail`이다; 윈도우 크기 값(2B 65~68 ms, 4B 144~149 ms)은 측정 범위보다 짧은 12.8K~13.2K로의 **외삽**(R² 0.25~0.59)이라 첫 5틱 평균(2B 71~74 ms)과 같이 읽고, 둘 다 예산 안일 때의 기계 판독 `passes_10hz_window`(2B 참, 4B 거짓; `--from-report`로 다시 요약)를 flag 옆에 둔다(03 §"지연 예산"). **다음**: G0b(2B·4B의 `stream` 경로: 정적 윈도우 KV, CUDA graph/`torch.compile`, 마스크 없는 attention 커널 — sdpa+마스크가 윈도우 틱의 ≈40%; readout-only 적응·무학습 품질) → `--limit 1000` rollout sweep → 128k → 클라우드 학습.

## 4. git 밖에 있는 것 — 전달 목록

| 항목 | 위치 | 전달/재생성 | 검증 |
| --- | --- | --- | --- |
| tokenizer | `artifacts/tokenizers/Qwen/Qwen3.8-27B/tokenizer.json` + 뿌리 `artifacts/tokenizers/manifest.json` | id `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`. 재다운로드 또는 파일 전송 | `load_tokenizer`가 manifest 해시와 대조 |
| 개발 장부·브리프·보고서·리뷰 패키지 | `.superpowers/sdd/` (≈9MB; `progress.md`가 태스크별 이월 항목의 정본) | rsync | `progress.md` 끝 항목이 "BRANCH FINISHED" 이후인지 |
| Claude 메모리 | `~/.claude/projects/-Users-user-Documents-robo-jev/memory/` | Spark의 프로젝트 경로 키(`-<경로를 -로 치환>`)로 복사 | `MEMORY.md` 한 줄 |
| backbone 가중치 | `artifacts/models/<id>/` + 뿌리 `artifacts/models/manifest.json`(2B/4B/9B/27B, 55.6 GB; revision·파일별 sha256·digest·라이선스·파라미터 수) | **전달하지 않는다** — manifest만 복사 후 `uv run python scripts/fetch_backbone.py --from-manifest artifacts/models/manifest.json --id <id>`로 같은 revision·해시 재수신 | `describe_backbone`이 적재 전 크기·소형 파일 해시(`--verify-full`이면 전부) 대조 |
| Spark 지연 선별 결과 | `artifacts/reports/backbone-screen.json`(563 KB, 틱별 기록 포함) + `backbone-screen-27b.json` | rsync(작음); 재생성은 ≈81분 GPU | JSON `environment.kernels`가 fla·causal_conv1d 활성, `verdict`가 03의 표와 일치 |
| 로봇 데이터 batch-0·rollout 100 | `artifacts/datasets/d1-robot/batch-0` (+ `rollouts/`), QA `artifacts/reports/d1-robot-batch-0-qa.json` | **전달하지 않는다** — `uv run python scripts/generate_episodes.py --config configs/data/d1_robot.yaml --count 40 --out artifacts/datasets/d1-robot/batch-0`(≈1분) 뒤 `uv run python scripts/rollout_keyframes.py --dataset artifacts/datasets/d1-robot/batch-0 --limit 100 --workers 8`로 재생성. 계약 v0.3 이전의 batch·rollout·DAgger probe는 지문 변경으로 거부된다 | manifest의 `versions` 집합이 하나(h0.4·e0.3·rj0.4), QA 위반 0·누출 0 |
| B2 토큰 측정 | `artifacts/reports/tokens-b2.json` (v0.3), `tokens-b2-v0.json` (옛 서식) | 재생성: `uv run python scripts/measure_tokens.py --episodes artifacts/datasets/d1-robot/batch-0` | 08 §3.4의 수치와 일치 |
| Spark v0.3 재실행 | `artifacts/reports/backbone-screen-v03.json` (2B·4B, lower/upper/instruction_change) | 재생성 ≈20~30분 GPU (§3-5의 명령) | 03 §"지연 예산" 표와 일치 |
| 리뷰 10·11의 재현 스크립트 | `artifacts/reviews/` | 필요 시 rsync(참고용) | — |

## 5. 작업 규칙

superpowers SDD 루프(브리프 파일 → 구현 에이전트 → 리뷰 패키지 → 리뷰 → 수정 → 재리뷰 → `progress.md` 한 줄). 커밋은 사용자 승인 하에, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. `.superpowers/`·`.venv/`·`artifacts/`·`data/`·`.claude/worktrees/`는 커밋 금지. docs/08이 로봇 스트림 계약의 정본, README의 구현 상태 표가 태스크 상태의 정본. 외부 리뷰는 코드로 검증한 뒤 동의를 받아 반영(07·09·10·11 반영 완료).
