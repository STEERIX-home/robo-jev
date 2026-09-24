# HANDOFF — 다른 머신(DGX Spark)에서 개발을 이어가기

기준: `main` @ `df5e00a`(PR #1 merge) 이후 문서 커밋 포함, 이 파일이 있는 커밋. 검증 상태: `uv run pytest -q` → **계약 v0.3 브랜치(task-v03, 리뷰 1 수정 뒤)에서 962 passed, 0 xfailed, 0 skipped (203 s)**(E1 seed 29의 xfail은 지시 조합 예약이 고쳐 지웠다; tokenizer가 있어야 skip 0 — 보고서 `.superpowers/sdd/task-v03-contract-report.md`). Spark 인수는 2026-09-19 완료(§3-1~4).

역할 분리: **학습은 클라우드**, Spark는 에이전트 개발 루프·테스트·데이터 생성·**배포급 지연 측정**(배포·실행 장비가 Spark/Jetson급). ARM64·CUDA 실행은 2026-09-19 검증했다(GB10, driver 580.159.03, CUDA 13.0, torch 2.14.0+cu130, triton 3.8.0; fla Triton·causal-conv1d 커널 sm_121에서 활성).

## 1. 완료된 것 (docs/README.md의 구현 상태 표가 정본)

Task 1·2·3a·3b·3c·4(CPU)·5(CPU) 전부 리뷰 통과·merge. Task 2b-G0a(Spark native 지연 선별: `candidates.yaml`·`fetch_backbone.py`·`measure_candidates.py`, `artifacts/reports/backbone-screen.json`)는 2026-09-19 Spark에서 구현·실측·리뷰 통과. 규칙 판단기·전문가 모두 E0 폐루프 완료, E1 3 seed 완료. 첫 40 에피소드 배치와 첫 100 rollout 비용 측정 완료(0.345 s/rollout → D1 128k ≈ 12.3 CPU-h). CPU 학습 경로는 D0 + 생성한 2 에피소드 batch로만 검증.

## 2. 결정 (사용자)

1. **확정(2026-09-19, Task v0.3)** — 계약 v0.3: 서식 축약 + 변화분 틱 + 후보 공간 축소(결합 키에서 `profile` 제거, 영역 쪽 push, 지시 조합 예약, K≤12) + 비키프레임 비용 허용 집합(τ=0.15)·hold∉A. 실측(리뷰 1 수정 뒤): 10물체·K=12에서 틱당 p50 332 / p95 599 / 첫 틱 880(에피소드 prefix 451을 더한 첫 호출 1,331 ≈ 2B에서 80 ms), 100틱 37K(08 §3.4); E1 sweep(seed 1~24·29·43) 퇴화 틱 49.7% → 0%, 완료 20/26 → 23/26 — 그중 A6 장면 수정이 계획을 바꾼 seed 3·16·17·22를 뺀 22편은 16/22 → 19/22, 바꾼 4편은 4/4(seed 16은 옛 코드에서 틱 0 완료) → 4/4. 버전 h0.4·ts0.5·e0.3·rj0.4(실행기 c0.5 그대로), batch-0 재생성(40편, E0·E1 40/40 완료). 놓기 국면이 운반 높이에서 여는 결함(E1 seed 13·15·19; 08 §4 놓기 규칙)은 D1-prep에서 고쳤다 — h0.5·e0.4·c0.6: 영역 안의 빈 자리·관측된 바닥 높이·hold 틱 open 없음·명령의 `place_mm`; E1 sweep 23/26 → 26/26, 놓기 높이 위 그리퍼 이벤트 16/30 → 0/33, batch-0 재생성(3,455틱, 40/40). 대조 sibling이 생성기의 1급 산출물이 됐다(04 §3·§6): 비로봇은 기본 레코드마다 사실 하나를 바꾼 sibling + 삭제 검사(pilot 2,000건 → 767쌍, QA 실패 0), 로봇은 틱 대조 쌍 `contrast/records.jsonl`(batch-0 115쌍; `d1_robot.yaml` `contrast.per_episode`). `--limit 1000` sweep(+ batch-0 전체 9,880 rollout, `rollout_keyframes.py --summarise`): 놓기 93.7 %, 파지 54.0 %(≤150 mm 85 %), 밀기 49.4 %(접근 ≤2 s 64 %), censoring 0.56 %(후보 이탈뿐), 0.49 s/rollout → 128k ≈ 17.3 CPU-h(8 worker 2.2 h). 접촉점에 닿지 못한 밀기 486건은 horizon이 아니라 대부분 **접근 중 충돌**(contact_force 308·horizon 140·censored 38)이고 ±y에 몰린다(+y 13.3 %, −y 14.8 %, +x 3.8 %; 손몸통이 물체 윗면과 겹치는 물체에서만) — D1-prep 수정 1(h0.6)이 밀기 접촉 거리를 축·손몸통별로 바꿨다(08 §4). 표: `artifacts/reports/d1-robot-sweep-{1000,full}.json`(`push_stage_by_direction`·`push_reasons_by_approach_s`). **D1 첫 버전(2026-09-20, Task D1):** 이월 수정 h0.7(놓기 정체 감시 절대 상한 45틱, 밀기 명령 구간에 stand-off 여유 — 같은 3,368 job의 3팔 A/B(`scripts/push_contact_ab.py`)에서 +y 손가락 밀기 horizon 실패 116 → 78, 경로 답과 무관한 `conflict{zone_full}`), 로봇 400편(36,690틱 — 설계의 120K틱은 상한이었다; 완료 395/400; 대조 990쌍; 대조 레코드의 경로 후보 순서도 섞음 — 400편 QA에서 정답 위치 편향이 걸려 고침), 128k 계획의 실현 100,640 rollout(성공 64.5 %, censored 0.90 %, 0.51 s/rollout ≈ 14 CPU-h) → 계보 버전 `d1-rollout-labels`(rollout 라벨 2,000: high 1,806 / low 194), 비로봇 2,000건(951 기본·760쌍), 검수 표본 900문항 = 공개 870(29 핵심 층 × 30, `sample.*`) + 봉인 30(`ood_test`, 별도 검수자용 `sealed.*` — 개발자에게 건네지 않는다), 이중 120, D-OOD stub(`configs/data/d_ood{,_single}.yaml`), Task 2c 소형 scorer 표(`artifacts/reports/tiny-scorer.json`; 실행 `python -m robo_jev.baselines.tiny_scorer`). 보고서 `.superpowers/sdd/task-d1-report.md`.
2. **확정(2026-09-19, Task v0.3; 리뷰 1 뒤 OOD 몫 ≈10~15 %로 조정)** — holdout 봉인(04 §5 표): 분야마다 템플릿 변형 하나·개념 하나, 로봇은 zoneF 목표 계열 + 지시 변형 3번(v1#2·v2#2, group 해시로 선택) + E1 계열 하나; OOD는 `ood_dev`/`ood_test`로 반분, QA가 누출 0을 확인한다. 봉인 id(한국어)는 그대로 두고 생성 비중(봉인 문구 10 %, 봉인 개념의 근원 장면 4~15 %)으로 몫을 맞췄다: pilot 2,000상태 분야별 10.6~12.5 %, 로봇 400편 일정 14.5 %(2,000편 11.7 %). 비로봇 개념 어휘의 선택(spatial goal-zone zoneC, dom reveal, workflow resource_offline, rules escort-policy)은 보고서(`.superpowers/sdd/task-v03-contract-report.md`)에서 확인.
3. **최종 확정(2026-09-20, Task 2b G0b)** — backbone = **`Qwen/Qwen3.5-2B` + `fused` 지렛대**(틱 몸통과 10개 결정 분기를 한 forward로). 실제 `stream` 경로 실측(정적 윈도우 KV·마스크 없는 flash·배치 분기; 03 §"지연 예산" 표, `artifacts/reports/backbone-stream{,-levers}.json`): 2B baseline upper p95 83.8 / batch-0(40편 1,348틱) 80.0 ms(upper 3.8 ms 미달; 100 ms를 넘긴 19틱은 모두 E1 60번째 틱의 소개 틱 800~1,011토큰), **fused(40편 재실측, 리뷰 1 I1) upper 62.7 / batch-0 60.2 ms, 100 ms 초과 2/1,348 = 0.15 %(E1 소개 틱 100.2·101.0 ms) → 10 Hz 통과(여유 17 / 20 ms)**, `all` 52.5 / 52.0 ms·초과 0(여유 27 / 28 ms; 기본 서빙 구성); 4B는 baseline 182 / 176, fused 144.6 / 122.6 ms(앞 8편)로 **5 Hz 대비**(어떤 지렛대로도 10 Hz 불가). 귀속 측정으로 절편이 weight-read(launch 빈틈 5~13 %)임을 확인해 FP8-9B는 닫음. 품질(readout-only T0 2B batch-0 dev 88.3 %·pilot dev 46.5 %; 무학습 2B 48.0 / 4B 75.7 % batch-0 dev)은 06 Task 2b 2단계 결과와 `backbone-selection.json`. 이전 단계(2026-09-19 native 선별: 2B 주·4B 5 Hz 대비·9B/27B 제외)는 그대로 전제다. **품질 미결은 닫히지 않았다(2026-09-21, Task P1; 판정은 리뷰 1 수정 라운드에서 다시 썼다 — 처음 적힌 "이 규모에서는 4B가 이긴다"는 같이 찍힌 대조군이 반증한다)**: D1에서 같은 데이터·step·seed·고정 평가 집합(`configs/eval/pilot.yaml`, 적응 5 run `eval_set.sha256 79d09793eab5`, 무학습 2 run은 16틱마다라 `c772c669c7be`)으로 재고 **run마다 자기 자신의** 상태 섞기 대조군에 대고 읽으면 판정 칸(`robot/ood_dev`의 `q_main`, 844틱 = **8편** — **옛 모집단**이다: `ood_dev` 24편 가운데 8편이고, 그 8편 목록이 정하는 기증자 회전에서 잰 값이다. Task P3가 24편 2,530틱에서 다시 쟀다, 아래)의 차는 2B LoRA(40) **+0.168**(0.570/0.402) · 2B T0(200) **+0.165**(0.454/0.289) · 2B T1(40) **+0.111**(0.712/0.601) · 4B LoRA(40) **+0.098**(0.642/0.544) · **4B T0(200) +0.072**(0.871/0.799)로, **원값이 가장 높은 4B T0의 대조군 위 여유가 가장 작다** — 원값 차 0.42는 순서가 뒤집힌다. 4B T0는 목표와 상태를 통째로 갈아 끼워도 0.799를 답해 목표를 읽는 규칙 기준군 0.821과 0.022 차다. 그리고 **어느 run도 지시 섞기 대조군을 못 넘는다**(−0.014 ~ +0.013, 전부 ±0.014 안). 전제 둘: 이 집합은 분할 전체가 아닌 8편 부분집합이고(같은 규칙 기준군이 분할 전체에서는 **0.731** — 0.090 차로 4B T0의 여유 +0.072보다 크다(2B 두 줄의 +0.165·+0.168보다는 작다)), 844틱은 8편에서 나왔는데 레코드별 예측이 남아 있지 않아 편 단위 구간을 계산할 수 없다(틱 독립이면 95 % 반폭 ±0.023~0.034, 편 안에서 완전 상관이면 ±0.23~0.35 — 모든 여유가 뒤쪽 안이다). **읽는 법(2026-09-22 Task P3가 넓힌 모집단에서 다시 쓴 것): 이 판정은 옛 8편 칸의 집계값에 대한 것이었고, 24편 2,530틱에서 주 지표(정답이 그 틱의 commitment가 **아닌** 525틱)로 다시 재면 이렇게 갈린다 — 세 가지 확인(틱 가중 구간·편 균등 구간·24번의 편 제거)을 모두 통과하는 줄이 **셋** 있고 그것이 정확히 T1 세 줄이며, fp32 master T1은 **+0.080 [+0.022, +0.165]**(편 균등 +0.100 [+0.074, +0.133])이다. 그러나 그 여유는 사실상 **한 갈래**에서 나온다 — 61틱짜리 `grasp`(목표를 읽는 규칙 판정기 1.000, 아무것도 읽지 않는 기계적 기준군 0.000)이고, 관측 게이트 210틱에서는 **어느 줄도** 자기 대조군을 넘지 못한다. 곧 "목표를 읽는다"가 보인 곳은 2,530틱 가운데 **61틱**이고, 그것도 상태 섞기 대조군에 대해서만이다: **지시 섞기 대조군을 넘는 줄은 모든 층·모든 run에서 여전히 하나도 없으므로 목표 *텍스트*는 아직 읽히지 않는다.** 4B의 원값이 더 높았던 것은 옛 칸의 사실이고(넓힌 칸에서는 0.871 → 0.726), 넓힌 칸에서도 4B T0의 읽기 층 여유는 +0.021 [−0.032, +0.100]로 0을 포함한다. 그리고 P2의 +0.257 가운데 큰 몫은 모델이 아니라 **자**였다 — 상태 섞기 대조군은 기증자에 의존하고 이 모집단에서 20.7 %의 틱이 얼어붙은 완료 상태와 섞인다.** **backbone 결정은 그대로다**(10 Hz 게이트가 선정했고 4B는 어떤 지렛대로도 들지 못한다). **이 파일럿은 5 Hz로 갈 품질 근거를 주지 않는다**; M-d 전제도 그대로다(4B의 fused/all 판정은 아직 앞 8편(E0뿐) 값이라 batch-0 40편 재측정 전에는 결정 없음). 무학습 품질은 D1에서 4B가 모든 분할에서 낫다(로봇 dev 0.776 대 0.484, 16틱마다).
4. 첫 실가중치 학습은 클라우드(5-CPU 뒤 Task 5 GPU 부분). **파일럿 예외 확정(2026-09-21, Task P1)**: D1 규모의 T0·LoRA·T1(2B)과 T0·LoRA(4B)는 Spark 한 대에서 끝났다 — 2B T0 200 step 33분(peak 10.1 GiB), 4B T0 200 step 74분(20.8 GiB), 2B LoRA 40 step 42분(26.8 GiB), 2B T1 40 step 38분(60.0 GiB). 클라우드는 D2 규모·seed 반복·4B full 학습과 **fp32 master weight가 필요한 진짜 T1**에 쓴다.

## 3. Spark에서 이어갈 순서

1. `git clone <origin> && cd robo-jev && uv sync`
2. tokenizer를 **같은 revision·해시로** 받는다(아래 §4의 값; `uv run python scripts/fetch_tokenizer.py --from-manifest artifacts/tokenizers/manifest.json` — manifest 파일을 먼저 복사; 또는 `--revision … --expect-sha256 …`; 다른 후보로의 fallback은 `--allow-fallback`을 줄 때만). **tokenizer 확보 뒤에** `uv run pytest -q`(962 passed, 0 xfailed 기대; skip이 남으면 tokenizer가 없는 것).
3. `.superpowers/sdd/`와 Claude 메모리 디렉터리를 §4대로 복사한다.
4. **완료(2026-09-19)** Spark 첫 측정: `uv sync --group backbone`(transformers 5.17.0·flash-linear-attention 0.5.2; causal-conv1d 1.7.0은 `uv pip install --no-build-isolation causal-conv1d==1.7.0`로 소스 빌드 ≈6분) → `uv run python scripts/fetch_backbone.py --id Qwen/Qwen3.5-2B`(4B·9B도; 재현은 `--from-manifest artifacts/models/manifest.json`) → `uv run python scripts/measure_candidates.py --config configs/model/candidates.yaml --path native --ticks 70 --report artifacts/reports/backbone-screen.json`(≈81분, BF16, 대표 입력 = D0 스트림 + 합성 하한/상한/지시 변경 + 500토큰 길이 대용 + 단일 요청). 결과: 현재 서식(≈1.85K/틱)에서 p95 모델 시간 2B 928 / 4B 2,299 / 9B 2,447 / 27B 5,640 ms, 500토큰 틱에서 123 / 305 / 378 / 886 ms(윈도우 크기 캐시로 **외삽** 83 / 197 / 278 ms) → 결정 3. 계약 v0.3 직렬화가 나오면 같은 명령에 `--profiles lower,v03_target`로 재실행.
5. **완료(2026-09-19, Task v0.3)** 결정 1~2 반영 → 하네스·직렬화 v0.3 구현 → batch-0 재생성 → B2 재측정(`artifacts/reports/tokens-b2.json`, 옛 값은 `tokens-b2-v0.json`) → 2b 재실행(`uv run python scripts/measure_candidates.py --config configs/model/candidates.yaml --path native --candidates Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B --profiles lower,upper,instruction_change --ticks 70 --report artifacts/reports/backbone-screen-v03.json`; `v03_target` 대역은 은퇴). 읽는 법: 문자 그대로의 판정은 자라는 캐시(15.7K~31K)의 p95라 2B도 `fails_10hz`이고 upper·지시 변경은 초과율 0.11·0.14로 `deadline_fail`이다; 윈도우 크기 값(2B 65~68 ms, 4B 144~149 ms)은 측정 범위보다 짧은 12.8K~13.2K로의 **외삽**(R² 0.25~0.59)이라 첫 5틱 평균(2B 71~74 ms)과 같이 읽고, 둘 다 예산 안일 때의 기계 판독 `passes_10hz_window`(2B 참, 4B 거짓; `--from-report`로 다시 요약)를 flag 옆에 둔다(03 §"지연 예산"). **완료(2026-09-20, Task 2b G0b)**: `stream` 경로 실측·지렛대·귀속(`uv run python scripts/measure_candidates.py --path stream --candidates Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B --profiles lower,upper,instruction_change --episodes artifacts/datasets/d1-robot/batch-0 --ticks 70 --levers baseline,fused,graphs,compile,readout_bf16,all`, `scripts/attribution.py`), readout-only T0·짧은 LoRA·무학습 점수·구간 메모리(`scripts/adapt_readout.py --mode t0|lora|zero-shot|chunk-memory`), 선정(`scripts/select_backbone.py`) → 결정 3 최종. GPU 진입점은 모두 `robo_jev.gpu`의 통합 메모리 울타리(0.6 = 73 GiB) 안에서, 긴 작업은 `choom -n 1000`으로 돌린다 — 2026-09-20에 울타리 없는 full backward가 121 GB 통합 메모리를 다 채워 커널이 데스크톱 세션을 세 번 죽였다. **완료(2026-09-20~21, Task D1)**: 128k(실현 100,640) → D1 400편 → Task 2c 표 → 검수 표본(결정 항목 1의 D1 첫 버전 문단). **완료(2026-09-21, Task P1 파일럿)**: D1에서 같은 step·seed·**고정 평가 집합**(`configs/eval/pilot.yaml`)으로 2B·4B를 나란히 돌렸다 — 돌린 run: 2b-t0, 4b-t0, 2b-zero-shot, 4b-zero-shot, 2b-lora, 4b-lora, 2b-t1. Task 5의 GPU 인수 검사도 실제 2B에서 끝났다(T0 동결 320 tensor 0 변경, LoRA·T1 gradient·parameter 이동, 저장 후 **프로세스 재시작** 재개가 비트 동일 → G0 통과, 200 step profile). **판정(D4, 2026-09-21 리뷰 1 수정 라운드에서 다시 썼다)**: 판정 칸(`robot/ood_dev`의 `q_main`, 844틱 = 8편)에서 run마다 **자기** 상태 섞기 대조군과의 차는 2B LoRA **+0.168** · 2B T0 **+0.165** · 2B T1 **+0.111** · 4B LoRA **+0.098** · **4B T0 +0.072**(0.871/0.799)로, 원값이 가장 높은 줄의 여유가 가장 작다. 4B T0는 목표·상태를 통째로 갈아 끼운 대조군에서도 0.799(규칙 기준군 0.821과 0.022 차)이고, **어느 run도 지시 섞기 대조군을 못 넘는다**(±0.014 안). 규칙 기준군은 이 8편 부분집합에서 0.821, 분할 전체에서 **0.731**이고(0.090 차), 레코드별 예측이 없어 편 단위 구간은 계산할 수 없다. **읽는 법: D1 파일럿 규모에서는 어느 적응 모델도 판정 칸에서 목표를 읽는다는 것을 보이지 못한다 — 4B의 더 높은 원값은 사실이고 그것을 의미 판단으로 돌릴 근거는 없다.** backbone 결정(2B + fused)은 지연이 정한 것이라 **그대로**이고, **이 파일럿은 5 Hz로 갈 품질 근거를 주지 않는다**(M-d 전제도 그대로: 4B의 fused/all을 batch-0 40편에서 다시 재기 전에는 5 Hz 결정 없음). 함께 나온 것: 어느 적응 run도 대조 `choice`에서 자기 무학습보다 낫지 않고(2B 0.594 → 0.490/0.281/0.344, 4B 0.688 → 0.604/0.406), 한 필드만 바뀐 대조 쌍에서 `q_main` 답이 바뀐 쌍은 20쌍 중 **0~2쌍**, **지시 대조 6쌍에서는 0~1쌍**이다(2B T0가 두 분할에서 각각 1/6, 4B LoRA가 ood_dev에서 1/6; `_all`은 LoRA 네 줄이 모두 2/20). BF16 master weight 때문에 T1은 40 step·1e-5에서 표본 24개의 4.83 %가 움직였는데 그 표본의 79.5 %가 `embed_tokens`(0.068 % 이동)라 **embedding을 빼면 23.32 %**가 움직였고 |w|가 가장 큰 norm tensor 네 개는 66~92 % 움직였다 — 좁혀진 것은 걸음 **크기**(움직인 원소 평균 |Δ|가 계획값 4.0e-4의 7~9 %)이고, fp32 master weight는 **증거 있는 미결 항목**이지 증명된 판정이 아니다 — **이 문장은 P1 시점의 읽기이고, 아래 Task P2 블록이 그 항목을 닫는다**(fp32 master로 돌린 T1이 있다: `artifacts/runs/p2-t1-qwen3.5-2b-20260921-174445`). 실행: `scripts/adapt_readout.py --config configs/train/qwen35-{2b,4b}-pilot.yaml --mode t0|lora|t1|zero-shot --eval-config configs/eval/pilot.yaml`, 인수 검사 `scripts/p1_acceptance.py --check frozen,trains,resume`; 보고서 `.superpowers/sdd/task-p1-report.md`, 표 `artifacts/reports/p1-*.json`. **완료(2026-09-21, Task P2)**: P1이 남긴 두 구멍을 메웠다. (1) **fp32 master weight** — optimizer가 학습 대상 가운데 fp32가 아닌 파라미터(= T1의 BF16 backbone)의 fp32 사본을 들고 fp32로 갱신한 뒤 되쓴다(`MasterWeightAdamW`, 설정 `fp32_master_weights` 기본 켜짐; readout·LoRA는 이미 fp32라 T0·LoRA는 P1과 같은 run이다). 측정 전에 고정한 기준 위의 시험 한 쌍이 그 차이를 붙든다(갱신폭 비 **0.9766** 대 **0.0000**, 허용 0.05). 대가는 2B에서 backward 상주 **+14.03 GiB**(예측 +14.02)와 step 시간 +0.2 %이고, 그래서 T1 구간이 10초 → **5초**다(10초는 3 step에 67.01 GiB, 40 step이면 울타리 밖; 5초 53.90; 2초 42.13 = optimizer 바닥). **4B의 T1은 이 상자에서 못 돈다** — optimizer step의 바닥만 94.0 GiB다. 되찾은 것: 같은 표본 24개에서 임베딩 뺀 이동 비율 23.32 % → **65.70 %**, 모든 원소 평균 |Δ|가 기대치의 1.93 % → **8.49 %**(4.4배). (2) **편 단위 불확실성** — `aggregate`가 질문 칸마다 편 단위 집계를 남기고 `episode_bootstrap`이 **쌍 부트스트랩**으로 여유의 구간을 낸다. 저장된 checkpoint로 판정 칸을 다시 재서(P1의 값을 소수점 셋째 자리까지 재현) 나온 것: 2B LoRA +0.168 [+0.131, +0.225] · 2B T0 +0.165 [+0.111, +0.249] · 2B T1 +0.111 [+0.066, +0.161] · 4B LoRA +0.098 [+0.060, +0.128] · **4B T0 +0.072 [−0.015, +0.134] — 유일하게 0을 포함한다**. 지시 섞기 여유는 **다섯 줄 모두 0을 포함한다**. (3) fp32 master로 다시 돌린 2B T1(같은 데이터·step·seed·평가 집합 `79d09793eab5`): 판정 칸 원값 0.712 → **0.994**인데 **자기 상태 섞기 대조군이 0.904**(목표·물체·영역·장면을 통째로 갈아 끼워도 90.4 %를 맞힌다 — 목표를 읽는 규칙 기준군 0.821보다 0.083 높다)이고 **지시 섞기 대비 여유는 −0.001 [−0.002, +0.000]**이다. 같은 5초 구간의 bf16 대조 run을 따로 돌려 가른 결과 +0.282 가운데 구간 몫은 +0.064, fp32 master 몫은 +0.218이다. **읽는 법(2026-09-21 리뷰 1 수정 라운드에서 다시 썼다): 판정 칸은 70 %가 'commitment 되풀이'라 집계 여유가 읽기를 가린다.** 844틱 가운데 **595틱(70.5 %)의 정답이 그 틱 자신의 `commitment.action_ref`**이고(commitment가 있는 604틱의 98.5 %) 상태 섞기는 그 줄을 일부러 남기므로, 아무것도 읽지 않는 기계적 정책("commitment가 있으면 그것, 없으면 `observe` 게이트 키")이 **751/844 = 0.890**을 받는다 — 섞인 모델의 0.904와 0.014 차이고, 섞인 fp32 T1은 실제로 604틱의 **97.0 %**에서 그 commitment를 글자 그대로 답한다(맞힌 763틱의 76.3 %가 그 되풀이). **정답이 commitment가 아닌 249틱으로 나누면 그림이 갈린다**: fp32 T1 **0.984 대 0.727, 여유 +0.257 [+0.073, +0.318]**(0을 포함하지 않는다 — **이 수의 큰 몫은 자다, 아래 Task P3**: 같은 249틱을 24편 회전으로 재면 대조군이 0.727이 아니라 0.960이고 여유는 +0.024다)인데 4B T0는 **0.863 대 0.904, −0.040 [−0.157, +0.125]**(0을 포함하므로 "진다"도 판정은 아니다). commitment 층에서는 fp32 T1 0.998/0.978, 4B T0 0.874/0.755이고 **일곱 줄 모두** 여유가 0을 넘는다(대조군이 답을 베껴 넘기는 층이라 당연하다). 비-commitment 층에서 **자기 대조군을 넘는 run은 fp32 T1 하나뿐**이고, 나머지 여섯 줄(2B T0 +0.008 · 2B LoRA +0.076 · 2B T1 bf16 +0.000 · 4B LoRA −0.008 · 4B T0 −0.040 · bf16 5초 +0.096)의 여유는 전부 0을 포함한다. **그러므로 P1의 판정(어느 run도 목표를 읽는다고 보이지 못했다)은 P1이 잰 일곱 run에 대해서는 그대로이고, 고쳐서 학습한 T1에는 더 이상 해당하지 않는다** — 근거는 층화 표(`artifacts/reports/p2-decision-cell-strata.json`, `scripts/decision_cell_strata.py`). 한계 둘: 249틱 가운데 **159틱(63.9 %)이 한 편**(`ep-E1-000235`; 전체 칸에서도 300/844 = 35.5 %)이고, **지시 섞기 대비 여유는 두 층 모두 0을 포함한다**(목표 **텍스트**는 여전히 읽지 않는다). (4) 함께 고친 것: **T1 checkpoint를 다시 실을 수 없던 버그**(묶인 lm_head ↔ embedding을 저장은 한 이름으로, 적재는 두 이름으로 세어 거절했다 — 평가도 재개도 같은 함수를 지난다), P1 인수 검사의 **pytest 덮개**(`tests/test_acceptance.py` 14개; 판정부를 `compare_resume`로 떼어 일곱 가지 어긋남마다 떨어지는 것을 고정), 재개 게이트가 학습 범위를 받는다(`--resume-modes t0|lora|t1`). 보고서 `.superpowers/sdd/task-p2-report.md`, 표 `artifacts/reports/p2-*.json`. **완료(2026-09-22, Task P3)**: **자를 고쳤다** — 과제·데이터를 고치기 전에 무엇을 어디서 재는지부터 정했다. (1) **모집단**: 판정 칸은 이제 **`ood_dev` 분할의 에피소드 전부를 통째로** = 24편 2,530틱이다(`configs/eval/p3-decision-cell.yaml`, `eval_set.sha256 bf80f9a54063`; 같은 규칙의 `dev` 참고 칸은 48편 3,762틱, `selection: false`). 틱은 솎지 않는다 — 스트림은 틱 0부터 재생하므로 솎기가 GPU를 아끼지 못하고, 뒤를 자르면 읽기 층 525틱 중 251틱(편 끝 `hold` 꼬리)이 사라진다. 한 편의 몫은 35.5 % → **11.9 %**, 읽기 층에서 63.9 % → **30.3 %**이고 **24편 모두가 그 층에 틱을 낸다**. (2) **주 지표**를 docs/08 §10에 못 박았다: **정답이 그 틱의 현재 commitment가 아닌 틱**(525/2,530). commitment 층(2,005)은 언제나 같이 싣되 "대조군이 답을 베껴 넘기는 층"이라는 주석 없이 인용하지 않는다. (3) **기계적 기준군**(`mechanical_baseline`, "commitment 있으면 그것, 없으면 observe")이 상시 열이 됐다 — 새 모집단에서 **0.875**(읽기 층 0.398, commitment 층 1.000). 넘지 못하는 주장은 주장이 아니다. (4) **commitment 섞기** 셋째 대조군을 만들어 재고 **표준 열은 바꾸지 않기로** 했다: 기증 틱의 참조를 그대로 실으면 이 틱의 후보 목록에 16.9 %만 들어맞고(하네스가 자리를 예약하므로 자기 참조는 2,033/2,033이 목록 안이다), 자리만 굴린 판도 commitment 층에서는 **저장된 정답을 거짓으로 만든다**(그 층 0.010). 읽기 층에서만 읽는다. (5) **일곱 run 재측정**: fp32 T1의 읽기 층 여유가 **+0.257 → +0.080 [+0.022, +0.165]**(편 균등 +0.100 [+0.074, +0.133])로 **살아남되 3분의 1**이고, 한 편을 빼도 24번 모두 0을 제외한다. **P2의 +0.257 가운데 큰 몫은 모델이 아니라 자였다** — 상태 섞기 대조군은 설정 목록의 **다음 레코드**를 기증자로 쓰고 기증자가 짧으면 마지막 틱으로 고정하는데, 8편 회전에서 `ep-E1-000235`(300틱)가 82틱짜리를 받아 **218틱(그 편의 `observe` 157틱 전부)이 얼어붙은 완료 상태와 섞였다**. 같은 249틱을 24편 회전으로 재면 같은 checkpoint가 **+0.024**다(모델 열은 844틱 전부 동일, 상태 섞기 열은 9.7 %가 다르다). (6) **여유의 출처를 갈래로 갈랐다**: 읽기 층 525틱은 `hold` 251(편 끝 꼬리, 두 열 다 0.97+) · `observe` 210(3편) · `grasp` 61(24편) · `place` 3이고, **일곱 줄 모두 `grasp`에서만 자기 대조군을 넘는다**(fp32 T1 +0.656 [+0.552, +0.840], 나머지 +0.16~+0.36; 그 갈래에서 목표를 읽는 규칙 판정기는 1.000, 기계적 기준군은 0.000). **관측 게이트 210틱에서는 어느 줄도 자기 대조군을 넘지 못한다.** (7) 편이 8 → 24가 되며 구간이 좁아져 **읽기 층에서 0을 제외하는 줄이 하나 → 넷**이 됐다 — "고쳐 학습한 run만 읽는다"는 옛 칸의 **분해능**에 대한 말이었다. 지시 섞기 여유는 모든 층·모든 줄에서 여전히 0을 포함한다. (8) **T1 재개 허용 오차를 사전 등록했다**(두 run 기준선 → 규칙 → 등록 → 판정; 상대 L2의 0에 가까운 분모 가드 포함) — `--gate t1`이 이제 exit 3이 아니라 판정한다. 보고서 `.superpowers/sdd/task-p3-report.md`, 표 `artifacts/reports/p3-*.json`.

**과제 재설계는 2026-09-22 Task R1이 했다 (다음은 R2).** P1~P3이 고친 것은 **자**였고(모집단·주 지표·기계적 기준군·쌍 부트스트랩), R1이 고친 것은 **재료**다. 셋을 바꿨다. (1) **모델의 입력에서 풀어 놓은 목표를 뺐다** — `goal` 줄은 버전과 주기적 지시 **문장**뿐이고 물체 소개 줄에 `attr=`가 없다(`ts0.6`·서식 v0.4). 대신 지시 문장이 취약·금지 물체를 **전부** 부른다(`s0.3`의 금지 물체 슬롯). 레코드·전문가·규칙 판정기는 그대로 구조화 목표를 읽는다. **옛 체크포인트는 계약 digest가 달라 거절된다** — P1~P3의 값은 "옛 계약·옛 모집단"으로 범위를 붙여 보존한다. (2) **에피소드를 사건이 잦게 만들었다** — 지시 변경 1~5회, 외란이 에피소드 안에서 대상을 밀고, 완료 꼬리 1 s → 0.3 s, 틱 안에서 버려지던 사건의 복원, 정체 감시(`h0.9`)와 앞단의 자기 가림 이어 들기(`pw0.2`). **수정 라운드 1(2026-09-23)이 코퍼스를 다시 만들었다** — 첫 판(`r1-robot-v0.1`, 52,079틱)은 45초를 hold·관측 **극한 순환**에 쓴 27편이 틱의 23.3 %와 읽기 틱의 **60.6 %**를 차지했고, 감시가 `m_hold`=15에서 끊고 같은 후보를 다시 채택하는 바람에 "가장 긴 죽은 구간"이 언제나 정확히 15여서 정체 지표가 구조적으로 0을 냈다(리뷰 1 C1). h0.9가 쿨다운·총량 상한·기하 나이 상한·멈춘 팔 감시로 순환을 끊고, 못 끊으면 에피소드를 `stall_exhausted`로 **명시적으로** 끝낸다. 실측(`r1-robot-v0.2`, 400편 **42,509틱**): 세계 사건 틱 1.6 % → **15.75 %**, 목표 변경 0.16 % → **1.33 %**, 모델이 보는 `instruction_changed` 줄 **7/57 → 564/564**, 완료 꼬리 11.8 % → 3.5 %, `max_ms` 종료 27 → **0**편(대신 32편이 `stall_exhausted`). **못 미친 것 — 정직한 수로**: 꼬리를 뺀 읽기 틱 2.6 % → **4.11 %**(완료한 368편만 보면 **3.32 %**; 목표 8 %), 그 가중 손실 몫 9.3 % → **11.7 %**(목표 30 %), 새 정의의 정체한 편 **4**편(목표 0). 첫 판의 7.8 %·20.1 %는 병적 편이 끌어올린 값이었다 — 손잡이는 `instruction.changes`·프로파일 비중이고 `tick_weights`는 학습 쪽이라 건드리지 않았다(보고서 Fix round 1). (3) **사건을 재는 지표**를 더했다(반응 지연·안정성·`q_stop` 지연/오경보·지시 대조 쌍 전체)와 두 계기 수리: **지시 섞기가 "지시를 읽는가"를 재는 열**이 됐고(모델이 문장을 보는 세 자리를 전부 굴린다), 상태 섞기의 **기증자 고정**(P3가 찾은, 값을 편 순서에 매달던 결함)을 없앴다. 보고서 `.superpowers/sdd/task-r1-report.md`.

**Task R2(2026-09-23)는 재개 게이트에서 한 번 멈췄다가, 진단 뒤 사용자 승인으로 판정 기준을 옮기고 이어갔다.** R2의 일은 고친 재료 위에서 fp32 T1을
1 epoch 돌려 "주 층에서 지시 섞기 여유의 구간이 0을 제외하는가"에 한 줄로 답하는 것이었고, docs/06 Task 5의 규칙대로
**긴 run 전에 그 학습 범위의 재개 게이트**를 먼저 돌렸다. (1) P3가 남긴 이월 항목을 닫았다 — 규칙을 재기 **전에**
"그 범위에서 잰 **모든** 재시작 없는 쌍의 최악값 위에"로 고쳐 커밋하고, 셋째 쌍을 R2의 경로에서 쟀다(loss |Δ|
**0.076560** — 셋 가운데 가장 크다; P2 0.064136 · P3 0.023574, 퍼짐 3.2배). 등록값 **loss_abs 0.2 · loss_rel 0.08 ·
param_max_abs 0.01 · param_rel_l2 0.05**. (2) 게이트(6 step = 3 + 실제 재시작 + 3)는 **`fail`**이다 — 정수 기준
(sampler 위치·뽑힌 단위·optimizer step 6=6·빠진 tensor 0)과 parameter 기준(5.959e-4 / 3.518e-3, 등록값의 15분의 1)은
전부 통과하고 **loss 기준만** 떨어진다(0.314955 > 0.2). **그 판정 위에서는 Stage B를 띄우지 않았다.**
(3) 진단: 같은 6 step 일정에서 **재시작 없는** 세 run을 맞대면 쌍마다 최악 |Δloss|가 **0.0786 ·
0.3304 · 0.4090**이다 — **재개한 쌍의 0.3150은 재시작이 전혀 없는 최악 쌍보다 작다.** 재개는 깨지지 않았고, 오차를
**5 step** 쌍에서 재어 **6 step** 비교를 판정한 것이 문제다(`max_steps`가 일정을 정하고 한 step이 더 있으면 교란이
한 번 더 증폭된다). (4) **그 진단 위에서, 사용자 승인으로 `t1`의 판정 기준을 옮겼다** — 게이트 실패를 **본 뒤의**
규칙 변경이고 순서를 그대로 적는다(등록 → 실행 → 실패 → 진단 → 변경). `RESUME_VERDICT_CRITERIA["t1"] =
("param",)`: 판정은 정수·parameter 기준이 지고 loss는 퍼짐과 함께 **기록만** 된다. **등록된 loss 값(0.2 / 0.08)은
느슨해지지 않았다** — 규칙을 글자 그대로 적용해 0.9 / 0.5로 늘렸다면 발견을 가렸을 것이다. 판정은 GPU 없이 같은
check의 snapshot에서 다시 냈고(측정값 불변을 먼저 assert한다) `--gate t1`은 **exit 0 `pass`**다. **Stage B는 그
판정 위에서 돌았다.** 함께 한 것: rollout 라벨판 T0를 다시 돌려 R1 Deviation 1을 닫았고(A2),
소형 scorer를 벽 예산 90 → 480분으로 제대로 돌렸다(A3, R1 Deviation 4). 보고서 `.superpowers/sdd/task-r2-report.md`,
표 `artifacts/reports/r2-*.json`.

**R2의 첫 학습 값 (Stage B).** fp32 master T1 **1 epoch**(233 step, 15,555.9 s = 63.79 s/step, peak 56.05 GiB,
loss 2.672 → 0.200, 40 step 체크포인트 별도). **판정 칸 주 층(235틱·26편)의 지시 섞기 여유 = +0.111
[+0.065, +0.158] — 0을 제외한다. 이 재료에서 모델은 지시를 읽는다.** 확인: 편 하나 빼기 26번 전부 0 제외(최악
+0.0950), 둘째 칸 `dev`(42편 276틱) +0.101 [+0.059, +0.150], 상태 섞기 +0.068, commitment 섞기 +0.064.
여유는 `grasp` 97틱(규칙 0.959 / 기계 0.000)에 몰려 **+0.258 [+0.144, +0.373]**이다. **학습이 만들었다** —
40 step +0.009 [−0.007, +0.027](0 포함), T0 200 step −0.004 [−0.035, +0.029](0 포함), 세 줄이 같은 평가
집합 해시 위에 있다. 칸 전체 **0.982**로 기계적 기준군 0.931·규칙 판정기 0.812를 둘 다 넘었다(처음이다).
사건 지표: 목표 변경 반응 **중앙 0틱·즉시 60.0 %·검열 0 %**, 전환율 **0.016**·왕복 **1**(T0는 0.420 / 477),
`q_stop`이 **처음으로 발화했지만 정지 사건 10건 중 1건뿐**(9건 검열 = 90 %; 잡은 1건의 지연 중앙 2틱,
오경보 0.00 %), 안전 위반 2.25 % → 0.92 %(남은 28틱 중 26틱이 `stop_ignored` — 같은 검열을 안전 쪽에서
본 것이다). 지시를 섞으면 즉시 반응이 60.0 % →
24.4 %로 떨어진다. **4B T0(200 step)도 자기 지시 섞기를 넘지만**(+0.043 [+0.012, +0.077]) 2B T1의 2.6분의
1이고 원값·안정성 모두 뒤진다 — **backbone 결정은 그대로**이고 이제 품질 증거도 지연과 같은 방향이다.
**다만 4B의 T1은 한 번도 돌지 않았고 이 상자에서는 못 돈다**(optimizer step 바닥만 94.0 GiB 대 울타리 73 GiB,
P2가 산술로 닫았다) — 곧 이것은 **전체를 학습한 2B 대 readout만 학습한 4B**의 비교이지 같은 조건끼리가 아니다.
**한계 넷**: 지시 대조 쌍 표는 판정에 못 쓴다(`false_change` null + 단일 레코드는 로봇 학습 분포 밖 —
이 checkpoint는 거기서 boolean 0.272로 무너지고 민감도 0.000이다), `grasp` 0.629는 천장 0.959에 못 미친다,
`place` 7틱·`push` 5틱은 너무 적다, seed 하나다. 소형 scorer를 3 epoch 다 돌린 표에서 **`dev`의 `q_main`은
다시 패턴으로 풀린다**(`ood_dev`는 아니다) — 둘째 칸은 복제이지 주장이 아니다.

**R2가 남긴 게이트 이월 둘 — 2026-09-23 Task R3a Stage A에서 닫혔다**: (a) `RESUME_TOLERANCE_RULE`이
"기준선 쌍은 그것이 판정할 비교와 **같은 `max_steps`**로 돌린다"를 담고, 진단 퍼짐은 기록에 이름이 있는 **네** 짝
(0.0786 / 0.3304 / 0.4090 / **0.4460** — fix round 1의 넷째 run)으로 등록됐다. 같은 네 run의 나머지 두 짝
(0.5246 · 0.3124)은 넣지 않았다: 퍼짐을 넓히는 것은 진단을 너그럽게 만드는 방향이다. **등록된 loss 오차 0.2 / 0.08도
판정 기준 `("param",)`도 그대로다** — 그 값이 3·5·5 step 쌍에서 나왔다는 사실만 소스·산출물에 적힌다.
(b) `collect_rng_state`가 이제 **CUDA generator를 담는다**(`torch.cuda.is_initialized()`일 때만 — 그 호출이 CUDA를
초기화하므로 CPU 학습 프로세스에 쓸데없는 context를 만들지 않게; 그 키가 없는 옛 checkpoint는 그대로 읽히고, 있는데
못 받는 상자는 이름으로 거절한다). 지금 설정에서 CUDA generator를 뽑는 연산이 없으므로 **R2의 수는 바뀌지 않는다**.

**Task R3a (2026-09-24) — seed를 넘어 서는가, 둘째 epoch은 `grasp`를 올리는가.** 재료·하네스·컨트롤러·평가 칸은
R2 그대로이고(평가 집합 해시 `6a3b69131243`·`9b484441b23b`가 R2의 것과 같다) **seed와 step 수만** 바꿨다.
**seed 18** (233 step = 1 epoch, 14,944.5 s, 61.17 s/step, peak 56.05 GiB, loss 2.633 → 0.130): 판정 칸 주 층
235틱에서 모델 **0.889** 대 지시 섞기 0.643, 여유 **+0.247 [+0.179, +0.308] — 0을 제외한다**(seed 17은 +0.111
[+0.065, +0.158]). `grasp` 97틱은 **0.876**으로 규칙 판정기 천장 **0.959**에 0.083까지 붙었고(seed 17은 0.629)
여유가 **+0.536 [+0.362, +0.708]**, 편 하나 빼기 26번 전부 0 제외(최악 `ep-E2-420167` +0.2329), 둘째 칸 `dev`
**+0.261 [+0.203, +0.321]**. 사건 지표는 더 나아졌다 — 목표 변경 즉시 반응 **86.7 %**, 전환율 **0.007**·왕복 1,
안전 위반 **0.36 %**(seed 17 0.92 %)이고, `q_stop`은 정지 사건 **10건 중 8건**을 잡는다(2건 검열 = 20 %, 잡은
여덟의 지연 중앙 1틱, 오경보 0.19 % — 학습된 줄 가운데 처음으로 0이 아닌 오경보다). **(정정, 2026-09-24 Stage D:
이 자리에 처음 적힌 "10건 중 2건만 잡는다(8건 검열)"는 검열 열을 거꾸로 읽은 것이다. 저장된 `stop_timing`은
`reacted 8, censored 2`이고 생성된 표는 검열 수를 찍으므로 `2 of 10 (20.0 %)`이 맞다. 아래 seed 19가 이 수가 왜
중요한지 보인다 — 세 seed가 1 / 8 / 0을 잡는다.)** **seed마다 자기 구간을 읽는다 — 세 seed의 값으로 구간을
만들지 않는다.**

**둘째 epoch (466 step, seed 17, 29,205.5 s = 8.11 h).** 이것은 R2의 step 233을 **이어간 run이 아니다** —
`adapt_readout.py`가 `resume`을 Trainer에 넘기지 않고 있어서(그 결함은 고쳤고 시험이 붙었다) **처음부터 466 step,
곧 한 일정으로 2 epoch**을 돌았다. 증거는 산출물이다: `summary.rescheduled`가 `null`, 곡선 첫 줄이 step 1 ·
loss 2.6720505952835083(R2의 step 1과 자릿수까지 같다), `sampler.epochs`가 `{robot 2, non_robot 2}`. R2의
checkpoint는 읽지도 쓰지도 않아 무사하다. 233 step 대신 466을 돌았으니 **≈4.3 GPU-h가 더 갔다**. 비교는
"이어 붙였다"가 아니라 **"각자 자기 일정을 끝까지 돈 1 epoch 대 2 epoch"**로 읽는다(그쪽이 더 깨끗하다).
값: 주 층 **0.860**, 지시 섞기 여유 **+0.191 [+0.137, +0.247] — 0을 제외한다**, `grasp` **0.845**(천장 0.959까지
**0.114** — 1 epoch은 0.629로 0.330이었다), 여유 +0.412 [+0.279, +0.562], 편 하나 빼기 26번 전부 0 제외,
둘째 칸 +0.170 [+0.122, +0.226]. `q_stop`이 정지 사건 **10건 중 6건**을 잡는다(검열 90 % → **40 %**, 오경보
0.00 %), 안전 위반 0.92 % → **0.26 %**, 목표 변경 즉시 반응 60.0 % → 80.0 %(다만 2.2 %가 검열된다 — 1 epoch은
0 %였다).

**seed 19 (233 step = 1 epoch, 14,860.8 s = 4.13 h, 60.75 s/step, peak 56.05 GiB, loss 3.092 → 0.200).** 주 층
235틱에서 모델 **0.804** 대 지시 섞기 0.677, 여유 **+0.128 [+0.084, +0.175] — 0을 제외한다**(편 균등 +0.105
[+0.066, +0.145]). 상태 섞기 +0.166 [+0.127, +0.203], `grasp` 97틱 **0.722**(여유 +0.309 [+0.202, +0.439]),
편 하나 빼기 26번 전부 0 제외(최악 `ep-E2-420109` +0.1131), 둘째 칸 `dev` **+0.156 [+0.116, +0.198]**. 7틱뿐인
`place`에서만 지시 섞기 열이 모델을 앞선다(−0.286 [−0.571, 0.000] — 0을 포함하고 편마다 한 틱이라 부호를 실을
수 없다). 사건: 목표 변경 즉시 반응 62.2 %(지시 섞으면 35.6 %), 전환율 0.011·왕복 1.

**판정 1 — 여유는 seed를 넘어 선다 (D1).** 판정 칸 주 층의 지시 섞기 여유는 **세 seed 모두 0을 제외한다** —
seed 17 **+0.111 [+0.065, +0.158]**, seed 18 **+0.247 [+0.179, +0.308]**, seed 19 **+0.128 [+0.084, +0.175]**.
곧 "이 모델은 지시를 읽는다"는 seed 의존적이지 않다. **크기는 그렇지 않다**: 세 값의 **범위는 +0.111~+0.247**
(폭 0.136으로 가장 작은 값보다 크다)이고 주 층 원값도 0.762~0.889다. **세 값으로 구간을 만들지 않았다** — 학습
seed 세 번 뽑기는 조리법의 여유를 추정할 표본이 아니고, 세 점으로 만든 구간은 실제 불확실성보다 좁아진다.
읽는 법은 하나다: **부호와 0 제외는 재현되고, 크기는 약 2배 안쪽까지만 안다.** 편 하나 빼기는 26회 × 대조군 3 ×
seed 3 = **234회 전부 0을 제외**하고, 둘째 칸도 세 seed 모두 부호를 재현한다(+0.101 / +0.261 / +0.156; C4 할인
그대로). **재현되지 않는 것이 하나 있다 — `q_stop`이다**: 세 seed가 10건 중 **1 / 8 / 0**건을 잡는다(검열 90 /
20 / 100 %, 안전 위반 0.92 / 0.36 / 0.99 %). 같은 조리법에서 하나는 풀린 것처럼, 하나는 학습되지 않은 것처럼
보인다 — **이 코퍼스에서 정지 탐지는 seed 잡음이고**, run 하나에서 나온 `q_stop` 문장은 그 run의 seed에 대한
문장이다.

**판정 2 — 둘째 epoch은 `grasp`를 천장 쪽으로 옮긴다 (D2).** seed 17에서 1 epoch 대 2 epoch(둘 다 처음부터,
각자 자기 일정 끝까지): `grasp` **0.629 → 0.845**로 규칙 판정기 천장 0.959까지의 거리가 0.330 → **0.114**
(간격의 **65 %**), 주 층 여유 **+0.111 → +0.191**(둘 다 0 제외), 주 층 원값 0.762 → 0.860, 칸 전체 0.982 →
0.989, 안전 위반 0.92 % → 0.26 %. 비용은 4.32 h → **8.11 h**다. **두 가지 단서**: (a) epoch 효과와 seed 효과가
같은 크기다 — seed 18은 **1 epoch**에서 `grasp` **0.876** / 여유 **+0.247**로 2 epoch seed 17(0.845 / +0.191)을
**넘는다**. 2 epoch은 seed 하나에서만 돌았으므로 "2 epoch이 1 epoch보다 낫다"는 seed 17에 대해 참이고 조리법에
대해서는 아직 아니다. (b) `q_stop`의 1건 → 6건은 읽기의 증거가 아니다 — 466에서 상태·지시·commitment 섞기 열이
**전부 10건 중 7건**을 잡아 모델보다 많고, seed 사이 1 / 8 / 0과 나란히 놓으면 그 지표의 잡음 안이다.

**Task R3b(2026-09-23)가 클라우드 실행기를 만들었다 — 그리고 한 푼도 쓰지 않았다.** R3a가 seed·epoch를 반복하는 동안
CPU에서 `scripts/launch_run.py {prepare,launch,status,fetch,cancel,resume}` · `src/robo_jev/launch/` ·
`infra/run-manifest.schema.json`을 만들고 **localhost 왕복**으로 인수했다(SSH 백엔드, `configs/train/tiny_cpu.yaml`
5 step **145.688 s**(러너 자신의 시계로 잰 경과 — 학습기의 `metrics.json`은 144.85 s다), 다섯 산출물 sha256 전부 일치). 고의 실패 넷도 실제로 냈다: 벽시계 상한 · USD 상한 · `kill -9` ·
확인되지 않은 `cancel`(→ `unknown`). 읽는 법 셋. (1) **상한을 재는 것은 실행기가 아니라 원격 러너다** — 벽시계·GPU
시간·USD 셋이 하나의 마감으로 환산되고(`max_usd / hourly_usd` 등) 가장 이른 것이 구속하며, 넘으면 원격이 스스로
checkpoint를 쓰고 `failed(reason=budget)`으로 끝난다. 노트북을 닫아도 지켜진다. 그리고 **아무 상한도 구속하지 않는 run은 `prepare`가 거절한다**(`--no-cap`으로 이름을 부르면 그 사실이 명세에 남는다); **재개의 남은 예산도 부모가 실제로 쓴 시간**(가져온 `state.json`)에서 빼며, 그것이 관측된 적 없으면 `status`부터 하라고 거절한다. (2) **`cancel`은 확인될 때까지
기다린다** — 확인하지 못하면 `cancelled`가 아니라 **`unknown`**이고 종료 코드 3으로 사람을 부른다(돈이 계속 나갈 수
있다). (3) **데이터는 묶음에 담기지 않는다** — 경로와 sha256만 가고 원격이 시작 전에 대조하며, 다르면
`failed(reason=inputs)`로 학습을 시작조차 하지 않는다.

**다음은 사용자 결정이다 — 공급자·계정·첫 유료 run.** 실행기는 "이미 있는 상자에 SSH"까지이고 인스턴스 생성·종료
(`Provider.create/destroy/describe`)는 **인터페이스만** 두고 이월했다. 그러므로 다음 세 가지는 코드가 아니라 사람이
정한다: (a) **공급자와 계정** — docs/05 §5의 공개 표(Lambda 1×H100 PCIe $3.29 · Runpod 표시값 $2.89 · H200 $4.59)는
2026-09-18 확인값이고, 예약 직전 콘솔의 상품·수량·지역·시간 단가를 명세의 `budget.price_source`에 복사한다.
(b) **첫 유료 run의 승인** — 준비된 명세는 `artifacts/scratch/r3b/r2-t1/run.json`(R2의 fp32 T1 1 epoch를 1×H100
$2.5/h 가정으로: 예상 4.32 h · **$10.80**, 상한 6 h / 6 GPU-h / $12.5 → 마감 5.0 h, 구속 `max_usd`)이고 `host`·
`identity_path`는 자리표시다. **H100 대 GB10의 처리량 배수는 아직 재지 않았다** — 4.32 h는 1.0× 가정의 상한이고
첫 run의 tokens/s로 대체한다. (c) **카드 크기** — 2B의 fp32 T1은 실측 peak 56.05 GiB로 80 GB 카드에 여유 22.6 %로
들지만, **4B의 fp32 T1은 optimizer 바닥만 94.0 GiB라 80 GB에 들지 않는다**(H200 141 GB / B200 180 GB 한 장, 또는
FSDP sharding부터). 자격 증명은 저장소에도 명세에도 넣지 않는다 — 개인키는 **경로**로만 가리킨다.

## 4. git 밖에 있는 것 — 전달 목록

| 항목 | 위치 | 전달/재생성 | 검증 |
| --- | --- | --- | --- |
| tokenizer | `artifacts/tokenizers/Qwen/Qwen3.8-27B/tokenizer.json` + 뿌리 `artifacts/tokenizers/manifest.json` | id `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, sha256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`. 재다운로드 또는 파일 전송 | `load_tokenizer`가 manifest 해시와 대조 |
| 개발 장부·브리프·보고서·리뷰 패키지 | `.superpowers/sdd/` (≈9MB; `progress.md`가 태스크별 이월 항목의 정본) | rsync | `progress.md` 끝 항목이 "BRANCH FINISHED" 이후인지 |
| Claude 메모리 | `~/.claude/projects/-Users-user-Documents-robo-jev/memory/` | Spark의 프로젝트 경로 키(`-<경로를 -로 치환>`)로 복사 | `MEMORY.md` 한 줄 |
| backbone 가중치 | `artifacts/models/<id>/` + 뿌리 `artifacts/models/manifest.json`(2B/4B/9B/27B, 55.6 GB; revision·파일별 sha256·digest·라이선스·파라미터 수) | **전달하지 않는다** — manifest만 복사 후 `uv run python scripts/fetch_backbone.py --from-manifest artifacts/models/manifest.json --id <id>`로 같은 revision·해시 재수신 | `describe_backbone`이 적재 전 크기·소형 파일 해시(`--verify-full`이면 전부) 대조 |
| Spark 지연 선별 결과 | `artifacts/reports/backbone-screen.json`(563 KB, 틱별 기록 포함) + `backbone-screen-27b.json` | rsync(작음); 재생성은 ≈81분 GPU | JSON `environment.kernels`가 fla·causal_conv1d 활성, `verdict`가 03의 표와 일치 |
| 로봇 데이터 batch-0·rollout 100 | `artifacts/datasets/d1-robot/batch-0` (+ `rollouts/`, `contrast/records.jsonl`, `rollouts-1000/`·`rollouts-full/` sweep), QA `artifacts/reports/d1-robot-batch-0-qa.json`; 비로봇 pilot `artifacts/datasets/d1/single`(2,000건, seed 17), QA `artifacts/reports/d1-pilot-single-qa.json` | **전달하지 않는다** — `uv run python scripts/generate_episodes.py --config configs/data/d1_robot.yaml --count 40 --out artifacts/datasets/d1-robot/batch-0`(≈1분) 뒤 `uv run python scripts/rollout_keyframes.py --dataset artifacts/datasets/d1-robot/batch-0 --limit 100 --workers 8`로 재생성. 계약 v0.3 이전의 batch·rollout·DAgger probe는 지문 변경으로 거부된다 | manifest의 `versions` 집합이 하나(h0.7·e0.4·c0.6·rj0.5), QA 위반 0·누출 0 |
| **D1 (2026-09-20, Task D1)** | 로봇 `artifacts/datasets/d1-robot/d1`(400편·36,690틱·대조 990쌍, 613 MB — rollouts/ 308 MB 포함) → 후속 버전 `d1-rollout-labels`(rollout 라벨 2,000, 314 MB); 비로봇 `artifacts/datasets/d1/single`; 버전 manifest `artifacts/datasets/d1/manifest.json`; QA `artifacts/reports/d1-robot-d1{,-rollout-labels}-qa.json`·`d1-pilot-single-qa.json`; sweep `artifacts/reports/d1-robot-sweep-d1.json`; 검수 표본 `artifacts/reports/d1-review-sample/`; Task 2c 표 `artifacts/reports/tiny-scorer.json`; 공개 세트 `artifacts/datasets/public/` | 재생성: `generate_episodes.py --config configs/data/d1_robot.yaml --count 400 --out artifacts/datasets/d1-robot/d1`(10분) → `rollout_keyframes.py --dataset … --limit 0 --workers 8 --chunk-episodes 8`(2시간, `--resume` 가능, systemd 단위로) → `python -m robo_jev.data.lineage --dataset … --out …/d1-rollout-labels` → `python -m robo_jev.data.validate`; 비로봇 `python -m robo_jev.data.generate --config configs/data/pilot.yaml --count 2000 --seed 17 --output artifacts/datasets/d1/single`; 표본 `scripts/export_review_sample.py`; 공개 세트 `uv run --with pyarrow python scripts/convert_public_sets.py` | QA 위반 0·누출 0(세 보고서), rollout 100,640·censored 904(`candidate_unavailable`뿐), manifest `versions` 한 집합(h0.7·labels-rollout-v1) |
| B2 토큰 측정 | `artifacts/reports/tokens-b2.json` (v0.3), `tokens-b2-v0.json` (옛 서식) | 재생성: `uv run python scripts/measure_tokens.py --episodes artifacts/datasets/d1-robot/batch-0` | 08 §3.4의 수치와 일치 |
| Spark v0.3 재실행 | `artifacts/reports/backbone-screen-v03.json` (2B·4B, lower/upper/instruction_change) | 재생성 ≈20~30분 GPU (§3-5의 명령) | 03 §"지연 예산" 표와 일치 |
| **P1 파일럿(2026-09-21)** | `artifacts/reports/p1-{2b,4b}-{t0,lora,t1,zero-shot}.json`(run마다 학습 곡선·s/step·peak 메모리·RSS·토큰·고정 평가 집합의 해시와 분할별 표), `p1-acceptance.json`(C1·C2·C3), checkpoint `artifacts/runs/p1-*/`(T0는 readout만 3.4 MiB, LoRA·T1은 크다), 로그 `artifacts/scratch/p1/` | **전달하지 않는다** — 재생성은 위 실행 줄(2B T0 200 step ≈ 47분, 4B T0 ≈ 1.8시간, LoRA 40 step ≈ 1~2시간) | run마다 `evaluation.eval_set.sha256`가 같아야 두 수를 나란히 놓을 수 있다; `manifest.contract_sha256`가 지금 체크아웃과 같아야 checkpoint가 실린다 |
| 리뷰 10·11의 재현 스크립트 | `artifacts/reviews/` | 필요 시 rsync(참고용) | — |

## 5. 작업 규칙

superpowers SDD 루프(브리프 파일 → 구현 에이전트 → 리뷰 패키지 → 리뷰 → 수정 → 재리뷰 → `progress.md` 한 줄). 커밋은 사용자 승인 하에, trailer `Co-Authored-By: <그 작업을 한 Claude 모델> <noreply@anthropic.com>`(2026-09-20까지는 Claude Fable 5.1, 2026-09-21부터 Claude Opus 5 (1M context)). `.superpowers/`·`.venv/`·`artifacts/`·`data/`·`.claude/worktrees/`는 커밋 금지. docs/08이 로봇 스트림 계약의 정본, README의 구현 상태 표가 태스크 상태의 정본. 외부 리뷰는 코드로 검증한 뒤 동의를 받아 반영(07·09·10·11 반영 완료).
