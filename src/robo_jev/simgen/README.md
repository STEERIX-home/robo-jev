# simgen — 시뮬레이션 판단 데이터 엔진 (레코드 형식 `simgen-request/0`)

정본 기획은 `docs/12-sim-judgment-data-engine.md`, 이 단계의 지시서는 Task v0a다. 이 문서는 **레코드 형식**(필드 뜻), **정답 정의**,
**버전 규칙**을 적는다.

한 줄 = 한 판단 순간 = 한 요청. 판단이 필요한 순간 하나를 만들고(①), 결정 여유에 맞춘 참 세계를 풀고(②), 로봇 감각만 담은 관측을
만들어 글로 쓰고(③), 관측에서 뽑은 사후 세계 K개에서 선택지마다 갈라 굴려(④) 확률 정답을 붙인다(⑤).

```
{"schema", "id", "generator", "meta", "request", "labels", "truth", "rationale"}
```

**모델 입력은 `request`뿐이다.** `meta`·`labels`·`truth`·`rationale`은 모델에 들어가지 않는다.

## 1. 필드

### `schema`, `id`

* `schema` = `"simgen-request/0"`.
* `id` = `"<생성기 버전>-<6자리 번호>"` (예: `v0a-000123`). 번호는 묶음 안의 순서다.

### `generator` — 다시 만들기 위한 열쇠

| 필드 | 뜻 |
|---|---|
| `version` | 생성기 버전 (`v0a`) |
| `catalog` / `values` / `skins` | 속성 목록·가치표·상황 옷 카드의 버전 (`configs/simgen/*.yaml`의 `version`) |
| `seed` | 레코드 seed (32비트). 모든 난수는 이 seed의 이름 붙은 갈래(`spec.stream`)에서 나온다 |
| `git` | 생성한 커밋 (짧은 sha) |
| `variant` | `null` / `"numeric"` / `"wording"` — 사실 하나만 바꾼 쌍의 변형 쪽이면 그 종류 |
| `forced` | 층화로 정해 준 값(언제나 `domain`; 예시용이면 `scene`·`body`·`style` …) |
| `attempts` | 여유를 맞추는 참 세계를 찾기까지의 시도 수 |

`generate.generate_record(seed, index, forced["domain"], forced=…, variant=…, pair=meta["pair"], git=…)`가 같은 레코드를 바이트 단위로
다시 만든다(시험 `test_simgen_determinism.py`).

### `meta` — 장부 (모델 입력 아님)

| 필드 | 뜻 |
|---|---|
| `scene` | `p1_step_in` · `p1_crossing` · `p1_head_on` · `p5_braking` · `p5_turning` · `p1p5_step_in_carrying` |
| `phenomena` | `["P1"]` / `["P5"]` / `["P1", "P5"]` |
| `domain` | 분야 10개: `warehouse factory cafe hospital lab sidewalk home retail greenhouse transit` |
| `body` | `mobile_base` · `humanoid` · `forklift_amr` · `arm` |
| `sensor_suite` | `lidar_camera` · `stereo_camera` · `arm_cell` |
| `variant` | `moving`, 또는 `p1_crossing`에서 멈춘 로봇이 건널지 정하는 `stopped` |
| `agent` | 상대 주체 `{kind, group, cue, velocity_mode (direct / two_fix), age_s}` 또는 `null` |
| `container` | 용기 `{kind, content (hot / chemical / other), fill_mode (phrase / percent), lid_view (open / lid / unknown)}` 또는 `null` |
| `style` | 서술 문체 `facts` · `prose` · `planner` |
| `language` | `en` |
| `units` | 글의 단위 표기 `{distance, speed, time}` |
| `hint` | 계산된 힌트 문장이 글에 있나 |
| `recommendation` | 계획기 권고 `{present, option (기본 행동 질문의 열쇠), canonical, wrong, question}` — `wrong`이면 기대 비용이 최선보다 2τ 이상 높은 선택지 |
| `margins` | 참 세계의 결정 여유(정의는 §3) |
| `pair` | `null` 또는 `{of: 짝의 id, kind: numeric / wording, role: base / variant[, flipped: 옮긴 여유 이름]}` |
| `key_question` | 핵심 noul의 이름(장면의 결정 여유를 묻는 질문) |
| `key_style` | 선택지 열쇠 문체 `word` · `code` · `verb` |
| `criteria` | 행동 질문마다 지시문에 적힌 기준(`avoid_contact` · `avoid_spill` · `time_critical` 또는 `null`) |
| `distractors` | 관계없는 사실의 수(0~3) |
| `basis` | 질문마다 정답을 계산한 법(사건·선택지·상태 이름) |
| `options` | 행동·상태 질문의 열쇠 → 정식 선택지(또는 상태 값) |
| `posterior_k` | 사후 표본 수 K (기본 256) |
| `margin_redraws` | 여유를 다시 뽑은 횟수(보통 0) |

### `request` — 모델 입력

```json
{"state": "<로봇 감각만 담은 글>",
 "questions": {
   "<name>": {"type": "choice", "instructions": "...", "criteria": {"<key>": "<설명>"}},
   "<name>": {"type": "noul", "instructions": "..."},
   "<name>": {"type": "score", "instructions": "...", "criteria": ["0 = ...", "1 = ..."]}}}
```

Decisions 형식 제약: `state` ≤ 16,000자; 질문 1~12개(생성기는 3~8개); 이름 `^[A-Za-z0-9_]{1,40}$`; `choice`는 선택지 2~8개(열쇠도
같은 정규식); `score`의 `criteria`는 문자열 목록. 검사기는 `record.validate_record`.

글의 거리는 **면 사이 간격**이다: 로봇 앞면 → 상대(장애물)의 가까운 면, 로봇 진로 가장자리 → 상대의 가까운 면. 교차선 거리는 로봇 앞면에서
상대가 지나는 줄까지. 통로 폭은 벽에서 벽까지.

### `labels` — 정답

| 질문 | 정답 |
|---|---|
| `noul` | `{"type": "noul", "p_yes"}` — K개 사후 세계에서 그 사건이 일어난 비율 |
| `score` | `{"type": "score", "value"}` — [0, 1] 양의 K개 평균(사건 확률, 가까움, 빈 높이 사용, 3초로 자른 지연) |
| 행동 `choice` | `{"type": "choice", "kind": "action", "probabilities", "expected_cost"}` — 아래 |
| 상태 `choice` | `{"type": "choice", "kind": "state", "probabilities"}` — 이산 상태(의도, 채움 구간, 뚜껑, 공의 멈춤)의 사후 비율 |
| 규칙 적용 `choice` | `{"type": "choice", "kind": "rule", "probabilities"}` — 질문에 적힌 규칙을 글에 보인 수에 적용한 결과(확률 1) |

**행동 choice (docs/12 §5 정정본).** 선택지 a마다 기대 비용

  E[c_a] = (1/K) Σ_k [ Σ_사건 비용(사건) · 1{사건이 세계 k에서 a로 일어남} + 진행 손실_k(a) · `delay_per_s` ],

확률 p_a = softmax(−(E[c_a] − min_b E[c_b]) / τ). **argmax는 언제나 기대 비용 최소**다(2,000개 전부에서 시험). '각 선택지가 최선이었던
빈도'는 쓰지 않는다 — 대부분의 세계에서 시간을 아끼지만 가끔 큰 사고를 내는 선택지가 높게 나오기 때문이다. 비용표는
`configs/simgen/values-v0.yaml`(사람 100, 아이·노인 150, 차량 120, 물체 15, 쏟아짐 뜨거움 25·약품 40·그 밖 8, 지연 1/초, τ = 2.0).
지시문에 기준이 적힌 질문(약 20 %)은 그 기준의 덮어쓰기(`criteria`)를 쓴다.

반올림: 기대 비용은 소수 4자리, 확률은 그 기대 비용으로 계산해 소수 8자리(합 오차 ≤ 4e-8, 계약은 ±1e-6). 기대 비용이 같으면 확률도 같다.

### `truth` — 참 세계와 실제로 일어난 것 (평가용)

* `world` — 숨은 값과 참값(`values`: 의도, 움직임 매개변수, 참 채움, 출렁임 계수 k, 바닥 기울기, 뚜껑, 가려진 쪽의 장애물 …),
  관측 시점의 사건(`at_observation`: 그때의 의도, 단서, 관측 나이, 옆 공간·뚜껑의 보임 상태), 참 여유(`margins`), 알려진 기하(`geometry`).
* `realized` — 질문마다 참 세계(K = 1)에서 일어난 것: noul → bool, score → 값, 행동 → `{costs, best}`, 상태 → 참 상태의 열쇠, 규칙 → 답.

학습은 사후 확률(`labels`)로 한다. 참 결과는 평가용이다(docs/12 §5).

### `rationale`

정답의 계산 과정 한 문단(보고된 수치 → 여유, 사후 의도·채움, 기대 비용 분해와 소프트맥스, 참 세계에서 일어난 것). 사람이 읽는 용도이고
'계산을 안으로 접기' 실험 재료다.

## 2. 누출 경계 (코드 구조 + 시험)

* `world.py`의 `World`(참값·숨은 값)와 `observation.py`의 `Observation`(감각 보고)은 다른 타입이다.
* 서술기(`narrate`)·질문 생성기(`questions`, `phrases`)·사후 표본(`posterior`)·굴리기(`rollout`)·정답기(`labels`)는 `World`를 import하지
  않는다. 세계와 관측을 함께 보는 곳은 `scenes`(참 세계 풀기 → 관측)와 `truth`·`rationale`뿐이다.
* 시험(`test_simgen_leak.py`): (a) import 검사, (b) 같은 관측 + 숨은 값만 다른 세계 → `request`와 `labels`가 바이트 단위로 같다,
  (c) 숨은 전용 이름(intent, slosh, true_ …)이 모델 입력에 없다.

## 3. 결정 여유와 표본

장면마다 목표 여유 m ~ 0.6·N(0, σ_m) + 0.4·U(−M, M)(혼합분포)을 뽑고 표면 변수를 그 여유에 정확히 맞춘다(σ_m·M은 catalog-v0 `margins`).

| 여유 | 정의 | 단위 |
|---|---|---|
| `step_in` | gap(중심) − (반응 거리 + moderate 멈춤 거리) − 반지름 합 = 면 간격 − 멈춤 거리 | m |
| `crossing` | 상대가 진로 가장자리에 닿는 시각 − (로봇이 교차선을 비우는 시각 + 0.3 s) | s |
| `head_on` | 통로 폭 − 두 몸 폭 − 필요한 폭(`passing_clearance`) | m |
| `spill` | (H − f·H − 1.3·(D/2)·a_ref/g) / H — 기준 가속(moderate 감속, 또는 지금 속도의 회전 옆 가속)에서 남는 림 높이 비율 | 림 높이 비율 |
| `stop` | 장애물 면 간격 − moderate 멈춤 거리 (p5_braking) | m |

레코드의 약 10 %는 사실 하나만 바꾼 쌍이다: `numeric`(결정 변수 하나만 경계 너머로 — 같은 잡음, 같은 질문 갈래), `wording`(숫자는 그대로,
상황 옷 단어만 바꿈 — 정답이 같다).

## 4. 버전 규칙

* **`schema`** (`simgen-request/N`): 필드를 빼거나 뜻을 바꾸면 N을 올린다. 필드를 더하기만 하면 그대로 둔다.
* **생성기 `version`** (`v0a` …): 같은 seed에서 다른 레코드가 나오게 바꾸면(물리·표본·서술·질문 코드) 올린다.
* **`catalog` / `skins` / `values`**: YAML의 값을 바꾸면 그 파일의 `version`을 올린다. 기본 가치표는 가치 판단이라 공개하고 버전을 붙인다.
* 레코드는 (생성기 버전, 세 설정 버전, seed, 강제한 분야, 쌍 변형)으로 다시 만든다. 물리가 좋아지면 옛 데이터도 새로 뽑는다(docs/12 §12).

## 5. 모듈과 명령

| 모듈 | 하는 일 |
|---|---|
| `catalog` | 설정 셋 적재·구조 검사 |
| `spec` | 중립 타입(Reading, RobotSpec, Batch), 이름 붙은 난수 갈래, 채움 구간 |
| `physics` | 멈춤 거리, 가감속, 가로지름 시각, 정적 액면 쏟아짐, 회전 지연 |
| `world` / `observation` | 참 세계 / 관측 |
| `scenes` | 여유에서 참 세계 풀기, 감각 모형, 사실 하나 바꾸기 |
| `posterior` | 관측 → 사후 세계 K개 |
| `rollout` | 선택지 × 세계 굴리기 → 사건·양 |
| `labels` | 정답(noul·score·행동·상태·규칙)과 참 결과 |
| `textfmt` · `phrases` · `narrate` · `questions` | 단위 표기, 문구 묶음, 서술기 3종, 질문 생성기 |
| `record` · `rationale` · `generate` | 형식 검사·직렬화, 계산 과정 문단, 생성 흐름·층화·쌍 |
| `stats` · `examples` | 묶음 통계, 검수용 예시 |

```bash
uv run python scripts/simgen_generate.py --n 2000 --seed 0 --out artifacts/simgen/v0a/sample-2000.jsonl --stats artifacts/simgen/v0a/stats.json
uv run python scripts/simgen_examples.py --out-dir artifacts/simgen/v0a
```
