# simgen — 시뮬레이션 판단 데이터 엔진 (레코드 형식 `simgen-request/0`)

정본 기획은 `docs/12-sim-judgment-data-engine.md`, 지시서는 Task v0a(엔진 1단계)와 Task v0b(새 현상 P3·P4·P6, 조합, 한국어, 봉인된 시험 세트
T1~T7, 학습 세트 — §5)다. 이 문서는 **레코드 형식**(필드 뜻), **정답 정의**, **버전 규칙**을 적는다.

한 줄 = 한 판단 순간 = 한 요청. 판단이 필요한 순간 하나를 만들고(①), 결정 여유에 맞춘 '닻' 세계를 풀고(②), 로봇 감각만 담은 관측을
만들어 글로 쓰고(③), 관측에서 뽑은 사후 세계 K개에서 선택지마다 갈라 굴려(④) 확률 정답을 붙인다(⑤). 평가용 참 세계는 그 사후에서 한 번
더 뽑는다(리뷰 1 I4 — 닻 세계는 관측을 만드는 데만 쓴다).

```
{"schema", "id", "generator", "meta", "request", "labels", "truth", "rationale"}
```

**모델 입력은 `request`뿐이다.** `meta`·`labels`·`truth`·`rationale`은 모델에 들어가지 않는다.

## 1. 필드

### `schema`, `id`

* `schema` = `"simgen-request/0"`.
* `id` = `"v0a-<6자리 번호>"` (예: `v0a-000123`; 검수 예시는 `v0a-ex-…`). 번호는 묶음 안의 순서다. 앞머리는 과제 이름이라 생성기
  버전이 올라도 그대로다 — 버전은 `generator.version`.

### `generator` — 다시 만들기 위한 열쇠

| 필드 | 뜻 |
|---|---|
| `version` | 생성기 버전 (`v0a.1` = 리뷰 1 수정; 첫 묶음은 `v0a`) |
| `catalog` / `values` / `skins` | 속성 목록·가치표·상황 옷 카드의 버전 (`configs/simgen/*.yaml`의 `version`) |
| `seed` | 레코드 seed (32비트). 모든 난수는 이 seed의 이름 붙은 갈래(`spec.stream`)에서 나온다 |
| `git` | 생성한 커밋 (짧은 sha) |
| `variant` | `null` / `"numeric"` / `"wording"` — 사실 하나만 바꾼 쌍의 변형 쪽이면 그 종류 |
| `forced` | 층화로 정해 준 값(언제나 `domain`; 예시용이면 `scene`·`body`·`style` …) |
| `attempts` | 여유를 맞추는 닻 세계를 찾기까지의 시도 수 |

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
| `style` | 서술 문체 `facts` · `prose` · `planner` (+ 빼 둔 `log` — T2 전용) |
| `language` | `en` · `ko`(v0b) |
| `units` | 글의 단위 표기 `{distance, speed, time}` |
| `hint` | 계산된 힌트 문장이 글에 있나 |
| `recommendation` | 계획기 권고 `{present, canonical, wrong, gap, costs, option, question}` — 그 순간 **가능한 모든 행동**(`costs` = 정식 선택지 → 기본 비용표의 기대 비용, 기준 없음)으로 판정한다: 맞는 권고는 최선과의 차 `gap` ≤ τ, `wrong`이면 `gap` ≥ 2τ. `option`은 기본 행동 질문(`question`)에 그 행동이 있을 때의 열쇠(없으면 `null`) |
| `margins` | 닻 세계(관측을 만든 세계)의 결정 여유 = 설계 목표(정의는 §3). 경계 집중은 이 값(곧 관측)에 걸린다 |
| `margin_norm` | `margins`와 같은 이름: 여유 / σ_m(그 여유·몸의 설계 σ, catalog `margins.*.sigma`) — 어려움(경계에서의 거리)으로 묶을 때 쓴다(Task v0c 요청). \|값\| < 1 이 경계 가까이 |
| `pair` | `null` 또는 `{of: 짝의 id, kind: numeric / wording, role: base / variant[, flipped: 옮긴 여유 이름]}` |
| `key_question` | 핵심 noul의 이름(장면의 결정 여유를 묻는 질문) |
| `key_style` | 선택지 열쇠 문체 `word` · `code` · `verb` |
| `criteria` | 행동 질문마다 지시문에 적힌 기준(`avoid_contact` · `avoid_spill` · `time_critical` 또는 `null`) |
| `distractors` | 관계없는 사실의 수(0~3) |
| `basis` | 질문마다 정답을 계산한 법(사건·선택지·상태 이름) |
| `options` | 행동·상태 질문의 열쇠 → 정식 선택지(또는 상태 값) |
| `posterior_k` | 사후 표본 수 K (기본 256) |
| `exact_labels` | 정답을 표본 빈도가 아니라 닫힌 꼴로 계산한 질문의 이름(의도·뚜껑·채움 구간 상태, stops_short·passable, 감속의 넘어섬 — 리뷰 1 m2) |
| `sides` | 비켜설 수 있는 몸의 옆 공간 보임 상태 `[왼쪽, 오른쪽]`(clear / blocked / occluded) 또는 `null` |
| `margin_redraws` | 여유를 다시 뽑은 횟수(보통 0) |
| `margin_fallback` | 닻이 목표 여유에 닿지 못해 가장 가까운 띠를 쓴 여유 이름(P3, 보통 빈 목록) — v0b |
| `group` · `maneuver` · `split` · `holdout` · `pool` · `holdout_vars` | 현상·조합 묶음, P3 움직임 갈래, 나눔, 들어간 빼 둔 칸, 문구 묶음, T3 변수의 보인 값 — §5 |
| `load` · `stack` · `place` · `stale` · `corner` · `conflict` | 새 장면의 관측 요약(종류·클래스·단서 …) — v0b |
| `t6` | T6 교란 레코드만: `{base, variant, map}` |

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
상대가 지나는 줄까지. 통로 폭은 벽에서 벽까지. 앞 거리(간격·교차선·장애물)는 지금 값이다. **마주 옴(head-on)의 위치 보고는 그 시각의 둘
사이 간격**이다("as of 1.15 s ago the gap between you was 3.36 m") — 지금 간격 = 보고 − (상대 속도 + 로봇 속도)·나이, 위치 두 개의 차이 /
간격 시간 = 다가오는 속도(리뷰 1 C1; 장면·사후·계산 과정이 같은 뜻). 바닥에 붙은 보고는 "… or less"로 쓴다(검열된 관측). km/h는 소수
두 자리(속도 격자 0.05 m/s = 0.18 km/h가 정확히 보인다).

### `labels` — 정답

| 질문 | 정답 |
|---|---|
| `noul` | `{"type": "noul", "p_yes"}` — K개 사후 세계에서 그 사건이 일어난 비율(닫힌 꼴이 있으면 그 값 — `meta.exact_labels`) |
| `score` | `{"type": "score", "value"}` — [0, 1] 양의 K개 평균(사건 확률, 가까움, 빈 높이 사용, 3초로 자른 지연) |
| 행동 `choice` | `{"type": "choice", "kind": "action", "probabilities", "expected_cost"}` — 아래 |
| 상태 `choice` | `{"type": "choice", "kind": "state", "probabilities"}` — 이산 상태(의도, 채움 구간, 뚜껑 — 닫힌 꼴; 공의 멈춤 — 비율)의 사후 확률 |
| 규칙 적용 `choice` | `{"type": "choice", "kind": "rule", "probabilities"}` — 질문에 적힌 규칙을 글에 보인 수에 적용한 결과(확률 1) |

**행동 choice (docs/12 §5 정정본).** 선택지 a마다 기대 비용

  E[c_a] = (1/K) Σ_k [ Σ_사건 비용(사건) · 1{사건이 세계 k에서 a로 일어남} · s_k(a) + 진행 손실_k(a) · `delay_per_s` ] + h(a),

확률 p_a = softmax(−(E[c_a] − min_b E[c_b]) / τ). **argmax는 언제나 기대 비용 최소**다(2,000개 전부에서 시험). '각 선택지가 최선이었던
빈도'는 쓰지 않는다 — 대부분의 세계에서 시간을 아끼지만 가끔 큰 사고를 내는 선택지가 높게 나오기 때문이다. 비용표는
`configs/simgen/values-v0.yaml`(사람 100, 아이·노인 150, 차량 120, 물체 15, 쏟아짐 뜨거움 25·약품 40·그 밖 8, 지연 1/초, τ = 2.0).
지시문에 기준이 적힌 질문(약 20 %)은 그 기준의 덮어쓰기(`criteria`)를 쓰고, 기준 문구는 덮어쓰기와 같은 뜻으로 쓴다(avoid_contact =
상대든 무엇이든 닿지 말 것·그 밖은 평소대로, time_critical = 지연 1초 ×3·사고 비용은 그대로 — 리뷰 1 m4).

가치표의 두 기제(리뷰 1 질문 1·6; 값은 사용자가 고른다): s_k(a) = max(floor, min(1, (v / v_ref)²)) — v는 처음 닿는 순간의 상대 속도
(넘어섬은 장애물에 닿는 속도), 닿음·넘어섬 항목에만, 기준이 덮어쓴 항목에는 걸지 않는다; h(a) = 급제동(stop_hard, 또는 회전 전 최대 감속)의
몸별 비용(짐 실은 지게차는 따로). `values-v0`은 둘 다 꺼짐(floor 1, h 0 — 기제가 없는 것과 같은 결과), `values-v0.1`은 제안값
(floor 0.25, v_ref 1.0 m/s·팔 0.5 m/s, h = 바퀴 2·휴머노이드 5·지게차 3·짐 실은 지게차 6·팔 1). 산출물은 사용자가 고르기 전까지 `values-v0`.

**진행 손실(지연)의 정의**(장면마다, 지연 score 질문의 지시문이 같은 말을 한다 — 리뷰 1 m3): P1 장면은 앞으로 3초 동안 잃는 진행(초; 기준 =
등속 또는 바로 출발). P5 감속은 목표(장애물 앞, 몸별 허용치 — 팔 0.01 m, 이동 로봇·휴머노이드 0.05 m, 지게차 0.10 m)보다 일찍 섰을 때
남은 거리 d를 gentle로 다시 다가가는 시간 2·√(d/a_gentle)(넘어서면 0 — 대신 넘어섬 비용) — 급제동이 공짜가 아니다(리뷰 1 I3). P5 회전은
감속·느린 호·재가속까지 회전을 마치는 시각이 늦어지는 만큼(접근 구간 전체에 걸친 필요한 감속 (v0² − v²)/(2·접근)을 편안함 하한과 최대 감속
사이로 자름 — 리뷰 1 I2; '일찍/늦게 줄이기'를 따로 선택지로 두지 않는다).

반올림: 기대 비용은 소수 4자리, 확률은 그 기대 비용으로 계산해 소수 8자리(합 오차 ≤ 4e-8, 계약은 ±1e-6). 기대 비용이 같으면 확률도 같다.

### `truth` — 참 세계와 실제로 일어난 것 (평가용)

* `world` — **정답의 사후에서 한 번 더 뽑은 세계**(`source: "posterior_draw"`, 갈래 `truth`): 숨은 값과 참값(`values`: 의도, 움직임
  매개변수, 참 채움, 출렁임 계수 k, 바닥 기울기, 뚜껑, 가려진 쪽의 장애물, 반응 여부 …), 그 세계의 여유(`margins`), 알려진 기하(`geometry`).
* `realized` — 질문마다 참 세계(K = 1)에서 일어난 것: noul → bool, score → 값, 행동 → `{costs, best}`, 상태 → 참 상태의 열쇠, 규칙 → 답.

참 세계를 정답과 같은 사후에서 뽑으므로 `realized`는 정답에 대해 보정되어 있다(시험). 닻 세계(여유에 맞춰 푼 세계)는 관측을 만드는 데만
쓰고 기록하지 않는다 — 닻을 풀 때 생긴 결합(채움을 로봇의 감속으로 맞춘 것 등)이 참값에 남지 않게(리뷰 1 I4의 곁길). 사후가 감각 모형의
올바른 역인지는 순방향 보정 시험(`test_simgen_sbc.py`: 정답의 사전에서 뽑은 세계 → 관측 → 정답 → 그 세계의 결과)이 본다.
학습은 사후 확률(`labels`)로 한다. 참 결과는 평가용이다(docs/12 §5).

### `rationale`

정답의 계산 과정 한 문단(보고된 수치 → 여유, 사후 의도·채움, 기대 비용 분해와 소프트맥스, 참 세계에서 일어난 것). 사람이 읽는 용도이고
'계산을 안으로 접기' 실험 재료다.

## 2. 누출 경계 (코드 구조 + 시험)

* `world.py`의 `World`(참값·숨은 값)와 `observation.py`의 `Observation`(감각 보고)은 다른 타입이다.
* 서술기(`narrate`)·질문 생성기(`questions`, `phrases`)·사후 표본(`posterior`)·굴리기(`rollout`)·정답기(`labels`)는 `World`를 import하지
  않는다. 세계와 관측을 함께 보는 곳은 `scenes`(닻 세계 풀기 → 관측)와 `generate`(참 세계 = 사후 표본을 World로)·`rationale`뿐이다.
* 시험(`test_simgen_leak.py`): (a) import 검사, (b) **숨은 값만 다른 두 World가 `observe()`를 거쳐 같은 관측을 내고**, 레코드 전체가
  바이트 단위로 같다(닻의 설계 여유만 숨은 값에 걸리면 다를 수 있다); 대조로 관측되는 값 하나를 바꾸면 관측과 글이 달라진다,
  (c) 숨은 전용 이름(intent, slosh, true_ …)이 모델 입력에 없다.

## 3. 결정 여유와 표본

장면마다 목표 여유 m ~ 0.6·N(0, σ_m) + 0.4·U(−M, M)(혼합분포)을 뽑고 닻 세계의 표면 변수를 그 여유에 정확히 맞춘다(σ_m·M은
catalog-v0.1 `margins`). 정답은 설계와 무관한 사전(속성 목록: 채움은 용기의 채움 범위, 의도·뚜껑·출렁임·반응은 속성 목록 과정, 직접 잰
거리·속도·폭은 국소적으로 평평)으로 계산한다 — 설계를 알면 감속 사양으로 채움을 맞히는 곁길이 생긴다(리뷰 1 질문 3).

| 여유 | 정의 | 단위 |
|---|---|---|
| `step_in` | gap(중심) − (반응 거리 + moderate 멈춤 거리) − 반지름 합 = 면 간격 − 멈춤 거리 | m |
| `crossing` | 상대가 진로 가장자리에 닿는 시각 − (로봇이 교차선을 비우는 시각 + 0.3 s) | s |
| `head_on` | 통로 폭 − 두 몸 폭 − 필요한 폭(`passing_clearance`) | m |
| `spill` | (H − f·H − k_ref·(D/2)·a_ref/g) / H, k_ref = 1.3 — 기준 가속(moderate 감속, 또는 지금 속도의 회전 옆 가속)에서 남는 림 높이 비율 | 림 높이 비율 |
| `stop` | 장애물 면 간격 − moderate 멈춤 거리 (p5_braking) | m |

레코드의 약 10 %는 사실 하나만 바꾼 쌍이다: `numeric`(결정 변수 하나만 경계 너머로 — 같은 잡음, 같은 질문 갈래; 보고가 실제로 바뀌는
후보만), `wording`(숫자는 그대로, 상황 옷 단어만 바꿈 — 정답이 같다).

## 4. 버전 규칙

* **`schema`** (`simgen-request/N`): 필드를 빼거나 뜻을 바꾸면 N을 올린다. 필드를 더하기만 하면 그대로 둔다.
* **생성기 `version`** (`v0a` → `v0a.1` …): 같은 seed에서 다른 레코드가 나오게 바꾸면(물리·표본·서술·질문 코드) 올린다.
* **`catalog` / `skins` / `values`**: YAML의 값을 바꾸면 그 파일의 `version`을 올린다. 기본 가치표는 가치 판단이라 공개하고 버전을 붙인다.
* 레코드는 (생성기 버전, 세 설정 버전, seed, 강제한 분야, 쌍 변형)으로 다시 만든다. 물리가 좋아지면 옛 데이터도 새로 뽑는다(docs/12 §12).

## 5. Task v0b — 새 현상·조합·한국어·빼 둔 칸·시험 세트

**장면(18).** v0a의 여섯에 더해: `p3_carrying_tall`(키 큰 물건을 싣고 감속 `brake`·회전 `turn`·출발 `start` — `meta.maneuver`),
`p3_ramp`(경사로 `down`·`up`), `p3_place_stack`(쌓기), `p4_place_check`·`p4_insert`(안착·끼우기 확인), `p6_stale_track`(가려진 상대가
아직 있나), `p6_blind_corner`(모퉁이), `p6_sensor_conflict`(라이다·카메라 불일치), 조합 `p3p5_carrying_mixed`·`p1p6_emerging_step_in`·
`p4p6_stale_check`, 그리고 **학습에 없는** `p1p3_step_in_carrying_tall`(T4 전용). 묶음(`meta.group`) = P1·P3·P4·P5·P6·P1+P5·P3+P5·P1+P6·P4+P6
(학습)·P1+P3(T4). 학습 세트는 분야 × 묶음으로 층화한다(분야 한 바퀴마다 다음 묶음).

**결정 여유(새 것).** `tip` = (B/2)/(com_mid·H) − r_ref(관측으로 정해지는 명목값 — com_mid는 무게중심 높이 비율 범위의 가운데; 기준 움직임의 쏠림 비 —
단단한 제동/g, 지금 속도의 v²/(R·g), 보통 출발/g, 경사로 (sin φ + a/g)/cos φ), `stack` = L_low/2 − |d|(명목 — 무게 치우침 c·놓기 오차 ε는 숨은 값), `seat` = logit P(제대로 | 보고),
`sight` = 다시 보이는 거리 − 최대 제동 멈춤 거리, `corner` = 모퉁이까지 거리 − 최대 제동 멈춤 거리, `late` = 늦은 감지 거리 − 최대 제동 멈춤 거리.
P3는 로봇 쪽 결정 변수(감속·회전 속도·출발 가속 — 글에 보이는 해상도의 값)와 물건 기하를 함께 풀어 정확히 맞추고, 그 상황의 물건으로는
닿을 수 없는 목표만 가장 가까운 띠를 쓴다(`meta.margin_fallback`). 정확히 맞출 수 없는 여유: `seat` ±0.35(거절 표본), `sight` ±0.025,
`stack` ±0.0025(글의 해상도).

**사후.** P4는 상태(좋음·둘째·셋째±)의 정확한 베이즈(무게 있는 세계 — `Batch.weights`), P6의 있음·출현·따라옴은 세계를 둘로 나눠 정확한
무게로 센다. 정답은 무게 평균(`labels.wmean`). 닫힌 꼴(`meta.exact_labels`): P4 `ok_now`·상태 분포, P6 `present`·`emerge_3s`·
`stops_before_corner`·`follower`, P3 쌓기 `topple|…`·무게 쏠림.

**'다시 볼까'(정보의 가치).** 다시 보기 선택지(`look_again`, 쌓기는 `check_weight`)의 기대 비용 = 보는 시간 × 지연 비용 +
E_신호[min_a E[c_a | 신호]] — 결정 규칙은 정답의 사후에서 정하고 참 세계에는 그 규칙을 적용한다(`truth.realized`가 보정된다). 이산 신호
(다시 봤더니 있다/없다, 두 센서의 새 보고, 무게 확인 값의 구간)는 닫힌 꼴, 재측정(P4)은 준난수 적분(할턴 4,096점 — 차원별 부호 대칭).
질문 종류 `voi`(choice: 지금 정하기 / 먼저 보기)는 `labels`에서 `kind: action`(기대 비용 소프트맥스)이다.

**한국어(`meta.language` = `ko`, 학습·T1~T7의 약 15 %).** 서술기 `narrate_ko`(같은 사실·같은 수, 단위 기호는 영어와 같은 규칙), 질문 문구
`phrases_ko_data`(영어와 같은 열쇠·같은 묶음 크기), 상황 옷 `configs/simgen/skins-ko-v0.2.yaml`, 조사는 `korean.kfmt`가 받침에 맞춘다.
같은 순간을 두 언어로 쓰면 관측·정답이 같다(시험).

**빼 둔 칸과 나눔(`configs/simgen/holdout-v0b.yaml`이 정본).** 학습 나눔은 문체 `log`, 분야마다 다섯째 상황, 빼 둔 바꿔 쓰기 묶음
(`pool: heldout`), 조합 P1+P3, T3 범위 칸 10개(로봇 속도 상위·추적 나이 > 1 s·높이/바닥 폭 > 5·경사 > 7°·거친 게이지·오래된 이미지 > 3 s·
채움 보고 > 0.92·오래된 목격 > 4 s·모퉁이 옆 거리 < 0.5 m)을 **만들 수 없다** — 고를 수 없게 막고, 범위는 좁혀 뽑고, 보고값이 넘으면 시도마다
새 잡음으로 다시 뽑는다(`generate.build_moment`, `holdout.violations`). 레코드: `meta.split`, `meta.holdout`(이 레코드가 들어간 빼 둔 칸),
`meta.pool`, `meta.holdout_vars`(T3 변수의 글에 보인 값 — 학습 세트 전부를 레코드만으로 다시 잰다), `generator.split`(나눔 규칙).

**시험 세트(`artifacts/simgen/v0b/tests/T{1..7}.jsonl` + `manifest.json` — sha256·seed 범위·빼 둔 칸의 정의·생성기 git).** T1 = 학습과 같은
분포, T2 = 레코드마다 빼 둔 축 하나 이상(문체·상황·바꿔 쓰기 묶음), T3 = 칸 하나의 시험 쪽(칸마다 같은 수), T4 = P1+P3만, T5 = 사실 하나만
바꾼 쌍, T6 = 기준 300 × 교란 12(`perturb.VARIANTS` — 상태·지시·선택지 설명 바꿔 쓰기, 순서 뒤집기, 섞기, 열쇠 바꾸기, 서식 잡음, 한 가지 더
정하기, 단위, 숫자 표기, 좌우 거울상, 사실 순서; `meta.t6 = {base, variant, map}` — map은 변형의 질문 → 같은 것을 묻는 기준 질문; 정답이
대응으로 정확히 같다, 거울상은 사후 세계를 공통 난수로 뒤집는다; 마주 옴은 좌우가 다른 사전이라 기준에서 뺀다), T7 = 보정(K = 1024).
seed 범위는 나눔마다 겹치지 않는다(학습 1,000,000~, T1 2,000,000~ …); 요청 글의 해시(`holdout.request_hash`)가 학습과 겹치지 않는다(시험).

## 6. 모듈과 명령

| 모듈 | 하는 일 |
|---|---|
| `catalog` | 설정 셋 적재·구조 검사 |
| `spec` | 중립 타입(Reading, RobotSpec, Batch), 이름 붙은 난수 갈래, 채움 구간 |
| `physics` | 멈춤 거리, 가감속, 가로지름 시각, 정적 액면 쏟아짐, 회전 감속·지연, 구간 최근접·처음 닿는 속도 |
| `world` / `observation` | 참 세계 / 관측 |
| `skins` | 상황 옷 항목 고르기 규칙(상황 태그·몸·장면 제약 — 의미 일관성) |
| `scenes` | 여유에서 닻 세계 풀기, 감각 모형, 사실 하나 바꾸기 |
| `posterior` | 관측 → 사후 세계 K개, 닫힌 꼴 정답 |
| `rollout` | 선택지 × 세계 굴리기 → 사건·양·충격 속도·급제동 |
| `labels` | 비용 모형(두 기제), 정답(noul·score·행동·상태·규칙)과 참 결과 |
| `textfmt` · `phrases` · `narrate` · `questions` | 단위 표기, 문구 묶음, 서술기 3종, 질문 생성기 |
| `record` · `rationale` · `generate` | 형식 검사·직렬화, 계산 과정 문단, 생성 흐름·층화·쌍 |
| `stats` · `examples` | 묶음 통계, 검수용 예시 |
| `p3` · `p4` · `p6` (관측 쪽) / `p3_world` · `p4_world` · `p6_world` (세계 쪽) | 새 현상의 사후·굴리기·닫힌 꼴 / 닻 세계·여유·감각 모형 |
| `voi` | '다시 볼까' 신호 모형(이산 닫힌 꼴, 연속 준난수 적분) |
| `holdout` · `testsets` · `perturb` | 빼 둔 칸과 나눔, 시험 세트 T1~T7과 manifest, T6 교란 |
| `korean` · `narrate_ko` · `phrases_ko` · `phrases_ko_data` · `state_text` · `rule_text` | 조사·한국어 서술기·한국어 문구, 상태·규칙 글(두 언어) |
| `phrasing` · `phrases_v01` · `phrases_ko_v01` | 문구 판(v0b·v01)의 표·고르기 흐름·빼 둔 문구 근접 규칙·쓰임 통계, v01의 새 학습 문구(영어·한국어) — §7 |

```bash
uv run python scripts/simgen_generate.py --n 2000 --seed 0 --out artifacts/simgen/v0a/sample-2000.jsonl --stats artifacts/simgen/v0a/stats.json [--values configs/simgen/values-v0.1.yaml]
uv run python scripts/simgen_examples.py --out-dir artifacts/simgen/v0a
uv run python scripts/simgen_value_compare.py --n 2000 --out artifacts/simgen/v0a/value-compare.json   # 두 가치표의 최선 행동 분포
uv run python scripts/simgen_mutants.py --work artifacts/scratch/v0a/mutants --out artifacts/scratch/v0a/mutants/result.json   # 변이 시험
# Task v0b
uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v0b/train-50k.jsonl --workers 12 [--values configs/simgen/values-v0.1.yaml]
uv run python scripts/simgen_v0b.py tests --out-dir artifacts/simgen/v0b/tests --workers 7
uv run python scripts/simgen_v0b.py examples --out artifacts/simgen/v0b/examples-30
# Task v0.1a — 학습 세트 v01(같은 seed·설정, 학습 문구 판 v01 + 사실 순서 섞기 0.5) + v0b와 레코드마다 맞대기
uv run python scripts/simgen_v0b.py train --out artifacts/simgen/v01/train-50k.jsonl --phrasing v01 --workers 12
uv run python scripts/simgen_v0b.py compare --v0b artifacts/simgen/v0b/train-50k.jsonl --v01 artifacts/simgen/v01/train-50k.jsonl \
    --v0b-stats artifacts/simgen/v0b/train-50k.stats.json --v01-stats artifacts/simgen/v01/train-50k.stats.json --out artifacts/simgen/v01/train-50k.compare.json
```

## 7. Task v0.1a — 학습 문구 판 v01과 사실 순서 섞기

v0 학습 모델은 학습 문구(열쇠마다 3개)로 물으면 가정 질문을 읽지만 T2의 빼 둔 바꿔 쓰기에서 무너지고(가정 행동 대신 '계획대로'로 답함),
사실 순서를 바꾸면 기반보다 더 흔들렸다. v0.1은 **데이터만** 고친다(모델·학습 설정은 v0 그대로 — v0 → v0.1의 차이를 이 두 변화에 돌린다).

* **문구 판(`phrasing`).** `v0b`(기본) = 기존 `train`·`heldout` 목록 그대로(바이트 단위 — 시험이 해시로 고정). `v01` = 학습 묶음
  `train_v01` = v0b `train` + 새 문구(`phrases_v01`: 영어 질문 열쇠 92개마다 새 문구 5~7개 — 가정 질문(`{action}`을 쓰는 39개) 7개, 선택지
  설명 41개마다 5개 — 743개; `phrases_ko_v01`: 한국어 같은 표의 열쇠마다 3개 — 399개). 빼 둔 묶음(T2 전용)은 두 판이 같다. 새 문구는 뜻이 같다(가정 행동·시간 창·
  세는 것·점수 기준 방향·답 형식)·빼 둔 문구와 같지 않고 낱말(어절) 유사도 < 0.8·글자 유사도 < 0.75·빼 둔 문구에만 있는 3-gram 금지·자리
  표시는 그 열쇠의 v0b 학습 문구의 것만, 그 열쇠의 v0b 학습 문구 모두에 있는 것은 반드시(가정 질문은 행동 자리)·**언어 전체에서** 빼 둔 문구에만
  있는 내용어(불용어가 아닌 낱말·어절)와 내용어를 담은 3-gram 금지 — 다른 열쇠의 빼 둔 말도 학습에 들어가지 않게(리뷰 1 M1~M3;
  `phrasing.violations`, 시험 `test_simgen_phrasing.py`).
* **고르기는 주 흐름을 건드리지 않는다.** v0b 묶음 크기로 뽑던 수를 그대로 뽑고(질문 이름·조리법·선택지·순서·정답이 v0b와 같다), 실제 문구는
  레코드 seed + 열쇠 이름의 별도 흐름(`stream(seed, "phrasing-v01", 열쇠)`)에서 넓은 묶음 전체 가운데 고른다. 고른 것은
  `meta.phrasing.picks`([열쇠, v0b 번호, v01 번호]) — 같은 seed의 v0b 레코드가 쓴 문구도 여기서 안다.
* **사실 순서 섞기(`fact_shuffle`, v01 기본 0.5).** 레코드마다 그 확률로 T6 '사실 순서'와 같은 장치(`RenderPlan.fact_seed` → `narrate.ordered`)를
  켠다; 동전과 seed는 별도 갈래 `train-facts`. 섞는 단위는 T6과 같다: 사실 조각(한 조각 안의 순서 — 두 시각의 위치 보고 등 — 는 그대로),
  첫 줄(상황·몸·과제)과 끝(방해 문구·힌트·권고·가능한 동작)은 제자리, 문단·칸(ROBOT / PERCEIVED)은 그대로. `meta.fact_order` =
  `{p, shuffled, seed, changed}`(changed = 글이 실제로 바뀌었나 — 조각이 하나뿐이거나 섞어도 같은 순서면 False).
* **학습 나눔에서만.** 기본값이 아닌 손잡이를 시험 나눔(T1~T7)이나 교란 변형에 주면 `assemble`이 거절한다 — 시험 세트 생성 경로는 늘 v0b이고,
  봉인된 T1~T7과 v0b 학습 세트는 지금 코드로 바이트 단위로 다시 만들어진다(시험이 앞부분의 해시를 고정). 기본값이면 레코드에
  `generator.phrasing`·`generator.fact_shuffle`·`meta.phrasing`·`meta.fact_order`를 적지 않는다(없음 = v0b).
* **생성기 버전.** `generator.version`은 `v0b` 그대로다(기본 경로의 출력이 같다). v01 레코드는 `generator.phrasing = "v01"`·
  `generator.fact_shuffle`로 구별되고 그 칸으로 다시 만들어진다(`rebuild_record`).
* **알려진 v0b 결함(고치지 않음 — v0b 글을 바꾸면 봉인된 시험 세트를 다시 만들 수 없다).** P6 오래된 추적의 `{spot}`은 전치사구('in the
  corridor outside room 214')인데 v0b의 상태 글('was standing at {spot}')·규칙 글('the distance to {spot}')·학습 문구 둘('still at {spot}',
  'toward {spot}')이 앞에 전치사를 붙인다. v01의 새 문구는 `{spot}`을 전치사구로 쓴다.


## 8. Task v0.2 — wording balance `v01b`

v0.1 trained on `v01`, whose uniform pick over the wide pool cut the original (v0b) wordings — the ones T1–T7 use — to about a third
of the English picks (0.36 EN, 0.50 KO). v0.1 then lost a little in-distribution action regret. `v01b` restores them to half:

* **Pick.** Same wide pool `train_v01` (no new wordings, no new rules). Each use keeps, with probability 0.5
  (`phrasing.V01B_ORIGINAL_SHARE`), **the v0b wording the main stream drew for this seed** — byte-for-byte the v0b record's wording —
  and otherwise takes one of the key's new wordings uniformly (`phrasing.pick_v01b`).
* **Streams.** The coin and the new index come from a third stream, `stream(seed, "phrasing-v01b", key)`; each use draws both. The main
  stream is untouched, so moments, questions, options and labels are byte-identical to v0b and v01 for the same seeds. With the same
  `fact_shuffle` (0.5, stream `train-facts`) the state text is byte-identical to v01's as well — `v01b` differs from `v01` only in the
  instruction and option texts (`scripts/simgen_v0b.py compare --same-facts`).
* **Same guard and log.** Training split only (`assemble` refuses it elsewhere); `generator.phrasing = "v01b"`; `meta.phrasing.picks` keeps
  the `[key, v0b index, wide index]` form, so `wording_stats` / `wording_summary` read it unchanged (original share = picks below the v0b
  list length).
* **Data.** `scripts/simgen_v0b.py train --phrasing v01b --out artifacts/simgen/v02/train-50k.jsonl` (fact shuffle 0.5 by default).
