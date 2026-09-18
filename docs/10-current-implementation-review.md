# 현재 기획·구현 결과 리뷰

검토일: 2026-09-18. 기준 commit: `47258d6102f8dbaaa944e5a58655220dae326830`. 리뷰 시작 시 작업 트리는 깨끗했다. 검토 대상은 01~09 문서, 현재 `src/robo_jev`, 설정과 테스트다. 기존 구현을 수정하지 않고 검증용 산출물과 이 리뷰를 추가했다.

## 1. 현재 위치

**입출력·데이터·시뮬레이션 기반은 실제 코드로 진전됐지만, 모델 개발과 학습의 결과는 아직 없다.** 기존 테스트 432개는 모두 통과했다. 그러나 추가한 연결 검증에서는 작업 실행·후보 제공·관측 정보 경계의 오류를 재현했다. 이 상태에서 로봇 데이터를 대량 수집하면 하네스의 결함을 모델 정답이나 모델 실패로 기록할 수 있다.

| 영역 | 확인된 결과 | 남은 범위 |
| --- | --- | --- |
| 설계 | 08을 정본으로 L1-a, 10개 질문, 그리퍼 상태, commitment 조건, TBPTT, 부분 라벨, DAgger 이력 보존을 문서화 | 모델·학습·성능으로 검증되지 않은 설계 |
| 계약·D0 | 입력 경계, 타입·라벨·참조 검증. 단일 요청 64건, 합성 스트림 4개/400틱 fixture | manifest의 `reviewed_by`가 비어 있어 사람 검수 완료 gold는 아님. 실제 물리 rollout 데이터와 구분 |
| 비로봇 데이터 | 4개 분야의 결정적 생성기, 계보 분할·표현/후보 변형·QA. seed 17의 2,000건 재생성에서 QA 오류 0 | 현재 기본 설정의 OOD holdout 목록은 비어 있음. 의미 일반화 시험과 독립 라벨 검수 필요 |
| 시뮬레이터·실행기 | 실제 MuJoCo/robosuite, E0/E1 장면·외란·지시 변경, snapshot/restore, 50Hz 목표/500Hz 물리, lease·반사·ACK | 국면별 실제 동작과 충돌 거절 등 아래 결함 해결, 작업 완료 통합 검증 |
| 지각·하네스 | 구조화 상태 추출, 참값 어댑터, 후보·경유점 생성, 규칙 판단기, 조합·레코드 조립 | 아래 후보·정보 경계·조합 문제. 참값 어댑터는 3D 재구성 구현이 아님 |
| 로봇 데이터 제작 | 에피소드 레코드 도구와 짧은 연결 테스트 | 스크립트 전문가, 키프레임 counterfactual rollout 라벨러, D1 400 에피소드, DAgger 수집 사이클 |
| 모델·학습 | 문서 설계 | tokenizer 직렬화, backbone 연결, pointer readout, hybrid 분기/stream state, loss, TBPTT, optimizer/checkpoint, GPU 선정·성능 측정 |

모델용 모듈과 학습 실행 스크립트는 현재 파일 목록에 없다. 10Hz는 목표이며 이번 검증은 LLM 추론 지연을 포함하지 않는다.

## 2. 재현된 문제

### I1 · P1 — 알려진 장애물이 있는 직선 경로가 그대로 승인된다

위치: [harness/robot.py](../src/robo_jev/harness/robot.py)의 `_path_block`, [sim/controller.py](../src/robo_jev/sim/controller.py)의 `_check_reach`와 `apply`.

하네스는 후보에 `path_clear=False`, `blocker=o1`을 계산해 놓고도 선택된 direct 경로를 그대로 명령으로 보낸다. 컨트롤러의 `_check_reach`는 원점 기준 거리와 바닥 높이만 검사한다. 전환 구간 검사는 실제 궤적 검사 대신 호출자가 준 `forbidden_segment` flag에 의존하지만, 하네스가 내는 명령에는 그 flag가 없다.

**재현:** 말단 `[0,0,200]`, 목표 물체 `[400,0,0]`, 장애물 `[200,0,140]`. 후보는 막힌 경로라고 보고했지만 direct 명령은 `ack.applied=True`였다. 최초 commitment 선택에서 기본 경로가 direct이므로 정상적인 새 행동 선택에도 나타날 수 있다.

명령으로 실제 적용할 경로와 혼합 구간을 관측된 장애물에 대조하고, 막힌 경로는 거절·정지·유효한 경유 경로로 처리해야 한다. 금지 flag를 수동 주입하는 테스트만으로 이 연결을 검증할 수 없다.

### I2 · P1 — `lift`와 `transport`가 놓기 위치로 바로 이동한다

위치: [harness/robot.py](../src/robo_jev/harness/robot.py)의 `_geometry_for`, `_phase`, `_path_block`(특히 1463행).

물체를 들고 있으면 후보의 `action_mm`은 목적지의 낮은 놓기 위치다. 그런데 `_path_block`은 approach를 제외한 모든 국면에서 `action_mm`을 쓴다. lift에서 현재 XY를 유지하며 들어 올리거나, transport에서 이동 높이를 유지하는 동작이 없다. place 국면도 XY 거리만으로 진입해 높이 도달을 확인하지 않는다.

**재현:** 물체를 든 말단 `[300,0,0]`에서 phase는 `lift`인데 명령 목표는 `[30,240,-52]`였다. 들어 올려야 할 때 오히려 더 낮은 목적지로 이동한다.

접근·파지·lift·transport·place마다 목표 자세와 완료/readiness 조건을 분리해야 한다. 그리퍼 개방은 놓기 위치와 지지/해제 조건을 확인한 뒤 허용해야 한다. 이 검증은 전문가 rollout을 만들기 전에 필요하다.

### I3 · P1 — 후보 절단이 기능·목적지를 체계적으로 없앤다

위치: [harness/robot.py](../src/robo_jev/harness/robot.py)의 `_prune`(559~583행).

대상별 round-robin은 적용하지만 대상 안에서는 의미 키를 사전순으로 정렬한 앞부분만 고른다. `grasp`가 `push`보다, `side`가 `top`보다, 앞 이름의 목적지가 뒤 이름보다 먼저 온다. 후보가 많을수록 같은 종류만 남는다.

**재현:** 보이는 물체 8개·영역 3개에서 실행 가능 조합 160개 중 29개를 남겼다. 남은 29개는 전부 grasp이고 목적지는 z0 또는 z1뿐이었다. 지시가 요구한 o7→z2 후보는 0개이고 push도 0개였다. 세 고정 후보를 합치면 K=32는 지키지만 올바른 행동은 고를 수 없다.

대상뿐 아니라 기능·접근·목적지의 범위를 보존하는 표본 방식과 현재 commitment 예약이 필요하다. 담을 수 없으면 추가 후보 요청/단계적 좁히기 정책을 정의해야 한다. 현재 `inclusion_rate=kept/feasible`은 목록 보존 비율이지 **정답 후보 포함률**이 아니다. 후자는 offline 정답과 비교해 별도로 측정해야 한다.

### I4 · P1 — 행동 전환 틱에 이전 commitment의 그리퍼 답이 적용된다

위치: [harness/robot.py](../src/robo_jev/harness/robot.py)의 `_aux`(1360행), [기존 테스트](../tests/test_harness.py)의 `test_aux_answers_are_discarded_on_the_switch_tick`.

문서와 함수 설명은 전환 틱에 경로·속도·힘·그리퍼 네 답을 버린다고 한다. 실제 코드는 앞의 세 값만 switch 분기에서 바꾸고, `q_gripper`는 분기 바깥에서 항상 읽는다. 기존 테스트도 세 값만 확인한다.

**재현:** 현재 그리퍼가 closed인 상태에서 새 행동을 선택하고 이전 조건의 `q_gripper=open`을 주면, `aux_discarded`를 기록하면서도 open 명령을 만든다. readiness 하중 조건을 만족하는 실행기는 개방 이벤트 `ge-0001`을 발생시켰다.

전환 틱의 그리퍼는 실제 현재 상태를 유지해야 한다. discarded 기록, adopted 결과, 실제 command/ACK가 모두 일치하는지 확인해야 한다.

### I5 · P1 — 가려진 물체의 simulator 외란이 모델 입력으로 샌다

위치: [perception/pointworld.py](../src/robo_jev/perception/pointworld.py)의 `GroundTruthAdapter.reconstruct`(328~343행), [sim/environment.py](../src/robo_jev/sim/environment.py)의 `_apply_disturbance`.

숨은 물체의 자세는 이전 관측으로 유지하지만, simulator가 만드는 `disturbance_applied` 이벤트는 가시성 검사 없이 `Reconstruction.events`에 복사된다. 이 이벤트는 `extract`를 통해 모델 상태로 전달되고, 이동 대상 판정에도 사용된다. 08은 관측 가능한 원시 신호로부터 사건을 만들도록 명시한다.

**재현:** o1을 가리고 참값 위치를 바꿨을 때 모델에 보이는 자세는 이전 `[200,220,-80]`을 유지했지만, 이벤트에는 `disturbance_applied: o1`이 그대로 들어갔다. 모델은 실제 앞단에서 알 수 없는 외란의 대상과 발생을 알게 된다.

simulator 사건은 evidence/평가용으로 보존하고, 모델 입력 사건은 관측된 변화나 실제 센서 신호로 생성해야 한다. 숨은 세계만 바꾼 대조 검사에는 자세뿐 아니라 events·moving·derived도 포함해야 한다.

### I6 · P2 — 기하 나이가 모델 처리·전송 중 증가하지 않는다

위치: [harness/robot.py](../src/robo_jev/harness/robot.py)의 `_geometry_fault`(957행), [sim/controller.py](../src/robo_jev/sim/controller.py)의 `_lifetime_fault`(398~405행).

두 검사 모두 요청을 만들 때 기록한 `geometry_age_ms`를 그대로 사용한다. 적용 시각까지 경과한 시간을 더하지 않아, 문서의 기하 유효기간을 넘긴 명령도 승인된다.

**재현:** 요청 시 moving 대상의 기하 나이 190ms, 답 도착까지 100ms. 실제 기하는 290ms로 200ms 제한을 넘지만 observe로 분기하지 않고 컨트롤러도 `applied=True`로 승인했다. 로봇 고유 감각의 관측 나이는 100ms여서 별도 deadline도 통과한다.

대상의 절대 `geometry_observed_at`을 전달해 적용 시각에 나이를 계산하는 방식이 명확하다. 상대 나이를 유지한다면 기준 시각을 함께 전달하고 하네스·실행기가 중복 또는 누락 없이 갱신해야 한다.

## 3. 실제 검증 결과와 한계

### 기존 테스트

`uv run pytest -q` → **432 passed in 11.22s**. 계약, 생성 결정성·QA, 지각 입력, controller 전이, simulation replay, 하네스 조합의 기존 검사다. 이 통과가 위 여섯 반례의 부재나 작업 성공을 뜻하지는 않는다.

### 비로봇 데이터

`generate_records(2000, seed=17)`와 `validate_dataset`을 실행했다.

| 항목 | 결과 |
| --- | --- |
| 상태 | 2,000 |
| 질문 / 라벨 | 15,969 / 14,018 |
| 마스킹 질문 | 1,951 |
| 원본 계열 | 1,543 |
| split별 레코드 | train 1,419 / dev 199 / calibration 213 / test 169 |
| 계약·자동 QA 오류 | 0 |
| 분야 | dom 501 / spatial 500 / rules 500 / workflow 499 |

자동 QA는 구조·계보·참조·분포·위치 편향을 확인한다. 이 숫자는 모든 의미 라벨을 독립적인 사람이 검증했다는 뜻이 아니다. 이번 생성은 리뷰용 재생성이고 D1 최종 동결 완료로 취급하지 않는다.

### 실제 E0 폐루프

기존 `test_the_harness_drives_a_real_e0_episode_and_records_it`은 실제 환경에서 10번의 판단 틱을 돌려 명령 승인과 레코드 정합성을 확인한다. 작업 완료를 요구하지 않는다. 이를 30초/300 판단 틱으로 늘려 seed 17·29·43을 실행했다. 규칙 판단기와 현재 하네스를 쓰고 매 판단 틱 사이에 50Hz 제어 5회를 진행했다.

| seed | 파지 관측 | 완료 게이트 | 머문 상태 | 명령 승인 |
| --- | --- | --- | --- | --- |
| 17 | 없음 | 없음 | instr/replan 300틱 | 300/300 |
| 29 | 없음 | 없음 | approach 10틱 뒤 observe 290틱 | 300/300 |
| 43 | 없음 | 없음 | approach 11틱 뒤 observe 289틱 | 300/300 |

이는 작은 연결 점검이며 일반적인 성공률 추정이 아니다. 이 세 실행은 lift에 도달하지 않았으므로 실패 원인을 I2 하나로 귀속하지 않는다. 명령이 승인되는 것과 작업이 진행되는 것이 다르다는 증거다. 관측 부족과 지시 부족의 구분, 재관측/재계획 분기의 실제 복구 효과를 포함한 E0 완료 검증이 필요하다.

### 재현 파일

- [여섯 반례와 2K QA 재현 스크립트](../artifacts/reviews/current-implementation/probe.py)
- [실행 결과 JSON](../artifacts/reviews/current-implementation/evidence.json)
- [30초 E0 연결 점검 스크립트](../artifacts/reviews/current-implementation/closed_loop_probe.py)

```sh
uv run pytest -q
uv run python artifacts/reviews/current-implementation/probe.py
uv run python artifacts/reviews/current-implementation/closed_loop_probe.py
```

반례 스크립트의 Controller readiness 입력 일부는 문제를 분리하기 위한 합성 관측이다. 실제 물리 실험과 섞어서 성공률로 집계하지 않는다. 현재 반례 출력은 실패 동작을 기록하는 것이며 기대 동작의 통과 결과가 아니다.

## 4. 기획과 구현의 연결 평가

이전 리뷰의 구조 수정은 문서에 상당 부분 반영됐다. 특히 공통 상태와 일시적 결정 분기, L1-a의 격리 예외, 에피소드 단위 state 전달, 그리퍼 상태 출력, 실제 실행 이력을 보존하는 계약은 이전보다 구체적이다. 실제 구현에서는 타입·입력 경계·snapshot·명령 기록을 함께 검사하는 기반도 생겼다.

다만 현재 진행 상황을 문서가 따라가지 못한다. [06](06-execution-roadmap.md) 13행은 여전히 저장소에 문서와 DOCX만 있다고 하고 모든 Task가 미완료 표시다. [README](README.md)도 완료한 것이 계획 문서뿐이라고 설명한다. 각 Task를 구현·단위 검증·폐루프 인수·학습 실측으로 나눠 현재 산출물과 연결해야 한다. 이 리뷰를 완료 체크의 대체로 사용하면 안 된다.

연구 결과로 아직 주장할 수 없는 것은 다음과 같다.

- Jev식 직접 readout의 품질·범용 의미 일반화·확률 보정.
- hybrid 공유와 분기의 logits/loss/gradient 정합성.
- 선택한 backbone에서 10Hz 및 100ms deadline 달성.
- 규칙 기준군 대비 모델의 추가 기여.
- 3D 카메라 재구성 앞단에서의 성능과 실로봇 전이.

## 5. 다음 작업의 우선순위

1. **로봇 데이터 확대 전에 I1~I6을 해결한다.** 반례를 회귀 검사로 옮기고 후보 포함률·관측 정보 경계까지 검증한다.
2. **E0에서 실제 작업을 끝내는 경로를 만든다.** 접근→파지→lift→transport→place의 물리 조건, 재관측과 재계획의 효과를 확인한다. 이후 지시 변경·외란이 있는 E1을 연다. 100개 rollout 속도 측정도 이 경로가 유효해진 뒤 수행한다.
3. **모델 최소 경로 구현을 병행한다.** tokenizer 기반 실제 토큰 집계, P0 pointer readout, D0 loss/gradient와 checkpoint, 작은 hybrid의 일시적 분기·상태 이어가기 검증을 우선한다. 환경 기능만 계속 늘리며 모델 검증이 뒤로 밀리지 않도록 한다.
4. **그 결과로 Task 2b와 학습 예산을 실행 가능한 값으로 바꾼다.** 모델·데이터·하네스 각각의 통과 근거를 갖춘 뒤 D1 400 에피소드와 본 학습으로 확대한다.

이번 단계의 판단은 “계획만 있는 상태를 벗어났다”까지다. “로봇 작업 파이프라인이 성립했고 모델 학습을 검증했다”는 단계에는 아직 도달하지 않았다.
