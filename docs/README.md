# 병렬 판단 모델 연구 문서

이 폴더는 연구 설계 문서와 외부 검토의 색인이다. 프로젝트 소개는 뿌리의 [README.md](../README.md)를 본다. [개발 계획서 v2](../Steerix_Robotics_Jev_Development_Plan_v2.docx)를 출발 자료로 삼으며 현재 연구 방향은 아래 문서에 반영한다.

**하네스가 상태·분야별 질문·타입·후보를 구성하고, 모델은 여러 질문의 선택·확률·점수를 병렬로 반환한다. 하네스가 답을 조합해 실행한다.** 모델이 질문과 후보를 생성하는 역할까지 맡지는 않는다.

직접 개발할 범위는 모델 구조, 학습 데이터 생성 파이프라인, 실제 학습·체크포인트, 판단 평가와 로봇 폐루프 검증 전체다. 로봇은 첫 응용이며 질문 의미와 후보를 특정 행동 목록에 고정하지 않는다.

## 현재 문서

| 문서 | 내용 | 상태 |
| --- | --- | --- |
| [연구 컨셉과 목표](01-research-concept-and-goals.md) | 병렬 판단 모델과 하네스의 책임, 입출력, 연구 목표와 비교 원칙 | 연구 설계 초안 |
| [첫 로봇 하네스와 학습 문제](02-task-and-learning-problem.md) | 대표 작업, 질문 묶음, 답의 조합·실행, 데이터 생성과 실제 학습 | 단일 팔·평행 그리퍼를 전제로 한 초안 |
| [모델 구조와 학습 설계](03-model-and-training-design.md) | backbone 후보, 공유 prefix·질문 분기·후보 readout, 손실, 실제 학습 단계와 기준군 | 구현 전 설계 |
| [데이터 생성 계획](04-data-generation-plan.md) | 다분야 질문 묶음, 라벨 근거, 시뮬레이션 결과, split·QA, 5K→50K와 반복 수집 | 제작 계획 |
| [실험 환경·GPU 클라우드·비용](05-experiment-and-cloud-plan.md) | 시뮬레이터, 학습 환경, GPU 메모리·가격·비용, checkpoint·평가 | 공개 가격 확인, 자원 미생성 |
| [첫 학습 파이프라인 실행 계획](06-execution-roadmap.md) | 파일·인터페이스·인수 검사, 최초 학습까지의 순서, 12주 산출물과 판단 기준 | 구현 대기 |
| [설계 변경 검토](07-design-change-review.md) | 다섯 가지 변경에 대한 외부 검토(R1~R6)와 공개 근거 해석 정정 | 검토 결과, R1~R6 반영 완료 |
| [스트리밍 입출력·데이터 계약](08-streaming-io-and-data-contract.md) | robojev의 계층(L2), 10Hz 스트림 입력(v0.3: 변화분 틱·서식 v0.3), 10개 질문 세트(K≤12), 조합 규칙, 컨트롤러 계약, 라벨(비용 허용 집합·hold∉A)·레코드 정의, 01~06 반영 목록 | **계약 v0.3**(2026-09-19, HANDOFF 결정 1·2). 로봇 스트림의 정본이며 01~06에 반영 완료 |
| [스트리밍 계약 검토](09-streaming-contract-review.md) | 08에 대한 외부 검토(S1~S7)와 수치 검산 | 검토 결과, 08에 반영 완료 |
| [현재 구현 리뷰](10-current-implementation-review.md) | 구현 상태 외부 검토: 재현된 결함 I1~I6, E0 폐루프 연결 점검, 비로봇 2K 재생성 QA, 우선순위 | 검토 결과. I1~I6 모두 타당 확인(47258d6에서 재현). 계약 보강은 08·02에 반영, 코드 수정은 Task 3b 수정 라운드 |
| [DGX Spark 인계 검토](11-dgx-spark-handoff-review.md) | main merge 뒤 외부 검토: 재개 정체 검사(S1), 미지원 model_id(S2), 인계 자료 전달(S3), 토큰 실측과 예산(S4), tokenizer 재현(S5) | 검토 결과. S1~S5 모두 타당. S3·S4는 `HANDOFF.md`·03·05·08에 반영, S1·S2·S5는 코드 수정(fix/handoff-review-11) |

## 상태와 다음 단계

프로젝트의 취지·핵심 아이디어·검증된 것과 아직 아닌 것·로드맵은 저장소 뿌리의 [README.md](../README.md)(영어)에 있다. 그 문서의 Status 표가 태스크 상태의 정본이며, 다른 머신에서 이어가는 절차는 [HANDOFF.md](../HANDOFF.md)에 있다.

## 설계 변경 이력 (요약)

| 시점 | 변경 | 근거 |
| --- | --- | --- |
| 2026-09-18 근본 검토 | readout을 후보별 분기에서 결정 위치 pointer로; backbone을 지연 예산에서 역산; 주 결정을 결합 행동 후보의 단일 선택으로; 질문 선행 배치(L1)와 연속 틱 지연; 규칙 기준군·결정 유지 규칙·안정성 지표; 로봇 데이터 60% | Jev 핵심 컨셉의 재현 가능성과 실시간 적합성 검토 |
| 2026-09-18 OM-1 검토 → 08 | robojev를 L2 "의미에 대한 System 1"으로 정의, 10Hz 스트리밍 typed 출력 계약(질문 세트 v0·조합 규칙·컨트롤러 계약·라벨·레코드), 앞단을 공통 구조화 스키마를 채우는 교체 가능 모듈(주 앞단 3D 재구성)로 | [08](08-streaming-io-and-data-contract.md), 리뷰 [09](09-streaming-contract-review.md) |
| 2026-09-18 리뷰 10 | 전환 틱 초기 경로·실제 명령 구간 대조·국면별 목표점·적용 시각 기준 기하 나이·기하 나이 기준 실행 가능성·관측된 변위 기준 사건·q_instr/q_observe 귀속·관측 자세 이동 | [10](10-current-implementation-review.md): I1~I6 재현, E0 폐루프 원인(시점 배치·자기 가림) |
| 2026-09-19 구현 실측 | 결정 표지 토큰 규칙(03 §3), 도중 지시 조각은 틱 토큰(08 §3.1), 로봇/비로봇 60%는 step 유효 loss 비중(04), 밀기 국면의 기하 나이 예외(08), 전문가 에피소드의 네 출력 필드(08 §8), 버전·지문·DAgger id 규칙 | 4a/4b/5-CPU/3c 리뷰 |
| 2026-09-19 Spark 실측(Task 2b-G0a) | 배포 장비(DGX Spark GB10)에서 backbone 후보의 native BF16 지연 선별: 현재 서식은 전 후보가 80 ms의 ≥7×, 500토큰 틱에서 2B만 사정권(≈83~123 ms) → G0b 후보 2B(주)·4B(5 Hz 대비), 9B·27B 제외(HANDOFF 결정 3 확정); 03·05·06·08·11·README·HANDOFF 갱신 | `artifacts/reports/backbone-screen.json`, `.superpowers/sdd/task-2b-g0a-{report,review-1}.md` |
| 2026-09-19 리뷰 11·cua-s1 검토 | 토큰 실측(틱당 1,764~3,712)과 엣지 배포 전제로 지연·학습 구간 예산 정정(03·05·08); 소형 scorer 기준군(Task 2c)·문맥 섞기 대조군·개념 수준 holdout·혼동 후보 규칙 | [11](11-dgx-spark-handoff-review.md) |
| 2026-09-19 계약 v0.3 (Task v0.3, 결정 1·2) | 서식 v0.3(짧은 이름, 물체 소개/동적 분리, 변화분 틱, 키 기반 후보 줄; 10물체·K=12에서 틱당 p50 392·p95 662·첫 틱 943), 후보 공간(프로파일 없는 결합 키, 영역 쪽 밀기, K≤12 + 지시 조합 예약; E1 퇴화 틱 49.7% → 0%, 완료 20/26 → 23/26), 비키프레임 비용 허용 집합(τ=0.15)과 hold∉A, holdout 봉인(템플릿 변형·개념 계열, ood_dev/ood_test, 누출 QA; 04 §5 표), batch-0 재생성, Spark 2B·4B 재실행 | [08](08-streaming-io-and-data-contract.md), [04 §5](04-data-generation-plan.md), `.superpowers/sdd/decisions-1-2-v03.md`, `task-v03-contract-report.md` |
| 2026-09-19 계약 v0.3 리뷰 1 수정 | 로봇 문구 변형을 origin group 해시로(걸친 group 0), OOD 몫을 생성 비중으로 ≈10~15 %에(비로봇 분야별 10.6~12.5 %, 로봇 400편 14.5 %), 변화분 줄이 기본값 복귀(`vis=1`·`yaw=0`)를 적음, `<id> gone` 줄, soft token 슬롯 거절, 후보 줄의 `path=ok`·중복 `clr` 생략(10물체·K12 틱당 p50 332·p95 599·첫 틱 880), Spark 판정에 `passes_10hz_window`와 `deadline_fail`·외삽 표기, E1 sweep의 동일 seed 대조, 놓기 국면의 운반 높이 낙하 결함(seed 15·19) 이월 | [08](08-streaming-io-and-data-contract.md), [04 §5](04-data-generation-plan.md), [03](03-model-and-training-design.md), `.superpowers/sdd/task-v03-contract-review-1.md`, `task-v03-fix-round-1-report.md` |
| 2026-09-19 D1-prep (놓기 규칙, 대조 쌍) | 놓기점을 영역 안의 빈 자리로(관측된 바닥 높이, 명령의 `place_mm`, hold·retreat 틱에는 open 없음; h0.5·e0.4·c0.6 — E1 sweep 23/26 → 26/26, 놓기 높이 위 이벤트 0), batch-0 재생성; 대조 sibling·삭제 검사(04 §3·§6) | [08 §4](08-streaming-io-and-data-contract.md), [04 §3](04-data-generation-plan.md), `.superpowers/sdd/task-d1-prep-report.md` |

가설·잠정 목표·측정 결과를 구분한다. 하네스가 구성할 내용과 모델이 학습할 내용을 분리하고 각각의 버전을 기록한다.
