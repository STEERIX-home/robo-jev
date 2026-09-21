# 첫 학습 파이프라인 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 검증 가능한 질문 묶음 데이터를 만들고, 직접 판단 모델의 가중치를 실제로 업데이트·저장·재개·평가하는 첫 파이프라인을 완성한다.

**Architecture:** 데이터 제작과 하네스는 상태·질문·후보를 구성한다. 로봇 하네스의 주 결정은 결합 행동 후보의 단일 선택이다. 판단 주기의 지연 예산 안에서 선정한 backbone(첫 후보 Qwen3.8-27B)이 공통 상태와 질문 분기를 읽고 결정 위치의 pointer readout으로 후보 점수를 반환한다. Full attention과 DeltaNet 양쪽에서 상태 격리·공유 gradient를 검증하며 학습·평가·launcher는 같은 데이터 계약을 사용한다.

**Tech Stack:** Python, PyTorch/FSDP2/FlexAttention, Transformers, MuJoCo/robosuite, JSONL/Parquet, Docker, pytest. 클라우드 제품은 실행 환경 adapter로만 연결한다.

**Spec:** [모델 설계](03-model-and-training-design.md), [데이터 설계](04-data-generation-plan.md), [실험·GPU 계획](05-experiment-and-cloud-plan.md), [스트리밍 입출력·데이터 계약](08-streaming-io-and-data-contract.md)(로봇 스트림의 정본).

작성일: 2026-09-18. 같은 날 근본 검토를 반영해 Task 2b(backbone 선정 게이트), 결합 행동 후보 하네스, 규칙 기반 기준군, 결정 유지 규칙, pointer readout과 L0/L1 배치를 추가했고, 이어 [스트리밍 계약](08-streaming-io-and-data-contract.md)에 따라 컨트롤러·스크립트 전문가·스트림 하네스·스트림 상태·TBPTT·DAgger 사이클과 D1 규모를 반영했다. 아래 경로·명령은 구현할 계약이다. 어느 Task가 어느 단계(구현·단위 검증·폐루프 인수·학습 실측)까지 통과했는지는 [뿌리 README의 Status 표](../README.md#status)가 정본이며, 이 문서의 Task 목록은 완료 표시를 대신하지 않는다.

## Global Constraints

- 하네스가 질문·타입·후보를 구성하고 모델은 판단만 한다.
- 초기 지원 타입은 `choice`, `boolean`, `ordinal`이다. 의미와 후보 목록은 요청마다 바뀐다.
- 초기 프로파일은 `Q=1~16`, 질문별 후보 `K=2~32`, 공통 상태 최대 2,048 tokens, 요청 전체 최대 8,192 tokens로 시작한다. 로봇 스트림의 결합 후보는 계약 v0.3부터 `K≤12`(9 + 예약 3, 08 §4)다.
- `N=1, Q>1`의 새 상태 판단과 같은 질문 세트의 10Hz 연속 틱 판단을 핵심 지연 조건으로 둔다. 판정은 모델 시간, 관측→명령 적용 시간, 100ms deadline 초과율이다.
- 모델 규모·초기 가중치·정보·데이터 예산·하네스·실행기를 맞춰 비교한다. backbone 규모 자체는 판단 주기의 지연 예산에서 역산한 상한 안에서 후보를 측정해 정한다.
- 선정한 backbone(첫 후보 Qwen3.8-27B)의 text 가중치 전체와 readout을 학습한다. vision encoder는 고정하고 시각 입력 학습은 별도 실험으로 둔다.
- readout은 질문마다 결정 위치 하나를 두는 pointer 방식이다. 후보별 분기 readout은 추가 연산을 쓰는 비교군으로만 학습한다.
- 로봇 하네스는 [스트리밍 계약](08-streaming-io-and-data-contract.md)을 따른다. 에피소드는 append-only 스트림(L1-a)이고 틱마다 고정 10질문(결합 행동 후보의 주 결정, 게이팅 4, 정지, 그리퍼 상태, 경로·속도·힘)을 병렬로 묻는다. 부가 질문은 틱 시작 시 commitment 기준이다. 분해형 질문 구성은 비교군이다.
- 결정 위치는 틱 끝 공통 상태에서 갈라지는 일시적 분기이고, 다음 틱은 분기 이전 공통 상태에서 이어간다. 실행 이력 입력은 어떤 데이터에서도 라벨로 대체하지 않는다.
- 스트림 학습은 에피소드 순서의 truncated BPTT(10초 구간, 같은 optimizer step 안에서 상태 전달)로 한다.
- 컨트롤러 계약(관측 deadline 200ms, 발행 lease 300ms, 혼합 100ms, 정지 전이, 반사, 그리퍼 이벤트 ACK)은 모든 비교군에 동일하다.
- 폐루프 비교에는 규칙 기반 판단기 기준군을 포함하고, 결정 안정성 지표를 성공률과 함께 보고한다.
- Attention mask뿐 아니라 DeltaNet recurrent state·causal convolution history도 분기별로 격리한다.
- 입력에 정답·미래 결과·비관측 counterfactual 정보를 넣지 않는다. 하네스 파생 값은 관측된 자세로만 계산한다.
- 데이터 분할은 origin group 기준으로 증강보다 먼저 수행하고, 로봇 장면 계열은 에피소드 생성 전에 배정한다.
- 비용은 [GPU 계획](05-experiment-and-cloud-plan.md)의 단계별 예산과 실제 견적으로 관리한다.

## 1. 구현 단위와 파일 경계

프로젝트 루트 기준 예정 경로다. 데이터·checkpoint·cache는 Git에 넣지 않고 manifest와 작은 검증 fixture만 추적한다.

| 예정 파일/디렉터리 | 책임 |
| --- | --- |
| `pyproject.toml`, `uv.lock` | 패키지·개발 도구·검증된 버전 고정 |
| `src/robo_jev/contracts.py` | 요청·라벨 유효성, 모델 입력과 근거 분리 |
| `src/robo_jev/data/generate.py` | seed 기반 합성 상태·질문·정답 생성 |
| `src/robo_jev/data/split.py` | origin group·의미 holdout 분할 |
| `src/robo_jev/data/validate.py` | 데이터 QA·집계·manifest |
| `src/robo_jev/data/episode.py` | 에피소드 스트림 레코드(prefix·틱·model_output·adopted·ack·labels)와 집계 |
| `src/robo_jev/perception/pointworld.py` | 3D 재구성 결과 → 공통 구조화 상태 추출(추적 id, 자세·정밀도, OBB·파지면, 자유 공간, 소스별 나이), 버전 고정 |
| `src/robo_jev/perception/sim_camera.py` | 시뮬 3D 카메라: 깊이 렌더와 잡음·결손 모델, 재구성 파이프라인 입력 |
| `src/robo_jev/sim/environment.py` | reset/step/snapshot/restore·외란·지시 변경 일정 |
| `src/robo_jev/sim/expert.py` | 국면 기반 스크립트 전문가와 국소 경유점 플래너 |
| `src/robo_jev/sim/label.py` | 고정 정책의 counterfactual 사건 결과(키프레임, paired seed) |
| `src/robo_jev/sim/controller.py` | 컨트롤러 계약: 명령 수명·혼합·정지 전이·반사·그리퍼 이벤트 ACK·OSC 어댑터 |
| `src/robo_jev/harness/robot.py` | 스트림 요청(10질문·결합 후보·경유점·실행 이력) 구성, 조합 규칙 v0(유효성·정지·게이팅·결정 유지·부가 답), 명령 생성 |
| `src/robo_jev/harness/rule_judge.py` | 모델 자리에 들어가는 규칙 기반 판단기 기준군(10답) |
| `src/robo_jev/model/serialize.py` | 알려진 입력의 토큰·구간·position 생성, 상태 선행(L0)·스트림(L1-a) 배치, 결정 분기 인덱스와 prefix 경계 |
| `src/robo_jev/model/stream.py` | 스트림 상태(recurrent/conv·윈도우 KV)의 유지·분기·복원과 증분/전체 계산 정합성 |
| `src/robo_jev/model/attention.py` | 기준 mask와 최적화 mask |
| `src/robo_jev/model/hybrid.py` | DeltaNet·conv 상태의 미분 가능한 분기와 backbone adapter |
| `src/robo_jev/model/judge.py` | backbone·결정 위치 pointer readout·후보별 분기 참고군·타입별 반환 |
| `src/robo_jev/train.py`, `src/robo_jev/loss.py` | 실제 가중치 업데이트·loss 정규화 |
| `src/robo_jev/checkpoint.py` | atomic 저장·재개 상태·export |
| `src/robo_jev/evaluate.py`, `src/robo_jev/profile.py` | 질문 품질·일반화·지연·메모리 |
| `configs/`, `infra/`, `scripts/` | 데이터·모델·실험 설정, container, 실행·종료 |
| `tests/`, `tests/fixtures/` | 계약·누출·mask·gradient·재개·시뮬레이터 재현 검증 |

JSONL을 첫 상호 교환 포맷으로 쓰고, 규모가 커지면 동일 schema의 Parquet shard를 추가한다. model code가 generator나 simulator를 import하지 않게 한다. Matrix 연동은 이 계약 위에 붙인다.

## 2. 첫 구현 순서와 인수 검사

아래 작업은 순서대로 수행한다. 병렬 처리가 가능하더라도 입력 계약을 먼저 고정하고, 각 단위의 인수 검사를 통과한 뒤 연결한다. 소형 random-weight 모델은 계산 검사용이며 연구 성능 비교군이 아니다.

### Task 1: 입력·라벨 계약과 D0 fixture

**Files:** `contracts.py`, `tests/test_contracts.py`, `tests/fixtures/d0.jsonl`, `pyproject.toml`.

**Interfaces:** `validate_record(record: dict) -> None`, `model_input(record: dict) -> dict`. 유효하지 않으면 필드 경로가 포함된 `ValueError`를 낸다. 모델 입력을 반환할 때 원본을 수정하지 않는다.

- [ ] D0 64상태(단일 요청)와 짧은 에피소드 스트림 4개(약 400틱)를 작성하고 3타입·복수 정답·정보 부족·결측 라벨, 결합 후보 주 결정·그리퍼 상태·정지·조건부 부가 라벨, 지시 변경·commitment 대조 쌍을 수동 확인한다.
- [ ] 존재하지 않는 정답 ID, 중복 후보, ordinal value 누락, 확률 합 오류, 빈 후보, `action_ref`·국면 참조 불일치, 실행 이력과 라벨의 혼동의 거절 검사를 먼저 작성한다.
- [ ] `request`(스트림에서는 실행 이력·commitment 포함)만 허용 목록으로 반환하고 labels/provenance/evidence/split/model_output/ack를 제외하는 입력 경계를 구현한다. 가려진 참값만 바꿨을 때 입력이 불변인지 검사한다.
- [ ] 아래 검사와 유효성 검사를 통과시키고 D0 해시·검수자를 manifest에 기록한다.

```python
import copy
import json
from pathlib import Path
import pytest
from robo_jev.contracts import model_input, validate_record

def first_record():
    return json.loads(Path("tests/fixtures/d0.jsonl").read_text().splitlines()[0])

def test_future_label_cannot_change_model_input():
    a = first_record()
    b = copy.deepcopy(a)
    b["evidence"] = {"future_success": True}
    b["labels"] = []
    assert model_input(a) == model_input(b)

def test_unknown_answer_is_rejected():
    r = first_record()  # 첫 fixture는 choice + valid_set으로 고정
    r["labels"][0]["candidate_ids"] = ["absent-candidate"]
    with pytest.raises(ValueError, match="candidate_ids"):
        validate_record(r)
```

실행: `python -m pytest tests/test_contracts.py -q`. 통과 산출물은 모델을 호출하지 않고 검증할 수 있는 데이터 계약이다.

### Task 2: 생성기·그룹 분할·QA

**Files:** `data/generate.py`, `data/split.py`, `data/validate.py`, `configs/data/pilot.yaml`, `tests/test_data.py`.

**Interfaces:** `generate_records(count: int, seed: int) -> list[dict]`, `assign_split(origin_group: str) -> str`, `validate_dataset(records: list[dict]) -> dict`.

- [ ] 색·위치·영역 제약, 합성 DOM의 접근 가능한 요소, workflow 선행 조건, 명시 규칙의 우선순위 문제를 각각 생성한다. 기하 문제를 실제 물리 성공 라벨로 표시하지 않는다.
- [ ] 생성된 사실에 따라 대상 선택·명제·수준을 각각 산출하는 독립 규칙을 구현한다. 표현 변형으로 사실이 바뀌면 정답도 다시 계산한다.
- [ ] 의미/분야 holdout을 먼저 제외하고, 나머지 origin group의 stable hash로 `70/10/10/10` 분할을 구현한다. 로봇 장면 계열은 에피소드 생성 전에 배정하고 파생본(틱·rollout·구간·재라벨링)은 이 값을 승계한다.
- [ ] QA에서 후보 참조·분포·라벨 출처·계보 충돌, 정보 경계(가려진 참값의 파생 값 유입), 스트림의 실행 이력/라벨 혼동과 `action_ref`·국면 참조를 검증하고, 요청 수·질문 수·에피소드 수·틱 수·각 split의 원본 group 수를 출력한다.
- [x] 비로봇 500상태 smoke 후 D1(비로봇 2,000상태 + 로봇 400 에피소드는 Task 3에서)로 확대한다. 아래 재현·계보 검사를 먼저 통과시킨다. (D1 2026-09-20: 비로봇 2,000건(`artifacts/datasets/d1/single`, QA 0/0/0) + 로봇 400편(`d1-robot/d1`, 36,690틱, QA 0/0/0) — `.superpowers/sdd/task-d1-report.md` stage B)

```python
from collections import defaultdict
from robo_jev.data.generate import generate_records
from robo_jev.data.validate import validate_dataset

def test_generation_is_reproducible_and_groups_do_not_leak():
    rows = generate_records(count=500, seed=17)
    assert rows == generate_records(count=500, seed=17)
    groups = defaultdict(set)
    for row in rows:
        groups[row["origin_group"]].add(row["split"])
    assert all(len(parts) == 1 for parts in groups.values())
    report = validate_dataset(rows)
    assert report["invalid_records"] == 0
    assert report["states"] == 500
```

예정 CLI:

```bash
python -m robo_jev.data.generate --config configs/data/pilot.yaml --count 2000 --seed 17 --output artifacts/datasets/d1/single
python -m robo_jev.data.validate --dataset artifacts/datasets/d1 --report artifacts/reports/d1-qa.json
```

통과 산출물은 재생성 가능한 D1 초안과 QA 보고서다. [검수 기준](04-data-generation-plan.md)을 통과한 후에만 동결 train 데이터로 승격한다.

### Task 2b: backbone 선정 게이트 (2단계)

**Files:** `scripts/measure_candidates.py`, `scripts/adapt_readout.py`, `configs/model/candidates.yaml`, `artifacts/reports/backbone-screen.json`, `artifacts/reports/backbone-selection.json`.

**Interfaces:** `measure_latency(model_id: str, requests: list[dict], layout: str, warm_prefix: bool, path: str) -> dict`는 틱당 p50/p95/p99, CUDA event 모델 시간, 관측→명령 적용 시간, 100ms deadline 초과율을 반환한다. `path`는 `native`(공식 forward) 또는 `stream`(Task 4의 실제 스트림 경로)이다. `zero_shot_label_scores(model_id: str, records: list[dict]) -> dict`는 학습 없이 후보 라벨 토큰 점수를 읽어 D0·D1 dev의 정확도·NLL을 반환한다. `adapt_readout(model_id: str, records: list[dict], steps: int) -> dict`는 readout만 짧게 학습한 뒤 D1 dev 품질과 학습 구간 메모리를 반환한다.

**1단계 · 예비 선별(G0a, 1×GPU).** native forward와 무학습 측정으로 명백히 맞지 않는 후보를 거른다. 통과 후보는 최대 2개다.

- [x] [모델 설계](03-model-and-training-design.md)의 후보 표를 측정 직전에 공식 공개로 다시 확인하고 `candidates.yaml`에 revision·라이선스·층 구성을 고정한다. (2026-09-19: Qwen3.5-2B/4B/9B + 참고 Qwen3.8-27B, 40-hex revision·Apache-2.0·층 구성·센 파라미터 수; `scripts/fetch_backbone.py`가 `artifacts/models/manifest.json`에 파일별 해시를 적고 적재 전에 대조)
- [x] 실제 tokenizer로 틱당 토큰 분포(물체 6/10개, K=12/32, 지시 변경, 3D 요약 필드 포함)를 측정한다. (`scripts/measure_tokens.py`, 08 §3.4)
- [x] 로봇 10질문 스트림과 대표 셀을 D0 스트림·D1 smoke에서 뽑아 후보마다 native forward의 10Hz 연속 틱 지연을 잰다. 스트림 상태 warm과 cold, L1-a 스트림과 무상태 L0 요청을 모두 잰다. (2026-09-19 DGX Spark, D1 대신 합성 하한/상한/지시 변경 + 500토큰 길이 대용; 아래 결과)
- [x] 같은 후보에 대해 D0와 D1 dev의 무학습 라벨 점수 읽기 품질을 잰다. (2026-09-21, Task P1 stage D: **D1**의 고정 평가 집합(`configs/eval/pilot.yaml`)에서 Nimble 방식 라벨 점수를 두 후보에 대해 잰다 — 스트림은 **16틱마다**, 프롬프트는 (틱 × 질문)마다 하나. 그래서 이 두 줄은 적응 run과 **점수를 매긴 모집단이 다르고**(로봇 `ood_dev` `q_main` 56틱 대 844틱) 평가 집합 해시도 다르다(`c772c669c7be` 대 `79d09793eab5`; 리뷰 1 I6). 로봇 dev 전체/`q_main` **2B 0.484 / 0.333 대 4B 0.776 / 0.595**, ood_dev 0.500 / 0.518 대 0.676 / 0.571, test 0.491 / 0.440 대 0.661 / 0.640; 비로봇 dev 0.382 / 0.291 대 0.455 / 0.389, ood_dev 0.430 / 0.392 대 0.518 / 0.523; 로봇 대조 단일 dev 0.403 / 0.594 대 0.605 / 0.688. **4B의 무학습 판단이 모든 분할에서 낫다**(G0b가 batch-0에서 본 것과 같은 순서). **D1만 쟀다** — 상자 글은 "D0와 D1 dev"를 요구하지만 D0는 이 파일럿의 평가 집합에 없다. 우연 수준 통과를 기준으로 쓰지 않는다는 이 Task의 규칙 때문이고 D1이 그 자리를 대신하지만, 상자가 적은 대로 한 것은 아니다(2026-09-21 리뷰 1 M5). `artifacts/reports/p1-{2b,4b}-zero-shot.json`)
- [x] 후보별로 10초 학습 구간(약 55K token)의 활성 메모리와 윈도우 KV(prefix + 30틱)를 산정해 학습 가능 노드와 비용을 기록한다. (config 기반 추정 + 실측 캐시 바이트, 05 §4; 노드·비용은 확정 후보 기준으로 05에서 재산정)
- [x] 탈락 기준: 토큰 하한에서도 native 지연이 예산을 넘음, 학습 메모리가 예산 노드에 맞지 않음, 무학습 품질이 D1 dev에서 다른 후보보다 뚜렷이 낮음. 결과와 근거를 `backbone-screen.json`에 남긴다. (지연·메모리로 판정; 무학습 품질은 2026-09-21 Task P1에서 측정 — 위 항목)

**1단계 결과(2026-09-19, DGX Spark GB10, native BF16, transformers 5.17 + fla·causal-conv1d 커널 활성, sdpa; `artifacts/reports/backbone-screen.json`).** stream warm(prefix + 30틱 캐시 뒤 연속 틱, native 캐시는 윈도우 없이 자람)의 p95 모델 시간: 현재 서식(틱당 ≈1.85K 토큰) 2B 928 / 4B 2,299 / 9B 2,447 / 27B 5,640 ms — 모두 80 ms의 ≥7×, 100 ms 초과율 1.0. 500토큰 길이 대용 틱: p95 123 / 305 / 378 / 886 ms, 윈도우 크기 캐시(prefix + 29틱)로 외삽하면 83 / 197 / 278 ms. 단일 요청(≤315토큰, 캐시 없음) p50 33 / 70 / 122 ms = 절편 26.5 / 51.6 / 93.6 ms + 1K 토큰당 38.6 / 107.7 / 152 ms. cold(무상태 재계산)는 warm의 7~25×. 문자 그대로는 **통과 후보가 없다** → "예산 안의 후보가 없으면" 조항대로 토큰 예산(계약 v0.3)이 먼저이고, 2단계 후보(≤2)는 v0.3 길이에서 예산에 가장 가까운 **Qwen3.5-2B(주, 10~15% 개선이면 10 Hz)와 Qwen3.5-4B(5 Hz 대비)**로 한다. 9B는 10 Hz 트랙 제외(거리 3.5~4.7×; FP8은 2단계의 profiler·CUDA graph 귀속 측정 뒤), 27B 제외. 2단계의 지렛대 순서: 마스크 없는 윈도우 attention 커널(sdpa+마스크가 윈도우 틱의 ≈40%, FLOP 시간의 ≈8×) → 정적 윈도우 KV 선할당(틱마다 `torch.cat`이 새 segment를 만들어 reserved가 allocated의 2~4×) → CUDA graph/`torch.compile`(절편).

**2단계 · 최종 선정(G0b, 실제 경로).** 1단계를 통과한 후보(≤2)에 대해 Task 4의 최소 실제 경로(P0 pointer readout + `StreamState` 분기·상태 전달 + 짧은 TBPTT 구간)를 구현하고 readout-only 적응을 거친 뒤 잰다. native 측정은 실제 kernel·cache 복제·메모리 이동·readout 비용을 포함하지 않으므로 최종 속도나 학습 후 품질의 근거로 쓰지 않는다.

- [x] 후보별로 `stream` 경로의 10Hz 연속 틱 p50/p95/p99와 100ms deadline 초과율을 잰다(윈도우 KV·분기 포함). (2026-09-20, DGX Spark: v0.3 lower/upper/지시 변경 + batch-0 40편, `--levers baseline,fused,graphs,compile,readout_bf16,all`; `artifacts/reports/backbone-stream.json`·`backbone-stream-levers.json` — 아래 결과)
- [x] D0·D1 smoke로 readout-only 적응(T0급, 수백 step)을 수행하고 별도 D1 dev(의미 holdout 포함)의 품질을 잰다. D0의 우연 수준 통과는 기준으로 쓰지 않는다. (`scripts/adapt_readout.py --mode t0|lora|zero-shot`; T0 2B 120 step·4B 50 step, 짧은 LoRA, 무학습 라벨 점수; 평가 = batch-0 dev 5편·ood_dev 1편(작다)·pilot dev/ood_dev·D0 dev, 위치 편향(치환 답 변경률)·문맥 섞기·규칙 기준군 열 포함; `adapt-{2b,4b}-{t0,lora}.json`, `zero-shot-{2b,4b}.json`)
- [x] 10초 구간 학습의 실제 peak 메모리를 재고, 후보별 D1·D2 학습 시간을 [GPU 계획](05-experiment-and-cloud-plan.md)의 시간 풀과 대조한다. (`--mode chunk-memory`, 05 §4 표: 2B readout-only 6.2 GiB / full+층 checkpointing 47.3 GiB, 4B 14.4 / 5초 58.2 GiB·10초 OOM; 처리량과 시간은 05 §6)
- [x] 지연·품질·학습 비용을 함께 놓고 본 실험 backbone을 확정한다. 탈락 후보의 측정값과 근거를 `backbone-selection.json`에 남기고, 확정 후보 기준으로 용량·비용 표를 다시 계산한다. (`scripts/select_backbone.py` → `artifacts/reports/backbone-selection.json`; 03 §"지연 예산", 05 §4·§6 갱신)

**2단계 결과(2026-09-20, DGX Spark GB10, BF16, `scripts/measure_candidates.py --path stream`; 보고서 `.superpowers/sdd/task-g0b-report.md`).** 실제 경로(정적 prefix KV + 30틱 윈도우 버퍼, DeltaNet 상태 명시 전달, 마스크 없는 varlen flash attention, 10개 결정 표지의 1토큰 배치 forward, pointer readout)의 틱당 모델 p95: **2B** baseline lower 74.7 · upper **83.8** · 지시 변경 82.3 · batch-0(40편 1,348틱) **80.0** ms → upper에서 3.8 ms 미달, batch-0는 0.0 ms(초과율 0.029 / 0.014); 지렛대 **`fused`**(틱 몸통과 분기를 한 forward로, 가중치 한 번 읽기; 40편 재실측 `backbone-stream-fused.json`) upper **62.7** / batch-0 **60.2** ms(p99 91.1 / 84.7, max 91.1 / 101.0), 100 ms 초과 2/1,348틱 = 0.15 %(둘 다 E1 60번째 틱의 ≈1,000토큰 소개 틱) → **10 Hz 통과(여유 17 / 20 ms)**; `all`(+ dense `torch.compile` + bf16 readout, 기본 서빙 구성) 52.5 / 52.0 ms·초과 0(여유 27 / 28 ms). 앞 8편(E0뿐) 부분집합의 지렛대 선별값 65.0 / 57.3은 E1 소개 틱이 없어 판정에 쓰지 않는다. **4B** baseline 182.3 / 175.7 ms(10 Hz·5 Hz 모두 미달), fused 144.6 / 122.6 ms → 5 Hz만(여유 5 / 27 ms), all 129.9 / 110.5 ms. 분기 forward의 CUDA graph는 −2 ms뿐(launch-bound가 아님), 귀속 측정(단일 요청 186토큰)에서 절편은 launch 빈틈 5~13 %가 아니라 weight-read(하한이 graph 시간의 59~70 %) → FP8-9B 닫음. 메모리: 2B 가중치 3.51 + peak 4.14 GiB(틱 사이 증가 0), 4B 7.83 / 9.45 GiB. **품질**(정확도 %; 괄호는 문맥 섞기 대조군 / 규칙 기준군): readout-only T0 2B(120 step) batch-0 dev **88.3**(87.4 / 96.2)·ood_dev 83.6(83.6 / 96.0)·pilot dev **46.5**(41.6)·pilot ood_dev 45.5(39.6); T0 4B(50 step) batch-0 dev 83.8(82.4)·ood_dev 85.0(85.0)·pilot dev 42.4(39.2); 무학습(Nimble 방식, 16틱마다) 2B batch-0 dev 48.0·pilot dev 42.1 / **4B 75.7·51.8** — 4B의 무학습 판단이 뚜렷이 낫고(Nimble의 61 / 66 %와 같은 방향), 로봇 스트림의 대조군은 (G0b 당시) **지시 섞기**(지시·목표 텍스트만 굴리고 물리 상태·후보는 그대로; D1 fix round 1부터 표준 열은 id 재매핑 상태 섞기이며 다음 backbone 평가는 그 열로 읽는다)라 상태에 달린 질문에서 모델과 같은 것이 정상이고, 읽을 것은 q_main이다 — T0 2B dev 64.6 % vs 지시 섞기 63.5 %, **ood_dev 23.4 % = 23.4 %**(규칙 기준군 75.5 %): 적응한 2B는 zoneF holdout 계열에서 목표를 읽지 않는다(LoRA 2B는 74.2 vs 71.9 / 60.6 vs 55.3으로 읽기 시작). 비로봇(상태 섞기 대조군)에서 T0의 이득은 +5 pt에 치환 답 변경률 33 %(위치 편향 큼). **품질은 미결이다**: 무학습 4B(pilot dev/ood_dev 51.8 / 51.3 %)가 pilot에서는 적응한 모든 run(최고 2B T0 46.5 / 45.5)을 앞서고 batch-0 dev에서는 적응한 2B(88.3)가 앞선다 — 지연이 선정을 결정하며, D1 dev(≈40편)에서 같은 step·seed의 비교가 나오기 전에는 4B를 품질 문제에서 물리지 않는다. 짧은 LoRA: 2B LoRA(r=16, 40 step, 59 s/step, peak 26.9 GiB) batch-0 dev 86.5·ood_dev 84.6·pilot dev 45.4·pilot ood_dev 44.3 %(문맥 섞기 대조군 87.1 / 84.6 / 41.2 / 38.3); 4B LoRA(r=16, 30 step, 120 s/step, peak 64.8 GiB) batch-0 dev 75.7·ood_dev 75.2·pilot dev 41.4·pilot ood_dev 42.9 %(문맥 섞기 대조군 76.5 / 75.2 / 35.4 / 35.7) — 수치 전체는 보고서 §S3.1과 `backbone-selection.json`. **확정: `Qwen/Qwen3.5-2B` + `fused` 지렛대**(10 Hz 게이트를 통과하는 유일한 후보; 4B는 어떤 지렛대로도 10 Hz에 못 들고 fused/all로 5 Hz 대비). 4B는 무학습 품질이 더 낫지만 지연 게이트가 결정적이며, 5 Hz로 주기를 낮추는 결정이 있을 때만 후보다.

**파일럿의 like-for-like 결과(2026-09-21, Task P1; 판정 문단은 2026-09-21 리뷰 1 수정 라운드에서 다시 썼다 — `.superpowers/sdd/task-p1-report.md` §D4, `task-p1-review-1.md` C1, `artifacts/reports/p1-*.json`).** G0b가 남긴 "품질 미결"을 D1에서 **같은 데이터·같은 step·같은 seed·같은 고정 평가 집합**(`configs/eval/pilot.yaml`, 적응 run 5개의 `eval_set.sha256 = 79d09793eab5`, 무학습 2개는 16틱마다 재서 `c772c669c7be`; 로봇 dev 8편 635틱·ood_dev 8편 844틱·test 4편 382틱, 대조 쌍 20쌍씩, 비로봇 48건씩)으로 재고, **닫지 못했다**. 판정 칸은 **`robot/ood_dev`의 `q_main`**(844틱 = **8편**)이고, 읽는 방법은 run마다 **자기 자신의** 상태 섞기 대조군과의 차다:

| run (step) | 모델 | 자기 상태 섞기 | **차** | 자기 지시 섞기 | 차 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2B LoRA (40) | 0.570 | 0.402 | **+0.168** | 0.568 | +0.002 |
| 2B T0 (200) | 0.454 | 0.289 | **+0.165** | 0.468 | −0.014 |
| 2B T1 (40, text backbone 전체) | 0.712 | 0.601 | **+0.111** | 0.705 | +0.007 |
| 4B LoRA (40) | 0.642 | 0.544 | **+0.098** | 0.633 | +0.009 |
| **4B T0 (200)** | **0.871** | **0.799** | **+0.072** | 0.858 | +0.013 |
| 규칙 기준군(구조화된 목표를 읽는다) | 0.821 (이 부분집합) / **0.731**(분할 전체) | — | | — | |
| 무학습 2B / 4B (16틱마다, n=56) | 0.518 / 0.571 | — | | — | |

**읽는 법: D1 파일럿 규모에서는 어느 적응 모델도 판정 칸에서 목표를 읽는다는 것을 보이지 못한다.** (1) **원값이 가장 높은 4B T0가 자기 대조군 위 여유는 가장 작다**(+0.072 대 2B LoRA +0.168·2B T0 +0.165) — 원값 차 0.42(0.871 − 0.454)는 각 run을 자기 대조군에 대고 읽으면 **순서가 뒤집힌다**. 4B의 원값이 더 높다는 것은 사실이고, 그것을 **의미 판단**으로 돌릴 근거가 이 측정에는 없다는 것이 정직한 진술이다. (2) 목표와 상태를 통째로 다른 에피소드 것으로 갈아 끼워도 4B T0는 **0.799**를 답한다 — 목표를 읽는 규칙 기준군 0.821과 0.022 차다. 대조군이 약한 것이 아니다(같은 섞기가 2B T0에서 0.165, 2B LoRA에서 0.168을 깎는다). (3) **어느 run도 지시를 읽지 않는다**: 지시·목표 텍스트를 굴린 대조군과의 차가 −0.014 ~ +0.013으로 전부 ±0.014 안이고, 로봇 스트림 15칸 전체에서도 −0.024 ~ +0.073(유일한 예외인 4B LoRA `robot/test`는 2c가 패턴 칸으로 표시한 곳)이다.

**두 가지 전제.** (a) **모집단**: 이 집합은 분할 전체가 아니라 **8편/196건 부분집합**이고, 규칙 기준군·소형 scorer 값은 **분할 전체를 2틱마다** 잰 다른 모집단이다 — 같은 규칙 기준군이 여기서 0.821, `ood_dev` 분할 전체에서 **0.7311**로 **0.090** 차다(4B T0의 여유 +0.072보다 크고, 2B 두 줄의 +0.165·+0.168보다는 작다). 소형 scorer의 0.397/0.375도 분할 전체 값이라 **이 부분집합에서 판정 칸이 여전히 "패턴으로 안 풀리는" 칸인지는 재지 않았다**(4B T0의 대조군 0.799이 그 반대를 시사한다). Task 2c의 0.731과 여기의 0.821을 규칙 기준군이 좋아진 것으로 읽으면 안 된다. (b) **불확실성**: 844틱은 **8편**에서 나왔고 독립 단위는 틱이 아니라 편이다. P1은 이 수를 산출물로 낼 수 없었고(`evaluate_items`가 표를 쓰기 전에 `_predictions`를 버렸다) 두 한계(틱 독립 ±0.023~±0.034, 완전 상관 ±0.23~±0.35)만 적을 수 있었다. **2026-09-21 Task P2에서 실제로 쟀다** — `aggregate`가 질문 칸마다 편 단위 집계를 표에 남기고, 편을 표본 단위로 재표집한 **쌍 부트스트랩**(모델과 자기 대조군을 같은 재표집 안에서 함께 센다)이 여유의 구간을 낸다. 저장된 checkpoint로 판정 칸만 다시 평가해(`configs/eval/pilot-decision-cell.yaml`, 같은 8편 844틱; 모델·대조군 값이 P1의 수를 소수점 셋째 자리까지 재현한다) 나온 값: 2B LoRA +0.168 **[+0.131, +0.225]**, 2B T0 +0.165 **[+0.111, +0.249]**, 2B T1 +0.111 **[+0.066, +0.161]**, 4B LoRA +0.098 **[+0.060, +0.128]**, **4B T0 +0.072 [−0.015, +0.134] — 유일하게 0을 포함한다**. 즉 **원값이 가장 높은 줄의 여유만 잡음과 구분되지 않는다.** 지시 섞기 대비 여유는 **다섯 줄 모두 0을 포함한다**(가장 넓은 것이 2B T0의 [−0.033, +0.018]) — "어느 run도 지시를 읽지 않는다"가 이제 눈대중이 아니라 구간이다. 덧붙여: **원값**의 편 단위 반폭은 0.005~0.214로 뒤쪽 한계에 가깝지만 **여유**의 반폭은 0.025~0.075로 앞쪽 한계의 1~3배다 — 편의 난이도가 차이에서 상쇄되기 때문이고, 그래서 8편으로도 여유는 가려낼 수 있다.

**함께 나온 것.** 비로봇 `choice` dev(48건/`choice` 203문항)의 자기 대조군 위 여유는 4B T0 +0.069 · 4B LoRA +0.059 · 2B T1 +0.044 · 2B LoRA +0.034 · 2B T0 −0.010이고(±0.02 안이라는 말은 `_all` 열의 2B 세 줄에만 참이다), ood_dev `choice`에서는 **2B T1의 +0.085**가 가장 크다. 로봇 대조 `choice` dev/ood_dev: 2B T0 0.490 / 0.443, 4B T0 0.604 / 0.500 — **어느 적응 run도 자기 무학습보다 낫지 않다**(2B 무학습 0.594/0.545, 4B 무학습 0.688/0.670). 한 필드만 바뀐 대조 쌍에서 `q_main` 답이 바뀐 쌍은 20쌍 중 **0~2쌍**, **지시 대조 6쌍에서는 0~1쌍**이다(2B T0가 dev·ood_dev 각각 1/6, 4B LoRA가 ood_dev 1/6, 나머지 8칸은 0/6; `_all`은 LoRA 네 줄이 모두 2/20). 바뀐 지시 쌍에서 두 쪽이 다 맞은 경우는 열 칸 모두 0이다. 후보 순서 치환의 답 변경률(`robot/dev q_main`): 2B T0 0.575 · **4B LoRA 0.546** · 2B LoRA 0.487 · 4B T0 0.238 · 2B T1 0.117.

**fp32 master weight로 다시 잰 T1(2026-09-21, Task P2; `.superpowers/sdd/task-p2-report.md` §C, `artifacts/reports/p2-2b-t1-fp32.json`).** P1의 T1은 갱신이 bf16 격자에 반올림돼 반쪽이었다. 그것을 고치고 **같은 데이터·step·seed·평가 집합**(`79d09793eab5`, 토큰 수까지 같다)으로 40 step 다시 돌렸다. 바뀐 것은 둘이다 — 갱신이 fp32 master를 거치는 것과, 그 메모리 때문에 구간이 10초 → 5초로 내려온 것. **두 번째를 가르려고 같은 5초 구간의 bf16 run을 따로 돌렸다.**

| `robot/ood_dev` `q_main` | P1 bf16 10초 | bf16 5초 (구간 대조) | **fp32 master 5초** |
| --- | ---: | ---: | ---: |
| 모델 | 0.712 | 0.776 | **0.994** |
| 자기 상태 섞기 | 0.601 | 0.692 | **0.904** |
| 여유 (편 단위 95 %) | +0.111 [+0.066, +0.161] | +0.084 [+0.039, +0.119] | +0.090 [+0.020, +0.145] |
| 자기 지시 섞기 | 0.705 | 0.770 | **0.995** |
| 여유 (편 단위 95 %) | +0.007 [−0.007, +0.024] | +0.006 [−0.011, +0.017] | **−0.001 [−0.002, +0.000]** |
| 마지막 손실 | 0.624 | 0.630 | **0.506** |
| peak allocated | 60.00 GiB | 40.63 GiB | 54.65 GiB |

원값이 오른 +0.282 가운데 **구간 몫이 +0.064, fp32 master 몫이 +0.218**이다(손실도 같은 말을 한다 — 구간만 바꾸면 0.624 → 0.630으로 제자리고, master가 0.506으로 내린다). 그런데 **여유는 따라 오르지 않는다**: 상태 섞기 대비 +0.111 → +0.084 → +0.090으로 세 구간이 겹치고, 지시 섞기 대비는 +0.007 → +0.006 → **−0.001**로 셋 다 0을 포함하며 fp32 줄의 구간이 이 과제에서 가장 좁다. 목표·물체·영역·장면을 통째로 갈아 끼운 대조군이 **0.904**로, 목표를 읽는 규칙 기준군(0.821)보다 0.083 높다.

**읽는 법: fp32 master는 학습을 바꾸고 원값을 크게 올리지만 판정을 바꾸지 않는다.** 제대로 학습한 줄이 지시 섞기 대조군과 **가장 정확히 일치하는** 줄이다. 상태를 지워도 0.904로 풀리는 칸은 답이 후보 목록과 자기 실행 이력에 실려 있다는 뜻이고, 이것은 optimizer가 아니라 **과제 설계**를 가리킨다 — 다음 후보 (c)의 근거다. 함께: 위치 편향(`robot/dev q_main` 답 변경률) 0.117 → 0.014, 선택적 정확도 0.937 → 0.993(잘못된 대상 0.000), 놓기 국면 284틱에서 grasp 282 / place 0.

**결정.** **backbone 결정은 그대로다**(지연이 선정한다: 4B는 어떤 지렛대로도 10 Hz에 들지 못한다). 그리고 **이 파일럿은 5 Hz로 갈 품질 근거를 주지 않는다** — 4B의 원값 우위가 자기 대조군을 넘어서지 못하므로 주기를 반으로 줄이는 대가와 저울에 올릴 의미 판단의 이득이 측정되지 않았다. G0b 리뷰 2 **M-d 전제도 그대로**다: 4B의 `fused`/`all` 판정은 아직 앞 8편(E0뿐) 선별값이라 batch-0 40편 재측정 전에는 5 Hz 결정을 하지 않는다. 품질 질문을 실제로 닫으려면 이 파일럿에 없는 측정이 필요하다 — 편 수를 늘리고, 편 단위 구간을 견디는 대조군 위 여유를 내고, 지시 섞기 대조군을 이기는 run이 하나는 나와야 한다. 덧붙여 BF16 master weight 때문에 T1은 `backbone_lr`이 국소 bf16 눈금의 절반보다 작으면 갱신 **크기**가 반올림으로 사라진다: 40 step·1e-5에서 표본 24개의 4.83 %가 움직였지만 그 표본의 79.5 %가 `embed_tokens`(0.068 %만 이동)라, **embedding을 빼면 23.32 %가 움직였고** |w|가 가장 큰 `linear_attn.norm.weight` 네 개(중앙값 0.79~0.97)는 66/74/84/92 % 움직였다(P1 C2·D, 리뷰 1 I3). 실제로 좁혀진 것은 **걸음 크기**다 — 움직인 원소의 평균 |Δ|가 계획대로면 4.0e-4일 자리에서 2.95e-5~3.49e-5(7~9 %)이고 5e-5에서는 3 step 만에 발산한다. fp32 master weight는 **증거 있는 미결 항목**이지 증명된 판정이 아니다(fp32로 돌린 T1이 없다).

실행: `uv run python scripts/measure_candidates.py --config configs/model/candidates.yaml --path native --ticks 70 --report artifacts/reports/backbone-screen.json`(입력은 D0 스트림·`measure_tokens`의 합성 장면·D0 단일 요청; 계약 v0.3 뒤 `--profiles lower,v03_target`로 재실행), 이어서 통과 후보에 `--path stream --candidates Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B --profiles lower,upper,instruction_change --episodes artifacts/datasets/d1-robot/batch-0 --ticks 70 --levers baseline,fused,…`과 `uv run python scripts/adapt_readout.py --model Qwen/Qwen3.5-2B --mode t0|lora|zero-shot|chunk-memory --out …`, `scripts/attribution.py`, `scripts/select_backbone.py`(모든 GPU 진입점은 `robo_jev.gpu`의 통합 메모리 울타리 0.6 안에서 돈다 — 05 §4). 예산 안의 후보가 없으면 토큰 예산·변화분 틱·서빙 GPU 수·판단 주기를 조정한 뒤 다시 잰다. **2단계를 통과하기 전에는 8 GPU 학습 예산(G1 이후)을 집행하지 않는다.** 1단계는 1주차, 2단계는 Task 4의 최소 경로가 나오는 2~3주차에 수행한다.

### Task 2c: 소형 scorer 기준군과 대조 지표 (CPU, 하루)

**Files:** `baselines/tiny_scorer.py`, `evaluate.py`(문맥 섞기 대조군·선택적 지표), `configs/baselines/tiny_scorer.yaml`, `tests/test_tiny_scorer.py`.

**Interfaces:** `train_tiny_scorer(config: dict) -> dict`는 D1(비로봇 단일 요청 + 로봇 틱)을 문맥/후보 byte 텍스트로 읽어 option-attention scorer를 처음부터 학습하고 분할별 정확도·NLL·ECE와 문맥 섞기 대조군 값을 돌려준다. `selective_metrics(predictions, records) -> dict`는 coverage·abstention·selective accuracy·wrong target·unsafe action rate를 낸다(08 §10).

- [x] jevlike/cua-s1 계열 구조(byte 임베딩 → 작은 Transformer 인코더 → 후보가 문맥에 attention → 공유 dot product → 후보 위 softmax)를 ≈1M 파라미터로 구현한다. 문맥은 직렬화한 상태·질문(허용 필드만), 후보는 후보 줄이다. (2026-09-20, Task D1: `robo_jev.baselines.tiny_scorer` 884K 파라미터; 문맥 = `state_lines` / `serialize.full_tick_sections`(틱을 첫 틱처럼 전부, 1,024 byte), 후보 = 직렬화의 후보 줄·계약의 고정 후보; D1 train 10,303 예제·90분 상한에서 1.98 epoch, ≈4.1 s/step)
- [x] D1의 train/dev/test/ood에서 학습·평가하고, 같은 분할에서 규칙 기준군과 나란히 보고한다. **이 기준군이 높은 분할·질문은 의미 판단이 아니라 패턴으로 풀린다**는 표지이며, 그 분할은 backbone의 성과 주장에 쓰지 않는다. (표: `artifacts/reports/tiny-scorer.json`, `.superpowers/sdd/task-d1-report.md` stage C·Fix round 1. 로봇 대조군은 fix round 1부터 id를 재매핑한 **상태 섞기**(goal·물체·영역·장면을 다음 에피소드의 것으로, 후보·commitment·robot·exec 줄은 유지 — `evaluate.context_shuffle_records(robot="state")`)이고 지시 텍스트 섞기는 표지를 정하지 않는 둘째 열이다(D1 리뷰 1 I1). 로봇 게이트·부가 질문 전부(0.86~1.00 = 상태 섞기 = 지시 섞기)와 dev/test의 `q_main`은 패턴으로 풀린다 — `q_main` dev/test scorer 0.79/0.77 vs **상태 섞기 0.78/0.78**(목표·장면을 다른 에피소드의 것으로 바꿔도 답이 그대로: 후보 줄 + 자기 commitment·robot 줄로 푼다)이고 규칙 기준군 0.74/0.74보다 높다(0.9M byte 모델이 목표를 읽는 규칙을 이긴다) — backbone 주장에서 제외; **ood_dev의 `q_main`은 아니다**(scorer 0.40, 상태 섞기 0.38, 지시 섞기 0.39, 규칙 0.73 — 목표를 읽는 규칙만 유지된다); 비로봇은 0.43~0.45 = 상태 섞기(패턴 없음); 로봇 대조 단일의 boolean은 패턴(0.92), choice는 아니다(0.60~0.71 vs 0.58~0.64))
- [x] 문맥 섞기 대조군을 모든 모델 평가의 표준 열로 넣는다. (`robo_jev.evaluate`: `context_shuffle_records`·`answer_change_rate`·`rule_judge_predictions`·`calibration_error`·`selective_metrics`(08 §10) — G0b 평가와 소형 scorer 표가 같은 열을 쓴다; fix round 1부터 `evaluate_items`의 표준 열도 로봇 스트림에서 id 재매핑 상태 섞기(`context_shuffle_kind = "state"`)이고 지시 텍스트 섞기는 `instruction_shuffle=True`의 둘째 열이다 — G0b 리뷰 2가 미룬 항목을 닫았다; G0b 표(아래 2단계 결과)의 로봇 대조군 값은 텍스트 대조군이다)

실행: `python -m pytest tests/test_tiny_scorer.py -q`. 통과 산출물은 분할별 (규칙, 소형 scorer, 문맥 섞기) 표다. 인수 기준은 성능이 아니라 **표가 있다는 것**이다 — 이 값이 backbone 실험의 해석 기준이 된다.

### Task 3: 시뮬레이션·컨트롤러·스크립트 전문가·로봇 스트림 하네스

**Files:** `sim/environment.py`, `sim/expert.py`, `sim/controller.py`, `sim/label.py`, `harness/robot.py`, `harness/rule_judge.py`, `data/episode.py`, `configs/sim/tidy_clutter.yaml`, `configs/harness/robot.yaml`, `configs/controller/osc_v0.yaml`, `tests/test_sim_replay.py`, `tests/test_controller.py`, `tests/test_harness.py`.

**Interfaces:** `Environment.reset(seed: int) -> dict`, `step(command: dict) -> dict`, `snapshot() -> bytes`, `restore(snapshot: bytes) -> None`. `Controller.apply(command: dict, now_ms: int) -> dict`는 수명 검사·혼합·정지 전이·반사·그리퍼 이벤트를 처리하고 ACK를 반환한다. `Expert.act(observation: dict, commitment: dict | None) -> dict`는 10질문의 답과 국면을 반환한다. `build_request(observation: dict, exec_history: dict, commitment: dict | None) -> dict`는 10질문 스트림 요청(결합 후보·경유점·실행 이력)을 만든다. `compose(request: dict, results: dict, commitment: dict | None, now_ms: int) -> dict`는 [조합 규칙 v0](08-streaming-io-and-data-contract.md)를 적용해 명령·채택·전환 기록을 반환한다. `rule_judge(request: dict) -> dict`는 모델과 같은 결과 형식을 반환한다. `rollout_event(snapshot: bytes, action: dict, event: dict, seed: int) -> dict`는 success/failure/censored와 evidence를 반환한다.

- [ ] E0의 reset·목표·성공 판정과 전체 snapshot 저장/복원을 만든다. E1의 다물체 장면(6~10개, 취약·금지 속성), 지시 변경 일정(5~15초), 외란 일정을 seed·모의 시간으로 정의한다.
- [ ] `Controller`에 명령 수명(관측 deadline 200ms, 발행 lease 300ms, 순번 역행 폐기), 100ms 혼합과 전환 구간 충돌 검사, 정지 전이, 반사, 그리퍼 이벤트(readiness·ID·ACK·멱등)를 구현한다. 늦은 응답에 lease가 새로 붙지 않고, `stop`이 혼합을 우회하며, 같은 그리퍼 상태가 반복돼도 이벤트가 한 번만 나는 검사를 둔다.
- [ ] `Expert`를 국면(접근·파지·들기·이동·놓기·밀기)과 국소 경유점 플래너, 가림·차단 시 관측·보류 규칙으로 구현하고 10Hz로 에피소드를 실행해 스트림 레코드를 기록한다.
- [ ] 하네스가 IK·충돌 여유·거리 등 기하 값을 관측된 자세로 계산해 결합 행동 후보(최대 32개)와 관측·보류·재계획 후보, 경유점 후보(≤3)를 만들고 10질문을 붙인다. 정답을 알고 후보를 줄이거나 정렬하지 않는 검사와 후보 포함률 집계를 추가한다.
- [ ] `perception/pointworld.py`의 추출 인터페이스(`extract(recon: Reconstruction, robot: dict, now_ms: int) -> dict`)를 공통 스키마로 고정하고, D1에서는 참값 어댑터가 같은 스키마를 채우게 한다. E2용 시뮬 3D 카메라와 재구성 결함 모델은 D1 이후에 붙이되 인터페이스는 지금 고정한다.
- [ ] `compose`에 유효성 검사, 반사·정지 우선, 게이팅, 결정 유지 규칙(`δ=0.15`, `m=2`, 같은 도전자, 목적지 포함 의미 키, 해제 조건), commitment 기준 부가 답 적용과 전환 틱 폐기를 구현하고 전환·해제·차단·폐기 횟수를 기록한다. 현재 후보보다 `δ` 미만으로 높은 후보는 `m` 틱 전에 선택되지 않고, 도전자가 바뀌면 카운터가 초기화되며, 게이팅 발동 시 즉시 전환되는 검사를 둔다. 분해형 비교 구성에서는 다른 대상의 접근 답을 조합하지 않는 검사를 유지한다.
- [ ] `rule_judge`를 구조화된 목표·제약의 적합성 필터, 고정 가중 기하 비용, 규칙 임계값으로 구현해 10답을 내고 버전·가중치를 config에 고정한다. 모델과 같은 요청을 받아 같은 형식으로 답하는지 검사한다.
- [ ] event의 action·후속 정책·horizon·성공 기준·외란 분포를 config로 고정한다. 키프레임 선택(에피소드당 5틱)과 paired seed 실행을 구현한다.
- [ ] 같은 snapshot/seed에서 trajectory가 허용 오차 안에 재현되는지 확인한다. 사건 구간을 완주한 목표 미달은 실패, simulator 오류와 인프라 timeout은 censor 사유로 구분한다.
- [x] 128,000개 D1 rollout 계획 중 먼저 100개를 실행해 속도·저장량을 재고 나머지 제작 시간과 CPU 비용을 산정한다. 후보별 결과에서 [08 7절의 규칙](08-streaming-io-and-data-contract.md)(의미 적합성 → 성과 → commitment, `unknown` 처리)으로 주 결정 라벨을 산출한다. (D1 2026-09-20: 2,000 키프레임 × 후보 4~8 × seed 8 = 100,640 rollout(≈14 CPU-h, 에피소드 묶음 단위 재개), 라벨 2,000 → 계보 버전 `d1-rollout-labels`; 들고 있는 틱의 후보 4·6은 grasp/place 쌍둥이 키(08 §7) — stage B)
- [ ] 학습 버전이 나온 뒤의 DAgger 사이클(200 에피소드 실행, 실행 이력 보존, 전문가 답은 labels에만)을 스크립트로 만든다.

```python
import numpy as np
from robo_jev.sim.environment import Environment

def test_snapshot_restores_controller_and_rng():
    env = Environment(config_path="configs/sim/tidy_clutter.yaml")
    env.reset(seed=17)
    state = env.snapshot()
    command = {"kind": "HOLD", "duration_ms": 100}
    a = env.step(command)
    env.restore(state)
    b = env.step(command)
    np.testing.assert_allclose(a["qpos"], b["qpos"], atol=1e-8)
    assert a["sim_time_ms"] == b["sim_time_ms"]
    assert a["disturbance_log"] == b["disturbance_log"]
```

실행: `python -m pytest tests/test_sim_replay.py tests/test_controller.py tests/test_harness.py -q`. GPU가 필요한 렌더 검사는 별도 표시한다. 통과 산출물은 D1 로봇 400 에피소드의 스트림 레코드와 컨트롤러 계약 검사이며, 키프레임 rollout 라벨은 계보를 유지한 D1 후속 버전에 반영한다.

### Task 4: Hybrid 상태 분기·스트림 상태·readout과 학습 loss

**Files:** `model/serialize.py`, `model/tokenizer.py`, `model/attention.py`, `model/hybrid.py`, `model/stream.py`, `model/judge.py`, `loss.py`, `configs/model/tiny_hybrid.yaml`, `scripts/fetch_tokenizer.py`, `scripts/measure_tokens.py`, `tests/test_serialize.py`, `tests/test_attention.py`, `tests/test_hybrid.py`, `tests/test_stream.py`, `tests/test_judge.py`, `tests/test_loss.py`, `tests/test_measure_tokens.py`.

**Interfaces:** `serialize_request(request: dict, tokenizer, layout: str = "state_first") -> dict`는 알려진 토큰·구간 ID·논리적 분기·local position·후보 경계 index·결정 위치 index·candidate mapping을 반환한다. `layout`은 상태 선행 `state_first`(L0)와 스트림 `stream_l1a`를 지원하며 스트림에서는 prefix 경계와 틱 경계도 반환한다. `StreamState.fork(n: int) -> list[StreamState]`는 결정 분기용 일시 상태를 만들고, `StreamState.advance(tokens) -> StreamState`는 분기 이전 공통 상태에 다음 틱 토큰을 이어 붙이며, 윈도우(정적 prefix + 30틱) 밖의 KV를 내보낸다. `build_reference_mask(layout: dict) -> Tensor`는 full-attention 층의 query×key 허용 행렬이다. `fork_delta_state(state: dict, branches: int) -> list[dict]`는 한 층의 `recurrent`·`conv` tensor를 분기하되 gradient 연결을 유지한다. `Judge.forward(batch: dict) -> dict`는 질문별 logits를 반환하며, `judgment_loss(outputs: dict, labels: dict) -> Tensor`는 상태 평균 loss를 반환한다.

- [ ] [정보 접근 규칙](03-model-and-training-design.md)에 따라 각 질문의 `S+T_i`를 독립 causal 경로로 실행하는 P0를 만든다. `T_i`에는 전체 후보 목록, 후보 경계, 결정 위치가 들어간다. 같은 backbone의 native 결과를 정답 기준으로 보존한다.
- [ ] Full-attention의 명시적 mask와 DeltaNet·conv 상태의 미분 가능한 분기를 각각 구현한다. state 공유 경로가 in-place 갱신되는 kernel은 독립된 상태 버퍼로 격리한다.
- [ ] 공유 pointer readout `z_ik = (U h_{d_i})ᵀ (V h_{c_ik}) / √r + b`와 choice/boolean/ordinal의 결과 변환을 구현한다. 후보별 분기 readout(참고군 R)은 같은 직렬화 계약의 선택지로 두되 주 경로의 인수 검사를 먼저 통과시킨다. 실행 command 생성은 넣지 않는다.
- [ ] 스트림 배치(L1-a)를 구현한다. prefix 캐시, 틱마다 `[상태][실행 이력][동적 후보]` 뒤에 10개 결정 위치를 일시적 분기로 실행(attention mask 차단, recurrent/conv fork), 분기 상태 폐기, 분기 이전 공통 상태에서 다음 틱 이어가기, 윈도우 = 정적 prefix + 30틱과 학습·추론 동일 mask·position. 같은 요청의 L0/L1-a 틱당 새 토큰 수를 집계한다. L1-b는 full-attention 후보에서만 별도 옵션으로 둔다.
- [ ] 스트림 정합성 검사: 증분 계산과 처음부터 계산의 logits가 일치(윈도우 절단 제외), 결정 분기가 서로의 출력을 바꾸지 않음, 질문 하나를 바꿨을 때 다른 분기·다음 틱 상태의 변화량을 품질 지표로 기록.
- [ ] valid-set CE·분포 CE·관측 사건 NLL·결측 mask·상태별 정규화와 `unknown` 후보를 정규화에서 빼는 부분 라벨 손실을 구현한다.
- [ ] 작은 random-weight hybrid fixture와 실제 backbone 양쪽에서 질문 단독/묶음, 복제/공유의 logits·loss·gradient를 비교한다. recurrent 초기 상태·conv history로 돌아가는 gradient를 포함한다.
- [ ] Full-attention은 FlexAttention, DeltaNet은 지원되는 상태 입출력 kernel로 최적화한다. prefix gradient가 지원되지 않으면 미분 가능한 기준 연산에서 시작하고 해당 kernel을 학습 지원으로 표시하지 않는다.

계산을 검증할 최소 mask 예시는 `[S0,S1,T1,C11,C12,D1,T2,C21,D2]`다. `C`는 후보 경계, `D`는 결정 위치다. 각 구간 길이가 짧더라도 다른 질문을 볼 수 없고, 같은 질문 안에서는 결정 위치가 모든 후보를 본다는 조건을 검사한다.

```python
import torch
from robo_jev.model.attention import build_reference_mask

def test_questions_cannot_read_each_other():
    layout = {
        "state": [0] * 9,
        "question": [-1, -1, 1, 1, 1, 1, 2, 2, 2],
        "candidate": [-1, -1, -1, 1, 2, -1, -1, 1, -1],
        "kind": ["state", "state", "question", "candidate", "candidate", "decision",
                 "question", "candidate", "decision"],
        "position": [0, 1, 2, 3, 4, 5, 2, 3, 4],
    }
    expected = torch.tensor([
        [1,0,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0], [1,1,1,1,0,0,0,0,0], [1,1,1,1,1,0,0,0,0], [1,1,1,1,1,1,0,0,0],
        [1,1,0,0,0,0,1,0,0], [1,1,0,0,0,0,1,1,0], [1,1,0,0,0,0,1,1,1],
    ], dtype=torch.bool)
    assert torch.equal(build_reference_mask(layout), expected)
```

위 mask 검사는 full-attention 부분만 다룬다. DeltaNet 상태의 alias와 gradient를 검사하는 최소 예시는 다음과 같다.

```python
import torch
from robo_jev.model.hybrid import fork_delta_state

def test_fork_preserves_prefix_gradient():
    state = {
        "recurrent": torch.ones(2, requires_grad=True),
        "conv": torch.ones(3, requires_grad=True),
    }
    children = fork_delta_state(state, branches=2)
    loss = sum(child["recurrent"].sum() + child["conv"].sum() for child in children)
    loss.backward()
    for tensor in state.values():
        torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 2))

def test_branch_updates_cannot_mutate_siblings():
    state = {"recurrent": torch.ones(2), "conv": torch.ones(3)}
    children = fork_delta_state(state, branches=2)
    children[0]["conv"].zero_()
    children[0]["recurrent"].zero_()
    for key in state:
        torch.testing.assert_close(children[1][key], state[key])
        assert torch.all(state[key] == 1)
```

스트림의 분기·상태 전달을 검사하는 최소 예시는 다음과 같다. 분기가 공통 상태를 바꾸지 않고, 다음 틱이 분기 이전 상태에서 이어져야 한다.

```python
import torch
from robo_jev.model.stream import StreamState

def test_decision_branches_do_not_leak_into_next_tick():
    base = StreamState.from_tokens(prefix_tokens, tick_tokens)
    snapshot = base.clone()
    branches = base.fork(9)
    for branch in branches:
        branch.step(decision_token)          # 일시적 분기, 결과는 버림
    torch.testing.assert_close(base.recurrent, snapshot.recurrent)
    torch.testing.assert_close(base.kv, snapshot.kv)
    next_state = base.advance(next_tick_tokens)
    full = StreamState.from_tokens(prefix_tokens, tick_tokens + next_tick_tokens)
    torch.testing.assert_close(next_state.recurrent, full.recurrent)
```

이 검사들은 시작 조건이다. 실제 DeltaNet 연산과 전체 backbone의 P0/P1 정합성을 대신하지 않는다. 실행은 `python -m pytest tests/test_attention.py tests/test_hybrid.py tests/test_stream.py tests/test_loss.py -q`이며, FP32 기준을 먼저 확인하고 BF16 허용 오차는 실측하여 고정한다.

### Task 5: 실제 학습·저장·중단 후 재개

**Files:** `train.py`, `checkpoint.py`, `sampler.py`, `configs/train/tiny_cpu.yaml`(CPU fixture), `configs/train/qwen38-27b-pilot.yaml`, `infra/train.Dockerfile`, `tests/test_checkpoint.py`, `tests/test_sampler.py`, `tests/test_train.py`, `tests/test_resume.py`.

**Interfaces:** `train(config: dict) -> dict`는 run ID·checkpoint 경로·마지막 step·지표를 반환한다. `save_checkpoint(path: str, state: dict) -> None`, `load_checkpoint(path: str) -> dict`는 model/optimizer/scheduler/RNG/sampler/config/manifest를 다룬다.

- [x] T0 readout-only 학습에서 optimizer step 전후 readout이 바뀌고 frozen backbone은 그대로인지 검사한다. (2026-09-21, Task P1 C1, 실제 2B: backbone tensor **320개 전부**의 sha256을 step 전후로 대조해 **0개 변경**·requires_grad 0개, readout 3/3 이동(U·V 최대 0.0003, bias 1.03e-05), `trainable_state_dict` 저장 키 ['U.weight', 'V.weight', 'bias']; `scripts/p1_acceptance.py --check frozen`, `artifacts/reports/p1-acceptance.json`)
- [x] T1에서 text backbone gradient와 실제 parameter 변경을 확인한다. loss만 감소하고 backbone 업데이트가 빠지는 오류를 차단한다. vision encoder는 고정 상태를 유지한다. (2026-09-21, Task P1 C2, 실제 2B 3 step: gradient가 text backbone **320개 tensor 전부**에 0이 아니게 닿고(최대 30.2) 고정 표본 8개 중 **7개**가 움직였다 — 0인 것은 `layers.12.linear_attn.dt_bias` **하나뿐**이고, 여섯은 정확히 bf16 격자 한 칸(2^-13 = 1.22e-4), 나머지 하나 `layers.21.linear_attn.dt_bias`는 다음 구간의 한 칸(2^-14 = 6.10e-5)을 **원소 하나에서** 움직였다(`l2 == max_abs`). **BF16 master weight의 반올림 손실**이다(2026-09-21 리뷰 1 I2에서 정정 — 전에는 "7개가 움직였다"와 "`dt_bias` 2개는 0"이 한 문장에 같이 있었다). `backbone_lr` 5e-5에서는 손실이 2.63 → 28.93로 발산했으므로 파일럿 T1은 §5 계획값 1e-5를 쓴다. LoRA(진단 조건) 같은 검사: gradient tensor 300개 중 첫 backward에 0이 아닌 것 150개(peft가 `lora_B`를 0으로 초기화하므로 `lora_A`는 첫 step에 0이다), 표본 8/8 이동, 손실 2.63 → 1.76. vision encoder는 이 backbone(Qwen3.5 text-only)에 없다 — `freeze_vision_encoder`는 계약상 true로 두고 대상이 없다는 것을 여기 적는다. **fp32 master weight는 2026-09-21 Task P2에서 붙였다**(`MasterWeightAdamW`, 설정 `fp32_master_weights` 기본 켜짐; readout·LoRA는 이미 fp32라 사본이 생기지 않아 T0·LoRA는 P1과 같은 run이다). 측정 전에 고정한 기준 위의 시험 한 쌍이 그 차이를 붙든다: 상수 gradient 200 step에서 bf16 파라미터가 움직인 거리 / 반올림 없는 기대치 N·lr이 fp32 master에서 **0.9766**(통과), bf16 직접 갱신에서 **0.0000**(떨어진다). 대가는 2B에서 backward 상주량 +14.03 GiB(실측; 예측 +14.02)와 step 시간 +0.2 %이고, 그래서 T1의 구간이 10초 → **5초**로 내려왔다(P2 A2 사다리). **기계 판정은 실패다**: `p1-acceptance.json`의 `checks.trains_t1.passed`·`backbone_moved`가 `false`인데, 그 판정은 "표본 8개가 **전부** 움직였는가"를 묻고 `backbone_lr 5e-5`에서 하나가 0이었기 때문이다. 이 상자를 체크하는 근거는 그 판정이 아니라 **gradient가 320개 tensor 전부에 0이 아니게 닿았고 표본 7/8이 실제로 움직였다**는 것, 그리고 파일럿 T1은 발산하지 않는 1e-5로 돌렸다는 것이다 — 40 step·1e-5에서는 embedding을 뺀 표본의 23.32 %가 움직였다(리뷰 1 M2·I3))
- [x] 스트림 데이터의 truncated BPTT를 구현한다. 에피소드를 10초 구간으로 나눠 같은 optimizer step 안에서 차례로 forward·backward하고 recurrent/conv 상태와 윈도우 KV를 detach해 전달하며, 한 에피소드를 하나의 accumulation 단위로 둔다. 구간 경계의 상태가 추론 시 증분 계산과 일치하는지, 분기 gradient가 공통 상태로 합쳐지는지 검사한다. 단일 요청 데이터와 스트림 데이터를 같은 step에 섞는 sampler(로봇/비로봇 축, 70/20/10 축, 정상 유지 틱 하향 가중)를 두고 실제 유효 loss 비중을 기록한다. (2026-09-21, 리뷰 1 I8에서 근거를 달았다 — 전에는 수·날짜·출처 없이 체크만 되어 있었다. 구간 경계 상태와 분기 gradient는 CPU 검사가 고정한다: `tests/test_train.py::test_chunk_boundary_state_equals_the_incremental_inference_state`(구간 경계 상태 = 증분 추론 상태), `::test_branch_gradient_merges_into_the_carried_state_and_stops_at_the_detach_boundary`(분기 gradient가 이월 상태로 합쳐지고 detach 경계에서 멈춘다), `::test_sum_of_chunk_losses_equals_the_whole_episode_loss`, `::test_episode_chunks_follow_the_simulated_time`. **실제 유효 loss 비중은 D1 규모에서 측정했다**(2B T0 200 step, `artifacts/reports/p1-2b-t0.json`의 `curve[*].loss_by_domain`): 200 step **전부**에 두 분야가 다 들었고, 설정값 `robot_loss_share 0.6`은 **분야별 평균 손실에 걸리는 가중치**라 step 손실에서 로봇이 실제로 차지한 몫은 평균 **0.215**·중앙값 0.182·범위 0.066~0.680이다(4B T0 0.220, 2B T1 0.285) — 비로봇 손실이 훨씬 커서 0.6이 그대로 비중이 되지 않는다. 타입별 평균 손실(200 step): choice 0.758 · ordinal 0.773 · boolean 0.123)
- [ ] 단일 GPU의 readout-only 저장·재개를 검증한 다음, 8 GPU FSDP2의 text backbone full training에서도 재검증한다. 재개 시 진행 중이던 에피소드의 구간 위치와 상태를 복원한다. (**앞쪽 절반 완료** 2026-09-21, Task P1 C3 — 단일 GPU readout-only 저장·재개가 비트 동일; 구간 경계 재개(진행 중 에피소드의 구간 index·상태·누적 gradient)는 CPU에서 `tests/test_resume.py`가 검사한다. **8 GPU FSDP2는 아직** — 이 상자에 GPU가 하나다. **2026-09-21 Task P2에서 게이트를 학습 범위별로 넓혔다**(`--resume-modes t0|lora|t1`; 판정 부분을 `compare_resume`로 떼어 `tests/test_acceptance.py`가 GPU 없이 일곱 가지 어긋남마다 떨어지는 것을 고정한다). **T1(fp32 master)에서 돌린 결과는 기계 판정 `false`이고, 그 이유는 재개가 아니다**: sampler 위치·뽑힌 레코드·optimizer step 수가 정확히 같고 파라미터 최대 차가 5.70e-4(backbone은 6.10e-5 = bf16 한 눈금)인데, **재시작이 전혀 없는 두 프로세스의 같은 3 step이 이미 손실에서 0.064까지 벌어진다**(step 1은 비트 동일 — 비결정성은 1.88B 파라미터의 backward/optimizer에 있다). P1의 허용 오차는 이 GPU가 비트 결정적이던 **T0**에서 고정한 값이라 T1에는 맞지 않는다. **수를 보고 허용 오차를 고치지 않았다** — T1용 허용 오차는 재시작 없는 두 run의 spread를 먼저 재서 그 위로 정하는 것이 이월 항목이고, 상대 L2 기준은 norm이 0에 가까운 tensor(`bias`, 절대 차 4.85e-7인데 비 0.16)에서 분모 때문에 떨어지므로 P1이 이동 표에 넣은 `reference_l2` 식의 보호가 필요하다)
- [x] 아래 시작 설정으로 200 step profile을 수행하고 실제 메모리·처리량·checkpoint 크기를 기록한다. (2026-09-21, Task P1 C4/D: 확정 backbone 2B의 T0 200 step — **9.94 s/step**(p50 9.07, max 24.94), peak allocated **10.08 GiB**, 4664 tokens/s(총 9,267,745), 학습 33.1분 + 고정 평가 집합 13.5분, checkpoint 3.4 MiB(readout만), RSS 7.11 GiB. 아래 YAML은 실제로 돌린 값으로 고쳤다)
- [x] 재개 직후 다음 batch·loss·업데이트를 중단 없는 실행과 비교하고 G0를 판정한다. (2026-09-21, Task P1 C3, 실제 2B T0 20 step: 연속 실행 대 10 step 저장 + **프로세스 재시작** + 10 step에서 step마다의 loss 차 최대 0.0, 학습 대상 tensor 3/3 **비트 동일**, sampler 위치·뽑힌 레코드·optimizer step 수 일치. 허용 오차 {'loss_abs': 0.02, 'loss_rel': 0.02, 'param_max_abs': 0.01, 'param_rel_l2': 0.05}는 비교 **전에** `scripts/p1_acceptance.py`에 고정했다 — 다른 프로세스는 kernel 선택이 달라질 수 있어 비트 동일을 요구하지 않았는데 결과는 비트 동일이었다. **G0 통과**)

**실제로 돌린 설정(2026-09-21, Task P1; 정본은 `configs/train/qwen35-2b-pilot.yaml`).** 아래 값은 예시가 아니라 이 profile을 만든 설정이다 —
옛 예시의 `model_id: Qwen/Qwen3.8-27B`와 `world_size: 8`은 backbone 확정(2B) 전·다중 GPU 계획의 것이었다.

```yaml
model_id: Qwen/Qwen3.5-2B              # Task 2b에서 확정한 backbone (옛 예시는 Qwen3.8-27B)
model_revision_manifest: artifacts/models/manifest.json
dataset_manifests:                     # 로봇 rollout 라벨판 + 비로봇, 한 run에 (splits: [train])
  - {path: artifacts/datasets/d1-robot/d1-rollout-labels/manifest.json, domain: robot, files: ["episodes/*/streams.jsonl"]}
  - {path: artifacts/datasets/d1/single/manifest.json, domain: non_robot}
dtype: bfloat16
readout: decision_pointer
readout_rank: 64
readout_dtype: float32
execution_backend: independent_paths
layout: {single: state_first, stream: stream_l1a}
stream_chunk_seconds: 10               # T0·T1; LoRA는 5
stream_window_ticks: 30
trainable: readout_only                # 이 profile은 T0. T1은 text_backbone_and_readout + backbone_lr 1e-5
freeze_vision_encoder: true            # 이 backbone에는 vision encoder가 없다 (text-only)
optimizer: adamw
backbone_lr: 0.00005                   # LoRA·T1용. T1은 1e-5 (5e-5는 발산 — C2)
readout_lr: 0.0003                     # G0b 실측 (1e-3은 한 step에 발산, 계획값은 1e-4)
weight_decay: 0.01
gradient_clip: 1.0
warmup_ratio: 0.05
microbatch_states_per_rank: 1
gradient_accumulation: 2               # 로봇 스트림 단위 1 + 비로봇 묶음 1 (계획값 4는 8 GPU 목표)
world_size: 1                          # GB10 한 장 (옛 예시는 8)
max_total_tokens: 8192
activation_checkpointing: false        # T0. LoRA·T1은 true
max_steps: 200
seed: 17
```

**실측(2B T0 200 step, DGX Spark GB10).** 9.94 s/step(p50 9.07, max 24.94) · peak allocated 10.08 GiB ·
4664 tokens/s(총 9,267,745) · 학습 33.1분 · 고정 평가 집합 13.5분 ·
checkpoint 3.4 MiB(readout만; T1은 backbone 전체라 GB 단위) · 프로세스 RSS 7.11 GiB ·
비용은 이 상자에 시간 단가가 없어 적지 않는다(Task 6의 launcher가 붙인다).

학습 설정은 `dataset_manifests: [...]`로 로봇 batch와 비로봇 데이터의 manifest를 여러 개 받아 한 run에 넣는다(manifest별 domain 기본값, 레코드의 `provenance.domain`이 우선). step마다 로봇 스트림 단위와 비로봇 묶음 단위를 둘 다 넣고 손실은 04의 0.6/0.4 혼합이다(CPU 검증 완료). `independent_paths`는 질문별 causal 경로를 복제하는 P0다. Q개 경로를 내부 microbatch로 나누어 상태별 loss를 구성하고 실제 처리량을 기록한다. `readout: candidate_branch`는 비교군 R이며 경로가 Q×K개로 늘어난다. 로봇 스트림 레코드는 `layout: stream_l1a`로 읽히며 `stream_chunk_seconds`·`stream_window_ticks`가 truncated BPTT와 윈도우를 정한다. `model_id`는 Task 2b에서 확정한 backbone(`Qwen/Qwen3.5-2B`)이고 위 블록은 예시가 아니라 **실제로 돌린 설정**이다(`configs/train/qwen35-2b-pilot.yaml`가 정본). 이후 `shared_hybrid` P1을 따로 profile한다. 이 경로는 full-attention과 DeltaNet의 정합성 검사를 모두 통과해야 한다. 입력 축소만으로 최대 지원 길이를 통과한 것처럼 표시하지 않는다. profile 후 본 학습용 step·epoch 상한을 다시 산정한다.

예정 CLI:

```bash
torchrun --standalone --nproc_per_node=8 -m robo_jev.train --config configs/train/qwen38-27b-pilot.yaml
```

**클라우드(GPU) 단계의 전제 — CPU 검증에서 드러난 것(2026-09-19).** (1) ~~sampler의 즉시 직렬화가 D1에서 10GB를 넘는다~~ — **실측으로 틀렸다(2026-09-21, Task P1 stage A1)**: 실제 학습 설정(`configs/train/qwen35-2b-pilot.yaml`, D1 rollout 라벨판 로봇 train 253편 + 비로봇 train 1,259건)의 `Trainer` 적재는 **35.2 s · 최대 RSS 3.23 GiB**(item 1,512 = 스트림 253 + 단일 1,259, 23,214틱, 토큰 11,299,218)이고 한 step을 뽑아도 늘지 않으며, 고정 평가 집합(`configs/eval/pilot.yaml`, 7분할 2,037상태 1.04M 토큰)까지 더해 **3.87 GiB**다 (`artifacts/scratch/p1/trainer-load-d1.json`). 추정 '>10 GB'는 400편 전부(× 165K 토큰)를 가정한 것이고 train split은 253편, 실측 토큰은 에피소드당 평균 44.7K다. **지연 직렬화·배열 layout은 하지 않는다** — 121 GB 상자에서 3~4 GiB는 병목이 아니다. D2(4,000편)에서 다시 잰다(선형이면 train 2,530편 ≈ 32 GiB로 그때는 바꿔야 한다). (2) `torch` 핀은 Linux에서 CUDA 13 휠 묶음을 받으므로 클라우드 이미지·드라이버에 맞는 index를 05에 고정한다. (3) `configs/`는 wheel에 포함되지 않으므로 Dockerfile이 저장소째 담는다. (4) step 중간 checkpoint는 누적 gradient(실제 backbone 크기)를 담으므로 선점(preemption) 환경에서만 켠다. (5) 질문 수·후보 수의 프로파일 상한(Q≤16, K≤32)을 `validate_record`가 강제한다(`contracts.PROFILE_LIMITS`, `limits=`로 다른 프로파일을 준다; 2026-09-21 Task P1 stage A2). D1 실측 최댓값은 상한 안이다 — 로봇 스트림 틱 Q=10(세트 v0)·**K≤12**(q_main; 후보 수 분포 3:36,418 · 7:13,726 · 12:17,671 …), 로봇 대조 단일 Q≤10·K≤12, 비로봇 단일 **Q≤16**(`rules-0002-0`)·K≤14 (`artifacts/scratch/p1/profile-caps-d1.json`). 넘는 레코드는 없었고 상한을 넓히지 않았다. (6) 서빙 클라이언트는 응답에 `meta`(observed_at·goal_version·후보 집합 버전·seq)를 반드시 실어야 하네스의 유효성 검사(08 §5.0)가 동작한다.

인수 검사의 핵심 비교는 `동일 seed의 20 step 연속 실행` 대 `10 step 저장 + 프로세스 재시작 + 10 step`이다. loss뿐 아니라 sampler 위치·optimizer step·선택한 parameter tensor의 차이를 보고한다. GPU 정밀도에 맞는 허용 오차를 고정하고 실패하면 긴 run을 시작하지 않는다.

### Task 6: 평가·비용·실행 종료를 포함한 첫 run

**Files:** `evaluate.py`, `profile.py`, `scripts/launch_run.py`, `configs/eval/pilot.yaml`, `infra/run-manifest.schema.json`.

**Interfaces:** `evaluate(config: dict) -> dict`, `profile(config: dict) -> dict`. launcher는 공급자 instance ID, 시간 단가, artifact 위치, 종료 결과를 기록한다. 평가 결과에는 데이터·모델·하네스·하드웨어 버전이 모두 있어야 한다.

- [ ] ID·의미 holdout·후보 순서·질문 묶음 정합성의 개별 지표를 출력한다. label-only와 전체 분포 계약을 구분한다. 관측 잡음 변형에서 주 결정이 뒤집히는 비율을 모델과 규칙 기반 판단기 양쪽에 대해 낸다.
- [ ] model CUDA 시간과 end-to-end wall-clock 시간을 각각 측정한다. 새로운 상태마다 재계산한다. 연속 틱 조건(같은 질문 세트, 스트림 상태 warm, 10Hz)은 개발용 에피소드 재생으로 L1-a 스트림과 무상태 L0 요청 모두 측정하고, 관측→명령 적용 시간, 100ms deadline 초과율, 관측 deadline 폐기율, lease 만료율을 낸다.
- [ ] 개발용 폐루프 100 seed에서 모델, 규칙 기반 판단기, 분해형 구성의 성공률과 결정 안정성 지표(전환율·왕복 전환·유지 시간, 목표 변경 후 반응 지연), 그리퍼 이벤트 누락·중복·시각 오차, `q_stop` 지연·오경보, 컨트롤러 거절률과 전환 구간 충돌을 기하 충분 층과 의미 판단 층으로 나눠 낸다. 규칙 기준군이 두 층 모두 포화하면 과제 조건 재설계를 G1 판정에 올린다.
- [ ] 스트림 격리 지표: 질문 하나를 바꿨을 때 다른 답과 다음 틱 상태의 변화, 지시 변경·commitment 대조 쌍의 정답 변화와 유지 학습 여부, 전환 틱의 부가 답 폐기.
- [ ] 시간·비용 상한, 실패 상태, 저장 성공 여부를 launcher에 연결한다. 종료 API 실패는 미종료 상태로 명시한다.
- [ ] 일부러 학습을 실패시켜 로그·checkpoint의 영속 보존과 GPU 종료 절차를 확인한다.
- [ ] D1과 첫 checkpoint·QA·profile·비용을 하나의 run report로 묶고 G1 확대 여부를 판정한다.

```bash
python -m robo_jev.evaluate --config configs/eval/pilot.yaml --checkpoint artifacts/checkpoints/qwen38-27b-pilot
python -m robo_jev.profile --config configs/eval/pilot.yaml --checkpoint artifacts/checkpoints/qwen38-27b-pilot --fresh-states 1000
python -m robo_jev.profile --config configs/eval/pilot.yaml --checkpoint artifacts/checkpoints/qwen38-27b-pilot --streaming-episodes artifacts/episodes/dev100 --layout stream_l1a --tick-hz 10
```

여기까지의 완료 조건은 “환경 설치”가 아니라 **직접 학습된 가중치, 재현 가능한 데이터, 재개 검증, 질문별 품질·실측 비용을 함께 제시하는 것**이다.

## 3. 이후 연구 사이클과 일정

기존의 12주안은 아래처럼 산출물 기준으로 사용한다. 역할은 모델·데이터·로봇/하네스·학습 인프라/평가의 네 축이며, 반드시 4명을 이미 확보했다는 뜻은 아니다. 인원·GPU 대기에 따라 달력 일정은 바뀌고 통과 기준은 유지한다.

| 기간 | 작업과 연결 | 산출물·통과 기준 |
| --- | --- | --- |
| 1주 | Task 1~2, Task 2b 1단계 예비 선별(G0a), Task 4의 mask·DeltaNet 분기·스트림 상태 검사 | D0(단일 요청 64 + 스트림 4 에피소드)·500상태 smoke, schema·누출·정보 경계·공유 상태의 gradient 확인, 후보 ≤2로 압축한 예비 선별 보고서 |
| 2~3주 | Task 3~6의 최소 실행, Task 4의 최소 실제 스트림 경로, Task 2b 2단계 최종 선정(G0b), G0 | 컨트롤러 계약·스크립트 전문가·스트림 하네스, 실제 optimizer update(TBPTT 포함)·재개·측정 환경, D1 제작(비로봇 2,000 + 로봇 400 에피소드)·검수, 규칙 기반 판단기, **backbone 확정** |
| 3~4주 | T1/G1(확정 backbone), LR·pointer readout 검증, R·L1/L0의 D1 규모 비교, 첫 DAgger 사이클 | 첫 일반화 곡선, 라벨 오류·기반 능력·구조 오류 구분, R 대비 품질과 L1-a/L0 틱당 지연·deadline 초과율 |
| 5~6주 | D2 제작(에피소드 4,000·키프레임 rollout 확대), T2/G1b, P1·B1-S 공유 최적화, E1 | 두 학습 방식(B1, pointer)의 에피소드 400/1,600/4,000 곡선, 각 checkpoint의 공유 없음/있음 실행 비교 |
| 7~8주 | 같은 backbone의 T3/G2 | 최종 규모에서 2 방식 × 3 seeds = 6 checkpoint(시간 풀이 허용하는 범위), 비용·성능 표 |
| 9~10주 | E2, calibration, 폐루프 개발 시험(모델·규칙 기준군·분해형 구성) | 부분 관측·지연 하네스, 결정 안정성 지표, 조건 층별 결과, test 전 판정 규약 동결 |
| 11~12주 | 봉인 ID/OOD·로봇 최종 시험 | 독립 평가·checkpoint·원시 로그·기술 보고서 |

실제 로봇 장비가 준비되면 9~12주 구간에 별도 검증을 추가한다. 장비가 준비되지 않으면 결과를 시뮬레이션 검증으로 한정하고 실제 환경 성공을 완료했다고 쓰지 않는다.

## 4. 각 단계에서 다음 단계로 넘어가는 판단

| 관측된 문제 | 먼저 확인할 것 | 다음 조치 |
| --- | --- | --- |
| D0도 학습하지 못함 | 직렬화·정답 참조·mask·gradient·loss | 대규모 데이터/GPU 확대 전에 구현을 수정 |
| D0는 맞지만 새 의미에 실패 | 라벨·표현 다양성·문맥 부족·backbone 능력 | 오류별 생성 보강, 기반 모델 진단 |
| 묶음에 다른 질문을 넣으면 답이 바뀜 | attention·position·cache key·recurrent/conv 상태 오염 | 정합성 문제를 먼저 해결 |
| 품질은 좋지만 공유가 느림 | mask kernel·padding·compile·메모리 이동 | profiler 근거로 병목 최적화, 속도 목표를 미달로 보고 |
| 오프라인은 좋고 로봇이 실패 | 후보 누락·상태 오차·지연·답 조합·실행기 | 모델과 하네스의 원인을 나눠 고정 조건에서 재검증 |
| 규칙 기반 판단기가 모델과 같은 성공률 | 조건 층 구성, 후보 생성이 답을 결정하는지 | 의미 판단이 필요한 조건을 강화하거나 그 층에서는 모델 기여를 주장하지 않음 |
| 결정이 틱마다 흔들림 | 결정 유지 규칙, 이력·현재 후보 입력, 대조 데이터, 관측 잡음 | 전환율과 반응 지연의 교환을 재설정하고 안정성 대조 데이터를 보강 |
| 실측 지연이 판단 주기 예산 초과(100ms deadline 초과율) | 틱당 새 토큰 수, 윈도우 attention, 서빙 GPU 수, active 파라미터 | 토큰 예산(변화분 틱) → 서빙 구성 → backbone 재선정 → 판단 주기 순으로 조정하고 비용 재산정 |
| 학습·추론 스트림 상태 불일치 | TBPTT 구간 경계, 윈도우 절단, 분기 상태 폐기, 실행 이력 입력 | 증분/전체 계산 정합성 검사를 먼저 통과시키고 DAgger 데이터의 이력 보존을 확인 |
| 그리퍼 이벤트 누락·중복 | 원하는 상태 라벨, readiness 조건, ACK·멱등 처리 | 라벨의 전환 틱 허용과 실행기 변환 규칙을 재검토 |
| 본 모델이 메모리 기준을 못 맞춤 | 실제 tensor/activation peak | 큰 VRAM 노드로 동일 비교 조건을 옮기고 비용 재산정 |

첫 반복의 우선순위는 **계약과 근거가 검증된 데이터 → 지연 예산 안의 backbone 선정 → 실제 학습·재개 → 의미 일반화 → 공유 계산 가속 → 규칙 기준군 대비 동적 로봇 성과**다. 이후에도 데이터 버전과 모델 버전을 함께 올리고, 성공한 checkpoint를 오류 보강의 다음 출발점으로 유지한다.
