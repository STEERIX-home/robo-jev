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
- [ ] 같은 후보에 대해 D0와 D1 dev의 무학습 라벨 점수 읽기 품질을 잰다.
- [x] 후보별로 10초 학습 구간(약 55K token)의 활성 메모리와 윈도우 KV(prefix + 30틱)를 산정해 학습 가능 노드와 비용을 기록한다. (config 기반 추정 + 실측 캐시 바이트, 05 §4; 노드·비용은 확정 후보 기준으로 05에서 재산정)
- [x] 탈락 기준: 토큰 하한에서도 native 지연이 예산을 넘음, 학습 메모리가 예산 노드에 맞지 않음, 무학습 품질이 D1 dev에서 다른 후보보다 뚜렷이 낮음. 결과와 근거를 `backbone-screen.json`에 남긴다. (지연·메모리로 판정; 무학습 품질은 미측정)

**1단계 결과(2026-09-19, DGX Spark GB10, native BF16, transformers 5.17 + fla·causal-conv1d 커널 활성, sdpa; `artifacts/reports/backbone-screen.json`).** stream warm(prefix + 30틱 캐시 뒤 연속 틱, native 캐시는 윈도우 없이 자람)의 p95 모델 시간: 현재 서식(틱당 ≈1.85K 토큰) 2B 928 / 4B 2,299 / 9B 2,447 / 27B 5,640 ms — 모두 80 ms의 ≥7×, 100 ms 초과율 1.0. 500토큰 길이 대용 틱: p95 123 / 305 / 378 / 886 ms, 윈도우 크기 캐시(prefix + 29틱)로 외삽하면 83 / 197 / 278 ms. 단일 요청(≤315토큰, 캐시 없음) p50 33 / 70 / 122 ms = 절편 26.5 / 51.6 / 93.6 ms + 1K 토큰당 38.6 / 107.7 / 152 ms. cold(무상태 재계산)는 warm의 7~25×. 문자 그대로는 **통과 후보가 없다** → "예산 안의 후보가 없으면" 조항대로 토큰 예산(계약 v0.3)이 먼저이고, 2단계 후보(≤2)는 v0.3 길이에서 예산에 가장 가까운 **Qwen3.5-2B(주, 10~15% 개선이면 10 Hz)와 Qwen3.5-4B(5 Hz 대비)**로 한다. 9B는 10 Hz 트랙 제외(거리 3.5~4.7×; FP8은 2단계의 profiler·CUDA graph 귀속 측정 뒤), 27B 제외. 2단계의 지렛대 순서: 마스크 없는 윈도우 attention 커널(sdpa+마스크가 윈도우 틱의 ≈40%, FLOP 시간의 ≈8×) → 정적 윈도우 KV 선할당(틱마다 `torch.cat`이 새 segment를 만들어 reserved가 allocated의 2~4×) → CUDA graph/`torch.compile`(절편).

**2단계 · 최종 선정(G0b, 실제 경로).** 1단계를 통과한 후보(≤2)에 대해 Task 4의 최소 실제 경로(P0 pointer readout + `StreamState` 분기·상태 전달 + 짧은 TBPTT 구간)를 구현하고 readout-only 적응을 거친 뒤 잰다. native 측정은 실제 kernel·cache 복제·메모리 이동·readout 비용을 포함하지 않으므로 최종 속도나 학습 후 품질의 근거로 쓰지 않는다.

- [x] 후보별로 `stream` 경로의 10Hz 연속 틱 p50/p95/p99와 100ms deadline 초과율을 잰다(윈도우 KV·분기 포함). (2026-09-20, DGX Spark: v0.3 lower/upper/지시 변경 + batch-0 40편, `--levers baseline,fused,graphs,compile,readout_bf16,all`; `artifacts/reports/backbone-stream.json`·`backbone-stream-levers.json` — 아래 결과)
- [x] D0·D1 smoke로 readout-only 적응(T0급, 수백 step)을 수행하고 별도 D1 dev(의미 holdout 포함)의 품질을 잰다. D0의 우연 수준 통과는 기준으로 쓰지 않는다. (`scripts/adapt_readout.py --mode t0|lora|zero-shot`; T0 2B 120 step·4B 50 step, 짧은 LoRA, 무학습 라벨 점수; 평가 = batch-0 dev 5편·ood_dev 1편(작다)·pilot dev/ood_dev·D0 dev, 위치 편향(치환 답 변경률)·문맥 섞기·규칙 기준군 열 포함; `adapt-{2b,4b}-{t0,lora}.json`, `zero-shot-{2b,4b}.json`)
- [x] 10초 구간 학습의 실제 peak 메모리를 재고, 후보별 D1·D2 학습 시간을 [GPU 계획](05-experiment-and-cloud-plan.md)의 시간 풀과 대조한다. (`--mode chunk-memory`, 05 §4 표: 2B readout-only 6.2 GiB / full+층 checkpointing 47.3 GiB, 4B 14.4 / 5초 58.2 GiB·10초 OOM; 처리량과 시간은 05 §6)
- [x] 지연·품질·학습 비용을 함께 놓고 본 실험 backbone을 확정한다. 탈락 후보의 측정값과 근거를 `backbone-selection.json`에 남기고, 확정 후보 기준으로 용량·비용 표를 다시 계산한다. (`scripts/select_backbone.py` → `artifacts/reports/backbone-selection.json`; 03 §"지연 예산", 05 §4·§6 갱신)

**2단계 결과(2026-09-20, DGX Spark GB10, BF16, `scripts/measure_candidates.py --path stream`; 보고서 `.superpowers/sdd/task-g0b-report.md`).** 실제 경로(정적 prefix KV + 30틱 윈도우 버퍼, DeltaNet 상태 명시 전달, 마스크 없는 varlen flash attention, 10개 결정 표지의 1토큰 배치 forward, pointer readout)의 틱당 모델 p95: **2B** baseline lower 74.7 · upper **83.8** · 지시 변경 82.3 · batch-0(40편 1,348틱) **80.0** ms → upper에서 3.8 ms 미달, batch-0는 0.0 ms(초과율 0.029 / 0.014); 지렛대 **`fused`**(틱 몸통과 분기를 한 forward로, 가중치 한 번 읽기; 40편 재실측 `backbone-stream-fused.json`) upper **62.7** / batch-0 **60.2** ms(p99 91.1 / 84.7, max 91.1 / 101.0), 100 ms 초과 2/1,348틱 = 0.15 %(둘 다 E1 60번째 틱의 ≈1,000토큰 소개 틱) → **10 Hz 통과(여유 17 / 20 ms)**; `all`(+ dense `torch.compile` + bf16 readout, 기본 서빙 구성) 52.5 / 52.0 ms·초과 0(여유 27 / 28 ms). 앞 8편(E0뿐) 부분집합의 지렛대 선별값 65.0 / 57.3은 E1 소개 틱이 없어 판정에 쓰지 않는다. **4B** baseline 182.3 / 175.7 ms(10 Hz·5 Hz 모두 미달), fused 144.6 / 122.6 ms → 5 Hz만(여유 5 / 27 ms), all 129.9 / 110.5 ms. 분기 forward의 CUDA graph는 −2 ms뿐(launch-bound가 아님), 귀속 측정(단일 요청 186토큰)에서 절편은 launch 빈틈 5~13 %가 아니라 weight-read(하한이 graph 시간의 59~70 %) → FP8-9B 닫음. 메모리: 2B 가중치 3.51 + peak 4.14 GiB(틱 사이 증가 0), 4B 7.83 / 9.45 GiB. **품질**(정확도 %; 괄호는 문맥 섞기 대조군 / 규칙 기준군): readout-only T0 2B(120 step) batch-0 dev **88.3**(87.4 / 96.2)·ood_dev 83.6(83.6 / 96.0)·pilot dev **46.5**(41.6)·pilot ood_dev 45.5(39.6); T0 4B(50 step) batch-0 dev 83.8(82.4)·ood_dev 85.0(85.0)·pilot dev 42.4(39.2); 무학습(Nimble 방식, 16틱마다) 2B batch-0 dev 48.0·pilot dev 42.1 / **4B 75.7·51.8** — 4B의 무학습 판단이 뚜렷이 낫고(Nimble의 61 / 66 %와 같은 방향), 로봇 스트림의 대조군은 (G0b 당시) **지시 섞기**(지시·목표 텍스트만 굴리고 물리 상태·후보는 그대로; D1 fix round 1부터 표준 열은 id 재매핑 상태 섞기이며 다음 backbone 평가는 그 열로 읽는다)라 상태에 달린 질문에서 모델과 같은 것이 정상이고, 읽을 것은 q_main이다 — T0 2B dev 64.6 % vs 지시 섞기 63.5 %, **ood_dev 23.4 % = 23.4 %**(규칙 기준군 75.5 %): 적응한 2B는 zoneF holdout 계열에서 목표를 읽지 않는다(LoRA 2B는 74.2 vs 71.9 / 60.6 vs 55.3으로 읽기 시작). 비로봇(상태 섞기 대조군)에서 T0의 이득은 +5 pt에 치환 답 변경률 33 %(위치 편향 큼). **품질은 미결이다**: 무학습 4B(pilot dev/ood_dev 51.8 / 51.3 %)가 pilot에서는 적응한 모든 run(최고 2B T0 46.5 / 45.5)을 앞서고 batch-0 dev에서는 적응한 2B(88.3)가 앞선다 — 지연이 선정을 결정하며, D1 dev(≈40편)에서 같은 step·seed의 비교가 나오기 전에는 4B를 품질 문제에서 물리지 않는다. 짧은 LoRA: 2B LoRA(r=16, 40 step, 59 s/step, peak 26.9 GiB) batch-0 dev 86.5·ood_dev 84.6·pilot dev 45.4·pilot ood_dev 44.3 %(문맥 섞기 대조군 87.1 / 84.6 / 41.2 / 38.3); 4B LoRA(r=16, 30 step, 120 s/step, peak 64.8 GiB) batch-0 dev 75.7·ood_dev 75.2·pilot dev 41.4·pilot ood_dev 42.9 %(문맥 섞기 대조군 76.5 / 75.2 / 35.4 / 35.7) — 수치 전체는 보고서 §S3.1과 `backbone-selection.json`. **확정: `Qwen/Qwen3.5-2B` + `fused` 지렛대**(10 Hz 게이트를 통과하는 유일한 후보; 4B는 어떤 지렛대로도 10 Hz에 못 들고 fused/all로 5 Hz 대비). 4B는 무학습 품질이 더 낫지만 지연 게이트가 결정적이며, 5 Hz로 주기를 낮추는 결정이 있을 때만 후보다.

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

- [ ] T0 readout-only 학습에서 optimizer step 전후 readout이 바뀌고 frozen backbone은 그대로인지 검사한다.
- [ ] T1에서 text backbone gradient와 실제 parameter 변경을 확인한다. loss만 감소하고 backbone 업데이트가 빠지는 오류를 차단한다. vision encoder는 고정 상태를 유지한다.
- [ ] 스트림 데이터의 truncated BPTT를 구현한다. 에피소드를 10초 구간으로 나눠 같은 optimizer step 안에서 차례로 forward·backward하고 recurrent/conv 상태와 윈도우 KV를 detach해 전달하며, 한 에피소드를 하나의 accumulation 단위로 둔다. 구간 경계의 상태가 추론 시 증분 계산과 일치하는지, 분기 gradient가 공통 상태로 합쳐지는지 검사한다. 단일 요청 데이터와 스트림 데이터를 같은 step에 섞는 sampler(로봇/비로봇 축, 70/20/10 축, 정상 유지 틱 하향 가중)를 두고 실제 유효 loss 비중을 기록한다.
- [ ] 단일 GPU의 readout-only 저장·재개를 검증한 다음, 8 GPU FSDP2의 text backbone full training에서도 재검증한다. 재개 시 진행 중이던 에피소드의 구간 위치와 상태를 복원한다.
- [ ] 아래 시작 설정으로 200 step profile을 수행하고 실제 메모리·처리량·checkpoint 크기를 기록한다.
- [ ] 재개 직후 다음 batch·loss·업데이트를 중단 없는 실행과 비교하고 G0를 판정한다.

```yaml
model_id: Qwen/Qwen3.8-27B
model_revision_manifest: artifacts/models/qwen38-27b-manifest.json
dataset_manifest: artifacts/datasets/d1/manifest.json
dtype: bfloat16
execution_backend: independent_paths
readout: decision_pointer
layout: state_first          # 비로봇 단일 요청. 로봇 스트림은 stream_l1a
stream_chunk_seconds: 10     # 스트림의 truncated BPTT 구간
stream_window_ticks: 30      # full-attention 윈도우(정적 prefix 별도 보존)
trainable: text_backbone_and_readout
freeze_vision_encoder: true
optimizer: adamw
backbone_lr: 0.00001
readout_lr: 0.0001
weight_decay: 0.01
gradient_clip: 1.0
warmup_ratio: 0.05
microbatch_states_per_rank: 1
gradient_accumulation: 4
world_size: 8
max_total_tokens: 8192
activation_checkpointing: true
max_steps: 200
max_wall_hours: 2
estimated_hourly_usd: 31.92
budget_usd: 63.84
seed: 17
```

학습 설정은 `dataset_manifests: [...]`로 로봇 batch와 비로봇 데이터의 manifest를 여러 개 받아 한 run에 넣는다(manifest별 domain 기본값, 레코드의 `provenance.domain`이 우선). step마다 로봇 스트림 단위와 비로봇 묶음 단위를 둘 다 넣고 손실은 04의 0.6/0.4 혼합이다(CPU 검증 완료). `independent_paths`는 질문별 causal 경로를 복제하는 P0다. Q개 경로를 내부 microbatch로 나누어 상태별 loss를 구성하고 실제 처리량을 기록한다. `readout: candidate_branch`는 비교군 R이며 경로가 Q×K개로 늘어난다. 로봇 스트림 레코드는 `layout: stream_l1a`로 읽히며 `stream_chunk_seconds`·`stream_window_ticks`가 truncated BPTT와 윈도우를 정한다. `model_id`는 Task 2b에서 확정한 backbone으로 바꾸며, 위 값은 첫 후보의 예시다. 이후 `shared_hybrid` P1을 따로 profile한다. 이 경로는 full-attention과 DeltaNet의 정합성 검사를 모두 통과해야 한다. 입력 축소만으로 최대 지원 길이를 통과한 것처럼 표시하지 않는다. profile 후 본 학습용 step·epoch 상한을 다시 산정한다.

예정 CLI:

```bash
torchrun --standalone --nproc_per_node=8 -m robo_jev.train --config configs/train/qwen38-27b-pilot.yaml
```

**클라우드(GPU) 단계의 전제 — CPU 검증에서 드러난 것(2026-09-19).** (1) sampler가 에피소드를 토큰 단위 Python 리스트로 미리 직렬화해 들고 있어 D1 규모(400 에피소드 × ≈165K 토큰)에서는 10GB를 넘는다 — 단위별 지연 직렬화와 배열 기반 layout으로 바꾼 뒤에 D1 학습을 시작한다. (2) `torch` 핀은 Linux에서 CUDA 13 휠 묶음을 받으므로 클라우드 이미지·드라이버에 맞는 index를 05에 고정한다. (3) `configs/`는 wheel에 포함되지 않으므로 Dockerfile이 저장소째 담는다. (4) step 중간 checkpoint는 누적 gradient(실제 backbone 크기)를 담으므로 선점(preemption) 환경에서만 켠다. (5) 질문 수·후보 수의 프로파일 상한(Q≤16, K≤32)은 `validate_record`가 아직 강제하지 않는다 — 실제 backbone 실행 전에 넣는다. (6) 서빙 클라이언트는 응답에 `meta`(observed_at·goal_version·후보 집합 버전·seq)를 반드시 실어야 하네스의 유효성 검사(08 §5.0)가 동작한다.

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
