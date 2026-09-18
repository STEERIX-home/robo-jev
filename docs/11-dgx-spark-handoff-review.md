# DGX Spark 인계 준비 리뷰

검토일: 2026-09-19. 기준 HEAD: `df5e00a16324e33f7fb928b814388634a7681b7f`. 리뷰 시작 시 작업 트리는 깨끗했다. 대상은 현재 코드·설정·문서, 로컬 `.superpowers/sdd/HANDOFF.md`와 작업 보고서다. 검증은 현재 Mac에서 수행했다. Spark에 접속하거나 ARM64/CUDA 실행을 검증하지 않았다. 기존 소스와 인계 문서는 수정하지 않았다.

## 1. 판단

**Spark에서 개발을 이어받을 코드 기반은 준비됐다. 다만 현재 문서와 저장소만으로 재현 가능한 인계가 완결되지는 않았다.** 실제 학습은 클라우드, Spark는 개발·데이터 생성·배포 지연 측정이라는 역할 분리는 타당하다. 아래 재개·모델 식별 문제와 인계 자료 전달을 해결한 뒤 개발을 이어갈 수 있다. D1 대량 제작·본 학습·10Hz 판정은 별도 게이트다.

이전 리뷰와 달리 지금은 소형 계산 모델에서 pointer readout, 스트림 상태, loss, 실제 optimizer step, 혼합 sampler, TBPTT와 checkpoint/재개까지 구현돼 있다. 이것은 모델 개발의 실질적인 진전이다. 다만 `TinyHybrid`는 난수 가중치의 계산 fixture이고, 공개 사전학습 backbone에 대한 의미 판단·GPU kernel·학습 성능은 아직 검증되지 않았다.

| 범위 | 이번 확인 |
| --- | --- |
| 현재 코드의 기존 검사 | `uv run pytest -q`: **885 passed, 1 xfailed in 122.85s** |
| 예상 실패 | E1 seed 29의 후보 상한 문제. 알려진 실패를 숨기지 않고 유지한 상태 |
| CPU 학습 | 기존 검사에 가중치 업데이트, 스트림 분기, 혼합 손실, 동일 데이터에서 중단 후 재개 검사가 포함됨 |
| 실 tokenizer | 받아 둔 Qwen tokenizer로 B2 재측정 완료. prefix 382~385, 합성 하네스 최초 틱 1,764~3,469, 후속 지시 변경 틱 최대 **3,712** |
| Spark 인수 | 미실행. `uv sync` 성공, GPU 인식, CUDA 연산, 실모델 forward, 실제 스트림 지연은 별도 확인 필요 |
| 실모델 경로 | 아직 fixture만 구현. `measure_candidates.py`, 사전학습 backbone adapter, GPU/BF16 학습 경로는 다음 구현 작업 |

## 2. 수정이 필요한 항목

### S1 · P1 — 재개 시 데이터 내용이 바뀌어도 기존 run으로 계속 진행한다

위치: [train.py](/Users/user/Documents/robo-jev/src/robo_jev/train.py:1058), [sampler.py](/Users/user/Documents/robo-jev/src/robo_jev/sampler.py:527).

`Trainer._load`는 설정값이 같은지만 확인한다. checkpoint에 저장한 데이터 manifest 해시와 현재 manifest 해시를 비교하지 않는다. sampler도 bucket과 인덱스의 존재를 검사할 뿐 같은 레코드·정답인지 확인하지 않는다. 적재 시 파일과 현재 manifest의 해시가 일치하는 검사는 있지만, 그것이 이전 run과 같은 데이터임을 보장하지 않는다.

**재현:** D0 사본으로 1 step 후 저장 → 첫 train 레코드의 정답을 `c0`에서 `c1`로 변경하고 사본 manifest의 파일 해시도 갱신 → 같은 설정·경로로 재개. manifest 해시가 달라졌는데도 오류 없이 step 1로 복원됐다. 원본 fixture는 건드리지 않았다.

머신 이동 후 데이터를 재생성하거나 같은 경로에 새 버전을 받으면 발생할 수 있다. 구간 중간 재개에서는 이전 데이터에서 계산한 recurrent/KV 상태를 바뀐 후반 구간에 연결할 수도 있다. 데이터의 내용 해시, tokenizer revision·파일 해시, 직렬화 버전 등 run 정체를 재개 전에 대조해야 한다. 의도적으로 데이터를 바꾸는 후속 학습은 별도 run으로 구분해야 한다.

### S2 · P1 — 실모델 ID를 설정해도 소형 fixture를 만들고 그 ID로 기록한다

위치: [train.py](/Users/user/Documents/robo-jev/src/robo_jev/train.py:298), [manifest 작성](/Users/user/Documents/robo-jev/src/robo_jev/train.py:620).

`build_model`은 `model_id`와 관계없이 `Judge.from_config`로 `TinyHybrid`를 만든다. 설정 검사는 미지원 BF16·분산·backend를 거절하지만 미지원 `model_id`는 거절하지 않는다. manifest의 `model.id`에는 사용자가 입력한 값을 그대로 적는다.

**재현:** 배포된 `tiny_cpu.yaml`의 `model_id`만 `Qwen/Qwen3.5-9B`로 바꿨다. 설정이 통과했고 실제 생성물은 **376,745개 파라미터의 CPU `TinyHybrid`**였다.

실모델 adapter가 없는 것은 현재 단계의 명시된 범위다. 문제는 잘못된 설정이 성공처럼 보이는 것이다. fixture 전용 경로에서는 `tiny_hybrid` 이외 ID를 거절하고, 실모델 경로가 생기면 실제 로드한 모델 종류·revision·파라미터 수·장치를 검증해 기록해야 한다.

### S3 · P1 — `git clone`에 핵심 인계 정보가 포함되지 않는다

위치: [HANDOFF.md](/Users/user/Documents/robo-jev/.superpowers/sdd/HANDOFF.md:28), [.gitignore](/Users/user/Documents/robo-jev/.gitignore:2).

인계 절차는 `git clone → uv sync → pytest → tokenizer fetch`다. 하지만 HANDOFF 자체, 작업 보고서, 진행 기록은 Git에서 제외한 `.superpowers/`에 있고, tokenizer manifest·데이터 manifest·측정 결과도 제외된 `artifacts/`에 있다. 문서는 Claude 메모리 복사만 언급하고, 이 자료들의 전달 또는 정확한 재생성 절차는 정의하지 않는다.

또한 HANDOFF 9행은 main 병합 완료인데 10행은 모델 브랜치 병합과 최종 리뷰를 다음 작업으로 안내한다. 테스트 기대값도 7행의 863과 28행의 494/577로 남아 있어 현재 885와 다르다. 다음 담당자가 어느 상태를 이어받아야 하는지 직접 재구성해야 한다.

코드 기준 SHA, 현재 완료 상태, 남은 작업, 전달할 파일·해시·목적지와 재생성할 파일을 하나의 인계 목록으로 정리해야 한다. scratch를 커밋하지 않는 원칙은 유지할 수 있다. 추적되는 인계 문서나 별도 전송 묶음으로 전달 여부를 검증하면 된다.

### S4 · P1 — 실제 입력 길이·배포 장비와 지연·학습 예산이 어긋난다

위치: [HANDOFF.md](/Users/user/Documents/robo-jev/.superpowers/sdd/HANDOFF.md:14), [03의 지연 산정](/Users/user/Documents/robo-jev/docs/03-model-and-training-design.md:38), [08의 토큰 예산](/Users/user/Documents/robo-jev/docs/08-streaming-io-and-data-contract.md:91).

인계 문서는 Spark를 지연 판정 장비로 정하고 500/1,800토큰의 native forward를 첫 측정으로 제안한다. 반면 정본 03·08의 수치는 여전히 H100과 틱당 450~700토큰, 10초 학습 구간 약 55K토큰을 전제한다.

이번 B2 재측정에서 최초 틱은 **1,764~3,469**, 실행 이력과 지시 변경이 있는 후속 틱은 **최대 3,712토큰**이었다. 500토큰은 아직 구현하지 않은 압축·변화분 계약의 목표값이다. 1,800토큰은 현재 조건의 하단만 대표한다. 동일한 길이가 100틱 지속된다고 가정하면 10초 구간은 약 **176K~371K토큰**이다. 이는 실제 에피소드 길이 분포의 측정값은 아니지만 55K를 계속 메모리·시간 예산에 사용할 수 없음을 보여준다.

Spark는 ARM64·128GB 공유 메모리·273GB/s 장비다. H100의 실효 연산량 가정이나 단일 배속 환산만으로 모델별 지연을 확정할 수 없다. [NVIDIA 하드웨어 명세](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [포팅 개요](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)

현재 서식의 하단·K=32·지시 변경·긴 의미 설명을 포함한 대표 입력과 warm history 길이를 측정해야 한다. 압축·변화분 입력은 별도 버전으로 만들고 품질·정합성을 함께 검사한다. FP8은 05의 BF16 주 비교와 별도 조건이며, 지원 kernel·양자화 대상·품질 차이를 명시해야 한다. 2~4B 또는 9B의 본선 선정은 속도와 미학습 판단 품질을 함께 측정한 결과로 결정해야 한다.

### S5 · P2 — tokenizer를 다른 머신에서 같은 버전으로 재현할 경로가 없다

위치: [fetch_tokenizer.py](/Users/user/Documents/robo-jev/scripts/fetch_tokenizer.py:39), [tokenizer.py](/Users/user/Documents/robo-jev/src/robo_jev/model/tokenizer.py:64).

fetch는 매번 Hub의 현재 revision을 조회하며 `--revision`이나 기존 manifest를 입력받지 않는다. 기본 호출은 첫 후보를 받지 못하면 다른 모델 tokenizer로 넘어간다. 다운로드 결과에 SHA를 남기는 것은 좋지만, 다음 머신이 이전 SHA를 요청할 수는 없다. `load_tokenizer`도 manifest의 파일 해시와 실제 파일을 대조하지 않는다.

따라서 같은 명령이 실행돼도 토큰 수·후보 경계·어휘가 달라질 수 있다. 현재 tokenizer의 ID·revision·파일 해시를 인계하고, 고정 revision 다운로드 또는 파일 전송 후 해시 확인을 수행해야 한다. 이번 측정에 사용한 revision은 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`이다. 설치 인수 순서도 tokenizer 확보 후 전체 검사로 바꾸는 것이 맞다. 현재 순서는 실 tokenizer 검사 없이 일부 skip된 통과를 먼저 얻게 한다.

## 3. 이미 알려졌지만 다음 단계 전에 유지해야 할 조건

- **대량 로봇 데이터 제작:** README는 E1 틱의 약 44%가 후보 누락·실행 불가 때문에 낮은 신뢰도의 hold 라벨이 된다고 보고한다. 이번 리뷰에서는 40개 배치를 다시 생성하지 않았다. 현재 전문가 코드도 비키프레임 `q_main`에 단일 선택만 정답으로 적는다. 0.25 가중치는 완화이며 후보 공간·허용 집합 문제의 해결은 아니다. 후보 계약을 정한 후 D1을 다시 만들고 128K rollout을 실행한다.
- **일반화 검증:** 비로봇 pilot의 holdout 목록은 여전히 비어 있고 D0 사람 검수도 미완이다. 개발 인계를 막을 이유는 없지만 의미 일반화·학습 품질 판정 전에 완료해야 한다.
- **CPU와 GPU 검증 경계:** 현재 Trainer는 CPU FP32/FP64, 난수 fixture, 단일 프로세스만 지원한다. 장치 이동, GPU RNG, BF16 오차, 실제 backbone fork/gradient와 kernel은 별도 작업이다. 소형 계산 모델 검사 통과를 실제 Qwen 구현의 통과로 승계하지 않는다.
- **데이터 적재 메모리:** 현재 eager 직렬화와 Python 목록 기반 layout은 큰 데이터에서 메모리를 많이 쓴다. Spark의 CPU·GPU가 같은 메모리를 공유하므로 데이터 worker와 모델 측정의 동시 부하도 기록해야 한다.

## 4. Spark에서 이어갈 순서

1. 기준 SHA와 인계 묶음·tokenizer 파일 해시를 확인하고, `uv sync --locked` 후 실제 tokenizer를 포함한 CPU 검사를 재현한다. 재개 시 내용 변경 거절과 미지원 모델 ID 거절도 먼저 보강한다.
2. ARM64·GB10에서 쓸 PyTorch/CUDA/driver 또는 container digest를 고정하고 CUDA tensor 연산·동기화·실모델 load/forward를 확인한다. CPU 테스트 통과와 별도 결과로 기록한다. NVIDIA는 Spark용 NGC 환경과 버전 고정을 안내한다. [NVIDIA NGC 안내](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)
3. 후보 공간과 서식/변화분 계약을 확정하고 실제 tokenizer로 토큰 분포를 다시 잰다. 압축 목표값과 현재 측정값을 섞지 않는다.
4. Spark에서 native 예비 측정 후, pointer·상태 분기·윈도우를 포함한 실제 경로로 지연을 잰다. 같은 후보의 의미 판단 품질을 함께 확인해 backbone과 배포 정밀도를 정한다. `scripts/measure_candidates.py`는 현재 파일이 없으므로 이 단계의 구현 산출물로 명시한다.
5. 클라우드에서 실제 backbone의 T0와 작은 T1을 먼저 수행하고, Spark 평가 결과와 연결한 뒤 D1·rollout·본 학습을 확대한다.

Spark를 통과한 결과를 다른 Jetson 구성의 지연 보장으로 일반화하지 않는다. 실제 최종 배포 장비가 별도라면 그 장비에서 같은 판정을 한 번 더 수행한다.

## 5. 검증 산출물

- [반례 재현 스크립트](/Users/user/Documents/robo-jev/artifacts/reviews/spark-handoff/probe.py)
- [모델 식별·데이터 변경 후 재개 결과](/Users/user/Documents/robo-jev/artifacts/reviews/spark-handoff/evidence.json)
- [현재 코드의 B2 재측정](/Users/user/Documents/robo-jev/artifacts/reviews/spark-handoff/tokens-b2.json)

```sh
uv run pytest -q
uv run python artifacts/reviews/spark-handoff/probe.py
uv run python scripts/measure_tokens.py --out artifacts/reviews/spark-handoff/tokens-b2.json
```

리뷰 산출물의 `artifacts/` 파일도 Git에서 제외되므로 인계 시 별도 전달 대상이다. 이번 반례는 실패 동작을 확인한 결과이며 수정 완료 검사가 아니다.
