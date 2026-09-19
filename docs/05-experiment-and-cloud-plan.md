# 실험 환경·GPU 클라우드·비용 계획

작성일: 2026-09-18 · 상태: 환경 구축 전 계획. GPU 예약, 유료 자원 생성, 모델 다운로드와 학습은 아직 실행하지 않았다. 같은 날 근본 검토를 반영해 후보 backbone의 지연·품질 측정 단계(G0a), 연속 틱 지연 측정, 규칙 기반 기준군과 결정 안정성 지표를 추가했고, 이어 [스트리밍 계약](08-streaming-io-and-data-contract.md)에 따라 판단 10Hz·OSC 50Hz·명령 수명 지표와 학습 시퀀스·윈도우 메모리 기준을 반영했고, 6절의 시간 풀을 스트림 토큰 기준으로 다시 산정해 2단계 backbone 게이트(G0a/G0b)와 2×2 실험표에 맞췄다.

클라우드 선호와 총예산은 미정으로 가정한다. 먼저 작은 자원으로 학습·재개·측정을 검증하고, [모델 설계](03-model-and-training-design.md)의 본 비교 실험으로 확대한다. 아래 소요 시간은 구매 가능한 실험 시간의 예시이며 학습 완료 시간의 예측이 아니다.

## 1. 환경을 네 부분으로 나누기

| 환경 | 첫 구성 | 역할 |
| --- | --- | --- |
| 로컬 개발 | 현재 Mac, CPU 테스트와 소형 synthetic 모델 | 계약·serializer·데이터 QA·리포트 작성 |
| 데이터 생성 | Linux CPU worker, 필요할 때만 렌더링 GPU | 합성 상태·규칙 라벨·병렬 simulator rollout |
| 모델 학습 | Linux NVIDIA GPU, 전용 학습 container | 직접 readout·backbone 학습, checkpoint 저장 |
| 평가·서빙 | 고정 GPU 한 대를 우선 사용, simulator 별도 프로세스 | N=1 질문 묶음 지연, 연속 틱 지연, 고정 하네스 폐루프. 여러 GPU에 나눈 서빙은 별도 조건으로 기록하고 모든 비교군에 같은 구성을 적용 |

훈련 중인 GPU에서 지연 benchmark를 동시에 실행하지 않는다. simulator worker가 학습 GPU의 메모리와 CPU를 잠식하지 않게 분리한다. Mac의 실행 시간을 NVIDIA 추론 성능으로 해석하지 않는다.

학습 코드와 데이터 포맷은 공급자에 종속되지 않게 하고, 공급자별 부분은 instance 생성·스토리지 연결·종료 설정으로 제한한다. 첫 단계에 Kubernetes나 다중 노드 cluster를 도입하지 않는다.

## 2. 로봇 실험 환경

첫 시뮬레이터는 **MuJoCo + robosuite 1.5 계열**, 로봇 모델은 **Panda + PandaGripper**로 제안한다. 이는 개발 기준체이며 실제 구매할 로봇을 정한 것은 아니다. robosuite는 headless 실행을 지원하며, 로봇·그리퍼·controller 구성을 분리한다. [설치 문서](https://robosuite.ai/docs/installation.html), [로봇 구성 문서](https://robosuite.ai/docs/modules/robots.html)

| 환경 버전 | 내용 | 필요한 추가 구현 |
| --- | --- | --- |
| E0 | 정적 Lift/PickPlace, 참조 상태 | 고정 목표·성공 판정·snapshot 재현 |
| E1 | 다물체 정리, 도중 지시 변경, 대상 이동(외란)·접근 방해·집기 실패·밀기 복구 | 외란·지시 변경 scheduler, 스크립트 전문가, 스트림 하네스·실행 adapter, episode 기록 |
| E2 | 부분 관측·지연·오차가 있는 상태: 시뮬 3D 카메라(깊이 렌더 + 잡음·결손 모델) → 3D 재구성 → 고정 추출 모듈 | 시뮬 3D 카메라 모델, 재구성 결함(표면 결손·추적 id 흔들림·지연) 재현, 소스별 관측 나이, 오래된 관측 처리 |
| E3 | 실제 3D 카메라 → 3D 재구성 → 같은 추출 모듈, 또는 실제 로봇 | timestamp 동기화(소스별), 좌표 변환, 정밀도 추정 전달, 같은 질문 계약 |

기본 PickPlace가 E1의 모든 기능을 제공한다고 가정하지 않는다. 외란과 pushing segment는 프로젝트 환경 wrapper로 구현한다. 운동 명령은 실행 adapter에서 controller 입력으로 변환한다. controller의 좌표계·회전 표현·delta/absolute 설정을 기록하고 변환 검사를 한다. [공식 controller 문서](https://robosuite.ai/docs/modules/controllers.html)

초기 실험 설정은 물리 timestep 2ms(500Hz, OSC 토크 계산 포함), OSC 말단 목표 갱신 50Hz(명령 혼합·보간은 이 주기), 판단 스트림 10Hz로 제안한다. 실제 속도는 측정해 조정하며 판단 모델에 저수준 500Hz 제어를 요구하지 않는다. 10Hz는 모델 시간 예산 약 70~80ms(관측→명령 적용 100ms deadline 안)에 해당하며, [모델 설계](03-model-and-training-design.md)의 지연 예산 산식에 따라 backbone 규모·틱당 토큰의 상한을 정한다. 실측 지연이 예산을 넘으면 토큰 예산(변화분 틱), 서빙 GPU 수, backbone, 판단 주기의 순서로 조정하고 변경을 기록한다. 명령 수명(관측 deadline 200ms, 발행 lease 300ms), 혼합 100ms, 정지 전이, 반사는 [스트리밍 계약](08-streaming-io-and-data-contract.md) 6절의 컨트롤러 계약을 따른다. 요청이 늦어지면 오래된 요청을 무한히 쌓지 않고 최신 상태와 명령 유효성을 관리한다.

참조 상태 E0와 부분 관측 E2의 성과는 별도로 보고한다. 모델 응답을 기다릴 때도 동적 평가의 세계 시간이 흐르게 한다. 방법은 simulator를 실시간 비동기로 돌리거나, 측정한 end-to-end 지연만큼 모의 시간을 전진시킨 뒤 명령을 적용하는 것이다. 일반 `env.step` 루프가 모델을 기다리며 세계를 멈추는 실험으로 반응성을 주장하지 않는다.

snapshot에는 물리 상태뿐 아니라 controller 내부 상태, 목표·외란 일정, wrapper 상태, 모든 RNG를 저장한다. 동일 snapshot·seed에서 실행을 재현하는지 먼저 확인한다. 외란은 정책의 호출 횟수 대신 모의 시간과 seed로 정의해 비교군 사이의 조건을 맞춘다.

## 3. 소프트웨어와 재현성

첫 호환성 검증 대상은 Linux x86_64, Python 3.11, PyTorch 2.14 계열의 FSDP2/FlexAttention과 Qwen3.8-27B의 `qwen3_5` 구현을 지원하는 Transformers다. 공개 config의 `transformers_version`은 `5.8.0.dev0`이지만, 이 필드만으로 최소 지원 버전이나 분기 학습 호환성을 확정하지 않는다. 실제 load·forward·backward·분산 검사에 통과한 패키지 버전 또는 commit을 lock한다. 이전 Qwen3의 `4.51.0` 조건을 승계하지 않는다. [공식 config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json), [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

학습 container와 simulator container의 dependency lock을 분리한다. 각 image에는 digest, Python·패키지 버전, CUDA runtime을 기록하고, 호스트 NVIDIA driver와 GPU 모델도 run manifest에 담는다. 선택한 driver/runtime 조합에서 full-attention mask, DeltaNet 초기 상태·conv history의 gradient, 분산 통신을 실제로 확인한 뒤 버전을 고정한다. 선택한 DeltaNet·causal-conv kernel도 버전과 기준 연산 대비 오차를 기록한다.

첫 학습 분산 방식은 한 노드의 FSDP2 + activation checkpointing + BF16이다. optimizer state는 FP32를 기본으로 하며 loss와 확률 정규화도 FP32에서 계산한다. 양자화와 CPU offload는 주 비교 설정에 넣지 않는다. offload가 필요하면 그 비용을 포함한 별도 조건으로 기록한다.

FSDP2는 파라미터·gradient·optimizer state를 GPU들에 나눠 보관하는 수단이다. GPU 수를 늘리는 것만으로 메모리가 단순 합산되는 것은 아니며 layer all-gather와 통신 버퍼의 peak가 남는다. [공식 FSDP2 설명](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## 4. GPU 용량 설계

다음은 decimal GB 기준의 거친 tensor 용량 계산이다. Adam 계열 full training의 예산을 파라미터당 16 bytes로 잡았다. 실제로 master weight를 별도 보관하는지, gradient와 parameter의 저장 dtype이 무엇인지에 따라 달라지므로 코드의 tensor inventory와 peak memory로 대체해야 한다.

| 모델 | BF16 가중치만 | 학습 상태의 계획값 `16P` | 첫 full training 후보 |
| --- | --- | --- | --- |
| Qwen3.8-27B의 언어 모델 약 27B | 약 54GB | 약 432GB | 8×H100 80GB, 짧은 요청부터 |

위 값은 모델 카드의 언어 모델 규모를 이용한 근사치다. 실제 가중치 tensor와 학습 대상 parameter를 집계해 대체하며, 고정된 vision encoder를 GPU에 상주시킬 때의 메모리도 별도 더한다. activation, full-attention workspace, recurrent/conv 분기 상태, 임시 all-gather, CUDA context도 필요하다. 4×80GB는 위 full training 상태 계획값만으로 용량이 부족하므로 기본안에서 제외한다.

rank당 상태 1건·경로 길이 4K에서 시작해 지원 상한으로 늘린다. P0에서는 상태 하나에 질문 수 Q개의 복제 경로가 생기고, 후보별 분기 참고군(R)에서는 Q×K개가 생기므로 논리적 요청 tokens와 실제 처리 tokens를 별도 집계한다. P1도 recurrent state 분기·역전파의 메모리를 포함한다.

로봇 스트림 학습은 에피소드를 10초 구간(약 100틱; 실측 서식으로 176K~371K token, 계약 v0.3 목표로 ≈50K — 55K 가정은 v0.3 이후 값)으로 나눠 같은 optimizer step 안에서 상태를 전달하는 truncated BPTT로 수행하므로, 구간당 활성 메모리와 full-attention 윈도우(정적 prefix + 30틱)의 KV(27B 기준 약 1.0GiB, 분기·임시 메모리 제외 — Spark 실측 재현 1.007 GiB; 실측 캐시 바이트는 현재 서식 윈도우(prefix + 30틱 ≈ 55.7K 토큰)에서 2B 663 MiB / 4B·9B 1,767 MiB, 500토큰 윈도우(≈15.4K)에서 190 / 506 MiB, recurrent 상태는 런타임이 fp32로 유지 — `artifacts/reports/backbone-screen.json`)를 후보 backbone의 메모리 기준에 넣는다. 이 조건은 27B보다 9B 후보에 유리하며 Task 2b의 선정 기준이다. 지연 게이트(G0b의 10Hz·100ms 초과율)는 배포·실행 장비가 DGX Spark/Jetson급 엣지이므로 **그 장비에서** 잰다(클라우드는 학습 처리량만); 엣지 실측(2026-09-19, Spark native; 03 §"지연 예산" 표)으로 G0b 후보는 **2B(주)·4B(5 Hz 대비)**로 확정했고 9B는 10 Hz 트랙에서 제외(FP8-9B는 G0b 귀속 측정 뒤), 27B는 제외했다(HANDOFF 결정 3). 학습 노드·비용은 2B/4B 기준으로 다시 산정한다: 학습 상태 계획값 `16P`는 2B ≈30GB, 4B ≈67GB(activation·분기 상태 별도)라 단일 80GB 또는 소수 GPU 노드로 가능하다.

위 용량표는 첫 후보 27B dense 기준의 상한 산정이다. 후보가 바뀌면 다시 계산한다. 작은 dense 후보(예: 약 9B)는 학습 상태 계획값이 약 144GB로 줄어 더 작은 노드로도 가능하다. MoE 후보는 추론 연산이 active 파라미터에 비례하지만 **학습 메모리는 총 파라미터에 비례**한다. 예를 들어 총 35B의 MoE는 `16P` 기준 약 560GB로 8×80GB에서 activation 여유가 거의 없으므로 8×H200 또는 8×B200급 노드가 필요할 수 있다. 후보별 학습 노드와 비용은 선정 시점에 같은 방법으로 산정한다. peak allocated/reserved, 최대 rank의 메모리, gradient accumulation의 영향을 측정한다. Spark native 추론 실측에서는 peak allocated 13.7 / 28.3 / 37.9 GiB(2B/4B/9B)에 reserved가 54.8 / 57.0 / 83.7 GiB로 2~4×였다 — 틱마다 자라는 KV의 `torch.cat`이 새 segment를 만드는 allocator 현상이라(`num_alloc_retries`·`num_ooms` 0) 2단계는 정적 윈도우 KV를 미리 할당한다. 10초 학습 구간 activation의 하한 추정(토큰 × hidden × 층 × 2 B)은 50K 토큰에서 4.6 / 7.6 / 12.2 GiB, 실측 서식 184,844토큰에서 16.9 / 28.2 / 45.1 GiB다(추정이며 실측은 G0b).

**메모리 통과 기준은 정한 최대 길이에서 200 step 동안 OOM이 없고 실사용 장치 메모리의 최소 10%가 남는 것**으로 둔다. 기준을 못 맞추면 checkpointing·불필요 tensor 보존을 먼저 수정하고, 여전히 부족하면 8×H200 또는 8×B200처럼 여유가 있는 동일 노드 구성으로 변경한다. 이때 모든 본 비교군의 하드웨어와 비용표를 함께 바꾼다. 비교군별로 입력을 다르게 잘라 맞추지 않는다.

27B readout-only와 추론은 1×80GB에서 짧은 요청으로 먼저 시험한다. 실제 가중치·분기 수·kernel workspace를 포함한 실측 통과가 필요하다. 추론 메모리 초과 시 더 큰 VRAM의 단일 GPU를 공통 benchmark 장치로 선택한다. 학습·평가 모두 양자화하지 않은 동일 backbone을 기본으로 한다.

## 5. 클라우드 선택과 공개 가격

첫 조달안은 **다중 GPU 노드가 필요한 학습에는 Lambda**, **단일 GPU 파일럿과 대체 재고에는 Runpod**를 비교하는 것이다. 이미 보유한 GPU나 계정이 있으면 같은 실행 계약에 연결한다. 가격만으로 정하지 않고 한 노드의 GPU 수·NVLink/NVSwitch 연결·CPU RAM·로컬 SSD·저장소 위치·재고를 함께 확인한다.

2026-09-18 확인한 공개 USD 가격은 아래와 같다. 세금·지역·재고·약정·부가 저장소에 따라 실제 견적은 달라질 수 있다. 다중 GPU 행은 **GPU 한 대당 시간 요금**과 **노드 전체 요금**을 구분했다.

| 공급자·공개 구성 | GPU당 시간 | 노드당 시간 |
| --- | --- | --- |
| Lambda 1×H100 PCIe 80GB | $3.29 | $3.29 |
| Lambda 4×H100 SXM 80GB | $4.09 | $16.36 |
| Lambda 8×H100 SXM 80GB | $3.99 | $31.92 |
| Lambda 8×B200 180GB | $6.69 | $53.52 |
| Runpod Pods H100 PCIe 80GB 표시값 | $2.89 | 실제 선택한 수량·상품으로 확인 |
| Runpod Pods H100 SXM 80GB 표시값 | $3.49 | 실제 선택한 수량·상품으로 확인 |
| Runpod Pods H200 141GB 표시값 | $4.59 | 실제 선택한 수량·상품으로 확인 |

근거: [Lambda 가격표](https://lambda.ai/pricing), [Runpod 가격표](https://www.runpod.io/pricing). Runpod 공개 카드의 표시값은 특정 지역·Secure/Community 선택·다중 GPU 연결이 확보된 견적과 동일하다고 가정하지 않는다. 예약 직전 콘솔의 상품·수량·지역·시간 단가를 run budget에 복사해 검증한다.

동일 노드인지 `nvidia-smi topo -m`으로 확인하고 collective bandwidth를 측정한다. 서로 다른 장치 이름이 비슷하다는 이유로 H100 PCIe와 SXM의 학습 성능을 동등하게 취급하지 않는다. spot 자원은 중단·복구가 검증된 이후 보조 실험에만 검토한다.

## 6. 첫 연구 사이클의 비용 봉투

클라우드 예산 답변이 아직 없으므로 아래는 조정 가능한 자원 배분안이다. full training의 기본 구성은 8×H100 SXM 노드다. G0~G1까지 약 $1,116의 GPU 시간 풀로 후보 선정과 학습 경로를 검증하며, 이후 단계는 처리량과 품질을 확인한 뒤 나눠 집행한다. 본 실험 backbone이 확정되면 G1 이후의 노드 구성과 단가를 그 후보 기준으로 다시 계산한다. 이전 14B→32B 가정의 약 $504·전체 $8,900~10,200 추정과 "상태당 4,000 token·5K/50K 상태" 기준의 산정은 더 이상 쓰지 않는다.

| 단계 | 배정한 실행 시간 | GPU 비용 예시 | 확보할 결과 |
| --- | --- | --- | --- |
| G0 환경·재개 검증 | 1×H100 PCIe 12h + 8×H100 SXM 2h | $103.32 | 모델 load·상태 분기·학습 step·저장/재개·profile |
| G0a backbone 1단계 예비 선별 | 1×H100 PCIe 8h | $26.32 | native 지연, 무학습 품질, 학습 메모리 산정, 후보 ≤2 |
| G0b backbone 2단계 최종 선정 | 1×H100 PCIe 16h + 8×H100 SXM 4h | $180.32 | 실제 스트림 경로의 지연·deadline 초과율, readout-only 적응 후 D1 dev 품질, 구간 학습 peak 메모리, backbone 확정 |
| G1 pilot(확정 backbone) | 8×H100 SXM 24h + 단일 PCIe 평가 12h | $805.56 | LR 진단·첫 학습 checkpoint·R과 L1/L0의 D1 비교 |
| G1b 데이터 확대·2×2 | 8×H100 SXM 총 96 node-hours + 단일 PCIe 평가 16h | $3,116.96 | 두 학습 방식의 곡선(로봇 에피소드 400/1,600/4,000 + 비로봇 2K/8K/20K), 각 checkpoint의 공유 없음/있음 실행 |
| G2 반복·최종 평가 | 8×H100 SXM 총 144 node-hours + 단일 PCIe 평가 48h | $4,754.40 | 최종 규모 2 방식 × 3 seeds를 목표로 하는 시간 풀, 봉인 평가·폐루프 최종 시험 |
| 합계 | 위 여섯 단계 | $8,986.88 | 완료 실험 수는 실측 처리량에 따름 |
| 재시도 예비 20% 포함 | 합계×1.2 | $10,784.26 | compile·실패·재검증의 여유 |

### 시간 풀 적합성 (스트림 기준 재산정)

아래는 스트림 데이터 정의로 다시 계산한 학습 토큰과 소요 시간이다. 가정: 틱당 600 token(450~700의 중앙값 — **실측 1,764~3,712와 맞지 않으므로 계약 v0.3 뒤 재측정 값으로 교체**), 비로봇 상태당 4,000 처리 token, epoch은 D1 3회·에피소드 1,600 2회·4,000 1회(스트림 틱의 중복 때문에 큰 규모에서 epoch을 줄임), 8×H100 처리량은 27B 5,000 tokens/s와 9B 15,000 tokens/s(27B의 FLOPs 비 3배로 환산한 추정). **처리량은 실측값이 아니며 Task 2b·G0에서 대체한다.**

| 학습 항목 | 토큰 | 27B 시간 | 9B 시간 |
| --- | --- | --- | --- |
| epoch당: D1 (로봇 72M + 비로봇 8M) / 1,600 (288M + 32M) / 4,000 (720M + 80M) | 80M / 320M / 800M | | |
| 1순위 pointer 곡선 (D1 ×3, 1,600 ×2, 4,000 ×1) | 1.68B | 93h | 31h |
| 2순위 B1 학습 (D1 ×3, 4,000 ×1) | 1.04B | 58h | 19h |
| 3순위 R(D1, 토큰 약 1.7배)과 무상태 L0(D1, 틱당 1,400) | 0.94B | 52h | 17h |
| LR 진단 (D1 1 epoch × 3값) | 0.24B | 13h | 4h |
| 1~3순위 + LR 합계 | 3.9B | **216h** | **72h** |
| 4순위 seeds (최종 규모 2 방식 × 추가 2 seeds) | 3.2B | 178h | 59h |
| 1~4순위 합계 | 7.1B | **394h** | **131h** |

G1·G1b·G2의 8×H100 시간 풀은 24 + 96 + 144 = **264 node-hours**다. 따라서 이 가정에서 9B는 seeds까지 포함해 풀 안에 들어가고(약 50% 여유), 27B는 seeds 없이 1~3순위까지만 들어간다. 27B로 seeds를 채우려면 약 130 node-hours(약 $4,150)를 더하거나 곡선의 중간 규모(1,600)를 빼야 한다. 이 차이는 Task 2b 최종 선정의 판단 근거에 포함한다. 평가·프로파일 시간은 단일 PCIe 시간(76h)에 별도로 매핑하며, 개발용 폐루프 100 에피소드는 27B 기준 약 40분, 최종 시험 약 640 paired 에피소드 × 2 arm은 약 8시간으로 본다.

G1b와 G2는 사용할 수 있는 시간 풀이다. R과 L1/L0는 G1에서 D1 규모로 비교하고, 확대는 2×2의 결과가 나온 뒤 남은 풀에서 결정한다. 규칙 기반 판단기 기준군은 GPU를 쓰지 않는다. 본 비교의 training seed는 `17, 29, 43`을 제안하고, 가능한 학습량과 반복 수를 profile로 다시 정한다. GPU 시간 풀을 늘리지 않고 완료하지 못하면 결과 범위를 줄여 명시하며, 세 번 반복했다고 가정하지 않는다.

예비비 포함 GPU 비용 외에 CPU 데이터 생성 $300~900, 저장·전송 $200~400, 선택적 teacher 생성 $300~1,000을 **임시 지출 한도**로 잡으면 첫 사이클은 약 $11,600~13,100이다. CPU 데이터 생성에는 스크립트 전문가의 에피소드 실행(D1 400개, D2 4,000개), 키프레임 rollout(D1 128,000개 ≈ 178 모의 시간, D2 2,560,000개 ≈ 3,556 모의 시간), DAgger 재라벨링 에피소드를 포함하며, 첫 100개 rollout의 실측 속도로 이 항목을 다시 산정한다. 이 세 항목은 공급자 견적이 아닌 계획상 예산이며, 주석 인건비·로봇 장비·3D 카메라·세금은 제외한다. teacher를 쓰지 않는 데이터 제작도 가능하다.

27B 학습 장치를 8×B200으로 바꾸면 G2 학습 부분만 `144×53.52=$7,706.88`로 변한다. H100 가정의 `$4,596.48`보다 `$3,110.40` 많다. G0·G1·G1b까지 장치를 바꾸면 그 단계들도 별도로 다시 계산한다. 처리량이 높아 소요 시간이 줄 수 있지만 측정 전에는 그 절감을 미리 반영하지 않는다.

### 시간 추정 방법

실제 tokenize 결과의 토큰 수와 동일한 학습 경로의 steady-state 처리량을 사용한다. 상태 공유 버전과 복제 버전은 처리하는 토큰 수가 다르므로 **상태/초, 질문/초, 실제 계산 토큰/초를 모두 보고** 같은 데이터 학습 비용을 비교한다.

```text
학습 시간 ≈ (실제 학습 토큰 수 × epoch / 측정 tokens_per_second)
             + 평가 + checkpoint I/O + compile/시작 비용
GPU 비용 = 실제 node_hours × 해당 노드 시간 단가
```

예를 들어 D1의 로봇 400 에피소드는 120,000틱 × 600 token = 72M, 비로봇 2,000상태 × 4,000 token = 8M으로 1 epoch가 80M tokens다. 3 epoch는 240M이다. 8 GPU 노드 합계 5,000 tokens/s라는 가상 실측값을 넣으면 순수 학습 약 13.3시간이며 여기에 기타 비용이 붙는다. 스트림은 prefix를 에피소드당 한 번만 처리하고 틱의 새 토큰만 세지만, 무상태 L0 요청은 틱마다 prefix를 다시 세므로 같은 데이터라도 토큰이 약 2.3배다. 이 처리량은 관측값이 아닌 계산 예시다.

## 7. 저장·재개·종료 설정

원본 데이터·동결 checkpoint는 영속 저장소, 실행 중 임시 shard·cache는 로컬 SSD에 둔다. 언어 모델 약 27B의 BF16 가중치는 약 54GB이며, vision 등 포함 범위에 따른 실제 export 크기를 별도 집계한다. optimizer와 master state를 포함한 재개본은 구현에 따라 수백 GB다. 초기에 영속 2TB·노드 scratch 1TB 이상을 예산 대상으로 두고 실제 checkpoint 크기로 조정한다.

스토리지는 종료한 GPU와 수명이 다를 수 있다. 특히 Runpod의 container disk·volume·network storage의 보존 조건을 구분하고, 삭제될 위치에 유일한 checkpoint를 두지 않는다. [Runpod 저장소 문서](https://docs.runpod.io/pods/storage/types)

재개 저장은 PyTorch Distributed Checkpoint를 검토한다. 이 포맷도 다른 버전·다른 world size에서 자동으로 완벽히 재개된다고 가정하지 않고, 먼저 동일 구성 복구를 검증한다. [공식 checkpoint 문서](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)

첫 운영 설정은 30분마다 또는 500 optimizer step마다 먼저 도달하는 시점에 저장, 최근 재개본 2개와 best-dev 1개 보관이다. 업로드 후 해시·완료 marker를 확인한 다음 오래된 재개본을 정리한다. 업로드 중단·디스크 부족·손상된 checkpoint를 성공으로 표시하지 않는다.

run config에 `max_steps`, `max_wall_hours`, `estimated_hourly_usd`, `budget_usd`를 넣는다. 학습이 끝나거나 실패하면 로그·산출물을 동기화하고 GPU 종료를 요청하는 launcher를 만든다. 외부 watchdog이 비정상 종료와 종료 API 실패를 확인한다. 완료된 job과 남아 있는 유료 저장소를 별도 기록한다. 이 자동화 역시 구축 계획이며 현재 생성된 자원은 없다.

## 8. 장시간 학습 전 필수 검증

1. GPU 개수·VRAM·topology·driver와 container 정보를 manifest에 기록한다.
2. 후보 backbone마다 같은 요청 세트로 연속 틱 조건의 지연과 무학습 라벨 점수 읽기 품질을 측정하고, 지연 예산 산식과 함께 선정 근거를 기록한다.
3. 결정적 64상태 fixture에서 입력·라벨 분리, 세 타입의 loss, readout gradient를 확인한다.
4. 1 GPU에서 readout-only step, 8 GPU에서 text backbone full training step을 실행하고 모든 rank의 loss/gradient가 유효한지 확인한다. detached prefix·in-place 상태 공유로 분기 gradient가 손실되지 않는지도 검사한다.
5. 가장 긴 지원 요청으로 200 step을 실행해 메모리와 steady-state 처리량을 측정한다. compile 시간은 별도로 기록한다.
6. checkpoint를 저장한 뒤 프로세스를 종료·재시작한다. 같은 데이터 순서의 다음 step loss·가중치 업데이트를 중단 없는 기준 실행과 허용 오차 안에서 비교한다.
7. 저장소 업로드·복원, job 실패 후 비용 집계와 종료 처리를 시험한다.
8. 측정한 처리량으로 이번 run의 시간·비용을 계산해 예산 안의 `max_steps`를 확정한다.

단순히 첫 loss가 출력되었다고 학습 환경이 완료된 것으로 보지 않는다.

## 9. 평가 환경과 최종 판정

오프라인 평가는 ID·새 의미·새 후보 조합·새 분야를 분리하고 accuracy/허용 답 적중률, NLL, Brier, calibration, ordinal 오차를 기록한다. Q/K/길이별 품질도 보고한다. 라벨이 부족한 질문은 평가 분모에서 제외한 수를 공개한다.

지연은 cold start, compile, warm inference를 구분한다. 대표 셀은 warmup 후 새 상태 요청 1,000건 이상으로 p50/p95/p99와 95% 신뢰구간을 구한다. 나머지 stress 셀은 우선 200건으로 병목을 찾고, 핵심 주장을 할 셀은 충분한 표본으로 확대한다. CUDA event의 모델 시간과 wall-clock의 tokenize·mask·전송·반환 시간을 함께 기록한다. 로봇 조건에서는 연속 틱 측정을 추가한다. 같은 질문 세트로 개발용 에피소드 100개를 10Hz로 재생해 스트림 상태가 warm인 틱당 모델 시간과 관측→명령 적용 시간의 p50/p95/p99, 100ms deadline 초과율, 관측 deadline 폐기율, lease 만료율을 재고, L1-a 스트림·무상태 L0 요청과 후보 backbone을 같은 재생으로 비교한다.

로봇 폐루프는 우선 개발용 100개 paired seed로 실패율과 분산을 측정한다. 최종 독립 seed 수는 성공률 차이의 분산으로 산정하고 시험 전에 고정한다. −3%p 비열등 한계와 2배 p95 개선은 기존의 잠정 목표이며, 신뢰구간이 기준을 충족하지 못하면 결론을 유보한다. 난이도별로 성공, 충돌/제약 위반, 완료 시간, 관측 나이, stale 응답과 폐기, 컨트롤러 거절률, 혼합·정지 전이의 속도 불연속과 전환 구간 충돌, 그리퍼 이벤트 누락·중복·시각 오차, `q_stop` 지연·오경보, 재계획과 관측 추가의 빈도, 결정 전환율·왕복 전환·유지 시간, 목표 변경 후 반응 지연을 보고한다. 모든 폐루프 비교에는 규칙 기반 판단기 기준군을 같은 seed로 포함하고, 기하만으로 충분한 층과 의미 판단이 필요한 층을 나눠 보고한다.

후보 생성 시간·포함률, 하네스 조합 시간과 실행기 제한을 함께 기록한다. 후보를 잘 만들어 준 덕분의 개선과 모델 계산 구조의 개선을 분리한다. E0/E1/E2와 실제 로봇의 결과는 합쳐 한 성공률로 제시하지 않는다.
