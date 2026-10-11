"""검사 전용 감싸개 — 스크립트 하나를 transformers의 **torch 참조 커널**로 돌린다 (Task cloud1의 CPU 끝까지 검사).

CUDA를 숨긴 CPU에서 소형 난수 Qwen을 하위 프로세스로 돌리면 transformers가 import 시점에 fla(Triton) 커널을 골라 "0 active drivers"로
멈춘다. 같은 프로세스 검사는 :class:`robo_jev.model.backbone_qwen.torch_reference_kernels`로 그것을 바꾸는데, job의 단계는 하위
프로세스라 이 감싸개가 그 일을 한다::

    python tests/fixtures/launch/with_reference_kernels.py scripts/diagnose_decisions.py --model tiny …

실제 job(GPU)에는 쓰지 않는다.
"""

import runpy
import sys

from robo_jev.model.backbone_qwen import torch_reference_kernels

script = sys.argv[1]
sys.argv = sys.argv[1:]
with torch_reference_kernels():
    runpy.run_path(script, run_name="__main__")
