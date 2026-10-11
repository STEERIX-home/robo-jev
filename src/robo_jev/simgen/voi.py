"""'다시 볼까' — 정보의 가치 (Task v0b 지시서 §1 P6).

E[비용 | 보기] = 보는 시간 × 지연 비용 + E_신호[ min_a E[c_a | 지금 관측, 신호] ]. 보는 동안 상황은 그대로라고 둔다(멈춰서 본다 — 정적 근사,
README). 사후 세계 K개(가중치 w_k)와 지금 가능한 행동(base)의 세계별 비용 C[b, k]가 있으면:

* 결정 규칙 a*(신호) = argmin_b Σ_k w_k·p(신호 | 세계 k)·C[b, k]  (본 뒤 갱신된 믿음에서의 최선 — 사후의 정답 세계로 정한다).
* 세계 k에서 보고 정했을 때의 기대 비용 = Σ_신호 p(신호 | k)·C[a*(신호), k] — 정답의 기대 비용은 이것의 사후 평균(+ 보는 시간).

신호가 이산이면(:class:`DiscreteSignal` — 다시 봤더니 '있다/없다', 두 센서의 새 보고, 무게 확인 값의 구간) 신호 위의 합이 정확하다(닫힌 꼴).
연속 측정을 다시 재면(:class:`GaussianSignal` — P4: 같은 참값을 새 잡음으로) 세계마다 Gauss–Hermite 마디로 적분한다(차원마다 7마디).

**참 세계의 실현 비용**: 결정 규칙은 정답의 사후에서 정하고(로봇이 실제로 아는 것), 그 규칙을 참 세계의 신호 분포에 적용한다 — 그래서
`truth.realized`가 정답에 대해 보정된다. 세계·관측 타입을 모른다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DiscreteSignal", "GaussianSignal", "gh_nodes"]


@dataclass
class DiscreteSignal:
    """이산 신호: likelihood[s, k] = P(신호 s | 세계 k) (열 합 1)."""

    base: tuple[str, ...]          # 보고 나서 고를 수 있는 행동(정식 열쇠)
    time: float                    # 보는 데 드는 시간 s
    likelihood: np.ndarray         # (S, K)
    option: str = "look_again"     # 다시 보기 선택지의 정식 열쇠(쌓기는 check_weight)

    def decide(self, costs: np.ndarray, weights: np.ndarray | None):
        """신호마다 최선 행동의 번호(base 안) — costs (B, K)."""
        w = np.full(costs.shape[1], 1.0 / costs.shape[1]) if weights is None else weights
        expected = (self.likelihood * w[None, :]) @ costs.T  # (S, B) — 정규화하지 않아도 argmin은 같다
        return np.argmin(expected, axis=1)

    def after(self, costs: np.ndarray, decision) -> np.ndarray:
        """세계마다 보고 정했을 때의 기대 비용 (K,) — 보는 시간은 빼고."""
        return np.sum(self.likelihood * costs[np.asarray(decision), :], axis=0)


_GH_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def gh_nodes(n: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """표준정규 N(0, 1)의 Gauss–Hermite 마디와 무게(합 1)."""
    if n not in _GH_CACHE:
        x, w = np.polynomial.hermite.hermgauss(n)
        _GH_CACHE[n] = (x * np.sqrt(2.0), w / np.sqrt(np.pi))
    return _GH_CACHE[n]


@dataclass
class GaussianSignal:
    """연속 신호: 세계 k에서 새 측정 m' ~ N(means[k], diag(sds[k]²)) (차원끼리 독립)."""

    base: tuple[str, ...]
    time: float
    means: np.ndarray              # (K, D)
    sds: np.ndarray                # (K, D)
    nodes: int = 7
    option: str = "look_again"

    def _density(self, points: np.ndarray) -> np.ndarray:
        """points (N, D) → (N, K) 세계마다의 밀도."""
        z = (points[:, None, :] - self.means[None, :, :]) / self.sds[None, :, :]
        return np.exp(-0.5 * np.sum(z * z, axis=2)) / np.prod(self.sds, axis=1)[None, :]

    def decide(self, costs: np.ndarray, weights: np.ndarray | None):
        """결정 규칙(함수): 새 측정 (N, D) → 최선 행동 번호 (N,)."""
        w = np.full(costs.shape[1], 1.0 / costs.shape[1]) if weights is None else weights

        def rule(points: np.ndarray) -> np.ndarray:
            dens = self._density(points) * w[None, :]   # (N, K)
            return np.argmin(dens @ costs.T, axis=1)    # (N, B)

        return rule

    def after(self, costs: np.ndarray, decision) -> np.ndarray:
        x, wx = gh_nodes(self.nodes)
        d = self.means.shape[1]
        grid = np.stack(np.meshgrid(*([x] * d), indexing="ij"), axis=-1).reshape(-1, d)
        weight = np.prod(np.stack(np.meshgrid(*([wx] * d), indexing="ij"), axis=-1).reshape(-1, d), axis=1)
        out = np.empty(self.means.shape[0])
        for k in range(self.means.shape[0]):
            points = self.means[k][None, :] + self.sds[k][None, :] * grid
            choice = decision(points)
            out[k] = float(np.sum(weight * costs[choice, k]))
        return out
