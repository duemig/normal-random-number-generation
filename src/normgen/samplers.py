"""Four ways to turn uniform numbers into normal numbers.

Every sampler takes a uniform source (any object with a ``random(n)`` method returning
uniforms on (0, 1)), the number of draws n, and the target mean and standard deviation.
All of them are vectorised: uniforms are drawn in arrays, and the two rejection methods
draw a batch, keep the accepted values and repeat until n values are available.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from .inverse_cdf import norm_ppf


class UniformSource(Protocol):
    def random(self, n: int) -> np.ndarray: ...


#: Envelope constant of the rejection sampler, c = sqrt(2e / pi) ~ 1.3155.
REJECTION_C = float(np.sqrt(2.0 * np.e / np.pi))


def inverse_transform(uniform: UniformSource, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """X = mu + sigma * Phi^{-1}(U), with Phi^{-1} from Algorithm AS 241."""
    return mu + sigma * norm_ppf(uniform.random(n))


def rejection(uniform: UniformSource, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Acceptance-rejection for |Z| with an exponential(1) proposal and a random sign.

    Each proposal uses three uniforms: Y = -log(U1), accept if U2 <= exp(-(Y - 1)**2 / 2),
    sign from U3. The acceptance probability is 1 / c ~ 0.760.
    """
    out = np.empty(n)
    filled = 0
    while filled < n:
        need = n - filled
        m = int(need * REJECTION_C * 1.1) + 16
        u = uniform.random(3 * m)
        y = -np.log(u[0::3])
        accept = u[1::3] <= np.exp(-0.5 * (y - 1.0) ** 2)
        sign = np.where(u[2::3] <= 0.5, 1.0, -1.0)
        z = (sign * y)[accept]
        take = min(z.size, need)
        out[filled : filled + take] = z[:take]
        filled += take
    return mu + sigma * out


def box_muller(uniform: UniformSource, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Box-Muller: each pair (U1, U2) gives two independent normals."""
    pairs = (n + 1) // 2
    u = uniform.random(2 * pairs)
    radius = np.sqrt(-2.0 * np.log(u[0::2]))
    angle = 2.0 * np.pi * u[1::2]
    z = np.empty(2 * pairs)
    z[0::2] = radius * np.cos(angle)
    z[1::2] = radius * np.sin(angle)
    return mu + sigma * z[:n]


def polar(uniform: UniformSource, n: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Marsaglia's polar method: Box-Muller without sine and cosine.

    Draw V1, V2 uniform on (-1, 1) and keep the pair if S = V1**2 + V2**2 < 1 (probability
    pi / 4 ~ 0.785). Then V1 * sqrt(-2 log(S) / S) and V2 * sqrt(-2 log(S) / S) are
    independent standard normals.
    """
    out = np.empty(n)
    filled = 0
    while filled < n:
        need = n - filled
        pairs = int(need / 2 / (np.pi / 4) * 1.1) + 16
        u = uniform.random(2 * pairs)
        v1 = 2.0 * u[0::2] - 1.0
        v2 = 2.0 * u[1::2] - 1.0
        s = v1 * v1 + v2 * v2
        ok = (s > 0.0) & (s < 1.0)
        f = np.sqrt(-2.0 * np.log(s[ok]) / s[ok])
        z = np.empty(2 * int(ok.sum()))
        z[0::2] = v1[ok] * f
        z[1::2] = v2[ok] * f
        take = min(z.size, need)
        out[filled : filled + take] = z[:take]
        filled += take
    return mu + sigma * out


Sampler = Callable[..., np.ndarray]

SAMPLERS: dict[str, Sampler] = {
    "Inverse transform": inverse_transform,
    "Rejection sampling": rejection,
    "Box-Muller": box_muller,
    "Polar method": polar,
}
