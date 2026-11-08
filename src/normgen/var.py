"""Monte Carlo value at risk and expected shortfall for a two-ETF portfolio in EUR.

The portfolio holds EUR 1.5m in an S&P 500 ETF and EUR 1.0m in a FTSE 100 ETF. Its
one-day value change depends on four risk factors: the two index returns and the changes
in USD/EUR and GBP/EUR. The factor returns are modelled as multivariate normal with the
mean vector and covariance matrix in ``data/var_inputs.json``. Correlated returns are
obtained from independent standard normals with the Cholesky factor L of the covariance
matrix: r = mu + L z.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class VarInputs:
    factors: list[str]  # S&P 500, FTSE 100, USD/EUR, GBP/EUR
    position_spx: float  # EUR
    position_ftse: float  # EUR
    mean: np.ndarray  # daily mean returns, shape (4,)
    cov: np.ndarray  # daily covariance matrix, shape (4, 4)
    alpha: float

    @property
    def total(self) -> float:
        return self.position_spx + self.position_ftse


def load_inputs(path: str | Path) -> VarInputs:
    d = json.loads(Path(path).read_text())
    pos = d["positions_eur"]
    return VarInputs(
        factors=d["factors"],
        position_spx=float(pos["S&P 500 ETF"]),
        position_ftse=float(pos["FTSE 100 ETF"]),
        mean=np.array(d["mean_daily_return"], dtype=float),
        cov=np.array(d["covariance_daily"], dtype=float),
        alpha=float(d["alpha"]),
    )


def cholesky(a: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L L^T = a (Cholesky-Banachiewicz, row by row)."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("input must be a square matrix")
    L = np.zeros_like(a)
    for i in range(n):
        for j in range(i + 1):
            s = a[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0.0:
                    raise ValueError("matrix is not positive definite")
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def value_change(inp: VarInputs, z: np.ndarray) -> np.ndarray:
    """One-day value change in EUR for each row of standard normals z (shape (n, 4))."""
    r = inp.mean + np.asarray(z) @ cholesky(inp.cov).T
    spx, ftse, usd, gbp = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    return inp.position_spx * (1 + spx) * (1 + usd) + inp.position_ftse * (1 + ftse) * (1 + gbp) - inp.total


def var_es(dv: np.ndarray, alpha: float) -> tuple[float, float]:
    """Empirical VaR and ES of simulated value changes dv at confidence level alpha.

    With n scenarios, k = floor((1 - alpha) n + 1). VaR is minus the (k - 1) / (n - 1)
    percentile of dv (linear interpolation, as Excel's PERCENTILE). ES is minus the
    average of the (1 - alpha) n worst outcomes, where the k-th worst outcome gets the
    weight 1 + (1 - alpha) n - k so that the weights add up to (1 - alpha) n.
    """
    dv = np.sort(np.asarray(dv, dtype=float))
    n = dv.size
    tail = (1.0 - alpha) * n
    k = math.floor(tail + 1.0)
    var = -float(np.percentile(dv, 100.0 * (k - 1) / (n - 1)))
    w = np.zeros(n)
    w[: k - 1] = 1.0
    w[k - 1] = 1.0 + tail - k
    es = -float(np.sum(w * dv) / tail)
    return var, es


def delta_normal(inp: VarInputs, alpha: float | None = None) -> tuple[float, float]:
    """VaR and ES of the linearised value change, which is exactly normal.

    dV ~ w'r with exposures w = (1.5m, 1.0m, 1.5m, 1.0m): the products of returns are dropped.
    """
    alpha = inp.alpha if alpha is None else alpha
    w = np.array([inp.position_spx, inp.position_ftse, inp.position_spx, inp.position_ftse])
    m = float(w @ inp.mean)
    s = float(math.sqrt(w @ inp.cov @ w))
    q = stats.norm.ppf(alpha)
    var = q * s - m
    es = s * stats.norm.pdf(q) / (1.0 - alpha) - m
    return var, es
