"""Statistical tests for the uniform generator and for the normal samples.

Normality tests (null hypothesis: the sample is standard normal, or normal for the
moment- and correlation-based tests):

* Kolmogorov-Smirnov, two-sided, against N(0, 1)
* Anderson-Darling against N(0, 1), with the p-value of Marsaglia and Marsaglia (2004)
* Shapiro-Wilk
* D'Agostino-Pearson K^2
* Jarque-Bera

Uniformity tests: chi-square on equal-width bins, two-sided Kolmogorov-Smirnov, and a
chi-square test on non-overlapping pairs (u[2i], u[2i+1]) on a 2D grid.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

# ----------------------------------------------------------------------------------
# Anderson-Darling test against a fully specified N(0, 1)
# ----------------------------------------------------------------------------------


def ad_statistic(x: np.ndarray) -> float:
    """A^2 = -n - (1/n) sum_i (2i - 1) [log Phi(x_(i)) + log(1 - Phi(x_(n+1-i)))]."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    i = np.arange(1, n + 1)
    log_cdf = special.log_ndtr(x)
    log_sf_rev = special.log_ndtr(-x[::-1])
    return float(-n - np.sum((2 * i - 1) * (log_cdf + log_sf_rev)) / n)


def _ad_inf(z: float) -> float:
    """Limiting distribution P(A^2 < z) for n -> infinity (Marsaglia and Marsaglia, 2004)."""
    if z <= 0.0:
        return 0.0
    if z < 2.0:
        return (
            math.exp(-1.2337141 / z)
            / math.sqrt(z)
            * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
        )
    return math.exp(-math.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))


def _ad_errfix(n: int, x: float) -> float:
    """Finite-sample correction to the limiting distribution (Marsaglia and Marsaglia, 2004)."""
    if x > 0.8:
        return (-130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * x) * x) * x) * x) * x) / n
    c = 0.01265 + 0.1757 / n
    if x < c:
        t = x / c
        t = math.sqrt(t) * (1.0 - t) * (49.0 * t - 102.0)
        return t * (0.0037 / (n * n) + 0.00078 / n + 0.00006) / n
    t = (x - c) / (0.8 - c)
    t = -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t
    return t * (0.04213 / n + 0.01365 / n / n)


def ad_pvalue(a2: float, n: int) -> float:
    """P-value of the Anderson-Darling statistic for a fully specified null distribution."""
    cdf = _ad_inf(a2)
    cdf += _ad_errfix(n, cdf)
    return float(min(max(1.0 - cdf, 0.0), 1.0))


# ----------------------------------------------------------------------------------
# Normality tests
# ----------------------------------------------------------------------------------

NORMALITY_TESTS = ("Kolmogorov-Smirnov", "Anderson-Darling", "Shapiro-Wilk", "D'Agostino K2", "Jarque-Bera")


def normality_pvalues(z: np.ndarray) -> dict[str, float]:
    """P-values of the five normality tests for one sample."""
    z = np.asarray(z, dtype=np.float64)
    return {
        "Kolmogorov-Smirnov": float(stats.kstest(z, "norm").pvalue),
        "Anderson-Darling": ad_pvalue(ad_statistic(z), z.size),
        "Shapiro-Wilk": float(stats.shapiro(z).pvalue),
        "D'Agostino K2": float(stats.normaltest(z).pvalue),
        "Jarque-Bera": float(stats.jarque_bera(z).pvalue),
    }


def sample_moments(z: np.ndarray) -> dict[str, float]:
    z = np.asarray(z, dtype=np.float64)
    return {
        "mean": float(z.mean()),
        "std": float(z.std(ddof=1)),
        "skewness": float(stats.skew(z)),
        "excess kurtosis": float(stats.kurtosis(z)),
    }


# ----------------------------------------------------------------------------------
# Uniformity tests
# ----------------------------------------------------------------------------------


def uniform_pvalues(u: np.ndarray, bins: int = 100, grid: int = 20) -> dict[str, float]:
    """P-values of three uniformity tests for one sample of uniforms."""
    u = np.asarray(u, dtype=np.float64)
    counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    pairs = u[: 2 * (u.size // 2)].reshape(-1, 2)
    counts2d, _, _ = np.histogram2d(pairs[:, 0], pairs[:, 1], bins=grid, range=[[0, 1], [0, 1]])
    return {
        "Chi-square (100 bins)": float(stats.chisquare(counts).pvalue),
        "Kolmogorov-Smirnov": float(stats.kstest(u, "uniform").pvalue),
        "Chi-square on pairs (20 x 20)": float(stats.chisquare(counts2d.ravel()).pvalue),
    }
