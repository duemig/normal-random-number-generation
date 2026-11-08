"""Inverse of the standard normal CDF.

Implementation of Wichura (1988), Algorithm AS 241 (PPND16). The algorithm uses three
rational approximations: one for the centre (|p - 0.5| <= 0.425) and two for the tails,
in terms of r = sqrt(-log(min(p, 1 - p))). Its relative accuracy is about 1e-16.
"""

from __future__ import annotations

import numpy as np

_A = (
    3.3871328727963666080e0,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
)
_B = (
    1.0,
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
)
_C = (
    1.42343711074968357734e0,
    4.63033784615654529590e0,
    5.76949722146069140550e0,
    3.64784832476320460504e0,
    1.27045825245236838258e0,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
)
_D = (
    1.0,
    2.05319162663775882187e0,
    1.67638483018380384940e0,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
)
_E = (
    6.65790464350110377720e0,
    5.46378491116411436990e0,
    1.78482653991729133580e0,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
)
_F = (
    1.0,
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
)


def _poly(coef: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    """Evaluate coef[0] + coef[1] x + ... + coef[7] x**7 with Horner's rule."""
    out = np.full_like(x, coef[-1])
    for c in coef[-2::-1]:
        out = out * x + c
    return out


def norm_ppf(p: np.ndarray | float) -> np.ndarray:
    """Return x with Phi(x) = p for p in (0, 1). Values outside (0, 1) give nan."""
    p = np.asarray(p, dtype=np.float64)
    x = np.full(p.shape, np.nan)
    q = p - 0.5

    central = np.abs(q) <= 0.425
    if np.any(central):
        qc = q[central]
        r = 0.180625 - qc * qc
        x[central] = qc * _poly(_A, r) / _poly(_B, r)

    tail = ~central & (p > 0.0) & (p < 1.0)
    if np.any(tail):
        qt = q[tail]
        r = np.where(qt < 0.0, p[tail], 1.0 - p[tail])
        r = np.sqrt(-np.log(r))
        xt = np.empty_like(r)
        near = r <= 5.0
        rn = r[near] - 1.6
        xt[near] = _poly(_C, rn) / _poly(_D, rn)
        rf = r[~near] - 5.0
        xt[~near] = _poly(_E, rf) / _poly(_F, rf)
        x[tail] = np.where(qt < 0.0, -xt, xt)

    return x
