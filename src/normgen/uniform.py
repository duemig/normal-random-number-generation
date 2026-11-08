"""Uniform pseudo-random numbers.

The normal samplers in this package only need a source of uniform numbers on (0, 1).
Two sources are provided:

* ``LCG``: a linear congruential generator, z[i+1] = (a z[i] + c) mod 2**32, written
  out here so that every step from integer recurrence to normal draw is visible.
* ``NumpyUniform``: numpy's PCG64 generator, used as a high-quality reference.
"""

from __future__ import annotations

from math import gcd

import numpy as np

M32 = 2**32


class LCG:
    """Linear congruential generator with modulus m = 2**32.

    The default constants a = 1103515245 and c = 12345 are the ones used in the example
    rand() implementation of the ANSI C standard. They satisfy the Hull-Dobell conditions,
    so the generator has the full period 2**32 for every seed.

    The recurrence is sequential, but it can be evaluated in blocks: for k = 1, ..., B,

        z[i + k] = (A_k z[i] + C_k) mod m,  A_k = a**k mod m,  C_k = c (a**(k-1) + ... + 1) mod m,

    so a whole block is one vectorised multiply-add. The result is identical to running
    the recurrence one step at a time.

    Uniforms are returned as (z + 0.5) / m, which lies strictly inside (0, 1), so that
    log(u) and the inverse normal CDF are always finite.
    """

    def __init__(self, seed: int = 1, a: int = 1103515245, c: int = 12345, block: int = 65536):
        self.a, self.c, self.m = int(a), int(c), M32
        self.state = int(seed) % self.m
        self._A = np.empty(block, dtype=np.uint64)
        self._C = np.empty(block, dtype=np.uint64)
        ak, ck = 1, 0
        for k in range(block):
            ak = (ak * self.a) % self.m
            ck = (ck * self.a + self.c) % self.m
            self._A[k] = ak
            self._C[k] = ck

    def has_full_period(self) -> bool:
        """Check the Hull-Dobell conditions for m = 2**32.

        (i) c and m are coprime, (ii) a - 1 is divisible by every prime factor of m (here 2),
        (iii) a - 1 is divisible by 4 because m is divisible by 4.
        """
        return gcd(self.c, self.m) == 1 and (self.a - 1) % 2 == 0 and (self.a - 1) % 4 == 0

    def step(self) -> int:
        """Advance one step with plain integer arithmetic and return the new state."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def integers(self, n: int) -> np.ndarray:
        """Return the next n states z[i + 1], ..., z[i + n] as uint64."""
        out = np.empty(n, dtype=np.uint64)
        mask = np.uint64(self.m - 1)
        z = np.uint64(self.state)
        pos = 0
        while pos < n:
            b = min(len(self._A), n - pos)
            # A_k < 2**32 and z < 2**32, so A_k * z + C_k < 2**64: no overflow in uint64.
            block = (self._A[:b] * z + self._C[:b]) & mask
            out[pos : pos + b] = block
            z = block[-1]
            pos += b
        self.state = int(z)
        return out

    def random(self, n: int) -> np.ndarray:
        """Return n uniforms on (0, 1)."""
        return (self.integers(n).astype(np.float64) + 0.5) / self.m


class NumpyUniform:
    """Uniforms on (0, 1) from numpy's PCG64 generator, with 52 random bits each.

    (k + 0.5) / 2**52 with k < 2**52 is exact in double precision, so 0 and 1 never occur.
    """

    def __init__(self, seed: int = 1):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def random(self, n: int) -> np.ndarray:
        k = self.rng.integers(0, 2**52, size=n, dtype=np.int64)
        return (k.astype(np.float64) + 0.5) / 2.0**52
