import numpy as np

from normgen.diagnostics import uniform_pvalues
from normgen.uniform import LCG, NumpyUniform


def test_first_state_from_seed_one():
    assert LCG(seed=1).step() == (1103515245 * 1 + 12345) % 2**32


def test_block_evaluation_matches_plain_recurrence():
    fast = LCG(seed=12345, block=1000)
    slow = LCG(seed=12345)
    z = fast.integers(3500)  # several blocks plus a partial one
    expected = [slow.step() for _ in range(3500)]
    assert z.tolist() == expected
    assert fast.state == slow.state


def test_uniforms_inside_open_interval():
    for src in (LCG(seed=0), NumpyUniform(seed=0)):
        u = src.random(200_000)
        assert u.min() > 0.0 and u.max() < 1.0


def test_hull_dobell_conditions():
    assert LCG().has_full_period()
    assert not LCG(a=1103515245, c=12346).has_full_period()  # c even: shares factor 2 with m


def test_lowest_bit_alternates():
    # With m = 2**32 and a, c odd, z[i+1] = z[i] + 1 (mod 2): the last bit has period 2.
    z = LCG(seed=7).integers(1000)
    bits = (z & np.uint64(1)).astype(int)
    assert np.all(bits[1:] != bits[:-1])


def test_lcg_passes_basic_uniformity_tests():
    p = uniform_pvalues(LCG(seed=1).random(100_000))
    assert min(p.values()) > 0.001
