import numpy as np
import pytest
from scipy import special, stats

from normgen.diagnostics import _ad_inf, ad_pvalue, ad_statistic, normality_pvalues
from normgen.inverse_cdf import norm_ppf
from normgen.samplers import REJECTION_C, SAMPLERS
from normgen.uniform import LCG, NumpyUniform


def test_inverse_cdf_matches_scipy():
    p = np.concatenate(
        [np.logspace(-300, -1, 5000), np.linspace(1e-4, 1 - 1e-4, 50001), 1 - np.logspace(-15, -1, 5000)]
    )
    x, ref = norm_ppf(p), special.ndtri(p)
    assert np.max(np.abs(x - ref) / np.maximum(np.abs(ref), 1.0)) < 1e-14


def test_inverse_cdf_edges():
    assert norm_ppf(0.5) == 0.0
    assert np.isnan(norm_ppf(0.0)) and np.isnan(norm_ppf(1.0))
    assert norm_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-15)


def test_rejection_constant():
    assert REJECTION_C == pytest.approx(1.3155, abs=1e-4)


@pytest.mark.parametrize("name", list(SAMPLERS))
@pytest.mark.parametrize("source", [LCG, NumpyUniform])
def test_samplers_produce_standard_normals(name, source):
    z = SAMPLERS[name](source(seed=3), 50_000)
    assert z.shape == (50_000,)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03
    assert stats.kstest(z, "norm").pvalue > 0.001


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_location_and_scale(name):
    x = SAMPLERS[name](NumpyUniform(seed=4), 50_000, mu=2.0, sigma=3.0)
    assert x.mean() == pytest.approx(2.0, abs=0.1) and x.std() == pytest.approx(3.0, abs=0.1)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_samplers_are_reproducible(name):
    a = SAMPLERS[name](LCG(seed=11), 1001)
    b = SAMPLERS[name](LCG(seed=11), 1001)
    assert np.array_equal(a, b)


def test_ad_limiting_distribution_matches_known_critical_values():
    # Critical values of A^2 for a fully specified distribution: 10%, 5%, 1%.
    assert _ad_inf(1.933) == pytest.approx(0.90, abs=1e-3)
    assert _ad_inf(2.492) == pytest.approx(0.95, abs=1e-3)
    assert _ad_inf(3.857) == pytest.approx(0.99, abs=1e-3)


def test_ad_rejects_shifted_sample():
    x = np.random.default_rng(0).standard_normal(2000) + 0.2
    assert ad_pvalue(ad_statistic(x), x.size) < 1e-4


def test_normality_tests_reject_uniform_data():
    p = normality_pvalues(np.random.default_rng(1).uniform(-2, 2, 5000))
    assert max(p.values()) < 1e-6
