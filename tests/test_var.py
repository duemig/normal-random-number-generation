import json
from pathlib import Path

import numpy as np
import pytest

from normgen.var import cholesky, delta_normal, load_inputs, value_change, var_es

DATA = Path(__file__).resolve().parents[1] / "data"


def test_cholesky_matches_numpy():
    inp = load_inputs(DATA / "var_inputs.json")
    L = cholesky(inp.cov)
    assert np.allclose(L, np.linalg.cholesky(inp.cov), rtol=0, atol=1e-15)
    assert np.allclose(L @ L.T, inp.cov, rtol=0, atol=1e-18)


def test_cholesky_rejects_indefinite_matrix():
    with pytest.raises(ValueError):
        cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_reproduces_spreadsheet_var_with_same_draws():
    inp = load_inputs(DATA / "var_inputs.json")
    ref = json.loads((DATA / "spreadsheet_draws.json").read_text())
    var, _ = var_es(value_change(inp, np.array(ref["z"])), inp.alpha)
    assert var == pytest.approx(ref["var_99_1d_eur"], abs=1e-6)


def test_var_es_on_known_sample():
    dv = -np.arange(1, 1001, dtype=float)  # losses 1..1000
    var, es = var_es(dv, 0.99)
    # k = 11: VaR is the 11th worst loss (990); ES is the mean of the 10 worst (995.5).
    assert var == pytest.approx(990.0) and es == pytest.approx(995.5)


def test_monte_carlo_close_to_delta_normal():
    inp = load_inputs(DATA / "var_inputs.json")
    z = np.random.default_rng(0).standard_normal((400_000, 4))
    var, es = var_es(value_change(inp, z), inp.alpha)
    dn_var, dn_es = delta_normal(inp)
    assert var == pytest.approx(dn_var, rel=0.02) and es == pytest.approx(dn_es, rel=0.02)
