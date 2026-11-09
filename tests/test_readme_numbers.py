"""The numbers quoted in the README must match results/ (written by scripts/run_study.py)."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_matches_results():
    readme = (ROOT / "README.md").read_text()
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    rates = pd.read_csv(ROOT / "results" / "rejection_rates.csv", index_col=0).to_numpy()
    inv = pd.read_csv(ROOT / "results" / "inverse_cdf_accuracy.csv", index_col=0)
    conv = pd.read_csv(ROOT / "results" / "var_convergence.csv").set_index("n")
    var = s["var"]
    exact = [
        f"{100 * rates.min():.1f}% to {100 * rates.max():.1f}%",
        f"VaR is EUR {var['reference']['var']:,.0f}",
        f"ES is EUR {var['reference']['es']:,.0f}",
        f"standard deviation of EUR {conv.loc[500, 'VaR sd']:,.0f}",
        f"Over {s['repetitions']:,} samples of {s['sample_size']:,} draws",
    ]
    for text in exact:
        assert text in readme, text

    # The inverse-CDF accuracy is at the level of the last bit of a double, so its leading
    # digit shifts with the platform and numpy version; compare the quoted figure with the
    # results numerically, to within that rounding, rather than character by character.
    quoted = re.search(r"maximum relative error of ([0-9.]+e-[0-9]+)", readme)
    assert quoted, "inverse-CDF accuracy figure not found in README"
    assert float(quoted.group(1)) == pytest.approx(inv["max rel error"].max(), abs=5e-16)
