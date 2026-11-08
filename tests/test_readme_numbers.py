"""The numbers quoted in the README must match results/ (written by scripts/run_study.py)."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_readme_matches_results():
    readme = (ROOT / "README.md").read_text()
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    rates = pd.read_csv(ROOT / "results" / "rejection_rates.csv", index_col=0).to_numpy()
    inv = pd.read_csv(ROOT / "results" / "inverse_cdf_accuracy.csv", index_col=0)
    conv = pd.read_csv(ROOT / "results" / "var_convergence.csv").set_index("n")
    var = s["var"]
    expected = [
        f"{inv['max rel error'].max():.1e}".replace("e-", "e-"),
        f"{100 * rates.min():.1f}% to {100 * rates.max():.1f}%",
        f"VaR is EUR {var['reference']['var']:,.0f}",
        f"ES is EUR {var['reference']['es']:,.0f}",
        f"standard deviation of EUR {conv.loc[500, 'VaR sd']:,.0f}",
        f"Over {s['repetitions']:,} samples of {s['sample_size']:,} draws",
    ]
    for text in expected:
        assert text in readme, text
