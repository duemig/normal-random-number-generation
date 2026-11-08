"""Build report.pdf from report/report.tex and the files in results/.

Every number in the report comes from results/ (written by scripts/run_study.py): this
script turns them into LaTeX macros and tables in report/generated/, then runs XeLaTeX.

Usage: python scripts/build_report.py [--no-pdf]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
GEN = REPORT / "generated"

METHODS = ["Inverse transform", "Rejection sampling", "Box-Muller", "Polar method", "NumPy (PCG64, ziggurat)"]
SHORT = ["Inverse transform", "Rejection", "Box-Muller", "Polar", "NumPy"]
TESTS = ["Kolmogorov-Smirnov", "Anderson-Darling", "Shapiro-Wilk", "D'Agostino K2", "Jarque-Bera"]
TEST_LABEL = {"D'Agostino K2": "D'Agostino $K^2$"}


def p_fmt(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def eur(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def num(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}".replace(",", "{,}")


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def table(colspec: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true", help="only write report/generated/")
    args = ap.parse_args()
    GEN.mkdir(parents=True, exist_ok=True)

    s = json.loads((RESULTS / "summary.json").read_text())
    single = pd.read_csv(RESULTS / "single_sample.csv", index_col=0)
    rates = pd.read_csv(RESULTS / "rejection_rates.csv", index_col=0)
    urates = pd.read_csv(RESULTS / "uniform_rejection_rates.csv", index_col=0)
    tails = pd.read_csv(RESULTS / "tail_counts.csv", index_col=0)
    inv = pd.read_csv(RESULTS / "inverse_cdf_accuracy.csv", index_col=0)
    speed = pd.read_csv(RESULTS / "speed.csv")
    conv = pd.read_csv(RESULTS / "var_convergence.csv").set_index("n")
    var = s["var"]
    acc = s["acceptance"]

    reps = s["repetitions"]
    se = (0.05 * 0.95 / reps) ** 0.5
    all_rates = rates.to_numpy().ravel()
    sp = speed.set_index(["method", "uniform source"])["million draws per second"]

    macros = {
        "SampleSize": num(s["sample_size"]),
        "Reps": num(reps),
        "RateSE": f"{100 * se:.1f}",
        "RateMin": pct(all_rates.min()),
        "RateMax": pct(all_rates.max()),
        "RateMaxZ": f"{abs(all_rates - 0.05).max() / se:.1f}",
        "TailDraws": f"$10^{{{len(str(int(tails['draws'].iloc[0]))) - 1}}}$",
        "TailExpFour": num(tails["expected |z|>4"].iloc[0]),
        "TailExpFive": num(tails["expected |z|>5"].iloc[0], 1),
        "TailMinP": f"{tails['p-value |z|>5'].min():.2f}",
        "TailMaxAbsZ": f"{tails['z-score |z|>4'].abs().max():.1f}",
        "InvMaxRel": sci(inv["max rel error"].max()),
        "AccRej": f"{acc['rejection_measured']:.3f}",
        "AccRejTheory": f"{acc['rejection_theory']:.3f}",
        "AccPolar": f"{acc['polar_measured']:.3f}",
        "AccPolarTheory": f"{acc['polar_theory']:.3f}",
        "UniPerDrawRej": f"{acc['uniforms_per_draw']['Rejection sampling']:.2f}",
        "UniPerDrawPolar": f"{acc['uniforms_per_draw']['Polar method']:.2f}",
        "SpeedNumpy": f"{sp[('NumPy (PCG64, ziggurat)', 'PCG64')]:.0f}",
        "SpeedPolarLCG": f"{sp[('Polar method', 'LCG')]:.0f}",
        "SpeedInvLCG": f"{sp[('Inverse transform', 'LCG')]:.0f}",
        "CPU": s["cpu"].replace("(R)", "").replace("@", "at"),
        "UniPKS": f"{s['uniform']['single_sample_pvalues']['Kolmogorov-Smirnov']:.2f}",
        "UniPChi": f"{s['uniform']['single_sample_pvalues']['Chi-square (100 bins)']:.2f}",
        "UniPPairs": f"{s['uniform']['single_sample_pvalues']['Chi-square on pairs (20 x 20)']:.2f}",
        "VaRSheet": eur(var["spreadsheet_draws"]["var"]),
        "ESSheetReported": eur(var["spreadsheet_draws"]["es_in_spreadsheet"]),
        "VaRSDPct": f"{100 * conv.loc[500, 'VaR sd'] / var['reference']['var']:.0f}",
        "ESSheet": eur(var["spreadsheet_draws"]["es"]),
        "VaRRef": eur(var["reference"]["var"]),
        "ESRef": eur(var["reference"]["es"]),
        "VaRDN": eur(var["delta_normal"]["var"]),
        "ESDN": eur(var["delta_normal"]["es"]),
        "RefDraws": num(var["reference"]["n"] / 1e6) + " million",
        "VaRRankSheet": f"{100 * var['n500_rank_of_spreadsheet_var']:.0f}",
        "ConvReps": num(var["convergence_reps"]),
        "VaRSDFiveHundred": eur(conv.loc[500, "VaR sd"]),
        "VaRMeanFiveHundred": eur(conv.loc[500, "VaR mean"]),
        "VaRLoFiveHundred": eur(conv.loc[500, "VaR q05"]),
        "VaRHiFiveHundred": eur(conv.loc[500, "VaR q95"]),
        "ESLoFiveHundred": eur(conv.loc[500, "ES q05"]),
        "ESHiFiveHundred": eur(conv.loc[500, "ES q95"]),
        "VaRSDHundredK": eur(conv.loc[100_000, "VaR sd"]),
        "VaRSDTenK": eur(conv.loc[10_000, "VaR sd"]),
        "VaRBiasPct": f"{100 * (1 - conv.loc[500, 'VaR mean'] / var['reference']['var']):.1f}",
    }
    tex = ["% Generated by scripts/build_report.py from results/. Do not edit by hand."]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (GEN / "numbers.tex").write_text("\n".join(tex) + "\n")

    # Table: uniform generator
    tests_u = list(urates.index)
    rows = [
        [t, f"{s['uniform']['single_sample_pvalues'][t]:.3f}", pct(urates.loc[t, "LCG"]), pct(urates.loc[t, "PCG64"])]
        for t in tests_u
    ]
    (GEN / "table_uniform.tex").write_text(
        table("lccc", ["Test", "p-value, LCG", "Rejected at 5\\%, LCG", "Rejected at 5\\%, PCG64"], rows)
    )

    # Table: inverse CDF accuracy
    rows = [
        [
            r.replace("<=", "$\\le$")
            .replace("<", "$<$")
            .replace("1e-11", "$10^{-11}$")
            .replace("1e-300", "$10^{-300}$"),
            sci(inv.loc[r, "max abs error"]),
            sci(inv.loc[r, "max rel error"]),
        ]
        for r in inv.index
    ]
    (GEN / "table_inverse_cdf.tex").write_text(
        table("lcc", ["Range of $p$", "Max. absolute error", "Max. relative error"], rows)
    )

    # Table: one sample per method
    rows = []
    for stat, label in [
        ("mean", "Mean"),
        ("std", "Standard deviation"),
        ("skewness", "Skewness"),
        ("excess kurtosis", "Excess kurtosis"),
    ]:
        rows.append([label] + [f"{single.loc[m, stat]:.3f}" for m in METHODS])
    rows.append(["\\multicolumn{6}{l}{\\textit{p-values}}"])
    for t in TESTS:
        rows.append([TEST_LABEL.get(t, t)] + [p_fmt(single.loc[m, t]) for m in METHODS])
    (GEN / "table_single.tex").write_text(table("l" + "c" * len(METHODS), ["", *SHORT], rows))

    # Table: rejection rates
    rows = [[TEST_LABEL.get(t, t)] + [pct(rates.loc[m, t]) for m in METHODS] for t in TESTS]
    (GEN / "table_rejection.tex").write_text(table("l" + "c" * len(METHODS), ["Test", *SHORT], rows))

    # Table: tails
    rows = [
        [sh, num(tails.loc[m, "count |z|>4"]), num(tails.loc[m, "count |z|>5"]), f"{tails.loc[m, 'p-value |z|>5']:.2f}"]
        for m, sh in zip(METHODS, SHORT, strict=True)
    ]
    rows.append(["Expected", num(tails["expected |z|>4"].iloc[0]), num(tails["expected |z|>5"].iloc[0], 1), ""])
    (GEN / "table_tails.tex").write_text(
        table("lccc", ["Method", "$|Z| > 4$", "$|Z| > 5$", "p-value, $|Z| > 5$"], rows)
    )

    # Table: speed
    upd = acc["uniforms_per_draw"]
    rows = [["Uniforms only", f"{sp[('Uniforms only', 'LCG')]:.0f}", f"{sp[('Uniforms only', 'PCG64')]:.0f}", ""]]
    for m in ["Inverse transform", "Rejection sampling", "Box-Muller", "Polar method"]:
        rows.append([m, f"{sp[(m, 'LCG')]:.0f}", f"{sp[(m, 'PCG64')]:.0f}", f"{upd[m]:.2f}"])
    rows.append(["NumPy \\texttt{standard\\_normal}", "", f"{sp[('NumPy (PCG64, ziggurat)', 'PCG64')]:.0f}", ""])
    (GEN / "table_speed.tex").write_text(table("lccc", ["Method", "LCG", "PCG64", "Uniforms per draw"], rows))

    # Table: VaR and ES
    rows = [
        [
            "Monte Carlo, the spreadsheet's 500 draws",
            eur(var["spreadsheet_draws"]["var"]),
            eur(var["spreadsheet_draws"]["es"]),
        ],
        [
            f"Monte Carlo, 500 draws, 5\\% to 95\\% range over {var['convergence_reps']} runs",
            f"{eur(conv.loc[500, 'VaR q05'])} to {eur(conv.loc[500, 'VaR q95'])}",
            f"{eur(conv.loc[500, 'ES q05'])} to {eur(conv.loc[500, 'ES q95'])}",
        ],
        [
            f"Monte Carlo, {num(var['reference']['n'] / 1e6)} million draws",
            eur(var["reference"]["var"]),
            eur(var["reference"]["es"]),
        ],
        ["Delta-normal (linearised value change)", eur(var["delta_normal"]["var"]), eur(var["delta_normal"]["es"])],
    ]
    (GEN / "table_var.tex").write_text(table("lcc", ["", "VaR (EUR)", "ES (EUR)"], rows))

    if args.no_pdf:
        return
    for _ in range(2):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
            cwd=REPORT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    shutil.copy(REPORT / "report.pdf", ROOT / "report.pdf")
    for ext in ("aux", "log", "out", "pdf"):
        (REPORT / f"report.{ext}").unlink(missing_ok=True)
    print("wrote report.pdf")


if __name__ == "__main__":
    main()
