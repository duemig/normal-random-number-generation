"""Run every computation behind the report and write results/ and figures/.

Usage: python scripts/run_study.py [--quick]

--quick uses fewer repetitions and draws (for a fast check); the report uses the full run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 12,
    }
)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import special, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from normgen.diagnostics import NORMALITY_TESTS, normality_pvalues, sample_moments, uniform_pvalues  # noqa: E402
from normgen.inverse_cdf import norm_ppf  # noqa: E402
from normgen.samplers import REJECTION_C, SAMPLERS  # noqa: E402
from normgen.uniform import LCG, NumpyUniform  # noqa: E402
from normgen.var import delta_normal, load_inputs, value_change, var_es  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_SAMPLE = 5000
SEED = 1
REFERENCE = "NumPy (PCG64, ziggurat)"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def numpy_reference(seed: int):
    rng = np.random.default_rng(seed)
    return lambda n: rng.standard_normal(n)


def _bind(sampler, source):
    return lambda n: sampler(source, n)


def generators(seed: int) -> dict:
    """Each method draws from its own LCG stream started at the same seed."""
    gens = {name: _bind(f, LCG(seed=seed)) for name, f in SAMPLERS.items()}
    gens[REFERENCE] = numpy_reference(seed)
    return gens


# ----------------------------------------------------------------------------------


def uniform_section(reps: int) -> dict:
    lcg = LCG(seed=SEED)
    single = uniform_pvalues(lcg.random(N_SAMPLE))
    rates = {}
    for label, src in (("LCG", LCG(seed=SEED)), ("PCG64", NumpyUniform(seed=SEED))):
        p = pd.DataFrame([uniform_pvalues(src.random(N_SAMPLE)) for _ in range(reps)])
        rates[label] = (p < 0.05).mean().to_dict()
    z = LCG(seed=SEED).integers(10_000)
    bits = (z & np.uint64(1)).astype(int)
    out = {
        "full_period": LCG().has_full_period(),
        "single_sample_pvalues": single,
        "rejection_rate_5pct": rates,
        "lowest_bit_alternates": bool(np.all(bits[1:] != bits[:-1])),
        "reps": reps,
    }
    pd.DataFrame(rates).to_csv(RESULTS / "uniform_rejection_rates.csv")
    return out


def single_sample_section() -> tuple[pd.DataFrame, dict]:
    rows, samples = [], {}
    for name, gen in generators(SEED).items():
        z = gen(N_SAMPLE)
        samples[name] = z
        rows.append({"method": name, **sample_moments(z), **normality_pvalues(z)})
    df = pd.DataFrame(rows).set_index("method")
    df.to_csv(RESULTS / "single_sample.csv")
    return df, samples


def repeated_section(reps: int) -> pd.DataFrame:
    rates = {}
    for name, gen in generators(SEED).items():
        p = pd.DataFrame([normality_pvalues(gen(N_SAMPLE)) for _ in range(reps)])
        rates[name] = (p < 0.05).mean()
    df = pd.DataFrame(rates).T[list(NORMALITY_TESTS)]
    df.to_csv(RESULTS / "rejection_rates.csv")
    return df


def tail_section(n_total: int, chunk: int = 10_000_000) -> pd.DataFrame:
    expected4 = n_total * 2 * stats.norm.sf(4.0)
    expected5 = n_total * 2 * stats.norm.sf(5.0)
    rows = []
    for name, gen in generators(SEED).items():
        c4 = c5 = 0
        done = 0
        while done < n_total:
            z = gen(min(chunk, n_total - done))
            a = np.abs(z)
            c4 += int((a > 4.0).sum())
            c5 += int((a > 5.0).sum())
            done += z.size
        rows.append(
            {
                "method": name,
                "draws": n_total,
                "count |z|>4": c4,
                "expected |z|>4": expected4,
                "z-score |z|>4": (c4 - expected4) / np.sqrt(expected4),
                "count |z|>5": c5,
                "expected |z|>5": expected5,
                "p-value |z|>5": float(
                    min(1.0, 2 * min(stats.poisson.cdf(c5, expected5), stats.poisson.sf(c5 - 1, expected5)))
                ),
            }
        )
    df = pd.DataFrame(rows).set_index("method")
    df.to_csv(RESULTS / "tail_counts.csv")
    return df


def inverse_cdf_section() -> pd.DataFrame:
    regions = {
        "Centre, |p - 0.5| <= 0.425": np.linspace(0.075, 0.925, 1_000_001),
        "Tail, 1e-11 < p < 0.075": np.logspace(-11, np.log10(0.075), 1_000_001),
        "Far tail, 1e-300 < p < 1e-11": np.logspace(-300, -11, 1_000_001),
    }
    rows = []
    for label, p in regions.items():
        x, ref = norm_ppf(p), special.ndtri(p)
        rows.append(
            {
                "region": label,
                "max abs error": float(np.max(np.abs(x - ref))),
                "max rel error": float(np.max(np.abs(x - ref) / np.maximum(np.abs(ref), 1e-300))),
            }
        )
    df = pd.DataFrame(rows).set_index("region")
    df.to_csv(RESULTS / "inverse_cdf_accuracy.csv")
    return df


def _best_time(fn, repeat: int) -> float:
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def speed_section(n: int, repeat: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    for src_label, src_cls in (("LCG", LCG), ("PCG64", NumpyUniform)):
        src = src_cls(seed=SEED)
        t = _best_time(partial(src.random, n), repeat)
        rows.append({"method": "Uniforms only", "uniform source": src_label, "million draws per second": n / t / 1e6})
        for name, f in SAMPLERS.items():
            src = src_cls(seed=SEED)
            t = _best_time(partial(f, src, n), repeat)
            rows.append({"method": name, "uniform source": src_label, "million draws per second": n / t / 1e6})
    rng = np.random.default_rng(SEED)
    t = _best_time(partial(rng.standard_normal, n), repeat)
    rows.append({"method": REFERENCE, "uniform source": "PCG64", "million draws per second": n / t / 1e6})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "speed.csv", index=False)

    # Acceptance rates and uniforms used per normal draw, measured on 10**6 proposals.
    u = NumpyUniform(seed=SEED).random(3 * 1_000_000)
    y = -np.log(u[0::3])
    acc_rej = float(np.mean(u[1::3] <= np.exp(-0.5 * (y - 1.0) ** 2)))
    v = 2 * NumpyUniform(seed=SEED + 1).random(2 * 1_000_000) - 1
    acc_pol = float(np.mean(v[0::2] ** 2 + v[1::2] ** 2 < 1.0))
    acceptance = {
        "rejection_measured": acc_rej,
        "rejection_theory": 1 / REJECTION_C,
        "polar_measured": acc_pol,
        "polar_theory": np.pi / 4,
        "uniforms_per_draw": {
            "Inverse transform": 1.0,
            "Rejection sampling": 3 * REJECTION_C,
            "Box-Muller": 1.0,
            "Polar method": 4 / np.pi,
        },
    }
    return df, acceptance


def var_section(n_ref: int, ns: list[int], reps: int) -> tuple[dict, pd.DataFrame, np.ndarray]:
    inp = load_inputs(ROOT / "data" / "var_inputs.json")
    ref = json.loads((ROOT / "data" / "spreadsheet_draws.json").read_text())
    sheet_var, sheet_es = var_es(value_change(inp, np.array(ref["z"])), inp.alpha)

    def normals(src, n):
        return norm_ppf(src.random(4 * n)).reshape(n, 4)

    src = NumpyUniform(seed=SEED)
    dv_ref = np.concatenate([value_change(inp, normals(src, 1_000_000)) for _ in range(n_ref // 1_000_000)])
    ref_var, ref_es = var_es(dv_ref, inp.alpha)
    dn_var, dn_es = delta_normal(inp)

    rows = []
    src = NumpyUniform(seed=SEED + 100)
    est500 = None
    for n in ns:
        est = np.array([var_es(value_change(inp, normals(src, n)), inp.alpha) for _ in range(reps)])
        if n == 500:
            est500 = est
        rows.append(
            {
                "n": n,
                "VaR mean": est[:, 0].mean(),
                "VaR sd": est[:, 0].std(ddof=1),
                "VaR q05": np.quantile(est[:, 0], 0.05),
                "VaR q95": np.quantile(est[:, 0], 0.95),
                "ES mean": est[:, 1].mean(),
                "ES sd": est[:, 1].std(ddof=1),
                "ES q05": np.quantile(est[:, 1], 0.05),
                "ES q95": np.quantile(est[:, 1], 0.95),
            }
        )
    conv = pd.DataFrame(rows)
    conv.to_csv(RESULTS / "var_convergence.csv", index=False)
    out = {
        "alpha": inp.alpha,
        "positions_eur": {"S&P 500 ETF": inp.position_spx, "FTSE 100 ETF": inp.position_ftse},
        "spreadsheet_draws": {
            "n": len(ref["z"]),
            "var": sheet_var,
            "es": sheet_es,
            "var_in_spreadsheet": ref["var_99_1d_eur"],
            "es_in_spreadsheet": ref["es_99_1d_eur_reported"],
        },
        "reference": {"n": int(dv_ref.size), "var": ref_var, "es": ref_es},
        "delta_normal": {"var": dn_var, "es": dn_es},
        "n500_rank_of_spreadsheet_var": float(np.mean(est500[:, 0] <= sheet_var)) if est500 is not None else None,
        "convergence_reps": reps,
    }
    (RESULTS / "var.json").write_text(json.dumps(out, indent=2))
    return out, conv, dv_ref[:1_000_000]


# ----------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------


def fig_histograms(samples: dict, u: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax = axes.ravel()
    ax[0].hist(u, bins=40, density=True, edgecolor="white")
    ax[0].axhline(1.0, color="black", lw=1)
    ax[0].set_title(f"LCG uniforms ({N_SAMPLE:,} draws)")
    grid = np.linspace(-4, 4, 400)
    for a, (name, z) in zip(ax[1:], samples.items(), strict=True):
        a.hist(z, bins=40, range=(-4, 4), density=True, edgecolor="white")
        a.plot(grid, stats.norm.pdf(grid), color="black", lw=1)
        a.set_title(name)
        a.set_xlim(-4, 4)
    fig.tight_layout()
    fig.savefig(FIGURES / "histograms.png", dpi=200)
    plt.close(fig)


def fig_qq(samples: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    ax = axes.ravel()
    for a, (name, z) in zip(ax[: len(samples)], samples.items(), strict=True):
        n = z.size
        theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        a.plot(theo, np.sort(z), ".", ms=2)
        a.plot([-4, 4], [-4, 4], color="black", lw=1)
        a.set_xlim(-4, 4)
        a.set_ylim(-4.5, 4.5)
        a.set_title(name)
        a.set_xlabel("Theoretical quantiles")
        a.set_ylabel("Sample quantiles")
    ax[-1].boxplot(
        list(samples.values()),
        tick_labels=["Inverse", "Rejection", "Box-Muller", "Polar", "NumPy"],
        flierprops={"markersize": 2},
    )
    ax[-1].set_title("Box plots")
    fig.tight_layout()
    fig.savefig(FIGURES / "qq_plots.png", dpi=200)
    plt.close(fig)


def fig_var(var: dict, conv: pd.DataFrame, dv: np.ndarray) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.hist(dv / 1e3, bins=200, density=True)
    a1.axvline(-var["reference"]["var"] / 1e3, color="black", lw=1.2, label="VaR (99%)")
    a1.axvline(-var["reference"]["es"] / 1e3, color="black", lw=1.2, ls="--", label="ES (99%)")
    a1.set_xlim(-150, 150)
    a1.set_xlabel("One-day value change (EUR thousand)")
    a1.set_title(f"Simulated value changes ({dv.size:,} draws)")
    a1.legend()
    n = conv["n"].to_numpy()
    a2.fill_between(n, conv["VaR q05"] / 1e3, conv["VaR q95"] / 1e3, alpha=0.3, label="VaR, 5% to 95% of estimates")
    a2.plot(n, conv["VaR mean"] / 1e3, marker="o", ms=3, label="VaR, mean of estimates")
    a2.axhline(var["reference"]["var"] / 1e3, color="black", lw=1, label="VaR, 10 million draws")
    a2.plot(
        [var["spreadsheet_draws"]["n"]],
        [var["spreadsheet_draws"]["var"] / 1e3],
        "x",
        color="black",
        ms=8,
        label="Spreadsheet (500 draws)",
    )
    a2.set_xscale("log")
    a2.set_xlabel("Number of simulated scenarios")
    a2.set_ylabel("EUR thousand")
    a2.set_title(f"VaR estimates over {var['convergence_reps']} repetitions")
    a2.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "var.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    reps = 100 if args.quick else 1000
    tail_n = 10_000_000 if args.quick else 100_000_000
    var_reps = 50 if args.quick else 500
    n_ref = 1_000_000 if args.quick else 10_000_000

    t0 = time.time()
    summary: dict = {
        "sample_size": N_SAMPLE,
        "seed": SEED,
        "repetitions": reps,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": cpu_name(),
    }
    summary["uniform"] = uniform_section(reps)
    single, samples = single_sample_section()
    summary["single_sample"] = single.to_dict(orient="index")
    summary["rejection_rates"] = repeated_section(reps).to_dict(orient="index")
    summary["tails"] = tail_section(tail_n).to_dict(orient="index")
    summary["inverse_cdf"] = inverse_cdf_section().to_dict(orient="index")
    speed, acceptance = speed_section(1_000_000, 7)
    summary["speed"] = speed.to_dict(orient="records")
    summary["acceptance"] = acceptance
    var, conv, dv = var_section(n_ref, [500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000], var_reps)
    summary["var"] = var
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig_histograms(samples, LCG(seed=SEED).random(N_SAMPLE))
    fig_qq(samples)
    fig_var(var, conv, dv)
    print(f"done in {summary['runtime_seconds']} s")


if __name__ == "__main__":
    main()
