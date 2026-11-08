# Generating normally distributed pseudo-random numbers

Report: [report.pdf](report.pdf)

Four ways to turn uniform random numbers into normal ones, written from scratch in Python, tested properly, and used in a Monte Carlo value-at-risk calculation.

## What it covers

| Part | Content |
|---|---|
| Uniform generator | Linear congruential generator with a = 1103515245, c = 12345, m = 2^32. Full period by the Hull-Dobell theorem. Vectorised by evaluating the recurrence in blocks (jump-ahead), with output identical to the step-by-step version |
| Inverse transform | Inverse normal CDF implemented from Wichura's Algorithm AS 241 |
| Rejection sampling | Half-normal target, exponential proposal, random sign |
| Box-Muller | Each pair of uniforms gives two independent normals |
| Polar method | Marsaglia's version of Box-Muller without sine and cosine |
| Tests | Two-sided Kolmogorov-Smirnov and Anderson-Darling against N(0, 1), Shapiro-Wilk, D'Agostino K^2, Jarque-Bera; chi-square and KS tests of the uniforms |
| Application | Monte Carlo VaR and ES (99%, one day) of EUR 2.5m in an S&P 500 ETF and a FTSE 100 ETF, with four risk factors correlated through a Cholesky factor |

## Main results

- **Accuracy of the inverse CDF.** The AS 241 implementation matches `scipy.special.ndtri` to a maximum relative error of 1.2e-15, from the centre of the distribution down to p = 1e-300.
- **Normality.** Over 1,000 samples of 5,000 draws, each of the five tests rejects each method in 3.6% to 6.3% of samples at the 5% level. That is within two standard errors of the nominal 5%, the same as for NumPy's own generator.
- **Tails.** In 10^8 draws per method, the numbers of draws beyond 4 and 5 standard deviations match the normal distribution.
- **Speed.** In the vectorised code, the polar method is the fastest of the four, followed by Box-Muller. Inverse transform and rejection sampling come out about level, roughly half the speed of the polar method. NumPy's compiled ziggurat is about twice as fast as the polar method (report, Table 6).
- **VaR.** With 10 million scenarios, the one-day 99% VaR is EUR 79,166 and the ES is EUR 90,413. With 500 scenarios, the VaR estimate has a standard deviation of EUR 5,458, about 7% of the VaR.

## Reproducing

```bash
pip install -r requirements.txt
pytest                          # unit tests, a few seconds
python scripts/run_study.py     # about 1.5 minutes; writes results/ and figures/
python scripts/build_report.py  # needs XeLaTeX; writes report.pdf from report/report.tex and results/
```

Every number, table and figure in the report is produced by these two scripts. Seeds are fixed. The files in `results/` were produced with Python 3.13, NumPy 2.5.3 and SciPy 1.18.1; other versions change the timings and, in a few statistics, the last digits.

## Layout

| Path | Content |
|---|---|
| `src/normgen/uniform.py` | LCG and a PCG64-based reference source |
| `src/normgen/inverse_cdf.py` | Inverse normal CDF (AS 241) |
| `src/normgen/samplers.py` | The four samplers |
| `src/normgen/diagnostics.py` | Normality and uniformity tests, including the Anderson-Darling p-value |
| `src/normgen/var.py` | Cholesky decomposition, simulated value changes, VaR and ES estimators, delta-normal benchmark |
| `tests/` | Unit tests |
| `scripts/` | Study and report build |
| `results/`, `figures/` | Output of `run_study.py` |
| `report/` | LaTeX source of the report; `report/generated/` holds the tables and numbers written by `build_report.py` |
| `data/var_inputs.json` | Mean vector and covariance matrix of the four risk factors (estimated from 245 daily returns) and the positions |
| `data/spreadsheet_draws.json` | The 500 normal draws and results of an earlier spreadsheet version of the VaR example, used to check this implementation |
