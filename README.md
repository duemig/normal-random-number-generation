# Generating normally distributed random numbers

Self-directed project, March 2019.
Notebook: [UCL Project.ipynb](UCL%20Project.ipynb). Report: [UCL Report.pdf](UCL%20Report.pdf).

## Question

Starting from a uniform random number generator, how do the standard methods for producing normal random variables compare in accuracy and speed?

## Methods

| Method | Implementation in the notebook |
|---|---|
| Uniform generator | Linear congruential generator $z_{i+1} = (a z_i + c) \bmod m$ with $a = 1103515245$, $c = 12345$, $m = 2^{32}$ |
| Inverse transform | $X = \Phi^{-1}(U)$, using `scipy.stats.norm.ppf` for the inverse CDF |
| Rejection sampling | Half-normal target, exponential proposal with envelope constant $c = \max f/g \approx 1.32$, random sign |
| Box-Muller | $Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2)$, $Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2)$ |
| Reference | `numpy.random.normal` |

Each method draws 5,000 standard normal values. These are checked with histograms, Q-Q and P-P plots, box plots, the empirical CDF, and the Shapiro-Wilk, D'Agostino K², Jarque-Bera, Kolmogorov-Smirnov and Anderson-Darling tests.

## Results

From the saved notebook outputs (5,000 draws each, a 2019 laptop, pure Python loops):

| Method | Mean | Std. dev. | Run time |
|---|---|---|---|
| Inverse transform | 0.011 | 1.009 | 0.93 s |
| Rejection sampling | -0.016 | 1.007 | 2.04 s |
| Box-Muller | -0.018 | 1.002 | 0.03 s |
| `numpy.random.normal` | 0.024 | 1.003 | 0.02 s |

Sample means and standard deviations are close to 0 and 1 for every method. The normality test results are in the notebook as interactive plotly tables, which GitHub does not render. Box-Muller is the fastest of the hand-written methods because it needs no inverse CDF evaluation and no rejection loop. The rejection sampler accepts about 1/c, roughly 76% of proposals, and is slowest in this implementation because it calls `scipy.stats` density functions inside the loop.

## Limitations

- The inverse-transform method relies on scipy's inverse CDF rather than an approximation written from scratch, and it draws its uniforms from numpy, not from the LCG.
- The Kolmogorov-Smirnov test is run with a one-sided alternative (`alternative="greater"`); a two-sided test is the appropriate choice for checking normality.
- Run times come from Python loops of single draws, so they compare implementations, not the algorithms' intrinsic cost. Vectorised versions would change the ranking.
- The report's Monte Carlo value-at-risk example was done in a separate spreadsheet that is not part of this repository.

## Reproducing

Python 3 with `numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn` and `plotly`. The notebook was run with Python 3.6.
